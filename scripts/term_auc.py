"""One-feature AUC for the transfer outcome, per screened unit: 'thing = word' or 'thing = concept'.

Per feature, three encodings of the same window (admit - 24 h up to the decision point):
  presence  mentioned on >= 1 day (0/1)
  days      number of distinct days mentioned
  count     raw number of mentions (same-day / same-report repeats all count)
A one-variable logistic regression ranks samples exactly as the variable itself, so its
AUC = the Mann-Whitney AUC of the variable; no model fit needed.
AUC > 0.5: more in transfer cases; < 0.5: more in controls. days/count vs presence differences are
oriented (x sign of presence AUC - 0.5): positive = separates the groups better.

Features = label-blind screen: `final` if filled, else `auto`; keep + maybe are scored.
  --unit word     runs/term_frequency/screen.csv  (word/bigram + structured items, term_screen.py)
  --unit concept  runs/term_concepts/[cohort/]screen.csv (UMLS concept x polarity, term_concept_table.py)
  --cohort        sample definition (transfer_bow_model.COHORTS); the word screen is shared across cohorts
CI: patient-level bootstrap (resample subject_id, as in transfer_recurrence.py).
Report-text features are also scored within samples that have >= 1 radiology report in the window.

Outputs runs/term_auc/{unit}s.csv and {unit}s.md.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata

ROOT = Path(__file__).resolve().parents[1]
ACTION_PREFIX = ("drug:", "order:", "rad:", "micro:")


def auc_cols(y, X):
    """Mann-Whitney AUC of every column of X (dense, n x k) against binary y; ties averaged."""
    r = rankdata(X, axis=0)
    n1, n0 = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot(y, X, groups, n, seed):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in ug}
    out = np.full((n, X.shape[1]), np.nan)
    for b in range(n):
        idx = np.concatenate([idx_by[g] for g in rng.choice(ug, len(ug))])
        out[b] = auc_cols(y[idx], X[idx])
    return out


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def screened(path):
    s = pd.read_csv(path).fillna({"final": ""})
    s["label"] = np.where(s.final != "", s.final, s.auto)
    return s[s.label.isin(["keep", "maybe"])].reset_index(drop=True)


def load_words(cohort):
    from scripts.transfer_bow_model import COHORTS, build_samples, events
    from scripts.transfer_recurrence import encode, window_units
    samples, _ = build_samples(3, 20261011, **COHORTS[cohort])
    samples = samples[samples.hours >= 48].reset_index(drop=True)
    tu, ku, _ = window_units(samples, events(set(samples.hadm_id)), 0, facts_only=False)
    X, names = encode(tu, ku, len(samples))
    days, raw = X["distinct"].copy(), X["raw"].copy()
    days.data, raw.data = np.expm1(days.data).round(), np.expm1(raw.data).round()
    has_report = np.zeros(len(samples), bool)
    has_report[[u[0] for u in tu]] = True
    s = screened(ROOT / "runs/term_frequency/screen.csv").rename(columns={"term": "feature"})
    col = {n: i for i, n in enumerate(names)}
    s = s[s.feature.isin(col)].reset_index(drop=True)
    D = days[:, [col[t] for t in s.feature]].toarray()
    C = raw[:, [col[t] for t in s.feature]].toarray()
    s["kind"] = np.where(s.feature.str.startswith(ACTION_PREFIX), "action", "fact")
    s["note"] = ""
    s["report"] = s.category.str.startswith("report")
    return samples, D, C, s[["feature", "category", "label", "kind", "note", "report"]], has_report


def load_concepts(cohort):
    base = ROOT / "runs/term_concepts" / ("" if cohort == "all" else cohort)
    samples = pd.read_parquet(base / "samples.parquet")
    sf = pd.read_parquet(base / "sample_features.parquet")
    s = screened(base / "screen.csv").fillna({"note": ""})
    s["category"] = s.group + " " + s.polarity
    s["report"] = True
    col = {f: j for j, f in enumerate(s.feature)}
    D, C = np.zeros((len(samples), len(s))), np.zeros((len(samples), len(s)))
    hit = sf[sf.feature.isin(col)]
    D[hit["sample"].values, hit.feature.map(col).values] = hit.days.values
    C[hit["sample"].values, hit.feature.map(col).values] = hit.mentions.values
    has_report = np.zeros(len(samples), bool)
    has_report[sf["sample"].unique()] = True  # >= 1 linked concept in the window ~ >= 1 report
    return samples, D, C, s[["feature", "category", "label", "kind", "note", "report"]], has_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", choices=["word", "concept"], default="word")
    parser.add_argument("--cohort", default="all", help="sample definition, see transfer_bow_model.COHORTS")
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/term_auc")
    args = parser.parse_args()
    if args.cohort != "all":
        args.out = args.out / args.cohort
    args.out.mkdir(parents=True, exist_ok=True)

    samples, D, C, res, has_report = (load_words if args.unit == "word" else load_concepts)(args.cohort)
    P = (D > 0).astype(float)
    y, groups = samples.label.values, samples.subject_id.values

    bp, bd, bc = (boot(y, X, groups, args.boot, 0) for X in (P, D, C))  # same draws -> paired diffs
    res["share_cases"] = P[y == 1].mean(axis=0).round(3)
    res["share_controls"] = P[y == 0].mean(axis=0).round(3)
    res["days_cases"] = (D[y == 1].sum(axis=0) / np.maximum(P[y == 1].sum(axis=0), 1)).round(2)
    res["days_controls"] = (D[y == 0].sum(axis=0) / np.maximum(P[y == 0].sum(axis=0), 1)).round(2)
    res["auc_presence"] = auc_cols(y, P).round(3)
    res["ci_presence"] = [f"{a:.2f}–{b:.2f}" for a, b in np.nanpercentile(bp, [2.5, 97.5], axis=0).T]
    res["auc_days"] = auc_cols(y, D).round(3)
    # "better" = further from 0.5 in the feature's own direction (controls-side features have AUC < 0.5)
    sign = np.sign(auc_cols(y, P) - 0.5)
    lo, hi = np.nanpercentile((bd - bp) * sign, [2.5, 97.5], axis=0)
    res["days_minus_presence"] = ((res.auc_days - res.auc_presence) * sign).round(3)
    res["diff_ci"] = [f"{a:+.3f}–{b:+.3f}" for a, b in zip(lo, hi)]
    res["days_better"] = lo > 0
    res["auc_count"] = auc_cols(y, C).round(3)
    lo_c, hi_c = np.nanpercentile((bc - bp) * sign, [2.5, 97.5], axis=0)
    res["count_minus_presence"] = ((res.auc_count - res.auc_presence) * sign).round(3)
    res["count_diff_ci"] = [f"{a:+.3f}–{b:+.3f}" for a, b in zip(lo_c, hi_c)]
    res["count_better"], res["count_worse"] = lo_c > 0, hi_c < 0
    pval = [mannwhitneyu(P[y == 1, j], P[y == 0, j]).pvalue for j in range(P.shape[1])]
    res["q_presence"] = bh(pval).round(4)  # BH over all scored features (ignores patient clustering)
    sub = has_report
    res["auc_presence_imaged"] = np.where(res.report, auc_cols(y[sub], P[sub]).round(3), np.nan)
    res["strength"] = (res.auc_presence - 0.5).abs()
    res.sort_values("strength", ascending=False).drop(columns="strength").to_csv(args.out / f"{args.unit}s.csv",
                                                                                  index=False)

    n1, n0 = int(y.sum()), int((1 - y).sum())
    lines = [f"# 逐{'词' if args.unit == 'word' else '概念'} AUC（{args.unit} × 转科）", "",
             f"样本 {len(samples)}（转科 {n1} / 对照 {n0}）；决定点前有影像报告 {has_report.sum()}"
             f"（转科 {int(has_report[y == 1].sum())} / 对照 {int(has_report[y == 0].sum())}）。",
             f"打分 {len(res)} 个（keep {int((res.label == 'keep').sum())} / maybe {int((res.label == 'maybe').sum())}）；"
             f"q<0.05（多重比较校正后）{int((res.q_presence < 0.05).sum())} 个；"
             f"'天数'显著优于'出现过' {int(res.days_better.sum())} 个；'次数'显著优于 {int(res.count_better.sum())} 个、"
             f"显著差于 {int(res.count_worse.sum())} 个。", "",
             "AUC > 0.5 = 转科组更常见；< 0.5 = 对照组更常见。比例 = 提到的样本比例；天数 = 提到者平均提到天数；"
             "imaged = 只在有影像报告的样本里。", ""]
    show = ["feature", "note", "share_cases", "share_controls", "auc_presence", "ci_presence", "q_presence",
            "auc_presence_imaged", "auc_days", "diff_ci"]
    for k in ("fact", "action"):
        for lab in ("keep", "maybe"):
            d = res[(res.kind == k) & (res.label == lab)]
            d = d[d.q_presence < 0.05].sort_values("auc_presence", ascending=False)
            lines += [f"## {k} · {lab}：q<0.05（{len(d)}）", "",
                      "| " + " | ".join(show) + " |", "|" + "---|" * len(show)]
            lines += ["| " + " | ".join(str(getattr(r, c)) for c in show) + " |" for r in d.itertuples()]
            lines.append("")
    d = res[res.days_better]
    lines += [f"## '天数'显著优于'出现过'（{len(d)}）", "",
              "| feature | kind | auc_presence | auc_days | diff_ci |", "|---|---|---|---|---|"]
    lines += [f"| {r.feature} | {r.kind} | {r.auc_presence} | {r.auc_days} | {r.diff_ci} |" for r in d.itertuples()]
    for flag, title in (("count_better", "'次数'显著优于'出现过'"), ("count_worse", "'次数'显著差于'出现过'")):
        d = res[res[flag]].sort_values("count_minus_presence")
        lines += ["", f"## {title}（{len(d)}）", "",
                  "| feature | kind | auc_presence | auc_days | auc_count | count_diff_ci |", "|---|---|---|---|---|---|"]
        lines += [f"| {r.feature} | {r.kind} | {r.auc_presence} | {r.auc_days} | {r.auc_count} | {r.count_diff_ci} |"
                  for r in d.itertuples()]
    (args.out / f"{args.unit}s.md").write_text("\n".join(lines))
    print("\n".join(lines[:5]))


if __name__ == "__main__":
    main()
