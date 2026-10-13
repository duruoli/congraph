"""Bag-of-words baseline: can plain token presence predict a MED→SURG transfer?

Cases: admissions that start on MED whose first transfer is MED→SURG.
Time zero = DECISION point: the first trace that the operation/transfer is
already decided — earliest of service change, "Transfer to=Surgery" ADT order
(≤120 h before), start of the operation day (procedure codes are date-only),
oral neomycin (bowel prep), pre-op imaging. The QUESTION point (first surgical
consult order) is recorded alongside but not modelled (few cases have one).
Controls: MED-only admissions still in hospital at the same hours-since-admit
(risk-set matching, K per case).

Every event that became available inside a window is one token: words/bigrams
from radiology reports, plus prefixed structured tokens (order type, drug,
abnormal lab, micro specimen/organism, exam name). Presence is binary.
Because t0 is the decision point, decision traces fall after t0 by
construction; only ADT orders are dropped. Negation words are kept in the
text (default English stop list removes no/not/without). L1 logistic
regression, outer CV grouped by patient.
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.sparse import hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw_data/ibd_mimiciv_3_1"
H = pd.Timedelta(hours=1)
NEGATION = {"no", "not", "nor", "without", "none", "never", "neither", "cannot"}
STOP = sorted(ENGLISH_STOP_WORDS - NEGATION)
PREOP = re.compile(r"pre[- ]?op", re.I)


def operative(r):
    """Open or laparoscopic abdominal operation (bedside/IR/endoscopic excluded)."""
    if r.icd_version == 10:
        return r.icd_code[:2] in ("0D", "0F", "0W", "0K") and \
            bool(re.search(r"Open Approach|Percutaneous Endoscopic Approach", r.long_title))
    c = r.icd_code
    return c[:2] in ("45", "46", "47", "48", "49", "53", "54") and \
        not c.startswith(("451", "452", "454", "482", "492", "5491", "5495"))
# (name, window start, window end) in hours relative to t0; None start = from ED (admit − 24 h)
WINDOWS = [("cum_lag0", None, 0), ("cum_lag12", None, -12), ("cum_lag24", None, -24), ("cum_lag48", None, -48),
           ("slice_0_12", -12, 0), ("slice_12_24", -24, -12), ("slice_24_48", -48, -24)]


def load(name, hadms, columns=None):
    return ds.dataset(SOURCE / name, format="parquet").to_table(
        columns=columns, filter=ds.field("hadm_id").isin(list(hadms))).to_pandas()


def build_samples(k, seed):
    co = ds.dataset(SOURCE / "cohort", format="parquet").to_table().to_pandas()
    sv = ds.dataset(SOURCE / "services", format="parquet").to_table().to_pandas()
    sv = sv.sort_values(["hadm_id", "transfertime"])
    first = sv.groupby("hadm_id").head(1)
    tr = sv[sv.prev_service.notna()]
    ft = tr.groupby("hadm_id").head(1)
    med_start = set(first[first.curr_service == "MED"].hadm_id)
    cases = ft[(ft.prev_service == "MED") & (ft.curr_service == "SURG") & ft.hadm_id.isin(med_start)]
    cases = cases.merge(co[["hadm_id", "subject_id", "admittime", "dischtime"]])

    det = load("poe_detail", cases.hadm_id)
    orders = det[(det.field_name == "Transfer to") & (det.field_value == "Surgery")]
    proc = load("procedures", cases.hadm_id)
    proc = proc[[operative(r) for r in proc.itertuples()]]
    rx = load("prescriptions", cases.hadm_id, ["hadm_id", "starttime", "drug", "route"])
    prep = rx[(rx.drug == "Neomycin Sulfate") & (rx.route == "PO")]
    rad = load("radiology", cases.hadm_id, ["hadm_id", "charttime", "exam_name", "text"])
    preop = rad[rad.exam_name.str.contains(PREOP, na=False) | rad.text.str.contains(PREOP, na=False)]
    poe = load("poe", cases.hadm_id, ["hadm_id", "ordertime", "order_type", "order_subtype"])
    consult = poe[(poe.order_type == "Consults") & poe.order_subtype.str.contains("Surg", na=False)]
    t0, src, tq = [], [], []
    for r in cases.itertuples():
        def first(df, col):
            x = df[(df.hadm_id == r.hadm_id) & (df[col] >= r.admittime - 24 * H) & (df[col] <= r.transfertime)][col]
            return x.min() if len(x) else None
        cand = {"service": r.transfertime, "bowel_prep": first(prep, "starttime"),
                "preop_imaging": first(preop, "charttime")}
        o = orders[(orders.hadm_id == r.hadm_id) & (orders.ordertime <= r.transfertime)
                   & (orders.ordertime >= r.transfertime - 120 * H)].ordertime
        if len(o):
            cand["order"] = o.min()
        d = pd.to_datetime(proc[proc.hadm_id == r.hadm_id].chartdate)
        d = d[(d >= r.admittime.normalize()) & (d <= r.transfertime)]
        if len(d):
            cand["operation_day"] = max(d.min(), r.admittime)
        cand = {k: v for k, v in cand.items() if v is not None}
        first_src = min(cand, key=cand.get)
        t0.append(cand[first_src]), src.append(first_src)
        q = first(consult, "ordertime")
        tq.append(q if q is not None and q <= cand[first_src] else pd.NaT)
    cases = cases.assign(t0=t0, t0_source=src, t_question=tq, label=1)
    cases["hours"] = (cases.t0 - cases.admittime) / H

    ctrl = co[co.hadm_id.isin(med_start - set(tr.hadm_id))]
    rng = random.Random(seed)
    rows = []
    for r in cases.itertuples():
        t = ctrl.admittime + pd.Timedelta(hours=r.hours)
        pool = ctrl[(t < ctrl.dischtime) & (ctrl.subject_id != r.subject_id)]
        for c in rng.sample(list(pool.itertuples()), min(k, len(pool))):
            rows.append({"hadm_id": c.hadm_id, "subject_id": c.subject_id, "admittime": c.admittime,
                         "t0": c.admittime + pd.Timedelta(hours=r.hours), "hours": r.hours,
                         "label": 0, "case_hadm": r.hadm_id, "t0_source": "matched"})
    cases["case_hadm"] = cases.hadm_id
    cols = ["hadm_id", "subject_id", "admittime", "t0", "hours", "label", "case_hadm", "t0_source"]
    return pd.concat([cases[cols], pd.DataFrame(rows)[cols]], ignore_index=True), cases


def norm(s):
    return re.sub(r"\W+", "_", str(s).strip().lower()).strip("_")


def events(hadms):
    """(hadm_id, available_time, kind, payload). kind: 'text' = report free text (fact),
    'fact' = result token (abnormal lab, organism), 'tok' = action token (order, drug, exam, specimen sent)."""
    out = []
    rad = load("radiology", hadms, ["hadm_id", "charttime", "storetime", "exam_name", "text"])
    for r in rad.itertuples():
        t = r.storetime if pd.notna(r.storetime) else r.charttime
        if not re.search(r"pre-?op", r.exam_name or "", re.I):
            out.append((r.hadm_id, t, "tok", "rad:" + norm(r.exam_name)))
        out.append((r.hadm_id, t, "text", r.text or ""))
    poe = load("poe", hadms, ["hadm_id", "ordertime", "order_type", "order_subtype", "transaction_type"])
    poe = poe[(poe.transaction_type == "New") & ~poe.order_type.isin(["ADT orders", "Medications", "Lab"])]
    out += [(r.hadm_id, r.ordertime, "tok", f"order:{norm(r.order_type)}/{norm(r.order_subtype)}")
            for r in poe.itertuples()]
    rx = load("prescriptions", hadms, ["hadm_id", "starttime", "drug"])
    out += [(r.hadm_id, r.starttime, "tok", "drug:" + norm(r.drug)) for r in rx.itertuples()]
    lab = load("labs", hadms, ["hadm_id", "charttime", "storetime", "label", "valuenum",
                                "ref_range_lower", "ref_range_upper", "flag"])
    lab = lab[lab.flag.notna()]
    for r in lab.itertuples():
        side = "high" if pd.notna(r.ref_range_upper) and pd.notna(r.valuenum) and r.valuenum > r.ref_range_upper \
            else "low" if pd.notna(r.ref_range_lower) and pd.notna(r.valuenum) and r.valuenum < r.ref_range_lower \
            else "abn"
        out.append((r.hadm_id, r.storetime if pd.notna(r.storetime) else r.charttime, "fact",
                    f"lab:{norm(r.label)}_{side}"))
    mic = load("microbiology", hadms, ["hadm_id", "charttime", "chartdate", "storetime", "storedate",
                                       "spec_type_desc", "org_name"])
    for r in mic.itertuples():
        t = next((x for x in (r.storetime, r.storedate, r.charttime, r.chartdate) if pd.notna(x)), None)
        out.append((r.hadm_id, t, "tok", "micro:" + norm(r.spec_type_desc)))
        if pd.notna(r.org_name):
            out.append((r.hadm_id, t, "fact", "org:" + norm(r.org_name)))
    ev = pd.DataFrame(out, columns=["hadm_id", "t", "kind", "payload"]).dropna(subset=["t"])
    ev["t"] = pd.to_datetime(ev.t)
    return ev


def documents(samples, ev, start, end, facts_only):
    by = dict(tuple(ev.groupby("hadm_id")))
    text, toks, n = [], [], []
    for s in samples.itertuples():
        lo = s.admittime - 24 * H if start is None else s.t0 + start * H
        e = by.get(s.hadm_id)
        if e is None:
            text.append(""), toks.append(""), n.append(0)
            continue
        w = e[(e.t >= lo) & (e.t < s.t0 + end * H)]
        text.append(" ".join(w[w.kind == "text"].payload))
        toks.append(" ".join(w[(w.kind == "fact") if facts_only else (w.kind != "text")].payload))
        n.append(int((w.kind != "tok").sum()) if facts_only else len(w))
    return text, toks, np.array(n)


def vectorizers(min_df):
    return (CountVectorizer(binary=True, ngram_range=(1, 2), stop_words=STOP, min_df=min_df,
                            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"),
            CountVectorizer(binary=True, min_df=min_df, lowercase=False, token_pattern=r"\S+"))


def evaluate(samples, text, toks, n, min_df, seed):
    y, groups = samples.label.values, samples.subject_id.values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    p_bow, p_cnt = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in cv.split(text, y, groups):
        tv, kv = vectorizers(min_df)
        Xtr = hstack([tv.fit_transform([text[i] for i in tr]), kv.fit_transform([toks[i] for i in tr])]).tocsr()
        Xte = hstack([tv.transform([text[i] for i in te]), kv.transform([toks[i] for i in te])]).tocsr()
        m = LogisticRegressionCV(Cs=8, cv=4, penalty="l1", solver="liblinear", scoring="roc_auc",
                                 max_iter=2000).fit(Xtr, y[tr])
        p_bow[te] = m.predict_proba(Xte)[:, 1]
        # baseline: how much is just "more stuff happened"
        lc = np.log1p(n).reshape(-1, 1)
        p_cnt[te] = LogisticRegression().fit(lc[tr], y[tr]).predict_proba(lc[te])[:, 1]
    return roc_auc_score(y, p_bow), roc_auc_score(y, p_cnt)


def top_features(samples, text, toks, min_df, k=30):
    tv, kv = vectorizers(min_df)
    X = hstack([tv.fit_transform(text), kv.fit_transform(toks)]).tocsr()
    names = np.r_[tv.get_feature_names_out(), kv.get_feature_names_out()]
    m = LogisticRegressionCV(Cs=8, cv=5, penalty="l1", solver="liblinear", scoring="roc_auc",
                             max_iter=2000).fit(X, samples.label.values)
    coef = m.coef_[0]
    y = samples.label.values
    prev = lambda j: (X[y == 1, j].mean(), X[y == 0, j].mean())
    order = np.argsort(coef)
    pick = lambda idx: [(names[j], round(coef[j], 3), *map(lambda v: round(float(v), 3), prev(j)))
                        for j in idx if coef[j] != 0]
    return pick(order[::-1][:k]), pick(order[:k]), int((coef != 0).sum()), X.shape[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3, help="controls per case")
    parser.add_argument("--min-hours", type=float, default=48, help="keep cases with t0 at least this far into the stay")
    parser.add_argument("--min-df", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--facts-only", action="store_true", help="drop action tokens (orders, drugs, exams sent)")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/transfer_bow")
    args = parser.parse_args()
    args.out = args.out / ("facts" if args.facts_only else "all")
    args.out.mkdir(parents=True, exist_ok=True)

    samples, cases = build_samples(args.k, args.seed)
    print("cases", len(cases), "| t0 source:", cases.t0_source.value_counts().to_dict())
    lead = (cases.t0 - cases.t_question) / H
    print("question point (surgical consult) present:", lead.notna().sum(),
          "| hours before decision point, quartiles:", lead.quantile([.25, .5, .75]).round(1).tolist())
    cases.to_csv(args.out.parent / "cases_anchors.csv", index=False)
    samples = samples[samples.hours >= args.min_hours].reset_index(drop=True)
    print("after min-hours:", samples.label.sum(), "cases,", (samples.label == 0).sum(), "controls")
    ev = events(set(samples.hadm_id))

    results, report = [], {}
    for name, start, end in WINDOWS:
        text, toks, n = documents(samples, ev, start, end, args.facts_only)
        auc, auc_cnt = evaluate(samples, text, toks, n, args.min_df, args.seed)
        results.append({"window": name, "start_h": start, "end_h": end, "auc_bow": round(auc, 3),
                        "auc_event_count_only": round(auc_cnt, 3),
                        "empty_case": int(((n == 0) & (samples.label == 1)).sum()),
                        "empty_ctrl": int(((n == 0) & (samples.label == 0)).sum())})
        print(results[-1])
        if name in ("cum_lag0", "cum_lag48", "slice_0_12"):
            pos, neg, nz, nf = top_features(samples, text, toks, args.min_df)
            report[name] = {"nonzero": nz, "n_features": nf, "toward_transfer": pos, "toward_stay": neg}
    pd.DataFrame(results).to_csv(args.out / "auc_by_window.csv", index=False)
    (args.out / "top_features.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    samples.to_csv(args.out / "samples.csv", index=False)


if __name__ == "__main__":
    main()
