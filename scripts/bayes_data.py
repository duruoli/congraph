#!/usr/bin/env python3
"""Bayes env v0 inputs: hidden problems s and lab / micro observations o (see BAYES_ENV.md section 7).

  s  one row per primary-IBD admission, one 0/1 column per problem: the admission carries a discharge ICD code
     of that problem. Rules = prefix lists in scripts/icd_problems.csv (draft, to review).
  o  one row per admission x day x item = the MOST abnormal result of that item that day (worst of a longer
     window = worst of its days). Labs: low / normal / high, or abnormal / normal for items that are only
     ever flagged without a direction (urine dipstick words, smear morphology); micro: positive / negative. Items: done in >= 10% of admissions, minus the drops / merges in
     scripts/bayes_items.csv (same measurement twice, derived values, specimen metadata, background screening);
     a `fallback` item is dropped when its twin is in the same draw and stands in for it otherwise.

Outputs (restricted MIMIC-derived, runs/ only) in runs/bayes_env/v0/:
  s.parquet  o.parquet  items.csv  icd_review.csv (every matched code + unmatched look-alikes)  summary.md

  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/bayes_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw_data/ibd_mimiciv_3_1"
UNITS = ROOT / "runs/terrain_units/word/units.parquet"
OUT = ROOT / "runs/bayes_env/v0"
MIN_COVER = 0.10


def cohort():
    c = pd.read_parquet(RAW / "cohort")
    return c[c.is_primary_ibd_diagnosis].drop_duplicates("hadm_id")


# ---------------------------------------------------------------- s
def build_s(coh):
    rules = pd.read_csv(ROOT / "scripts/icd_problems.csv", dtype=str).fillna("")
    d = pd.read_parquet(RAW / "diagnoses")
    d = d[d.hadm_id.isin(coh.hadm_id)]
    s = pd.DataFrame(0, index=pd.Index(coh.hadm_id, name="hadm_id"), columns=rules.problem.tolist())
    any_hit = pd.Series(False, index=d.index)
    review = []
    for r in rules.itertuples():
        hit = ((d.icd_version.eq(9) & d.icd_code.str.startswith(tuple(r.icd9.split())))
               | (d.icd_version.eq(10) & d.icd_code.str.startswith(tuple(r.icd10.split()))))
        any_hit |= hit
        s.loc[s.index.isin(d[hit].hadm_id), r.problem] = 1
        review.append(d[hit].assign(problem=r.problem, matched=True))
    for r in rules.itertuples():  # look-alikes no rule caught: candidates for a missed code
        near = ~any_hit & d.long_title.str.contains(r.keywords, case=False, na=False)
        review.append(d[near].assign(problem=r.problem, matched=False))
    rev = (pd.concat(review).groupby(["problem", "matched", "icd_version", "icd_code", "long_title"])
           .hadm_id.nunique().rename("admissions").reset_index())
    order = {p: i for i, p in enumerate(rules.problem)}
    rev = rev.sort_values(["problem", "matched", "admissions"], key=lambda c: c.map(order) if c.name == "problem"
                          else c, ascending=[True, False, False])
    return s, rev, rules


# ---------------------------------------------------------------- o
def build_o(coh):
    u = pd.read_parquet(UNITS, columns=["hadm_id", "t", "day", "source", "unit", "name", "status", "dev"])
    u = u[u.source.isin(["lab", "micro"]) & u.hadm_id.isin(coh.hadm_id)]
    rules = pd.read_csv(ROOT / "scripts/bayes_items.csv")
    # fallback = the same measurement as `into`: dropped when the same draw (admission x charttime) has `into`,
    # otherwise it stands in for it (a stand-alone hematocrit = the hemoglobin observation of that draw)
    fb = rules[rules.action.eq("fallback")].set_index("unit").into
    is_fb = u.unit.isin(fb.index)
    present = pd.MultiIndex.from_frame(u[["hadm_id", "t", "unit"]])
    u = u[~(is_fb & pd.MultiIndex.from_arrays([u.hadm_id, u.t, u.unit.map(fb)]).isin(present))]
    names = u.drop_duplicates("unit").set_index("unit").name
    is_fb = u.unit.isin(fb.index)
    u = u.assign(name=u.name.mask(is_fb, u.unit.map(fb).map(names)), unit=u.unit.map(fb).fillna(u.unit))
    merge = rules[rules.action.eq("merge")].set_index("unit")
    u = u.assign(item=u.unit.map(merge.into).fillna(u.unit),
                 name=u.unit.map(merge.into.str.split("|").str[-1]).fillna(u.name))
    n = coh.hadm_id.nunique()
    items = (u.groupby(["item", "source"]).agg(name=("name", "first"), admissions=("hadm_id", "nunique"))
             .reset_index())
    items["coverage"] = (items.admissions / n).round(3)
    drop = rules[rules.action.eq("drop")].set_index("unit").reason
    items["reason"] = items.item.map(drop)
    items.loc[items.reason.isna() & items.coverage.lt(MIN_COVER), "reason"] = f"done in < {MIN_COVER:.0%}"
    items["kept"] = items.reason.isna()

    u = u[u.item.isin(items.item[items.kept])]
    # "abnormal" without a direction = a flagged result whose value is redacted (e.g. CRP ">300"). On an item that
    # only ever goes one way it is that direction; on a two-way item the direction is unknown -> not an observation
    dirs = u[u.status.isin(["high", "low"])].groupby("item").status.agg(set)
    one_way = dirs[dirs.map(len).eq(1)].map(lambda s: next(iter(s)))
    flag = u.status.eq("abnormal")
    u = u.assign(status=u.status.mask(flag & u.item.isin(one_way.index), u.item.map(one_way)))
    u = u[~(u.status.eq("abnormal") & u.item.isin(dirs.index))]
    # most abnormal result of the item that day: any abnormal over normal, then farthest outside the range
    sev = u.status.isin(["high", "low", "abnormal", "positive"]).astype(float) + u.dev.abs().fillna(0)
    o = (u.assign(sev=sev).sort_values(["sev", "t"]).groupby(["hadm_id", "day", "item"]).tail(1)
         .rename(columns={"status": "value"})[["hadm_id", "day", "item", "value", "dev", "t"]]
         .sort_values(["hadm_id", "day", "item"]).reset_index(drop=True))
    lv = o.groupby("item").value.agg(lambda v: " / ".join(sorted(v.unique())))
    items["levels"] = items.item.map(lv).fillna("")
    return o, items.sort_values(["kept", "source", "admissions"], ascending=[False, True, False])


# ---------------------------------------------------------------- summary
def summary(s, rev, rules, o, items):
    n = len(s)
    zh = rules.set_index("problem").zh
    L = ["# Bayes env v0 inputs", "", f"Cohort: primary-IBD admissions, n = {n:,}.", "",
         "## s: problems (discharge ICD, draft rules in `scripts/icd_problems.csv`)", "",
         "| problem | | admissions | % | codes matched |", "|---|---|---:|---:|---:|"]
    for p in s.columns:
        k = int(s[p].sum())
        L.append(f"| {p} | {zh[p]} | {k:,} | {100 * k / n:.1f} | {int(rev[rev.problem.eq(p) & rev.matched].shape[0])} |")
    L += ["", f"All 10 = 0: {int((s.sum(1) == 0).sum()):,} admissions. "
              "Review every matched code and the unmatched look-alikes in `icd_review.csv`.", "",
          "## o: lab + micro, most abnormal result per admission x day x item", "",
          f"Items kept: {int(items.kept.sum())} "
          f"(lab {int((items.kept & items.source.eq('lab')).sum())}, "
          f"micro {int((items.kept & items.source.eq('micro')).sum())}); "
          f"rows {len(o):,}; admissions with any o {o.hadm_id.nunique():,} of {n:,}.", "",
          "| item | name | admissions | levels |", "|---|---|---:|---|"]
    L += [f"| `{r.item}` | {r.name} | {r.admissions:,} | {r.levels} |" for r in items[items.kept].itertuples()]
    L += ["", "Dropped although done in >= 10% (reason in `scripts/bayes_items.csv`):", ""]
    L += [f"- {r.name}: {r.reason}" for r in items[~items.kept & items.coverage.ge(MIN_COVER)].itertuples()]
    vc = o.value.value_counts()
    L += ["", "Values: " + ", ".join(f"{k} {v:,}" for k, v in vc.items()) + "."]
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coh = cohort()
    s, rev, rules = build_s(coh)
    o, items = build_o(coh)
    s.reset_index().to_parquet(OUT / "s.parquet", index=False)
    o.to_parquet(OUT / "o.parquet", index=False)
    items.to_csv(OUT / "items.csv", index=False)
    rev.to_csv(OUT / "icd_review.csv", index=False)
    (OUT / "summary.md").write_text(summary(s, rev, rules, o, items))
    print(f"s {s.shape}, o {len(o):,} rows / {int(items.kept.sum())} items -> {OUT}")


if __name__ == "__main__":
    main()
