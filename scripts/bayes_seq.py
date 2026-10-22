#!/usr/bin/env python3
"""Bayes env v0, step 1.5b: repeated measurements by conditional LR (BAYES_ENV.md 7.4, recurrence - replication).

v0 keeps ONE result per item per window (the worst), so the day-by-day change is lost. Replaying day by day
(step 6) needs every day, but summing each day's ln LR as if independent counts a persistent state again and
again. The chain rule, first-order Markov given s:

  p(o_1 .. o_T | s_k) = p(o_1 | s_k) * prod_t p(o_t | o_{t-1}, s_k)
  ln LR(o_t | o_{t-1}) = ln p(o_t | o_{t-1}, s=1) - ln p(o_t | o_{t-1}, s=0)

A repeat only adds what the previous result did not predict: "still low" scores ~0 when it is as common in
s=1 as in s=0. o_{t-1} = the item's previous observed day in the window (gaps skipped); the first observation
uses prev = "first", i.e. the plain likelihood of a first result. Same estimator as bayes_v0.ln_lr (shrunk
toward the item's pooled distribution, now per previous value), same 5 folds by patient.

Compared, per window (day < N):
  worst   v0: worst result per item in the window, plain LR                 (bayes_v0.py)
  daily   every day's worst result, plain LR, summed as if independent      (the double count, for reference)
  cond    every day's worst result, conditional LR                          (this step)
each raw and with one alpha per problem fitted out of fold (bayes_calib method A).

Outputs runs/bayes_env/v0/seq/: summary.md, metrics.csv, lr_transition.csv (all admissions, for reading)

  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/bayes_seq.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from scripts.bayes_calib import fit_predict
from scripts.bayes_v0 import DATA, calib_slope, ln_lr, load, logit, out_of_fold, window

OUT = DATA / "seq"
WINDOWS = [1, 3, None]
FIRST = "first"


def daily(o, n):
    """worst result per admission x day x item (already so in o.parquet) inside the window, with the previous one."""
    w = (o if n is None else o[o.day < n]).sort_values(["hadm_id", "item", "day"])
    prev = w.groupby(["hadm_id", "item"]).value.shift().fillna(FIRST)
    return w.assign(prev=prev)[["hadm_id", "item", "day", "prev", "value"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s, o, coh, problems, levels = load()
    fold = coh.fold.values
    rows = []
    for n in WINDOWS:
        wname = "all" if n is None else f"day<{n}"
        d = daily(o, n)
        evid = {"worst": out_of_fold(window(o, n), coh, problems, levels)[1],
                "daily": out_of_fold(d, coh, problems, levels)[1],
                "cond": out_of_fold(d, coh, problems, levels, keys=("item", "prev", "value"))[1]}
        prior, _ = out_of_fold(window(o, n), coh, problems, levels)
        per_adm = d.groupby("hadm_id").size().reindex(coh.index, fill_value=0)
        repeats = int((d.prev != FIRST).sum())
        for k in problems:
            y = coh[k].values
            lp = pd.Series(logit(prior[k]), index=coh.index)
            for m, ev in evid.items():
                ev = ev[k].astype(float)
                z_raw = (lp + ev).values
                z_a, a = fit_predict(lp, ev, y, None, fold)
                for tag, z in [("raw", z_raw), ("alpha", z_a)]:
                    p = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)
                    rows.append({"window": wname, "problem": k, "method": m, "fit": tag, "auc": roc_auc_score(y, z),
                                 "log_loss": log_loss(y, p), "slope": calib_slope(y, z),
                                 "alpha": a["all"] if tag == "alpha" else np.nan})
        print(f"{wname}: {len(d):,} daily facts ({per_adm.median():.0f} per admission), {repeats:,} repeats")
    met = pd.DataFrame(rows)
    met.to_csv(OUT / "metrics.csv", index=False)

    # transitions on all admissions, whole stay, for reading: does a repeat add evidence?
    d = daily(o, None)
    plain = ln_lr(d[d.prev == FIRST], s, levels, problems)
    cond = ln_lr(d, s, levels, problems, keys=("item", "prev", "value"))
    name = pd.read_csv(DATA / "items.csv").set_index("item").name
    cond.insert(0, "name", cond.index.get_level_values("item").map(name))
    cond.insert(1, "n", d.groupby(["item", "prev", "value"]).size().reindex(cond.index))
    cond.to_csv(OUT / "lr_transition.csv")

    L = ["# Bayes v0, 5b: repeated measurements by conditional LR", "",
         "Out of fold (5 folds by patient), primary-IBD admissions. `worst` = v0 (one worst result per item per "
         "window); `daily` = every day's result as independent evidence; `cond` = every day, each repeat by "
         "ln LR(o_t | o_{t-1}). `raw` = prior + summed evidence; `alpha` = one discount per problem fitted out of "
         "fold. Slope 1 = honest, < 1 = overconfident; alpha near 1 = little double counting left.", ""]
    for w, g in met.groupby("window", sort=False):
        t = g.groupby(["method", "fit"], sort=False)[["auc", "log_loss", "slope", "alpha"]].mean()
        L += [f"## {w}: mean over the 10 problems", "", "| method | fit | AUC | log loss | slope | alpha |",
              "|---|---|---:|---:|---:|---:|"]
        L += [f"| {m} | {f} | {r.auc:.3f} | {r.log_loss:.4f} | {r.slope:.2f} | "
              f"{'' if np.isnan(r.alpha) else f'{r.alpha:.2f}'} |" for (m, f), r in t.iterrows()]
        L.append("")
    L += ["## does a repeat add evidence? (whole stay, all admissions)", "",
          "ln LR of a first result vs the same result again the next measured day. Persistence carries little "
          "news when it is common in both s=1 and s=0.", "",
          "| fact | problem | first | again (same as before) | n again |", "|---|---|---:|---:|---:|"]
    for item, value, k in [("lab:51222", "low", "anemia"), ("lab:50912", "high", "aki"), ("lab:50912", "low", "malnutrition"),
                           ("lab:50983", "high", "aki"), ("lab:50889", "high", "abscess_fistula"),
                           ("lab:51275", "high", "thrombus"), ("lab:50862", "low", "malnutrition")]:
        if (item, value) in plain.index and (item, value, value) in cond.index:
            L.append(f"| {name.get(item, item)} {value} | {k} | {plain.loc[(item, value), k]:+.2f} | "
                     f"{cond.loc[(item, value, value), k]:+.2f} | {int(cond.loc[(item, value, value), 'n'])} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
