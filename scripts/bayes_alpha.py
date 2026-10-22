#!/usr/bin/env python3
"""Bayes env v0: one set of evidence discounts that stays calibrated at every replay time (BAYES_ENV.md 7.8).

The conditional-LR posterior (bayes_seq.py) is still overconfident, and by an amount that grows with the
number of days: alpha is an empirical "total discount" for dependence the model does not represent. A replay
needs ONE set of discounts valid on day 1 and on day 10, so the discounts are fitted on all replay times at
once (stacked snapshots: every admission at every cutoff day < N), then judged per cutoff.

  one      logit b = b0 + b1 logit prior + alpha * (all evidence)
  split    ... + alpha_first * (first observations) + alpha_rep * (repeats)
  by_rep   ... + alpha_first * first + sum_r alpha_r * (repeats whose index falls in bin r)
           repeat index = how many earlier observations of the same item (1 = the first repeat)

Evidence at cutoff N uses ln LR estimated in that window (as in bayes_v0 / bayes_seq), out of fold. The
discounts are cross-fitted: fold f's prediction uses discounts fitted on the other folds' snapshots.

Outputs runs/bayes_env/v0/seq/: alpha_summary.md, alpha_metrics.csv, alpha_coef.csv

  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/bayes_alpha.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from scripts.bayes_seq import FIRST, OUT, daily
from scripts.bayes_v0 import FOLDS, calib_slope, ln_lr, load, logit, out_of_fold, window

CUTOFFS = [1, 2, 3, 5, 8, None]
REP_BINS = [(1, 1, "rep 1"), (2, 3, "rep 2-3"), (4, 7, "rep 4-7"), (8, 10 ** 6, "rep 8+")]
KEYS = ["item", "prev", "value"]


def split_evidence(o, coh, problems, levels, n):
    """out-of-fold evidence per admission, split into first observations and repeat-index bins."""
    d = daily(o, n)
    d = d.assign(rep=d.groupby(["hadm_id", "item"]).cumcount())
    parts = ["first"] + [b[2] for b in REP_BINS]
    out = {p: pd.DataFrame(0.0, index=coh.index, columns=problems) for p in parts}
    for f in range(FOLDS):
        tr, te = coh[coh.fold != f], coh[coh.fold == f]
        lr = ln_lr(d[d.hadm_id.isin(tr.index)], tr[problems].reset_index(), levels, problems, KEYS)
        ev = d[d.hadm_id.isin(te.index)].join(lr, on=KEYS).fillna({k: 0 for k in problems})
        part = pd.Series("first", index=ev.index)
        for lo, hi, name in REP_BINS:
            part[ev.rep.between(lo, hi)] = name
        for p in parts:
            out[p].loc[te.index] = ev[part.eq(p)].groupby("hadm_id")[problems].sum().reindex(te.index, fill_value=0).values
    prior = out_of_fold(window(o, n), coh, problems, levels)[0]
    return prior, out


def design(snap, k, method):
    lp = logit(snap["prior"][k].astype(float))
    first = snap["first"][k].values
    reps = [snap[b[2]][k].values for b in REP_BINS]
    if method == "one":
        return np.column_stack([lp, first + sum(reps)])
    if method == "split":
        return np.column_stack([lp, first, sum(reps)])
    return np.column_stack([lp, first, *reps])


def main():
    s, o, coh, problems, levels = load()
    snaps = []
    for n in CUTOFFS:
        prior, ev = split_evidence(o, coh, problems, levels, n)
        snaps.append({"cutoff": "all" if n is None else f"day<{n}", "prior": prior, **ev})
        print("evidence ready:", snaps[-1]["cutoff"])
    fold = np.concatenate([coh.fold.values] * len(snaps))
    cut = np.concatenate([[sn["cutoff"]] * len(coh) for sn in snaps])
    rows, coefs = [], []
    for k in problems:
        y = np.concatenate([coh[k].values] * len(snaps))
        for method in ["one", "split", "by_rep"]:
            X = np.vstack([design(sn, k, method) for sn in snaps])
            z = np.empty(len(y))
            for f in range(FOLDS):
                tr, te = fold != f, fold == f
                z[te] = LogisticRegression(C=1e6, max_iter=2000).fit(X[tr], y[tr]).decision_function(X[te])
            c = LogisticRegression(C=1e6, max_iter=2000).fit(X, y).coef_[0, 1:]
            names = {"one": ["alpha"], "split": ["first", "repeat"],
                     "by_rep": ["first"] + [b[2] for b in REP_BINS]}[method]
            coefs.append({"problem": k, "method": method, **dict(zip(names, c))})
            for cname in dict.fromkeys(cut):
                m = cut == cname
                p = np.clip(1 / (1 + np.exp(-z[m])), 1e-6, 1 - 1e-6)
                rows.append({"cutoff": cname, "problem": k, "method": method, "auc": roc_auc_score(y[m], z[m]),
                             "log_loss": log_loss(y[m], p), "slope": calib_slope(y[m], z[m])})
    met, co = pd.DataFrame(rows), pd.DataFrame(coefs)
    met.to_csv(OUT / "alpha_metrics.csv", index=False)
    co.to_csv(OUT / "alpha_coef.csv", index=False)

    L = ["# One set of discounts for every replay time", "",
         "Discounts fitted once on all cutoffs stacked (cross-fitted by patient fold), then judged per cutoff. "
         "Slope 1 = calibrated at that replay time. Mean over the 10 problems.", "",
         "## calibration slope per cutoff", "", "| cutoff | " + " | ".join(["one", "split", "by_rep"]) + " |",
         "|---|---:|---:|---:|"]
    t = met.groupby(["cutoff", "method"], sort=False)[["slope", "log_loss", "auc"]].mean()
    for cname in dict.fromkeys(met.cutoff):
        L.append(f"| {cname} | " + " | ".join(f"{t.loc[(cname, m), 'slope']:.2f}" for m in ["one", "split", "by_rep"]) + " |")
    L += ["", "## log loss per cutoff (lower = better)", "", "| cutoff | one | split | by_rep |", "|---|---:|---:|---:|"]
    for cname in dict.fromkeys(met.cutoff):
        L.append(f"| {cname} | " + " | ".join(f"{t.loc[(cname, m), 'log_loss']:.4f}" for m in ["one", "split", "by_rep"]) + " |")
    L += ["", "## AUC per cutoff", "", "| cutoff | one | split | by_rep |", "|---|---:|---:|---:|"]
    for cname in dict.fromkeys(met.cutoff):
        L.append(f"| {cname} | " + " | ".join(f"{t.loc[(cname, m), 'auc']:.3f}" for m in ["one", "split", "by_rep"]) + " |")
    L += ["", "## discounts (all admissions, median over problems)", ""]
    for m, g in co.groupby("method", sort=False):
        L.append(f"- {m}: " + ", ".join(f"{c} {g[c].median():.2f}" for c in g.columns[2:] if g[c].notna().any()))
    (OUT / "alpha_summary.md").write_text("\n".join(L) + "\n")
    print((OUT / "alpha_summary.md").read_text())


if __name__ == "__main__":
    main()
