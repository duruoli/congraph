#!/usr/bin/env python3
"""Bayes env v0: calibrating the overconfident posterior (BAYES_ENV.md 7.3, "s too coarse": severity).

Naive Bayes sums ln LR as if the facts were independent given s; a sicker patient has many abnormal facts
that move together, so the sum is too large. Methods compared, all out of fold (5 folds by patient; the
calibration weights of fold f are fitted on the other folds' out-of-fold predictions):

  raw        logit prior + sum ln LR                                 (bayes_v0.py)
  A          logit prior * b + alpha * sum ln LR                      one discount per problem
  B DRG      alpha depends on the APR-DRG severity of illness (1-4, or none): sicker -> discount more?
             DRG severity is assigned AFTER discharge from the codes, so it peeks; reference only
  C count    alpha depends on how many items were abnormal in the window (tertiles): same idea, but the
             severity is visible at the time
  D strat    likelihood within patients of similar severity: p(o_j | s_k, bin), bin = tertile of the number
             of OTHER abnormal items in the window (leave j out); then one alpha as in A
  D2         D plus the severity itself as one piece of evidence, ln p(bin | s=1) / p(bin | s=0):
             the full factorisation p(o, bin | s) = p(bin | s) prod p(o_j | s, bin) (D alone drops it)

Metrics per problem: AUC (ranking), log loss (are the probabilities right; lower = better), calibration
slope of the final logit (1 = honest).

Outputs runs/bayes_env/v0/calib/: summary.md, metrics.csv, alpha.csv

  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/bayes_calib.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from scripts.bayes_v0 import ABNORMAL, DATA, FOLDS, RAW_DIR, calib_slope, load, ln_lr, logit, out_of_fold, window

OUT = DATA / "calib"
WINDOWS = [1, 3]


def severity(coh, ow):
    drg = pd.read_parquet(RAW_DIR / "drgcodes")
    drg = drg[drg.drg_type.eq("APR")].groupby("hadm_id").drg_severity.max()
    sev = pd.DataFrame(index=coh.index)
    sev["drg"] = drg.reindex(coh.index).map(lambda x: "none" if pd.isna(x) else str(int(x)))
    n_abn = ow.assign(a=ow.value.isin(ABNORMAL)).groupby("hadm_id").a.sum().reindex(coh.index, fill_value=0)
    cuts = np.quantile(n_abn, [1 / 3, 2 / 3])
    sev["count"] = pd.cut(n_abn, [-np.inf, *cuts, np.inf], labels=["low", "mid", "high"]).astype(str)
    return sev, n_abn, cuts


def fit_predict(lp, ev, y, group, fold):
    """cross-fitted logit = b0 + b1 * logit prior + alpha_g * evidence; group None = one alpha."""
    g = pd.Series("all" if group is None else group, index=lp.index)
    levels = sorted(g.unique())
    X = np.column_stack([lp] + [ev * (g == v) for v in levels])
    out = np.empty(len(y))
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        m = LogisticRegression(C=1e6, max_iter=1000).fit(X[tr], y[tr])
        out[te] = m.decision_function(X[te])
    alpha = LogisticRegression(C=1e6, max_iter=1000).fit(X, y).coef_[0, 1:]  # for reading only
    return out, dict(zip(levels, alpha))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s, o, coh, problems, levels = load()
    rows, alphas, nonspec = [], [], []
    for n in WINDOWS:
        ow = window(o, n)
        sev, n_abn, cuts = severity(coh, ow)
        prior, evid = out_of_fold(ow, coh, problems, levels)
        # D: likelihood stratified by the severity of the OTHER facts
        loo = ow.hadm_id.map(n_abn).values - ow.value.isin(ABNORMAL).values
        ow_b = ow.assign(bin=pd.cut(loo, [-np.inf, *cuts, np.inf], labels=["low", "mid", "high"]).astype(str))
        _, evid_d = out_of_fold(ow_b, coh, problems, levels, keys=("item", "value", "bin"))
        # D2: the full factorisation p(o, bin | s) = p(bin | s) * prod p(o_j | s, bin): "how sick" counted ONCE
        ow_sev = pd.DataFrame({"hadm_id": coh.index, "item": "__severity__", "value": sev["count"].values})
        _, evid_sev = out_of_fold(ow_sev, coh, problems, pd.concat([levels, pd.Series({"__severity__": 3})]))
        fold = coh.fold.values
        for k in problems:
            y = coh[k].values
            lp = pd.Series(logit(prior[k]), index=coh.index)
            ev, ev_d = evid[k].astype(float), evid_d[k].astype(float)
            preds = {"raw": (lp + ev).values}
            preds["A one alpha"], a = fit_predict(lp, ev, y, None, fold)
            alphas.append({"window": n, "problem": k, "method": "A", **a})
            preds["B by DRG"], a = fit_predict(lp, ev, y, sev.drg, fold)
            alphas.append({"window": n, "problem": k, "method": "B", **a})
            preds["C by count"], a = fit_predict(lp, ev, y, sev["count"], fold)
            alphas.append({"window": n, "problem": k, "method": "C", **a})
            preds["D strat raw"] = (lp + ev_d).values
            preds["D strat + alpha"], a = fit_predict(lp, ev_d, y, None, fold)
            alphas.append({"window": n, "problem": k, "method": "D", **a})
            ev_d2 = ev_d + evid_sev[k].astype(float)
            preds["D2 strat + severity once"] = (lp + ev_d2).values
            preds["D2 + alpha"], a = fit_predict(lp, ev_d2, y, None, fold)
            alphas.append({"window": n, "problem": k, "method": "D2", **a})
            for m, z in preds.items():
                p = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)
                rows.append({"window": f"day<{n}", "problem": k, "method": m, "auc": roc_auc_score(y, z),
                             "log_loss": log_loss(y, p), "slope": calib_slope(y, z)})
        # does stratifying shrink the non-specific "sick" facts? mean |ln LR| over problems, all admissions
        plain = ln_lr(ow, s, levels, problems)
        strat = ln_lr(ow_b, s, levels, problems, keys=("item", "value", "bin"))
        w = ow_b.groupby(["item", "value", "bin"]).size()
        strat_avg = strat.mul(w, axis=0).groupby(level=["item", "value"]).sum().div(
            w.groupby(level=["item", "value"]).sum(), axis=0)
        for item, value, label in [("lab:50983", "high", "Sodium high"), ("lab:50971", "high", "Potassium high"),
                                   ("lab:50912", "low", "Creatinine low"), ("lab:50912", "high", "Creatinine high"),
                                   ("lab:51222", "normal", "Hemoglobin normal")]:
            nonspec.append({"window": f"day<{n}", "fact": label,
                            "plain": plain.loc[(item, value)].abs().mean(), "strat": strat_avg.loc[(item, value)].abs().mean()})

    met = pd.DataFrame(rows)
    met.to_csv(OUT / "metrics.csv", index=False)
    al = pd.DataFrame(alphas)
    al.to_csv(OUT / "alpha.csv", index=False)
    ns = pd.DataFrame(nonspec)

    L = ["# Calibrating the v0 posterior", "",
         "Out of fold, primary-IBD admissions. AUC = ranking; log loss = are the probabilities right (lower = better); "
         "slope 1 = honest, < 1 = overconfident. B uses the post-discharge DRG severity (peeks; reference only).", ""]
    for w, g in met.groupby("window", sort=False):
        t = g.groupby("method", sort=False)[["auc", "log_loss", "slope"]].mean()
        L += [f"## {w}: mean over the 10 problems", "", "| method | AUC | log loss | slope |", "|---|---:|---:|---:|"]
        L += [f"| {m} | {r.auc:.3f} | {r.log_loss:.4f} | {r.slope:.2f} |" for m, r in t.iterrows()]
        L += ["", "Per problem, log loss (lower = better):", "",
              "| problem | " + " | ".join(t.index) + " |", "|---|" + "---:|" * len(t)]
        pv = g.pivot(index="problem", columns="method", values="log_loss")[t.index].reindex(problems)
        L += [f"| {p} | " + " | ".join(f"{v:.4f}" for v in r) + " |" for p, r in pv.iterrows()]
        L.append("")
    L += ["## discount alpha by severity stratum (fitted on all admissions, median over problems)", "",
          "Hypothesis: sicker -> more facts move together -> smaller alpha.", ""]
    for (n, m), g in al.groupby(["window", "method"]):
        cols = [c for c in ["all", "1", "2", "3", "4", "none", "low", "mid", "high"] if c in g and g[c].notna().any()]
        L.append(f"- day<{n} {m}: " + ", ".join(f"{c} {g[c].median():.2f}" for c in cols))
    L += ["", "## non-specific facts: mean |ln LR| over the 10 problems, plain vs stratified by severity", "",
          "| window | fact | plain | stratified |", "|---|---|---:|---:|"]
    L += [f"| {r.window} | {r.fact} | {r.plain:.2f} | {r.strat:.2f} |" for r in ns.itertuples()]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
