#!/usr/bin/env python3
"""Bayes env v0, step 1.6 (c): first-time, non-routine orders only, and per-problem weights.

Step 6 (bayes_ev.py) found doctors' lab / micro orders nearly unrelated to EV, dominated by habit (the daily
panels) and monitoring (re-ordering what was already done). The orders most likely to be information seeking
are the FIRST time a non-routine test is ordered in a stay. Routine = ordered on >= 50% of decision days.

On that subset (rows = decision day x non-routine action not done before in the stay):
  rank      percentile of the chosen action's EV among the not-yet-done non-routine actions of that day
  within    per action, AUC of EV for "ordered today" within day strata
  pooled    ordered ~ FE(action x day) + slope(action) x total uncertainty + beta x EV (std within action)
  weights   the same with the 10 per-problem EV_k (std within action) in place of EV: the weight each problem
            carries in the doctors' choices, if they were seeking information (w_k = 1 in step 6)

Reads runs/bayes_env/v0/ev/ev.parquet. Outputs runs/bayes_env/v0/ev/first_summary.md, first_weights.csv

  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/bayes_ev_first.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from scripts.bayes_ev import BOOT, OUT, strat_auc

ROUTINE = 0.5


def fit_fe(ev, cols, rng):
    """logistic with FE(action x day) + action-specific uncertainty slope + cols; coefficients of cols + CI."""
    n = len(ev)
    r = np.arange(n)
    cell = pd.factorize(ev.action + "@" + ev.day.astype(str))[0]
    act = pd.factorize(ev.action)[0]
    X = sparse.hstack([sparse.csr_matrix((np.ones(n), (r, cell)), shape=(n, cell.max() + 1)),
                       sparse.csr_matrix((ev.H_std.values, (r, act)), shape=(n, act.max() + 1)),
                       sparse.csr_matrix(ev[cols].values)]).tocsr()
    y = ev.ordered.values.astype(int)

    def fit(idx):
        return LogisticRegression(C=1e4, max_iter=3000, fit_intercept=False).fit(X[idx], y[idx]).coef_[0, -len(cols):]

    est = fit(r)
    adm = ev.hadm_id.values
    pos = pd.Series(r).groupby(adm).apply(np.array)
    boots = np.array([fit(np.concatenate(pos.loc[rng.choice(pos.index, len(pos))].values)) for _ in range(BOOT)])
    return est, np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


def main():
    ev = pd.read_parquet(OUT / "ev.parquet")
    problems = [c for c in ev.columns if c not in {"hadm_id", "day", "action", "EV", "ordered", "H", "before",
                                                    "EV_std", "H_std"}]
    share = ev.groupby("action").ordered.mean()
    routine = sorted(share.index[share >= ROUTINE])
    f = ev[~ev.action.isin(routine) & ~ev.before].copy()
    f["EV_std"] = f.groupby("action").EV.transform(lambda x: (x - x.mean()) / (x.std() or 1))
    for k in problems:
        f[k + "_std"] = f.groupby("action")[k].transform(lambda x: (x - x.mean()) / (x.std() or 1))
    print(f"{len(f):,} rows (decision day x not-yet-done non-routine action), {int(f.ordered.sum()):,} ordered")

    # 1. rank on the menu of not-yet-done non-routine actions
    g = f.groupby(["hadm_id", "day"])
    f["menu"] = g.action.transform("size")
    f["pct"] = (g.EV.rank(method="average") - 1) / (f.menu - 1).replace(0, np.nan)
    pop = f.groupby("action").ordered.sum()
    f["pop_pct"] = (f.groupby(["hadm_id", "day"]).action.transform(lambda a: a.map(pop).rank(method="average"))
                    - 1) / (f.menu - 1).replace(0, np.nan)
    ch = f[f.ordered & f.menu.gt(1)]
    top = f[f.menu.gt(1)].loc[lambda x: x.groupby(["hadm_id", "day"]).EV.idxmax()]
    k_per_day = ch.groupby(["hadm_id", "day"]).size().reindex(top.set_index(["hadm_id", "day"]).index, fill_value=0)
    rand_top = (k_per_day.values / top.menu.values).mean()

    # 2. within action
    w = []
    for a, gg in f.groupby("action"):
        y = gg.ordered.values.astype(int)
        w.append({"action": a, "first_orders": int(y.sum()), "rows": len(gg),
                  "auc_EV": strat_auc(y, gg.EV.values, gg.day.values),
                  "auc_H": strat_auc(y, gg.H.values, gg.day.values)})
    w = pd.DataFrame(w).sort_values("first_orders", ascending=False)

    # 3. pooled, 4. per-problem weights
    rng = np.random.default_rng(0)
    beta, lo, hi = fit_fe(f, ["EV_std"], rng)
    wk, wlo, whi = fit_fe(f, [k + "_std" for k in problems], rng)
    wt = pd.DataFrame({"problem": problems, "weight": wk, "ci_lo": wlo, "ci_hi": whi})
    wt.to_csv(OUT / "first_weights.csv", index=False)
    # which action carries each problem's EV: for reading the weights
    lead = {k: f.groupby("action")[k].mean().sort_values(ascending=False).head(3).index.tolist() for k in problems}

    L = ["# Step 6 (c): first-time non-routine orders, and per-problem weights", "",
         f"Routine (ordered on >= {ROUTINE:.0%} of decision days, excluded): {', '.join(routine)}. Rows = decision "
         f"day x non-routine action NOT done earlier in the stay: {len(f):,}; first-time orders {int(f.ordered.sum()):,}.",
         "", "## 1. rank among the not-yet-done non-routine actions of the day (0.5 = random)", "",
         f"- chosen action's EV percentile **{ch.pct.mean():.2f}**; by habit (how often each is ordered first): "
         f"{ch.pop_pct.mean():.2f}",
         f"- the day's highest-EV option was ordered on {top.ordered.mean():.1%} of decision days "
         f"(random, given how many were ordered: {rand_top:.1%})", "",
         "| day | EV percentile of chosen | n |", "|---|---:|---:|"]
    L += [f"| {d} | {x.pct.mean():.2f} | {len(x):,} |" for d, x in ch.groupby("day")]
    L += ["", "## 2. within an action (AUC within day strata; 0.5 = no relation)", "",
          "| action | first orders | AUC EV | AUC H |", "|---|---:|---:|---:|"]
    L += [f"| `{r.action}` | {r.first_orders:,} | {r.auc_EV:.2f} | {r.auc_H:.2f} |" for r in w.itertuples()]
    L += ["", f"Median: AUC EV {w.auc_EV.median():.2f}, AUC H {w.auc_H.median():.2f}.", "",
          "## 3. pooled: ordered ~ FE(action x day) + slope(action) x uncertainty + beta x EV", "",
          f"beta = **{beta[0]:+.3f}** (OR per SD {np.exp(beta[0]):.2f}), 95% CI {lo[0]:+.3f} .. {hi[0]:+.3f} "
          f"({BOOT} bootstraps over admissions)", "",
          "## 4. implied problem weights: the 10 EV_k side by side (std within action)", "",
          "Positive = the doctors order more where the test would tell more about this problem.", "",
          "| problem | weight | 95% CI | actions carrying most of its EV |", "|---|---:|---|---|"]
    L += [f"| {r.problem} | {r.weight:+.3f} | {r.ci_lo:+.3f} .. {r.ci_hi:+.3f} | {', '.join(lead[r.problem])} |"
          for r in wt.sort_values("weight", ascending=False).itertuples()]
    (OUT / "first_summary.md").write_text("\n".join(L) + "\n")
    print((OUT / "first_summary.md").read_text())


if __name__ == "__main__":
    main()
