#!/usr/bin/env python3
"""Bayes env v0, step 1.6 (d): how much of the EV / doctor gap is the code being read off the test itself?

A discharge code is a summary of observations: C. diff is coded from the C. diff test, anemia from haemoglobin,
AKI from creatinine / BUN, sepsis from the blood culture (bayes_v0.DEFINITIONAL, fixed beforehand, not picked
by LR). For such a pair the model's EV is "look up the answer": theta is near 0 / 1 by construction, so the
test that defines the label tops the menu whether or not a doctor would need it.

  A  decompose (from ev.parquet): each action's EV by problem; which (action, problem) the day's top-EV action
     comes from
  B  recompute with each problem's defining items left out of that problem's belief AND EV (bayes_ev.build
     skip=DEFINITIONAL), then the same rank readouts: all orders, and first-time non-routine orders

Outputs runs/bayes_env/v0/ev/: def_summary.md, ev_nodef.parquet

  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/bayes_ev_def.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.bayes_ev import OUT, build
from scripts.bayes_ev_first import ROUTINE
from scripts.bayes_v0 import DEFINITIONAL

META = {"hadm_id", "day", "action", "EV", "ordered", "H", "before", "EV_std", "H_std"}


def defines(J):
    """{(action, problem)} where J(action) holds an item the problem's code is assigned from."""
    return {(a, k) for a, its in J.items() for k, d in DEFINITIONAL.items() if set(its) & set(d)}


def readout(ev):
    """rank of chosen actions on the full menu and on the first-time non-routine menu; EV sizes."""
    n = ev.groupby(["hadm_id", "day"]).action.transform("size")
    pct = (ev.groupby(["hadm_id", "day"]).EV.rank(method="average") - 1) / (n - 1)
    ch = ev.ordered
    top = ev.loc[ev.groupby(["hadm_id", "day"]).EV.idxmax()]
    rand_top = (ev[ch].groupby(["hadm_id", "day"]).size() / n.groupby([ev.hadm_id, ev.day]).first()).mean()
    day_max = ev.groupby(["hadm_id", "day"]).EV.max()
    best_chosen = ev[ch].groupby(["hadm_id", "day"]).EV.max()
    out = {"pct_all": pct[ch].mean(), "top_ordered_all": top.ordered.mean(), "top_random_all": rand_top,
           "ev_median": ev.EV.median(), "daymax_median": day_max.median(),
           "gap_median": (day_max - best_chosen).dropna().median(),
           "top_actions": top.action.value_counts(normalize=True).head(6)}
    share = ev.groupby("action").ordered.mean()
    f = ev[~ev.action.isin(share.index[share >= ROUTINE]) & ~ev.before]
    g = f.groupby(["hadm_id", "day"])
    menu = g.action.transform("size")
    fp = (g.EV.rank(method="average") - 1) / (menu - 1).replace(0, np.nan)
    keep = menu.gt(1)
    ftop = f[keep].loc[lambda x: x.groupby(["hadm_id", "day"]).EV.idxmax()]
    k_day = f[keep & f.ordered].groupby(["hadm_id", "day"]).size()
    k_day = k_day.reindex(ftop.set_index(["hadm_id", "day"]).index, fill_value=0)
    out.update({"pct_first": fp[f.ordered & keep].mean(), "top_ordered_first": ftop.ordered.mean(),
                "top_random_first": (k_day.values / menu[ftop.index].values).mean(),
                "top_actions_first": ftop.action.value_counts(normalize=True).head(6)})
    return out


def main():
    base = pd.read_parquet(OUT / "ev.parquet")
    problems = [c for c in base.columns if c not in META]
    nodef, calib, J, *_ = build(skip=DEFINITIONAL)
    nodef.to_parquet(OUT / "ev_nodef.parquet", index=False)
    pairs = defines(J)

    # A. each action's EV by problem (mean over decision days), and where the day's top comes from
    by = base.groupby("action")[problems].mean()
    tot = by.sum(1)
    top = base.loc[base.groupby(["hadm_id", "day"]).EV.idxmax()]
    top_k = top[problems].idxmax(1)
    top_src = (top.action + " → " + top_k).value_counts(normalize=True)
    def_share = np.mean([(a, k) in pairs for a, k in zip(top.action, top_k)])

    L = ["# Step 6 (d): is the EV / doctor gap the code being read off its own test?", "",
         "Defining items (fixed beforehand, `bayes_v0.DEFINITIONAL`): "
         + "; ".join(f"{k} ← {', '.join(v)}" for k, v in DEFINITIONAL.items()) + ".",
         f"Actions that carry a defining item: {', '.join(sorted(f'{a} → {k}' for a, k in pairs))}.", "",
         "## A. where the EV comes from (v0 as run)", "",
         f"The day's top-EV action, by the problem that gives it most of its EV ({len(top):,} decision days); "
         f"**{def_share:.0%}** of days the top is a defining pair.", "",
         "| top action → main problem | share of days | defining? |", "|---|---:|---|"]
    for s, v in top_src.head(10).items():
        a, k = s.split(" → ")
        L.append(f"| `{a}` → {k} | {v:.1%} | {'yes' if (a, k) in pairs else ''} |")
    L += ["", "Each action's mean EV (nats ×1000) and its two largest problems (share of its EV):", "",
          "| action | mean EV ×1000 | 1st | 2nd |", "|---|---:|---|---|"]
    for a in tot.sort_values(ascending=False).index[:15]:
        r = (by.loc[a] / tot[a]).sort_values(ascending=False)
        L.append(f"| `{a}` | {tot[a] * 1000:.2f} | {r.index[0]} {r.iloc[0]:.0%}"
                 f"{' (def)' if (a, r.index[0]) in pairs else ''} | {r.index[1]} {r.iloc[1]:.0%} |")

    # B. with / without the defining items
    rb, rn = readout(base), readout(nodef)
    rows = [("chosen action's EV percentile, all orders (0.5 = random)", "pct_all", "{:.2f}"),
            ("day's top-EV action ordered, all orders", "top_ordered_all", "{:.1%}"),
            ("  … if one were picked at random", "top_random_all", "{:.1%}"),
            ("chosen percentile, first-time non-routine", "pct_first", "{:.2f}"),
            ("top-EV ordered, first-time non-routine", "top_ordered_first", "{:.1%}"),
            ("  … at random", "top_random_first", "{:.1%}"),
            ("EV per action x day, median (nats ×1000)", "ev_median", "{:.2f}"),
            ("day's max EV, median (×1000)", "daymax_median", "{:.2f}"),
            ("gap day's max − best chosen, median (×1000)", "gap_median", "{:.2f}")]
    L += ["", "## B. recomputed without the defining items (belief and EV of that problem)", "",
          "| | v0 | without defining |", "|---|---:|---:|"]
    for name, key, fmt in rows:
        scale = 1000 if "×1000" in name else 1
        L.append(f"| {name} | {fmt.format(rb[key] * scale)} | {fmt.format(rn[key] * scale)} |")
    for key, title in [("top_actions", "all orders"), ("top_actions_first", "first-time non-routine")]:
        L += ["", f"Top-EV action, {title}: v0 = "
              + ", ".join(f"`{a}` {v:.0%}" for a, v in rb[key].items())
              + "; without defining = " + ", ".join(f"`{a}` {v:.0%}" for a, v in rn[key].items()) + "."]
    au0 = pd.read_csv(OUT / "calib.csv").query("problem in @DEFINITIONAL").groupby("problem").auc.mean()
    au = calib[calib.problem.isin(DEFINITIONAL)].groupby("problem").auc.mean()
    L += ["", "Belief AUC, v0 → without the defining items (mean over decision days): "
          + ", ".join(f"{k} {au0[k]:.2f} → {v:.2f}" for k, v in au.items()) + "."]
    (OUT / "def_summary.md").write_text("\n".join(L) + "\n")
    print((OUT / "def_summary.md").read_text())


if __name__ == "__main__":
    main()
