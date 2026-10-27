#!/usr/bin/env python3
"""Bayes env v0, step 1.6: is the doctor's (non-imaging) test choice related to its expected information gain?

Decision moment = admission x day d (day -1 = before admission .. day 7) on which >= 1 candidate action was done.
The belief at d uses only facts of days < d (results the doctor could have seen); the menu = the 28 candidate
actions (runs/bayes_env/v0/actions/actions.csv, in_space).

Belief (one fixed environment, BAYES_ENV.md 7.8): ONE conditional likelihood table p(o_t | o_{t-1}, s_k)
(prev = "first" for a first observation), estimated on whole stays of the training folds; calibrated logit
  z_k = b0 + b1 logit prior + alpha_first * (evidence of first observations) + alpha_rep * (evidence of repeats)
fitted on all decision days stacked, cross-fitted by patient fold.

EV(a) = sum_k I(s_k; o_J(a)) under that belief (section 4.1), exactly: the items of J(a) are independent given
s_k, so the distribution of the discounted log-likelihood ratio L = sum_j alpha_j ln LR_j(o_j) under s=1 and
s=0 is built by convolution (rounded to a grid), then
  EV_k = sum_L [b p1(L) + (1-b) p0(L)] KL( sigmoid(z+L) || sigmoid(z) ).
Each item is predicted from its own last result (conditional theta); its alpha is first or repeat accordingly.
w_k = 1. The reflex smear riding on a CBC is left out of EV(cbc) (4.7% of CBCs).

Tests:
  rank      percentile of each chosen action's EV among the 28 that day (0 = lowest, 1 = highest);
            baselines: random 0.5, and ranking by how often each action is ordered (habit)
  within    per action, over decision days: is it ordered more on days its EV is high? AUC within day strata,
            next to the same AUC for the patient's total uncertainty sum_k H(b_k) (is it the action-specific
            information or just "the patient is unclear")
  pooled    logistic: ordered ~ action x day fixed effects + action-specific slope on total uncertainty
            + beta * EV (standardised within action); beta's CI by bootstrap over admissions

Outputs runs/bayes_env/v0/ev/: summary.md, ev.parquet (decision x action), within.csv, calib.csv

  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/bayes_ev.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from scripts.bayes_seq import FIRST, daily
from scripts.bayes_v0 import A, DATA, FOLDS, calib_slope, demographic_prior, load, logit

OUT = DATA / "ev"
DAYS = range(-1, 8)
GRID = 0.005
BOOT = 30
KEYS = ["item", "prev", "value"]


# ---------------------------------------------------------------- likelihood
def theta_table(d, s, levels, problems):
    """theta_k(value | item, prev, s=1 / s=0), shrunk as in bayes_v0.ln_lr. Index item, prev, value."""
    x = d.merge(s, on="hadm_id")
    pooled = x.groupby(KEYS).size()
    lv = pooled.index.get_level_values("item").map(levels).values
    pooled = (pooled + 1) / (pooled.groupby(level=["item", "prev"]).transform("sum") + lv)
    out = {}
    for k in problems:
        g = x.assign(one=x[k], zero=1 - x[k]).groupby(KEYS)[["one", "zero"]].sum()
        tot = g.groupby(level=["item", "prev"]).transform("sum")
        out[k] = (g + A * pooled.values[:, None]) / (tot + A)
    return out


def as_dict(th):
    """{(item, prev): (theta1, theta0)} over the item's values, for fast lookup."""
    return {key: (g.one.values, g.zero.values) for key, g in th.groupby(level=["item", "prev"])}


# ---------------------------------------------------------------- belief
def row_evidence(dd, coh, s, levels, problems):
    """ln LR of every daily fact, out of fold, plus the fold tables (for EV)."""
    parts, tables = [], {}
    for f in range(FOLDS):
        tr, te = coh[coh.fold != f], coh[coh.fold == f]
        th = theta_table(dd[dd.hadm_id.isin(tr.index)], tr[problems].reset_index(), levels, problems)
        tables[f] = th
        lr = pd.DataFrame({k: np.log(th[k].one / th[k].zero) for k in problems})
        parts.append(dd[dd.hadm_id.isin(te.index)].join(lr, on=KEYS))
    ev = pd.concat(parts)
    # a (prev, value) pair never seen in training: use the first-observation LR of that value, else 0
    miss = ev[problems[0]].isna()
    if miss.any():
        alt = ev.loc[miss, ["item", "value"]].assign(prev=FIRST)
        for f in range(FOLDS):
            m = miss & ev.hadm_id.map(coh.fold).eq(f)
            th = tables[f]
            lr = pd.DataFrame({k: np.log(th[k].one / th[k].zero) for k in problems})
            ev.loc[m, problems] = alt[m[miss].values].join(lr, on=KEYS)[problems].values
    return ev.fillna({k: 0.0 for k in problems}), tables


def beliefs(dec, ev, coh, problems):
    """calibrated z_k at every decision (hadm, day): facts of days < day, cross-fitted two-alpha model."""
    prior = pd.DataFrame(index=coh.index, columns=problems, dtype=float)
    for f in range(FOLDS):
        tr, te = coh[coh.fold != f], coh[coh.fold == f]
        prior.loc[te.index] = demographic_prior(tr, te, problems)
    first = ev.prev.eq(FIRST)
    snaps = []
    for d in DAYS:
        hs = dec[dec.day.eq(d)].hadm_id.unique()
        e = ev[ev.day.lt(d) & ev.hadm_id.isin(hs)]
        e1 = e[first.loc[e.index]].groupby("hadm_id")[problems].sum().reindex(hs, fill_value=0)
        e2 = e[~first.loc[e.index]].groupby("hadm_id")[problems].sum().reindex(hs, fill_value=0)
        snaps.append((d, hs, e1, e2))
    H = np.concatenate([s[1] for s in snaps])
    D = np.concatenate([[s[0]] * len(s[1]) for s in snaps])
    fold = coh.fold.reindex(H).values
    Z = pd.DataFrame(index=pd.MultiIndex.from_arrays([H, D], names=["hadm_id", "day"]), columns=problems, dtype=float)
    coef, calib = {}, []
    for k in problems:
        X = np.column_stack([logit(prior[k].reindex(H).astype(float)),
                             np.concatenate([s[2][k].values for s in snaps]),
                             np.concatenate([s[3][k].values for s in snaps])])
        y = coh[k].reindex(H).values
        z = np.empty(len(y))
        for f in range(FOLDS):
            m = LogisticRegression(C=1e6, max_iter=2000).fit(X[fold != f], y[fold != f])
            z[fold == f] = m.decision_function(X[fold == f])
            coef[(k, f)] = m.coef_[0, 1:]  # alpha_first, alpha_rep used by EV for fold f
        Z[k] = z
        for d in DAYS:
            m = D == d
            calib.append({"problem": k, "day": d, "n": int(m.sum()), "slope": calib_slope(y[m], z[m]),
                          "auc": roc_auc_score(y[m], z[m]) if 0 < y[m].sum() < m.sum() else np.nan})
    return Z, coef, pd.DataFrame(calib)


# ---------------------------------------------------------------- EV
def ll_dist(items):
    """items = [(theta1 array, theta0 array, discounted lnLR array)] -> grid L values, p1(L), p0(L)."""
    v, p1, p0 = np.zeros(1), np.ones(1), np.ones(1)
    for t1, t0, l in items:
        v = (v[:, None] + l[None, :]).ravel()
        p1 = (p1[:, None] * t1[None, :]).ravel()
        p0 = (p0[:, None] * t0[None, :]).ravel()
        key = np.round(v / GRID).astype(np.int64)
        u, inv = np.unique(key, return_inverse=True)
        v, p1, p0 = u * GRID, np.bincount(inv, p1), np.bincount(inv, p0)
    return v, p1, p0


def info_gain(z, v, p1, p0):
    """exact expected KL of the binary belief, for an array of prior logits z."""
    b = 1 / (1 + np.exp(-z))[:, None]
    post = 1 / (1 + np.exp(-(z[:, None] + v[None, :])))
    post = np.clip(post, 1e-12, 1 - 1e-12)
    bb = np.clip(b, 1e-12, 1 - 1e-12)
    kl = post * np.log(post / bb) + (1 - post) * np.log((1 - post) / (1 - bb))
    return ((b * p1[None, :] + (1 - b) * p0[None, :]) * kl).sum(1)


def expected_gain(dec, Z, coef, tables, prev, J, coh, problems, skip=None):
    """EV_k(a) for every decision x candidate action; grouped by (fold, action, previous results of J(a)).
    skip = {problem: [items]}: items left out of problem k's channel (the items its code is assigned from)."""
    skip = skip or {}
    dec = dec.assign(fold=dec.hadm_id.map(coh.fold).values)
    th = {(f, k): as_dict(tables[f][k]) for f in tables for k in problems}
    rows = []
    for a, items in J.items():
        pv = prev.reindex(columns=items).fillna(FIRST)  # prev is in dec's row order
        pat = pv.astype(str).agg("|".join, axis=1).values
        for (f, p), idx in pd.DataFrame({"f": dec.fold.values, "p": pat}).groupby(["f", "p"]).indices.items():
            prevs = p.split("|")
            sub = dec.iloc[idx]
            out = {"hadm_id": sub.hadm_id.values, "day": sub.day.values, "action": a}
            for k in problems:
                a1, a2 = coef[(k, f)]
                its = []
                for it, pr in zip(items, prevs):
                    if it in skip.get(k, ()):
                        continue
                    # predicted from its own last result; an unseen last value falls back to a first observation
                    seen = pr != FIRST and (it, pr) in th[(f, k)]
                    t = th[(f, k)].get((it, pr) if seen else (it, FIRST))
                    if t is None:
                        continue
                    its.append((t[0], t[1], (a2 if seen else a1) * np.log(t[0] / t[1])))
                z = Z[k].loc[pd.MultiIndex.from_frame(sub[["hadm_id", "day"]])].values.astype(float)
                out[k] = info_gain(z, *ll_dist(its)) if its else np.zeros(len(z))
            rows.append(pd.DataFrame(out))
    ev = pd.concat(rows, ignore_index=True)
    ev["EV"] = ev[problems].sum(1)
    return ev


# ---------------------------------------------------------------- tests
def strat_auc(y, x, strata):
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        if 0 < y[m].sum() < m.sum():
            num += roc_auc_score(y[m], x[m]) * m.sum()
            den += m.sum()
    return num / den if den else np.nan


def pooled(ev, rng):
    """beta of standardised EV in: ordered ~ FE(action x day) + slope(action) * uncertainty + beta * EV."""
    acts = sorted(ev.action.unique())
    n = len(ev)
    cell = pd.factorize(ev.action + "@" + ev.day.astype(str))[0]
    act = pd.factorize(ev.action)[0]
    r = np.arange(n)
    fe = sparse.csr_matrix((np.ones(n), (r, cell)), shape=(n, cell.max() + 1))
    hu = sparse.csr_matrix((ev.H_std.values, (r, act)), shape=(n, act.max() + 1))
    X = sparse.hstack([fe, hu, sparse.csr_matrix(ev.EV_std.values[:, None])]).tocsr()
    y = ev.ordered.values.astype(int)

    def fit(idx):
        return LogisticRegression(C=1e4, max_iter=3000, fit_intercept=False).fit(X[idx], y[idx]).coef_[0, -1]

    beta = fit(np.arange(len(y)))
    adm = ev.hadm_id.values
    ua = np.unique(adm)
    pos = pd.Series(np.arange(len(adm))).groupby(adm).apply(np.array)
    boots = []
    for _ in range(BOOT):
        pick = rng.choice(ua, len(ua))
        boots.append(fit(np.concatenate(pos.loc[pick].values)))
    return beta, np.percentile(boots, [2.5, 97.5]), len(acts)


def build(skip=None):
    """the decision x action EV table; skip = {problem: [items]} leaves those items out of problem k's belief
    and EV. Returns ev (with ordered, H, before), calib, J, chosen, cand, events."""
    s, o, coh, problems, levels = load()
    acts = pd.read_csv(DATA / "actions/actions.csv")
    cand = acts[acts.in_space]
    J = {r.action: r.J.split(";") for r in cand.itertuples()}
    events = pd.read_parquet(DATA / "actions/events.parquet")
    chosen = events[events.action.isin(J) & events.day.between(min(DAYS), max(DAYS))]
    dec = chosen[["hadm_id", "day"]].drop_duplicates().astype({"day": int}).sort_values(["hadm_id", "day"])
    dec = dec.reset_index(drop=True)
    print(f"{len(dec):,} decision days, {len(J)} candidate actions")

    dd = daily(o, None)
    rev, tables = row_evidence(dd, coh, s, levels, problems)
    for k, its in (skip or {}).items():
        rev.loc[rev.item.isin(its), k] = 0.0
    Z, coef, calib = beliefs(dec, rev, coh, problems)
    print("beliefs ready")

    # each item's last result before the decision day
    left = dec.merge(pd.DataFrame({"item": sorted(set(sum(J.values(), [])))}), how="cross")
    right = dd[["hadm_id", "item", "day", "value"]].astype({"day": int})
    pv = pd.merge_asof(left.sort_values("day"), right.sort_values("day"), on="day", by=["hadm_id", "item"],
                       allow_exact_matches=False)
    prev = pv.pivot_table(index=["hadm_id", "day"], columns="item", values="value", aggfunc="first")
    prev = prev.reindex(pd.MultiIndex.from_frame(dec[["hadm_id", "day"]]))

    ev = expected_gain(dec, Z, coef, tables, prev, J, coh, problems, skip)
    got = chosen.groupby(["hadm_id", "day"]).action.agg(set)
    ev["ordered"] = [a in got.get((h, d), set()) for h, d, a in zip(ev.hadm_id, ev.day, ev.action)]
    b = 1 / (1 + np.exp(-Z.astype(float)))
    Hd = -(b * np.log(b) + (1 - b) * np.log(1 - b)).sum(1).rename("H")
    ev = ev.join(Hd, on=["hadm_id", "day"])
    # already done on an earlier day of the stay: a repeat (monitoring) has low EV by construction (alpha_rep,
    # static s), so every within-action comparison is also made within this flag
    first_day = events.groupby(["hadm_id", "action"]).day.min().astype(float)
    fd = first_day.reindex(pd.MultiIndex.from_frame(ev[["hadm_id", "action"]])).values
    ev["before"] = np.nan_to_num(fd, nan=1e9) < ev.day.values
    ev["EV_std"] = ev.groupby("action").EV.transform(lambda x: (x - x.mean()) / (x.std() or 1))
    ev["H_std"] = (ev.H - ev.H.mean()) / ev.H.std()
    return ev, calib, J, chosen, cand, events


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev, calib, J, chosen, cand, events = build()
    calib.to_csv(OUT / "calib.csv", index=False)
    ev.to_parquet(OUT / "ev.parquet", index=False)
    print("EV ready")

    # rank among the menu
    ev["pct"] = ev.groupby(["hadm_id", "day"]).EV.rank(pct=False, method="average").sub(1) / (len(J) - 1)
    pop = chosen.groupby("action").size().reindex(list(J)).fillna(0)
    ev["pop_pct"] = ev.action.map((pop.rank(method="average") - 1) / (len(J) - 1))
    ch = ev[ev.ordered]
    top = ev.loc[ev.groupby(["hadm_id", "day"]).EV.idxmax()]
    # within action
    w = []
    for a, g in ev.groupby("action"):
        y, st = g.ordered.values.astype(int), g.day.values
        st2 = (g.day.astype(str) + np.where(g.before, "b", "n")).values
        w.append({"action": a, "zh": cand.set_index("action").zh.get(a, ""), "ordered_days": int(y.sum()),
                  "share": y.mean(), "p_before": g.ordered[g.before].mean(),
                  "auc_EV_day": strat_auc(y, g.EV.values, st),
                  "auc_EV": strat_auc(y, g.EV.values, st2), "auc_H": strat_auc(y, g.H.values, st2)})
    w = pd.DataFrame(w).sort_values("ordered_days", ascending=False)
    w.to_csv(OUT / "within.csv", index=False)
    beta0, ci0, _ = pooled(ev, np.random.default_rng(0))
    beta, ci, _ = pooled(ev.assign(day=ev.day.astype(str) + np.where(ev.before, "b", "n")), np.random.default_rng(0))

    cs = calib.groupby("day")[["slope", "auc"]].mean()
    L = ["# Step 6: doctors' test choices vs expected information gain (labs + micro, v0)", "",
         f"{len(dec):,} decision days (admission x day, day -1 .. 7), {len(J)} candidate actions, "
         f"{len(ch):,} actions ordered. Belief at day d = facts of days < d; one conditional likelihood "
         "table + two alphas (first / repeat), cross-fitted. EV in nats, sum over the 10 problems (w = 1).", "",
         "## belief calibration at the decision days (mean over problems)", "",
         "| day | " + " | ".join(str(d) for d in cs.index) + " |", "|---|" + "---:|" * len(cs),
         "| slope | " + " | ".join(f"{v:.2f}" for v in cs.slope) + " |",
         "| AUC | " + " | ".join(f"{v:.2f}" for v in cs.auc) + " |", "",
         "## 1. where does the chosen action rank on the menu (percentile; 0.5 = random)", "",
         f"- by EV: **{ch.pct.mean():.2f}**; by habit (how often each action is ordered overall): {ch.pop_pct.mean():.2f}",
         f"- the day's highest-EV action was ordered on {top.ordered.mean():.1%} of decision days "
         f"(random pick of one of {len(J)}: {ch.groupby(['hadm_id', 'day']).size().mean() / len(J):.1%} "
         "given the number ordered per day)", "",
         "| day | EV percentile of chosen | n chosen |", "|---|---:|---:|"]
    L += [f"| {d} | {g.pct.mean():.2f} | {len(g):,} |" for d, g in ch.groupby("day")]
    L += ["", "## 2. within an action: ordered more on its high-EV days? (AUC; 0.5 = no relation)", "",
          "`AUC EV | day` = within day strata only; `AUC EV` = within day x already-done strata (a repeat has low "
          "EV by construction); `AUC H` = same strata, the patient's total uncertainty instead of the action's EV. "
          "`P | done before` = how often it is ordered again once it has been done.", "",
          "| action | | days ordered | P \\| done before | AUC EV \\| day | AUC EV | AUC H |",
          "|---|---|---:|---:|---:|---:|---:|"]
    L += [f"| `{r.action}` | {r.zh if isinstance(r.zh, str) else ''} | {r.ordered_days:,} | {r.p_before:.2f} | "
          f"{r.auc_EV_day:.2f} | {r.auc_EV:.2f} | {r.auc_H:.2f} |" for r in w.itertuples()]
    L += ["", f"Median over actions: AUC EV | day {w.auc_EV_day.median():.2f}, AUC EV {w.auc_EV.median():.2f}, "
          f"AUC H {w.auc_H.median():.2f}.", "",
          "## 3. pooled: ordered ~ FE(action x day [x already done]) + slope(action) x total uncertainty "
          "+ beta x EV (std within action)", "",
          f"- FE action x day: beta = {beta0:+.3f} (OR per SD {np.exp(beta0):.2f}), 95% CI {ci0[0]:+.3f} .. {ci0[1]:+.3f}",
          f"- FE action x day x already done: beta = **{beta:+.3f}** (OR per SD {np.exp(beta):.2f}), "
          f"95% CI {ci[0]:+.3f} .. {ci[1]:+.3f}", f"- CIs: {BOOT} bootstraps over admissions."]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
