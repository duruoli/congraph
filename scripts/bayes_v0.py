#!/usr/bin/env python3
"""Bayes env v0, step 1.3-1.4: estimate p(o | s), update beliefs, validate on held-out patients.

Per problem k, independently (BAYES_ENV.md sections 1-2):
  prior     p(s_k = 1 | same sex, age +-5) among training admissions, shrunk toward the overall rate:
            (n_k + M * p_all) / (n + M)
  theta     p(o_j = v | s_k = x, did j), shrunk toward the item's pooled distribution with A pseudo-admissions
            (rare results -> LR near 1), estimated in the SAME window
  belief    logit b_k = logit prior + sum_j ln LR_kj(o_j)

Window: facts with day < N (day -1 = before admission, 0 = first 24 h; N = all -> whole stay), worst result
per item in the window. Validation: 5 folds by PATIENT (one patient's admissions never split), every admission
predicted once out of fold. Reported per problem: AUC of the prior alone and of the posterior; calibration
slope (logistic regression of the outcome on the posterior logit; < 1 = overconfident, the signature of
double-counted evidence); and the posterior AUC without the items the code is defined by.

Outputs runs/bayes_env/v0/eval/: summary.md, metrics.csv, lr.csv (ln LR on all admissions, for reading)

  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/bayes_v0.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from scripts.bayes_data import OUT as DATA, RAW as RAW_DIR, cohort

OUT = DATA / "eval"
WINDOWS = [1, 3, None]  # first 24 h (+ before admission), first 3 days, whole stay
M = 20                  # prior shrinkage strength, in admissions
A = 20                  # likelihood shrinkage toward the item's pooled distribution, in admissions
AGE_BAND = 5
FOLDS = 5
# items the discharge code is (largely) assigned from: reported with and without them
DEFINITIONAL = {"anemia": ["lab:51222"], "aki": ["lab:50912", "lab:51006"],
                "cdiff": ["micro:STOOL|C. difficile"], "sepsis": ["micro:BLOOD CULTURE|Blood Culture, Routine"]}
ABNORMAL = {"high", "low", "abnormal", "positive"}


def window(o, n):
    w = o if n is None else o[o.day < n]
    sev = w.value.isin(ABNORMAL).astype(float) + w.dev.abs().fillna(0)
    return w.assign(sev=sev).sort_values("sev").groupby(["hadm_id", "item"]).tail(1)[["hadm_id", "item", "value"]]


def demographic_prior(train, test, problems):
    """p(s_k | same sex, age +-AGE_BAND) from train, shrunk toward the train rate; one row per test admission."""
    p_all = train[problems].mean()
    out = pd.DataFrame(index=test.index, columns=problems, dtype=float)
    ages = np.arange(0, 121)
    for g, tr in train.groupby("gender"):
        n_age = tr.groupby("age").size().reindex(ages, fill_value=0)
        k_age = tr.groupby("age")[problems].sum().reindex(ages, fill_value=0)
        band = lambda x: x.rolling(2 * AGE_BAND + 1, center=True, min_periods=1).sum()
        n_b, k_b = band(n_age), band(k_age)
        prior = (k_b + M * p_all).div(n_b + M, axis=0)
        te = test[test.gender.eq(g)]
        out.loc[te.index] = prior.loc[te.age].values
    return out


def ln_lr(ow, s, levels, problems, keys=("item", "value")):
    """ln LR_kj(v) for every (item, value) seen in ow; ow = windowed o of the training admissions.
    Extra keys (e.g. a severity bin) make the likelihood conditional on them: p(o_j | s_k, bin)."""
    keys = list(keys)
    by = [k for k in keys if k != "value"]  # theta normalises over the values of one item (x bin)
    d = ow.merge(s, on="hadm_id")
    # shrink theta toward the item's pooled distribution (A pseudo-admissions): a result seen once or twice
    # moves both theta_1 and theta_0 to the same pooled rate, so its LR stays near 1 instead of exploding
    pooled = d.groupby(keys).size()
    lv = pooled.index.get_level_values("item").map(levels).values
    pooled = (pooled + 1) / (pooled.groupby(level=by).transform("sum") + lv)
    rows = {}
    for k in problems:
        g = d.assign(one=d[k], zero=1 - d[k]).groupby(keys)[["one", "zero"]].sum()
        tot = g.groupby(level=by).transform("sum")
        theta = (g + A * pooled.values[:, None]) / (tot + A)
        rows[k] = np.log(theta.one / theta.zero)
    return pd.DataFrame(rows)


def out_of_fold(ow, coh, problems, levels, keys=("item", "value"), skip=None):
    """prior and summed evidence for every admission, each from the folds it is not in.
    skip = {problem: [items]} left out of that problem's evidence."""
    prior = pd.DataFrame(index=coh.index, columns=problems, dtype=float)
    evid = prior.copy()
    for f in range(FOLDS):
        tr, te = coh[coh.fold != f], coh[coh.fold == f]
        prior.loc[te.index] = demographic_prior(tr, te, problems)
        lr = ln_lr(ow[ow.hadm_id.isin(tr.index)], tr[problems].reset_index(), levels, problems, keys)
        ev = ow[ow.hadm_id.isin(te.index)].join(lr, on=list(keys)).fillna({k: 0 for k in problems})
        for k in problems:
            e = ev[~ev.item.isin((skip or {}).get(k, []))]
            evid.loc[te.index, k] = e.groupby("hadm_id")[k].sum().reindex(te.index, fill_value=0)
    return prior, evid


def load():
    s = pd.read_parquet(DATA / "s.parquet")
    o = pd.read_parquet(DATA / "o.parquet")
    problems = [c for c in s.columns if c != "hadm_id"]
    coh = cohort()[["hadm_id", "subject_id", "gender", "age"]].merge(s, on="hadm_id").set_index("hadm_id")
    rng = np.random.default_rng(0)
    subj = coh.subject_id.unique()
    fold_of = pd.Series(rng.permutation(len(subj)) % FOLDS, index=subj)
    coh["fold"] = coh.subject_id.map(fold_of).values
    return s, o, coh, problems, o.groupby("item").value.nunique()


def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1 - p))


def calib_slope(y, logit):
    m = LogisticRegression(C=1e6).fit(logit.reshape(-1, 1), y)
    return float(m.coef_[0, 0])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s, o, coh, problems, levels = load()
    metrics = []
    for n in WINDOWS:
        ow = window(o, n)
        prior, evid = out_of_fold(ow, coh, problems, levels)
        _, evid_nodef = out_of_fold(ow, coh, problems, levels, skip=DEFINITIONAL)
        post = logit(prior) + evid
        post_nodef = logit(prior) + evid_nodef
        for k in problems:
            y = coh[k].values
            lp = post[k].values.astype(float)
            metrics.append({"window": "all" if n is None else f"day<{n}", "problem": k, "n_pos": int(y.sum()),
                            "auc_prior": roc_auc_score(y, prior[k].astype(float)),
                            "auc_post": roc_auc_score(y, lp),
                            "auc_post_no_def": roc_auc_score(y, post_nodef[k].astype(float)) if k in DEFINITIONAL else np.nan,
                            "calib_slope": calib_slope(y, lp),
                            "mean_pred": float((1 / (1 + np.exp(-lp))).mean()), "rate": float(y.mean())})
    met = pd.DataFrame(metrics)
    met.to_csv(OUT / "metrics.csv", index=False)

    # ln LR on all admissions, whole-stay window, for reading which facts carry each problem
    w_all = window(o, None)
    lr_all = ln_lr(w_all, s, levels, problems)
    name = pd.read_csv(DATA / "items.csv").set_index("item").name
    lr_all.insert(0, "name", lr_all.index.get_level_values("item").map(name))
    lr_all.insert(1, "admissions", w_all.groupby(["item", "value"]).size().reindex(lr_all.index))
    lr_all.to_csv(OUT / "lr.csv")

    L = ["# Bayes v0: likelihoods, belief update, held-out validation", "",
         f"Primary-IBD admissions n = {len(coh):,}; {FOLDS} folds by patient; prior = same sex, age ±{AGE_BAND}, "
         f"shrunk toward the overall rate (M = {M}). Window: day < N (day −1 = before admission, 0 = first 24 h).", "",
         "AUC: prior alone vs prior + facts. Calibration slope: 1 = honest, < 1 = overconfident "
         "(evidence counted twice). `no def` = posterior without the items the code is defined by.", ""]
    for w, g in met.groupby("window", sort=False):
        L += [f"## window {w}", "", "| problem | cases | AUC prior | AUC post | AUC post no def | calib slope | mean pred / rate |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        L += [f"| {r.problem} | {r.n_pos} | {r.auc_prior:.2f} | {r.auc_post:.2f} | "
              f"{'' if np.isnan(r.auc_post_no_def) else f'{r.auc_post_no_def:.2f}'} | {r.calib_slope:.2f} | "
              f"{r.mean_pred:.2f} / {r.rate:.2f} |" for r in g.itertuples()]
        L.append("")
    L += ["## strongest facts per problem (whole stay, all admissions, |ln LR| top 5 among results seen in ≥ 30)", ""]
    common = lr_all[lr_all.admissions >= 30]
    for k in problems:
        top = common[k].abs().sort_values(ascending=False).head(5).index
        L.append(f"- **{k}**: " + "; ".join(f"{lr_all.loc[i, 'name']} {i[1]} ({lr_all.loc[i, k]:+.2f})" for i in top))
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
