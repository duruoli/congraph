"""Does recurrence carry weight? Count encodings vs presence for predicting MED→SURG.

Same cases/controls/decision-point t0 as transfer_bow_model.py. For each term:
  binary   = mentioned at all in the window
  distinct = number of distinct calendar days with a mention (recurrence, same-day copies collapse)
  raw      = total mentions (every repeat counts, copies included)
Counts enter as log1p, every encoding is MaxAbs-scaled, and every model also gets
log1p(number of records in the window) so "more documentation" is not read as weight.
Feature sets: facts (report text, abnormal labs, organisms) and facts+actions.

Part 2: dose-response — share transferred by number of mention-days (0/1/2/3+)
for pre-specified problem terms (simple negation filter on report sentences) and
for terms chosen on one half of patients, plotted on the other half.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MaxAbsScaler

from scripts.transfer_bow_model import H, STOP, build_samples, events

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = [("cum_lag0", 0), ("cum_lag24", -24), ("cum_lag48", -48)]
ENCODINGS = ["binary", "distinct", "raw"]
NEG = re.compile(r"\b(no|not|without|negative for|free of|resolution of|resolved|rule out|r/o|evaluate for|eval for)\b", re.I)
PRESPECIFIED = {
    "perforation": r"perforat|free air|pneumoperitoneum|free intraperitoneal air",
    "obstruction": r"obstruct|dilated (small )?bowel|dilated loops|transition point",
    "abscess/collection": r"abscess|fluid collection|phlegmon",
    "fistula": r"fistul",
    "stricture": r"strictur|narrowing",
    "CRP high (lab)": r"^lab:c_reactive_protein_high$",
}


def window_units(samples, ev, end, facts_only):
    """Per sample, per calendar day: report texts and one token string. Returns unit tables."""
    by = dict(tuple(ev.groupby("hadm_id")))
    text_units, tok_units, n_records = [], [], []
    for i, s in enumerate(samples.itertuples()):
        e = by.get(s.hadm_id)
        if e is None:
            n_records.append(0)
            continue
        w = e[(e.t >= s.admittime - 24 * H) & (e.t < s.t0 + end * H)]
        w = w[w.kind != "tok"] if facts_only else w
        n_records.append(len(w))
        day = w.t.dt.normalize()
        for d, g in w.groupby(day):
            text_units += [(i, d, x) for x in g[g.kind == "text"].payload]
            toks = g[g.kind != "text"].payload
            if len(toks):
                tok_units.append((i, d, " ".join(toks)))
    return text_units, tok_units, np.array(n_records)


def encode(text_units, tok_units, n):
    """Sample × term matrices for the three encodings (+ term names)."""
    tv = CountVectorizer(ngram_range=(1, 2), stop_words=STOP, min_df=2, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    kv = CountVectorizer(min_df=2, lowercase=False, token_pattern=r"\S+")
    mats = []
    for units, vec in ((text_units, tv), (tok_units, kv)):
        U = vec.fit_transform([u[2] for u in units]).tocsr()
        # collapse units to (sample, day)
        keys = pd.Series([(u[0], u[1]) for u in units]).astype(str)
        codes, uniq = pd.factorize(keys)
        day_sample = pd.Series([u[0] for u in units]).groupby(codes).first().values
        to_day = csr_matrix((np.ones(len(units)), (codes, np.arange(len(units)))), shape=(len(uniq), len(units)))
        D = (to_day @ U).tocsr()
        to_sample = csr_matrix((np.ones(len(uniq)), (day_sample, np.arange(len(uniq)))), shape=(n, len(uniq)))
        raw = (to_sample @ D).tocsr()
        Db = D.copy(); Db.data[:] = 1
        distinct = (to_sample @ Db).tocsr()
        mats.append((raw, distinct, vec.get_feature_names_out()))
    raw = hstack([mats[0][0], mats[1][0]]).tocsr().astype(float)
    distinct = hstack([mats[0][1], mats[1][1]]).tocsr().astype(float)
    binary = distinct.copy(); binary.data[:] = 1
    names = np.r_[mats[0][2], mats[1][2]]
    lr, ld = raw.copy(), distinct.copy()
    lr.data, ld.data = np.log1p(lr.data), np.log1p(ld.data)
    return {"binary": binary, "distinct": ld, "raw": lr}, names


def oof_predictions(X, n_records, y, groups, seed, min_df=0.01):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    p, p_cnt = np.zeros(len(y)), np.zeros(len(y))
    cov = np.log1p(n_records).reshape(-1, 1)
    for tr, te in cv.split(X, y, groups):
        keep = np.asarray((X[tr] > 0).mean(axis=0)).ravel() >= min_df  # vocabulary from training fold only
        Xs = X[:, keep]
        sc, sc2 = MaxAbsScaler().fit(Xs[tr]), MaxAbsScaler().fit(cov[tr])
        Z = hstack([sc.transform(Xs), csr_matrix(sc2.transform(cov))]).tocsr()
        m = LogisticRegressionCV(Cs=8, cv=4, penalty="l1", solver="liblinear", scoring="roc_auc",
                                 max_iter=3000).fit(Z[tr], y[tr])
        p[te] = m.predict_proba(Z[te])[:, 1]
        p_cnt[te] = LogisticRegression().fit(cov[tr], y[tr]).predict_proba(cov[te])[:, 1]
    return p, p_cnt


def cluster_boot(y, groups, preds, n=1000, seed=0):
    """Patient-level bootstrap: AUC CI for each prediction vector, and paired differences vs the first."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in ug}
    out = np.full((n, len(preds)), np.nan)
    for b in range(n):
        idx = np.concatenate([idx_by[g] for g in rng.choice(ug, len(ug))])
        if y[idx].min() == y[idx].max():
            continue
        out[b] = [roc_auc_score(y[idx], p[idx]) for p in preds]
    return out


def mention_days(samples, ev, end, pattern):
    """Distinct days with an affirmed (simple negation filter) sentence/token match, per sample."""
    rx = re.compile(pattern, re.I)
    by = dict(tuple(ev.groupby("hadm_id")))
    counts = []
    for s in samples.itertuples():
        e = by.get(s.hadm_id)
        days = set()
        if e is not None:
            w = e[(e.t >= s.admittime - 24 * H) & (e.t < s.t0 + end * H)]
            for r in w.itertuples():
                if r.kind == "text":
                    for sent in re.split(r"(?<=[.;:])\s+|\n+", r.payload):
                        m = rx.search(sent)
                        if m and not NEG.search(sent[:m.start()]):
                            days.add(r.t.normalize())
                            break
                elif rx.search(r.payload):
                    days.add(r.t.normalize())
        counts.append(len(days))
    return np.array(counts)


def dose_table(counts, y):
    rows = []
    for lab, mask in [("0", counts == 0), ("1", counts == 1), ("2", counts == 2), ("3+", counts >= 3)]:
        k, n = int(y[mask].sum()), int(mask.sum())
        if n == 0:
            rows.append({"days": lab, "n": 0}); continue
        p = k / n; z = 1.96
        c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        rows.append({"days": lab, "n": n, "transferred": k, "share": round(p, 3), "ci": [round(c - h, 3), round(c + h, 3)]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261011, 1, 2])
    parser.add_argument("--min-hours", type=float, default=48)
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/transfer_recurrence")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    samples, _ = build_samples(3, args.seeds[0])
    samples = samples[samples.hours >= args.min_hours].reset_index(drop=True)
    y, groups = samples.label.values, samples.subject_id.values
    ev = events(set(samples.hadm_id))
    print("samples", len(samples), "cases", y.sum(), flush=True)

    rows, keep_for_dose = [], {}
    for fs in ("facts", "facts+actions"):
        for wname, end in WINDOWS:
            tu, ku, nrec = window_units(samples, ev, end, fs == "facts")
            X, names = encode(tu, ku, len(samples))
            if fs == "facts" and wname == "cum_lag0":
                keep_for_dose = {"X": X["distinct"], "names": names, "nrec": nrec}
            preds = {}
            for enc in ENCODINGS:
                ps = [oof_predictions(X[enc], nrec, y, groups, s) for s in args.seeds]
                preds[enc] = np.mean([p for p, _ in ps], axis=0)
                preds["count_only"] = ps[0][1]
                per_seed = [roc_auc_score(y, p) for p, _ in ps]
                rows.append({"features": fs, "window": wname, "encoding": enc,
                             "auc_mean": round(float(np.mean(per_seed)), 3),
                             "auc_seed_range": [round(min(per_seed), 3), round(max(per_seed), 3)]})
                print(rows[-1], flush=True)
            boot = cluster_boot(y, groups, [preds[e] for e in ENCODINGS] + [preds["count_only"]])
            ok = ~np.isnan(boot).any(axis=1)
            for j, enc in enumerate(ENCODINGS):
                r = next(r for r in rows if (r["features"], r["window"], r["encoding"]) == (fs, wname, enc))
                r["auc_ci95"] = np.percentile(boot[ok, j], [2.5, 97.5]).round(3).tolist()
                if j:
                    d = boot[ok, j] - boot[ok, 0]
                    r["diff_vs_binary"] = round(float(np.mean(d)), 3)
                    r["diff_ci95"] = np.percentile(d, [2.5, 97.5]).round(3).tolist()
            rows.append({"features": fs, "window": wname, "encoding": "count_only",
                         "auc_mean": round(roc_auc_score(y, preds["count_only"]), 3),
                         "auc_ci95": np.percentile(boot[ok, 3], [2.5, 97.5]).round(3).tolist()})
            print(rows[-4:], flush=True)
    (args.out / "encoding_auc.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))

    # permutation check: shuffled labels should give ~0.5
    rng = np.random.default_rng(0)
    perm = [roc_auc_score(yp, oof_predictions(keep_for_dose["X"], keep_for_dose["nrec"], yp, groups, args.seeds[0])[0])
            for yp in (rng.permutation(y) for _ in range(args.permutations))]
    print("permutation AUCs", np.round(perm, 3), flush=True)

    # dose-response, pre-specified problem terms
    dose = {"prespecified": {}, "split_sample": {}, "permutation_auc": np.round(perm, 3).tolist()}
    for wname, end in [("cum_lag0", 0), ("cum_lag48", -48)]:
        for name, pat in PRESPECIFIED.items():
            dose["prespecified"].setdefault(name, {})[wname] = dose_table(mention_days(samples, ev, end, pat), y)

    # dose-response, data-chosen terms: choose on half A (stability over bootstraps), show on half B
    X, names = keep_for_dose["X"], keep_for_dose["names"]
    subj = np.unique(groups); rng.shuffle(subj)
    A = np.isin(groups, subj[: len(subj) // 2]); B = ~A
    picks = np.zeros(X.shape[1])
    for b in range(30):
        idx = rng.choice(np.flatnonzero(A), A.sum())
        Xa = X[idx]; keep = np.asarray((Xa > 0).mean(axis=0)).ravel() >= 0.02
        m = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=3000).fit(
            MaxAbsScaler().fit_transform(Xa[:, keep]), y[idx])
        picks[np.flatnonzero(keep)[m.coef_[0] > 0]] += 1
    top = np.argsort(-picks)[:8]
    Xd = X.copy(); Xd.data = np.expm1(Xd.data)  # back to day counts
    for j in top:
        c = np.asarray(Xd[:, j].todense()).ravel().round().astype(int)
        dose["split_sample"][str(names[j])] = {"selected_in_A": f"{int(picks[j])}/30", "half_B": dose_table(c[B], y[B])}
    (args.out / "dose_response.json").write_text(json.dumps(dose, indent=1, ensure_ascii=False))
    print("done", flush=True)


if __name__ == "__main__":
    main()
