"""Gap analysis: which step-0 patient features predict simulator-vs-actual gaps.

Convention (doctor-side):
  commission = (actual_tests - sim_tests) - {Radiograph_Chest}   # doctor added extras
  omission   = sim_tests - actual_tests                          # doctor skipped rubric-recommended tests
  order_swap = relative order of shared tests differs

For each disease separately, build:
  - Univariate Fisher exact test of every (feature, gap) pair
  - L1-regularized logistic regression (CV) coefficients
  - Random forest feature importances
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")

ROOT     = Path("/Users/duruoli/A/A李杜若/1-科研/PhD/0/1-code/congraph")
RESULTS  = ROOT / "results"
OUT_DIR  = RESULTS / "gap_analysis"
OUT_DIR.mkdir(exist_ok=True, parents=True)
(OUT_DIR / "univariate").mkdir(exist_ok=True)
(OUT_DIR / "models").mkdir(exist_ok=True)

DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]
GAP_TYPES = ["commission", "omission", "order_swap"]
DROP_FEATS = {"CTSI_score", "tests_done", "lab_itemids"}
EXCLUDE_FROM_COMMISSION = {"Radiograph_Chest"}


# ── 1. Load comparison + features ───────────────────────────────────────────

comp = pd.read_csv(RESULTS / "test_seq_comparison" / "test_sequence_comparison.csv")
comp["patient_id"] = comp["patient_id"].astype(str)
comp["sim_list"]    = comp["simulated_sequence"].fillna("").apply(
    lambda s: [t.strip() for t in s.split(",") if t.strip()])
comp["actual_list"] = comp["actual_sequence"].fillna("").apply(
    lambda s: [t.strip() for t in s.split(",") if t.strip()])


feat_rows: list[dict] = []
for disease in DISEASES:
    data = json.load(open(RESULTS / f"{disease}_features.json"))["results"]
    for pid, steps in data.items():
        if not steps:
            continue
        f = steps[0].get("features", {})
        row = {"patient_id": str(pid), "disease": disease}
        row.update({k: v for k, v in f.items() if k not in DROP_FEATS})
        feat_rows.append(row)

feat_df = pd.DataFrame(feat_rows)
print(f"Loaded features for {len(feat_df)} patient-rows across {feat_df['disease'].nunique()} diseases")


# ── 2. Join and compute gap labels ──────────────────────────────────────────

df = comp.merge(feat_df, on=["patient_id", "disease"], how="left", validate="one_to_one")
missing = df[df.iloc[:, -5:].isna().all(axis=1)]
if len(missing):
    raise RuntimeError(f"{len(missing)} patients failed feature join")
print(f"After join: {len(df)} rows")


def compute_gaps(sim: list[str], actual: list[str]) -> dict:
    sim_set, act_set = set(sim), set(actual)
    commission = (act_set - sim_set) - EXCLUDE_FROM_COMMISSION
    omission   = sim_set - act_set
    shared     = sim_set & act_set
    sim_shared    = [t for t in sim    if t in shared]
    actual_shared = [t for t in actual if t in shared]
    order_swap = sim_shared != actual_shared
    return {
        "commission_tests": sorted(commission),
        "omission_tests":   sorted(omission),
        "has_commission":   int(len(commission) > 0),
        "has_omission":     int(len(omission) > 0),
        "has_order_swap":   int(order_swap),
        "n_commission":     len(commission),
        "n_omission":       len(omission),
    }


gap_records = df.apply(lambda r: compute_gaps(r["sim_list"], r["actual_list"]), axis=1)
gap_df = pd.DataFrame(list(gap_records))
df = pd.concat([df, gap_df], axis=1)


# Per-test commission/omission flags
all_commission = sorted({t for ts in df["commission_tests"] for t in ts})
all_omission   = sorted({t for ts in df["omission_tests"]   for t in ts})
for t in all_commission:
    df[f"commission__{t}"] = df["commission_tests"].apply(lambda xs: int(t in xs))
for t in all_omission:
    df[f"omission__{t}"] = df["omission_tests"].apply(lambda xs: int(t in xs))


# ── 3. Build feature matrix ─────────────────────────────────────────────────

raw_feat_cols = [c for c in feat_df.columns if c not in {"patient_id", "disease"}]

def build_X(sub: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    Xs = []
    names = []
    for c in raw_feat_cols:
        col = sub[c]
        if c == "pain_location":
            dummies = pd.get_dummies(col, prefix="pain_location", dummy_na=False).astype(int)
            Xs.append(dummies)
            names.extend(dummies.columns.tolist())
        else:
            # binary or numeric — coerce to numeric
            v = col.map(lambda x: 1 if x is True else (0 if x is False else x))
            v = pd.to_numeric(v, errors="coerce").fillna(0).astype(int)
            Xs.append(v.rename(c))
            names.append(c)
    X = pd.concat(Xs, axis=1)
    # Drop zero-variance columns
    nonconst = X.columns[X.nunique() > 1].tolist()
    return X[nonconst], nonconst


# ── 4. Save the joined dataset ──────────────────────────────────────────────

flat_cols = (
    ["patient_id", "disease", "simulated_sequence", "actual_sequence",
     "simulated_length", "actual_length"]
    + raw_feat_cols
    + ["has_commission", "has_omission", "has_order_swap",
       "n_commission", "n_omission"]
    + [c for c in df.columns if c.startswith("commission__") or c.startswith("omission__")]
    + ["commission_tests", "omission_tests"]
)
df_out = df[flat_cols].copy()
df_out["commission_tests"] = df_out["commission_tests"].apply(lambda xs: ", ".join(xs))
df_out["omission_tests"]   = df_out["omission_tests"].apply(lambda xs: ", ".join(xs))
df_out.to_csv(OUT_DIR / "joined_data.csv", index=False)
print(f"  wrote joined_data.csv ({len(df_out)} rows, {df_out.shape[1]} cols)")


# Per-disease gap-rate summary
summary_rows = []
for disease in DISEASES:
    sub = df[df["disease"] == disease]
    n = len(sub)
    row = {"disease": disease, "n": n}
    for g in GAP_TYPES:
        col = f"has_{g}"
        row[f"{g}_rate"] = sub[col].mean()
        row[f"{g}_n"]    = int(sub[col].sum())
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "gap_rates_per_disease.csv", index=False)
print(f"  wrote gap_rates_per_disease.csv")
print(summary_df.to_string(index=False))


# Per-test gap rates per disease
per_test_rows = []
for disease in DISEASES:
    sub = df[df["disease"] == disease]
    n = len(sub)
    for t in all_commission:
        per_test_rows.append({"disease": disease, "gap_type": "commission",
                              "test": t, "n_with_gap": int(sub[f"commission__{t}"].sum()),
                              "rate": sub[f"commission__{t}"].mean(), "n_total": n})
    for t in all_omission:
        per_test_rows.append({"disease": disease, "gap_type": "omission",
                              "test": t, "n_with_gap": int(sub[f"omission__{t}"].sum()),
                              "rate": sub[f"omission__{t}"].mean(), "n_total": n})
pd.DataFrame(per_test_rows).to_csv(OUT_DIR / "per_test_gap_rates.csv", index=False)
print(f"  wrote per_test_gap_rates.csv")


# ── 5. Per-disease modeling ─────────────────────────────────────────────────

metrics_all: list[dict] = []

for disease in DISEASES:
    sub = df[df["disease"] == disease].reset_index(drop=True)
    X, feat_names = build_X(sub)
    n = len(sub)

    for gap in GAP_TYPES:
        y = sub[f"has_{gap}"].values.astype(int)
        n_pos = int(y.sum()); n_neg = n - n_pos
        base = n_pos / n if n else 0.0

        m = {"disease": disease, "gap": gap, "n": n,
             "n_pos": n_pos, "n_neg": n_neg, "base_rate": base}

        # Skip degenerate outcomes
        if n_pos < 3 or n_neg < 3:
            m["status"] = "skipped_degenerate"
            metrics_all.append(m)
            continue

        # Univariate Fisher exact for each (binary-coded) feature
        uni_rows = []
        for c in feat_names:
            x = X[c].values
            # 2×2 contingency
            a = int(((x == 1) & (y == 1)).sum())
            b = int(((x == 1) & (y == 0)).sum())
            cc = int(((x == 0) & (y == 1)).sum())
            d = int(((x == 0) & (y == 0)).sum())
            # Skip uninformative
            if (a + b == 0) or (cc + d == 0):
                continue
            try:
                or_, p = fisher_exact([[a, b], [cc, d]], alternative="two-sided")
            except Exception:
                or_, p = float("nan"), float("nan")
            # Haldane-Anscombe corrected log OR for ranking when cell zero
            a_c, b_c, c_c, d_c = a + 0.5, b + 0.5, cc + 0.5, d + 0.5
            log_or_smoothed = math.log((a_c * d_c) / (b_c * c_c))
            uni_rows.append({
                "feature": c,
                "rate_when_gap": a / (a + cc) if (a + cc) else float("nan"),
                "rate_when_no_gap": b / (b + d) if (b + d) else float("nan"),
                "n_feat_pos": a + b,
                "odds_ratio": or_,
                "log_odds_ratio_smoothed": log_or_smoothed,
                "p_value": p,
            })
        uni_df = pd.DataFrame(uni_rows).sort_values("p_value")
        # Benjamini-Hochberg q-value
        if len(uni_df):
            p = uni_df["p_value"].values
            m_eff = len(p)
            order = np.argsort(p)
            ranked = p[order]
            q = ranked * m_eff / (np.arange(m_eff) + 1)
            q = np.minimum.accumulate(q[::-1])[::-1]
            qvals = np.empty(m_eff); qvals[order] = q
            uni_df["bh_q"] = np.clip(qvals, 0, 1)
        uni_df.to_csv(OUT_DIR / "univariate" / f"{disease}_{gap}.csv", index=False)

        # L1 logistic regression with CV
        n_splits = min(5, n_pos, n_neg)
        if n_splits >= 3:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            try:
                lr = LogisticRegressionCV(
                    Cs=10, cv=cv, penalty="l1", solver="liblinear",
                    scoring="roc_auc", max_iter=2000, random_state=42,
                )
                lr.fit(X.values, y)
                y_prob = cross_val_predict(
                    LogisticRegressionCV(
                        Cs=10, cv=cv, penalty="l1", solver="liblinear",
                        scoring="roc_auc", max_iter=2000, random_state=42,
                    ),
                    X.values, y, cv=cv, method="predict_proba",
                )[:, 1]
                cv_auc = roc_auc_score(y, y_prob)
                coefs = lr.coef_.ravel()
                coef_df = pd.DataFrame({
                    "feature": feat_names,
                    "coef": coefs,
                    "abs_coef": np.abs(coefs),
                }).sort_values("abs_coef", ascending=False)
                coef_df = coef_df[coef_df["abs_coef"] > 1e-8]
                coef_df.to_csv(
                    OUT_DIR / "models" / f"{disease}_{gap}_l1_coef.csv", index=False)
                m["l1_cv_auc"]      = float(cv_auc)
                m["l1_best_C"]      = float(lr.C_[0])
                m["l1_n_nonzero"]   = int((np.abs(coefs) > 1e-8).sum())
            except Exception as e:
                m["l1_error"] = str(e)
        else:
            m["l1_status"] = f"skipped_few_folds (n_pos={n_pos}, n_neg={n_neg})"

        # Random forest
        try:
            rf = RandomForestClassifier(
                n_estimators=500, max_depth=None, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            rf.fit(X.values, y)
            if n_splits >= 3:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                y_prob = cross_val_predict(
                    RandomForestClassifier(
                        n_estimators=500, max_depth=None, min_samples_leaf=2,
                        class_weight="balanced", random_state=42, n_jobs=-1,
                    ),
                    X.values, y, cv=cv, method="predict_proba",
                )[:, 1]
                rf_auc = roc_auc_score(y, y_prob)
                m["rf_cv_auc"] = float(rf_auc)
            imp_df = pd.DataFrame({
                "feature": feat_names,
                "importance": rf.feature_importances_,
            }).sort_values("importance", ascending=False)
            imp_df.to_csv(
                OUT_DIR / "models" / f"{disease}_{gap}_rf_importance.csv", index=False)
        except Exception as e:
            m["rf_error"] = str(e)

        metrics_all.append(m)

metrics_df = pd.DataFrame(metrics_all)
metrics_df.to_csv(OUT_DIR / "model_metrics.csv", index=False)
print()
print(metrics_df.to_string(index=False))


# ── 6. Markdown summary ─────────────────────────────────────────────────────

def _md_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    def cell(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else (f"{v:.3f}" if abs(v) < 1000 else f"{v:.2e}")
        return "" if v is None else str(v)
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows   = ["| " + " | ".join(cell(v) for v in row) + " |"
              for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


lines: list[str] = []
lines.append("# Gap analysis — per disease\n")
lines.append("**Convention** (doctor-side):\n")
lines.append("- `commission` = `actual \\ sim` − `{Radiograph_Chest}` "
             "(doctor ordered extra tests beyond rubric)\n")
lines.append("- `omission` = `sim \\ actual` (doctor skipped rubric-recommended tests)\n")
lines.append("- `order_swap` = relative order of shared tests differs\n\n")
lines.append("## Gap rates\n\n")
lines.append(summary_df.pipe(_md_table))
lines.append("\n\n## Model AUCs\n\n")
m_show = metrics_df.copy()
for c in ["base_rate", "l1_cv_auc", "rf_cv_auc"]:
    if c in m_show.columns:
        m_show[c] = m_show[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
lines.append(m_show[[c for c in [
    "disease", "gap", "n", "n_pos", "base_rate",
    "l1_cv_auc", "l1_n_nonzero", "rf_cv_auc"] if c in m_show.columns]].pipe(_md_table))
lines.append("\n\n## Top features (BH q ≤ 0.20) by Fisher exact, per (disease, gap)\n")
for (disease, gap), grp in [((d, g), None) for d in DISEASES for g in GAP_TYPES]:
    fp = OUT_DIR / "univariate" / f"{disease}_{gap}.csv"
    if not fp.exists():
        continue
    u = pd.read_csv(fp)
    sig = u[u["bh_q"] <= 0.20].head(10)
    if not len(sig):
        continue
    lines.append(f"\n### {disease} — {gap}\n")
    lines.append(sig[["feature", "rate_when_gap", "rate_when_no_gap",
                      "odds_ratio", "p_value", "bh_q"]].pipe(_md_table))
    lines.append("\n")
(OUT_DIR / "summary.md").write_text("\n".join(lines))
print(f"\n  wrote summary.md")
print(f"\nAll outputs in: {OUT_DIR}")
