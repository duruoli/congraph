"""Part 3 — Condition log-likelihood weights and step-0 CertaintyScore.

Pipeline:
  1. Enumerate clinically observable "condition features" appearing in any
     edge condition of the rubric graphs (rubric_graph.py).  Categorical
     `pain_location` is expanded to dummies for {RLQ, RUQ, LLQ, Epigastric}.
  2. Reconstruct the same train/test split as run_rubric_simulator_oracle.py
     (n_test=300, seed=42, min_tests=2).  Training cohort = 1044 patients.
  3. For each (condition feature c, disease d), estimate the smoothed
     log-likelihood ratio
            w(c,d) = log[ (P(c=1|d)+0.5)/(P(c=1|¬d)+0.5) ]
     using training step-0 features and disease labels.  Reliability flag
     unreliable=True when the positive feature count < 5.
  4. For each of the 300 test patients, compute
            CertaintyScore = Σ_{c: c=1 at step 0}  w(c, d_oracle)
     using the patient's true (oracle) disease label.  A secondary score
     using d=argmax-prevalence-class (top-disease proxy) is also recorded.
  5. Merge CertaintyScore into joined_data, refit L1 logistic regressions
     for commission/omission/order_swap and report AUC change.

Outputs
-------
  results/gap_analysis/part3_condition_weights.csv
  results/gap_analysis/part3_certainty_score.csv
  results/gap_analysis/part3_regression_comparison.txt
"""

from __future__ import annotations

import json
import math
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.knn_feature_eval import DISEASE_FILES, load_all  # noqa: E402

RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "gap_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]
GAP_TYPES = ["commission", "omission", "order_swap"]
SEED = 42
N_TEST = 300
MIN_TESTS = 2


# ---------------------------------------------------------------------------
# Condition feature list
# ---------------------------------------------------------------------------
# Binary clinical-evidence features that appear (directly or via helpers
# alvarado_score / bisap_score / _revised_atlanta_criteria_count / _tg18_*)
# in any edge condition of TRIAGE_GRAPH or the four disease sub-graphs.
# Post-test imaging flags are included for completeness even though they
# cannot trigger at step 0 (they remain False until the test is done).

CONDITION_FEATURES_BINARY: list[str] = [
    # Pain characteristics
    "pain_migration_to_RLQ",
    "epigastric_radiating_to_back",
    "bowel_habit_change",
    "symptom_duration_over_72h",
    # HPI symptom flags
    "anorexia",
    "nausea_vomiting",
    "prior_diverticular_disease",
    "fever_reported_in_hpi",
    # Physical examination
    "murphys_sign",
    "RUQ_tenderness",
    "RLQ_tenderness",
    "rebound_tenderness",
    "RUQ_mass",
    "fever_temp_ge_37_3",
    "fever_temp_ge_38",
    "peritoneal_signs",
    "impaired_mental_status",
    # Labs
    "WBC_gt_10k",
    "WBC_gt_18k",
    "left_shift",
    "CRP_elevated",
    "CRP_gt_200",
    "lipase_ge_3xULN",
    "BUN_gt_25",
    "bilirubin_elevated",
    "LFTs_elevated",
    "SIRS_criteria_ge_2",
    # Demographics
    "age_gt_60",
    # Imaging (post-test; will always be False at step 0 but listed for
    # completeness so the w-table is comprehensive)
    "US_appendix_inflamed",
    "US_gallstones",
    "US_GB_wall_thickening",
    "US_pericholecystic_fluid",
    "US_sonographic_murphys",
    "US_perforation_abscess",
    "CT_appendicitis_positive",
    "CT_perforation_abscess",
    "CT_cholecystitis_positive",
    "CT_GB_severe_findings",
    "CT_diverticulitis_confirmed",
    "CT_diverticulitis_complicated",
    "CT_phlegmon",
    "CT_abscess_lt_3cm",
    "CT_abscess_ge_3cm",
    "CT_purulent_peritonitis",
    "CT_fecal_peritonitis",
    "CT_pancreatitis_positive",
    "cholecystitis_additional_imaging_positive",
    "pleural_effusion_on_imaging",
    # Organ dysfunction
    "has_organ_dysfunction",
    "organ_failure_transient",
    "organ_failure_persistent",
    "local_complications_pancreatitis",
]

# pain_location categories explicitly compared in edge conditions
PAIN_LOCATION_CATEGORIES = ["RLQ", "RUQ", "LLQ", "Epigastric"]


def expand_features(f: dict) -> dict[str, int]:
    """Encode a feature dict into the {feature -> 0/1} matrix row."""
    row: dict[str, int] = {}
    for c in CONDITION_FEATURES_BINARY:
        v = f.get(c, False)
        row[c] = int(bool(v))
    pl = f.get("pain_location", "Other")
    for cat in PAIN_LOCATION_CATEGORIES:
        row[f"pain_location_{cat}"] = int(pl == cat)
    return row


ALL_CONDITION_FEATURES: list[str] = (
    CONDITION_FEATURES_BINARY
    + [f"pain_location_{c}" for c in PAIN_LOCATION_CATEGORIES]
)


# ---------------------------------------------------------------------------
# Step 1. Reproduce train/test split exactly as run_rubric_simulator_oracle.py
# ---------------------------------------------------------------------------

def build_split() -> tuple[list[tuple[str, str, list[dict]]], list[tuple[str, str, list[dict]]]]:
    all_patients = load_all()
    flat = [(d, pid, steps) for d, ps in all_patients.items() for pid, steps in ps.items()]
    min_steps = MIN_TESTS + 1
    flat = [(d, pid, s) for d, pid, s in flat if len(s) >= min_steps]
    random.seed(SEED)
    random.shuffle(flat)
    test_rows = flat[:N_TEST]
    train_rows = flat[N_TEST:]
    return train_rows, test_rows


def step0_features(steps: list[dict]) -> dict:
    return steps[0].get("features", {}) if steps else {}


# ---------------------------------------------------------------------------
# Step 2. w(c,d) on training cohort
# ---------------------------------------------------------------------------

SMOOTH = 0.5
MIN_POS_FOR_RELIABLE = 5


def compute_weights(train_rows: list[tuple[str, str, list[dict]]]) -> pd.DataFrame:
    rows = []
    for d, pid, steps in train_rows:
        feats = step0_features(steps)
        row = {"patient_id": pid, "disease": d}
        row.update(expand_features(feats))
        rows.append(row)
    df = pd.DataFrame(rows)

    out_rows = []
    for c in ALL_CONDITION_FEATURES:
        x = df[c].values.astype(int)
        n_pos_total = int(x.sum())
        for d in DISEASES:
            mask_d = (df["disease"] == d).values
            n_d = int(mask_d.sum())
            n_nd = len(df) - n_d
            n_c_d = int(((x == 1) & mask_d).sum())
            n_c_nd = n_pos_total - n_c_d
            p1 = (n_c_d + SMOOTH) / (n_d + 2 * SMOOTH)
            p0 = (n_c_nd + SMOOTH) / (n_nd + 2 * SMOOTH)
            w = math.log(p1 / p0)
            out_rows.append({
                "feature": c,
                "disease": d,
                "n_d": n_d,
                "n_nd": n_nd,
                "n_c_d": n_c_d,
                "n_c_nd": n_c_nd,
                "p_c_given_d": p1,
                "p_c_given_not_d": p0,
                "w": w,
                "unreliable": n_pos_total < MIN_POS_FOR_RELIABLE,
            })
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Step 3. CertaintyScore on test cohort
# ---------------------------------------------------------------------------

def compute_certainty(
    test_rows: list[tuple[str, str, list[dict]]],
    w_df: pd.DataFrame,
) -> pd.DataFrame:
    # Pivot for fast lookup: feature × disease → w
    w_pivot = w_df.pivot(index="feature", columns="disease", values="w")

    rows = []
    for d, pid, steps in test_rows:
        feats = step0_features(steps)
        enc = expand_features(feats)
        triggered = [c for c, v in enc.items() if v == 1]
        # primary: oracle disease
        score_oracle = float(sum(w_pivot.loc[c, d] for c in triggered if c in w_pivot.index))
        # secondary: score using the disease that *maximises* the sum of
        # triggered weights → "perceived top diagnosis" from step-0 evidence
        per_disease = {
            dd: float(sum(w_pivot.loc[c, dd] for c in triggered if c in w_pivot.index))
            for dd in DISEASES
        }
        top_dx = max(per_disease, key=per_disease.get) if per_disease else d
        score_top = per_disease.get(top_dx, 0.0)

        rows.append({
            "patient_id": pid,
            "disease": d,
            "n_triggered": len(triggered),
            "certainty_score_oracle": score_oracle,
            "perceived_top_diagnosis": top_dx,
            "certainty_score_top": score_top,
            "oracle_eq_top": int(d == top_dx),
            **{f"score_{dd}": per_disease.get(dd, 0.0) for dd in DISEASES},
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4. Regression comparison
# ---------------------------------------------------------------------------

DROP_FEATS = {"CTSI_score", "tests_done", "lab_itemids"}


def build_X(sub: pd.DataFrame, raw_feat_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Replicate gap_analysis.build_X with the same dummy / binary encoding."""
    Xs = []
    names = []
    for c in raw_feat_cols:
        col = sub[c]
        if c == "pain_location":
            dummies = pd.get_dummies(col, prefix="pain_location", dummy_na=False).astype(int)
            Xs.append(dummies)
            names.extend(dummies.columns.tolist())
        else:
            v = col.map(lambda x: 1 if x is True else (0 if x is False else x))
            v = pd.to_numeric(v, errors="coerce").fillna(0).astype(int)
            Xs.append(v.rename(c))
            names.append(c)
    X = pd.concat(Xs, axis=1)
    nonconst = X.columns[X.nunique() > 1].tolist()
    return X[nonconst], nonconst


def fit_l1_auc(X: np.ndarray, y: np.ndarray, n_splits: int) -> tuple[float, float, np.ndarray]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    lr = LogisticRegressionCV(
        Cs=10, cv=cv, penalty="l1", solver="liblinear",
        scoring="roc_auc", max_iter=2000, random_state=SEED,
    )
    lr.fit(X, y)
    y_prob = cross_val_predict(
        LogisticRegressionCV(
            Cs=10, cv=cv, penalty="l1", solver="liblinear",
            scoring="roc_auc", max_iter=2000, random_state=SEED,
        ),
        X, y, cv=cv, method="predict_proba",
    )[:, 1]
    auc = roc_auc_score(y, y_prob)
    ll = -log_loss(y, np.clip(y_prob, 1e-6, 1 - 1e-6), labels=[0, 1])
    return float(auc), float(ll), y_prob


def regression_comparison(certainty_df: pd.DataFrame) -> str:
    joined = pd.read_csv(OUT_DIR / "joined_data.csv")
    joined["patient_id"] = joined["patient_id"].astype(str)
    certainty_df["patient_id"] = certainty_df["patient_id"].astype(str)

    raw_feat_cols = [
        c for c in joined.columns
        if c in {"pain_location"} or c in expand_features({}).keys()
        # The expand_features call only matches binary names; we need to
        # explicitly include any feature columns from joined_data that
        # correspond to schema features.
    ]
    # Re-derive from the full feature_schema to match gap_analysis.py exactly:
    from pipeline.feature_schema import default_features
    schema_keys = set(default_features().keys())
    raw_feat_cols = [
        c for c in joined.columns
        if c in schema_keys and c not in DROP_FEATS
    ]

    df = joined.merge(
        certainty_df[["patient_id", "certainty_score_oracle", "certainty_score_top", "n_triggered"]],
        on="patient_id", how="left",
    )

    lines: list[str] = []
    lines.append("# Part 3 — CertaintyScore regression comparison")
    lines.append("")
    lines.append("Setup")
    lines.append("-----")
    lines.append(f"  Train cohort (for w(c,d)) : 1044 patients (split seed={SEED})")
    lines.append(f"  Test cohort               : 300 patients (gap labels)")
    lines.append("  Encoding                  : pain_location → dummies, others 0/1")
    lines.append("  Baseline X                : all schema features (mirrors gap_analysis.py)")
    lines.append("  +CertaintyScore X         : baseline + certainty_score_oracle")
    lines.append("")
    lines.append("Per-(disease, gap) AUC and log-likelihood under 5-fold (or fewer) CV")
    lines.append("L1 logistic regression with Cs=10 search, scoring=roc_auc.")
    lines.append("")

    rows = []
    for disease in DISEASES:
        sub = df[df["disease"] == disease].reset_index(drop=True)
        X_base, feat_names = build_X(sub, raw_feat_cols)
        for gap in GAP_TYPES:
            y = sub[f"has_{gap}"].values.astype(int)
            n_pos = int(y.sum()); n_neg = len(y) - n_pos
            if n_pos < 3 or n_neg < 3:
                rows.append({
                    "disease": disease, "gap": gap, "n": len(sub),
                    "n_pos": n_pos, "status": "degenerate",
                })
                continue
            n_splits = min(5, n_pos, n_neg)

            def _safe(X):
                try:
                    return fit_l1_auc(X, y, n_splits)
                except Exception:
                    return float("nan"), float("nan"), None

            auc_b, ll_b, _ = _safe(X_base.values)
            cs_oracle = sub["certainty_score_oracle"].fillna(0.0).values.reshape(-1, 1)
            X_plus = np.hstack([X_base.values, cs_oracle])
            auc_p, ll_p, _ = _safe(X_plus)
            auc_a, ll_a, _ = _safe(cs_oracle)
            cs_top = sub["certainty_score_top"].fillna(0.0).values.reshape(-1, 1)
            auc_at, ll_at, _ = _safe(cs_top)

            rows.append({
                "disease": disease, "gap": gap, "n": len(sub),
                "n_pos": n_pos,
                "auc_baseline": auc_b, "ll_baseline": ll_b,
                "auc_plus_cs": auc_p, "ll_plus_cs": ll_p,
                "delta_auc": auc_p - auc_b,
                "auc_cs_alone": auc_a, "ll_cs_alone": ll_a,
                "auc_cs_top_alone": auc_at,
            })

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(OUT_DIR / "part3_regression_comparison.csv", index=False)

    lines.append(cmp_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    lines.append("")
    lines.append("Interpretation guide")
    lines.append("--------------------")
    lines.append("  delta_auc > 0    : CertaintyScore adds incremental predictive power")
    lines.append("                     beyond the raw features (the certainty story holds).")
    lines.append("  auc_cs_alone     : Univariate predictive power of CertaintyScore.")
    lines.append("                     If close to auc_baseline, a single scalar captures")
    lines.append("                     most of what the many raw features encode.")
    lines.append("  auc_cs_top_alone : Same, but using the perceived-top-diagnosis score.")
    lines.append("                     Robustness check: if much weaker than oracle-score,")
    lines.append("                     it suggests the oracle label is doing heavy lifting.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building train/test split…")
    train_rows, test_rows = build_split()
    print(f"  train={len(train_rows)}  test={len(test_rows)}")

    print("Computing w(c,d) on training cohort…")
    w_df = compute_weights(train_rows)
    w_df.to_csv(OUT_DIR / "part3_condition_weights.csv", index=False)
    print(f"  wrote part3_condition_weights.csv  ({len(w_df)} rows, "
          f"{w_df['feature'].nunique()} features × {w_df['disease'].nunique()} diseases)")

    print("Computing CertaintyScore on test cohort…")
    c_df = compute_certainty(test_rows, w_df)
    c_df.to_csv(OUT_DIR / "part3_certainty_score.csv", index=False)
    print(f"  wrote part3_certainty_score.csv  ({len(c_df)} rows)")

    print("Refit gap regressions with CertaintyScore…")
    report = regression_comparison(c_df)
    (OUT_DIR / "part3_regression_comparison.txt").write_text(report + "\n")
    print("  wrote part3_regression_comparison.txt")


if __name__ == "__main__":
    main()
