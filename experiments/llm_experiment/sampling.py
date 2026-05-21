"""Stratified 30-patient sampling: disease x CertaintyScore tercile.

Reads:
  - results/gap_analysis/part3_certainty_score.csv  (CS per patient)
  - results/test_seq_comparison/test_sequence_comparison.csv  (sequences)

Returns rows of (patient_id, disease, certainty_score_oracle, cs_tercile,
actual_sequence, simulated_sequence). CS terciles are disease-specific
(matches handoff_summary.md convention).
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


_ROOT = Path(__file__).resolve().parents[2]
_CS_CSV = _ROOT / "results" / "gap_analysis" / "part3_certainty_score.csv"
_TSC_CSV = _ROOT / "results" / "test_seq_comparison" / "test_sequence_comparison.csv"


# (disease, low_count, mid_count, high_count) — sums to 30
SAMPLE_PLAN: list[tuple[str, int, int, int]] = [
    ("appendicitis",   3, 3, 2),
    ("cholecystitis",  3, 3, 2),
    ("diverticulitis", 2, 2, 2),
    ("pancreatitis",   3, 3, 2),
]


def _join_cs_tsc() -> pd.DataFrame:
    cs = pd.read_csv(_CS_CSV)
    tsc = pd.read_csv(_TSC_CSV)
    df = cs.merge(
        tsc[["patient_id", "disease", "simulated_sequence", "actual_sequence"]],
        on=["patient_id", "disease"],
        how="inner",
    )
    df["cs_tercile"] = df.groupby("disease", observed=True)[
        "certainty_score_oracle"
    ].transform(lambda s: pd.qcut(s, q=3, labels=["low", "mid", "high"]))
    return df


def sample_30(seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    df = _join_cs_tsc()
    picks: list[pd.Series] = []
    for disease, n_lo, n_mid, n_hi in SAMPLE_PLAN:
        for tercile, n in [("low", n_lo), ("mid", n_mid), ("high", n_hi)]:
            pool = df[(df["disease"] == disease) & (df["cs_tercile"] == tercile)]
            ids = pool["patient_id"].tolist()
            rng.shuffle(ids)
            chosen = ids[:n]
            picks.extend(pool[pool["patient_id"].isin(chosen)].to_dict("records"))
    out = pd.DataFrame(picks)
    keep = [
        "patient_id", "disease", "certainty_score_oracle", "cs_tercile",
        "perceived_top_diagnosis", "actual_sequence", "simulated_sequence",
    ]
    return out[keep].reset_index(drop=True)


def sample_order_sensitivity(seed: int = 42) -> pd.DataFrame:
    """Pick 3 patients, each from a different disease + different CS tercile."""
    rng = random.Random(seed)
    df = _join_cs_tsc()
    picks: list[pd.Series] = []
    # Mix it up: pancreatitis-high, cholecystitis-mid, appendicitis-low
    spec = [
        ("pancreatitis",  "high"),
        ("cholecystitis", "mid"),
        ("appendicitis",  "low"),
    ]
    for disease, tercile in spec:
        pool = df[(df["disease"] == disease) & (df["cs_tercile"] == tercile)]
        ids = pool["patient_id"].tolist()
        rng.shuffle(ids)
        chosen_id = ids[0]
        picks.extend(pool[pool["patient_id"] == chosen_id].to_dict("records"))
    out = pd.DataFrame(picks)
    keep = [
        "patient_id", "disease", "certainty_score_oracle", "cs_tercile",
        "perceived_top_diagnosis", "actual_sequence", "simulated_sequence",
    ]
    return out[keep].reset_index(drop=True)


if __name__ == "__main__":
    main = sample_30()
    print("== MAIN SAMPLE (30) ==")
    print(main.to_string())
    print("\n== ORDER-SENSITIVITY SAMPLE (3) ==")
    print(sample_order_sensitivity().to_string())
