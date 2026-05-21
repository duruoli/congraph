"""Compute Part-1-style descriptive gap stats for LLM recommendation sequences.

For each (condition x disease) cell:
  - exact_match_rate : LLM seq (as list) == actual seq (as list)
  - commission_rate  : >= 1 test in actual\sim − {Radiograph_Chest}
  - omission_rate    : >= 1 test in sim\actual
  - order_swap_rate  : among patients with non-empty intersection, relative order differs
  - mean_seq_length  : LLM seq length (with Lab_Panel prepended for parity)

Both `LLM seq` and `actual seq` have Lab_Panel prepended (same convention as
scripts/extract_test_sequences.py).

Also includes a `rubric_only` baseline from
results/test_seq_comparison/test_sequence_comparison.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EXCLUDE_FROM_COMMISSION = {"Radiograph_Chest"}

_ROOT = Path(__file__).resolve().parents[2]
_TSC = _ROOT / "results" / "test_seq_comparison" / "test_sequence_comparison.csv"


def _split(seq: str) -> list[str]:
    if isinstance(seq, str) and seq.strip():
        return [s.strip() for s in seq.split(",") if s.strip()]
    return []


def _prepend_lab(seq: list[str]) -> list[str]:
    if seq and seq[0] == "Lab_Panel":
        return seq
    return ["Lab_Panel"] + seq


def _gap_flags(sim: list[str], actual: list[str]) -> dict:
    sim_set = set(sim)
    act_set = set(actual)
    commission = (act_set - sim_set) - EXCLUDE_FROM_COMMISSION
    omission = sim_set - act_set
    shared = sim_set & act_set
    sim_shared = [t for t in sim if t in shared]
    act_shared = [t for t in actual if t in shared]
    order_swap = sim_shared != act_shared
    exact_match = sim == actual
    return {
        "exact_match": int(exact_match),
        "has_commission": int(len(commission) > 0),
        "has_omission": int(len(omission) > 0),
        "has_order_swap": int(order_swap),
        "n_commission": len(commission),
        "n_omission": len(omission),
    }


def _per_patient_gaps(llm_seq_df: pd.DataFrame, tsc: pd.DataFrame) -> pd.DataFrame:
    """Join LLM seqs with actual seqs and compute per-patient flags."""
    actual = tsc.set_index("patient_id")["actual_sequence"].to_dict()
    rubric_sim = tsc.set_index("patient_id")["simulated_sequence"].to_dict()
    out_rows = []
    for _, row in llm_seq_df.iterrows():
        pid = int(row["patient_id"])
        if pid not in actual:
            continue
        actual_seq = _split(actual[pid])  # already includes Lab_Panel
        llm_seq = _prepend_lab(_split(row["llm_sequence"]))
        flags = _gap_flags(llm_seq, actual_seq)
        out_rows.append({
            "patient_id": pid,
            "disease": row["disease"],
            "cs_tercile": row["cs_tercile"],
            "condition": row["condition"],
            "info_order": row["info_order"],
            "llm_sequence_with_lab": ", ".join(llm_seq),
            "actual_sequence": actual[pid],
            "rubric_sim_sequence": rubric_sim.get(pid, ""),
            "llm_length": len(llm_seq),
            "actual_length": len(actual_seq),
            **flags,
        })
    return pd.DataFrame(out_rows)


def _rubric_baseline_gaps(tsc: pd.DataFrame, sample_ids: set[int]) -> pd.DataFrame:
    rows = []
    sub = tsc[tsc["patient_id"].isin(sample_ids)]
    for _, row in sub.iterrows():
        sim = _split(row["simulated_sequence"])
        actual = _split(row["actual_sequence"])
        flags = _gap_flags(sim, actual)
        rows.append({
            "patient_id": row["patient_id"],
            "disease": row["disease"],
            "condition": "rubric_only",
            "info_order": "n/a",
            "llm_length": len(sim),
            "actual_length": len(actual),
            **flags,
        })
    return pd.DataFrame(rows)


def aggregate_by_condition_disease(per_patient: pd.DataFrame) -> pd.DataFrame:
    """Mean rates by (condition, disease), plus an all-disease row."""
    metrics = ["exact_match", "has_commission", "has_omission", "has_order_swap"]
    rows = []
    for (cond, disease), sub in per_patient.groupby(["condition", "disease"]):
        n = len(sub)
        row = {"condition": cond, "disease": disease, "n_patients": n}
        for m in metrics:
            row[m + "_rate"] = round(sub[m].mean(), 3)
        row["mean_llm_length"] = round(sub["llm_length"].mean(), 2)
        row["mean_actual_length"] = round(sub["actual_length"].mean(), 2)
        row["mean_n_commission"] = round(sub["n_commission"].mean(), 2)
        row["mean_n_omission"] = round(sub["n_omission"].mean(), 2)
        rows.append(row)
    for cond, sub in per_patient.groupby("condition"):
        n = len(sub)
        row = {"condition": cond, "disease": "ALL", "n_patients": n}
        for m in metrics:
            row[m + "_rate"] = round(sub[m].mean(), 3)
        row["mean_llm_length"] = round(sub["llm_length"].mean(), 2)
        row["mean_actual_length"] = round(sub["actual_length"].mean(), 2)
        row["mean_n_commission"] = round(sub["n_commission"].mean(), 2)
        row["mean_n_omission"] = round(sub["n_omission"].mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["disease", "condition"]).reset_index(drop=True)


def aggregate_by_cs_tercile(per_patient: pd.DataFrame) -> pd.DataFrame:
    metrics = ["exact_match", "has_commission", "has_omission", "has_order_swap"]
    rows = []
    for (cond, cs), sub in per_patient.groupby(["condition", "cs_tercile"]):
        n = len(sub)
        row = {"condition": cond, "cs_tercile": str(cs), "n_patients": n}
        for m in metrics:
            row[m + "_rate"] = round(sub[m].mean(), 3)
        row["mean_llm_length"] = round(sub["llm_length"].mean(), 2)
        row["mean_actual_length"] = round(sub["actual_length"].mean(), 2)
        row["mean_n_commission"] = round(sub["n_commission"].mean(), 2)
        row["mean_n_omission"] = round(sub["n_omission"].mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["condition", "cs_tercile"]).reset_index(drop=True)


def build_full_tables(llm_seq_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (per_patient, gap_by_cond_disease, gap_by_cs_tercile)."""
    llm_df = pd.read_csv(llm_seq_csv)
    tsc = pd.read_csv(_TSC)
    per_patient = _per_patient_gaps(llm_df, tsc)
    sample_ids = set(per_patient["patient_id"].unique())
    # Append rubric_only baseline rows (carry cs_tercile from llm rows)
    rubric_rows = _rubric_baseline_gaps(tsc, sample_ids)
    # Attach cs_tercile from llm sample
    cs_map = per_patient.drop_duplicates("patient_id").set_index("patient_id")["cs_tercile"].to_dict()
    rubric_rows["cs_tercile"] = rubric_rows["patient_id"].map(cs_map)
    combined = pd.concat([per_patient, rubric_rows], ignore_index=True)
    return combined, aggregate_by_condition_disease(combined), aggregate_by_cs_tercile(combined)
