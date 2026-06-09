"""Gap analysis: compare LLM / doctor sequences against the rubric as reference.

Convention (rubric-relative, identical to scripts/gap_analysis.py):
  commission = (agent_tests − rubric_tests) − {Radiograph_Chest}
               agent ordered extra tests not recommended by rubric
  omission   = rubric_tests − agent_tests
               agent skipped tests the rubric recommended
  order_swap = relative order of shared tests differs

Three conditions on the same rubric-relative axis:
  llm_features_only : LLM sequence (features only)
  llm_full          : LLM sequence (features + rubric hint + KNN)
  doctor            : actual physician sequences
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


def _gap_flags(rubric: list[str], agent: list[str]) -> dict:
    """Compute commission/omission of `agent` relative to `rubric`.

    commission = agent did extra tests not in rubric
    omission   = rubric-recommended tests the agent skipped
    """
    rubric_set = set(rubric)
    agent_set = set(agent)
    commission = (agent_set - rubric_set) - EXCLUDE_FROM_COMMISSION
    omission = rubric_set - agent_set
    shared = rubric_set & agent_set
    rubric_shared = [t for t in rubric if t in shared]
    agent_shared = [t for t in agent if t in shared]
    order_swap = rubric_shared != agent_shared
    return {
        "matches_rubric": int(rubric == agent),
        "has_commission": int(len(commission) > 0),
        "has_omission": int(len(omission) > 0),
        "has_order_swap": int(order_swap),
        "n_commission": len(commission),
        "n_omission": len(omission),
        "commission_tests": sorted(commission),
        "omission_tests": sorted(omission),
    }


def _per_patient_gaps(llm_seq_df: pd.DataFrame, tsc: pd.DataFrame) -> pd.DataFrame:
    """Compare LLM sequences against rubric (rubric is the reference)."""
    rubric = tsc.set_index("patient_id")["simulated_sequence"].to_dict()
    doctor = tsc.set_index("patient_id")["actual_sequence"].to_dict()
    out_rows = []
    for _, row in llm_seq_df.iterrows():
        pid = int(row["patient_id"])
        if pid not in rubric:
            continue
        rubric_seq = _split(rubric[pid])
        llm_seq = _prepend_lab(_split(row["llm_sequence"]))
        flags = _gap_flags(rubric_seq, llm_seq)
        out_rows.append({
            "patient_id": pid,
            "disease": row["disease"],
            "cs_tercile": row["cs_tercile"],
            "condition": row["condition"],
            "info_order": row["info_order"],
            "agent_sequence": ", ".join(llm_seq),
            "rubric_sequence": rubric.get(pid, ""),
            "doctor_sequence": doctor.get(pid, ""),
            "seq_length": len(llm_seq),
            "rubric_length": len(rubric_seq),
            **flags,
        })
    return pd.DataFrame(out_rows)


def _doctor_gaps(tsc: pd.DataFrame, sample_ids: set[int]) -> pd.DataFrame:
    """Compare doctor sequences against rubric — same axis as LLM conditions."""
    rows = []
    sub = tsc[tsc["patient_id"].isin(sample_ids)]
    for _, row in sub.iterrows():
        rubric_seq = _split(row["simulated_sequence"])
        doctor_seq = _split(row["actual_sequence"])
        flags = _gap_flags(rubric_seq, doctor_seq)
        rows.append({
            "patient_id": row["patient_id"],
            "disease": row["disease"],
            "condition": "doctor",
            "info_order": "n/a",
            "seq_length": len(doctor_seq),
            "rubric_length": len(rubric_seq),
            **flags,
        })
    return pd.DataFrame(rows)


def _aggregate(per_patient: pd.DataFrame, groupby: list[str]) -> pd.DataFrame:
    metrics = ["matches_rubric", "has_commission", "has_omission", "has_order_swap"]
    rows = []
    for keys, sub in per_patient.groupby(groupby):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(sub)
        row = dict(zip(groupby, keys))
        row["n_patients"] = n
        for m in metrics:
            row[m + "_rate"] = round(sub[m].mean(), 3)
        row["mean_seq_length"] = round(sub["seq_length"].mean(), 2)
        row["mean_rubric_length"] = round(sub["rubric_length"].mean(), 2)
        row["mean_n_commission"] = round(sub["n_commission"].mean(), 2)
        row["mean_n_omission"] = round(sub["n_omission"].mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_by_condition_disease(per_patient: pd.DataFrame) -> pd.DataFrame:
    by_cd = _aggregate(per_patient, ["condition", "disease"])
    by_c = _aggregate(per_patient, ["condition"])
    by_c.insert(1, "disease", "ALL")
    return pd.concat([by_cd, by_c], ignore_index=True).sort_values(
        ["disease", "condition"]
    ).reset_index(drop=True)


def aggregate_by_cs_tercile(per_patient: pd.DataFrame) -> pd.DataFrame:
    return _aggregate(per_patient, ["condition", "cs_tercile"]).sort_values(
        ["condition", "cs_tercile"]
    ).reset_index(drop=True)


def build_full_tables(llm_seq_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (per_patient, gap_by_cond_disease, gap_by_cs_tercile).

    All three conditions (llm_features_only, llm_full, doctor) are compared
    against the rubric on the same commission/omission axis.
    """
    llm_df = pd.read_csv(llm_seq_csv)
    tsc = pd.read_csv(_TSC)
    per_llm = _per_patient_gaps(llm_df, tsc)
    sample_ids = set(per_llm["patient_id"].unique())

    doctor_rows = _doctor_gaps(tsc, sample_ids)
    cs_map = (
        per_llm.drop_duplicates("patient_id")
        .set_index("patient_id")["cs_tercile"]
        .to_dict()
    )
    doctor_rows["cs_tercile"] = doctor_rows["patient_id"].map(cs_map)

    combined = pd.concat([per_llm, doctor_rows], ignore_index=True)
    return combined, aggregate_by_condition_disease(combined), aggregate_by_cs_tercile(combined)
