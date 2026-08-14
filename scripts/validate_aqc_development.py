"""Validate Track-B discovery artifacts and paired-prompt causal boundaries."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aqc import prompts  # noqa: E402


DATA = ROOT / "data" / "aqc_development"
SOURCE_FIELDS = (
    "differential",
    "other_hypothesis",
    "information_gap",
    "expected_finding",
    "reasoning",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> None:
    manifest = read_json(DATA / "discovery_sample_manifest.json")
    codebook = read_json(DATA / "provisional_assumption_codebook.json")
    rows = [json.loads(line) for line in (DATA / "discovery_open_coding.jsonl").read_text().splitlines()]

    assert manifest["n_trajectories"] == 16
    assert manifest["n_steps"] == len(rows) == 48
    assert manifest["available_corpus_profile"]["n_unique_trajectories"] == 38
    assert manifest["available_corpus_profile"]["n_decision_steps"] == 81
    assert manifest["trajectory_counts_by_disease"] == {
        "appendicitis": 4,
        "cholecystitis": 4,
        "diverticulitis": 4,
        "pancreatitis": 4,
    }
    assert len({row["sample_id"] for row in rows}) == len(rows)
    assert all("data/acr_normative" not in row["source_path"] for row in rows)
    assert all("source_verification_label" not in row for row in rows)

    type_ids = [item["id"] for item in codebook["types"]]
    status_ids = [item["id"] for item in codebook["statuses"]]
    assert len(type_ids) == len(set(type_ids))
    assert {"other", "unclear"}.issubset(type_ids)
    assert "unclear" in status_ids
    assert set(prompts.ASSUMPTION_TYPES) == set(type_ids)
    assert set(prompts.ASSUMPTION_STATUSES) == set(status_ids)
    assert all(set(row["open_type_codes"]).issubset(type_ids) for row in rows)
    codebook_counts = {item["id"]: item["sample_assignments"] for item in codebook["types"]}
    assert all(
        codebook_counts[code] == count
        for code, count in manifest["open_code_assignments"].items()
    )

    # Prove copied discovery text remains exactly equal to the source annotation.
    source_cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_path = row["source_path"]
        source = source_cache.setdefault(source_path, read_json(ROOT / source_path))
        step = next(item for item in source["steps"] if item["step"] == row["step"])
        ex_ante = step["representative_ex_ante"]
        for field in SOURCE_FIELDS:
            assert row["verbatim_assumption_material"][field] == ex_ante.get(field, {} if field == "differential" else "")

    # Sentinel values ensure neither builder serializes the hidden current result,
    # a local verification label, or arbitrary later-event fields.
    decision_point = {
        "ordered": "ORDER_VISIBLE",
        "visible_prior_imaging": [],
        "masked_result_of_this_test": "CURRENT_RESULT_SECRET",
        "later_event": "LATER_EVENT_SECRET",
    }
    baseline = {
        "patient_history": "history",
        "physical_examination": "exam",
        "laboratory_tests": "labs",
    }
    direct = prompts.build_direct_user(decision_point, baseline, None)
    assert "ORDER_VISIBLE" in direct
    assert "CURRENT_RESULT_SECRET" not in direct
    assert "LATER_EVENT_SECRET" not in direct

    old = {
        "differential": {"other": 1.0},
        "reasoning": "OLD_EX_ANTE_VISIBLE",
        "verification": "VERIFICATION_SECRET",
        "actual_finding": "CURRENT_RESULT_SECRET",
    }
    recode = prompts.build_recode_user(old, "ORDER_VISIBLE", None)
    assert "OLD_EX_ANTE_VISIBLE" in recode
    assert "VERIFICATION_SECRET" not in recode
    assert "CURRENT_RESULT_SECRET" not in recode

    print("A/Q/C development validation passed: 16 trajectories, 48 steps, source text exact, prompt masking intact")


if __name__ == "__main__":
    main()
