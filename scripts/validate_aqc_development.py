"""Validate the formal Track-B development artifacts and causal boundaries."""
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
DISEASES = {"appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"}
ALLOWED_EX_ANTE_FIELDS = {
    "differential", "other_hypothesis", "information_gap", "expected_finding",
    "action_role", "appropriateness", "appropriateness_reason", "grounding", "reasoning",
}
FORBIDDEN_CAUSAL_KEYS = {
    "verification", "actual_finding", "certainty_update", "dev_belief", "deviation",
    "later_event", "masked_result_of_this_test", "acr_rating",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(nested_keys(child))
    return keys


def assert_unique(values: list[str], label: str) -> None:
    assert len(values) == len(set(values)), f"duplicate {label}"


def main() -> None:
    split = read_json(DATA / "split_manifest.json")
    manifest = read_json(DATA / "development_sample_manifest.json")
    assumption_book = read_json(DATA / "assumption_codebook_v1.json")
    question_book = read_json(DATA / "question_codebook_v1.json")
    coverage_book = read_json(DATA / "coverage_contract_v1.json")
    audit = read_json(DATA / "audit_records.json")
    saturation = read_json(DATA / "saturation_audit.json")
    rows = [json.loads(line) for line in
            (DATA / "discovery_open_coding.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]

    # Corpus and patient-level split.
    patients = split["patients"]
    assert len(patients) == 293
    assert sum(row["n_steps"] for row in patients) == 542
    assert {row["disease"] for row in patients} == DISEASES
    patient_keys = [(row["disease"], int(row["hadm_id"])) for row in patients]
    assert len(patient_keys) == len(set(patient_keys))
    assert split["summary"]["development"]["n_patients"] == 235
    assert split["summary"]["development"]["n_steps"] == 433
    assert split["summary"]["final_test"]["n_patients"] == 58
    assert split["summary"]["final_test"]["n_steps"] == 109
    assert all(row["partition"] in {"development", "final_test"} for row in patients)

    development = {(row["disease"], int(row["hadm_id"])) for row in patients
                   if row["partition"] == "development"}
    final_test = {(row["disease"], int(row["hadm_id"])) for row in patients
                  if row["partition"] == "final_test"}
    assert development.isdisjoint(final_test)
    assert len(development | final_test) == 293
    assert ("appendicitis", 20123918) in development
    assert ("appendicitis", 20123918) not in final_test
    for disease in DISEASES:
        assert any(key[0] == disease for key in development)
        assert any(key[0] == disease for key in final_test)

    # Development sample and fresh-batch non-overlap.
    batches = manifest["batches"]
    assert [batch["name"] for batch in batches] == [
        "initial_24", "saturation_check_1", "saturation_check_2"
    ]
    assert [batch["summary"]["n_patients"] for batch in batches] == [24, 12, 12]
    selected: list[tuple[str, int]] = []
    for batch in batches:
        for patient in batch["patients"]:
            key = (patient["disease"], int(patient["hadm_id"]))
            selected.append(key)
            assert key in development
            assert key not in final_test
    assert len(selected) == len(set(selected)) == 48
    assert manifest["all_coded_development"]["n_steps"] == len(rows) == 137

    # Exact source traceability and two-view causal boundary.
    assert_unique([row["coding_id"] for row in rows], "coding_id")
    source_cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = (row["disease_stratum_sampling_only"], int(row["hadm_id"]))
        assert key in development and key not in final_test
        source_path = row["source_path"]
        assert source_path.startswith("results/annotation_experiment/full/")
        source = source_cache.setdefault(source_path, read_json(ROOT / source_path))
        step = next(item for item in source["steps"] if int(item["step"]) == int(row["step"]))
        ex_ante = step["representative_ex_ante"]
        copied = row["view_2_schema_light"]["source_fields_verbatim"]
        assert set(copied) == ALLOWED_EX_ANTE_FIELDS
        for field in ALLOWED_EX_ANTE_FIELDS:
            assert copied[field] == ex_ante.get(field)
        assert row["view_1_reasoning_only"]["reasoning_verbatim"] == ex_ante.get("reasoning", "")
        assert not (nested_keys(row) & FORBIDDEN_CAUSAL_KEYS)

    # Codebook and output-contract enums.
    assumption_types = [item["id"] for item in assumption_book["types"]]
    question_types = [item["id"] for item in question_book["types"]]
    requirement_types = list(question_book["answer_requirement_definitions"])
    assert_unique(assumption_types, "assumption type")
    assert_unique(question_types, "question type")
    assert {"other", "unclear"}.issubset(assumption_types)
    assert {"other", "unclear"}.issubset(question_types)
    assert set(prompts.ASSUMPTION_TYPES) == set(assumption_types)
    assert set(prompts.QUESTION_TYPES) == set(question_types)
    assert set(prompts.ANSWER_REQUIREMENT_TYPES) == set(requirement_types)
    assert set(coverage_book["required_fields"]["status"]) == set(prompts.COVERAGE_STATUSES)
    assert set(coverage_book["required_fields"]["direction"]) == set(prompts.COVERAGE_DIRECTIONS)
    assert set(coverage_book["aggregate"]["allowed"]) == set(prompts.COVERAGE_AGGREGATES)

    alias = question_book.get("candidate_aliases", {})
    allowed_question_candidates = set(question_types) | set(alias)
    allowed_requirements = set(requirement_types)
    for row in rows:
        for view_name in ("view_1_reasoning_only", "view_2_schema_light"):
            view = row[view_name]
            assert set(view["open_assumption_type_candidates"]) <= set(assumption_types)
            assert set(view["open_question_type_candidates"]) <= allowed_question_candidates
            assert set(view["open_answer_requirement_candidates"]) <= allowed_requirements

    rows_by_id = {row["coding_id"]: row for row in rows}
    for book in (assumption_book, question_book):
        for type_rule in book["types"]:
            for example in type_rule["source_examples"]:
                source_fields = rows_by_id[example["coding_id"]]["view_2_schema_light"]["source_fields_verbatim"]
                searchable = " ".join(str(source_fields.get(field, "")) for field in (
                    "reasoning", "information_gap", "expected_finding", "other_hypothesis",
                    "appropriateness_reason",
                ))
                assert example["quote"] in searchable

    contract = prompts.output_contract()
    assert "answer_requirements" in contract["current_question"]
    assert "coverage" in contract and "pre_order_coverage" not in contract
    assert "requirements" in contract["coverage"]

    # Deterministic lexical stability is not qualitative saturation.
    assert saturation["conclusion"] == "lexically_stable_but_not_yet_qualitatively_saturated"
    assert audit["freeze_status"] == "not_frozen"
    n_scaffold = sum(bool(row["view_comparison"]["possible_scaffold_induction"]) for row in rows)
    assert audit["paired_view_audit"]["steps_with_any_candidate_added_after_schema_light_view"] == n_scaffold

    # Sentinel prompt checks for current-result/later-event leakage.
    decision_point = {
        "ordered": "ORDER_VISIBLE", "visible_prior_imaging": [],
        "masked_result_of_this_test": "CURRENT_RESULT_SECRET",
        "later_event": "LATER_EVENT_SECRET",
    }
    baseline = {"patient_history": "history", "physical_examination": "exam",
                "laboratory_tests": "labs"}
    direct = prompts.build_direct_user(decision_point, baseline, None)
    assert "ORDER_VISIBLE" in direct
    assert "CURRENT_RESULT_SECRET" not in direct
    assert "LATER_EVENT_SECRET" not in direct

    old = {"differential": {"other": 1.0}, "reasoning": "OLD_EX_ANTE_VISIBLE",
           "verification": "VERIFICATION_SECRET", "actual_finding": "CURRENT_RESULT_SECRET"}
    recode = prompts.build_recode_user(old, "ORDER_VISIBLE", None)
    assert "OLD_EX_ANTE_VISIBLE" in recode
    assert "VERIFICATION_SECRET" not in recode
    assert "CURRENT_RESULT_SECRET" not in recode

    print(
        "A/Q/C development validation passed: 293 patients/542 steps split; "
        "48 development trajectories/137 coded steps; final-test isolation, source traceability, "
        "enum contracts, requirement coverage, and prompt masking intact"
    )


if __name__ == "__main__":
    main()
