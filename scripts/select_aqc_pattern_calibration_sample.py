#!/usr/bin/env python3
"""Select a deterministic, ACR-blind Stage-3 pattern calibration sample."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_aqc_patterns import step_snapshot


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "aqc_analysis" / "development_v1"
SAMPLE_OUT = DATA / "pattern_calibration_sample.jsonl"
MANIFEST_OUT = DATA / "pattern_calibration_sample_manifest.json"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def p01_class(row: dict[str, Any]) -> str | None:
    continuity = row["question_continuity"] in {"new", "reopened"}
    type_change = row["source_question_type"] != row["target_question_type"]
    if continuity and type_change:
        return "both_continuity_and_type_change"
    if continuity:
        return "continuity_only"
    if type_change:
        return "type_change_only"
    return None


def assumption_profiles(step: dict[str, Any]) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    types: dict[str, int] = defaultdict(int)
    statuses: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for assumption in step["effective_annotation"].get("assumptions") or []:
        assumption_type = str(assumption.get("type"))
        assumption_status = str(assumption.get("status"))
        types[assumption_type] += 1
        statuses[assumption_type][assumption_status] += 1
    return dict(types), {key: dict(value) for key, value in statuses.items()}


def p08_class(row: dict[str, Any], steps_by_id: dict[str, dict[str, Any]]) -> str | None:
    source_types, source_status = assumption_profiles(steps_by_id[row["source_step_id"]])
    target_types, target_status = assumption_profiles(steps_by_id[row["target_step_id"]])
    type_change = source_types != target_types
    shared = set(source_types) & set(target_types)
    status_change = any(source_status[key] != target_status[key] for key in shared)
    if type_change and status_change:
        return "both_type_multiset_and_shared_type_status_change"
    if type_change:
        return "type_multiset_change_only"
    if status_change:
        return "shared_type_status_change_only"
    return None


def add_selection(
    selections: dict[str, set[str]], transition_id: str, reason: str
) -> None:
    selections[transition_id].add(reason)


def first_by_group(
    rows: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    selected: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: item["transition_id"]):
        key = tuple(str(row[field]) for field in group_fields)
        selected.setdefault(key, row)
    return sorted(selected.items())


def main() -> None:
    manifest = read_json(DATA / "manifest.json")
    if manifest["counts"]["patients"] != 235 or manifest["counts"]["final_test_patients_included"] != 0:
        raise AssertionError("Calibration selection requires development-only data")
    steps = read_jsonl(DATA / "steps.jsonl")
    transitions = read_jsonl(DATA / "transitions.jsonl")
    opportunities = read_jsonl(DATA / "pattern_opportunities.jsonl")
    steps_by_id = {row["step_id"]: row for row in steps}
    transitions_by_id = {row["transition_id"]: row for row in transitions}
    candidate_ids: dict[str, set[str]] = defaultdict(set)
    opportunity_ids: dict[str, set[str]] = defaultdict(set)
    for row in opportunities:
        if row["unit"] != "transition":
            continue
        opportunity_ids[row["pattern_id"]].add(row["unit_id"])
        if row["is_candidate"]:
            candidate_ids[row["pattern_id"]].add(row["unit_id"])

    selections: dict[str, set[str]] = defaultdict(set)

    # P01: one candidate per disease-by-signal class.
    p01_rows = []
    for transition_id in candidate_ids["AQC_P01"]:
        row = dict(transitions_by_id[transition_id])
        row["structural_stratum"] = p01_class(row)
        p01_rows.append(row)
    for (disease, stratum), row in first_by_group(p01_rows, ("disease", "structural_stratum")):
        add_selection(selections, row["transition_id"], f"P01_candidate:{disease}:{stratum}")

    # P02 family: one example for every observed mutually exclusive subtype combination,
    # plus one no-current-mechanism example per disease.
    p02_rows = []
    subtypes = ["AQC_P03", "AQC_P04", "AQC_P05", "AQC_P07"]
    for transition_id in candidate_ids["AQC_P02"]:
        active = [pattern_id for pattern_id in subtypes if transition_id in candidate_ids[pattern_id]]
        row = dict(transitions_by_id[transition_id])
        row["mechanism_signature"] = "+".join(active) if active else "none_of_P03_P04_P05_P07"
        p02_rows.append(row)
    mechanism_rows = [row for row in p02_rows if row["mechanism_signature"] != "none_of_P03_P04_P05_P07"]
    for (signature,), row in first_by_group(mechanism_rows, ("mechanism_signature",)):
        add_selection(selections, row["transition_id"], f"P02_mechanism_combination:{signature}")
    no_mechanism_rows = [row for row in p02_rows if row["mechanism_signature"] == "none_of_P03_P04_P05_P07"]
    for (disease,), row in first_by_group(no_mechanism_rows, ("disease",)):
        add_selection(selections, row["transition_id"], f"P02_no_current_mechanism:{disease}")

    # P06: one candidate and one opportunity counterexample per disease.
    for status, ids in (
        ("candidate", candidate_ids["AQC_P06"]),
        ("counterexample", opportunity_ids["AQC_P06"] - candidate_ids["AQC_P06"]),
    ):
        rows = [transitions_by_id[transition_id] for transition_id in ids]
        for (disease,), row in first_by_group(rows, ("disease",)):
            add_selection(selections, row["transition_id"], f"P06_{status}:{disease}")

    # P08: one candidate per disease-by-structural class.
    p08_rows = []
    for transition_id in candidate_ids["AQC_P08"]:
        row = dict(transitions_by_id[transition_id])
        row["structural_stratum"] = p08_class(row, steps_by_id)
        p08_rows.append(row)
    for (disease, stratum), row in first_by_group(p08_rows, ("disease", "structural_stratum")):
        add_selection(selections, row["transition_id"], f"P08_candidate:{disease}:{stratum}")

    review_prompts = {
        "AQC_P01": "Is this a clinically meaningful question reorientation, or only a question-type taxonomy inconsistency?",
        "AQC_P02_family": "Does an open question truly persist, and which limitation/result/temporal mechanisms are supported without using the observed order as justification?",
        "AQC_P06": "After an informative prior result, is the target question genuinely new or rerouted rather than a relabeling of the same question?",
        "AQC_P08": "Is there substantive assumption revision, or only annotation composition/status drift without proposition identity?",
    }
    sample = []
    for transition_id, reasons in sorted(selections.items()):
        transition = transitions_by_id[transition_id]
        represented_patterns = sorted(
            pattern_id for pattern_id, ids in candidate_ids.items()
            if pattern_id.startswith("AQC_P0") and transition_id in ids
        )
        sample.append({
            "calibration_id": f"calibration:{transition_id}",
            "transition_id": transition_id,
            "patient_id": transition["patient_id"],
            "disease": transition["disease"],
            "selection_reasons": sorted(reasons),
            "represented_candidate_patterns": represented_patterns,
            "review_prompts": review_prompts,
            "transition_fields": transition,
            "source_state": step_snapshot(steps_by_id[transition["source_step_id"]]),
            "target_state": step_snapshot(steps_by_id[transition["target_step_id"]]),
        })

    reason_counts: dict[str, int] = defaultdict(int)
    for reasons in selections.values():
        for reason in reasons:
            reason_counts[reason.split(":", 1)[0]] += 1
    sample_manifest = {
        "schema_version": "0.1.0-stage-3-calibration-selection",
        "status": "selected_not_reviewed",
        "acr_used": False,
        "final_test_used": False,
        "sampling_purpose": "definition calibration and counterexample discovery, not frequency estimation",
        "selection_rule": {
            "AQC_P01": "lexicographically first transition per disease-by-signal class",
            "AQC_P02_family": "lexicographically first transition per observed mechanism combination, plus first no-current-mechanism transition per disease",
            "AQC_P06": "lexicographically first candidate and eligible counterexample per disease",
            "AQC_P08": "lexicographically first transition per disease-by-structural class",
            "deduplication": "one row per transition; retain all selection reasons",
        },
        "counts": {
            "sample_transitions": len(sample),
            "sample_patients": len({row["patient_id"] for row in sample}),
            "sample_diseases": len({row["disease"] for row in sample}),
            "selection_tags_by_family": dict(sorted(reason_counts.items())),
        },
        "output": "data/aqc_analysis/development_v1/pattern_calibration_sample.jsonl",
    }
    write_jsonl(SAMPLE_OUT, sample)
    write_json(MANIFEST_OUT, sample_manifest)
    print(f"selected {len(sample)} transitions from {sample_manifest['counts']['sample_patients']} patients")


if __name__ == "__main__":
    main()
