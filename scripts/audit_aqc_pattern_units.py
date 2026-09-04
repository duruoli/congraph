#!/usr/bin/env python3
"""Feasibility audit for the Stage-2 empirical A/Q/C pattern units.

This does not interpret or retain scientific patterns. It verifies that every draft rule has a
well-defined opportunity set in the development-only analysis layer and reports candidate counts
needed to revise definitions before Stage 3.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from aqc_pattern_rules import RULES, make_context, strict_eligible


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "aqc_analysis" / "development_v1"
CODEBOOK = ROOT / "data" / "aqc_analysis" / "pattern_codebook_draft_v1.json"
OUTPUT = DATA / "pattern_unit_feasibility.json"


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


def main() -> None:
    manifest = read_json(DATA / "manifest.json")
    codebook = read_json(CODEBOOK)
    steps = read_jsonl(DATA / "steps.jsonl")
    transitions = read_jsonl(DATA / "transitions.jsonl")
    requirements = read_jsonl(DATA / "requirements.jsonl")
    if manifest["counts"]["patients"] != 235 or manifest["counts"]["steps"] != 433:
        raise AssertionError("Stage-2 audit requires the frozen 235-patient / 433-step layer")
    if manifest["counts"]["final_test_patients_included"] != 0:
        raise AssertionError("final test entered the empirical pattern layer")

    context = make_context(steps, requirements)
    expected_ids = {item["pattern_id"] for item in codebook["patterns"]}
    if expected_ids != set(RULES):
        raise AssertionError("codebook and executable Stage-2 rules differ")

    results = []
    for definition in codebook["patterns"]:
        pattern_id = definition["pattern_id"]
        rows = transitions if definition["unit"] == "transition" else steps
        rule = RULES[pattern_id]
        opportunities, candidates = [], []
        for row in rows:
            is_opportunity, is_candidate = rule(row, context)
            if is_opportunity:
                opportunities.append(row)
            if is_candidate:
                candidates.append(row)
        patient_field = "patient_id"
        results.append({
            "pattern_id": pattern_id,
            "name": definition["name"],
            "unit": definition["unit"],
            "n_opportunities": len(opportunities),
            "n_candidate_units": len(candidates),
            "n_candidate_patients": len({row[patient_field] for row in candidates}),
            "candidate_by_disease": dict(sorted(Counter(row["disease"] for row in candidates).items())),
            "candidate_by_schema": dict(sorted(Counter(str(row.get("schema_version")) for row in candidates).items())),
            "candidate_by_action_relation": (
                dict(sorted(Counter(row["observed_action_relation"] for row in candidates).items()))
                if definition["unit"] == "transition" else None
            ),
            "n_candidates_with_unclear": sum(bool(row.get("has_unclear_value")) for row in candidates),
            "n_candidates_with_weak_support": sum(bool(row.get("has_weak_support")) for row in candidates),
            "n_strict_candidate_units": sum(
                strict_eligible(row) for row in candidates
            ),
            "n_strict_candidate_patients": len({
                row[patient_field] for row in candidates
                if strict_eligible(row)
            }),
        })

    write_json(OUTPUT, {
        "schema_version": "0.1.0-stage-2-feasibility",
        "interpretation": (
            "Mechanical opportunity/candidate counts used only to test operational definitions. "
            "They are not retained scientific findings or prevalence estimates."
        ),
        "uses_derived_transition_reference_as_rule_input": False,
        "source_manifest": "data/aqc_analysis/development_v1/manifest.json",
        "pattern_codebook": "data/aqc_analysis/pattern_codebook_draft_v1.json",
        "counts": {"patients": 235, "steps": len(steps), "transitions": len(transitions)},
        "patterns": results,
        "deferred_patterns": codebook["deferred_patterns"],
    })
    print(f"audited {len(results)} draft pattern units; wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
