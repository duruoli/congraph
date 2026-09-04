#!/usr/bin/env python3
"""Stage-3 ACR-blind exploration of empirical A/Q/C pattern candidates."""

from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from aqc_pattern_rules import RULES, make_context, strict_eligible


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "aqc_analysis" / "development_v1"
CODEBOOK = ROOT / "data" / "aqc_analysis" / "pattern_codebook_draft_v1.json"
OPPORTUNITIES_OUT = DATA / "pattern_opportunities.jsonl"
OCCURRENCES_OUT = DATA / "pattern_occurrences.jsonl"
REVIEW_OUT = DATA / "pattern_review_queue.jsonl"
SUMMARY_OUT = DATA / "pattern_exploration_summary.json"
REPORT_OUT = DATA / "pattern_exploration_report.md"


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


def unit_id(row: dict[str, Any], unit: str) -> str:
    return row["transition_id"] if unit == "transition" else row["step_id"]


def anchor_step_id(row: dict[str, Any], unit: str) -> str:
    return row["target_step_id"] if unit == "transition" else row["step_id"]


def step_snapshot(step: dict[str, Any]) -> dict[str, Any]:
    annotation = step["effective_annotation"]
    question = annotation.get("current_question") or {}
    coverage = annotation.get("coverage") or {}
    previous = annotation.get("previous_order_update") or {}
    return {
        "step_id": step["step_id"],
        "ordered": step["ordered"],
        "modality_family": step["modality_family"],
        "assumptions": [
            {
                "proposition": item.get("proposition"),
                "type": item.get("type"),
                "status": item.get("status"),
                "support": item.get("support"),
            }
            for item in annotation.get("assumptions", [])
        ],
        "question": {
            "primary": question.get("primary"),
            "target": question.get("target"),
            "type": question.get("type"),
            "secondary_questions": question.get("secondary_questions") or [],
        },
        "requirements": [
            {
                "requirement_key": item.get("requirement_key"),
                "type": item.get("id"),
                "dimension": item.get("dimension"),
            }
            for item in question.get("answer_requirements", [])
        ],
        "coverage": {
            "aggregate": coverage.get("aggregate"),
            "requirements": [
                {
                    "requirement_key": item.get("requirement_key"),
                    "type": item.get("requirement_id"),
                    "status": item.get("status"),
                    "direction": item.get("direction"),
                    "reason": item.get("reason"),
                }
                for item in coverage.get("requirements", [])
            ],
        },
        "previous_order_update": {
            "study_adequacy": previous.get("study_adequacy"),
            "test_question_capability": previous.get("test_question_capability"),
            "result_status": previous.get("result_status"),
            "effect_on_previous_question": previous.get("effect_on_previous_question"),
            "discordance": (previous.get("discordance") or {}).get("label"),
        },
        "assumption_change": (annotation.get("assumption_change") or {}).get("label"),
        "question_continuity": annotation.get("question_continuity"),
        "question_grounding": step.get("question_grounding"),
        "legacy_intent_support": step.get("legacy_intent_support"),
        "legacy_unsupported_residual": step.get("legacy_unsupported_residual"),
    }


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def stratify(
    opportunities: list[dict[str, Any]], candidates: list[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    opportunity_counts = Counter(str(row.get(field)) for row in opportunities)
    candidate_counts = Counter(str(row.get(field)) for row in candidates)
    result = {}
    for value in sorted(opportunity_counts):
        n_opportunities = opportunity_counts[value]
        n_candidates = candidate_counts[value]
        candidate_subset = [row for row in candidates if str(row.get(field)) == value]
        result[value] = {
            "n_opportunities": n_opportunities,
            "n_candidate_units": n_candidates,
            "candidate_fraction_of_opportunities": rate(n_candidates, n_opportunities),
            "n_candidate_patients": len({row["patient_id"] for row in candidate_subset}),
        }
    return result


def diverse_ids(rows: list[dict[str, Any]], unit: str, limit: int = 4) -> list[str]:
    selected: list[str] = []
    seen_diseases: set[str] = set()
    for row in sorted(rows, key=lambda item: (item["disease"], item["patient_id"], unit_id(item, unit))):
        if row["disease"] not in seen_diseases:
            selected.append(unit_id(row, unit))
            seen_diseases.add(row["disease"])
        if len(selected) == limit:
            return selected
    for row in sorted(rows, key=lambda item: unit_id(item, unit)):
        identifier = unit_id(row, unit)
        if identifier not in selected:
            selected.append(identifier)
        if len(selected) == limit:
            break
    return selected


def version_instability(strata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    eligible = {
        key: value["candidate_fraction_of_opportunities"]
        for key, value in strata.items()
        if value["n_opportunities"] >= 10 and value["candidate_fraction_of_opportunities"] is not None
    }
    if len(eligible) < 2:
        return {"assessable": False, "max_absolute_fraction_difference": None, "flag_ge_0_15": False}
    spread = round(max(eligible.values()) - min(eligible.values()), 4)
    return {"assessable": True, "max_absolute_fraction_difference": spread, "flag_ge_0_15": spread >= 0.15}


def opportunity_record(
    definition: dict[str, Any], row: dict[str, Any], candidate: bool
) -> dict[str, Any]:
    unit = definition["unit"]
    record = {
        "pattern_id": definition["pattern_id"],
        "pattern_name": definition["name"],
        "unit": unit,
        "unit_id": unit_id(row, unit),
        "anchor_step_id": anchor_step_id(row, unit),
        "patient_id": row["patient_id"],
        "disease": row["disease"],
        "prompt_version_sha256": row.get("prompt_version_sha256"),
        "schema_version": row.get("schema_version"),
        "is_candidate": candidate,
        "strict_sensitivity_eligible": strict_eligible(row),
        "has_unclear_value": bool(row.get("has_unclear_value")),
        "has_weak_assumption_support": bool(row.get("has_weak_assumption_support")),
    }
    if unit == "transition":
        record.update({
            "source_step_id": row["source_step_id"],
            "target_step_id": row["target_step_id"],
            "observed_action_relation": row["observed_action_relation"],
            "source_modality_family": row["source_modality_family"],
            "target_modality_family": row["target_modality_family"],
            "source_question_type": row["source_question_type"],
            "target_question_type": row["target_question_type"],
            "source_coverage_aggregate": row["source_coverage_aggregate"],
            "target_coverage_aggregate": row["target_coverage_aggregate"],
            "previous_study_adequacy": row["previous_study_adequacy"],
            "previous_test_question_capability": row["previous_test_question_capability"],
            "previous_result_status": row["previous_result_status"],
            "question_continuity": row["question_continuity"],
            "discordance": row["discordance"],
        })
    else:
        record.update({
            "step_index": row["step_index"],
            "ordered": row["ordered"],
            "modality_family": row["modality_family"],
            "question_type": row["question_type"],
            "coverage_aggregate": row["coverage_aggregate"],
            "question_grounding": row.get("question_grounding"),
            "legacy_intent_support": row.get("legacy_intent_support"),
        })
    return record


def review_reasons(occurrence: dict[str, Any], anchor: dict[str, Any]) -> list[str]:
    reasons = []
    if occurrence["pattern_id"] in {"AQC_P09", "AQC_P10"}:
        reasons.append("rare_boundary_pattern")
    if occurrence["pattern_id"] == "AQC_P11":
        reasons.append("order_support_concern")
    if occurrence["pattern_id"] == "AQC_P07":
        reasons.append("temporal_semantic_review")
    if occurrence["has_unclear_value"]:
        reasons.append("unclear_annotation")
    if occurrence["has_weak_assumption_support"]:
        reasons.append("weak_assumption_support")
    if anchor["evidence_qc_status"] == "not_uniformly_audited":
        reasons.append("not_uniformly_audited")
    return reasons


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Development A/Q/C pattern exploration",
        "",
        "Status: exploratory Stage-3 output; pattern definitions are not frozen.",
        "",
        "This report is ACR-blind. Counts describe the 235-patient development partition and are",
        "not prevalence estimates, causal effects, normative recommendations, or final-test results.",
        "No rule uses the annotation's `derived_transition_reference` as an input.",
        "",
        "## Candidate overview",
        "",
        "| ID | Pattern | Opportunities | Candidate units | Patients | Strict units | Diseases |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["patterns"]:
        lines.append(
            f"| {item['pattern_id']} | {item['name']} | {item['n_opportunities']} | "
            f"{item['n_candidate_units']} | {item['n_candidate_patients']} | "
            f"{item['n_strict_candidate_units']} | {item['n_diseases_with_candidates']} |"
        )
    lines.extend(["", "## Definition diagnostics", ""])
    for item in summary["patterns"]:
        diagnostics = []
        if item["candidate_fraction_of_opportunities"] is not None and item["candidate_fraction_of_opportunities"] >= 0.9:
            diagnostics.append("near-universal within its opportunity set; discriminative value requires review")
        if item["n_candidate_patients"] < 5:
            diagnostics.append("rare boundary candidate; do not estimate a stable rate")
        if item["schema_version_instability"]["flag_ge_0_15"]:
            diagnostics.append("candidate fraction differs by at least 0.15 across assessable schema versions")
        if not diagnostics:
            diagnostics.append("no automatic definition warning")
        lines.append(f"- `{item['pattern_id']}`: " + "; ".join(diagnostics) + ".")
    lines.extend(["", "## Example and counterexample queues", ""])
    for item in summary["patterns"]:
        examples = ", ".join(f"`{value}`" for value in item["example_candidate_unit_ids"]) or "none"
        counterexamples = ", ".join(f"`{value}`" for value in item["example_counterexample_unit_ids"]) or "none"
        lines.append(f"- `{item['pattern_id']}` candidates: {examples}; opportunity counterexamples: {counterexamples}.")
    lines.extend(["", "## Highest-overlap candidate pairs", ""])
    if summary["candidate_overlaps"]:
        for item in summary["candidate_overlaps"][:15]:
            lines.append(f"- `{item['pattern_a']}` + `{item['pattern_b']}`: {item['n_shared_anchor_steps']} target/anchor steps.")
    else:
        lines.append("- No overlaps.")
    lines.extend([
        "",
        "## Boundaries retained for Stage 4",
        "",
        "- `close_or_stop` remains unidentifiable without an explicit observation-window/censoring rule.",
        "- `escalation` remains undefined until modality, protocol, and intervention changes receive a purely observational hierarchy.",
        "- `AQC_P09` uses only `materially_discordant`; `indeterminate` cases remain a separate manual audit set.",
        "- `AQC_P11` must be reported separately by schema generation; its two support fields are not pooled as equivalent measurements.",
        "- High candidate fractions in P02--P05 may reflect how opportunity sets were defined and require counterexample review before freezing.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    manifest = read_json(DATA / "manifest.json")
    codebook = read_json(CODEBOOK)
    steps = read_jsonl(DATA / "steps.jsonl")
    transitions = read_jsonl(DATA / "transitions.jsonl")
    requirements = read_jsonl(DATA / "requirements.jsonl")
    if manifest["counts"]["patients"] != 235 or manifest["counts"]["final_test_patients_included"] != 0:
        raise AssertionError("Stage 3 requires the development-only v1 release")
    context = make_context(steps, requirements)
    step_by_id = context.step_by_id
    expected_ids = {item["pattern_id"] for item in codebook["patterns"]}
    if expected_ids != set(RULES):
        raise AssertionError("pattern codebook and executable rules differ")

    opportunity_rows: list[dict[str, Any]] = []
    occurrences: list[dict[str, Any]] = []
    candidates_by_pattern: dict[str, list[dict[str, Any]]] = {}
    opportunities_by_pattern: dict[str, list[dict[str, Any]]] = {}
    summaries = []
    candidate_patterns_by_anchor: dict[str, set[str]] = defaultdict(set)

    for definition in codebook["patterns"]:
        pattern_id = definition["pattern_id"]
        unit = definition["unit"]
        source_rows = transitions if unit == "transition" else steps
        opportunities, candidates = [], []
        for row in source_rows:
            is_opportunity, is_candidate = RULES[pattern_id](row, context)
            if not is_opportunity:
                continue
            opportunities.append(row)
            flat = opportunity_record(definition, row, is_candidate)
            opportunity_rows.append(flat)
            if is_candidate:
                candidates.append(row)
                anchor_id = anchor_step_id(row, unit)
                candidate_patterns_by_anchor[anchor_id].add(pattern_id)
                occurrence = {
                    **flat,
                    "occurrence_id": f"{pattern_id}:{unit_id(row, unit)}",
                    "source_state": (
                        step_snapshot(step_by_id[row["source_step_id"]]) if unit == "transition" else None
                    ),
                    "target_state": step_snapshot(step_by_id[anchor_id]),
                }
                occurrences.append(occurrence)
        opportunities_by_pattern[pattern_id] = opportunities
        candidates_by_pattern[pattern_id] = candidates
        strict_candidates = [row for row in candidates if strict_eligible(row)]
        by_schema = stratify(opportunities, candidates, "schema_version")
        noncandidates = [row for row in opportunities if row not in candidates]
        summaries.append({
            "pattern_id": pattern_id,
            "name": definition["name"],
            "unit": unit,
            "n_opportunities": len(opportunities),
            "n_candidate_units": len(candidates),
            "candidate_fraction_of_opportunities": rate(len(candidates), len(opportunities)),
            "n_candidate_patients": len({row["patient_id"] for row in candidates}),
            "n_strict_candidate_units": len(strict_candidates),
            "n_strict_candidate_patients": len({row["patient_id"] for row in strict_candidates}),
            "n_diseases_with_candidates": len({row["disease"] for row in candidates}),
            "by_disease": stratify(opportunities, candidates, "disease"),
            "by_schema_version": by_schema,
            "by_prompt_version": stratify(opportunities, candidates, "prompt_version_sha256"),
            "by_observed_action_relation": (
                stratify(opportunities, candidates, "observed_action_relation") if unit == "transition" else None
            ),
            "schema_version_instability": version_instability(by_schema),
            "example_candidate_unit_ids": diverse_ids(candidates, unit),
            "example_counterexample_unit_ids": diverse_ids(noncandidates, unit),
        })

    occurrence_by_id = {row["occurrence_id"]: row for row in occurrences}
    if len(occurrence_by_id) != len(occurrences):
        raise AssertionError("occurrence_id is not unique")
    review_queue = []
    for occurrence in occurrences:
        anchor = step_by_id[occurrence["anchor_step_id"]]
        reasons = review_reasons(occurrence, anchor)
        if not reasons:
            continue
        high = {"rare_boundary_pattern", "order_support_concern", "unclear_annotation"}
        priority = "high" if high.intersection(reasons) else "medium"
        review_queue.append({
            "review_id": f"review:{occurrence['occurrence_id']}",
            "priority": priority,
            "review_reasons": reasons,
            "occurrence_id": occurrence["occurrence_id"],
            "pattern_id": occurrence["pattern_id"],
            "unit_id": occurrence["unit_id"],
            "anchor_step_id": occurrence["anchor_step_id"],
            "patient_id": occurrence["patient_id"],
            "disease": occurrence["disease"],
            "schema_version": occurrence["schema_version"],
        })

    overlaps = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    for pattern_ids in candidate_patterns_by_anchor.values():
        for pair in itertools.combinations(sorted(pattern_ids), 2):
            pair_counts[pair] += 1
    for (pattern_a, pattern_b), count in pair_counts.most_common():
        overlaps.append({"pattern_a": pattern_a, "pattern_b": pattern_b, "n_shared_anchor_steps": count})

    p05_legacy = [
        row for row in transitions
        if row.get("previous_result_status") in {"not_appessed", "not_appraised", "mixed"}
        and row.get("target_coverage_aggregate") in {"unanswered", "partially_answered"}
        and row.get("question_continuity") in {"same", "refined", "reopened"}
    ]
    p09_indeterminate = [row for row in transitions if row.get("discordance") == "indeterminate"]
    summary = {
        "schema_version": "0.1.0-stage-3-exploration",
        "status": "exploratory_not_frozen",
        "acr_used": False,
        "uses_derived_transition_reference_as_rule_input": False,
        "source_manifest": "data/aqc_analysis/development_v1/manifest.json",
        "pattern_codebook": "data/aqc_analysis/pattern_codebook_draft_v1.json",
        "counts": {
            "patients": 235,
            "steps": len(steps),
            "transitions": len(transitions),
            "opportunity_rows": len(opportunity_rows),
            "candidate_occurrences": len(occurrences),
            "review_queue_rows": len(review_queue),
        },
        "patterns": summaries,
        "candidate_overlaps": overlaps,
        "boundary_sensitivity": {
            "AQC_P05_legacy_nonstandard_result_status_candidates": len(p05_legacy),
            "AQC_P05_legacy_nonstandard_result_status_unit_ids": [row["transition_id"] for row in p05_legacy],
            "AQC_P09_indeterminate_manual_audit_units": len(p09_indeterminate),
            "AQC_P09_indeterminate_manual_audit_unit_ids": [row["transition_id"] for row in p09_indeterminate],
        },
        "limitations": [
            "DIRECT reference annotation saw the actual order; recurrence can reflect order-driven reconstruction.",
            "Candidate fractions are development descriptions, not population prevalence or final replication estimates.",
            "Weak assumption support is common, so primary and strict-sensitivity counts must be shown together.",
            "Stop remains unidentifiable from the last observed step without a censoring model.",
            "No ACR Context or appropriateness information was used.",
        ],
    }
    write_jsonl(OPPORTUNITIES_OUT, opportunity_rows)
    write_jsonl(OCCURRENCES_OUT, occurrences)
    write_jsonl(REVIEW_OUT, sorted(review_queue, key=lambda row: (row["priority"] != "high", row["pattern_id"], row["unit_id"])))
    write_json(SUMMARY_OUT, summary)
    with REPORT_OUT.open("w", encoding="utf-8") as handle:
        handle.write(render_report(summary))
    print(
        f"wrote {len(opportunity_rows)} opportunities, {len(occurrences)} candidate occurrences, "
        f"and {len(review_queue)} review rows"
    )


if __name__ == "__main__":
    main()
