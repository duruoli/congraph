#!/usr/bin/env python3
"""Stage-3 structural refinement of broad A/Q/C pattern candidates."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "aqc_analysis" / "development_v1"
TARGETED_REVIEW = ROOT / "data" / "aqc_analysis" / "stage3_targeted_review_v1.json"
OUT = DATA / "pattern_structure_refinement.json"
REPORT = DATA / "pattern_structure_refinement_report.md"


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


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize_groups(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        value: {
            "n_units": len(items),
            "n_patients": len({item["patient_id"] for item in items}),
            "n_diseases": len({item["disease"] for item in items}),
        }
        for value, items in sorted(grouped.items())
    }


def assumption_profiles(step: dict[str, Any]) -> tuple[Counter[str], dict[str, Counter[str]]]:
    type_counts: Counter[str] = Counter()
    status_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for assumption in step["effective_annotation"].get("assumptions") or []:
        assumption_type = str(assumption.get("type"))
        assumption_status = str(assumption.get("status"))
        type_counts[assumption_type] += 1
        status_by_type[assumption_type][assumption_status] += 1
    return type_counts, status_by_type


def p01_decomposition(
    candidates: list[dict[str, Any]], transitions_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    rows = []
    for occurrence in candidates:
        transition = transitions_by_id[occurrence["unit_id"]]
        continuity_signal = transition["question_continuity"] in {"new", "reopened"}
        type_signal = transition["source_question_type"] != transition["target_question_type"]
        if continuity_signal and type_signal:
            signal_class = "both_continuity_and_type_change"
        elif continuity_signal:
            signal_class = "continuity_only"
        elif type_signal:
            signal_class = "type_change_only"
        else:
            raise AssertionError("P01 candidate has neither defining signal")
        rows.append({**transition, "signal_class": signal_class})
    return {
        "n_candidates": len(rows),
        "signal_classes": summarize_groups(rows, "signal_class"),
        "question_continuity": summarize_groups(rows, "question_continuity"),
        "interpretation": (
            "The two rule signals are not equivalent: continuity-only captures explicit new/reopened "
            "questions without a type change, while type-only captures a taxonomic change despite a "
            "non-new continuity label. Report them separately."
        ),
    }


def p02_mechanism_combinations(
    p02_candidates: list[dict[str, Any]], candidates_by_pattern: dict[str, set[str]],
    transitions_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    subtype_ids = ["AQC_P03", "AQC_P04", "AQC_P05", "AQC_P07"]
    rows = []
    for occurrence in p02_candidates:
        unit = occurrence["unit_id"]
        active = [pattern_id for pattern_id in subtype_ids if unit in candidates_by_pattern[pattern_id]]
        signature = "+".join(active) if active else "none_of_P03_P04_P05_P07"
        rows.append({**transitions_by_id[unit], "mechanism_signature": signature})
    with_any = [row for row in rows if row["mechanism_signature"] != "none_of_P03_P04_P05_P07"]
    return {
        "n_p02_candidates": len(rows),
        "n_with_at_least_one_current_mechanism": len(with_any),
        "fraction_with_at_least_one_current_mechanism": rate(len(with_any), len(rows)),
        "n_without_current_mechanism": len(rows) - len(with_any),
        "mutually_exclusive_mechanism_combinations": summarize_groups(rows, "mechanism_signature"),
        "interpretation": (
            "P03/P04/P05/P07 are overlapping explanatory flags nested under the continued-imaging "
            "backbone P02. Their conditional opportunity fractions are not independent effect sizes."
        ),
    }


def p08_decomposition(
    candidates: list[dict[str, Any]], transitions_by_id: dict[str, dict[str, Any]],
    steps_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for occurrence in candidates:
        transition = transitions_by_id[occurrence["unit_id"]]
        source_types, source_status = assumption_profiles(steps_by_id[transition["source_step_id"]])
        target_types, target_status = assumption_profiles(steps_by_id[transition["target_step_id"]])
        type_signal = source_types != target_types
        shared_types = set(source_types) & set(target_types)
        status_signal = any(source_status[key] != target_status[key] for key in shared_types)
        if type_signal and status_signal:
            structural_class = "both_type_multiset_and_shared_type_status_change"
        elif type_signal:
            structural_class = "type_multiset_change_only"
        elif status_signal:
            structural_class = "shared_type_status_change_only"
        else:
            raise AssertionError("P08 candidate cannot be structurally decomposed")
        rows.append({
            **transition,
            "structural_class": structural_class,
            "assumption_count_relation": (
                "increase" if sum(target_types.values()) > sum(source_types.values())
                else "decrease" if sum(target_types.values()) < sum(source_types.values())
                else "same"
            ),
            "declared_assumption_change": transition.get("assumption_change"),
        })
    return {
        "n_candidates": len(rows),
        "structural_classes": summarize_groups(rows, "structural_class"),
        "assumption_count_relation": summarize_groups(rows, "assumption_count_relation"),
        "declared_assumption_change": summarize_groups(rows, "declared_assumption_change"),
        "interpretation": (
            "The current P08 rule is a structural change detector. It cannot establish proposition-level "
            "belief revision because assumptions are not linked across steps by stable proposition IDs."
        ),
    }


def p11_schema_audit(
    opportunities: list[dict[str, Any]], candidates: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
) -> dict[str, Any]:
    relation_by_target = {row["target_step_id"]: row["observed_action_relation"] for row in transitions}
    candidate_ids = {row["unit_id"] for row in candidates}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in opportunities:
        schema_branch = "current_question_grounding" if row.get("question_grounding") is not None else "legacy_intent_support"
        support_value = row.get("question_grounding") or row.get("legacy_intent_support")
        grouped[schema_branch].append({
            **row,
            "support_value": support_value,
            "step_position": "initial" if row["step_index"] == 1 else "noninitial",
            "observed_action_relation": relation_by_target.get(row["unit_id"], "initial"),
            "is_candidate": row["unit_id"] in candidate_ids,
        })
    output = {}
    for branch, rows in sorted(grouped.items()):
        candidate_rows = [row for row in rows if row["is_candidate"]]
        output[branch] = {
            "n_opportunities": len(rows),
            "n_candidates": len(candidate_rows),
            "candidate_fraction": rate(len(candidate_rows), len(rows)),
            "n_candidate_patients": len({row["patient_id"] for row in candidate_rows}),
            "candidate_support_values": dict(sorted(Counter(row["support_value"] for row in candidate_rows).items())),
            "candidate_step_position": summarize_groups(candidate_rows, "step_position"),
            "candidate_observed_action_relation": summarize_groups(candidate_rows, "observed_action_relation"),
        }
    return {
        "schema_branches": output,
        "interpretation": (
            "This remains a schema-specific annotation audit. The two fields have different semantics "
            "and their fractions must not be pooled or treated as a trajectory-transition pattern."
        ),
    }


def render_report(result: dict[str, Any]) -> str:
    p01 = result["decompositions"]["AQC_P01"]
    p02 = result["decompositions"]["AQC_P02_family"]
    p08 = result["decompositions"]["AQC_P08"]
    p11 = result["decompositions"]["AQC_P11"]
    lines = [
        "# Stage-3 A/Q/C pattern structural refinement",
        "",
        "Status: development-only, ACR-blind, exploratory, and not frozen.",
        "",
        "## Question reorientation (P01)", "",
    ]
    for name, stats in p01["signal_classes"].items():
        lines.append(f"- `{name}`: {stats['n_units']} transitions / {stats['n_patients']} patients.")
    lines.extend(["", "## Persistent-open-question family (P02--P07)", ""])
    lines.append(
        f"- P02 contains {p02['n_p02_candidates']} transitions; "
        f"{p02['n_with_at_least_one_current_mechanism']} have at least one P03/P04/P05/P07 flag and "
        f"{p02['n_without_current_mechanism']} have none."
    )
    for name, stats in p02["mutually_exclusive_mechanism_combinations"].items():
        lines.append(f"- `{name}`: {stats['n_units']} transitions / {stats['n_patients']} patients.")
    lines.extend(["", "## Assumption composition (P08)", ""])
    for name, stats in p08["structural_classes"].items():
        lines.append(f"- `{name}`: {stats['n_units']} transitions / {stats['n_patients']} patients.")
    lines.extend(["", "## Order-support audit (P11)", ""])
    for name, stats in p11["schema_branches"].items():
        lines.append(
            f"- `{name}`: {stats['n_candidates']}/{stats['n_opportunities']} candidates "
            f"({stats['candidate_fraction']}); {stats['n_candidate_patients']} patients."
        )
    lines.extend(["", "## Draft disposition before codebook freeze", ""])
    for pattern_id, decision in result["draft_dispositions"].items():
        lines.append(f"- `{pattern_id}`: **{decision['disposition']}** — {decision['basis']}")
    lines.extend([
        "",
        "These dispositions organize the next manual calibration step. They are not frozen definitions,",
        "replication results, or evidence that A/Q/C predicts the next order.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    manifest = read_json(DATA / "manifest.json")
    if manifest["counts"]["patients"] != 235 or manifest["counts"]["final_test_patients_included"] != 0:
        raise AssertionError("Refinement requires the development-only release")
    steps = read_jsonl(DATA / "steps.jsonl")
    transitions = read_jsonl(DATA / "transitions.jsonl")
    opportunities = read_jsonl(DATA / "pattern_opportunities.jsonl")
    occurrences = read_jsonl(DATA / "pattern_occurrences.jsonl")
    targeted_review = read_json(TARGETED_REVIEW)
    steps_by_id = {row["step_id"]: row for row in steps}
    transitions_by_id = {row["transition_id"]: row for row in transitions}
    candidates_by_pattern_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    opportunities_by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in occurrences:
        candidates_by_pattern_rows[row["pattern_id"]].append(row)
    for row in opportunities:
        opportunities_by_pattern[row["pattern_id"]].append(row)
    candidates_by_pattern = {
        pattern_id: {row["unit_id"] for row in rows}
        for pattern_id, rows in candidates_by_pattern_rows.items()
    }
    review_counts = Counter(
        (item["pattern_id"], item["decision"]) for item in targeted_review["decisions"]
    )
    result = {
        "schema_version": "0.1.0-stage-3-structural-refinement",
        "status": "exploratory_not_frozen",
        "acr_used": False,
        "final_test_used": False,
        "decompositions": {
            "AQC_P01": p01_decomposition(candidates_by_pattern_rows["AQC_P01"], transitions_by_id),
            "AQC_P02_family": p02_mechanism_combinations(
                candidates_by_pattern_rows["AQC_P02"], candidates_by_pattern, transitions_by_id
            ),
            "AQC_P08": p08_decomposition(
                candidates_by_pattern_rows["AQC_P08"], transitions_by_id, steps_by_id
            ),
            "AQC_P11": p11_schema_audit(
                opportunities_by_pattern["AQC_P11"], candidates_by_pattern_rows["AQC_P11"], transitions
            ),
        },
        "draft_dispositions": {
            "AQC_P01": {
                "disposition": "retain_and_split_signals",
                "basis": "Retain as a structural descriptor, but report continuity and question-type signals separately.",
            },
            "AQC_P02": {
                "disposition": "retain_as_backbone_only",
                "basis": "It defines continued imaging with an open question; it is a conditional backbone, not a standalone predictive finding.",
            },
            "AQC_P03--AQC_P05": {
                "disposition": "retain_as_overlapping_mechanism_flags",
                "basis": "Use within P02 and report mutually exclusive combinations; do not present their selected-denominator fractions as effect sizes.",
            },
            "AQC_P06": {
                "disposition": "retain_under_P01",
                "basis": "It identifies reorientation after an informative result but cannot distinguish advance from reroute without review.",
            },
            "AQC_P07": {
                "disposition": "retain_with_semantic_guardrail",
                "basis": (
                    f"Complete targeted review retained 13 of 14 and marked one unclear; require an explicit temporal requirement plus the P02 backbone."
                ),
            },
            "AQC_P08": {
                "disposition": "refine_before_freeze",
                "basis": "Separate type-multiset from shared-type status changes; neither is proposition-level belief revision.",
            },
            "AQC_P09": {
                "disposition": "downgrade_to_annotation_audit",
                "basis": (
                    f"The sole material-discordance candidate was excluded as false discordance; no retained development occurrence remains."
                ),
            },
            "AQC_P10": {
                "disposition": "retain_as_rare_boundary_audit",
                "basis": "Complete review retained two audit cases and excluded one annotation inconsistency; this is not a stable empirical rate.",
            },
            "AQC_P11": {
                "disposition": "retain_as_schema_specific_annotation_audit",
                "basis": "It is not a transition pattern and the legacy/current support constructs cannot be pooled.",
            },
        },
        "targeted_review_crosscheck": {
            f"{pattern_id}:{decision}": count
            for (pattern_id, decision), count in sorted(review_counts.items())
        },
        "next_action": (
            "Calibrate P01, P02-family, P06, and the decomposed P08 on a deterministic, disease-diverse "
            "manual sample; then revise and freeze the development codebook before any predictive test."
        ),
    }
    write_json(OUT, result)
    with REPORT.open("w", encoding="utf-8") as handle:
        handle.write(render_report(result))
    print(f"wrote {OUT.relative_to(ROOT)} and {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
