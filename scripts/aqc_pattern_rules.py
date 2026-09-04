"""Single executable source for the ACR-blind empirical A/Q/C pattern rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


Rule = Callable[[dict[str, Any], "PatternContext"], tuple[bool, bool]]


@dataclass(frozen=True)
class PatternContext:
    step_by_id: dict[str, dict[str, Any]]
    requirement_types_by_step: dict[str, set[str]]


def present(row: dict[str, Any], fields: tuple[str, ...]) -> bool:
    return all(row.get(field) not in {None, ""} for field in fields)


def assumption_signature(step: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    assumptions = step["effective_annotation"].get("assumptions") or []
    return tuple(sorted((str(item.get("type")), str(item.get("status"))) for item in assumptions))


def p01(row: dict[str, Any], _: PatternContext) -> tuple[bool, bool]:
    opportunity = present(row, ("source_question_type", "target_question_type", "question_continuity"))
    match = row.get("question_continuity") in {"new", "reopened"} or row.get("source_question_type") != row.get("target_question_type")
    return opportunity, opportunity and match


def p02(row: dict[str, Any], _: PatternContext) -> tuple[bool, bool]:
    opportunity = present(row, ("target_coverage_aggregate", "question_continuity"))
    match = row.get("target_coverage_aggregate") in {"unanswered", "partially_answered"} and row.get("question_continuity") in {"same", "refined", "reopened"}
    return opportunity, opportunity and match


def p03(row: dict[str, Any], _: PatternContext) -> tuple[bool, bool]:
    opportunity = row.get("previous_study_adequacy") in {"limited_but_diagnostic", "nondiagnostic"}
    match = row.get("target_coverage_aggregate") in {"unanswered", "partially_answered"} and row.get("question_continuity") in {"same", "refined", "reopened"}
    return opportunity, opportunity and match


def p04(row: dict[str, Any], _: PatternContext) -> tuple[bool, bool]:
    opportunity = row.get("previous_test_question_capability") in {"partially_capable", "not_capable", "uncertain"}
    match = row.get("target_coverage_aggregate") in {"unanswered", "partially_answered"} and row.get("question_continuity") in {"same", "refined", "reopened"}
    return opportunity, opportunity and match


def p05(row: dict[str, Any], _: PatternContext) -> tuple[bool, bool]:
    opportunity = row.get("previous_result_status") in {"indeterminate", "not_assessed"}
    match = row.get("target_coverage_aggregate") in {"unanswered", "partially_answered"} and row.get("question_continuity") in {"same", "refined", "reopened"}
    return opportunity, opportunity and match


def p06(row: dict[str, Any], _: PatternContext) -> tuple[bool, bool]:
    opportunity = row.get("previous_result_status") in {"positive", "negative"} and row.get("previous_study_adequacy") in {"diagnostic", "limited_but_diagnostic"}
    reoriented = row.get("question_continuity") == "new" or row.get("source_question_type") != row.get("target_question_type")
    return opportunity, opportunity and reoriented


def p07(row: dict[str, Any], context: PatternContext) -> tuple[bool, bool]:
    types = context.requirement_types_by_step.get(row["target_step_id"], set())
    opportunity = bool(types)
    return opportunity, opportunity and "temporal_course_or_response" in types


def p08(row: dict[str, Any], context: PatternContext) -> tuple[bool, bool]:
    source_signature = assumption_signature(context.step_by_id[row["source_step_id"]])
    target_signature = assumption_signature(context.step_by_id[row["target_step_id"]])
    opportunity = bool(source_signature) and bool(target_signature)
    return opportunity, opportunity and source_signature != target_signature


def p09(row: dict[str, Any], _: PatternContext) -> tuple[bool, bool]:
    opportunity = row.get("discordance") in {"concordant", "materially_discordant", "indeterminate", "not_applicable"}
    return opportunity, opportunity and row.get("discordance") == "materially_discordant"


def p10(row: dict[str, Any], _: PatternContext) -> tuple[bool, bool]:
    opportunity = int(row["step_index"]) > 1 and present(row, ("coverage_aggregate", "ordered"))
    return opportunity, opportunity and row.get("coverage_aggregate") == "sufficiently_answered"


def p11(row: dict[str, Any], _: PatternContext) -> tuple[bool, bool]:
    new_value = row.get("question_grounding")
    legacy_value = row.get("legacy_intent_support")
    opportunity = (new_value is not None) ^ (legacy_value is not None)
    match = new_value in {"weakly_supported", "unclear"} or legacy_value in {"weakly_supported", "unclear"}
    return opportunity, opportunity and match


TRANSITION_RULES: dict[str, Rule] = {
    "AQC_P01": p01,
    "AQC_P02": p02,
    "AQC_P03": p03,
    "AQC_P04": p04,
    "AQC_P05": p05,
    "AQC_P06": p06,
    "AQC_P07": p07,
    "AQC_P08": p08,
    "AQC_P09": p09,
}

STEP_RULES: dict[str, Rule] = {"AQC_P10": p10, "AQC_P11": p11}
RULES: dict[str, Rule] = {**TRANSITION_RULES, **STEP_RULES}


def make_context(
    steps: list[dict[str, Any]], requirements: list[dict[str, Any]]
) -> PatternContext:
    step_by_id = {row["step_id"]: row for row in steps}
    requirement_types_by_step: dict[str, set[str]] = {}
    for row in requirements:
        requirement_types_by_step.setdefault(row["step_id"], set()).add(row["requirement_type"])
    return PatternContext(step_by_id, requirement_types_by_step)


def strict_eligible(row: dict[str, Any]) -> bool:
    return not row.get("has_unclear_value") and not row.get("has_weak_assumption_support")
