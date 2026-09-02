"""Deterministic per-step validator for A/Q/C annotation output.

This module checks structure and cross-field rules whose correct outcome is
mechanically knowable.  It deliberately does not infer clinical intent:
temporal-question detection and substantive discordance belong to targeted
semantic auditors.
"""
from __future__ import annotations

from typing import Any

from experiments.aqc import prompts

VALIDATOR_VERSION = "3.1.0"

ASSUMPTION_SUPPORT = {"well_supported", "weakly_supported", "unclear"}
QUESTION_CONTINUITIES = {"initial", "same", "refined", "new", "reopened"}
ASSUMPTION_CHANGES = {
    "establish", "retain", "refine", "challenge", "exclude", "replace", "initial"
}
DERIVED_TRANSITIONS = {
    "remedy", "adjudicate", "advance", "reroute", "reopen", "close", "initial", "unclear"
}
QUESTION_GROUNDING = {"well_supported", "weakly_supported", "unclear"}
TEST_CAPABILITY = {"capable", "partially_capable", "not_capable", "uncertain"}
PREVIOUS_STUDY_ADEQUACY = {
    "diagnostic", "limited_but_diagnostic", "nondiagnostic", "unknown", "not_applicable"
}
PREVIOUS_TEST_CAPABILITY = TEST_CAPABILITY | {"not_applicable"}
PREVIOUS_RESULT_STATUS = {
    "positive", "negative", "indeterminate", "not_assessed", "unknown", "not_applicable"
}
DISCORDANCE_LABELS = {
    "concordant", "materially_discordant", "indeterminate", "not_applicable"
}


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_quote_list(
    value: Any,
    *,
    field: str,
    require_one: bool,
    maximum: int = 2,
    validate_items: bool = True,
) -> list[str]:
    if not isinstance(value, list) or (require_one and not value) or len(value or []) > maximum:
        return [f"{field}_bad_count"]
    if validate_items and any(not _nonempty(item) for item in value):
        return [f"{field}_bad_item"]
    return []


def _validate_other_field(
    item: dict[str, Any], *, enum_field: str, other_field: str, prefix: str
) -> list[str]:
    other = item.get(other_field)
    if item.get(enum_field) == "other":
        return [] if _nonempty(other) else [f"{prefix}_missing_other_name"]
    return [] if other in {"", None} else [f"{prefix}_unexpected_other_name"]


def _validate_assumptions(value: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    assumptions = value.get("assumptions")
    if not isinstance(assumptions, list):
        return ["assumptions_not_list"]
    if len(assumptions) > 5:
        errors.append("too_many_assumptions")
    for index, item in enumerate(assumptions):
        prefix = f"assumption_{index}"
        if not isinstance(item, dict):
            errors.append(f"{prefix}_not_object")
            continue
        if item.get("type") not in prompts.ASSUMPTION_TYPES:
            errors.append(f"{prefix}_bad_type")
        if item.get("status") not in prompts.ASSUMPTION_STATUSES:
            errors.append(f"{prefix}_bad_status")
        if item.get("support") not in ASSUMPTION_SUPPORT:
            errors.append(f"{prefix}_bad_support")
        errors.extend(_validate_other_field(
            item, enum_field="type", other_field="other_proposed_type", prefix=prefix
        ))
        errors.extend(_validate_quote_list(
            item.get("evidence"), field=f"{prefix}_evidence", require_one=True
        ))
        if item.get("status") in {"established", "excluded"} and item.get("support") != "well_supported":
            errors.append(f"{prefix}_strong_status_without_strong_support")
    return errors


def _validate_question(
    value: dict[str, Any], ordered: str | None
) -> tuple[list[str], list[tuple[str, str]]]:
    errors: list[str] = []
    question = value.get("current_question")
    if not isinstance(question, dict):
        return ["current_question_not_object"], []
    if question.get("type") not in prompts.QUESTION_TYPES:
        errors.append("bad_question_type")
    errors.extend(_validate_other_field(
        question,
        enum_field="type",
        other_field="other_proposed_type",
        prefix="current_question",
    ))
    requirements = question.get("answer_requirements")
    declared: list[tuple[str, str]] = []
    if not isinstance(requirements, list):
        errors.append("answer_requirements_not_list")
    else:
        if len(requirements) > 5:
            errors.append("too_many_answer_requirements")
        for index, item in enumerate(requirements):
            prefix = f"answer_requirement_{index}"
            if not isinstance(item, dict):
                errors.append(f"{prefix}_not_object")
                continue
            key = item.get("requirement_key")
            requirement_type = item.get("id")
            if not _nonempty(key):
                errors.append(f"{prefix}_bad_key")
            if requirement_type not in prompts.ANSWER_REQUIREMENT_TYPES:
                errors.append(f"{prefix}_bad_type")
            errors.extend(_validate_other_field(
                item,
                enum_field="id",
                other_field="other_proposed_dimension",
                prefix=prefix,
            ))
            if _nonempty(key):
                declared.append((key, requirement_type))
        keys = [key for key, _ in declared]
        if len(keys) != len(set(keys)):
            errors.append("duplicate_answer_requirement_key")
    secondary = question.get("secondary_questions")
    if not isinstance(secondary, list) or len(secondary) > 2:
        errors.append("bad_secondary_questions")
    elif any(not _nonempty(item) for item in secondary):
        errors.append("bad_secondary_question_item")
    question_evidence = question.get("evidence")
    errors.extend(_validate_quote_list(
        question_evidence, field="question_evidence", require_one=True
    ))
    if isinstance(question_evidence, list) and _nonempty(ordered):
        normalized_order = ordered.strip().casefold()
        if any(
            isinstance(item, str) and item.strip().casefold() == normalized_order
            for item in question_evidence
        ):
            errors.append("current_order_used_as_question_evidence")
    return errors, declared


def _validate_coverage(
    value: dict[str, Any], declared: list[tuple[str, str]]
) -> list[str]:
    errors: list[str] = []
    coverage = value.get("coverage")
    if not isinstance(coverage, dict) or not isinstance(coverage.get("requirements"), list):
        return ["coverage_requirements_not_list"]
    covered: list[tuple[str, str]] = []
    for index, item in enumerate(coverage["requirements"]):
        prefix = f"coverage_{index}"
        if not isinstance(item, dict):
            errors.append(f"{prefix}_not_object")
            continue
        key = item.get("requirement_key")
        requirement_type = item.get("requirement_id")
        status = item.get("status")
        direction = item.get("direction")
        evidence = item.get("supporting_evidence")
        if not _nonempty(key):
            errors.append(f"{prefix}_bad_key")
        if requirement_type not in prompts.ANSWER_REQUIREMENT_TYPES:
            errors.append(f"{prefix}_bad_requirement_type")
        if _nonempty(key):
            covered.append((key, requirement_type))
        if status not in prompts.COVERAGE_STATUSES:
            errors.append(f"{prefix}_bad_status")
        if direction not in prompts.COVERAGE_DIRECTIONS:
            errors.append(f"{prefix}_bad_direction")
        errors.extend(_validate_quote_list(
            evidence,
            field=f"{prefix}_evidence",
            require_one=False,
            validate_items=False,
        ))
        if status == "unaddressed" and direction != "no_direction":
            errors.append(f"{prefix}_unaddressed_with_direction")
        if direction in {"supports", "refutes", "mixed"} and isinstance(evidence, list) and not evidence:
            errors.append(f"{prefix}_direction_without_evidence")
        if status == "sufficiently_addressed" and direction == "no_direction":
            errors.append(f"{prefix}_sufficient_without_direction")
    covered_keys = [key for key, _ in covered]
    if len(covered_keys) != len(set(covered_keys)):
        errors.append("duplicate_coverage_requirement_key")
    if covered != declared:
        errors.append("coverage_question_requirement_mismatch")
    aggregate = coverage.get("aggregate")
    if aggregate not in prompts.COVERAGE_AGGREGATES:
        errors.append("bad_coverage_aggregate")
    valid_items = [item for item in coverage["requirements"] if isinstance(item, dict)]
    if aggregate == "sufficiently_answered" and any(
        item.get("status") != "sufficiently_addressed" for item in valid_items
    ):
        errors.append("sufficient_aggregate_with_incomplete_requirement")
    if aggregate == "partially_answered" and valid_items and all(
        item.get("status") == "unaddressed" for item in valid_items
    ):
        errors.append("partial_aggregate_with_all_requirements_unaddressed")
    if aggregate == "unanswered" and any(
        item.get("status") == "sufficiently_addressed" for item in valid_items
    ):
        errors.append("unanswered_aggregate_with_sufficient_requirement")
    return errors


def _validate_previous_order_update(
    value: dict[str, Any], *, is_first_step: bool
) -> list[str]:
    errors: list[str] = []
    previous = value.get("previous_order_update")
    if not isinstance(previous, dict):
        return ["previous_order_update_not_object"]
    if previous.get("study_adequacy") not in PREVIOUS_STUDY_ADEQUACY:
        errors.append("bad_previous_study_adequacy")
    if previous.get("test_question_capability") not in PREVIOUS_TEST_CAPABILITY:
        errors.append("bad_previous_test_question_capability")
    if previous.get("result_status") not in PREVIOUS_RESULT_STATUS:
        errors.append("bad_previous_result_status")
    discordance = previous.get("discordance")
    if not isinstance(discordance, dict):
        errors.append("previous_discordance_not_object")
    else:
        label = discordance.get("label")
        stream_1 = discordance.get("evidence_stream_1")
        stream_2 = discordance.get("evidence_stream_2")
        reason = discordance.get("reason")
        if label not in DISCORDANCE_LABELS:
            errors.append("bad_discordance_label")
        if not isinstance(stream_1, str) or not isinstance(stream_2, str):
            errors.append("bad_discordance_evidence_stream_type")
        if not isinstance(reason, str):
            errors.append("bad_discordance_reason_type")
        if label in {"materially_discordant", "indeterminate"}:
            if not _nonempty(stream_1) or not _nonempty(stream_2):
                errors.append("discordance_label_without_two_evidence_streams")
            if not _nonempty(reason):
                errors.append("discordance_label_without_reason")
        if label == "not_applicable" and (
            _nonempty(stream_1) or _nonempty(stream_2)
        ):
            errors.append("not_applicable_discordance_with_evidence")
    if is_first_step:
        if previous.get("applicable") is not False:
            errors.append("first_step_previous_order_update_applicable")
        if previous.get("study_adequacy") != "not_applicable":
            errors.append("first_step_previous_study_adequacy_not_applicable")
        if previous.get("test_question_capability") != "not_applicable":
            errors.append("first_step_previous_capability_not_applicable")
        if previous.get("result_status") != "not_applicable":
            errors.append("first_step_previous_result_not_applicable")
        if isinstance(discordance, dict) and discordance.get("label") != "not_applicable":
            errors.append("first_step_discordance_not_applicable")
    elif previous.get("applicable") is not True:
        errors.append("later_step_previous_order_update_not_applicable")
    return errors


def _validate_sequence_state(value: dict[str, Any], *, is_first_step: bool) -> list[str]:
    errors: list[str] = []
    continuity = value.get("question_continuity")
    change = value.get("assumption_change")
    transition = value.get("derived_transition")
    if continuity not in QUESTION_CONTINUITIES:
        errors.append("bad_question_continuity")
    if not isinstance(change, dict) or change.get("label") not in ASSUMPTION_CHANGES:
        errors.append("bad_assumption_change")
    if transition not in DERIVED_TRANSITIONS:
        errors.append("bad_derived_transition")
    if is_first_step:
        if continuity != "initial":
            errors.append("first_step_question_continuity_not_initial")
        if not isinstance(change, dict) or change.get("label") != "initial":
            errors.append("first_step_assumption_change_not_initial")
        if transition != "initial":
            errors.append("first_step_derived_transition_not_initial")
    else:
        if continuity == "initial":
            errors.append("later_step_question_continuity_initial")
        if isinstance(change, dict) and change.get("label") == "initial":
            errors.append("later_step_assumption_change_initial")
        if transition == "initial":
            errors.append("later_step_derived_transition_initial")
    return errors


def _validate_current_order_fit(value: dict[str, Any]) -> list[str]:
    fit = value.get("current_order_fit")
    if not isinstance(fit, dict):
        return ["current_order_fit_not_object"]
    errors: list[str] = []
    if set(fit) != {"question_grounding", "test_question_capability"}:
        errors.append("bad_current_order_fit_fields")
    if fit.get("question_grounding") not in QUESTION_GROUNDING:
        errors.append("bad_question_grounding")
    if fit.get("test_question_capability") not in TEST_CAPABILITY:
        errors.append("bad_test_question_capability")
    return errors


def validate_output(
    value: Any,
    *,
    is_first_step: bool = False,
    ordered: str | None = None,
    is_repeat_order: bool = False,
) -> list[str]:
    """Return deterministic validation errors for one generated A/Q/C step.

    ``is_repeat_order`` is retained for API/provenance compatibility.  Repeat
    status alone is intentionally not a hard temporal rule: the prompt says a
    temporal requirement is needed only when the clinical question asks for
    change relative to an earlier study.
    """
    del is_repeat_order
    if not isinstance(value, dict):
        return ["not_a_json_object"]
    errors: list[str] = []
    errors.extend(_validate_assumptions(value))
    question_errors, declared = _validate_question(value, ordered)
    errors.extend(question_errors)
    errors.extend(_validate_coverage(value, declared))
    errors.extend(_validate_previous_order_update(value, is_first_step=is_first_step))
    errors.extend(_validate_sequence_state(value, is_first_step=is_first_step))
    errors.extend(_validate_current_order_fit(value))
    if "current_test" in value:
        errors.append("transitional_current_test_present")
    return errors
