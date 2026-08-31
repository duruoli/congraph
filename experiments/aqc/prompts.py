"""Paired, sequential prompts for the Track-B A/Q/C pilot.

Trajectory-level coherence is implemented by carrying the preceding A/Q/C state
forward one decision point at a time.  A single prompt containing every masked
view would leak an early order's result through a later view, so it is explicitly
not supported here.
"""
from __future__ import annotations

import json
from typing import Any


ASSUMPTION_TYPES = [
    "syndrome_or_source_frame",
    "disease_or_finding_identity",
    "etiology_or_mechanism",
    "severity_extent_or_course",
    "complication",
    "alternative_source",
    "intervention_or_device_state",
    "other",
    "unclear",
]
ASSUMPTION_STATUSES = ["suspected", "likely", "established", "challenged", "excluded", "unclear"]
QUESTION_TYPES = [
    "source_localization",
    "existence_or_identity",
    "etiology_or_mechanism",
    "severity_or_extent",
    "complication",
    "alternative_source",
    "intervention_or_device_state",
    "other",
    "unclear",
]

ANSWER_REQUIREMENT_TYPES = [
    "target_visualization_or_assessment",
    "presence_or_absence",
    "anatomic_localization",
    "finding_identity",
    "etiologic_agent_or_mechanism",
    "severity_or_extent",
    "temporal_course_or_response",
    "complication_presence_or_character",
    "alternative_source_discrimination",
    "device_position_or_integrity",
    "device_or_intervention_function",
    "other",
    "unclear",
]

COVERAGE_STATUSES = ["unaddressed", "partially_addressed", "sufficiently_addressed"]
COVERAGE_DIRECTIONS = ["supports", "refutes", "mixed", "no_direction"]
COVERAGE_AGGREGATES = ["unanswered", "partially_answered", "sufficiently_answered"]

COMMON_SYSTEM = """Reconstruct the A/Q/C state that best explains one observed imaging decision.

Input boundary: use only the visible pre-order chart, resulted prior imaging, the carried prior A/Q/C state, and the actual current order. The current result and later events are absent. The order identifies a decision to explain; chart evidence determines how strongly its inferred intent is supported.

Annotation rules:
1. Assumptions are atomic propositions. Never combine an observed or established syndrome/fact with a still-speculative cause, mechanism, or complication; split claims with different certainty into separate propositions. Use the minimum decision-relevant set: usually 1-3 uncertain hypotheses plus only the established facts needed to interpret them, with 5 propositions maximum in total. Give each proposition its own type, level, status, exact evidence quote, and support. `established` and `excluded` require explicit or adequately decisive evidence; nonvisualization and limited/nondiagnostic assessment are not exclusion.
2. The primary question is the central decision-relevant unknown, not the examination name. Evidence fidelity takes priority over specificity: use the most specific question supported by the visible chart, but keep it at a broader shared level when the exact target or trigger is unclear, lower `intent_support`, and preserve the ambiguity in `unsupported_residual`; never infer a specific target from the exam name alone. Question types describe what is being asked and may overlap semantically; choose the type most central to this order. `alternative_source_discrimination` is an answer-requirement type; a primary competing-source question uses question type `alternative_source`.
3. Declare only information dimensions necessary to answer the question (usually 2-4; maximum 5). Each requirement has a unique `requirement_key` and a concrete target/object. A type may repeat for different objects. `temporal_course_or_response` is a requirement, not a question type; use it only when comparison, progression, stability, or response is part of the stated question.
4. Coverage describes all evidence available before the current order. Provide exactly one coverage entry per declared requirement, matched by `requirement_key` and type. Keep study adequacy, test-question capability, result status, and aggregate coverage separate. An adequately assessed negative can sufficiently cover a requirement with direction `refutes`; nonvisualized, indeterminate, or not-assessed targets generally cannot.
5. Carry the trajectory forward and revise it only when newly available prior evidence supports a change. Record material discordance only when two exact, clinically important evidence streams conflict. Derive the transition after the assumption and question updates. `close` requires an explicit no-further-imaging action.
6. Preserve uncertainty. Put any portion of the observed order that the evidence does not explain in `unsupported_residual`; do not increase certainty merely to make the order appear rational.
7. When the current exam repeats the preceding exam or its target, explicitly consider change since the prior study and any new trigger across A/Q/C; fold this temporal logic into the existing question and requirements when relevant rather than mechanically adding a separate requirement.

Return strict JSON. Use at most two evidence quotes per item, at most two secondary questions, and one short sentence for each explanation."""

DIRECT_SYSTEM = COMMON_SYSTEM + """

This is the DIRECT arm. Infer A/Q/C from the masked chart plus the actual order. You do not see the old schema-free reconstruction."""

RECODE_SYSTEM = COMMON_SYSTEM + """

This is the RECODE arm. Recode the supplied old schema-free, ex-ante reconstruction into A/Q/C. Treat its wording as fallible source material: preserve what it supports, mark over-rationalized or internally unsupported claims as weak, and do not add facts from outside it."""


def output_contract() -> dict[str, Any]:
    """Machine-readable shape requested from both paired pilot arms."""
    return {
        "assumptions": [{
            "proposition": "one atomic proposition",
            "type": f"one of: {' | '.join(ASSUMPTION_TYPES)}",
            "other_proposed_type": "required free-text name when type=other; otherwise empty",
            "level": "clinical hierarchy level in plain language",
            "status": f"one of: {' | '.join(ASSUMPTION_STATUSES)}",
            "evidence": ["verbatim quote from visible input"],
            "support": "well_supported | weakly_supported | unclear",
        }],
        "previous_order_update": {
            "applicable": "boolean; false at the first decision order",
            "previous_question": "question text or null",
            "study_adequacy": "diagnostic | limited_but_diagnostic | nondiagnostic | unknown | not_applicable",
            "test_question_capability": "capable | partially_capable | not_capable | uncertain | not_applicable",
            "result_status": "positive | negative | indeterminate | not_assessed | unknown | not_applicable",
            "effect_on_previous_question": "how all newly available evidence changed coverage",
            "discordance": {
                "label": "concordant | materially_discordant | indeterminate | not_applicable",
                "evidence_stream_1": "quote or empty",
                "evidence_stream_2": "quote or empty",
                "reason": "brief explanation",
            },
        },
        "assumption_change": {
            "label": "establish | retain | refine | challenge | exclude | replace | initial",
            "changed_proposition": "specific proposition or null",
            "reason": "brief evidence-grounded explanation",
        },
        "current_question": {
            "primary": "decision-relevant unknown; do not restate the modality",
            "target": "anatomy, disease, finding, mechanism, complication, or intervention",
            "type": f"one of: {' | '.join(QUESTION_TYPES)}",
            "other_proposed_type": "required free-text name when type=other; otherwise empty",
            "positive_answer_changes": "decision or assumption change",
            "negative_answer_changes": "decision or assumption change",
            "secondary_questions": ["optional question"],
            "answer_requirements": [{
                "requirement_key": "short unique key within this step, e.g. req_1",
                "id": f"one of: {' | '.join(ANSWER_REQUIREMENT_TYPES)}",
                "other_proposed_dimension": "required free-text name when id=other; otherwise empty",
                "dimension": "concrete target/object and information dimension that must be addressed",
                "why_required": "why this dimension is necessary to answer the question",
            }],
            "evidence": ["verbatim quote from visible input"],
        },
        "question_continuity": "initial | same | refined | new | reopened",
        "coverage": {
            "requirements": [{
                "requirement_key": "must exactly match one declared answer_requirements requirement_key",
                "requirement_id": f"one of: {' | '.join(ANSWER_REQUIREMENT_TYPES)}",
                "status": f"one of: {' | '.join(COVERAGE_STATUSES)}",
                "direction": f"one of: {' | '.join(COVERAGE_DIRECTIONS)}",
                "supporting_evidence": ["verbatim quote or empty"],
                "reason": "how all pre-order evidence addresses this requirement",
            }],
            "aggregate": f"one of: {' | '.join(COVERAGE_AGGREGATES)}",
            "aggregate_reason": "optional summary; never a replacement for the entries above",
        },
        "current_order_fit": {
            "test_question_capability": "capable | partially_capable | not_capable | uncertain",
            "why_this_order_could_answer": "brief explanation",
            "intent_support": "well_supported | weakly_supported | unclear",
            "unsupported_residual": "what remains unexplained; empty only if genuinely supported",
        },
        "derived_transition": "remedy | adjudicate | advance | reroute | reopen | close | initial | unclear",
        "alternative_reconstruction": "second plausible A/Q/C account or empty",
    }


def _prior_imaging_text(prior: list[dict[str, Any]]) -> str:
    if not prior:
        return "(none)"
    return "\n\n".join(
        f"[Prior imaging {index}: {item.get('modality', '')} {item.get('region', '')} "
        f"({item.get('exam', '')})]\n{item.get('report', '')}"
        for index, item in enumerate(prior, start=1)
    )


def build_direct_user(
    decision_point: dict[str, Any],
    baseline: dict[str, Any],
    prior_aqc: dict[str, Any] | None,
) -> str:
    """Build one causally masked sequential DIRECT-arm request."""
    return f"""## Baseline workup
[History]
{baseline.get('patient_history', '')}

[Physical examination]
{baseline.get('physical_examination', '')}

[Laboratory tests]
{baseline.get('laboratory_tests', '')}

## Imaging resulted before this decision point
{_prior_imaging_text(decision_point.get('visible_prior_imaging', []))}

## Prior A/Q/C state carried from the preceding decision point
{json.dumps(prior_aqc, ensure_ascii=False) if prior_aqc else '(none; this is the first decision order)'}

## Actual current order to explain
{decision_point.get('ordered', '')}

The current result and every later event are hidden. Return exactly this JSON contract:
{json.dumps(output_contract(), ensure_ascii=False, indent=2)}"""


def build_recode_user(
    old_ex_ante: dict[str, Any],
    ordered: str,
    prior_aqc: dict[str, Any] | None,
) -> str:
    """Build one sequential RECODE-arm request without current result/verification."""
    allowed = {
        key: old_ex_ante.get(key)
        for key in (
            "differential",
            "other_hypothesis",
            "information_gap",
            "expected_finding",
            "action_role",
            "appropriateness",
            "appropriateness_reason",
            "grounding",
            "reasoning",
        )
    }
    return f"""## Old schema-free ex-ante reconstruction
{json.dumps(allowed, ensure_ascii=False, indent=2)}

## Prior A/Q/C state carried from the preceding decision point
{json.dumps(prior_aqc, ensure_ascii=False) if prior_aqc else '(none; this is the first decision order)'}

## Actual current order represented by that reconstruction
{ordered}

No current result, verification label, or later event is available. Return exactly this JSON contract:
{json.dumps(output_contract(), ensure_ascii=False, indent=2)}"""
