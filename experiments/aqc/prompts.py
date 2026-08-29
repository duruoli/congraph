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

COMMON_SYSTEM = """You reconstruct a plausible imaging-decision trajectory from causally available evidence. This is empirical, order-aware knowledge discovery, not guideline scoring and not next-test prediction.

At each decision point you see the actual imaging order, but never that order's result or later events. Use the order as a clue to intent only when the visible record supports it. Do not claim unique access to the treating physician's private thoughts. Preserve ambiguity and an unsupported residual instead of forcing every order into a clean rationale.

Hard rules:
- Maintain a coherent trajectory by using the supplied prior A/Q/C state, but revise it when the newly visible prior result warrants revision.
- Represent assumptions as separate atomic propositions. Each proposition has its own type, level, and status; never give one certainty label to the entire step.
- Separate disease existence from etiology, severity/course, complication, alternative source, and intervention/device state.
- Separate study adequacy, test-question capability, result status, and aggregate question coverage.
- A valid negative is informative. It is not the same as indeterminate, nonvisualized, or not assessed.
- Coverage is relative to all evidence and the current question, not a property of one test.
- For every question, enumerate the evidence dimensions required for an answer. These answer
  requirements describe information, not a recommended modality or protocol.
- Record coverage separately for every requirement. Never replace the requirement list with one
  scalar label; keep study adequacy, test-question capability, result status, and aggregate
  coverage distinct.
- Flag material discordance only with two quoted, clinically important evidence streams that the current assumption cannot comfortably explain.
- First describe assumption and question changes. Derive a transition summary only afterward.
- Use close only when the actual current action explicitly represents no further imaging; never infer stop merely because later events are hidden.
- Be concise enough to finish the JSON: include at most 5 assumptions, 5 answer requirements,
  2 evidence quotes per item, and 2 secondary questions. Keep explanations to one short sentence.
- Output strict JSON only."""

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
                "id": f"one of: {' | '.join(ANSWER_REQUIREMENT_TYPES)}",
                "other_proposed_dimension": "required free-text name when id=other; otherwise empty",
                "dimension": "what evidence dimension must be addressed",
                "why_required": "why this dimension is necessary to answer the question",
            }],
            "evidence": ["verbatim quote from visible input"],
        },
        "question_continuity": "initial | same | refined | new | reopened",
        "coverage": {
            "requirements": [{
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
