"""Sequential prompts for pre-order A/Q/C clinical reasoning annotation."""
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

ANNOTATION_SYSTEM = """You are a clinical reasoning annotator. Given the information available before an imaging order and the order itself, reconstruct the reasoning that most plausibly led to that decision:
- Assumptions (A): what the clinician believed, suspected, or considered;
- Question (Q): the main uncertainty the clinician wanted the test to resolve;
- Coverage (C): how much the available evidence had already answered that question.
Also judge whether the ordered test could answer Q.

Use the record and order together. The order is important evidence of the clinician's intended question, but it does not prove that the suspected condition was present. Do not ignore the order, and do not force the record to support it. If the order suggests a more specific Q than the record supports, retain that Q, mark `question_grounding` as `weakly_supported` or `unclear`, and give a materially different record-based account in `alternative_reconstruction`.

Evidence rules:
- Use only the visible pre-order history, examination, laboratory tests, and resulted prior imaging for clinical claims. Use the prior A/Q/C annotation only to maintain continuity.
- In `evidence`, `supporting_evidence`, and `evidence_stream_*`, copy only verbatim text from the visible clinical record. Put summaries, interpretations, and conclusions based on missing information in explanation or `reason` fields. Never quote the current order as clinical evidence.

Annotation rules:
1. Assumptions must be atomic and decision-relevant. Separate observed facts from uncertain diagnoses, causes, mechanisms, and complications, and separate claims with different certainty. Include only the facts needed to interpret the uncertain hypotheses; use at most five assumptions. `established` and `excluded` require decisive evidence. A limited, nondiagnostic, or nonvisualizing study does not establish exclusion.
2. State Q as the clinical unknown, not as the name of the ordered test. Declare two to four answer requirements when possible, at most five. Each requirement needs a unique key and a concrete target. `temporal_course_or_response` is an answer requirement, not a question type. Use `alternative_source` for the question type and `alternative_source_discrimination` for its requirement.
3. For Coverage C, assess all pre-order evidence against every answer requirement. Provide exactly one matching coverage entry for each requirement. Keep study adequacy, ability of a test to answer Q, result status, and coverage separate. An adequate negative result may address a requirement with direction `refutes`; an unassessed, indeterminate, or nonvisualized target usually cannot.
4. Across multiple orders, carry the previous A/Q/C state forward and change it only when new evidence supports a change. Mark discordance only when two explicit, clinically important evidence streams conflict. Use `close` only when no further imaging is explicitly intended.
5. For a repeated order, identify what is being reassessed and why it is repeated now. Without a documented trigger, timing rationale, or decision need, use weak/unclear `question_grounding`. If Q asks whether the condition has changed since an earlier study, include `temporal_course_or_response`. Do not add this requirement merely because an order is repeated.
6. In `current_order_fit`, record two relations: whether the visible record supports Q as the clinician's question (`question_grounding`), and whether the ordered test can answer Q (`test_question_capability`).

Return only the complete JSON object defined in the supplied output template. Use at most two evidence quotes per item, at most two secondary questions, and one short sentence per explanation."""


def output_contract() -> dict[str, Any]:
    """Machine-readable output template supplied with every annotation request."""
    return {
        "assumptions": [{
            "proposition": "one atomic proposition",
            "type": f"one of: {' | '.join(ASSUMPTION_TYPES)}",
            "other_proposed_type": "required free-text name when type=other; otherwise empty",
            "level": "clinical hierarchy level in plain language",
            "status": f"one of: {' | '.join(ASSUMPTION_STATUSES)}",
            "evidence": ["verbatim quote from the pre-order chart or resulted prior imaging; never quote the current order"],
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
            "question_grounding": "visible-record support that this was the clinician's question: well_supported | weakly_supported | unclear",
            "test_question_capability": "capable | partially_capable | not_capable | uncertain",
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


def build_annotation_user(
    decision_point: dict[str, Any],
    baseline: dict[str, Any],
    prior_aqc: dict[str, Any] | None,
) -> str:
    """Build one sequential pre-order clinical reasoning annotation request."""
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

## Current imaging order
{decision_point.get('ordered', '')}

The current result and every later event are hidden.

Fill every field in the JSON output template below. Its strings describe the expected content or allowed values; replace them with case-specific annotations. Do not add keys.
{json.dumps(output_contract(), ensure_ascii=False, indent=2)}"""
