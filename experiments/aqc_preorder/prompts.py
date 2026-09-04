"""Order-blinded prompts for pre-order A/Q/C extraction."""

from __future__ import annotations

import json
from typing import Any

from experiments.aqc.prompts import (
    ANSWER_REQUIREMENT_TYPES,
    ASSUMPTION_STATUSES,
    ASSUMPTION_TYPES,
    COVERAGE_AGGREGATES,
    COVERAGE_DIRECTIONS,
    COVERAGE_STATUSES,
    QUESTION_TYPES,
)


SYSTEM = """You are a clinical-state annotator. Reconstruct the A/Q/C state using only the
clinical information available at this decision time:
- Assumptions (A): decision-relevant beliefs, hypotheses, and established constraints;
- Question (Q): the most important unresolved imaging-answerable clinical question;
- Coverage (C): how much the visible evidence already answers each requirement of Q.

The current imaging order, its result, and all later events are deliberately hidden. Do not guess
which test was actually ordered and do not mention a modality merely to predict the hidden action.
Infer Q from the visible clinical state, not from knowledge of the eventual order. When several
questions remain plausible, select the most decision-relevant one and record alternatives only as
secondary questions. Use verbatim evidence only from the visible input.

Keep study adequacy, result status, test capability, and question coverage distinct. A technically
adequate test may still be unable to answer a particular question. A negative or indeterminate
result does not automatically close the broader clinical question. Across steps, carry the prior
blinded A/Q/C state forward only when supported by newly visible evidence.

Return only the complete JSON object defined by the output template. Do not add keys."""


def output_contract() -> dict[str, Any]:
    return {
        "assumptions": [{
            "proposition": "one atomic, decision-relevant proposition",
            "type": f"one of: {' | '.join(ASSUMPTION_TYPES)}",
            "other_proposed_type": "required when type=other; otherwise empty",
            "status": f"one of: {' | '.join(ASSUMPTION_STATUSES)}",
            "evidence": ["verbatim quote from visible input"],
            "support": "well_supported | weakly_supported | unclear",
        }],
        "latest_imaging_update": {
            "applicable": "boolean; false when no prior imaging is visible",
            "study_adequacy": "diagnostic | limited_but_diagnostic | nondiagnostic | unknown | not_applicable",
            "result_status": "positive | negative | indeterminate | not_assessed | unknown | not_applicable",
            "effect_on_prior_question": "brief evidence-grounded update or not_applicable",
        },
        "current_question": {
            "primary": "most decision-relevant unresolved imaging-answerable clinical unknown",
            "target": "anatomy, disease, finding, mechanism, complication, or intervention",
            "type": f"one of: {' | '.join(QUESTION_TYPES)}",
            "other_proposed_type": "required when type=other; otherwise empty",
            "positive_answer_changes": "decision or assumption change",
            "negative_answer_changes": "decision or assumption change",
            "secondary_questions": ["at most two plausible secondary questions"],
            "answer_requirements": [{
                "requirement_key": "short unique key within the step",
                "id": f"one of: {' | '.join(ANSWER_REQUIREMENT_TYPES)}",
                "other_proposed_dimension": "required when id=other; otherwise empty",
                "dimension": "concrete information dimension needed to answer Q",
                "why_required": "brief explanation",
            }],
            "evidence": ["verbatim quote from visible input"],
        },
        "question_continuity": "initial | same | refined | new | reopened | unclear",
        "coverage": {
            "requirements": [{
                "requirement_key": "must match one declared answer requirement",
                "requirement_id": f"one of: {' | '.join(ANSWER_REQUIREMENT_TYPES)}",
                "status": f"one of: {' | '.join(COVERAGE_STATUSES)}",
                "direction": f"one of: {' | '.join(COVERAGE_DIRECTIONS)}",
                "supporting_evidence": ["verbatim quote or empty"],
                "reason": "how visible evidence addresses this requirement",
            }],
            "aggregate": f"one of: {' | '.join(COVERAGE_AGGREGATES)}",
            "aggregate_reason": "brief summary",
        },
    }


def build_user(record: dict[str, Any], prior_blinded_aqc: dict[str, Any] | None) -> str:
    baseline = record["baseline"]
    prior_imaging = record.get("visible_prior_imaging") or []
    if prior_imaging:
        prior_text = "\n\n".join(
            f"[{index}] {item['modality']} {item['region']} ({item['exam']})\n{item['report']}"
            for index, item in enumerate(prior_imaging, start=1)
        )
    else:
        prior_text = "(none)"
    return f"""## Visible baseline workup
[History]
{baseline.get('patient_history', '')}

[Physical examination]
{baseline.get('physical_examination', '')}

[Laboratory tests]
{baseline.get('laboratory_tests', '')}

## Imaging resulted before this decision point
{prior_text}

## Prior order-blinded A/Q/C state
{json.dumps(prior_blinded_aqc, ensure_ascii=False) if prior_blinded_aqc else '(none)'}

The current order, current result, and later events are hidden. Describe the clinical A/Q/C state;
do not predict or name the hidden order.

{json.dumps(output_contract(), ensure_ascii=False, indent=2)}"""
