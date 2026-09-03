"""Small, causally masked prompts for targeted A/Q/C semantic repair."""
from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any


TEMPORAL_SYSTEM = """You audit whether interval comparison is an independent answer dimension in one local A/Q/C annotation. Do not redo the annotation.

Return `interval_comparison_required` only when answering the clinical question requires comparing the current state with an identifiable earlier study or treatment state and reporting improvement, worsening, progression, stability, or response. A current finding alone must be insufficient to answer that part of the question. When this is true, temporal course needs its own requirement and coverage entry even if comparison language also appears inside a severity, complication, or presence requirement, because prior evidence may cover the current-state dimension without covering interval change.

Return `current_state_question_only` when the actual information need is current presence, identity, anatomy, severity, or complication status. Repeated ordering, availability of prior imaging, or adjectives such as new, evolving, persistent, or worsening do not by themselves establish an independent interval-comparison need. If such wording creates a comparison that is not independently necessary or supported, rewrite only the question wording while preserving the current-state information need.

Return `unclear` only when the supplied pre-order record cannot distinguish those two cases. Do not use a generic aligned/consistent decision.

Use only the supplied pre-order clinical record. Never infer from the masked current result. Preserve all unrelated fields. Evidence must be verbatim clinical text; absence of information belongs in reason with an empty evidence list.

Return only the requested JSON object."""


DISCORDANCE_SYSTEM = """You audit one existing A/Q/C discordance flag for a false positive. Do not redo the annotation or search for missed discordance.

A proposition is one atomic, decision-relevant clinical claim that can be true or false at a specified level and time, such as disease presence, etiology, complication, anatomy, severity, or interval change. Prefer a proposition already represented in the carried-forward assumptions or the previous question; do not invent a broad topic after seeing the evidence.

True discordance requires two explicit, credible, clinically important evidence streams that bear on the same proposition in opposing directions, are comparable in target and time, cannot be comfortably reconciled, and could change the next question or action. A new result that merely challenges or excludes a working hypothesis is a normal assumption update, not discordance, unless another explicit evidence stream still supports that same proposition. Different technique or adequacy, nonvisualization followed by visualization, compatible coexisting findings, timing or test-sensitivity differences, and limited reassurance are not discordance.

Use only the supplied pre-order clinical record. Never infer from the masked current result. Return true_discordance only when every requirement above is met, false_discordance when any requirement clearly fails, and unclear only when the visible record cannot resolve the issue.

Return only the requested JSON object."""


TEMPORAL_OUTPUT = {
    "decision": "interval_comparison_required | current_state_question_only | unclear",
    "reason": "one concise sentence explaining whether comparison with a baseline is necessary to answer Q",
    "revised_primary": "replacement string or null",
    "revised_secondary_questions": "replacement list or null",
    "temporal_requirement": {
        "requirement_key": "new unique key",
        "id": "temporal_course_or_response",
        "other_proposed_dimension": "",
        "dimension": "specific comparison target and reference study/state",
        "why_required": "why interval comparison is necessary",
    },
    "temporal_coverage": {
        "requirement_key": "same new key",
        "requirement_id": "temporal_course_or_response",
        "status": "unaddressed | partially_addressed | sufficiently_addressed",
        "direction": "supports | refutes | mixed | no_direction",
        "supporting_evidence": ["zero to two verbatim quotes"],
        "reason": "how pre-order evidence covers the interval comparison",
    },
}


DISCORDANCE_OUTPUT = {
    "decision": "true_discordance | false_discordance | unclear",
    "proposition": "one atomic clinical claim, or null if no shared proposition exists",
    "reason": "one concise sentence applying the discordance criteria",
}


TEMPORAL_TASK = (
    "Decide whether interval comparison is independently required to answer the question. "
    "If required and the typed requirement is missing, supply exactly one temporal requirement "
    "and its matching coverage entry. If current-state characterization is the actual question, "
    "remove unnecessary comparison language from the primary/secondary wording; an existing "
    "temporal requirement and coverage entry will be removed deterministically. If a required "
    "temporal requirement already exists but Q does not state the comparison, revise Q to make "
    "the comparison explicit. Return exactly the six requested fields; use null for every "
    "revision or temporal object that the chosen decision does not require. Do not assess or "
    "revise aggregate coverage; code derives any necessary aggregate change after applying the "
    "requirement-level repair."
)


DISCORDANCE_TASK = (
    "Test only whether the existing positive or indeterminate discordance flag is genuine. "
    "Name the single proposition at issue, then decide whether both quoted evidence streams "
    "truly oppose each other on that proposition and the conflict remains clinically material "
    "after ordinary explanations are considered."
)


def _compact_baseline(record: dict[str, Any]) -> dict[str, Any]:
    baseline = record["baseline"]
    return {
        "patient_history": baseline.get("patient_history"),
        "laboratory_tests": baseline.get("laboratory_tests"),
    }


def build_temporal_user(
    record: dict[str, Any], decision_point: dict[str, Any], annotation: dict[str, Any]
) -> str:
    payload = {
        "task": TEMPORAL_TASK,
        "clinical_context": {
            "baseline_relevant": _compact_baseline(record),
            "resulted_prior_imaging": decision_point.get("visible_prior_imaging") or [],
            "current_order": decision_point.get("ordered"),
            "current_result": "MASKED_AND_NOT_AVAILABLE",
        },
        "annotation_subset": {
            "current_question": annotation.get("current_question"),
            "question_continuity": annotation.get("question_continuity"),
            "coverage": annotation.get("coverage"),
        },
        "output": TEMPORAL_OUTPUT,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_discordance_user(
    record: dict[str, Any],
    decision_point: dict[str, Any],
    annotation: dict[str, Any],
    newly_visible_imaging: list[dict[str, Any]],
    carried_forward_assumptions: list[dict[str, Any]] | None = None,
) -> str:
    previous = annotation.get("previous_order_update") or {}
    payload = {
        "task": DISCORDANCE_TASK,
        "clinical_context": {
            "baseline_relevant": _compact_baseline(record),
            "current_order": decision_point.get("ordered"),
            "current_result": "MASKED_AND_NOT_AVAILABLE",
        },
        "newly_visible_imaging_since_previous_decision": newly_visible_imaging,
        "annotation_subset": {
            "carried_forward_assumptions": carried_forward_assumptions or [],
            "updated_assumptions": annotation.get("assumptions"),
            "previous_question": previous.get("previous_question"),
            "effect_on_previous_question": previous.get("effect_on_previous_question"),
            "assumption_change": annotation.get("assumption_change"),
            "current_question": annotation.get("current_question"),
            "discordance": previous.get("discordance"),
        },
        "output": DISCORDANCE_OUTPUT,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


TEMPORAL_INPUT_CONTRACT = {
    "clinical_context": ["filtered patient history", "labs", "resulted prior imaging", "current order"],
    "annotation_subset": ["current question", "question continuity", "coverage"],
    "current_result": "masked",
}

DISCORDANCE_INPUT_CONTRACT = {
    "clinical_context": ["filtered patient history", "labs", "current order"],
    "new_evidence": "newly visible imaging since previous decision",
    "annotation_subset": [
        "carried-forward assumptions",
        "updated assumptions",
        "previous question",
        "effect on previous question",
        "assumption change",
        "current question",
        "discordance",
    ],
    "current_result": "masked",
}


def prompt_hash(
    system: str,
    task: str,
    output: dict[str, Any],
    input_contract: dict[str, Any],
    builder: Any,
) -> str:
    return hashlib.sha256(
        (
            system
            + task
            + json.dumps(output, sort_keys=True)
            + json.dumps(input_contract, sort_keys=True)
            + inspect.getsource(builder)
        ).encode("utf-8")
    ).hexdigest()


TEMPORAL_PROMPT_SHA256 = prompt_hash(
    TEMPORAL_SYSTEM,
    TEMPORAL_TASK,
    TEMPORAL_OUTPUT,
    TEMPORAL_INPUT_CONTRACT,
    build_temporal_user,
)
DISCORDANCE_PROMPT_SHA256 = prompt_hash(
    DISCORDANCE_SYSTEM,
    DISCORDANCE_TASK,
    DISCORDANCE_OUTPUT,
    DISCORDANCE_INPUT_CONTRACT,
    build_discordance_user,
)
