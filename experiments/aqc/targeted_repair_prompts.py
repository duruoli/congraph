"""Small, causally masked prompts for targeted A/Q/C semantic repair."""
from __future__ import annotations

import hashlib
import json
from typing import Any


TEMPORAL_SYSTEM = """You review one local A/Q/C consistency question. Do not redo the annotation.

Decide whether the current clinical question truly asks for change relative to an earlier study or treatment state. A temporal requirement is needed for interval improvement, worsening, progression, stability, or response. It is not needed merely because an order is repeated or the question asks for the current presence, identity, or severity of a finding.

Use only the supplied pre-order clinical record. Never infer from the masked current result. Preserve all unrelated fields. Evidence must be verbatim clinical text; absence of information belongs in reason with an empty evidence list.

Return only the requested JSON object."""


DISCORDANCE_SYSTEM = """You review one local A/Q/C discordance judgment. Do not redo the annotation.

Discordance requires two explicit, clinically important evidence streams that address the same proposition in opposing directions. Different technique or adequacy, nonvisualization followed by visualization, compatible coexisting findings, and a limited study with limited reassurance are not discordance. Use indeterminate only when the two streams plausibly conflict but the conflict cannot be resolved from the visible record.

Use only the supplied pre-order clinical record. Never infer from the masked current result. Evidence streams must be verbatim clinical text. Preserve all unrelated fields.

Return only the requested JSON object."""


TEMPORAL_OUTPUT = {
    "decision": "aligned | add_temporal_requirement | remove_temporal_wording | unclear",
    "reason": "one concise sentence",
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
    "revised_aggregate": "unanswered | partially_answered | sufficiently_answered | null",
    "revised_aggregate_reason": "replacement string or null",
}


DISCORDANCE_OUTPUT = {
    "decision": "aligned | revise | unclear",
    "reason": "one concise sentence",
    "discordance": {
        "label": "concordant | materially_discordant | indeterminate | not_applicable",
        "evidence_stream_1": "verbatim quote or empty",
        "evidence_stream_2": "verbatim quote or empty",
        "reason": "why the streams conflict or do not conflict",
    },
}


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
        "task": (
            "Check only temporal intent/requirement alignment. If aligned, return null for every "
            "revision field. If temporal intent is explicit but the requirement is missing, add one "
            "requirement and its matching coverage entry. If temporal wording is unsupported and "
            "current-state characterization is the actual question, revise only the question wording."
        ),
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
) -> str:
    previous = annotation.get("previous_order_update") or {}
    payload = {
        "task": (
            "Check only whether the two evidence streams conflict on the same clinical proposition. "
            "If the existing judgment is aligned, preserve it. Otherwise return a replacement for "
            "the discordance object only."
        ),
        "clinical_context": {
            "baseline_relevant": _compact_baseline(record),
            "current_order": decision_point.get("ordered"),
            "current_result": "MASKED_AND_NOT_AVAILABLE",
        },
        "newly_visible_imaging_since_previous_decision": newly_visible_imaging,
        "annotation_subset": {
            "previous_question": previous.get("previous_question"),
            "effect_on_previous_question": previous.get("effect_on_previous_question"),
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
    "annotation_subset": ["previous question", "effect on previous question", "discordance"],
    "current_result": "masked",
}


def prompt_hash(system: str, output: dict[str, Any], input_contract: dict[str, Any]) -> str:
    return hashlib.sha256(
        (
            system
            + json.dumps(output, sort_keys=True)
            + json.dumps(input_contract, sort_keys=True)
        ).encode("utf-8")
    ).hexdigest()


TEMPORAL_PROMPT_SHA256 = prompt_hash(TEMPORAL_SYSTEM, TEMPORAL_OUTPUT, TEMPORAL_INPUT_CONTRACT)
DISCORDANCE_PROMPT_SHA256 = prompt_hash(
    DISCORDANCE_SYSTEM, DISCORDANCE_OUTPUT, DISCORDANCE_INPUT_CONTRACT
)
