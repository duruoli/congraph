"""Stage-1 blinded prompt for open patient-Context extraction."""

from __future__ import annotations

import json
from typing import Any


FACTUAL_DIMENSIONS = [
    "symptoms",
    "signs_and_labs",
    "patient_characteristics",
    "disease_timing",
    "prior_test",
    "encounter_stage",
    "imaging_stage",
]
INFERENTIAL_DIMENSIONS = [
    "diagnosis",
    "severity_or_complication",
    "evidence_interpretation",
]
ALL_DIMENSIONS = FACTUAL_DIMENSIONS + INFERENTIAL_DIMENSIONS

EPISTEMIC_SOURCES = [
    "directly_documented_fact",
    "deterministic_derivation",
    "documented_clinical_judgment",
    "reconstructed_judgment",
]


SYSTEM = """You are annotating the open patient Context at one imaging decision step. Use only the
causally available record before that decision: history, examination, laboratory data, and reports
of prior imaging whose results were already available.

This is a blinded extraction pass. You are not given the current imaging order, its result, later
events, A/Q/C annotations, ACR variants, an ACR vocabulary, or action ratings. Do not guess any of
them. Preserve the patient's and chart's native wording; do not translate an item into guideline
terminology.

Context conditions have two epistemic kinds.

FACTUAL CONDITIONS are directly checkable from the visible record or deterministically derivable:
- symptoms: patient-reported manifestations, including explicit absence or persistence;
- signs_and_labs: examination findings, vital signs, laboratory states, explicit negatives, and
  documented changes over time;
- patient_characteristics: patient attributes that materially delimit the clinical scenario, not
  an indiscriminate demographic or problem-list dump;
- disease_timing: position relative to documented symptom onset or disease course;
- prior_test: relevant imaging completed before this decision; create one item per study and retain
  modality, region, protocol, and time in the native value when visible;
- encounter_stage: the current episode's position in the visit or presentation sequence;
- imaging_stage: the decision's position in the imaging sequence. Infer only what visible
  trajectory metadata supports; do not invent repeat, switch, or post-intervention intent.

INFERENTIAL CONDITIONS require a rule or clinical interpretation:
- diagnosis: a suspected, established, challenged, excluded, or unknown disease or etiologic frame;
- severity_or_complication: a rule-derived or clinically synthesized assessment of severity,
  deterioration, systemic response, or complication;
- evidence_interpretation: what symptoms, labs, or prior imaging mean, including a reported imaging
  finding or limitation, atypicality, uncertainty, confounding, and competing explanations. Link an
  interpretation of a prior study to its prior_test item.

Annotation rules:
1. Extract sparsely: include only conditions supported by the visible record and relevant to the
   active clinical workup. An empty dimension means not documented or unknown, never absent.
2. Separate facts from judgments. For example, record abnormal vital signs as facts and a global
   deterioration assessment as a separate inference only when supported. Record a prior study as a
   fact and its reported or reconstructed meaning as a linked evidence_interpretation.
3. Preserve explicit negation and uncertainty. Missing evidence is not a negative predicate.
4. Each item must be atomic when possible. If the source contains AND/OR logic that cannot be split
   without changing its meaning, preserve it in value_native and explain the logic briefly.
5. Every item needs exact verbatim evidence spans. A deterministic derivation or reconstructed
   judgment must cite the input facts and explain the reproducible rule or synthesis. Do not call a
   judgment documented unless the visible chart states it.
6. Use reconstructed_judgment sparingly. Do not infer a diagnosis merely from symptoms or from the
   fact that this is an imaging decision. When several interpretations remain possible, preserve
   the uncertainty or abstain.
7. If a relevant condition does not fit any of the ten dimensions, record it under
   additional_dimension_outside_acr_schema and propose a concise dimension name. Do not decide
   whether an extracted value has an ACR equivalent; that occurs in a later mapping pass.
8. If a relevant judgment appears necessary but cannot be recovered from the visible record,
   describe the gap under latent_or_unidentifiable. Do not invent a Context value for it.

Return only the complete JSON object defined by the supplied output template. Do not add keys."""


def _evidence_span_contract() -> dict[str, Any]:
    return {
        "section": "history | physical_examination | laboratory_tests | prior_imaging",
        "prior_imaging_index": "1-based integer when section=prior_imaging; otherwise null",
        "quote": "exact verbatim span from the visible input",
    }


def _item_contract() -> dict[str, Any]:
    return {
        "item_id": "unique ID within this decision step, e.g. ctx_01",
        "value_native": "one concise patient-native Context condition",
        "assertion_status": (
            "affirmed | negated | suspected | established | challenged | excluded | equivocal | unclear"
        ),
        "temporality": "current | historical | trajectory | unclear",
        "epistemic_source": f"one of: {' | '.join(EPISTEMIC_SOURCES)}",
        "evidence_spans": [_evidence_span_contract()],
        "based_on_item_ids": [
            "IDs of extracted factual items used by a derivation or judgment; otherwise empty"
        ],
        "applies_to_item_ids": [
            "IDs of items this condition interprets or modifies, especially prior_test; otherwise empty"
        ],
        "logic_note": "atomic, or a brief account of preserved AND/OR/threshold logic",
        "reasoning": (
            "brief rule or synthesis for deterministic/reconstructed items; empty for directly stated items"
        ),
    }


def output_contract() -> dict[str, Any]:
    """Machine-readable output template supplied with every extraction request."""
    return {
        "schema_version": "1.0.0-open-patient-context",
        "factual_context": {dimension: [_item_contract()] for dimension in FACTUAL_DIMENSIONS},
        "inferential_context": {dimension: [_item_contract()] for dimension in INFERENTIAL_DIMENSIONS},
        "additional_dimension_outside_acr_schema": [{
            "item_id": "unique ID within this decision step",
            "proposed_dimension": "concise intuitive name not duplicating an existing dimension",
            "epistemic_kind": "factual | inferential | unclear",
            "value_native": "relevant patient-native condition",
            "why_outside": "why none of the ten dimensions can represent this condition",
            "evidence_spans": [_evidence_span_contract()],
        }],
        "latent_or_unidentifiable": [{
            "description": "relevant Context judgment or operation that the visible record cannot recover",
            "related_dimension": f"one of: {' | '.join(ALL_DIMENSIONS)} | outside_schema | unclear",
            "why_unidentifiable": "specific missing evidence or undocumented judgment",
            "relevant_evidence_spans": [_evidence_span_contract()],
        }],
        "extraction_note": "brief note about material ambiguity or empty string",
    }


def _prior_imaging_text(prior: list[dict[str, Any]]) -> str:
    if not prior:
        return "(none)"
    return "\n\n".join(
        f"[Prior imaging {index}: {item.get('modality', '')} {item.get('region', '')} "
        f"({item.get('exam', '')})]\n{item.get('report', '')}"
        for index, item in enumerate(prior, start=1)
    )


def build_user(
    baseline: dict[str, Any],
    visible_prior_imaging: list[dict[str, Any]],
) -> str:
    """Build one order-blinded open patient-Context extraction request."""
    return f"""## Visible pre-order record

[History]
{baseline.get('patient_history', '')}

[Physical examination]
{baseline.get('physical_examination', '')}

[Laboratory tests]
{baseline.get('laboratory_tests', '')}

## Imaging resulted before this decision point
{_prior_imaging_text(visible_prior_imaging)}

The current imaging order, its result, later events, A/Q/C, and all ACR content are hidden.
Extract the open patient Context from only the visible material above.

Fill every field in the JSON output template below. Replace descriptive placeholder strings with
case-specific values, retain empty arrays for unsupported dimensions, and do not add keys.
{json.dumps(output_contract(), ensure_ascii=False, indent=2)}"""
