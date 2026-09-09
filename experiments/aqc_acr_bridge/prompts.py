"""Blinded prompt for the text half of hybrid patient-Context extraction."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from experiments.aqc_acr_bridge.algorithmic_predicates import mask_labelled_vitals


LLM_DIMENSIONS = [
    "patient_attribute",
    "symptom_state",
    "sign_state",
    "lab_finding_state",
    "imaging_finding_state",
    "diagnostic_state",
    "test_history",
    "test_interpretation",
    "aggregate_assessment",
    "temporal_position",
    "diagnosis_presentation_stage",
]

SYSTEM = """Extract clinically relevant patient conditions from the supplied record into these
dimensions:

- patient_attribute(attribute, state): a patient characteristic that affects the clinical context.
- symptom_state(symptom, site, state): a patient-reported symptom, its site, and its state.
- sign_state(sign, state): an examination finding or vital-sign state.
- lab_finding_state(analyte, state): a laboratory analyte or laboratory-derived state.
- imaging_finding_state(finding, site, state): an anatomical or pathological imaging finding.
- diagnostic_state(condition, status, role): a disease or complication and its diagnostic status
  and role.
- test_history(test, status): a test mentioned as completed, incomplete, or planned.
- test_interpretation(test, target, result): what a test showed with respect to a diagnostic target.
- aggregate_assessment(assessment, target, state): an integrated clinical assessment or score.
- temporal_position(event, relation, value, anchor): an event's position relative to a clinical
  time anchor.
- diagnosis_presentation_stage(diagnosis, stage): whether this is the first, initial, or recurrent
  presentation of a diagnosis.

Rules:
1. Use concise normalized English values in snake_case and retain exact source wording as evidence.
2. Preserve explicit negation and uncertainty. Information that is not documented is unknown, not
   absent.
3. If a relevant condition fits none of these dimensions, extract it under
   other_proposed_dimension with a concise dimension name and definition.
4. Keep the extraction sparse and decision-relevant. Extract at most 30 distinct items total,
   focusing on conditions material to the patient's abdominal presentation and imaging context.
   Do not inventory incidental normal anatomy, duplicate the same condition across synonymous
   items, or extract every sentence merely because it is documented.
5. Each evidence support must be one verbatim, contiguous substring copied from its named visible
   source. Preserve its spelling, punctuation, and whitespace exactly. Never join separate spans,
   add ellipses, paraphrase, or repair redacted text. Prefer one evidence span per item; add another
   only when it is necessary to support that same item.
6. Return only the complete JSON object matching the supplied template. Keep an empty array when a
   dimension has no supported item and do not add keys."""


def _evidence_contract() -> dict[str, str]:
    return {
        "source": (
            "history | physical_examination | prior_imaging_N"
        ),
        "support": "exact quote from the visible source",
    }


def _with_evidence(arguments: Mapping[str, str]) -> dict[str, Any]:
    return {**arguments, "evidence": [_evidence_contract()]}


def output_contract() -> dict[str, Any]:
    """Return the deliberately small, predicate-specific LLM output template."""
    return {
        "schema_version": "3.0.0-hybrid-text-context",
        "patient_context": {
            "patient_attribute": [_with_evidence({
                "attribute": "normalized patient attribute",
                "state": "current documented state",
            })],
            "symptom_state": [_with_evidence({
                "symptom": "normalized symptom",
                "site": "normalized anatomical site or unspecified",
                "state": "present | absent | persistent | improving | worsening | equivocal",
            })],
            "sign_state": [_with_evidence({
                "sign": "normalized sign",
                "state": "documented state",
            })],
            "lab_finding_state": [_with_evidence({
                "analyte": "normalized laboratory analyte or derived finding",
                "state": "documented result or state",
            })],
            "imaging_finding_state": [_with_evidence({
                "finding": "normalized prior-imaging finding",
                "site": "normalized anatomical site",
                "state": "present | absent | equivocal | limited",
            })],
            "diagnostic_state": [_with_evidence({
                "condition": "normalized diagnosis or complication",
                "status": "suspected | established | challenged | excluded | equivocal",
                "role": "primary_diagnosis | complication | alternative_diagnosis",
            })],
            "test_history": [_with_evidence({
                "test": "normalized test",
                "status": "completed | incomplete | planned",
            })],
            "test_interpretation": [_with_evidence({
                "test": "normalized completed test",
                "target": "normalized diagnostic target",
                "result": "positive | negative | equivocal | limited",
            })],
            "aggregate_assessment": [_with_evidence({
                "assessment": "normalized explicitly documented assessment or score",
                "target": "condition assessed or unspecified",
                "state": "documented assessment state",
            })],
            "temporal_position": [_with_evidence({
                "event": "clinical event",
                "relation": "before | after | since | within",
                "value": "documented duration or time value",
                "anchor": "named clinical anchor",
            })],
            "diagnosis_presentation_stage": [_with_evidence({
                "diagnosis": "normalized diagnosis",
                "stage": "first_presentation | initial_presentation | recurrent_presentation",
            })],
        },
        "other_proposed_dimension": [{
            "dimension": "concise proposed dimension name",
            "definition": "one concise extraction question",
            "value": "one concise patient condition",
            "evidence": [_evidence_contract()],
        }],
    }


def _prior_imaging_text(prior: Sequence[Mapping[str, Any]]) -> str:
    if not prior:
        return "(none)"
    return "\n\n".join(
        f"[Prior imaging {index}: {item.get('modality', '')} {item.get('region', '')} "
        f"({item.get('exam', '')})]\n"
        f"{item.get('report', '')}"
        for index, item in enumerate(prior, start=1)
    )


def build_user(
    baseline: Mapping[str, Any],
    visible_prior_imaging: Sequence[Mapping[str, Any]],
) -> str:
    """Build one order-blinded text-complement extraction request."""
    return f"""## Visible pre-order record

[History]
{baseline.get('patient_history', '')}

[Physical examination]
{mask_labelled_vitals(baseline.get('physical_examination', ''))}

[Prior resulted imaging]
{_prior_imaging_text(visible_prior_imaging)}

Return the extracted dimensions using this JSON template:
{json.dumps(output_contract(), ensure_ascii=False, indent=2)}"""
