"""Adjudicate the 12 manual patient Contexts against all 17 ACR Variants.

Direct patient-to-ACR links are carried forward mechanically.  The explicitly
hand-reviewed decisions below resolve judgment-dependent links, temporal
comparisons missed by the item mapper, candidate ranking, and case-level
correspondence.  No A/Q/C, current order/result, action rating, or outcome is
loaded by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ACR_AUDIT = ROOT / "data" / "aqc_acr_bridge" / "acr_variant_predicate_audit_v1.json"
DEFAULT_INPUT = (
    ROOT / "results" / "aqc_acr_bridge" / "pilot_v1" / "manual_acr_mapping_v1"
    / "outputs"
)
DEFAULT_OUTPUT = (
    ROOT / "results" / "aqc_acr_bridge" / "pilot_v1"
    / "manual_variant_adjudication_v1"
)
SCHEMA_VERSION = "1.0.0-manual-variant-adjudication"
PREDICATE_STATUSES = {"supported", "contradicted", "unknown"}
CORRESPONDENCE_LABELS = {"exact", "partial", "multiple", "uncertain", "out_of_scope"}
DIRECT_SUPPORT = {"exact_or_equivalent", "patient_value_narrower"}
JUDGMENT_RELATIONS = {"patient_value_broader", "related_judgment_required"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# These are the few condition judgments that cannot be obtained by applying
# the documented relation semantics directly.  In particular, explicit time
# values can support/contradict several ACR thresholds even when the upstream
# item mapper emitted only the nearest link.
MANUAL_CONDITION_JUDGMENTS: dict[tuple[str, str], dict[str, Any]] = {
    (
        "appendicitis:20123918:s2", "acr_126_v2:c06"
    ): {
        "status": "unknown",
        "item_refs": [
            "patient_context.patient_attribute[1]",
            "patient_context.diagnostic_state[0]",
        ],
        "rationale": (
            "Crohn disease is supported as medical history, but the quoted evidence does not "
            "establish that it is a current alternative to acute pancreatitis."
        ),
    },
    (
        "appendicitis:20123918:s3", "acr_126_v2:c06"
    ): {
        "status": "unknown",
        "item_refs": [
            "patient_context.patient_attribute[1]",
            "patient_context.diagnostic_state[0]",
        ],
        "rationale": (
            "Crohn disease is supported as medical history, but the quoted evidence does not "
            "establish that it is a current alternative to acute pancreatitis."
        ),
    },
    (
        "cholecystitis:21948836:s2", "acr_132_v2:c02"
    ): {
        "status": "supported",
        "item_refs": [
            "patient_context.diagnostic_state[2]",
            "patient_context.imaging_finding_state[1]",
            "patient_context.imaging_finding_state[4]",
        ],
        "rationale": (
            "Suspected cholangitis plus documented biliary-tree dilation/narrowing supports "
            "suspected biliary disease as the cause under consideration for RUQ pain."
        ),
    },
    (
        "cholecystitis:21948836:s2", "acr_132_v3:c04"
    ): {
        "status": "supported",
        "item_refs": [
            "patient_context.diagnostic_state[2]",
            "patient_context.imaging_finding_state[1]",
            "patient_context.imaging_finding_state[4]",
        ],
        "rationale": (
            "Suspected cholangitis and abnormal biliary-tree findings support the broader "
            "suspected-biliary-disease predicate."
        ),
    },
    (
        "cholecystitis:21948836:s2", "acr_132_v4:c04"
    ): {
        "status": "supported",
        "item_refs": [
            "patient_context.diagnostic_state[2]",
            "patient_context.imaging_finding_state[1]",
            "patient_context.imaging_finding_state[4]",
        ],
        "rationale": (
            "Suspected cholangitis and abnormal biliary-tree findings support the broader "
            "suspected-biliary-disease predicate."
        ),
    },
    (
        "pancreatitis:20001800:s2", "acr_126_v1:c06"
    ): {
        "status": "supported",
        "item_refs": ["patient_context.temporal_position[0]"],
        "rationale": (
            "One day from epigastric-pain onset to the presentation decision is within the "
            "ACR less-than-48-to-72-hour window."
        ),
    },
    (
        "pancreatitis:20720063:s3", "acr_126_v1:c06"
    ): {
        "status": "supported",
        "item_refs": ["patient_context.temporal_position[0]"],
        "rationale": (
            "Onset around 5 pm the prior evening and a morning presentation place the decision "
            "within 48 to 72 hours."
        ),
    },
    (
        "pancreatitis:20720063:s3", "acr_126_v3:c08"
    ): {
        "status": "contradicted",
        "item_refs": ["patient_context.temporal_position[0]"],
        "rationale": "Overnight symptom duration is not greater than 48 to 72 hours.",
    },
    (
        "pancreatitis:20720063:s3", "acr_126_v4:c06"
    ): {
        "status": "contradicted",
        "item_refs": ["patient_context.temporal_position[0]"],
        "rationale": "Overnight symptom duration is shorter than the stated 7-to-21-day position.",
    },
    (
        "pancreatitis:20720063:s3", "acr_126_v6:c08"
    ): {
        "status": "contradicted",
        "item_refs": ["patient_context.temporal_position[0]"],
        "rationale": "Overnight symptom duration is shorter than four weeks.",
    },
}


CASE_REVIEWS: dict[str, dict[str, Any]] = {
    "appendicitis:20123918:s2": {
        "label": "multiple",
        "candidates": [
            ("acr_21_v3", "partial", "Pregnancy and RLQ pain support the specific pregnant-patient frame; fever, leukocytosis, and suspected appendicitis remain unknown."),
            ("acr_21_v1", "exact", "The one-predicate generic RLQ-pain signature is fully supported."),
            ("acr_21_v2", "partial", "RLQ pain is supported, but the fever/WBC/appendicitis conjunction is unknown."),
        ],
        "rationale": "The generic RLQ Variant is exactly instantiated while the pregnancy-specific Variant is materially but incompletely instantiated.",
        "bridge_operations": [
            {
                "operation": "resolve_generic_specific_variant_overlap",
                "item_refs": ["patient_context.patient_attribute[0]", "patient_context.symptom_state[0]"],
                "rationale": "Pregnancy makes Variant 3 clinically salient, but missing fever/WBC/diagnostic evidence prevents silent promotion over the generic Variant.",
            },
            {
                "operation": "reject_unsupported_diagnostic_role",
                "item_refs": ["patient_context.patient_attribute[1]", "patient_context.diagnostic_state[0]"],
                "rationale": "Crohn disease is documented as history; the cited evidence does not independently establish a current alternative-diagnosis role.",
            },
        ],
    },
    "appendicitis:20123918:s3": {
        "label": "multiple",
        "candidates": [
            ("acr_21_v3", "partial", "Pregnancy and RLQ pain are supported; the negative RLQ-inflammation report does not establish or exclude suspected appendicitis."),
            ("acr_21_v1", "exact", "The generic RLQ-pain signature is fully supported."),
            ("acr_21_v2", "partial", "RLQ pain is supported while fever, leukocytosis, and suspected appendicitis remain unknown."),
        ],
        "rationale": "The generic and pregnancy-specific frames overlap, and the intervening ultrasound does not resolve the appendix question.",
        "bridge_operations": [
            {
                "operation": "prior_test_question_capability_assessment",
                "item_refs": ["patient_context.imaging_finding_state[2]", "patient_context.test_interpretation[1]"],
                "rationale": "No RLQ inflammatory focus is not equivalent to documented appendix visualization or a definitive negative appendix examination.",
            },
            {
                "operation": "reject_unsupported_diagnostic_role",
                "item_refs": ["patient_context.patient_attribute[1]", "patient_context.diagnostic_state[0]"],
                "rationale": "Crohn disease is documented as history, not explicitly as the active alternative diagnosis.",
            },
        ],
    },
    "appendicitis:20276429:s2": {
        "label": "multiple",
        "candidates": [
            ("acr_21_v2", "partial", "RLQ pain and established appendicitis support two required predicates; fever is subjective and WBC is unknown."),
            ("acr_21_v1", "exact", "The generic RLQ-pain signature is fully supported."),
        ],
        "rationale": "A generic exact match overlaps a strongly supported but incomplete suspected-appendicitis Variant.",
        "bridge_operations": [
            {
                "operation": "prior_test_answered_active_question",
                "item_refs": ["patient_context.test_interpretation[0]", "patient_context.diagnostic_state[0]"],
                "rationale": "The prior ultrasound is positive and establishes appendicitis, so a later imaging decision may concern a new question not represented by the compiled initial-imaging predicates.",
            }
        ],
    },
    "appendicitis:20689999:s2": {
        "label": "exact",
        "candidates": [
            ("acr_21_v1", "exact", "Worsening RLQ pain entails the sole required predicate."),
            ("acr_21_v2", "partial", "RLQ pain is supported, but reported fever is not an objective sign and WBC/appendicitis status are unknown."),
        ],
        "rationale": "The generic RLQ-pain Variant is fully supported; the more specific appendicitis Variant lacks several required predicates.",
        "bridge_operations": [
            {
                "operation": "prior_test_question_capability_assessment",
                "item_refs": ["patient_context.test_interpretation[0]", "patient_context.imaging_finding_state[0]"],
                "rationale": "A pelvic ultrasound limited for right-ovarian pathology does not answer an appendix-focused question.",
            }
        ],
    },
    "cholecystitis:20334898:s2": {
        "label": "multiple",
        "candidates": [
            ("acr_132_v2", "exact", "RUQ-predominant pain and equivocal acute cholecystitis support suspected biliary disease."),
            ("acr_21_v1", "exact", "The same pain item explicitly includes RLQ pain and therefore satisfies the generic RLQ-pain signature, although RUQ pain predominates."),
            ("acr_132_v4", "partial", "RUQ pain, high WBC, and suspected biliary disease are supported; fever and a qualifying prior ultrasound/result are unknown."),
        ],
        "rationale": "RUQ Variant 2 and generic RLQ Variant 1 are both instantiated by the multi-site pain record; RUQ predominance and biliary evidence favor Variant 2 but do not erase the cross-topic match.",
        "bridge_operations": [
            {
                "operation": "resolve_competing_pain_topics",
                "item_refs": ["patient_context.symptom_state[0]", "patient_context.diagnostic_state[0]"],
                "rationale": "The patient has both RUQ and RLQ pain; distribution and suspected biliary disease are needed to prioritize the RUQ topic over the generic RLQ frame.",
            },
            {
                "operation": "patient_feasibility_constraint_application",
                "item_refs": ["patient_context.patient_attribute[0]", "other_proposed_dimension[0]"],
                "rationale": "Severe COPD and active respiratory support may constrain feasible imaging despite not changing the ACR Context predicate match.",
            }
        ],
    },
    "cholecystitis:20660601:s2": {
        "label": "out_of_scope",
        "candidates": [],
        "rationale": "Established acute cholecystitis is present without abdominal/RUQ pain, whereas every selected RUQ Variant requires RUQ pain; no other selected topic supplies a compatible complete frame.",
        "bridge_operations": [
            {
                "operation": "acr_context_missing_painless_disease_presentation",
                "item_refs": ["patient_context.symptom_state[2]", "patient_context.diagnostic_state[0]"],
                "rationale": "The disease is imaging-established but the topic's shared pain predicate is explicitly absent.",
            }
        ],
    },
    "cholecystitis:21948836:s2": {
        "label": "multiple",
        "candidates": [
            ("acr_132_v2", "exact", "RUQ pain and suspected biliary disease are supported."),
            ("acr_132_v3", "partial", "RUQ pain, no fever, biliary suspicion, and prior ultrasound are supported; WBC and an acute-cholecystitis-targeted ultrasound interpretation remain unknown."),
        ],
        "rationale": "The initial suspected-biliary frame is exact, while the prior-ultrasound sequence makes Variant 3 materially plausible but incomplete.",
        "bridge_operations": [
            {
                "operation": "test_target_translation",
                "item_refs": ["patient_context.test_interpretation[0]", "patient_context.diagnostic_state[2]"],
                "rationale": "An equivocal ultrasound for obstruction/stone cannot silently become a negative-or-equivocal ultrasound for acute cholecystitis.",
            },
            {
                "operation": "resolve_initial_vs_sequential_variant_overlap",
                "item_refs": ["patient_context.test_history[0]", "patient_context.test_interpretation[0]"],
                "rationale": "Prior ultrasound makes a sequential frame plausible, but its diagnostic target and remaining required evidence do not fully identify Variant 3.",
            },
        ],
    },
    "diverticulitis:20180280:s2": {
        "label": "partial",
        "candidates": [
            ("acr_20_v3", "partial", "Established complicated diverticulitis supports the complication predicate, but LLQ-localized pain is not documented."),
            ("acr_20_v2", "partial", "Established diverticulitis supports suspected diverticulitis, but LLQ-localized pain is not documented."),
        ],
        "rationale": "The diagnosis and complications are strong, but all selected LLQ Variants require a location-specific pain predicate absent from the record.",
        "bridge_operations": [
            {
                "operation": "disease_to_symptom_topic_translation",
                "item_refs": ["patient_context.symptom_state[0]", "patient_context.diagnostic_state[0]"],
                "rationale": "Established complicated diverticulitis with diffuse pain does not entail the ACR topic's LLQ-localized symptom requirement.",
            }
        ],
    },
    "diverticulitis:21292285:s2": {
        "label": "uncertain",
        "candidates": [
            ("acr_20_v2", "uncertain", "Diverticulitis is equivocal and sigmoid inflammation is incompletely assessed; LLQ pain is not documented."),
            ("acr_20_v3", "uncertain", "Peritonitis may represent a complication, but attribution to diverticulitis is not established and LLQ pain is absent from the documented localization."),
        ],
        "rationale": "The record supports a severe acute abdominal process but does not resolve diverticulitis as its source or instantiate the required LLQ pain.",
        "bridge_operations": [
            {
                "operation": "resolve_complication_source_under_limited_imaging",
                "item_refs": ["patient_context.diagnostic_state[0]", "patient_context.diagnostic_state[1]", "patient_context.test_interpretation[0]"],
                "rationale": "Peritonitis is suspected, diverticulitis is equivocal, and noncontrast CT is limited for identifying the inflammatory source.",
            },
            {
                "operation": "patient_constraint_capability_tradeoff",
                "item_refs": ["patient_context.patient_attribute[2]", "other_proposed_dimension[0]"],
                "rationale": "Renal failure prevents contrast use and leaves the source question incompletely answered.",
            },
        ],
    },
    "pancreatitis:20001800:s2": {
        "label": "partial",
        "candidates": [
            ("acr_126_v1", "partial", "Suspected pancreatitis, characteristic epigastric-to-back pain, and early timing are supported; first presentation and both enzyme predicates are unknown."),
            ("acr_126_v2", "partial", "Suspected pancreatitis is supported, but first-presentation and atypical-presentation aggregates are unknown."),
        ],
        "rationale": "Variant 1 is the closest frame but its conjunctive amylase/lipase and first-presentation requirements are not documented.",
        "bridge_operations": [
            {
                "operation": "resolve_incomplete_conjunctive_laboratory_evidence",
                "item_refs": ["patient_context.diagnostic_state[0]", "patient_context.symptom_state[0]"],
                "rationale": "Clinical suspicion and characteristic pain cannot substitute for ACR's required increased-amylase AND increased-lipase predicates.",
            }
        ],
    },
    "pancreatitis:20720063:s3": {
        "label": "partial",
        "candidates": [
            ("acr_126_v1", "partial", "Established pancreatitis entails suspicion; epigastric pain and early timing are supported, while first presentation and enzyme predicates are unknown."),
            ("acr_126_v2", "partial", "Pancreatitis is supported, but first-presentation and atypical-presentation aggregates remain unknown."),
        ],
        "rationale": "Early timing rules out the later established-pancreatitis temporal frames; Variant 1 remains incomplete because laboratory and presentation-stage predicates are absent.",
        "bridge_operations": [
            {
                "operation": "separate_inflammatory_fluid_from_known_collection",
                "item_refs": ["patient_context.imaging_finding_state[0]"],
                "rationale": "Trace peripancreatic inflammatory fluid does not automatically establish the known organized collection required by Variant 6.",
            }
        ],
    },
    "pancreatitis:25133113:s6": {
        "label": "partial",
        "candidates": [
            ("acr_126_v3", "partial", "Established pancreatitis and timing beyond 48–72 hours are supported; critical illness is not explicitly established by a matching aggregate or severity score."),
            ("acr_126_v5", "uncertain", "Necrosis is not assessable on noncontrast imaging and significant abrupt deterioration is not established."),
            ("acr_126_v2", "partial", "First-presentation pancreatitis is supported, but marked lipase elevation contradicts the illustrative equivocal-lipase feature and the atypical aggregate remains unknown."),
        ],
        "rationale": "Variant 3 is closest, but critical illness cannot be inferred from severe imaging burden alone; contrast limitation leaves Variant 5 unresolved rather than supported.",
        "bridge_operations": [
            {
                "operation": "technical_limitation_to_diagnostic_uncertainty",
                "item_refs": ["patient_context.imaging_finding_state[2]", "patient_context.test_interpretation[1]", "other_proposed_dimension[1]"],
                "rationale": "Noncontrast imaging cannot assess necrosis, so unknown necrosis must not become absence or known necrotizing pancreatitis.",
            },
            {
                "operation": "distinguish_imaging_burden_from_validated_severity_state",
                "item_refs": ["patient_context.aggregate_assessment[0]"],
                "rationale": "Severe inflammatory burden is not equivalent to critical illness, SIRS, or a severe APACHE-II/BISAP/Marshall score.",
            },
        ],
    },
}


def mapping_index(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    items = {row["item_ref"]: row for row in document["mappings"]}
    by_instance: dict[str, list[dict[str, Any]]] = {}
    for row in document["mappings"]:
        for link in row["links"]:
            instance_id = link["acr_predicate_instance_id"]
            if instance_id is None:
                continue
            by_instance.setdefault(instance_id, []).append({
                "item_ref": row["item_ref"],
                "dimension": row["dimension"],
                "patient_item": row["patient_item"],
                "mapping_relation": link["relation"],
                "mapping_rationale": link["rationale"],
            })
    return items, by_instance


def evidence_for_refs(items: dict[str, Any], refs: list[str]) -> list[dict[str, Any]]:
    evidence = []
    for item_ref in refs:
        if item_ref not in items:
            raise ValueError(f"unknown patient item ref in manual judgment: {item_ref}")
        row = items[item_ref]
        evidence.append({
            "item_ref": item_ref,
            "dimension": row["dimension"],
            "patient_item": row["patient_item"],
            "mapping_relation": "adjudication_only",
            "mapping_rationale": "Added during manual Variant adjudication.",
        })
    return evidence


def adjudicate_condition(
    step_id: str,
    instance_id: str,
    items: dict[str, Any],
    linked_evidence: list[dict[str, Any]],
) -> tuple[str, str, str, list[dict[str, Any]]]:
    manual = MANUAL_CONDITION_JUDGMENTS.get((step_id, instance_id))
    if manual is not None:
        evidence = list(linked_evidence)
        linked_refs = {row["item_ref"] for row in evidence}
        evidence.extend(
            row for row in evidence_for_refs(items, manual["item_refs"])
            if row["item_ref"] not in linked_refs
        )
        return manual["status"], "manual_evidence_review", manual["rationale"], evidence

    supporting = [row for row in linked_evidence if row["mapping_relation"] in DIRECT_SUPPORT]
    contradicting = [row for row in linked_evidence if row["mapping_relation"] == "contradicted"]
    if supporting and contradicting:
        return (
            "unknown",
            "manual_conflict_preserved",
            "Patient items provide both direct support and contradiction at potentially different scope; the conflict is not silently resolved.",
            linked_evidence,
        )
    if supporting:
        return (
            "supported",
            "direct_mapping_relation",
            "At least one exact/equivalent or compatible patient-value-narrower link directly entails the ACR predicate.",
            linked_evidence,
        )
    if contradicting:
        return (
            "contradicted",
            "direct_mapping_relation",
            "At least one patient item directly negates the ACR predicate at compatible scope.",
            linked_evidence,
        )
    if any(row["mapping_relation"] in JUDGMENT_RELATIONS for row in linked_evidence):
        return (
            "unknown",
            "manual_relation_review",
            "The judgment-dependent or broader link was reviewed but does not provide entailment or direct negation without an unsupported additional inference.",
            linked_evidence,
        )
    return (
        "unknown",
        "no_comparable_patient_evidence",
        "No mapped patient item establishes or directly negates this predicate; missing evidence remains unknown.",
        [],
    )


def group_status(member_statuses: list[str], operator: str) -> str:
    if operator == "ONE_OR_MORE_OF":
        if "supported" in member_statuses:
            return "supported"
        if member_statuses and all(status == "contradicted" for status in member_statuses):
            return "contradicted"
        return "unknown"
    raise ValueError(f"unsupported computed group operator: {operator}")


def variant_evaluation(variant: dict[str, Any], condition_rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = {row["condition_id"]: row["status"] for row in condition_rows}
    required_units = [row["condition_id"] for row in condition_rows if row["role"] == "required"]
    computed_groups = []
    for logic in variant.get("logic", []):
        if logic.get("id") and logic["operator"] == "ONE_OR_MORE_OF":
            status = group_status([statuses[member] for member in logic["members"]], logic["operator"])
            computed_groups.append({
                "group_id": logic["id"],
                "operator": logic["operator"],
                "members": logic["members"],
                "status": status,
            })
            required_units.append(logic["id"])
            statuses[logic["id"]] = status
    required_statuses = [statuses[unit] for unit in required_units]
    if "contradicted" in required_statuses:
        overall = "contradicted"
    elif required_statuses and all(status == "supported" for status in required_statuses):
        overall = "satisfied"
    elif "supported" in required_statuses:
        overall = "partial"
    else:
        overall = "unknown"
    return {
        "required_units": required_units,
        "computed_groups": computed_groups,
        "n_required_supported": sum(statuses[unit] == "supported" for unit in required_units),
        "n_required_contradicted": sum(statuses[unit] == "contradicted" for unit in required_units),
        "n_required_unknown": sum(statuses[unit] == "unknown" for unit in required_units),
        "overall_signature_status": overall,
    }


def adjudicate_file(path: Path, acr: dict[str, Any]) -> dict[str, Any]:
    mapping = read_json(path)
    step_id = mapping["step_id"]
    if step_id not in CASE_REVIEWS:
        raise ValueError(f"missing case review: {step_id}")
    items, by_instance = mapping_index(mapping)
    variants = []
    for variant in acr["variants"]:
        conditions = []
        for condition in variant["conditions"]:
            instance_id = f"{variant['variant_key']}:{condition['id']}"
            status, method, rationale, evidence = adjudicate_condition(
                step_id, instance_id, items, by_instance.get(instance_id, [])
            )
            conditions.append({
                "acr_predicate_instance_id": instance_id,
                "condition_id": condition["id"],
                "surface": condition["surface"],
                "normalized": condition["normalized"],
                "predicate_type": condition["type"],
                "epistemic_kind": condition["kind"],
                "role": condition["role"],
                "status": status,
                "patient_evidence": evidence,
                "adjudication": {"method": method, "rationale": rationale},
            })
        evaluation = variant_evaluation(variant, conditions)
        variants.append({
            "variant_key": variant["variant_key"],
            "topic": variant["topic"],
            "variant_id": variant["variant_id"],
            "variant_text": variant["variant_text"],
            "logic": variant.get("logic", []),
            "aggregate_structure": variant.get("aggregate_structure", []),
            "predicate_matrix": conditions,
            "signature_evaluation": evaluation,
        })

    review = CASE_REVIEWS[step_id]
    by_variant = {row["variant_key"]: row for row in variants}
    candidates = []
    for rank, (variant_key, correspondence, rationale) in enumerate(review["candidates"], start=1):
        if variant_key not in by_variant:
            raise ValueError(f"unknown candidate Variant: {variant_key}")
        candidates.append({
            "rank": rank,
            "variant_key": variant_key,
            "topic": by_variant[variant_key]["topic"],
            "variant_id": by_variant[variant_key]["variant_id"],
            "candidate_correspondence": correspondence,
            "signature_evaluation": by_variant[variant_key]["signature_evaluation"],
            "rationale": rationale,
        })
    if review["label"] not in CORRESPONDENCE_LABELS:
        raise ValueError(f"invalid case label: {review['label']}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "manual_variant_adjudication_complete",
        "step_id": step_id,
        "source_patient_acr_mapping": str(path.relative_to(ROOT)),
        "source_patient_acr_mapping_sha256": sha256_file(path),
        "acr_registry": str(ACR_AUDIT.relative_to(ROOT)),
        "acr_registry_sha256": sha256_file(ACR_AUDIT),
        "blinding": {
            "aqc_hidden": True,
            "current_order_and_result_hidden": True,
            "acr_actions_and_ratings_hidden": True,
            "later_outcomes_hidden": True,
            "old_crosswalk_labels_hidden": True,
        },
        "variants": variants,
        "candidate_variants": candidates,
        "reviewed_correspondence": {
            "label": review["label"],
            "rationale": review["rationale"],
        },
        "candidate_bridge_operations": review["bridge_operations"],
    }


def validate_document(document: dict[str, Any]) -> None:
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("wrong schema version")
    if len(document["variants"]) != 17:
        raise ValueError("every document must contain all 17 Variants")
    instance_ids = []
    for variant in document["variants"]:
        for row in variant["predicate_matrix"]:
            if row["status"] not in PREDICATE_STATUSES:
                raise ValueError(f"invalid predicate status: {row['status']}")
            instance_ids.append(row["acr_predicate_instance_id"])
    if len(instance_ids) != 78 or len(set(instance_ids)) != 78:
        raise ValueError("every document must contain 78 unique predicate instances")
    if document["reviewed_correspondence"]["label"] not in CORRESPONDENCE_LABELS:
        raise ValueError("invalid correspondence label")
    expected_signature = {"exact": "satisfied", "partial": "partial"}
    for candidate in document["candidate_variants"]:
        expected = expected_signature.get(candidate["candidate_correspondence"])
        actual = candidate["signature_evaluation"]["overall_signature_status"]
        if expected is not None and actual != expected:
            raise ValueError(
                f"candidate {candidate['variant_key']} is labelled "
                f"{candidate['candidate_correspondence']} but its signature is {actual}"
            )


def readme_text() -> str:
    return """# Manual ACR Variant adjudication: pilot v1

This directory evaluates each of the 12 manually extracted patient Contexts against all 17
compiled ACR Variant signatures. Each output contains a 78-row predicate matrix, preserved ACR
logic/aggregate structure, a short candidate list, a reviewed case-level correspondence label,
and candidate bridge operations.

## Predicate decisions

- `exact_or_equivalent` and compatible `patient_value_narrower` links directly support a predicate.
- A direct negative at compatible time/scope contradicts it.
- Every `patient_value_broader` and `related_judgment_required` link is retained and reviewed. It
  remains `unknown` unless a hand-authored condition judgment supplies entailment or contradiction.
- Missing evidence is `unknown`, never absence.
- Aggregate members and indicators do not automatically satisfy their parent assessment.
- Variant 6's associated-state group is supported when one or more members are supported, while
  its other required predicates remain conjunctive.

## Case-level labels used in this calibration

- `exact`: one retained Variant signature is fully supported without an equally plausible
  unresolved competing frame.
- `partial`: the closest Variant has some supported and some unknown required predicates.
- `multiple`: generic/specific, sequential, or cross-topic Variants remain simultaneously plausible.
- `uncertain`: the active disease/topic or complication source cannot be resolved from available
  evidence.
- `out_of_scope`: no selected Variant provides a compatible frame.

The candidate rank is a human review aid, not an automatic assignment. A generic one-predicate
Variant can be satisfied while a more specific Variant remains partial. The case-level label and
rationale explicitly preserve that overlap.

## Blinding and provenance

Only the active manual patient-item-to-ACR mappings and the compiled ACR predicate registry are
loaded. A/Q/C, the current order/result, ACR Actions/ratings, old crosswalk labels, and later
outcomes remain hidden. Upstream extraction and mapping files are hash-bound in every output.

One upstream concern is deliberately preserved rather than silently repaired: Crohn disease in
`appendicitis:20123918:s2` and `s3` appears both as a background `patient_attribute` and as an
`alternative_diagnosis`. The cited diagnostic-state evidence establishes only history, not a
current alternative-diagnosis role. Variant adjudication therefore records this as an unsupported
role assignment and does not use it as independent Variant support.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_dir = output_root / "outputs"
    acr = read_json(ACR_AUDIT)
    paths = sorted(input_dir.glob("*.json"))
    if len(paths) != 12:
        raise ValueError(f"expected 12 mapping files, found {len(paths)}")

    documents = []
    for path in paths:
        document = adjudicate_file(path, acr)
        validate_document(document)
        write_json(output_dir / path.name, document)
        documents.append(document)

    labels = Counter(row["reviewed_correspondence"]["label"] for row in documents)
    bridge_operations = Counter(
        operation["operation"]
        for document in documents
        for operation in document["candidate_bridge_operations"]
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "manual_variant_adjudication_complete",
        "source_directory": str(input_dir.relative_to(ROOT)),
        "output_directory": str(output_dir.relative_to(ROOT)),
        "acr_registry": str(ACR_AUDIT.relative_to(ROOT)),
        "n_steps": len(documents),
        "n_variants_per_step": 17,
        "n_predicate_instances_per_step": 78,
        "n_predicate_decisions": len(documents) * 78,
        "correspondence_label_counts": dict(sorted(labels.items())),
        "candidate_bridge_operation_counts": dict(sorted(bridge_operations.items())),
        "steps": [
            {
                "step_id": document["step_id"],
                "reviewed_correspondence": document["reviewed_correspondence"],
                "candidate_variants": document["candidate_variants"],
                "candidate_bridge_operations": document["candidate_bridge_operations"],
            }
            for document in documents
        ],
    }
    write_json(output_root / "summary.json", summary)
    (output_root / "README.md").write_text(readme_text(), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
