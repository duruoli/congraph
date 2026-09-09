"""Map the manually extracted 12-step patient Context to ACR predicate instances.

The mapping decisions in this file are deliberately explicit and conservative.
The script performs no lexical retrieval and never assigns a Variant. It only
applies the hand-authored semantic rules below, verifies every target against
the compiled 78-instance ACR registry, and writes one auditable mapping record
for every extracted patient item.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "results" / "aqc_acr_bridge" / "pilot_v1" / "manual_extraction_v1"
    / "outputs"
)
DEFAULT_OUTPUT = (
    ROOT / "results" / "aqc_acr_bridge" / "pilot_v1" / "manual_acr_mapping_v1"
)
ACR_AUDIT = ROOT / "data" / "aqc_acr_bridge" / "acr_variant_predicate_audit_v1.json"
SCHEMA_VERSION = "1.0.0-manual-patient-item-to-acr-mapping"
RELATIONS = {
    "exact_or_equivalent",
    "patient_value_broader",
    "patient_value_narrower",
    "related_judgment_required",
    "contradicted",
    "no_acr_equivalent",
}

RLQ_PAIN = ["acr_21_v1:c01", "acr_21_v2:c01", "acr_21_v3:c02"]
RLQ_FEVER = ["acr_21_v2:c02", "acr_21_v3:c03"]
RLQ_WBC = ["acr_21_v2:c03", "acr_21_v3:c04"]
RLQ_APPENDICITIS = ["acr_21_v2:c04", "acr_21_v3:c05"]
PREGNANCY = ["acr_21_v3:c01"]

RUQ_PAIN = [
    "acr_132_v1:c01", "acr_132_v2:c01", "acr_132_v3:c01",
    "acr_132_v4:c01", "acr_132_v5:c01",
]
RUQ_NO_FEVER = ["acr_132_v3:c02"]
RUQ_FEVER = ["acr_132_v4:c02"]
RUQ_NO_HIGH_WBC = ["acr_132_v3:c03"]
RUQ_HIGH_WBC = ["acr_132_v4:c03"]
RUQ_BILIARY = ["acr_132_v2:c02", "acr_132_v3:c04", "acr_132_v4:c04"]
RUQ_ACALCULOUS = ["acr_132_v5:c02"]
RUQ_US_HISTORY = ["acr_132_v3:c05", "acr_132_v4:c05", "acr_132_v5:c03"]
RUQ_US_ACUTE_CHOLE = ["acr_132_v3:c06", "acr_132_v4:c06"]
RUQ_US_ACALCULOUS = ["acr_132_v5:c04"]

LLQ_PAIN = ["acr_20_v1:c01", "acr_20_v2:c01", "acr_20_v3:c01"]
LLQ_DIVERTICULITIS = ["acr_20_v2:c02"]
LLQ_DIVERTICULITIS_COMPLICATION = ["acr_20_v3:c02"]

AP_SUSPECTED = ["acr_126_v1:c01", "acr_126_v2:c01"]
AP_ESTABLISHED = ["acr_126_v3:c01", "acr_126_v4:c01", "acr_126_v6:c01"]
AP_NECROTIZING = ["acr_126_v5:c01"]
AP_FIRST_PRESENTATION = ["acr_126_v1:c02", "acr_126_v2:c02"]
AP_EPIGASTRIC_PAIN = ["acr_126_v1:c03"]
AP_AMYLASE_HIGH = ["acr_126_v1:c04"]
AP_LIPASE_HIGH = ["acr_126_v1:c05"]
AP_EARLY_TIME = ["acr_126_v1:c06"]
AP_ATYPICAL = ["acr_126_v2:c03"]
AP_AMYLASE_EQUIVOCAL = ["acr_126_v2:c04"]
AP_LIPASE_EQUIVOCAL = ["acr_126_v2:c05"]
AP_OTHER_DIAGNOSIS = ["acr_126_v2:c06"]
AP_CRITICAL = ["acr_126_v3:c02"]
AP_SIRS = ["acr_126_v3:c03", "acr_126_v4:c02"]
AP_SEVERE_SCORE = ["acr_126_v3:c04", "acr_126_v4:c03"]
AP_OVER_48_72H = ["acr_126_v3:c08"]
AP_WBC_HIGH = ["acr_126_v4:c04"]
AP_FEVER = ["acr_126_v4:c05"]
AP_OVER_7_21D = ["acr_126_v4:c06"]
AP_DETERIORATION = ["acr_126_v5:c02"]
AP_HYPOTENSION = ["acr_126_v5:c05"]
AP_TACHYPNEA = ["acr_126_v5:c07"]
AP_WBC_INCREASE = ["acr_126_v5:c09"]
AP_COLLECTION = ["acr_126_v6:c02"]
AP_PERSISTENT_PAIN = ["acr_126_v6:c03"]
AP_NAUSEA = ["acr_126_v6:c05"]
AP_VOMITING = ["acr_126_v6:c06"]
AP_OVER_4W = ["acr_126_v6:c08"]


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


def load_registry() -> dict[str, dict[str, Any]]:
    audit = read_json(ACR_AUDIT)
    registry: dict[str, dict[str, Any]] = {}
    for variant in audit["variants"]:
        for condition in variant["conditions"]:
            instance_id = f"{variant['variant_key']}:{condition['id']}"
            registry[instance_id] = {
                "acr_predicate_instance_id": instance_id,
                "variant_key": variant["variant_key"],
                "topic": variant["topic"],
                "variant_id": variant["variant_id"],
                "variant_text": variant["variant_text"],
                "condition_id": condition["id"],
                "surface": condition["surface"],
                "normalized": condition["normalized"],
                "predicate_type": condition["type"],
                "epistemic_kind": condition["kind"],
                "role": condition["role"],
            }
    if len(registry) != 78:
        raise ValueError(f"expected 78 ACR instances, found {len(registry)}")
    return registry


def semantic_links(step_id: str, dimension: str, item: dict[str, Any]) -> list[dict[str, str]]:
    """Return hand-authored semantic links for one native patient item."""
    links: list[dict[str, str]] = []

    def add(ids: list[str], relation: str, rationale: str) -> None:
        if relation not in RELATIONS - {"no_acr_equivalent"}:
            raise ValueError(f"invalid mapped relation: {relation}")
        for instance_id in ids:
            links.append({
                "acr_predicate_instance_id": instance_id,
                "relation": relation,
                "rationale": rationale,
            })

    if dimension == "patient_attribute":
        attribute = item["attribute"]
        if attribute == "pregnancy":
            add(PREGNANCY, "patient_value_narrower",
                "The patient item entails pregnancy and adds gravidity/gestational detail.")
        if attribute in {
            "chronic_kidney_disease", "chronic_renal_insufficiency",
            "end_stage_renal_disease", "acute_renal_failure",
        }:
            add(AP_ATYPICAL, "related_judgment_required",
                "Renal disease can confound pancreatic enzymes, but does not alone establish the ACR atypical-presentation aggregate.")

    elif dimension == "symptom_state":
        symptom = item["symptom"]
        site = item["site"]
        state = item["state"]
        if symptom == "abdominal_pain":
            if state == "absent":
                add(RLQ_PAIN + RUQ_PAIN + LLQ_PAIN + AP_EPIGASTRIC_PAIN + AP_PERSISTENT_PAIN,
                    "contradicted",
                    "General abdominal-pain absence negates these location-specific/persistent pain predicates at the documented time.")
            else:
                if "right_lower_quadrant" in site:
                    relation = (
                        "exact_or_equivalent"
                        if site == "right_lower_quadrant" and state == "present"
                        else "patient_value_narrower"
                    )
                    add(RLQ_PAIN, relation,
                        "The patient item contains right-lower-quadrant abdominal pain; additional sites or course make it narrower when present.")
                if "right_upper_quadrant" in site:
                    add(RUQ_PAIN, "patient_value_narrower",
                        "The patient item entails right-upper-quadrant pain and adds distribution, radiation, or course.")
                if "left_lower_quadrant" in site:
                    add(LLQ_PAIN, "patient_value_narrower",
                        "The patient item entails left-lower-quadrant pain and adds distribution or course.")
                if "epigastr" in site:
                    add(AP_EPIGASTRIC_PAIN, "patient_value_narrower",
                        "The patient item entails epigastric pain and adds radiation, another site, or course.")
                if state == "persistent":
                    add(AP_PERSISTENT_PAIN, "patient_value_narrower",
                        "Persistent pain is equivalent to continued pain while the patient item adds an anatomic distribution.")
                elif step_id.startswith("pancreatitis:") and state in {"present", "worsening"}:
                    add(AP_PERSISTENT_PAIN, "related_judgment_required",
                        "Active or worsening abdominal pain is related, but duration sufficient for ACR 'continued' pain requires judgment.")
        elif symptom == "postprandial_abdominal_pain" and site == "right_upper_quadrant":
            add(RUQ_PAIN, "patient_value_narrower",
                "Postprandial RUQ pain entails RUQ pain and adds a provoking context.")
        elif symptom in {"nausea", "nausea_and_vomiting"}:
            relation = "contradicted" if state == "absent" else (
                "patient_value_narrower" if state == "persistent" or symptom == "nausea_and_vomiting"
                else "exact_or_equivalent"
            )
            add(AP_NAUSEA, relation,
                "The patient item directly records nausea status; persistence or the combined symptom adds detail.")
            if symptom == "nausea_and_vomiting":
                add(AP_VOMITING, "contradicted" if state == "absent" else "patient_value_narrower",
                    "The combined item directly records vomiting status and adds nausea.")
        elif symptom == "vomiting":
            add(AP_VOMITING, "contradicted" if state == "absent" else "exact_or_equivalent",
                "The patient item directly records vomiting presence or absence.")
        elif symptom in {"fever", "subjective_fever"} and state == "present":
            add(RLQ_FEVER + RUQ_FEVER + AP_FEVER, "related_judgment_required",
                "Reported fever is clinically related to ACR fever, but objective/current compatibility requires confirmation.")
            add(RUQ_NO_FEVER, "contradicted",
                "Reported fever is incompatible with the no-fever predicate if timing is judged compatible.")
        elif symptom in {"fever_and_chills", "fever_chills_and_sweats"} and state == "absent":
            add(RUQ_NO_FEVER, "exact_or_equivalent",
                "Explicit denial of fever supports the ACR no-fever predicate.")
            add(RLQ_FEVER + RUQ_FEVER + AP_FEVER, "contradicted",
                "Explicit fever denial negates fever-present predicates at a compatible time and scope.")

    elif dimension == "sign_state":
        sign = item["sign"]
        state = item["state"]
        if sign == "hypotension":
            add(AP_HYPOTENSION, "patient_value_narrower",
                "A documented blood pressure of 75/40 is a specific instance of hypotension.")
        if sign == "respiratory_effort" and "tachypneic" in state:
            add(AP_TACHYPNEA, "related_judgment_required",
                "Qualitative tachypnea is related to the ACR vital-sign predicate, but no compatible rate/change is supplied.")
        if sign in {"somnolence", "confusion"}:
            add(AP_CRITICAL, "related_judgment_required",
                "Altered mental status may contribute to critical illness but cannot establish that aggregate alone.")

    elif dimension == "lab_finding_state":
        analyte = item["analyte"]
        state = item["state"]
        if analyte == "white_blood_cell_count":
            add(RLQ_WBC + RUQ_HIGH_WBC + AP_WBC_HIGH, "patient_value_narrower",
                "The numeric leukocyte value is a more specific instance of a high WBC/leukocytosis predicate.")
            add(RUQ_NO_HIGH_WBC, "contradicted",
                "The documented high WBC conflicts with the no-high-WBC predicate.")
            add(AP_WBC_INCREASE, "related_judgment_required",
                "A high value does not by itself demonstrate an interval increase.")
        elif analyte == "lipase":
            if state == "4k":
                add(AP_LIPASE_HIGH, "patient_value_narrower",
                    "A lipase of 4,000 is a numeric, markedly increased lipase result.")
                add(AP_LIPASE_EQUIVOCAL, "contradicted",
                    "A markedly elevated lipase is not an equivocal enzyme result.")
            else:
                add(AP_LIPASE_HIGH + AP_LIPASE_EQUIVOCAL, "related_judgment_required",
                    "A falling pair of values requires reference ranges and decision-time selection to classify as increased or equivocal.")

    elif dimension == "imaging_finding_state":
        finding = item["finding"]
        state = item["state"]
        if finding in {"enlarged_noncompressible_appendix", "appendiceal_wall_hyperemia"}:
            add(RLQ_APPENDICITIS, "related_judgment_required",
                "This is a component imaging sign that requires synthesis before assigning suspected appendicitis.")
        elif finding in {"right_lower_quadrant_inflammation", "periappendiceal_inflammation"}:
            add(RLQ_APPENDICITIS, "related_judgment_required",
                "Absence of this component finding lowers support but does not directly negate appendicitis.")
        elif finding in {
            "gallstones", "gallbladder_wall_thickening", "pericholecystic_fat_stranding",
            "pericholecystic_inflammation", "gallbladder_wall_perforation",
            "gallbladder_wall_edema_and_thickening", "gallbladder_wall_edema",
            "gallbladder_sludge", "acute_gallbladder_inflammation",
        }:
            add(RUQ_BILIARY, "related_judgment_required",
                "The gallbladder finding informs suspected biliary disease but does not alone assign the diagnostic Context.")
            add(RUQ_ACALCULOUS, "related_judgment_required",
                "The finding may inform acalculous cholecystitis, but stones, inflammation, and patient setting must be synthesized.")
        elif finding in {"sigmoid_wall_thickening_with_diverticular_disease", "sigmoid_diverticulosis"}:
            add(LLQ_DIVERTICULITIS, "related_judgment_required",
                "The colonic finding is related to diverticulitis but requires clinical/imaging synthesis.")
        elif finding in {"pneumoperitoneum", "pelvic_collections"} and state != "absent":
            add(LLQ_DIVERTICULITIS_COMPLICATION, "related_judgment_required",
                "The finding is a potential complication, but attribution to diverticulitis requires judgment.")
        elif finding in {"peripancreatic_inflammation", "peripancreatic_inflammatory_change"}:
            add(AP_SUSPECTED + AP_ESTABLISHED, "related_judgment_required",
                "A pancreatic inflammatory imaging sign supports pancreatitis but requires integrated diagnostic interpretation.")
        elif finding == "pancreatic_necrosis":
            add(AP_NECROTIZING, "related_judgment_required",
                "Necrosis was not assessed, so known necrotizing pancreatitis cannot be established or contradicted.")
        elif finding == "pancreatic_or_peripancreatic_collection":
            if state == "absent":
                add(AP_COLLECTION, "contradicted",
                    "The patient item explicitly records absence of a pancreatic/peripancreatic collection.")
            else:
                add(AP_COLLECTION, "related_judgment_required",
                    "The finding is related to the required known collection predicate but certainty/scope must be checked.")

    elif dimension == "diagnostic_state":
        condition = item["condition"]
        status = item["status"]
        role = item["role"]
        if condition == "acute_appendicitis" and status == "established":
            add(RLQ_APPENDICITIS, "patient_value_narrower",
                "Established acute appendicitis entails suspected appendicitis and adds certainty.")
        elif condition == "acute_cholecystitis":
            relation = "patient_value_narrower" if status in {"established", "equivocal"} else "related_judgment_required"
            add(RUQ_BILIARY, relation,
                "Acute cholecystitis is a more specific biliary diagnosis; its certainty must remain attached.")
            add(RUQ_ACALCULOUS, "related_judgment_required",
                "Whether the cholecystitis is specifically acalculous requires stone and clinical-context synthesis.")
        elif condition == "acute_pancreatitis":
            if status == "suspected":
                add(AP_SUSPECTED, "exact_or_equivalent",
                    "The diagnosis and suspected status match the ACR predicate.")
                add(AP_ESTABLISHED, "patient_value_broader",
                    "Suspected pancreatitis does not entail established pancreatitis.")
                add(AP_NECROTIZING, "patient_value_broader",
                    "Suspected pancreatitis does not entail the known necrotizing subtype.")
            elif status == "established":
                add(AP_SUSPECTED, "patient_value_narrower",
                    "Established pancreatitis entails suspected pancreatitis and adds certainty.")
                add(AP_ESTABLISHED, "exact_or_equivalent",
                    "The diagnosis and established status match the ACR predicate.")
                add(AP_NECROTIZING, "patient_value_broader",
                    "Established acute pancreatitis does not entail necrosis.")
            else:
                add(AP_SUSPECTED + AP_ESTABLISHED + AP_NECROTIZING,
                    "related_judgment_required",
                    "A challenged diagnosis cannot be treated as either a direct match or a direct negation without adjudicating the conflicting evidence.")
        elif condition == "complicated_diverticulitis" and status == "established":
            add(LLQ_DIVERTICULITIS, "patient_value_narrower",
                "Established complicated diverticulitis entails suspected diverticulitis and adds certainty/complication status.")
            add(LLQ_DIVERTICULITIS_COMPLICATION, "patient_value_narrower",
                "Established complicated diverticulitis entails a suspected diverticulitis complication and adds certainty.")
        elif condition in {"multiple_pelvic_collections", "pneumoperitoneum"}:
            add(LLQ_DIVERTICULITIS_COMPLICATION, "related_judgment_required",
                "The complication is compatible with complicated diverticulitis, but its cause is not encoded in this item alone.")
        elif condition == "diverticulitis":
            add(LLQ_DIVERTICULITIS, "related_judgment_required",
                "Equivocal diverticulitis is clinically related to suspected diverticulitis but does not have identical assertion status.")
        elif condition == "pancreatic_necrosis":
            add(AP_NECROTIZING, "related_judgment_required",
                "Equivocal necrosis does not establish known necrotizing pancreatitis.")

        if role == "alternative_diagnosis" and status != "excluded":
            add(AP_OTHER_DIAGNOSIS, "patient_value_narrower",
                "A named possible alternative is more specific than ACR's generic possibility of a non-pancreatitis diagnosis.")

    elif dimension == "test_history":
        test = item["test"]
        if "ultrasound" in test and item["status"] == "completed":
            add(RUQ_US_HISTORY, "patient_value_narrower",
                "The completed ultrasound item adds modality region, protocol, or serial detail to ACR's generic ultrasound history.")

    elif dimension == "test_interpretation":
        test = item["test"]
        target = item["target"]
        result = item["result"]
        is_ultrasound = "ultrasound" in test
        if is_ultrasound and target == "acute_cholecystitis":
            if result == "equivocal":
                add(RUQ_US_ACUTE_CHOLE, "patient_value_narrower",
                    "Equivocal is one explicit branch of ACR's negative-or-equivocal ultrasound predicate.")
                add(RUQ_US_ACALCULOUS, "patient_value_broader",
                    "The target acute cholecystitis is less specific than acalculous cholecystitis.")
            elif result == "positive":
                add(RUQ_US_ACUTE_CHOLE + RUQ_US_ACALCULOUS, "contradicted",
                    "A positive result conflicts with a negative-or-equivocal result at compatible scope.")
        elif is_ultrasound and target in {
            "biliary_disease", "biliary_obstruction_or_stone",
            "biliary_cause_of_pancreatitis", "gallbladder_change",
        }:
            add(RUQ_US_ACUTE_CHOLE + RUQ_US_ACALCULOUS, "related_judgment_required",
                "The ultrasound result concerns a related but nonidentical target; acute/acalculous cholecystitis requires additional interpretation.")

    elif dimension == "aggregate_assessment":
        assessment = item["assessment"]
        if assessment in {
            "copd_exacerbation_requiring_noninvasive_ventilation",
            "hemodynamic_instability_with_peritoneal_abdomen",
        }:
            add(AP_CRITICAL, "related_judgment_required",
                "The assessment may indicate critical illness, but its target is not acute pancreatitis and the ACR aggregate requires adjudication.")
        elif assessment == "severe_pancreatic_inflammatory_burden":
            add(AP_CRITICAL + AP_SEVERE_SCORE, "related_judgment_required",
                "Severe imaging burden is related to illness severity but is neither a critical-illness assertion nor a validated severe clinical score.")

    elif dimension == "temporal_position":
        if step_id == "pancreatitis:20001800:s2":
            add(AP_EARLY_TIME, "related_judgment_required",
                "One day of symptoms at presentation likely falls within 48-72 hours, but exact decision time and anchor alignment require confirmation.")
        elif step_id == "pancreatitis:20720063:s3":
            add(AP_EARLY_TIME, "related_judgment_required",
                "Onset the previous evening likely falls within 48-72 hours, but the current-decision interval is not explicitly calculated.")
        elif step_id == "pancreatitis:25133113:s6" and item["value"] == "4_days":
            add(AP_EARLY_TIME, "contradicted",
                "Four days since onset exceeds the less-than-48-to-72-hour window.")
            add(AP_OVER_48_72H, "patient_value_narrower",
                "Four days is a more precise duration beyond 48-72 hours.")
            add(AP_OVER_7_21D + AP_OVER_4W, "contradicted",
                "Four days is shorter than the stated later ACR temporal thresholds.")

    elif dimension == "diagnosis_presentation_stage":
        if item["diagnosis"] == "acute_pancreatitis" and item["stage"] == "first_presentation":
            add(AP_FIRST_PRESENTATION, "exact_or_equivalent",
                "The patient and ACR predicates both state first presentation of acute pancreatitis.")

    # Preserve the order above while removing duplicate targets produced by
    # overlapping semantic rules. Conflicting relations are a coding error.
    unique: dict[str, dict[str, str]] = {}
    for link in links:
        instance_id = link["acr_predicate_instance_id"]
        previous = unique.get(instance_id)
        if previous is not None and previous["relation"] != link["relation"]:
            raise ValueError(
                f"conflicting relations for {step_id} {dimension} {item}: "
                f"{previous} versus {link}"
            )
        unique[instance_id] = link
    return list(unique.values())


def enrich_links(
    links: list[dict[str, str]], registry: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    enriched = []
    for link in links:
        instance_id = link["acr_predicate_instance_id"]
        if instance_id not in registry:
            raise ValueError(f"unknown ACR predicate instance: {instance_id}")
        enriched.append({**registry[instance_id], **link})
    return enriched


def map_file(path: Path, registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    extraction = read_json(path)
    disease, hadm_id, step = path.stem.split("_", 2)
    step_id = f"{disease}:{hadm_id}:{step}"
    mappings = []
    relation_counts: Counter[str] = Counter()
    channel_counts: Counter[str] = Counter()

    def append_item(item_ref: str, dimension: str, item: dict[str, Any], is_other: bool) -> None:
        raw_links = semantic_links(step_id, dimension, item)
        if raw_links:
            links: list[dict[str, Any]] = enrich_links(raw_links, registry)
            channel = "mapped_to_acr"
        else:
            links = [{
                "acr_predicate_instance_id": None,
                "relation": "no_acr_equivalent",
                "rationale": (
                    "No comparable predicate instance exists in the selected 17-Variant ACR registry."
                ),
            }]
            channel = "other_proposed_dimension" if is_other else "unmapped_value_within_dimension"
        for link in links:
            relation_counts[link["relation"]] += 1
        channel_counts[channel] += 1
        mappings.append({
            "item_ref": item_ref,
            "dimension": dimension,
            "patient_item": item,
            "mapping_channel": channel,
            "links": links,
        })

    for dimension, items in extraction["patient_context"].items():
        for index, item in enumerate(items):
            append_item(f"patient_context.{dimension}[{index}]", dimension, item, False)
    for index, item in enumerate(extraction["other_proposed_dimension"]):
        append_item(f"other_proposed_dimension[{index}]", item["dimension"], item, True)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "manual_semantic_mapping_complete",
        "step_id": step_id,
        "source_extraction": str(path.relative_to(ROOT)),
        "source_extraction_sha256": sha256_file(path),
        "acr_registry": str(ACR_AUDIT.relative_to(ROOT)),
        "acr_registry_sha256": sha256_file(ACR_AUDIT),
        "mapping_direction": "patient_item_relative_to_acr_predicate",
        "mappings": mappings,
        "summary": {
            "n_patient_items": len(mappings),
            "n_mapped_items": channel_counts["mapped_to_acr"],
            "n_unmapped_values_within_dimension": channel_counts["unmapped_value_within_dimension"],
            "n_other_proposed_dimension_items": channel_counts["other_proposed_dimension"],
            "n_links": sum(relation_counts.values()),
            "relation_counts": dict(sorted(relation_counts.items())),
        },
    }


def validate_mapping(value: dict[str, Any], registry: dict[str, dict[str, Any]]) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("incorrect mapping schema_version")
    mappings = value.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        raise ValueError("mappings must be a non-empty list")
    refs = [row.get("item_ref") for row in mappings]
    if len(refs) != len(set(refs)):
        raise ValueError(f"duplicate item_ref in {value.get('step_id')}")
    for row in mappings:
        links = row.get("links")
        if not isinstance(links, list) or not links:
            raise ValueError(f"mapping has no links: {row.get('item_ref')}")
        channel = row.get("mapping_channel")
        for link in links:
            relation = link.get("relation")
            if relation not in RELATIONS:
                raise ValueError(f"invalid relation: {relation}")
            instance_id = link.get("acr_predicate_instance_id")
            if relation == "no_acr_equivalent":
                if instance_id is not None or channel == "mapped_to_acr":
                    raise ValueError("no_acr_equivalent must have null target and open channel")
            elif instance_id not in registry or channel != "mapped_to_acr":
                raise ValueError("mapped link must target the registry and use mapped_to_acr")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_dir = output_root / "outputs"
    registry = load_registry()
    input_paths = sorted(input_dir.glob("*.json"))
    if len(input_paths) != 12:
        raise ValueError(f"expected 12 manual extraction files, found {len(input_paths)}")

    summaries = []
    mapped_documents = []
    for path in input_paths:
        mapped = map_file(path, registry)
        validate_mapping(mapped, registry)
        write_json(output_dir / path.name, mapped)
        mapped_documents.append(mapped)
        summaries.append({"step_id": mapped["step_id"], **mapped["summary"]})

    aggregate_relations: Counter[str] = Counter()
    for row in summaries:
        aggregate_relations.update(row["relation_counts"])
    dimension_channels: dict[str, Counter[str]] = defaultdict(Counter)
    links_by_topic: Counter[str] = Counter()
    linked_instances: set[str] = set()
    for document in mapped_documents:
        for mapping in document["mappings"]:
            dimension_channels[mapping["dimension"]][mapping["mapping_channel"]] += 1
            for link in mapping["links"]:
                instance_id = link["acr_predicate_instance_id"]
                if instance_id is not None:
                    linked_instances.add(instance_id)
                    links_by_topic[link["topic"]] += 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "manual_semantic_mapping_complete",
        "source_directory": str(input_dir.relative_to(ROOT)),
        "output_directory": str(output_dir.relative_to(ROOT)),
        "acr_registry": str(ACR_AUDIT.relative_to(ROOT)),
        "n_acr_predicate_instances": len(registry),
        "n_linked_acr_predicate_instances": len(linked_instances),
        "n_unlinked_acr_predicate_instances": len(registry) - len(linked_instances),
        "n_steps": len(summaries),
        "n_patient_items": sum(row["n_patient_items"] for row in summaries),
        "n_mapped_items": sum(row["n_mapped_items"] for row in summaries),
        "n_unmapped_values_within_dimension": sum(
            row["n_unmapped_values_within_dimension"] for row in summaries
        ),
        "n_other_proposed_dimension_items": sum(
            row["n_other_proposed_dimension_items"] for row in summaries
        ),
        "n_links": sum(row["n_links"] for row in summaries),
        "relation_counts": dict(sorted(aggregate_relations.items())),
        "item_channel_counts_by_dimension": {
            dimension: dict(sorted(counts.items()))
            for dimension, counts in sorted(dimension_channels.items())
        },
        "link_counts_by_acr_topic": dict(sorted(links_by_topic.items())),
        "steps": summaries,
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
