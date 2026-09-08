#!/usr/bin/env python3
"""Map the 12-step manual open-context pilot to the audited 50-value ACR vocabulary.

The mappings below are deliberately semantic and exact-value scoped. They do
not use the observed imaging order, A/Q/C, action ratings, or later outcomes.
Every open item receives at least one mapping row; absence of an equivalent is
represented explicitly rather than by omission.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "data/aqc_acr_bridge/pilot_v1/open_context_manual_v1"
EXTRACTIONS = PILOT_DIR / "manual_extractions_v1.jsonl"
ACR_AUDIT = ROOT / "data/aqc_acr_bridge/acr_context_value_dimension_audit_v1.csv"
OUTPUT_JSONL = PILOT_DIR / "manual_acr_vocab_mappings_v1.jsonl"
OUTPUT_TSV = PILOT_DIR / "manual_acr_vocab_mapping_audit_v1.tsv"
OUTPUT_CASE_SUMMARY = PILOT_DIR / "manual_acr_vocab_mapping_case_summary_v1.tsv"

RELATIONS = {
    "exact_or_equivalent",
    "patient_value_broader",
    "patient_value_narrower",
    "related_judgment_required",
    "contradicted",
    "no_acr_equivalent",
}


def m(target: str, relation: str, rationale: str) -> dict[str, str]:
    return {"target": target, "relation": relation, "rationale": rationale}


# Exact-value adjudications. Repeated values reuse the same mapping, while the
# output remains item-level and retains case-specific evidence and links.
VALUE_RULES: dict[str, list[dict[str, str]]] = {
    # Symptoms
    "RUQ-predominant abdominal pain worse after eating, with bloating and nausea": [
        m("ctx_034", "patient_value_narrower", "The patient condition contains RUQ pain plus additional qualifiers."),
        m("ctx_026", "patient_value_narrower", "The patient condition contains nausea plus additional symptoms."),
    ],
    "abdominal pain and constipation, worsening over the last day with sharp pain and emesis": [
        m("ctx_006", "patient_value_narrower", "Worsening persistent abdominal pain entails continued abdominal pain."),
        m("ctx_049", "patient_value_narrower", "The condition contains vomiting/emesis plus additional symptoms."),
    ],
    "abdominal pain or diarrhea/constipation": [
        m("ctx_006", "contradicted", "The explicitly negated abdominal pain contradicts continued abdominal pain at this step."),
    ],
    "acute supraumbilical pain with nausea/vomiting": [
        m("ctx_026", "patient_value_narrower", "The condition contains nausea plus additional symptoms."),
        m("ctx_049", "patient_value_narrower", "The condition contains vomiting plus additional symptoms."),
    ],
    "constant sharp abdominal pain": [
        m("ctx_006", "patient_value_narrower", "Constant pain is a more specific continued-pain state."),
    ],
    "epigastric and right-upper-quadrant pain": [
        m("ctx_010", "patient_value_narrower", "The patient value includes epigastric pain and an additional location."),
        m("ctx_034", "patient_value_narrower", "The patient value includes RUQ pain and an additional location."),
    ],
    "fever or chills": [
        m("ctx_029", "patient_value_narrower", "Explicit denial of fever entails the ACR no-fever predicate; chills are an added negative."),
        m("ctx_012", "contradicted", "Explicit denial of fever contradicts fever."),
    ],
    "intermittent RLQ abdominal pain, worsening": [
        m("ctx_033", "patient_value_narrower", "The patient value adds intermittency and worsening to RLQ pain."),
        m("ctx_006", "patient_value_narrower", "A week of worsening pain entails continued abdominal pain."),
    ],
    "nausea with two episodes of emesis and inability to tolerate intake": [
        m("ctx_026", "patient_value_narrower", "The patient value contains nausea plus additional features."),
        m("ctx_049", "patient_value_narrower", "Two episodes of emesis instantiate vomiting more specifically."),
    ],
    "no nausea or vomiting": [
        m("ctx_026", "contradicted", "Explicit absence of nausea contradicts nausea."),
        m("ctx_049", "contradicted", "Explicit absence of vomiting contradicts vomiting."),
    ],
    "no vomiting, diarrhea, fever, or chills": [
        m("ctx_049", "contradicted", "Explicit absence of vomiting contradicts vomiting."),
        m("ctx_029", "patient_value_narrower", "Explicit absence of fever entails no fever; other negatives add specificity."),
        m("ctx_012", "contradicted", "Explicit absence of fever contradicts fever."),
    ],
    "pain progressed to the RLQ": [
        m("ctx_033", "exact_or_equivalent", "Pain is currently localized to the RLQ."),
    ],
    "pain radiated to the RLQ": [
        m("ctx_033", "exact_or_equivalent", "Pain is currently reported in the RLQ."),
    ],
    "pain radiating to the back with nausea and vomiting": [
        m("ctx_026", "patient_value_narrower", "The condition contains nausea plus pain and radiation."),
        m("ctx_049", "patient_value_narrower", "The condition contains vomiting plus pain and radiation."),
    ],
    "relatively sudden mid-epigastric pain radiating to the back": [
        m("ctx_010", "patient_value_narrower", "The patient value adds sudden onset and back radiation to epigastric pain."),
    ],
    "repeated vomiting with diffuse lower abdominal pain": [
        m("ctx_049", "patient_value_narrower", "Repeated vomiting is a more specific vomiting predicate."),
    ],
    "subjective fever at home": [
        m("ctx_012", "related_judgment_required", "Subjective unmeasured fever is related to, but not automatically equivalent to, an observed fever state."),
    ],
    "unrelenting worsening abdominal pain with nausea and one nonbloody emesis": [
        m("ctx_006", "patient_value_narrower", "Unrelenting worsening pain entails continued abdominal pain."),
        m("ctx_026", "patient_value_narrower", "The condition contains nausea plus additional features."),
        m("ctx_049", "patient_value_narrower", "A documented emesis episode instantiates vomiting more specifically."),
    ],

    # Signs and laboratory states
    "WBC within stated reference range": [
        m("ctx_030", "exact_or_equivalent", "A measured WBC within its reference interval entails no high WBC count."),
        m("ctx_009", "contradicted", "A normal WBC contradicts elevated WBC count."),
        m("ctx_025", "contradicted", "A normal WBC contradicts leukocytosis."),
    ],
    "blood pressure 75/40": [
        m("ctx_017", "patient_value_narrower", "The measured severe low blood pressure instantiates hypotension."),
    ],
    "leukocytosis (WBC 13.3 K/uL)": [
        m("ctx_009", "exact_or_equivalent", "The thresholded patient value is an elevated WBC count."),
        m("ctx_025", "exact_or_equivalent", "The thresholded patient value is leukocytosis."),
    ],
    "leukocytosis (WBC 14.0 K/uL)": [
        m("ctx_009", "exact_or_equivalent", "The thresholded patient value is an elevated WBC count."),
        m("ctx_025", "exact_or_equivalent", "The thresholded patient value is leukocytosis."),
    ],
    "leukocytosis (WBC 15.9 K/uL)": [
        m("ctx_009", "exact_or_equivalent", "The thresholded patient value is an elevated WBC count."),
        m("ctx_025", "exact_or_equivalent", "The thresholded patient value is leukocytosis."),
    ],
    "leukocytosis (WBC 17.1 K/uL)": [
        m("ctx_009", "exact_or_equivalent", "The thresholded patient value is an elevated WBC count."),
        m("ctx_025", "exact_or_equivalent", "The thresholded patient value is leukocytosis."),
    ],
    "mild leukocytosis (WBC 11.7 K/uL)": [
        m("ctx_009", "exact_or_equivalent", "The thresholded patient value is an elevated WBC count."),
        m("ctx_025", "exact_or_equivalent", "The thresholded patient value is leukocytosis."),
    ],
    "tachycardia at 120": [
        m("ctx_045", "patient_value_narrower", "The numeric heart rate is a specific tachycardic state."),
    ],
    "temperature 100.9": [
        m("ctx_012", "patient_value_narrower", "The measured temperature is above a conventional fever threshold."),
    ],
    "lipase 402": [
        m("ctx_019", "patient_value_broader", "Only elevated lipase is encoded; ACR requires increased amylase AND lipase."),
    ],
    "marked lipase elevation (4550 IU/L)": [
        m("ctx_019", "patient_value_broader", "Only elevated lipase is encoded; ACR requires increased amylase AND lipase."),
    ],
    "marked lipase elevation (5502 IU/L)": [
        m("ctx_019", "patient_value_broader", "Only elevated lipase is encoded; ACR requires increased amylase AND lipase."),
    ],
    "prior pancreatic/liver test abnormalities improved to near-normal current values": [
        m("ctx_011", "related_judgment_required", "Improved near-normal enzymes may be equivocal for current pancreatitis, but ACR specifically requires an amylase-and-lipase interpretation."),
    ],
    "acute-on-chronic renal dysfunction with creatinine 3.4/eGFR 17": [
        m("ctx_031", "related_judgment_required", "Renal dysfunction is present, but whether it confounds pancreatic enzymes is a separate causal judgment."),
    ],

    # Patient characteristics
    "G1P0 at 23 weeks 5 days gestation": [
        m("ctx_032", "patient_value_narrower", "Gestational age and parity add specificity to pregnant woman."),
    ],
    "G1P0 pregnancy": [
        m("ctx_032", "patient_value_narrower", "Parity adds specificity to pregnant woman."),
    ],
    "DM, chronic renal insufficiency, and COPD": [
        m("ctx_031", "related_judgment_required", "CRI is present in pancreatitis workup, but an effect on pancreatic-enzyme interpretation still requires judgment."),
    ],

    # Disease timing
    "acute onset on the morning of presentation": [
        m("ctx_024", "patient_value_narrower", "Same-morning onset is within the ACR less-than-48-to-72-hour window."),
    ],
    "less than one day into acute pain": [
        m("ctx_024", "patient_value_narrower", "Less than one day is within the ACR less-than-48-to-72-hour window."),
    ],
    "one day into acute pain": [
        m("ctx_024", "patient_value_narrower", "One day is within the ACR less-than-48-to-72-hour window."),
    ],
    "one day into an acute episode": [
        m("ctx_024", "patient_value_narrower", "One day is within the ACR less-than-48-to-72-hour window."),
    ],
    "onset the evening before morning ED presentation": [
        m("ctx_024", "patient_value_narrower", "Overnight onset is within the ACR less-than-48-to-72-hour window."),
    ],
    "four days into worsening pain": [
        m("ctx_015", "patient_value_narrower", "Four days is beyond the ACR 48-to-72-hour threshold range."),
    ],
    "earlier-week onset with worsening during the last day": [
        m("ctx_024", "related_judgment_required", "Earlier this week does not establish whether onset was below 48 hours."),
        m("ctx_015", "related_judgment_required", "Earlier this week does not establish whether onset exceeded 72 hours."),
    ],
    "one month of pain, worse the night before presentation": [
        m("ctx_014", "related_judgment_required", "One month is near, but does not unambiguously exceed, the greater-than-four-week boundary."),
        m("ctx_016", "related_judgment_required", "The source ACR threshold is compound; one month is temporally related but requires predicate adjudication."),
    ],
    "one week into symptoms, with worsening prompting presentation": [
        m("ctx_016", "related_judgment_required", "Exactly one week does not unambiguously satisfy ACR's greater-than-7-to-21-day wording."),
    ],

    # Encounter stage
    "initial emergency-department evaluation": [
        m("ctx_013", "exact_or_equivalent", "The current episode is explicitly at initial presentation."),
    ],
    "emergency-department presentation after worsening pain": [
        m("ctx_013", "patient_value_narrower", "This is the first documented presentation for the current episode, with an added trigger."),
    ],
    "emergency-department presentation for persistent pain": [
        m("ctx_013", "patient_value_narrower", "This is the first documented presentation for the current episode, with an added trigger."),
    ],
    "morning emergency-department presentation after overnight symptoms": [
        m("ctx_013", "patient_value_narrower", "This is the first documented presentation for the current episode, with added timing."),
    ],
    "re-presentation after transient improvement following IV fluids and antiemetics": [
        m("ctx_013", "contradicted", "A re-presentation after prior same-episode care is not a first-time presentation at this decision step."),
    ],
    "admitted after ED RUQ ultrasound": [
        m("ctx_013", "related_judgment_required", "Admission after ED imaging does not determine whether this is the first presentation of the ACR-framed disease."),
    ],
    "ICU evaluation after surgical and GI consultation recommending ERCP": [
        m("ctx_013", "related_judgment_required", "Later encounter stage does not determine whether this is the first presentation of the ACR-framed disease."),
    ],
    "MICU admission during COPD exacerbation requiring BiPAP": [
        m("ctx_013", "related_judgment_required", "MICU admission describes current care stage, not whether the ACR-framed disease is presenting for the first time."),
    ],
    "interfacility transfer after hypotension and a concerning abdominal examination": [
        m("ctx_013", "related_judgment_required", "Interfacility transfer does not determine whether this is the first presentation of the ACR-framed disease."),
    ],
    "post-I&D transfer for complete heart block with temporary pacing wire placed": [
        m("ctx_013", "related_judgment_required", "The current care stage is unrelated to whether the newly detected ACR-framed disease is a first presentation."),
    ],
    "transfer to labor and delivery after outside observation and GI/surgical review": [
        m("ctx_013", "related_judgment_required", "Outside observation and transfer do not determine whether this is the first presentation of the ACR-framed disease."),
    ],

    # Diagnoses
    "acute appendicitis": [
        m("ctx_040", "exact_or_equivalent", "The patient and ACR values both encode suspected appendicitis."),
    ],
    "acute pancreatitis": [
        m("ctx_003", "exact_or_equivalent", "The patient and ACR values both encode established acute pancreatitis."),
        m("ctx_039", "patient_value_narrower", "Established acute pancreatitis entails the less certain suspected acute pancreatitis predicate."),
    ],
    "alcohol-related pancreatitis": [
        m("ctx_039", "patient_value_narrower", "Suspected alcohol-related pancreatitis is more specific than suspected acute pancreatitis."),
    ],
    "alcohol-related versus occult/passed biliary pancreatitis": [
        m("ctx_041", "related_judgment_required", "Only one etiologic branch is biliary; resolving the disjunction is required before suspected biliary disease applies."),
    ],
    "biliary obstruction or recently passed stone as pancreatitis etiology": [
        m("ctx_041", "patient_value_narrower", "The suspected biliary mechanism is more specific than suspected biliary disease."),
    ],
    "complicated sigmoid diverticular disease with pelvic collections": [
        m("ctx_043", "patient_value_narrower", "Established, anatomically specified diverticular disease is more specific and more certain than suspected diverticulitis."),
    ],
    "undifferentiated acute abdomen; perforation or septic source not identified": [
        m("ctx_048", "exact_or_equivalent", "The active etiology remains explicitly unresolved."),
    ],
    "possible acute cholecystitis in the setting of cholelithiasis": [
        m("ctx_041", "patient_value_narrower", "Possible calculous cholecystitis is a more specific suspected biliary disease."),
        m("ctx_038", "contradicted", "Documented gallstones contradict the acalculous qualifier."),
    ],
    "acute cholecystitis without a consistently visualized stone": [
        m("ctx_041", "patient_value_narrower", "Suspected cholecystitis is more specific than suspected biliary disease."),
        m("ctx_038", "related_judgment_required", "Failure to consistently visualize a stone is not sufficient to establish an acalculous mechanism."),
    ],
    "acute cholecystitis with probable perforation": [
        m("ctx_041", "patient_value_narrower", "Complicated acute cholecystitis is more specific than suspected biliary disease."),
        m("ctx_038", "related_judgment_required", "No radiopaque stone is seen, but CT cannot establish that the disease is truly acalculous."),
    ],
    "passed gallstone with possible cholangitis versus congenital or autoimmune biliary abnormality": [
        m("ctx_041", "patient_value_narrower", "All documented alternatives concern a suspected biliary-tree disease process, with added etiologic detail."),
    ],
    "recent pancreatitis, now biochemically and radiologically challenged": [
        m("ctx_003", "related_judgment_required", "The prior diagnosis is documented, but whether acute pancreatitis still constitutes the current ACR context requires temporal adjudication."),
    ],

    # Severity or complication
    "critical concurrent respiratory decompensation requiring noninvasive ventilation": [
        m("ctx_007", "patient_value_narrower", "The patient value gives a specific basis for a critically ill state."),
    ],
    "no CT evidence of necrosis-related vascular complication or focal collection": [
        m("ctx_021", "contradicted", "Preserved enhancement/no necrosis evidence contradicts known necrotizing pancreatitis at this time."),
        m("ctx_022", "related_judgment_required", "No focal collection is not automatically equivalent to absence of every ACR pancreatic/peripancreatic fluid-collection subtype."),
    ],
    "persistent pneumoperitoneum with enlarging and new pelvic air/fluid collections": [
        m("ctx_042", "patient_value_narrower", "The patient value specifies established complications of diverticular disease more narrowly than ACR's suspected complications."),
    ],
    "severe phlegmonous pancreatitis with extensive tracking fluid and minimal interval progression": [
        m("ctx_022", "related_judgment_required", "Tracking fluid and phlegmon are related to, but not necessarily equivalent to, a defined pancreatic/peripancreatic fluid collection."),
        m("ctx_035", "related_judgment_required", "Imaging severity is documented, but no named clinical score or threshold is supplied."),
    ],
    "shock physiology with peritoneal signs and systemic inflammatory response": [
        m("ctx_007", "patient_value_narrower", "Shock with peritoneal signs is a specific critically ill state."),
        m("ctx_044", "related_judgment_required", "Systemic inflammatory physiology is present, but formal SIRS criteria and definition require adjudication."),
    ],

    # Prior-test interpretations
    "pelvic ultrasound did not fully assess the right adnexa because the right ovary was not visualized": [
        m("ctx_027", "related_judgment_required", "Nonvisualization may make the ultrasound equivocal for an adnexal question, but is not automatically a negative/equivocal study for the active question."),
    ],
    "ovarian abnormality not demonstrated": [
        m("ctx_027", "related_judgment_required", "A negative ovarian finding is narrower than, and does not alone establish, a globally negative/equivocal ultrasound."),
    ],
    "initial ultrasound shows no gallstones or gallbladder wall thickening": [
        m("ctx_027", "related_judgment_required", "Specific negative gallbladder findings do not by themselves establish that the ultrasound is negative for the active question."),
    ],
    "focused RLQ ultrasound did not show a fluid collection or sonographic inflammation": [
        m("ctx_027", "related_judgment_required", "The focused study has negative findings, but appendix visualization and adequacy for the active question are not documented."),
    ],
    "pancreas is incompletely visualized because of bowel gas": [
        m("ctx_027", "related_judgment_required", "Technical limitation may make ultrasound equivocal, but its relevance depends on the active question."),
    ],
    "pancreatic tail assessment was limited by bowel gas": [
        m("ctx_027", "related_judgment_required", "Partial nonvisualization may make ultrasound equivocal, but its relevance depends on the active question."),
    ],
    "CT and ultrasound show biliary duct dilation without a visualized gallstone or CBD stone": [
        m("ctx_027", "related_judgment_required", "The ultrasound component is equivocal for an obstructing stone, but the patient item also synthesizes CT and a specific biliary question."),
    ],
    "serial studies show gallbladder edema/sludge and mild CBD dilation but no persistent choledocholithiasis": [
        m("ctx_027", "related_judgment_required", "The serial ultrasound evidence is mixed rather than simply negative/equivocal, and must be interpreted for the active biliary question."),
    ],
    "ultrasound shows an enlarged noncompressible hyperemic appendix with surrounding fat edema and focal tenderness": [
        m("ctx_027", "contradicted", "This is a positive appendiceal ultrasound pattern, contradicting a negative-or-equivocal prior ultrasound state."),
    ],
    "no distinct pancreatic pseudocyst on serial CT": [
        m("ctx_022", "related_judgment_required", "Absence of a pseudocyst does not exclude every pancreatic or peripancreatic fluid-collection subtype."),
    ],
}


NO_EQ_RATIONALE = {
    "symptoms": "No audited ACR value represents this symptom or symptom combination at comparable meaning.",
    "signs_and_labs": "No audited ACR value represents this examination/laboratory condition at comparable meaning.",
    "patient_characteristics": "No audited ACR population predicate represents this patient attribute without adding an unsupported causal relation.",
    "disease_timing": "The patient timing does not determine any audited ACR timing threshold.",
    "prior_test": "The audited ACR prior-test vocabulary contains ultrasound only; this completed test has no equivalent.",
    "encounter_stage": "The patient encounter state does not map to the sole audited ACR encounter predicate.",
    "imaging_stage": "No audited ACR imaging-stage predicate is supported.",
    "diagnosis": "No audited ACR diagnostic predicate represents this diagnosis at comparable meaning.",
    "severity_or_complication": "No audited ACR severity/complication predicate represents this state at comparable meaning.",
    "evidence_interpretation": "No audited ACR evidence-interpretation predicate represents this finding or limitation at comparable meaning.",
}


def read_acr_vocab() -> dict[str, dict[str, str]]:
    with ACR_AUDIT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    vocab = {row["row_id"]: row for row in rows}
    if len(vocab) != 50:
        raise ValueError("expected exactly 50 audited ACR values")
    return vocab


def item_mappings(item: dict[str, Any], dimension: str) -> tuple[str, list[dict[str, str]]]:
    value = item["value_native"]

    if dimension == "prior_test" and "ultrasound" in value.lower():
        return "prior_test_ultrasound", [m(
            "ctx_047", "patient_value_narrower",
            "The patient value specifies the ultrasound region/protocol more narrowly than ACR's generic prior ultrasound.",
        )]
    if dimension == "imaging_stage" and value.startswith("after "):
        return "post_imaging_means_next", [m(
            "ctx_028", "patient_value_narrower",
            "A decision after one or more resulted studies is a more specifically described next imaging stage.",
        )]
    if value in VALUE_RULES:
        return "exact_value_adjudication", VALUE_RULES[value]
    return "manual_no_equivalent", [{
        "target": "",
        "relation": "no_acr_equivalent",
        "rationale": NO_EQ_RATIONALE[dimension],
    }]


def main() -> None:
    vocab = read_acr_vocab()
    extraction_rows = [json.loads(line) for line in EXTRACTIONS.read_text().splitlines() if line]
    output: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    source_items = 0

    for case in extraction_rows:
        annotation = case["annotation"]
        for epistemic_kind, context_key in (
            ("factual", "factual_context"),
            ("inferential", "inferential_context"),
        ):
            for dimension, items in annotation[context_key].items():
                for item in items:
                    source_items += 1
                    rule_id, mappings = item_mappings(item, dimension)
                    compiled = []
                    targets_seen: set[str] = set()
                    for mapping in mappings:
                        relation = mapping["relation"]
                        target = mapping["target"]
                        if relation not in RELATIONS:
                            raise ValueError(f"unknown relation {relation}")
                        if relation == "no_acr_equivalent":
                            if target:
                                raise ValueError("no_acr_equivalent must not have a target")
                            acr = None
                        else:
                            if target not in vocab:
                                raise ValueError(f"unknown ACR target {target}")
                            if target in targets_seen:
                                raise ValueError(f"duplicate target {target} for {case['case_id']}:{item['item_id']}")
                            targets_seen.add(target)
                            acr = vocab[target]
                        record = {
                            "acr_context_value_id": target or None,
                            "acr_native_value": acr["native_value"] if acr else None,
                            "acr_epistemic_kind": acr["epistemic_kind"] if acr else None,
                            "acr_dimension": acr["dimension"] if acr else None,
                            "acr_source_variant_ids": acr["source_variant_ids"].split(";") if acr else [],
                            "mapping_relation": relation,
                            "judgment_required": relation == "related_judgment_required",
                            "mapping_rationale": mapping["rationale"],
                            "mapping_rule_id": rule_id,
                        }
                        compiled.append(record)
                        audit_rows.append({
                            "case_id": case["case_id"],
                            "step_id": case["step_id"],
                            "patient_item_id": item["item_id"],
                            "patient_epistemic_kind": epistemic_kind,
                            "patient_dimension": dimension,
                            "patient_value_native": item["value_native"],
                            "assertion_status": item["assertion_status"],
                            "acr_context_value_id": target,
                            "acr_native_value": acr["native_value"] if acr else "",
                            "acr_dimension": acr["dimension"] if acr else "",
                            "mapping_relation": relation,
                            "judgment_required": str(relation == "related_judgment_required").lower(),
                            "mapping_rationale": mapping["rationale"],
                            "mapping_rule_id": rule_id,
                        })
                    output.append({
                        "schema_version": "1.0.0-patient-to-acr-vocab-mapping",
                        "case_id": case["case_id"],
                        "step_id": case["step_id"],
                        "mapping_status": "manual_contaminated_calibration_v1",
                        "patient_item_id": item["item_id"],
                        "patient_epistemic_kind": epistemic_kind,
                        "patient_dimension": dimension,
                        "patient_item": item,
                        "unmapped_value_within_dimension": all(
                            mapping["mapping_relation"] == "no_acr_equivalent"
                            for mapping in compiled
                        ),
                        "acr_vocab_mappings": compiled,
                    })

    if len(output) != source_items:
        raise ValueError("not every source item was mapped exactly once at the item-wrapper level")
    if any(not row["acr_vocab_mappings"] for row in output):
        raise ValueError("mapping lists must not be empty")

    OUTPUT_JSONL.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))
    fields = list(audit_rows[0])
    with OUTPUT_TSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    summary_rows = []
    for step_id in sorted({row["step_id"] for row in audit_rows}):
        step_rows = [row for row in audit_rows if row["step_id"] == step_id]
        item_ids = {row["patient_item_id"] for row in step_rows}
        linked_ids = {
            row["patient_item_id"] for row in step_rows
            if row["mapping_relation"] != "no_acr_equivalent"
        }
        def targets(relations: set[str]) -> str:
            return "|".join(sorted({
                row["acr_context_value_id"] for row in step_rows
                if row["mapping_relation"] in relations and row["acr_context_value_id"]
            }))
        summary_rows.append({
            "step_id": step_id,
            "patient_items": len(item_ids),
            "items_with_acr_link": len(linked_ids),
            "unmapped_items": len(item_ids - linked_ids),
            "entailed_acr_values": targets({"exact_or_equivalent", "patient_value_narrower"}),
            "partial_acr_values": targets({"patient_value_broader"}),
            "judgment_required_acr_values": targets({"related_judgment_required"}),
            "contradicted_acr_values": targets({"contradicted"}),
        })
    with OUTPUT_CASE_SUMMARY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    relation_counts = Counter(row["mapping_relation"] for row in audit_rows)
    mapped_items = sum(
        any(m["mapping_relation"] != "no_acr_equivalent" for m in row["acr_vocab_mappings"])
        for row in output
    )
    print(f"PASS: {source_items} patient items; {mapped_items} with >=1 ACR link; {source_items - mapped_items} unmapped")
    print("mapping rows:", len(audit_rows), dict(sorted(relation_counts.items())))
    print(OUTPUT_JSONL)
    print(OUTPUT_TSV)
    print(OUTPUT_CASE_SUMMARY)


if __name__ == "__main__":
    main()
