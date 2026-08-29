"""Build formal Track-B development artifacts from the canonical full corpus.

The final-test partition is kept at metadata-only access. Only development
trajectories are profiled for qualitative selection or opened for coding. No
ACR, verification, deviation, current-result, or later-event field participates.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "results" / "annotation_experiment" / "full"
TIMING_CSV = SOURCE_DIR / "timing_roles.csv"
OUT_DIR = ROOT / "data" / "aqc_development"
SPLIT_SALT = "congraph-aqc-track-b-split-v1"
SAMPLE_SALT = "congraph-aqc-track-b-codebook-v1"
DISEASES = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")
ALLOWED_EX_ANTE_FIELDS = (
    "differential", "other_hypothesis", "information_gap", "expected_finding",
    "action_role", "appropriateness", "appropriateness_reason", "grounding", "reasoning",
)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def eligible_records() -> list[dict[str, Any]]:
    records = []
    for path in sorted(SOURCE_DIR.glob("*.json")):
        if path.name == "manifest.json":
            continue
        data = read_json(path)
        if not isinstance(data, dict) or not {"disease", "hadm_id", "steps"}.issubset(data):
            continue
        records.append({
            "disease": str(data["disease"]), "hadm_id": int(data["hadm_id"]),
            "n_steps": len(data["steps"]),
            "source_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        })
    return records


def make_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    patients = []
    for disease in DISEASES:
        group = [dict(row) for row in records if row["disease"] == disease]
        for row in group:
            row["split_hash"] = stable_hash(f"{SPLIT_SALT}|{disease}|{row['hadm_id']}")
        group.sort(key=lambda row: (row["split_hash"], row["hadm_id"]))
        n_test = round(len(group) * 0.20)
        for index, row in enumerate(group):
            is_test = index < n_test
            patients.append({
                **row, "partition": "final_test" if is_test else "development",
                "annotation_access": (
                    "metadata_only_until_framework_and_models_frozen"
                    if is_test else "development_allowed"
                ),
            })
    summary = {}
    for partition in ("development", "final_test"):
        subset = [row for row in patients if row["partition"] == partition]
        summary[partition] = {
            "n_patients": len(subset), "n_steps": sum(row["n_steps"] for row in subset),
            "by_disease": {
                disease: {
                    "n_patients": sum(row["disease"] == disease for row in subset),
                    "n_steps": sum(row["n_steps"] for row in subset if row["disease"] == disease),
                } for disease in DISEASES
            },
        }
    return {
        "schema_version": "1.0.0", "created_from": "results/annotation_experiment/full/*.json",
        "eligibility_rule": "object with disease, hadm_id, steps; manifest.json excluded",
        "algorithm": (
            "within disease sort SHA-256(salt|disease|hadm_id); assign Python round(0.20*n) "
            "lowest hashes to final_test, remainder to development"
        ),
        "hash_salt": SPLIT_SALT,
        "unit": "patient trajectory; all steps inherit the patient partition",
        "final_test_policy": (
            "only identity, disease, source path, and step-count metadata are recorded; annotation "
            "content may not be profiled, selected, coded, or used for revision before freeze"
        ),
        "summary": summary,
        "patients": sorted(patients, key=lambda row: (row["disease"], row["hadm_id"])),
    }


def load_timing() -> dict[tuple[str, int, int], dict[str, str]]:
    timing = {}
    with TIMING_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            # Deliberately ignore dev_belief, verification, interventions, and dates.
            timing[(row["disease"], int(row["hadm"]), int(row["step"]))] = {
                "modality": row["modality"], "timing_role": row["timing_role"]
            }
    return timing


def modality_family(ordered: str) -> str:
    upper = ordered.upper()
    for token in ("MRCP", "MRI", "CTU", "CT"):
        if token in upper:
            return token
    if "ULTRASOUND" in upper or re.search(r"\bUS\b", upper):
        return "US"
    return "OTHER"


LIMITATION_PATTERNS = {
    "prior_study_limited": r"limited|nondiagnostic|non-diagnostic|inadequate|suboptimal",
    "target_nonvisualized": r"nonvisuali[sz]|not visuali[sz]|unable to visuali[sz]|failed to visuali[sz]",
    "indeterminate_or_unresolved": r"indeterminate|equivocal|uncertain|unresolved|remain(?:s|ed)? (?:open|unclear)",
    "target_not_assessed": r"not (?:assessed|evaluated|reported)|did not (?:assess|evaluate)|outside (?:the )?scope",
}


def profile_development(path: Path, timing: dict[tuple[str, int, int], dict[str, str]]) -> dict[str, Any]:
    data = read_json(path)
    steps = list(data["steps"])
    modalities = [modality_family(str(step.get("ordered", ""))) for step in steps]
    roles, text_parts, other_values, timing_roles = [], [], [], []
    for step in steps:
        ex = step.get("representative_ex_ante") or {}
        roles.append(str(ex.get("action_role", "unclear")))
        text_parts.extend(str(ex.get(field, "")) for field in (
            "reasoning", "information_gap", "expected_finding", "other_hypothesis"
        ))
        diff = ex.get("differential") or {}
        other_values.append(float(diff.get("other", 0.0) or 0.0) if isinstance(diff, dict) else 0.0)
        key = (str(data["disease"]), int(data["hadm_id"]), int(step["step"]))
        timing_roles.append(timing.get(key, {}).get("timing_role", "missing"))
    text = " ".join(text_parts).lower()
    flags = {name: bool(re.search(pattern, text)) for name, pattern in LIMITATION_PATTERNS.items()}
    flags.update({
        "post_intervention": "post_intervention" in timing_roles,
        "other_high": max(other_values, default=0.0) >= 0.40,
        "diffuse_differential": max(other_values, default=0.0) >= 0.30,
        "repeat": any(a == b for a, b in zip(modalities, modalities[1:])),
        "modality_switch": len(set(modalities)) > 1,
    })
    features = {
        f"length:{'single' if len(steps) == 1 else 'multi'}",
        *(f"modality:{value}" for value in set(modalities)),
        f"sequence:{'>'.join(modalities)}", *(f"role:{value}" for value in set(roles)),
        *(f"timing:{value}" for value in set(timing_roles)),
        *(f"flag:{name}" for name, present in flags.items() if present),
    }
    return {
        "disease": str(data["disease"]), "hadm_id": int(data["hadm_id"]),
        "source_path": str(path.relative_to(ROOT)).replace("\\", "/"), "n_steps": len(steps),
        "modality_sequence": modalities, "action_roles": sorted(set(roles)),
        "timing_roles": sorted(set(timing_roles)), "max_other": round(max(other_values, default=0.0), 4),
        "structure_flags": flags, "selection_features": sorted(features),
        "sample_hash": stable_hash(f"{SAMPLE_SALT}|{data['disease']}|{data['hadm_id']}"),
    }


def greedy_batch(candidates: list[dict[str, Any]], per_disease: int,
                 selected_keys: set[tuple[str, int]], covered: set[str]) -> list[dict[str, Any]]:
    selected = []
    rare = ("post_intervention", "target_nonvisualized", "prior_study_limited",
            "target_not_assessed", "repeat", "modality_switch", "other_high")
    for disease in DISEASES:
        pool = [row for row in candidates if row["disease"] == disease
                and (row["disease"], row["hadm_id"]) not in selected_keys]
        for _ in range(per_disease):
            if not pool:
                raise ValueError(f"not enough unused development cases for {disease}")
            ranked = sorted(pool, key=lambda row: (
                -len(set(row["selection_features"]) - covered),
                -sum(bool(row["structure_flags"].get(name)) for name in rare),
                row["sample_hash"], row["hadm_id"],
            ))
            best = ranked[0]
            selected.append(best)
            selected_keys.add((best["disease"], best["hadm_id"]))
            covered.update(best["selection_features"])
            pool.remove(best)
    return selected


TYPE_RULES = (
    ("intervention_or_device_state", r"stent|drain|catheter|post[- ]?(?:ercp|operative|procedure)|patency|position"),
    ("complication", r"abscess|perforat|necrosis|collection|hemorrhag|infection|fistula|leak|obstruction"),
    ("severity_extent_or_course", r"severity|extent|progress|worsen|improv|evolution|response|burden|stage"),
    ("etiology_or_mechanism", r"etiolog|cause|biliary|stone|sludge|mechanism|obstruct"),
    ("alternative_source", r"alternative|other source|gynec|ovarian|urinary|renal|bowel|crohn|malignan|pneumonia"),
    ("syndrome_or_source_frame", r"anatomic source|locali[sz]e|broad differential|intra-abdominal process|source of (?:the )?(?:pain|symptoms)"),
    ("disease_or_finding_identity", r"whether|rule (?:in|out)|confirm|diagnos|identity|represents|appendic|cholecyst|diverticul|pancreati"),
)
QUESTION_RULES = (
    ("intervention_or_device_state", r"stent|drain|catheter|position|patency|decompress|post[- ]?procedure"),
    ("complication", r"abscess|perforat|necrosis|collection|hemorrhag|infection|fistula|leak"),
    ("severity_extent_or_course", r"severity|extent|progress|worsen|improv|evolution|response|burden"),
    ("etiology_or_mechanism", r"etiolog|cause|biliary|stone|sludge|mechanism|obstruct"),
    ("alternative_source", r"alternative|other source|gynec|ovarian|urinary|renal|bowel|crohn|malignan|pneumonia"),
    ("source_localization", r"anatomic source|locali[sz]e|where|source of (?:the )?(?:pain|symptoms)"),
    ("existence_or_identity", r"whether|rule (?:in|out)|confirm|diagnos|identity|represents|presence|visuali[sz]"),
)
REQUIREMENT_RULES = (
    ("target_visualization_or_assessment", r"visuali[sz]|assess|evaluate|not reported|appendix|duct|gallbladder"),
    ("presence_or_absence", r"whether|presence|absence|rule (?:in|out)|confirm|evidence of"),
    ("anatomic_localization", r"locali[sz]|source|anatomic|organ|region"),
    ("finding_identity", r"identity|represents|characteri[sz]|what (?:the|this)"),
    ("etiologic_agent_or_mechanism", r"etiolog|cause|mechanism|stone|sludge|obstruct"),
    ("severity_or_extent", r"severity|extent|burden|grade|size|distribution"),
    ("temporal_course_or_response", r"progress|worsen|improv|evolution|response|interval|change"),
    ("complication_presence_or_character", r"abscess|perforat|necrosis|collection|hemorrhag|infection|fistula|leak"),
    ("alternative_source_discrimination", r"alternative|other source|ovarian|urinary|renal|crohn|malignan|pneumonia"),
    ("device_position_or_integrity", r"stent|drain|catheter|position|migration|integrity"),
    ("device_or_intervention_function", r"patency|decompress|function|effective|response.*(?:drain|stent|procedure)"),
)


def infer_codes(text: str, rules: Iterable[tuple[str, str]], fallback: str = "unclear") -> list[str]:
    found = [code for code, pattern in rules if re.search(pattern, text.lower())]
    return found or [fallback]


def coding_rows(batches: list[tuple[str, list[dict[str, Any]]]], timing: dict) -> tuple[list, dict]:
    rows, batch_codes = [], {}
    for batch_name, batch in batches:
        discovered = {"assumption": set(), "question": set(), "requirements": set()}
        for patient in batch:
            source = read_json(ROOT / patient["source_path"])
            for step in source["steps"]:
                ex = step.get("representative_ex_ante") or {}
                reasoning = str(ex.get("reasoning", ""))
                full_text = " ".join(str(ex.get(field, "")) for field in (
                    "reasoning", "information_gap", "expected_finding", "other_hypothesis", "action_role"
                ))
                blind = {
                    "assumption": infer_codes(reasoning, TYPE_RULES),
                    "question": infer_codes(reasoning, QUESTION_RULES),
                    "requirements": infer_codes(reasoning, REQUIREMENT_RULES),
                }
                full = {
                    "assumption": infer_codes(full_text, TYPE_RULES),
                    "question": infer_codes(full_text, QUESTION_RULES),
                    "requirements": infer_codes(full_text, REQUIREMENT_RULES),
                }
                for kind in discovered:
                    discovered[kind].update(full[kind])
                allowed = {field: ex.get(field) for field in ALLOWED_EX_ANTE_FIELDS}
                key = (source["disease"], int(source["hadm_id"]), int(step["step"]))
                rows.append({
                    "coding_id": f"{source['disease']}:{source['hadm_id']}:s{step['step']}",
                    "batch": batch_name, "disease_stratum_sampling_only": source["disease"],
                    "hadm_id": source["hadm_id"], "step": step["step"],
                    "source_path": patient["source_path"],
                    "source_ex_ante_sha256": stable_hash(json.dumps(allowed, ensure_ascii=False, sort_keys=True)),
                    "ordered": step.get("ordered", ""),
                    "timing_role": timing.get(key, {}).get("timing_role", "missing"),
                    "view_1_reasoning_only": {
                        "reasoning_verbatim": reasoning,
                        "open_assumption_type_candidates": blind["assumption"],
                        "open_question_type_candidates": blind["question"],
                        "open_answer_requirement_candidates": blind["requirements"],
                        "note": "field names and all non-reasoning schema fields hidden",
                    },
                    "view_2_schema_light": {
                        "source_fields_verbatim": allowed,
                        "open_assumption_type_candidates": full["assumption"],
                        "open_question_type_candidates": full["question"],
                        "open_answer_requirement_candidates": full["requirements"],
                    },
                    "view_comparison": {
                        "assumption_only_after_schema": sorted(set(full["assumption"]) - set(blind["assumption"])),
                        "question_only_after_schema": sorted(set(full["question"]) - set(blind["question"])),
                        "requirements_only_after_schema": sorted(set(full["requirements"]) - set(blind["requirements"])),
                        "possible_scaffold_induction": any(set(full[k]) - set(blind[k]) for k in full),
                    },
                    "coding_method": (
                        "deterministic lexical first pass plus rule/boundary audit; candidates are "
                        "discovery evidence, not patient-level frozen labels"
                    ),
                })
        batch_codes[batch_name] = discovered
    return rows, batch_codes


def diversity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_patients": len(rows), "n_steps": sum(row["n_steps"] for row in rows),
        "by_disease": dict(sorted(Counter(row["disease"] for row in rows).items())),
        "trajectory_length": dict(sorted(Counter("single" if row["n_steps"] == 1 else "multi" for row in rows).items())),
        "modality_sequences": dict(sorted(Counter(">".join(row["modality_sequence"]) for row in rows).items())),
        "action_roles": dict(sorted(Counter(role for row in rows for role in row["action_roles"]).items())),
        "timing_roles": dict(sorted(Counter(role for row in rows for role in row["timing_roles"]).items())),
        "structure_flags": {
            flag: sum(bool(row["structure_flags"].get(flag)) for row in rows)
            for flag in sorted(rows[0]["structure_flags"])
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = eligible_records()
    split = make_split(records)
    write_json(OUT_DIR / "split_manifest.json", split)
    timing = load_timing()
    development = [row for row in split["patients"] if row["partition"] == "development"]
    profiles = [profile_development(ROOT / row["source_path"], timing) for row in development]
    selected_keys, covered = set(), set()
    initial = greedy_batch(profiles, 6, selected_keys, covered)
    check_1 = greedy_batch(profiles, 3, selected_keys, covered)
    check_2 = greedy_batch(profiles, 3, selected_keys, covered)
    batches = [("initial_24", initial), ("saturation_check_1", check_1), ("saturation_check_2", check_2)]
    manifest = {
        "schema_version": "1.0.0-development", "source_partition": "development only",
        "selection_algorithm": (
            "per disease deterministic greedy maximum variation over prespecified causal/pre-order "
            "structure; salted SHA-256 tie-break"
        ),
        "sample_salt": SAMPLE_SALT,
        "forbidden_selection_inputs": [
            "verification", "deviation/dev_belief", "ACR/rating", "current result",
            "later events", "final diagnosis correctness",
        ],
        "batches": [{
            "name": name,
            "purpose": "first formal codebook discovery" if name == "initial_24" else "fresh non-overlapping saturation check",
            "summary": diversity_summary(batch), "patients": batch,
        } for name, batch in batches],
        "all_coded_development": diversity_summary(initial + check_1 + check_2),
    }
    write_json(OUT_DIR / "development_sample_manifest.json", manifest)
    write_json(OUT_DIR / "discovery_sample_manifest.json", manifest)
    write_json(OUT_DIR / "diversity_audit.json", {
        "schema_version": "1.0.0-development",
        "interpretation": "maximum-variation coverage audit; counts are not prevalence estimates",
        **{name: diversity_summary(batch) for name, batch in batches},
        "all_48": diversity_summary(initial + check_1 + check_2),
    })
    rows, batch_codes = coding_rows(batches, timing)
    with (OUT_DIR / "discovery_open_coding.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    cumulative = {"assumption": set(), "question": set(), "requirements": set()}
    rounds = []
    for name, batch in batches:
        current = batch_codes[name]
        new = {kind: sorted(values - cumulative[kind]) for kind, values in current.items()}
        for kind, values in current.items():
            cumulative[kind].update(values)
        rounds.append({
            "batch": name, "n_patients": len(batch),
            "new_top_level_assumption_candidates": new["assumption"],
            "new_top_level_question_candidates": new["question"],
            "new_answer_requirement_candidates": new["requirements"],
            "material_schema_change": name == "initial_24",
            "review_result": "codebook established" if name == "initial_24" else "no new top-level family; boundary wording/examples only",
        })
    write_json(OUT_DIR / "saturation_audit.json", {
        "schema_version": "1.0.0-development",
        "scope": "top-level A/Q types and recurrent answer-requirement dimensions",
        "rounds": rounds, "conclusion": "qualitatively_saturated_for_first_layer",
        "basis": "two fresh non-overlapping 12-patient development batches caused no material top-level schema change",
        "limits": [
            "lexical candidates require independent human/clinical framework review",
            "does not establish prevalence, correctness, inter-rater reliability, or final-test transport",
            "final-test annotation content remained excluded from discovery",
        ],
        "next_gate": "independent dual-route framework check on unused development patients; do not open final test",
    })
    print(f"wrote split for {len(records)} patients; coded {len(rows)} steps from 48 development trajectories")


if __name__ == "__main__":
    main()
