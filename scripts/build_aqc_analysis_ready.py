#!/usr/bin/env python3
"""Build the canonical effective A/Q/C development analysis layer.

Only development identities from the frozen split are eligible. Original model outputs are read
without modification, and only explicitly accepted final overlays are applied. Dry-run and
intermediate proposed overlays are deliberately excluded.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aqc_validator import VALIDATOR_VERSION as CURRENT_VALIDATOR_VERSION
from scripts.aqc_validator import validate_output


SPLIT = ROOT / "data" / "aqc_development" / "split_manifest.json"
OUT = ROOT / "data" / "aqc_analysis" / "development_v1"
MODEL = "openai/gpt-5.1"

RAW_GLOBS = (
    "results/aqc_direct/development/*/openai__gpt-5.1/*.json",
    "results/aqc_direct/pilot/*/openai__gpt-5.1/*.json",
)

# Order matters only if a future accepted file intentionally patches an earlier one. At present,
# the accepted overlays have disjoint coding IDs except where the later file is already a merged
# replacement. No dry-run or unadjudicated algorithmic proposed overlay belongs here.
FINAL_OVERLAYS = (
    "results/aqc_direct/development/d9f0c01a5056/manual_adjudication.json",
    "results/aqc_direct/development/c7f7ffae2271/manual_adjudication_batch_003.json",
    "results/aqc_direct/development/697923b99721/manual_adjudication_bridge_004.json",
    "results/aqc_direct/development/697923b99721/manual_adjudication_bridge_005.json",
    "results/aqc_direct/development/697923b99721/manual_adjudication_bridge_006.json",
    "results/aqc_direct/development/697923b99721/manual_adjudication_bridge_007.json",
    "results/aqc_direct/development/697923b99721/targeted_proposed_overlay_bridge_009.json",
)

# These machine-readable final audits establish zero current-validator errors and zero flagged
# low-evidence items for their listed steps. Other steps are retained with an explicit not-audited
# status rather than being silently treated as cleared.
FINAL_AUDITS = (
    "results/aqc_direct/development/c7f7ffae2271/batch_003_content_audit_adjudicated.json",
    "results/aqc_direct/development/697923b99721/audit_bridge_004_makeup.json",
    "results/aqc_direct/development/697923b99721/audit_bridge_005_adjudicated.json",
    "results/aqc_direct/development/697923b99721/audit_bridge_006_targeted.json",
    "results/aqc_direct/development/697923b99721/audit_bridge_007_targeted.json",
    "results/aqc_direct/development/697923b99721/audit_bridge_009_targeted.json",
)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def pointer_tokens(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise ValueError(f"JSON Patch path must start with '/': {pointer}")
    return [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]


def patch_parent(document: Any, pointer: str) -> tuple[Any, str]:
    tokens = pointer_tokens(pointer)
    if not tokens:
        raise ValueError("root-level JSON Patch operations are not supported")
    parent = document
    for token in tokens[:-1]:
        parent = parent[int(token)] if isinstance(parent, list) else parent[token]
    return parent, tokens[-1]


def apply_patch(document: dict[str, Any], operations: list[dict[str, Any]]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    for operation in operations:
        op = operation["op"]
        parent, token = patch_parent(result, operation["path"])
        if op == "remove":
            if isinstance(parent, list):
                del parent[int(token)]
            else:
                del parent[token]
        elif op in {"add", "replace"}:
            value = copy.deepcopy(operation["value"])
            if isinstance(parent, list):
                if token == "-":
                    if op != "add":
                        raise ValueError("only add may use the '-' list token")
                    parent.append(value)
                elif op == "add":
                    parent.insert(int(token), value)
                else:
                    parent[int(token)] = value
            else:
                parent[token] = value
        else:
            raise ValueError(f"unsupported JSON Patch operation: {op}")
    return result


def modality_family(order: str) -> str:
    upper = order.upper()
    if "MRCP" in upper:
        return "MRCP"
    if "MRI" in upper:
        return "MRI"
    if "CTU" in upper:
        return "CTU"
    if re.search(r"\bCT\b|COMPUTED TOMOGRAPH", upper):
        return "CT"
    if "ULTRASOUND" in upper or re.search(r"\bUS\b", upper):
        return "US"
    if "HIDA" in upper or "HEPATOBILIARY" in upper:
        return "HIDA"
    if "X-RAY" in upper or "XRAY" in upper or "RADIOGRAPH" in upper:
        return "XRAY"
    return "OTHER"


def contains_value(value: Any, targets: set[str]) -> bool:
    if isinstance(value, str):
        return value in targets
    if isinstance(value, list):
        return any(contains_value(item, targets) for item in value)
    if isinstance(value, dict):
        return any(contains_value(item, targets) for item in value.values())
    return False


def load_split() -> tuple[dict[tuple[str, int], dict[str, Any]], set[tuple[str, int]]]:
    split = read_json(SPLIT)
    development = {
        (row["disease"], int(row["hadm_id"])): row
        for row in split["patients"] if row["partition"] == "development"
    }
    final_test = {
        (row["disease"], int(row["hadm_id"]))
        for row in split["patients"] if row["partition"] == "final_test"
    }
    if len(development) != 235 or sum(row["n_steps"] for row in development.values()) != 433:
        raise AssertionError("frozen development split is not 235 patients / 433 steps")
    return development, final_test


def load_raw(development: dict[tuple[str, int], dict[str, Any]],
             final_test: set[tuple[str, int]]) -> dict[tuple[str, int], tuple[Path, dict[str, Any]]]:
    found: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for pattern in RAW_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            try:
                filename_disease, filename_hadm = path.stem.rsplit("_", 1)
                filename_key = (filename_disease, int(filename_hadm))
            except (ValueError, TypeError) as exc:
                raise AssertionError(f"unrecognized patient annotation filename: {path}") from exc
            # Resolve partition from filename metadata before opening the annotation payload.
            if filename_key in final_test:
                raise AssertionError(f"final-test output discovered without opening it: {filename_key}")
            if filename_key not in development:
                raise AssertionError(f"non-development output discovered without opening it: {filename_key}")
            data = read_json(path)
            if data.get("model") != MODEL:
                continue
            key = (str(data["disease_stratum_sampling_only"]), int(data["hadm_id"]))
            if key != filename_key:
                raise AssertionError(f"filename/payload identity mismatch at {path}: {key}")
            if key in found:
                raise AssertionError(f"duplicate GPT-5.1 patient output: {key}")
            found[key] = (path, data)
    missing = sorted(set(development) - set(found))
    extra = sorted(set(found) - set(development))
    if missing or extra:
        raise AssertionError(f"raw annotation identity mismatch; missing={missing}, extra={extra}")
    return found


def load_overlays() -> tuple[dict[str, tuple[str, dict[str, Any]]], list[dict[str, Any]]]:
    corrections: dict[str, tuple[str, dict[str, Any]]] = {}
    inventory = []
    for item in FINAL_OVERLAYS:
        path = ROOT / item
        data = read_json(path)
        inventory.append({
            "path": item,
            "sha256": sha256_file(path),
            "n_corrected_steps": len(data.get("corrections", [])),
        })
        for correction in data.get("corrections", []):
            coding_id = correction["coding_id"]
            if coding_id in corrections:
                raise AssertionError(f"accepted overlay collision for {coding_id}")
            corrections[coding_id] = (item, correction)
    return corrections, inventory


def load_final_audits() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    steps: dict[str, dict[str, Any]] = {}
    inventory = []
    for item in FINAL_AUDITS:
        path = ROOT / item
        data = read_json(path)
        inventory.append({"path": item, "sha256": sha256_file(path), "n_steps": len(data["steps"])})
        for row in data["steps"]:
            coding_id = row["coding_id"]
            if coding_id in steps:
                raise AssertionError(f"final audit collision for {coding_id}")
            steps[coding_id] = row
    return steps, inventory


def effective_annotation(accepted: dict[str, Any], correction: dict[str, Any] | None) -> dict[str, Any]:
    result = copy.deepcopy(accepted)
    if correction is None:
        return result
    if "operations" in correction:
        result = apply_patch(result, correction["operations"])
    if "replacement_fields" in correction:
        for field, value in correction["replacement_fields"].items():
            result[field] = copy.deepcopy(value)
    return result


def normalized_fit(annotation: dict[str, Any]) -> dict[str, Any]:
    fit = annotation.get("current_order_fit") or {}
    # Do not equate these fields: their meanings changed across schema generations.
    return {
        "test_question_capability": fit.get("test_question_capability"),
        "question_grounding": fit.get("question_grounding"),
        "legacy_intent_support": fit.get("intent_support"),
        "legacy_unsupported_residual": fit.get("unsupported_residual"),
    }


def build() -> None:
    development, final_test = load_split()
    raw = load_raw(development, final_test)
    corrections, overlay_inventory = load_overlays()
    audited_steps, audit_inventory = load_final_audits()

    patients: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    requirements: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    effective_patients: list[dict[str, Any]] = []
    used_corrections: set[str] = set()

    for disease, hadm_id in sorted(development):
        raw_path, record = raw[(disease, hadm_id)]
        patient_id = f"{disease}:{hadm_id}"
        trajectory_id = patient_id
        prompt_hash = record["prompt_version_sha256"]
        schema_version = record["schema_version"]
        validator_version = record.get("validator_version")
        patient_steps = []
        patient_overlays: set[str] = set()
        previous_step_row: dict[str, Any] | None = None

        for source_step in sorted(record["steps"], key=lambda row: int(row["step"])):
            step_index = int(source_step["step"])
            coding_id = f"{disease}:{hadm_id}:s{step_index}"
            step_id = coding_id
            if source_step.get("accepted") is None:
                raise AssertionError(f"missing accepted annotation: {coding_id}")
            if source_step.get("validation_errors"):
                raise AssertionError(f"stored validation errors remain at {coding_id}")

            overlay_item = corrections.get(coding_id)
            overlay_path = overlay_item[0] if overlay_item else None
            correction = overlay_item[1] if overlay_item else None
            annotation = effective_annotation(source_step["accepted"], correction)
            if overlay_item:
                used_corrections.add(coding_id)
                patient_overlays.add(overlay_path)

            validation_method = "stored_version_validation_plus_release_contract"
            if schema_version == "2.0.0-development" and validator_version == CURRENT_VALIDATOR_VERSION:
                effective_errors = validate_output(
                    annotation,
                    is_first_step=step_index == 1,
                    ordered=str(source_step.get("ordered", "")),
                )
                if effective_errors:
                    raise AssertionError(
                        f"effective annotation fails validator {CURRENT_VALIDATOR_VERSION} at "
                        f"{coding_id}: {effective_errors}"
                    )
                validation_method = f"current_validator_{CURRENT_VALIDATOR_VERSION}_plus_release_contract"

            question = annotation.get("current_question") or {}
            coverage = annotation.get("coverage") or {}
            fit = normalized_fit(annotation)
            assumption_types = [row.get("type") for row in annotation.get("assumptions", [])]
            assumption_statuses = [row.get("status") for row in annotation.get("assumptions", [])]
            has_weak_assumption_support = any(
                row.get("support") == "weakly_supported" for row in annotation.get("assumptions", [])
            )
            audit = audited_steps.get(coding_id)
            evidence_qc_status = "not_uniformly_audited"
            low_evidence_items = None
            if audit is not None:
                if audit.get("validation_errors_under_current_validator"):
                    raise AssertionError(f"final audit reports invalid step: {coding_id}")
                low_evidence_items = int(audit.get("low_evidence_items", 0))
                evidence_qc_status = "flagged_in_final_audit" if low_evidence_items else "cleared_in_final_audit"

            step_row = {
                "patient_id": patient_id,
                "trajectory_id": trajectory_id,
                "step_id": step_id,
                "step_index": step_index,
                "disease": disease,
                "hadm_id": hadm_id,
                "ordered": source_step.get("ordered", ""),
                "modality_family": modality_family(str(source_step.get("ordered", ""))),
                "prompt_version_sha256": prompt_hash,
                "schema_version": schema_version,
                "validator_version": validator_version,
                "model": record["model"],
                "raw_annotation_path": relative(raw_path),
                "raw_annotation_sha256": sha256_file(raw_path),
                "applied_overlay_path": overlay_path,
                "is_structurally_valid": True,
                "effective_validation_method": validation_method,
                "evidence_qc_status": evidence_qc_status,
                "low_evidence_items": low_evidence_items,
                "has_unclear_value": contains_value(annotation, {"unclear"}),
                "has_uncertain_value": contains_value(annotation, {"uncertain"}),
                "has_weak_support": contains_value(annotation, {"weakly_supported"}),
                "has_weak_assumption_support": has_weak_assumption_support,
                "assumption_types": assumption_types,
                "assumption_statuses": assumption_statuses,
                "assumption_change": (annotation.get("assumption_change") or {}).get("label"),
                "question_type": question.get("type"),
                "question_target": question.get("target"),
                "question_continuity": annotation.get("question_continuity"),
                "coverage_aggregate": coverage.get("aggregate"),
                "previous_study_adequacy": (annotation.get("previous_order_update") or {}).get("study_adequacy"),
                "previous_test_question_capability": (annotation.get("previous_order_update") or {}).get("test_question_capability"),
                "previous_result_status": (annotation.get("previous_order_update") or {}).get("result_status"),
                "previous_effect_on_question": (annotation.get("previous_order_update") or {}).get("effect_on_previous_question"),
                "discordance": ((annotation.get("previous_order_update") or {}).get("discordance") or {}).get("label"),
                "derived_transition_reference": annotation.get("derived_transition"),
                **fit,
                "effective_annotation": annotation,
            }
            steps.append(step_row)

            question_id = f"{step_id}:q0"
            question_requirements = question.get("answer_requirements") or []
            coverage_by_key = {
                row.get("requirement_key"): row for row in coverage.get("requirements", [])
            }
            if len(coverage_by_key) != len(coverage.get("requirements", [])):
                raise AssertionError(f"duplicate coverage requirement_key at {coding_id}")
            for index, requirement in enumerate(question_requirements, start=1):
                source_key = requirement.get("requirement_key")
                covered = coverage_by_key.get(source_key)
                if covered is None:
                    raise AssertionError(f"question/coverage requirement mismatch at {coding_id}: {source_key}")
                requirements.append({
                    "patient_id": patient_id,
                    "trajectory_id": trajectory_id,
                    "step_id": step_id,
                    "question_id": question_id,
                    "requirement_id": f"{question_id}:r{index:02d}",
                    "requirement_index": index,
                    "source_requirement_key": source_key,
                    "requirement_type": requirement.get("id"),
                    "dimension": requirement.get("dimension"),
                    "coverage_status": covered.get("status"),
                    "coverage_direction": covered.get("direction"),
                    "supporting_evidence": covered.get("supporting_evidence") or [],
                    "prompt_version_sha256": prompt_hash,
                    "schema_version": schema_version,
                    "has_unclear_value": contains_value({"requirement": requirement, "coverage": covered}, {"unclear"}),
                })

            if set(coverage_by_key) != {row.get("requirement_key") for row in question_requirements}:
                raise AssertionError(f"extra coverage requirement at {coding_id}")

            if previous_step_row is not None:
                relation = "repeat" if previous_step_row["modality_family"] == step_row["modality_family"] else "switch"
                transitions.append({
                    "transition_id": f"{previous_step_row['step_id']}->{step_id}",
                    "patient_id": patient_id,
                    "trajectory_id": trajectory_id,
                    "disease": disease,
                    "source_step_id": previous_step_row["step_id"],
                    "target_step_id": step_id,
                    "source_order": previous_step_row["ordered"],
                    "target_order": step_row["ordered"],
                    "source_modality_family": previous_step_row["modality_family"],
                    "target_modality_family": step_row["modality_family"],
                    "observed_action_relation": relation,
                    "source_assumption_types": previous_step_row["assumption_types"],
                    "target_assumption_types": step_row["assumption_types"],
                    "source_question_type": previous_step_row["question_type"],
                    "target_question_type": step_row["question_type"],
                    "source_coverage_aggregate": previous_step_row["coverage_aggregate"],
                    "target_coverage_aggregate": step_row["coverage_aggregate"],
                    "assumption_change": step_row["assumption_change"],
                    "question_continuity": step_row["question_continuity"],
                    "previous_study_adequacy": step_row["previous_study_adequacy"],
                    "previous_test_question_capability": step_row["previous_test_question_capability"],
                    "previous_result_status": step_row["previous_result_status"],
                    "previous_effect_on_question": step_row["previous_effect_on_question"],
                    "discordance": step_row["discordance"],
                    "derived_transition_reference": step_row["derived_transition_reference"],
                    "prompt_version_sha256": prompt_hash,
                    "schema_version": schema_version,
                    "has_unclear_value": previous_step_row["has_unclear_value"] or step_row["has_unclear_value"],
                    "has_weak_support": previous_step_row["has_weak_support"] or step_row["has_weak_support"],
                    "has_weak_assumption_support": (
                        previous_step_row["has_weak_assumption_support"]
                        or step_row["has_weak_assumption_support"]
                    ),
                })
            previous_step_row = step_row
            patient_steps.append({
                "step_id": step_id,
                "step_index": step_index,
                "ordered": source_step.get("ordered", ""),
                "applied_overlay_path": overlay_path,
                "effective_annotation": annotation,
            })

        patients.append({
            "patient_id": patient_id,
            "trajectory_id": trajectory_id,
            "disease": disease,
            "hadm_id": hadm_id,
            "partition": "development",
            "n_steps": len(patient_steps),
            "source_path": development[(disease, hadm_id)]["source_path"],
            "raw_annotation_path": relative(raw_path),
            "raw_annotation_sha256": sha256_file(raw_path),
            "prompt_version_sha256": prompt_hash,
            "schema_version": schema_version,
            "validator_version": validator_version,
            "model": record["model"],
            "applied_overlay_paths": sorted(patient_overlays),
        })
        effective_patients.append({
            "patient_id": patient_id,
            "trajectory_id": trajectory_id,
            "disease": disease,
            "hadm_id": hadm_id,
            "partition": "development",
            "prompt_version_sha256": prompt_hash,
            "schema_version": schema_version,
            "validator_version": validator_version,
            "model": record["model"],
            "raw_annotation_path": relative(raw_path),
            "applied_overlay_paths": sorted(patient_overlays),
            "steps": patient_steps,
        })

    unused = sorted(set(corrections) - used_corrections)
    if unused:
        raise AssertionError(f"accepted corrections do not match the development corpus: {unused}")
    if len(patients) != 235 or len(steps) != 433:
        raise AssertionError(f"effective layer count mismatch: {len(patients)} patients / {len(steps)} steps")
    if len({row["patient_id"] for row in patients}) != len(patients):
        raise AssertionError("patient_id is not unique")
    if len({row["step_id"] for row in steps}) != len(steps):
        raise AssertionError("step_id is not unique")
    if len({row["requirement_id"] for row in requirements}) != len(requirements):
        raise AssertionError("requirement_id is not unique")
    if len({row["transition_id"] for row in transitions}) != len(transitions):
        raise AssertionError("transition_id is not unique")
    expected_transitions = sum(max(0, row["n_steps"] - 1) for row in patients)
    if len(transitions) != expected_transitions:
        raise AssertionError("transition count does not equal sum(n_steps - 1)")

    write_jsonl(OUT / "patients.jsonl", patients)
    write_jsonl(OUT / "steps.jsonl", steps)
    write_jsonl(OUT / "requirements.jsonl", requirements)
    write_jsonl(OUT / "transitions.jsonl", transitions)
    write_jsonl(OUT / "effective_annotations.jsonl", effective_patients)

    def count(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))

    manifest = {
        "release": "aqc-development-v1",
        "partition": "development_only",
        "source_split": relative(SPLIT),
        "counts": {
            "patients": len(patients),
            "steps": len(steps),
            "requirements": len(requirements),
            "transitions": len(transitions),
            "corrected_steps": len(used_corrections),
            "final_test_patients_included": 0,
        },
        "by_disease_patients": count(patients, "disease"),
        "by_disease_steps": count(steps, "disease"),
        "by_prompt_patients": count(patients, "prompt_version_sha256"),
        "by_schema_patients": count(patients, "schema_version"),
        "by_validator_patients": count(patients, "validator_version"),
        "quality": {
            "structurally_invalid_steps": sum(not row["is_structurally_valid"] for row in steps),
            "steps_cleared_in_machine_readable_final_audit": sum(
                row["evidence_qc_status"] == "cleared_in_final_audit" for row in steps
            ),
            "steps_flagged_in_machine_readable_final_audit": sum(
                row["evidence_qc_status"] == "flagged_in_final_audit" for row in steps
            ),
            "steps_without_uniform_machine_readable_final_audit": sum(
                row["evidence_qc_status"] == "not_uniformly_audited" for row in steps
            ),
            "steps_with_unclear_value": sum(row["has_unclear_value"] for row in steps),
            "steps_with_uncertain_value": sum(row["has_uncertain_value"] for row in steps),
            "steps_with_weak_support": sum(row["has_weak_support"] for row in steps),
            "steps_with_weak_assumption_support": sum(
                row["has_weak_assumption_support"] for row in steps
            ),
            "effective_validation_methods": count(steps, "effective_validation_method"),
        },
        "policies": {
            "corrections": (
                "Apply only the accepted overlay allowlist in order; retain original outputs; "
                "do not serialize per-correction rationale into the analysis layer."
            ),
            "invalid": "Fail the build; never silently drop an invalid patient or step.",
            "low_evidence": (
                "Retain and flag. A zero is asserted only when a listed machine-readable final "
                "audit cleared the step; otherwise status is not_uniformly_audited."
            ),
            "unclear_or_weak": (
                "Retain as observed values. Exclusion is pattern-specific and must be paired with "
                "a sensitivity analysis."
            ),
            "schema_compatibility": (
                "Preserve original schema and prompt versions. Never equate legacy intent_support "
                "with question_grounding."
            ),
            "final_test": "Identity metadata is used only to prove exclusion; no final-test clinical content is read.",
        },
        "accepted_overlays": overlay_inventory,
        "final_audits": audit_inventory,
        "outputs": [
            "patients.jsonl", "steps.jsonl", "requirements.jsonl", "transitions.jsonl",
            "effective_annotations.jsonl",
        ],
    }
    write_json(OUT / "manifest.json", manifest)
    print(
        f"wrote {len(patients)} patients, {len(steps)} steps, {len(requirements)} requirements, "
        f"and {len(transitions)} transitions to {relative(OUT)}"
    )


if __name__ == "__main__":
    build()
