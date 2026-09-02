"""Detect and optionally repair local temporal/discordance A/Q/C issues.

Dry-run is the default and never calls an external model.  ``--execute`` sends
only flagged, causally masked subproblems and emits a non-destructive patch
overlay; it never overwrites original annotation files.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.annotation.annotate import call_json  # noqa: E402
from experiments.aqc.targeted_repair_prompts import (  # noqa: E402
    DISCORDANCE_PROMPT_SHA256,
    DISCORDANCE_SYSTEM,
    TEMPORAL_PROMPT_SHA256,
    TEMPORAL_SYSTEM,
    build_discordance_user,
    build_temporal_user,
)
from scripts.aqc_algorithmic_auditor import (  # noqa: E402
    ALGORITHM_VERSION as ALGORITHMIC_AUDITOR_VERSION,
    _load_filtered_record,
    apply_operations,
    audit_and_normalize,
)
from scripts.aqc_validator import VALIDATOR_VERSION, validate_output  # noqa: E402
from scripts.build_masked_view import RAW, load_lab_map  # noqa: E402
from scripts.run_aqc_direct import read_json, slug  # noqa: E402

SCHEMA_VERSION = "1.0.0-aqc-targeted-audit"
AUDITOR_VERSION = "1.0.0-temporal-discordance-targeted-repair"

STRONG_TEMPORAL_PATTERN = re.compile(
    r"\b(?:interval|progression|response to|changed? since|"
    r"since (?:the )?(?:prior|earlier|last)|"
    r"compared (?:with|to) (?:the )?(?:prior|earlier|last))\b",
    re.I,
)

IMAGING_CHANGE_PATTERN = re.compile(
    r"\b(?:finding|abnormalit\w*|collection|lesion|inflammation|fluid|dilat\w*|"
    r"size|extent|disease|condition)\b.{0,50}\b(?:improv\w*|worsen\w*|"
    r"progress\w*|evolv\w*|remain\w* stable)\b|"
    r"\b(?:improv\w*|worsen\w*|progress\w*|evolv\w*|remain\w* stable)\b.{0,50}"
    r"\b(?:finding|abnormalit\w*|collection|lesion|inflammation|fluid|dilat\w*|"
    r"size|extent|disease|condition)\b",
    re.I,
)

SUSPICIOUS_DISCORDANCE = re.compile(
    r"\b(?:can coexist|compatible|different technique|improved technique|"
    r"nonvisuali[sz]|limited study|markedly limited|not true discordance|"
    r"rather than (?:a |true )?discordance|distinct processes?)\b",
    re.I,
)


def temporal_candidate(annotation: dict[str, Any]) -> dict[str, Any] | None:
    question = annotation.get("current_question")
    if not isinstance(question, dict):
        return None
    text = " ".join([
        str(question.get("primary") or ""),
        *[str(item) for item in question.get("secondary_questions") or []],
    ])
    requirements = question.get("answer_requirements") or []
    has_temporal = any(
        isinstance(item, dict) and item.get("id") == "temporal_course_or_response"
        for item in requirements
    )
    explicit_temporal = bool(
        STRONG_TEMPORAL_PATTERN.search(text) or IMAGING_CHANGE_PATTERN.search(text)
    )
    if explicit_temporal and not has_temporal:
        return {"kind": "missing_temporal_requirement", "matched_text": text}
    if has_temporal and not explicit_temporal:
        return {"kind": "possibly_unnecessary_temporal_requirement", "matched_text": text}
    return None


def discordance_candidate(annotation: dict[str, Any], is_first_step: bool) -> dict[str, Any] | None:
    if is_first_step:
        return None
    previous = annotation.get("previous_order_update")
    discordance = previous.get("discordance") if isinstance(previous, dict) else None
    if not isinstance(discordance, dict):
        return None
    label = discordance.get("label")
    reason = str(discordance.get("reason") or "")
    if label == "indeterminate":
        return {"kind": "indeterminate_discordance_review", "reason": reason}
    if label == "materially_discordant" and SUSPICIOUS_DISCORDANCE.search(reason):
        return {"kind": "possibly_nonconflicting_streams", "reason": reason}
    if label == "concordant" and re.search(r"\b(?:conflict|oppos|discordant|inconsistent)\b", reason, re.I):
        return {"kind": "possibly_missed_discordance", "reason": reason}
    return None


def _temporal_operations(
    annotation: dict[str, Any], response: Any
) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(response, dict):
        return [], ["targeted_temporal_not_object"]
    decision = response.get("decision")
    if decision not in {"aligned", "add_temporal_requirement", "remove_temporal_wording", "unclear"}:
        return [], ["targeted_temporal_bad_decision"]
    if decision in {"aligned", "unclear"}:
        return [], []
    operations: list[dict[str, Any]] = []
    primary = response.get("revised_primary")
    secondary = response.get("revised_secondary_questions")
    if isinstance(primary, str) and primary.strip():
        operations.append({"op": "replace", "path": "/current_question/primary", "value": primary})
    if isinstance(secondary, list) and all(isinstance(item, str) and item.strip() for item in secondary):
        operations.append({
            "op": "replace", "path": "/current_question/secondary_questions", "value": secondary
        })
    if decision == "remove_temporal_wording":
        return (operations, []) if operations else ([], ["temporal_remove_without_revised_question"])

    requirement = response.get("temporal_requirement")
    temporal_coverage = response.get("temporal_coverage")
    if not isinstance(requirement, dict) or not isinstance(temporal_coverage, dict):
        return [], ["temporal_add_without_requirement_and_coverage"]
    key = requirement.get("requirement_key")
    if (
        not isinstance(key, str) or not key.strip()
        or requirement.get("id") != "temporal_course_or_response"
        or temporal_coverage.get("requirement_key") != key
        or temporal_coverage.get("requirement_id") != "temporal_course_or_response"
    ):
        return [], ["temporal_add_bad_requirement_binding"]
    question = annotation["current_question"]
    if key in {item.get("requirement_key") for item in question["answer_requirements"] if isinstance(item, dict)}:
        return [], ["temporal_add_duplicate_requirement_key"]
    requirements = [*question["answer_requirements"], requirement]
    coverage_requirements = [*annotation["coverage"]["requirements"], temporal_coverage]
    operations.extend([
        {"op": "replace", "path": "/current_question/answer_requirements", "value": requirements},
        {"op": "replace", "path": "/coverage/requirements", "value": coverage_requirements},
    ])
    aggregate = response.get("revised_aggregate")
    if aggregate in {"unanswered", "partially_answered", "sufficiently_answered"}:
        operations.append({"op": "replace", "path": "/coverage/aggregate", "value": aggregate})
    aggregate_reason = response.get("revised_aggregate_reason")
    if isinstance(aggregate_reason, str) and aggregate_reason.strip():
        operations.append({
            "op": "replace", "path": "/coverage/aggregate_reason", "value": aggregate_reason
        })
    return operations, []


def _discordance_operations(response: Any) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(response, dict):
        return [], ["targeted_discordance_not_object"]
    decision = response.get("decision")
    if decision not in {"aligned", "revise", "unclear"}:
        return [], ["targeted_discordance_bad_decision"]
    if decision in {"aligned", "unclear"}:
        return [], []
    discordance = response.get("discordance")
    if not isinstance(discordance, dict) or set(discordance) != {
        "label", "evidence_stream_1", "evidence_stream_2", "reason"
    }:
        return [], ["targeted_discordance_bad_replacement"]
    return [{
        "op": "replace",
        "path": "/previous_order_update/discordance",
        "value": discordance,
    }], []


def _newly_visible_imaging(
    decision_points: list[dict[str, Any]], step_index: int
) -> list[dict[str, Any]]:
    current = decision_points[step_index].get("visible_prior_imaging") or []
    if step_index == 0:
        return current
    previous = decision_points[step_index - 1].get("visible_prior_imaging") or []
    return current[len(previous):]


def audit_manifest(
    manifest_path: Path,
    prompt_hash: str,
    model: str,
    *,
    execute: bool,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    frames = {disease: pd.read_csv(ROOT / path) for disease, path in RAW.items()}
    labmap = load_lab_map()
    result_dir = ROOT / "results" / "aqc_direct" / "development" / prompt_hash[:12] / slug(model)
    rows: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    for patient in manifest["patients"]:
        disease = str(patient["disease"])
        hadm_id = int(patient["hadm_id"])
        result = read_json(result_dir / f"{disease}_{hadm_id}.json")
        record = _load_filtered_record(result, patient, frames, labmap)
        seen_orders: set[str] = set()
        for step_index, (stored_step, decision_point) in enumerate(
            zip(result["steps"], record["decision_points"], strict=True)
        ):
            accepted = stored_step.get("accepted")
            if not isinstance(accepted, dict):
                continue
            ordered = str(decision_point["ordered"])
            is_first = int(stored_step["step"]) == 1
            algorithmic = audit_and_normalize(
                accepted,
                record=record,
                decision_point=decision_point,
                is_first_step=is_first,
                is_repeat_order=ordered in seen_orders,
            )
            candidate = algorithmic["repaired"]
            temporal = temporal_candidate(candidate)
            discordance = discordance_candidate(candidate, is_first)
            if not temporal and not discordance:
                seen_orders.add(ordered)
                if algorithmic["operations"]:
                    coding_id = f"{disease}:{hadm_id}:s{stored_step['step']}"
                    corrections.append({
                        "coding_id": coding_id,
                        "reason": "Deterministic algorithmic normalization before targeted audit.",
                        "operations": algorithmic["operations"],
                    })
                continue
            coding_id = f"{disease}:{hadm_id}:s{stored_step['step']}"
            row: dict[str, Any] = {
                "coding_id": coding_id,
                "algorithmic_operations": algorithmic["operations"],
                "temporal_candidate": temporal,
                "discordance_candidate": discordance,
                "temporal_prompt_sha256": TEMPORAL_PROMPT_SHA256 if temporal else None,
                "discordance_prompt_sha256": DISCORDANCE_PROMPT_SHA256 if discordance else None,
                "temporal_user": build_temporal_user(record, decision_point, candidate) if temporal else None,
                "discordance_user": build_discordance_user(
                    record,
                    decision_point,
                    candidate,
                    _newly_visible_imaging(record["decision_points"], step_index),
                ) if discordance else None,
                "targeted_calls": [],
                "targeted_operations": [],
                "repair_errors": [],
            }
            targeted_operations: list[dict[str, Any]] = []
            if execute and temporal:
                call = call_json(
                    TEMPORAL_SYSTEM,
                    row["temporal_user"],
                    model=model,
                    temperature=0.0,
                    max_tokens=1400,
                )
                operations, errors = _temporal_operations(candidate, call.get("parsed"))
                row["targeted_calls"].append({"kind": "temporal", "call": call})
                row["repair_errors"].extend(errors)
                if not errors:
                    candidate = apply_operations(candidate, operations)
                    targeted_operations.extend(operations)
            if execute and discordance:
                call = call_json(
                    DISCORDANCE_SYSTEM,
                    row["discordance_user"],
                    model=model,
                    temperature=0.0,
                    max_tokens=900,
                )
                operations, errors = _discordance_operations(call.get("parsed"))
                row["targeted_calls"].append({"kind": "discordance", "call": call})
                row["repair_errors"].extend(errors)
                if not errors:
                    candidate = apply_operations(candidate, operations)
                    targeted_operations.extend(operations)
            post_target_audit = audit_and_normalize(
                candidate,
                record=record,
                decision_point=decision_point,
                is_first_step=is_first,
                is_repeat_order=ordered in seen_orders,
            )
            if execute and post_target_audit["operations"]:
                candidate = post_target_audit["repaired"]
                targeted_operations.extend(post_target_audit["operations"])
            pre_unresolved = {
                issue["path"] for issue in algorithmic["issues"]
                if issue.get("kind") == "nonverbatim_unresolved"
            }
            post_unresolved = {
                issue["path"] for issue in post_target_audit["issues"]
                if issue.get("kind") == "nonverbatim_unresolved"
            }
            new_unresolved = sorted(post_unresolved - pre_unresolved)
            if execute and new_unresolved:
                row["repair_errors"].append("targeted_repair_introduced_nonverbatim_evidence")
            row["post_target_algorithmic_issues"] = post_target_audit["issues"]
            row["new_unresolved_evidence_paths"] = new_unresolved
            final_errors = validate_output(
                candidate,
                is_first_step=is_first,
                ordered=ordered,
                is_repeat_order=ordered in seen_orders,
            )
            row["targeted_operations"] = targeted_operations
            row["validation_errors_after_targeted"] = final_errors
            all_operations = [*algorithmic["operations"], *targeted_operations]
            if all_operations and not final_errors and not row["repair_errors"]:
                corrections.append({
                    "coding_id": coding_id,
                    "reason": "Algorithmic normalization plus validated targeted semantic repair.",
                    "operations": all_operations,
                })
            rows.append(row)
            seen_orders.add(ordered)
    return {
        "schema_version": SCHEMA_VERSION,
        "auditor_version": AUDITOR_VERSION,
        "algorithmic_auditor_version": ALGORITHMIC_AUDITOR_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "manifest": str(manifest_path),
        "model": model,
        "execute": execute,
        "temporal_prompt_sha256": TEMPORAL_PROMPT_SHA256,
        "discordance_prompt_sha256": DISCORDANCE_PROMPT_SHA256,
        "n_targeted_steps": len(rows),
        "n_temporal_candidates": sum(bool(row["temporal_candidate"]) for row in rows),
        "n_discordance_candidates": sum(bool(row["discordance_candidate"]) for row in rows),
        "n_targeted_calls": sum(len(row["targeted_calls"]) for row in rows),
        "n_repaired_steps": sum(bool(row["targeted_operations"]) and not row["validation_errors_after_targeted"] for row in rows),
        "rows": rows,
        "adjudication": {
            "schema_version": "1.0.0-manual-adjudication",
            "source": "algorithmic-plus-targeted-llm" if execute else "algorithmic-dry-run",
            "source_prompt_hash": prompt_hash,
            "validator_version": VALIDATOR_VERSION,
            "model": model,
            "manifest": str(manifest_path),
            "corrections": corrections,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt-hash", required=True)
    parser.add_argument("--model", default="openai/gpt-5.1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--adjudication-output", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="send only flagged subproblems after explicit external-transmission authorization",
    )
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    report = audit_manifest(
        manifest_path,
        args.prompt_hash,
        args.model,
        execute=args.execute,
    )
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.adjudication_output:
        output = (
            args.adjudication_output
            if args.adjudication_output.is_absolute()
            else ROOT / args.adjudication_output
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report["adjudication"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = {key: value for key, value in report.items() if key not in {"rows", "adjudication"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.execute:
        print("dry run only; --execute requires explicit authorization for this targeted payload")


if __name__ == "__main__":
    main()
