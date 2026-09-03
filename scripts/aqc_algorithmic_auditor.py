"""Clinical-only, deterministic A/Q/C evidence auditor and normalizer.

The auditor never edits model result files.  It emits RFC-6902-style patch
operations plus provenance.  Only uniquely determined mechanical corrections
are proposed; unresolved evidence remains flagged for review.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aqc_validator import VALIDATOR_VERSION, validate_output  # noqa: E402
from scripts.audit_aqc_input_leakage import (  # noqa: E402
    apply_reviewed_redactions,
    load_reviews,
)
from scripts.build_masked_view import RAW, build_record, load_lab_map  # noqa: E402
from scripts.run_aqc_direct import read_json, slug  # noqa: E402

SCHEMA_VERSION = "1.0.0-aqc-algorithmic-audit"
ALGORITHM_VERSION = "1.0.0-clinical-only-evidence-normalizer"

FORBIDDEN_EVIDENCE = re.compile(
    r"(?:^|\n)\s*#{1,6}\s|"
    r"imaging resulted before this decision point|"
    r"output template|answer_requirements|"
    r"^\s*\(?none\)?\s*$",
    re.I,
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def lexical_normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalize(text)))


def token_count(text: str) -> int:
    return len(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalize(text)))


def clinical_blocks(record: dict[str, Any], decision_point: dict[str, Any]) -> list[str]:
    baseline = record["baseline"]
    return [
        str(baseline.get("patient_history") or ""),
        str(baseline.get("physical_examination") or ""),
        str(baseline.get("laboratory_tests") or ""),
        *[
            str(item.get("report") or "")
            for item in decision_point.get("visible_prior_imaging") or []
        ],
    ]


def evidence_units(blocks: list[str]) -> list[str]:
    units: dict[str, str] = {}
    for block in blocks:
        for line in re.split(r"[\r\n]+", block):
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue
            candidates = [line, *re.split(r"(?<=[.!?])\s+", line)]
            for candidate in candidates:
                candidate = candidate.strip()
                if token_count(candidate) >= 3:
                    units.setdefault(normalize(candidate), candidate)
    return list(units.values())


def evidence_locations(value: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, assumption in enumerate(value.get("assumptions") or []):
        if isinstance(assumption, dict):
            for quote_index, quote in enumerate(assumption.get("evidence") or []):
                rows.append({
                    "path": f"/assumptions/{index}/evidence/{quote_index}",
                    "quote": quote,
                    "container": "list",
                })
    question = value.get("current_question") or {}
    if isinstance(question, dict):
        for quote_index, quote in enumerate(question.get("evidence") or []):
            rows.append({
                "path": f"/current_question/evidence/{quote_index}",
                "quote": quote,
                "container": "list",
            })
    coverage = value.get("coverage") or {}
    if isinstance(coverage, dict):
        for index, requirement in enumerate(coverage.get("requirements") or []):
            if isinstance(requirement, dict):
                for quote_index, quote in enumerate(requirement.get("supporting_evidence") or []):
                    rows.append({
                        "path": f"/coverage/requirements/{index}/supporting_evidence/{quote_index}",
                        "quote": quote,
                        "container": "list",
                    })
    previous = value.get("previous_order_update") or {}
    discordance = previous.get("discordance") if isinstance(previous, dict) else None
    if isinstance(discordance, dict):
        for field in ("evidence_stream_1", "evidence_stream_2"):
            quote = discordance.get(field)
            if isinstance(quote, str) and quote:
                rows.append({
                    "path": f"/previous_order_update/discordance/{field}",
                    "quote": quote,
                    "container": "scalar",
                })
    return rows


def quote_is_verbatim(quote: str, blocks: list[str]) -> bool:
    normalized_blocks = [normalize(block) for block in blocks]
    whole = normalize(quote).strip(' "“”‘’')
    if whole and any(whole in block for block in normalized_blocks):
        return True
    # A single evidence string may join or reorder several verbatim source
    # sentences/lab lines. Every segment must still occur literally after
    # punctuation/whitespace normalization; semantic similarity is not enough.
    lexical_blocks = [lexical_normalize(block) for block in blocks]
    segments = [
        lexical_normalize(part)
        for part in re.split(r"\.{3,}|…|(?<=[.!?])\s+|[;\r\n]+", quote)
        if token_count(part) >= 2
    ]
    return bool(segments) and all(any(segment in block for block in lexical_blocks) for segment in segments)


def unique_embedded_source(quote: str, units: list[str]) -> str | None:
    normalized_quote = lexical_normalize(quote)
    candidates = [
        unit for unit in units
        if lexical_normalize(unit) in normalized_quote
        and token_count(unit) >= 4
        # Do not silently discard a second, potentially legitimate excerpt.
        and token_count(unit) / max(token_count(quote), 1) >= 0.70
    ]
    if not candidates:
        return None
    longest = max(token_count(candidate) for candidate in candidates)
    best = {lexical_normalize(candidate): candidate for candidate in candidates if token_count(candidate) == longest}
    return next(iter(best.values())) if len(best) == 1 else None


def ordered_operations(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Descending list indices prevent earlier removals from shifting later paths.
    def sort_key(operation: dict[str, Any]) -> tuple[str, int]:
        path = str(operation["path"])
        parent, _, leaf = path.rpartition("/")
        return parent, -(int(leaf) if leaf.isdigit() else -1)

    return sorted(operations, key=sort_key)


def apply_operations(value: dict[str, Any], operations: list[dict[str, Any]]) -> dict[str, Any]:
    result = copy.deepcopy(value)

    for operation in ordered_operations(operations):
        parts = [part.replace("~1", "/").replace("~0", "~") for part in operation["path"].split("/")[1:]]
        parent: Any = result
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        leaf = parts[-1]
        if isinstance(parent, list):
            if operation["op"] == "remove":
                parent.pop(int(leaf))
            elif operation["op"] == "add":
                if leaf == "-":
                    parent.append(copy.deepcopy(operation.get("value")))
                else:
                    parent.insert(int(leaf), copy.deepcopy(operation.get("value")))
            else:
                parent[int(leaf)] = copy.deepcopy(operation.get("value"))
        elif operation["op"] == "remove":
            del parent[leaf]
        else:
            parent[leaf] = copy.deepcopy(operation.get("value"))
    return result


def audit_and_normalize(
    value: dict[str, Any],
    *,
    record: dict[str, Any],
    decision_point: dict[str, Any],
    is_first_step: bool,
    is_repeat_order: bool,
) -> dict[str, Any]:
    blocks = clinical_blocks(record, decision_point)
    units = evidence_units(blocks)
    issues: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    for location in evidence_locations(value):
        quote = location["quote"]
        path = location["path"]
        if not isinstance(quote, str) or not quote.strip():
            issues.append({"path": path, "kind": "empty_evidence_item", "quote": quote})
            if location["container"] == "list":
                operations.append({"op": "remove", "path": path})
            continue
        if FORBIDDEN_EVIDENCE.search(quote):
            issues.append({"path": path, "kind": "prompt_scaffolding", "quote": quote})
            operation = {"op": "remove", "path": path} if location["container"] == "list" else {
                "op": "replace", "path": path, "value": ""
            }
            operations.append(operation)
            continue
        if quote_is_verbatim(quote, blocks):
            continue
        replacement = unique_embedded_source(quote, units)
        if replacement is not None:
            issues.append({
                "path": path,
                "kind": "explanatory_wrapper_removed",
                "quote": quote,
                "replacement": replacement,
            })
            operations.append({"op": "replace", "path": path, "value": replacement})
        else:
            issues.append({"path": path, "kind": "nonverbatim_unresolved", "quote": quote})

    # Removing a forbidden/empty item can leave an assumption with no evidence.
    # When the output has more than the required three assumptions, the only
    # deterministic repair is to remove that now-unsupported assumption rather
    # than emit an invalid empty evidence list.  If removal would violate the
    # minimum assumption count, retain the original item and leave the issue for
    # adjudication instead of manufacturing a validator failure.
    assumptions = value.get("assumptions")
    if isinstance(assumptions, list):
        provisional = apply_operations(value, operations)
        empty_indices = [
            index
            for index, assumption in enumerate(provisional.get("assumptions") or [])
            if isinstance(assumption, dict) and not (assumption.get("evidence") or [])
        ]
        removable = max(0, len(assumptions) - 3)
        for index in reversed(empty_indices):
            prefix = f"/assumptions/{index}/"
            operations = [
                operation for operation in operations
                if not str(operation["path"]).startswith(prefix)
            ]
            if removable:
                operations.append({"op": "remove", "path": f"/assumptions/{index}"})
                issues.append({
                    "path": f"/assumptions/{index}",
                    "kind": "unsupported_assumption_removed",
                })
                removable -= 1
            else:
                issues.append({
                    "path": f"/assumptions/{index}",
                    "kind": "unsupported_assumption_unresolved_minimum_count",
                })

    previous = value.get("previous_order_update")
    if is_first_step and isinstance(previous, dict):
        for field in ("study_adequacy", "test_question_capability", "result_status"):
            if previous.get(field) != "not_applicable":
                operations.append({
                    "op": "replace",
                    "path": f"/previous_order_update/{field}",
                    "value": "not_applicable",
                })
                issues.append({
                    "path": f"/previous_order_update/{field}",
                    "kind": "first_step_previous_field_normalized",
                })
        discordance = previous.get("discordance")
        if isinstance(discordance, dict) and discordance.get("label") != "not_applicable":
            operations.extend([
                {"op": "replace", "path": "/previous_order_update/discordance/label", "value": "not_applicable"},
                {"op": "replace", "path": "/previous_order_update/discordance/evidence_stream_1", "value": ""},
                {"op": "replace", "path": "/previous_order_update/discordance/evidence_stream_2", "value": ""},
            ])
            issues.append({
                "path": "/previous_order_update/discordance",
                "kind": "first_step_discordance_normalized",
            })

    coverage = value.get("coverage")
    if isinstance(coverage, dict):
        for index, item in enumerate(coverage.get("requirements") or []):
            if not isinstance(item, dict) or item.get("status") != "unaddressed":
                continue
            direction = item.get("direction")
            if direction not in {"supports", "refutes", "mixed"}:
                continue
            evidence = [quote for quote in item.get("supporting_evidence") or [] if isinstance(quote, str) and quote.strip()]
            if evidence:
                operations.append({
                    "op": "replace",
                    "path": f"/coverage/requirements/{index}/status",
                    "value": "partially_addressed",
                })
                repair = "status_to_partially_addressed"
            else:
                operations.append({
                    "op": "replace",
                    "path": f"/coverage/requirements/{index}/direction",
                    "value": "no_direction",
                })
                repair = "direction_to_no_direction"
            issues.append({
                "path": f"/coverage/requirements/{index}",
                "kind": "coverage_status_direction_normalized",
                "repair": repair,
            })

    operations = ordered_operations(operations)
    repaired = apply_operations(value, operations)
    before = validate_output(
        value,
        is_first_step=is_first_step,
        ordered=str(decision_point.get("ordered") or ""),
        is_repeat_order=is_repeat_order,
    )
    after = validate_output(
        repaired,
        is_first_step=is_first_step,
        ordered=str(decision_point.get("ordered") or ""),
        is_repeat_order=is_repeat_order,
    )
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "issues": issues,
        "operations": operations,
        "validation_errors_before": before,
        "validation_errors_after": after,
        "repaired": repaired,
    }


def _load_filtered_record(
    result: dict[str, Any], patient: dict[str, Any], frames: dict[str, pd.DataFrame], labmap: dict
) -> dict[str, Any]:
    disease = str(patient["disease"])
    hadm_id = int(patient["hadm_id"])
    row = frames[disease][frames[disease]["hadm_id"] == hadm_id]
    if row.empty:
        raise ValueError(f"raw record not found: {disease}/{hadm_id}")
    record = build_record(disease, hadm_id, row.iloc[0], labmap)
    review_path = Path(str(result.get("causal_mask_review") or ""))
    if not review_path.is_absolute():
        review_path = ROOT / review_path
    if hashlib.sha256(review_path.read_bytes()).hexdigest() != result.get("causal_mask_review_sha256"):
        raise ValueError(f"causal-mask review hash mismatch: {disease}/{hadm_id}")
    reviews = load_reviews(review_path)
    filtered, applied = apply_reviewed_redactions(
        record["baseline"]["patient_history"], disease, hadm_id, reviews
    )
    if applied != result.get("applied_hpi_redactions"):
        raise ValueError(f"redaction provenance mismatch: {disease}/{hadm_id}")
    record["baseline"]["patient_history"] = filtered
    return record


def audit_manifest(manifest_path: Path, prompt_hash: str, model: str) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    frames = {disease: pd.read_csv(ROOT / path) for disease, path in RAW.items()}
    labmap = load_lab_map()
    result_dir = ROOT / "results" / "aqc_direct" / "development" / prompt_hash[:12] / slug(model)
    steps: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    for patient in manifest["patients"]:
        disease = str(patient["disease"])
        hadm_id = int(patient["hadm_id"])
        result = read_json(result_dir / f"{disease}_{hadm_id}.json")
        record = _load_filtered_record(result, patient, frames, labmap)
        seen_orders: set[str] = set()
        for stored_step, decision_point in zip(result["steps"], record["decision_points"], strict=True):
            accepted = stored_step.get("accepted")
            if not isinstance(accepted, dict):
                continue
            coding_id = f"{disease}:{hadm_id}:s{stored_step['step']}"
            ordered = str(decision_point["ordered"])
            audit = audit_and_normalize(
                accepted,
                record=record,
                decision_point=decision_point,
                is_first_step=int(stored_step["step"]) == 1,
                is_repeat_order=ordered in seen_orders,
            )
            steps.append({
                "coding_id": coding_id,
                **{key: audit[key] for key in (
                    "issues", "operations", "validation_errors_before", "validation_errors_after"
                )},
            })
            if audit["operations"]:
                corrections.append({
                    "coding_id": coding_id,
                    "reason": "Deterministic clinical-only evidence and cross-field normalization.",
                    "operations": audit["operations"],
                })
            seen_orders.add(ordered)
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "manifest": str(manifest_path),
        "prompt_hash": prompt_hash,
        "model": model,
        "n_steps": len(steps),
        "n_steps_with_issues": sum(bool(step["issues"]) for step in steps),
        "n_issues": sum(len(step["issues"]) for step in steps),
        "n_deterministic_operations": sum(len(step["operations"]) for step in steps),
        "n_unresolved_nonverbatim": sum(
            issue.get("kind") == "nonverbatim_unresolved"
            for step in steps for issue in step["issues"]
        ),
        "n_steps_invalid_after_normalization": sum(
            bool(step["validation_errors_after"]) for step in steps
        ),
        "steps": steps,
        "adjudication": {
            "schema_version": "1.0.0-manual-adjudication",
            "source": "algorithmic",
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
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    report = audit_manifest(manifest_path, args.prompt_hash, args.model)
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
    print(json.dumps({key: value for key, value in report.items() if key not in {"steps", "adjudication"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
