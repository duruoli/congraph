"""Read-only content audit for one frozen DIRECT development batch."""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiments.aqc import prompts  # noqa: E402
from scripts.build_masked_view import RAW, build_record, load_lab_map  # noqa: E402
from scripts.run_aqc_direct import read_json, slug  # noqa: E402
from scripts.run_aqc_framework_check import validate_output  # noqa: E402


def apply_json_patch(value: dict[str, Any], operations: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the small RFC-6902 subset used by manual DIRECT adjudications."""
    result = copy.deepcopy(value)
    for operation in operations:
        op = operation.get("op")
        raw_path = operation.get("path")
        if op not in {"add", "replace", "remove"} or not isinstance(raw_path, str):
            raise ValueError(f"unsupported adjudication operation: {operation}")
        parts = [part.replace("~1", "/").replace("~0", "~") for part in raw_path.split("/")[1:]]
        if not parts:
            raise ValueError("root-level adjudication operations are not supported")
        parent: Any = result
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        leaf = parts[-1]
        if isinstance(parent, list):
            if op == "add" and leaf == "-":
                parent.append(copy.deepcopy(operation.get("value")))
            elif op == "remove":
                parent.pop(int(leaf))
            elif op in {"add", "replace"}:
                parent[int(leaf)] = copy.deepcopy(operation.get("value"))
        elif op == "remove":
            del parent[leaf]
        else:
            parent[leaf] = copy.deepcopy(operation.get("value"))
    return result


def normalized_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text)


def token_coverage(quote: str, visible: str) -> float:
    quote_tokens = [token for token in normalized_tokens(quote) if len(token) >= 3 or token.isdigit()]
    if not quote_tokens:
        return 1.0
    visible_counts = Counter(normalized_tokens(visible))
    quote_counts = Counter(quote_tokens)
    matched = sum(min(count, visible_counts[token]) for token, count in quote_counts.items())
    return matched / sum(quote_counts.values())


def evidence_items(value: dict[str, Any]) -> Iterable[tuple[str, str]]:
    for index, assumption in enumerate(value.get("assumptions") or []):
        if isinstance(assumption, dict):
            for evidence_index, quote in enumerate(assumption.get("evidence") or []):
                if isinstance(quote, str):
                    yield f"assumptions[{index}].evidence[{evidence_index}]", quote
    question = value.get("current_question")
    if isinstance(question, dict):
        for evidence_index, quote in enumerate(question.get("evidence") or []):
            if isinstance(quote, str):
                yield f"current_question.evidence[{evidence_index}]", quote
    coverage = value.get("coverage")
    if isinstance(coverage, dict):
        for index, requirement in enumerate(coverage.get("requirements") or []):
            if isinstance(requirement, dict):
                for evidence_index, quote in enumerate(requirement.get("supporting_evidence") or []):
                    if isinstance(quote, str):
                        yield (
                            f"coverage.requirements[{index}].supporting_evidence[{evidence_index}]",
                            quote,
                        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt-hash", required=True)
    parser.add_argument("--model", default="openai/gpt-5.1")
    parser.add_argument("--evidence-threshold", type=float, default=0.72)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    manifest = read_json(manifest_path)
    patients = manifest.get("patients")
    if not isinstance(patients, list):
        raise ValueError("manifest must contain patients")

    frames = {disease: pd.read_csv(ROOT / path) for disease, path in RAW.items()}
    labmap = load_lab_map()
    result_dir = (
        ROOT / "results" / "aqc_direct" / "development" / args.prompt_hash[:12] / slug(args.model)
    )
    corrections: dict[str, list[dict[str, Any]]] = {}
    if args.adjudication:
        adjudication_path = (
            args.adjudication if args.adjudication.is_absolute() else ROOT / args.adjudication
        )
        adjudication = read_json(adjudication_path)
        for correction in adjudication.get("corrections") or []:
            coding_id = correction.get("coding_id")
            operations = correction.get("operations")
            if not isinstance(coding_id, str) or not isinstance(operations, list):
                raise ValueError("adjudication corrections require coding_id and operations")
            corrections[coding_id] = operations
    findings: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    missing_files: list[str] = []

    for patient in patients:
        disease = str(patient["disease"])
        hadm_id = int(patient["hadm_id"])
        path = result_dir / f"{disease}_{hadm_id}.json"
        if not path.exists():
            missing_files.append(path.name)
            continue
        result = read_json(path)
        row = frames[disease][frames[disease]["hadm_id"] == hadm_id]
        if row.empty:
            raise ValueError(f"raw record not found: {disease}/{hadm_id}")
        record = build_record(disease, hadm_id, row.iloc[0], labmap)
        prior: dict[str, Any] | None = None
        seen_orders: set[str] = set()
        for step, dp in zip(result.get("steps") or [], record["decision_points"], strict=True):
            model_accepted = step.get("accepted")
            coding_id = f"{disease}:{hadm_id}:s{step.get('step')}"
            accepted = (
                apply_json_patch(model_accepted, corrections[coding_id])
                if coding_id in corrections and isinstance(model_accepted, dict)
                else model_accepted
            )
            ordered = str(step.get("ordered") or "")
            is_repeat = ordered in seen_orders
            errors = validate_output(
                accepted,
                is_first_step=int(step.get("step")) == 1,
                ordered=ordered,
                is_repeat_order=is_repeat,
            )
            visible = prompts.build_annotation_user(dp, record["baseline"], prior)
            low_evidence = []
            if isinstance(accepted, dict):
                for field, quote in evidence_items(accepted):
                    coverage = token_coverage(quote, visible)
                    if coverage < args.evidence_threshold:
                        low_evidence.append(
                            {"field": field, "coverage": round(coverage, 4), "quote": quote}
                        )
            if low_evidence:
                findings.append(
                    {"coding_id": coding_id, "kind": "low_evidence_fidelity", "items": low_evidence}
                )
            step_rows.append(
                {
                    "coding_id": coding_id,
                    "is_exact_repeat": is_repeat,
                    "validation_errors_under_current_validator": errors,
                    "low_evidence_items": len(low_evidence),
                }
            )
            if isinstance(model_accepted, dict):
                prior = model_accepted
            seen_orders.add(ordered)

    report = {
        "schema_version": "1.0.0-development-content-audit",
        "manifest": str(args.manifest),
        "source_prompt_hash": args.prompt_hash,
        "model": args.model,
        "evidence_threshold": args.evidence_threshold,
        "adjudication": str(args.adjudication) if args.adjudication else None,
        "n_adjudicated_steps": len(corrections),
        "n_manifest_patients": len(patients),
        "n_completed_patients": len(patients) - len(missing_files),
        "n_steps": len(step_rows),
        "n_exact_repeat_steps": sum(row["is_exact_repeat"] for row in step_rows),
        "n_steps_invalid_under_current_validator": sum(
            bool(row["validation_errors_under_current_validator"]) for row in step_rows
        ),
        "n_steps_with_low_evidence_fidelity": sum(row["low_evidence_items"] > 0 for row in step_rows),
        "n_low_evidence_items": sum(row["low_evidence_items"] for row in step_rows),
        "missing_files": missing_files,
        "findings": findings,
        "steps": step_rows,
        "interpretation": (
            "Token coverage is a screening heuristic, not adjudication. Low-coverage evidence must be "
            "checked manually; high coverage does not prove that an inference is clinically justified."
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output_path = args.output if args.output.is_absolute() else ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
