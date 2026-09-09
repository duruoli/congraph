"""Render and run the frozen 12-step AQC--ACR bridge extraction pilot.

This runner intentionally performs only the blinded LLM text-complement pass.
It does not merge deterministic predicates or expose the current order/result.
The 2026-09 pilot run may explicitly skip the HPI leakage preflight so that the
end-to-end extraction mechanics can be exercised; that choice is recorded in
every artifact and must not be mistaken for a validated causal input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiments.annotation.annotate import call_json  # noqa: E402
from experiments.aqc_acr_bridge.prompts import (  # noqa: E402
    LLM_DIMENSIONS,
    SYSTEM,
    build_user,
    output_contract,
)
from experiments.llm_experiment.env_loader import load_openrouter_key  # noqa: E402
from scripts.build_masked_view import RAW, build_record, load_lab_map  # noqa: E402


PILOT_MANIFEST = ROOT / "data" / "aqc_acr_bridge" / "pilot_v1" / "sample_manifest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "aqc_acr_bridge" / "pilot_v1"
RUNNER_VERSION = "0.2.0-sparse-llm-text-complement-pilot"
ALLOWED_EVIDENCE_SOURCES = {"history", "physical_examination"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def model_slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def parse_step_id(step_id: str) -> tuple[str, int, int]:
    try:
        disease, hadm_text, step_text = step_id.split(":")
        return disease, int(hadm_text), int(step_text.removeprefix("s"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid pilot step_id: {step_id!r}") from exc


def load_pilot_steps(path: Path) -> list[dict[str, Any]]:
    manifest = read_json(path)
    rows = manifest.get("decision_steps") if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("pilot manifest must contain a non-empty decision_steps list")
    step_ids = [row.get("step_id") for row in rows if isinstance(row, dict)]
    if len(step_ids) != len(rows) or not all(isinstance(item, str) for item in step_ids):
        raise ValueError("every pilot decision step must have a string step_id")
    if len(step_ids) != len(set(step_ids)):
        raise ValueError("pilot manifest contains duplicate step IDs")
    return rows


def resolve_steps(path: Path) -> list[dict[str, Any]]:
    """Resolve frozen step IDs to raw rows and causally ordered masked views."""
    selected = load_pilot_steps(path)
    needed_diseases = {parse_step_id(row["step_id"])[0] for row in selected}
    frames = {
        disease: pd.read_csv(ROOT / RAW[disease]) for disease in sorted(needed_diseases)
    }
    labmap = load_lab_map()
    record_cache: dict[tuple[str, int], dict[str, Any]] = {}
    raw_cache: dict[tuple[str, int], Mapping[str, Any]] = {}
    resolved = []

    for selection in selected:
        step_id = selection["step_id"]
        disease, hadm_id, step_number = parse_step_id(step_id)
        if disease not in RAW:
            raise ValueError(f"unknown disease in pilot step: {step_id}")
        key = (disease, hadm_id)
        if key not in record_cache:
            matching = frames[disease][frames[disease]["hadm_id"] == hadm_id]
            if len(matching) != 1:
                raise ValueError(
                    f"expected one raw row for {disease}/{hadm_id}, found {len(matching)}"
                )
            raw_row = matching.iloc[0]
            raw_cache[key] = raw_row
            record_cache[key] = build_record(disease, hadm_id, raw_row, labmap)
        record = record_cache[key]
        matches = [
            item for item in record["decision_points"] if int(item["step"]) == step_number
        ]
        if len(matches) != 1:
            raise ValueError(f"could not uniquely resolve decision point: {step_id}")
        resolved.append({
            "step_id": step_id,
            "selection_tags": selection.get("selection_tags", []),
            "disease": disease,
            "hadm_id": hadm_id,
            "step": step_number,
            "baseline": record["baseline"],
            "decision_point": matches[0],
            "raw_row": raw_cache[key],
        })
    return resolved


def visible_sources(baseline: Mapping[str, Any], prior: list[Mapping[str, Any]]) -> dict[str, str]:
    # build_user masks labelled vitals. Recover the exact rendered source sections
    # from the prompt instead of duplicating that masking logic here.
    user = build_user(baseline, prior)
    history = user.split("[History]\n", 1)[1].split("\n\n[Physical examination]", 1)[0]
    physical = user.split("[Physical examination]\n", 1)[1].split(
        "\n\n[Prior resulted imaging]", 1
    )[0]
    sources = {"history": history, "physical_examination": physical}
    for index, item in enumerate(prior, start=1):
        sources[f"prior_imaging_{index}"] = str(item.get("report", ""))
    return sources


def _expected_item_keys() -> dict[str, set[str]]:
    template = output_contract()
    return {
        dimension: set(items[0])
        for dimension, items in template["patient_context"].items()
    }


def validate_extraction(
    parsed: Any,
    baseline: Mapping[str, Any],
    prior: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Strictly validate shape, allowed keys, and exact-substring evidence."""
    errors: list[str] = []
    evidence_count = 0
    item_count = 0
    sources = visible_sources(baseline, prior)
    expected_keys = _expected_item_keys()

    if not isinstance(parsed, dict):
        return {"valid": False, "errors": ["response is not a JSON object"],
                "item_count": 0, "evidence_count": 0}

    top_keys = {"schema_version", "patient_context", "other_proposed_dimension"}
    if set(parsed) != top_keys:
        errors.append(f"top-level keys must equal {sorted(top_keys)}")
    if parsed.get("schema_version") != "3.0.0-hybrid-text-context":
        errors.append("incorrect schema_version")
    context = parsed.get("patient_context")
    if not isinstance(context, dict):
        errors.append("patient_context must be an object")
        context = {}
    if set(context) != set(LLM_DIMENSIONS):
        errors.append("patient_context keys must match the eleven dimensions")

    def validate_evidence(value: Any, location: str) -> None:
        nonlocal evidence_count
        if not isinstance(value, list) or not value:
            errors.append(f"{location}.evidence must be a non-empty array")
            return
        for evidence_index, evidence in enumerate(value):
            evidence_location = f"{location}.evidence[{evidence_index}]"
            if not isinstance(evidence, dict) or set(evidence) != {"source", "support"}:
                errors.append(f"{evidence_location} must contain only source and support")
                continue
            source = evidence.get("source")
            support = evidence.get("support")
            allowed_source = (
                source in ALLOWED_EVIDENCE_SOURCES
                or isinstance(source, str) and source.startswith("prior_imaging_")
            )
            if not allowed_source or source not in sources:
                errors.append(f"{evidence_location}.source is not a visible source: {source!r}")
                continue
            if not isinstance(support, str) or not support.strip():
                errors.append(f"{evidence_location}.support must be a non-empty string")
                continue
            if support not in sources[source]:
                errors.append(f"{evidence_location}.support is not an exact source substring")
                continue
            evidence_count += 1

    for dimension in LLM_DIMENSIONS:
        items = context.get(dimension)
        if not isinstance(items, list):
            errors.append(f"patient_context.{dimension} must be an array")
            continue
        for index, item in enumerate(items):
            item_count += 1
            location = f"patient_context.{dimension}[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{location} must be an object")
                continue
            if set(item) != expected_keys[dimension]:
                errors.append(
                    f"{location} keys must equal {sorted(expected_keys[dimension])}"
                )
            validate_evidence(item.get("evidence"), location)

    other = parsed.get("other_proposed_dimension")
    other_keys = {"dimension", "definition", "value", "evidence"}
    if not isinstance(other, list):
        errors.append("other_proposed_dimension must be an array")
    else:
        for index, item in enumerate(other):
            item_count += 1
            location = f"other_proposed_dimension[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{location} must be an object")
                continue
            if set(item) != other_keys:
                errors.append(f"{location} keys must equal {sorted(other_keys)}")
            validate_evidence(item.get("evidence"), location)

    return {
        "valid": not errors,
        "errors": errors,
        "item_count": item_count,
        "evidence_count": evidence_count,
    }


def render_input(row: Mapping[str, Any], model: str) -> dict[str, Any]:
    baseline = row["baseline"]
    prior = row["decision_point"].get("visible_prior_imaging") or []
    user = build_user(baseline, prior)
    return {
        "runner_version": RUNNER_VERSION,
        "step_id": row["step_id"],
        "selection_tags": row["selection_tags"],
        "model": model,
        "backend": "openrouter",
        "temperature": 0.0,
        "max_tokens": 6000,
        "hpi_leakage_preflight_skipped": True,
        "current_order_included": False,
        "current_result_included": False,
        "later_outcomes_included": False,
        "n_visible_prior_imaging": len(prior),
        "system_sha256": sha256_text(SYSTEM),
        "user_sha256": sha256_text(user),
        "system": SYSTEM,
        "user": user,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PILOT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="openai/gpt-5.1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    prompt_hash = sha256_text(SYSTEM)[:12]
    run_root = output_root / model_slug(args.model) / prompt_hash
    input_dir = run_root / "inputs"
    output_dir = run_root / "outputs"
    rows = resolve_steps(manifest_path)

    rendered_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        rendered = render_input(row, args.model)
        rendered_rows[row["step_id"]] = rendered
        write_json(input_dir / f"{row['step_id'].replace(':', '_')}.json", rendered)

    print(json.dumps({
        "runner_version": RUNNER_VERSION,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "model": args.model,
        "prompt_hash": prompt_hash,
        "backend": "openrouter",
        "n_steps": len(rows),
        "hpi_leakage_preflight_skipped": True,
        "input_dir": str(input_dir.relative_to(ROOT)),
        "output_dir": str(output_dir.relative_to(ROOT)),
        "execute": args.execute,
    }, ensure_ascii=False, indent=2))

    if not args.execute:
        print("Rendered inputs only; add --execute to call the model.")
        return

    load_openrouter_key()
    completed = 0
    valid = 0
    failed: list[str] = []
    skipped = 0
    total_usage: dict[str, int] = {}

    for index, row in enumerate(rows, start=1):
        step_id = row["step_id"]
        output_path = output_dir / f"{step_id.replace(':', '_')}.json"
        if output_path.exists() and not args.force:
            existing = read_json(output_path)
            if isinstance(existing, dict) and existing.get("parsed") is not None:
                skipped += 1
                print(f"[{index:02d}/{len(rows):02d}] skip existing {step_id}")
                continue
        rendered = rendered_rows[step_id]
        print(f"[{index:02d}/{len(rows):02d}] extracting {step_id}", flush=True)
        try:
            call = call_json(
                SYSTEM,
                rendered["user"],
                model=args.model,
                temperature=0.0,
                max_tokens=6000,
            )
            parsed = call.get("parsed")
            validation = validate_extraction(
                parsed,
                row["baseline"],
                row["decision_point"].get("visible_prior_imaging") or [],
            )
            artifact = {
                "runner_version": RUNNER_VERSION,
                "step_id": step_id,
                "model": args.model,
                "backend": "openrouter",
                "temperature": 0.0,
                "max_tokens": 6000,
                "hpi_leakage_preflight_skipped": True,
                "input_file": str(
                    (input_dir / f"{step_id.replace(':', '_')}.json").relative_to(ROOT)
                ),
                "system_sha256": rendered["system_sha256"],
                "user_sha256": rendered["user_sha256"],
                "request_id": call.get("request_id"),
                "usage": call.get("usage"),
                "validation": validation,
                "parsed": parsed,
                "raw": call.get("raw"),
            }
            write_json(output_path, artifact)
            completed += 1
            valid += int(validation["valid"])
            if not validation["valid"]:
                failed.append(step_id)
            for key, value in (call.get("usage") or {}).items():
                if isinstance(value, int):
                    total_usage[key] = total_usage.get(key, 0) + value
            print(
                f"  validation={'valid' if validation['valid'] else 'INVALID'} "
                f"items={validation['item_count']} evidence={validation['evidence_count']}"
            )
        except Exception as exc:  # preserve progress across provider/API failures
            failed.append(step_id)
            write_json(output_path, {
                "runner_version": RUNNER_VERSION,
                "step_id": step_id,
                "model": args.model,
                "backend": "openrouter",
                "hpi_leakage_preflight_skipped": True,
                "input_file": str(
                    (input_dir / f"{step_id.replace(':', '_')}.json").relative_to(ROOT)
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "parsed": None,
            })
            print(f"  ERROR {type(exc).__name__}: {exc}")

    summary = {
        "runner_version": RUNNER_VERSION,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "model": args.model,
        "prompt_hash": prompt_hash,
        "backend": "openrouter",
        "hpi_leakage_preflight_skipped": True,
        "n_steps": len(rows),
        "n_completed_this_run": completed,
        "n_skipped_existing": skipped,
        "n_valid_this_run": valid,
        "failed_or_invalid_step_ids": failed,
        "usage_this_run": total_usage,
    }
    write_json(run_root / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
