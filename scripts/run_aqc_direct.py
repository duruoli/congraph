"""Resumable DIRECT-only A/Q/C pilot and development annotation runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiments.annotation.annotate import call_json  # noqa: E402
from experiments.aqc import prompts  # noqa: E402
from experiments.llm_experiment.env_loader import load_openrouter_key  # noqa: E402
from scripts.build_masked_view import RAW, build_record, load_lab_map  # noqa: E402
from scripts.run_aqc_framework_check import validate_output  # noqa: E402

DEV = ROOT / "data" / "aqc_development"
PILOT = ROOT / "data" / "aqc_direct" / "pilot_manifest.json"
RESULTS = ROOT / "results" / "aqc_direct"
VALIDATOR_VERSION = "3.0.0"
RETRY_PROTOCOL_VERSION = "1.0.0-validator-feedback"
DEFAULT_MODELS = [
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5.1",
    "openai/gpt-5-mini",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def prompt_version_hash() -> str:
    return digest(prompts.ANNOTATION_SYSTEM + json.dumps(prompts.output_contract(), sort_keys=True))


def load_patients(scope: str) -> list[dict[str, Any]]:
    if scope == "pilot":
        return read_json(PILOT)["patients"]
    split = read_json(DEV / "split_manifest.json")
    pilot_keys = {
        (p["disease"], int(p["hadm_id"])) for p in read_json(PILOT)["patients"]
    }
    return [
        p for p in split["patients"]
        if p["partition"] == "development"
        and (p["disease"], int(p["hadm_id"])) not in pilot_keys
    ]


def usage_cost(call: dict[str, Any]) -> float:
    usage = call.get("usage") or {}
    return float(usage.get("cost") or call.get("cost") or 0.0)


def current_run_cost(result: dict[str, Any]) -> float:
    return sum(
        usage_cost(attempt.get("call") or {})
        for step in result.get("steps", [])
        if not step.get("reused")
        for attempt in step.get("attempts", [])
    )


def step_cost(step: dict[str, Any]) -> float:
    return sum(usage_cost(a.get("call") or {}) for a in step.get("attempts", [])) + (
        step_cost(step["superseded_step"])
        if isinstance(step.get("superseded_step"), dict) else 0.0
    )


def result_cost(result: dict[str, Any]) -> float:
    return sum(step_cost(step) for step in result.get("steps", [])) + sum(
        result_cost(old) for old in result.get("superseded_runs", [])
    )


def result_is_valid(result: dict[str, Any], expected_steps: int) -> bool:
    steps = result.get("steps")
    if not isinstance(steps, list) or len(steps) != expected_steps:
        return False
    seen_orders: set[str] = set()
    for index, step in enumerate(steps):
        ordered = str(step.get("ordered", ""))
        if validate_output(
            step.get("accepted"),
            is_first_step=index == 0,
            ordered=ordered,
            is_repeat_order=ordered in seen_orders,
        ):
            return False
        seen_orders.add(ordered)
    return True


def select_stratified_new(
    patients: list[dict[str, Any]], model: str, batch_size: int
) -> list[dict[str, Any]]:
    completed_names = {
        path.name
        for path in (RESULTS / "development").glob(f"*/{slug(model)}/*.json")
    }
    remaining = [
        patient for patient in patients
        if f"{patient['disease']}_{patient['hadm_id']}.json" not in completed_names
    ]
    diseases = sorted({patient["disease"] for patient in remaining})
    groups = {
        disease: sorted(
            (patient for patient in remaining if patient["disease"] == disease),
            key=lambda patient: digest(
                f"congraph-aqc-direct-development-batches-v1|{disease}|{patient['hadm_id']}"
            ),
        )
        for disease in diseases
    }
    selected: list[dict[str, Any]] = []
    while len(selected) < batch_size and any(groups.values()):
        for disease in diseases:
            if groups[disease] and len(selected) < batch_size:
                selected.append(groups[disease].pop(0))
    return selected


def select_invalid_existing(
    patients: list[dict[str, Any]], model: str, scope: str
) -> list[dict[str, Any]]:
    out_dir = RESULTS / scope / prompt_version_hash()[:12] / slug(model)
    selected = []
    for patient in patients:
        path = out_dir / f"{patient['disease']}_{patient['hadm_id']}.json"
        if path.exists() and not result_is_valid(read_json(path), int(patient["n_steps"])):
            selected.append(patient)
    return selected


def run_patient(patient: dict[str, Any], model: str, frames: dict[str, pd.DataFrame],
                labmap: dict, retries: int, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    disease, hadm_id = patient["disease"], int(patient["hadm_id"])
    source = read_json(ROOT / patient["source_path"])
    row = frames[disease][frames[disease]["hadm_id"] == hadm_id]
    if row.empty:
        raise ValueError(f"raw record not found: {disease}/{hadm_id}")
    record = build_record(disease, hadm_id, row.iloc[0], labmap)
    if len(record["decision_points"]) != len(source["steps"]):
        raise ValueError(f"step alignment mismatch: {disease}/{hadm_id}")

    prior = None
    steps = []
    old_steps = existing.get("steps", []) if isinstance(existing, dict) else []
    for step_index, (dp, source_step) in enumerate(
        zip(record["decision_points"], source["steps"], strict=True)
    ):
        if int(dp["step"]) != int(source_step["step"]):
            raise ValueError(f"step id mismatch: {disease}/{hadm_id}")
        old_step = old_steps[step_index] if step_index < len(old_steps) else None
        is_repeat_order = any(
            previous_dp["ordered"] == dp["ordered"]
            for previous_dp in record["decision_points"][:step_index]
        )
        if isinstance(old_step, dict) and not validate_output(
            old_step.get("accepted"),
            is_first_step=step_index == 0,
            ordered=str(dp["ordered"]),
            is_repeat_order=is_repeat_order,
        ):
            reused_step = dict(old_step)
            reused_step["reused"] = True
            steps.append(reused_step)
            prior = old_step["accepted"]
            continue
        user = prompts.build_annotation_user(dp, record["baseline"], prior)
        attempts = []
        retry_errors: list[str] = []
        for attempt_index in range(retries + 1):
            attempt_user = user
            if retry_errors:
                attempt_user += (
                    "\n\n## Validator feedback from the preceding attempt\n"
                    "Correct every listed structural error while preserving evidence-grounded "
                    "clinical uncertainty. Return the full JSON object again.\n- "
                    + "\n- ".join(retry_errors)
                )
            call = call_json(
                prompts.ANNOTATION_SYSTEM, attempt_user, model=model, temperature=0.0, max_tokens=4000
            )
            errors = validate_output(
                call.get("parsed"),
                is_first_step=step_index == 0,
                ordered=str(dp["ordered"]),
                is_repeat_order=is_repeat_order,
            )
            attempts.append({
                "attempt": attempt_index + 1,
                "request_prompt_sha256": digest(prompts.ANNOTATION_SYSTEM + "\n" + attempt_user),
                "call": call,
                "validation_errors": errors,
            })
            if not errors:
                break
            retry_errors = errors
        accepted = attempts[-1]["call"].get("parsed") if not attempts[-1]["validation_errors"] else None
        new_step = {
            "step": int(dp["step"]),
            "ordered": dp["ordered"],
            "request_prompt_sha256": digest(prompts.ANNOTATION_SYSTEM + "\n" + user),
            "attempts": attempts,
            "accepted": accepted,
            "validation_errors": attempts[-1]["validation_errors"],
        }
        if isinstance(old_step, dict):
            new_step["superseded_step"] = old_step
        steps.append(new_step)
        if accepted is None:
            break
        prior = accepted
    return {
        "schema_version": "2.0.0-development",
        "scope": "DIRECT empirical annotation",
        "disease_stratum_sampling_only": disease,
        "hadm_id": hadm_id,
        "source_path": patient["source_path"],
        "model": model,
        "prompt_version_sha256": prompt_version_hash(),
        "validator_version": VALIDATOR_VERSION,
        "retry_protocol_version": RETRY_PROTOCOL_VERSION,
        "causal_boundary": "pre-order chart + resulted prior imaging + prior A/Q/C + current order",
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("pilot", "development"), default="pilot")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stratified-new", type=int, metavar="N",
                        help="select N not-yet-written development patients round-robin by disease")
    parser.add_argument(
        "--patient-manifest",
        type=Path,
        help="use an exact frozen patient list for interruption-safe batch resume",
    )
    parser.add_argument("--retries", type=int, default=2)
    cost_group = parser.add_mutually_exclusive_group()
    cost_group.add_argument("--cost-stop-usd", type=float, default=5.0)
    cost_group.add_argument("--no-cost-stop", action="store_true",
                            help="disable the per-run recorded-cost stopping threshold")
    parser.add_argument("--execute", action="store_true", help="authorize calls after external approval")
    parser.add_argument("--repair-invalid", action="store_true",
                        help="rerun only incomplete/invalid stored trajectories and retain old runs")
    parser.add_argument("--repair-invalid-steps", action="store_true",
                        help="rerun only invalid/missing steps; reuse valid stored steps")
    parser.add_argument("--repair-existing-only", action="store_true",
                        help="select only stored trajectories failing the current validator")
    args = parser.parse_args()
    models = args.models or DEFAULT_MODELS
    patients = load_patients(args.scope)
    if args.patient_manifest is not None:
        if args.repair_existing_only or args.stratified_new is not None or args.limit is not None:
            parser.error(
                "--patient-manifest cannot be combined with --repair-existing-only, "
                "--stratified-new, or --limit"
            )
        manifest_path = (
            args.patient_manifest
            if args.patient_manifest.is_absolute()
            else ROOT / args.patient_manifest
        )
        requested = read_json(manifest_path)
        requested_rows = requested.get("patients") if isinstance(requested, dict) else None
        if not isinstance(requested_rows, list):
            parser.error("--patient-manifest must contain a patients list")
        available = {
            (p["disease"], int(p["hadm_id"])): p
            for p in patients
        }
        requested_keys = [
            (row.get("disease"), int(row.get("hadm_id")))
            for row in requested_rows if isinstance(row, dict)
        ]
        if len(requested_keys) != len(requested_rows):
            parser.error("every patient-manifest row must be an object with disease and hadm_id")
        if len(requested_keys) != len(set(requested_keys)):
            parser.error("patient-manifest contains duplicate patient keys")
        missing = [key for key in requested_keys if key not in available]
        if missing:
            parser.error(f"patient-manifest contains unavailable patients: {missing}")
        patients = [available[key] for key in requested_keys]
    elif args.repair_existing_only:
        if len(models) != 1 or not args.repair_invalid_steps:
            parser.error("--repair-existing-only requires one model and --repair-invalid-steps")
        patients = select_invalid_existing(patients, models[0], args.scope)
    elif args.stratified_new is not None:
        if args.scope != "development" or len(models) != 1:
            parser.error("--stratified-new requires --scope development and exactly one model")
        patients = select_stratified_new(patients, models[0], args.stratified_new)
    elif args.limit is not None:
        patients = patients[:args.limit]

    plan = {
        "scope": args.scope,
        "models": models,
        "n_patients": len(patients),
        "n_steps": sum(int(p["n_steps"]) for p in patients),
        "by_disease": {
            disease: sum(p["disease"] == disease for p in patients)
            for disease in sorted({p["disease"] for p in patients})
        },
        "pilot_patients_excluded": 6 if args.scope == "development" else 0,
        "retries": args.retries,
        "cost_stop_usd": None if args.no_cost_stop else args.cost_stop_usd,
        "prompt_version_sha256": prompt_version_hash(),
        "validator_version": VALIDATOR_VERSION,
        "retry_protocol_version": RETRY_PROTOCOL_VERSION,
        "repair_invalid": args.repair_invalid,
        "repair_invalid_steps": args.repair_invalid_steps,
        "patient_manifest": str(args.patient_manifest) if args.patient_manifest else None,
        "output_root": str(RESULTS.relative_to(ROOT)),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        print("dry run only; add --execute after explicit authorization for this batch/destination/use")
        return
    load_openrouter_key()

    frames = {disease: pd.read_csv(ROOT / path) for disease, path in RAW.items()}
    labmap = load_lab_map()
    spent = 0.0
    for model in models:
        out_dir = RESULTS / args.scope / prompt_version_hash()[:12] / slug(model)
        out_dir.mkdir(parents=True, exist_ok=True)
        selected_names = {
            f"{patient['disease']}_{patient['hadm_id']}.json" for patient in patients
        }
        prior_outputs = [path for path in out_dir.glob("*.json") if path.name in selected_names]
        spent += sum(result_cost(read_json(path)) for path in prior_outputs)
        print(f"recorded cumulative batch cost before resume: ${spent:.6f}")
        for index, patient in enumerate(patients, start=1):
            out = out_dir / f"{patient['disease']}_{patient['hadm_id']}.json"
            if out.exists():
                old_result = read_json(out)
                if not (args.repair_invalid or args.repair_invalid_steps) or result_is_valid(
                    old_result, int(patient["n_steps"])
                ):
                    print(f"[{model} {index}/{len(patients)}] resume skip {out.name}")
                    continue
                print(f"[{model} {index}/{len(patients)}] repair invalid {out.name}")
            else:
                old_result = None
            if not args.no_cost_stop and spent >= args.cost_stop_usd:
                raise RuntimeError(f"cost stop reached: ${spent:.6f}")
            print(f"[{model} {index}/{len(patients)}] {patient['disease']}/{patient['hadm_id']}")
            result = run_patient(
                patient, model, frames, labmap, args.retries,
                existing=old_result if args.repair_invalid_steps else None,
            )
            if old_result is not None and not args.repair_invalid_steps:
                result["superseded_runs"] = [
                    *old_result.get("superseded_runs", []),
                    {key: value for key, value in old_result.items() if key != "superseded_runs"},
                ]
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            spent += current_run_cost(result)
            print(f"recorded session cost: ${spent:.6f}")


if __name__ == "__main__":
    main()
