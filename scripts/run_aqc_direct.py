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
    return digest(prompts.DIRECT_SYSTEM + json.dumps(prompts.output_contract(), sort_keys=True))


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


def result_cost(result: dict[str, Any]) -> float:
    return sum(
        usage_cost(attempt.get("call") or {})
        for step in result.get("steps", [])
        for attempt in step.get("attempts", [])
    )


def run_patient(patient: dict[str, Any], model: str, frames: dict[str, pd.DataFrame],
                labmap: dict, retries: int) -> dict[str, Any]:
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
    for dp, source_step in zip(record["decision_points"], source["steps"], strict=True):
        if int(dp["step"]) != int(source_step["step"]):
            raise ValueError(f"step id mismatch: {disease}/{hadm_id}")
        user = prompts.build_direct_user(dp, record["baseline"], prior)
        attempts = []
        for attempt_index in range(retries + 1):
            call = call_json(
                prompts.DIRECT_SYSTEM, user, model=model, temperature=0.0, max_tokens=4000
            )
            errors = validate_output(call.get("parsed"), is_first_step=int(dp["step"]) == 1)
            attempts.append({
                "attempt": attempt_index + 1,
                "call": call,
                "validation_errors": errors,
            })
            if not errors:
                break
        accepted = attempts[-1]["call"].get("parsed") if not attempts[-1]["validation_errors"] else None
        steps.append({
            "step": int(dp["step"]),
            "ordered": dp["ordered"],
            "request_prompt_sha256": digest(prompts.DIRECT_SYSTEM + "\n" + user),
            "attempts": attempts,
            "accepted": accepted,
            "validation_errors": attempts[-1]["validation_errors"],
        })
        if accepted is None:
            break
        prior = accepted
    return {
        "schema_version": "1.0.0-development",
        "scope": "DIRECT empirical annotation",
        "disease_stratum_sampling_only": disease,
        "hadm_id": hadm_id,
        "source_path": patient["source_path"],
        "model": model,
        "prompt_version_sha256": prompt_version_hash(),
        "causal_boundary": "pre-order chart + resulted prior imaging + prior A/Q/C + current order",
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("pilot", "development"), default="pilot")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--cost-stop-usd", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true", help="authorize calls after external approval")
    args = parser.parse_args()
    models = args.models or DEFAULT_MODELS
    patients = load_patients(args.scope)
    if args.limit is not None:
        patients = patients[:args.limit]

    plan = {
        "scope": args.scope,
        "models": models,
        "n_patients": len(patients),
        "n_steps": sum(int(p["n_steps"]) for p in patients),
        "pilot_patients_excluded": 6 if args.scope == "development" else 0,
        "retries": args.retries,
        "cost_stop_usd": args.cost_stop_usd,
        "prompt_version_sha256": prompt_version_hash(),
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
                print(f"[{model} {index}/{len(patients)}] resume skip {out.name}")
                continue
            if spent >= args.cost_stop_usd:
                raise RuntimeError(f"cost stop reached: ${spent:.6f}")
            print(f"[{model} {index}/{len(patients)}] {patient['disease']}/{patient['hadm_id']}")
            result = run_patient(patient, model, frames, labmap, args.retries)
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            spent += result_cost(result)
            print(f"recorded session cost: ${spent:.6f}")


if __name__ == "__main__":
    main()
