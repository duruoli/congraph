"""Prepare and run the independent development-only A/Q/C two-arm framework check.

Arm DIRECT sees a causally masked chart plus the actual order, but not the old
reconstruction. Arm RECODE sees only the old schema-light ex-ante reconstruction
plus the order. Neither arm sees the current result, later events, verification,
deviation, ACR, or the disease sampling label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiments.annotation.annotate import call_json  # noqa: E402
from experiments.aqc import prompts  # noqa: E402
from scripts.build_aqc_discovery_sample import (  # noqa: E402
    DISEASES, ROOT as BUILDER_ROOT, greedy_batch, load_timing, profile_development,
    stable_hash,
)
from scripts.build_masked_view import RAW, build_record, load_lab_map  # noqa: E402

DATA = ROOT / "data" / "aqc_framework_check"
RESULTS = ROOT / "results" / "aqc_framework_check"
SPLIT_PATH = ROOT / "data" / "aqc_development" / "split_manifest.json"
DISCOVERY_PATH = ROOT / "data" / "aqc_development" / "development_sample_manifest.json"
SALT = "congraph-aqc-framework-check-v1"
DEFAULT_MODEL = os.environ.get("AQC_MODEL", "anthropic/claude-sonnet-4.6")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_manifest() -> dict[str, Any]:
    split = read_json(SPLIT_PATH)
    discovery = read_json(DISCOVERY_PATH)
    used = {
        (patient["disease"], int(patient["hadm_id"]))
        for batch in discovery["batches"] for patient in batch["patients"]
    }
    final_test = {
        (patient["disease"], int(patient["hadm_id"]))
        for patient in split["patients"] if patient["partition"] == "final_test"
    }
    development = [
        patient for patient in split["patients"]
        if patient["partition"] == "development"
        and (patient["disease"], int(patient["hadm_id"])) not in used
    ]
    timing = load_timing()
    profiles = [profile_development(ROOT / row["source_path"], timing) for row in development]
    for profile in profiles:
        profile["sample_hash"] = stable_hash(
            f"{SALT}|{profile['disease']}|{profile['hadm_id']}"
        )
    selected_keys = set(used)
    selected = greedy_batch(profiles, 4, selected_keys, set())
    keys = [(row["disease"], int(row["hadm_id"])) for row in selected]
    assert len(keys) == len(set(keys)) == 16
    assert not (set(keys) & used)
    assert not (set(keys) & final_test)
    manifest = {
        "schema_version": "1.0.0-development",
        "purpose": "independent DIRECT versus RECODE framework check",
        "source_partition": "development only",
        "selection_salt": SALT,
        "selection_algorithm": (
            "exclude all 48 codebook-development trajectories; choose four per disease by "
            "deterministic maximum structural variation with salted SHA-256 tie-breaking"
        ),
        "forbidden_selection_inputs": [
            "verification", "deviation/dev_belief", "ACR/rating", "current result",
            "later events", "final diagnosis correctness",
        ],
        "n_patients": len(selected),
        "n_source_steps": sum(row["n_steps"] for row in selected),
        "by_disease": {d: sum(row["disease"] == d for row in selected) for d in DISEASES},
        "patients": selected,
    }
    write_json(DATA / "sample_manifest.json", manifest)
    return manifest


def load_masked_record(disease: str, hadm_id: int, frames: dict[str, pd.DataFrame], labmap: dict) -> dict:
    frame = frames[disease]
    row = frame[frame["hadm_id"] == hadm_id]
    if row.empty:
        raise ValueError(f"raw record not found: {disease}/{hadm_id}")
    return build_record(disease, hadm_id, row.iloc[0], labmap)


def validate_output(value: Any, *, is_first_step: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["not_a_json_object"]
    assumptions = value.get("assumptions")
    if not isinstance(assumptions, list):
        errors.append("assumptions_not_list")
    else:
        if len(assumptions) > 5:
            errors.append("too_many_assumptions")
        for index, item in enumerate(assumptions):
            if not isinstance(item, dict):
                errors.append(f"assumption_{index}_not_object")
                continue
            if item.get("type") not in prompts.ASSUMPTION_TYPES:
                errors.append(f"assumption_{index}_bad_type")
            if item.get("status") not in prompts.ASSUMPTION_STATUSES:
                errors.append(f"assumption_{index}_bad_status")
            if item.get("support") not in {"well_supported", "weakly_supported", "unclear"}:
                errors.append(f"assumption_{index}_bad_support")
            evidence = item.get("evidence")
            if not isinstance(evidence, list) or not evidence or len(evidence) > 2:
                errors.append(f"assumption_{index}_bad_evidence_count")
            if item.get("status") in {"established", "excluded"} and item.get("support") != "well_supported":
                errors.append(f"assumption_{index}_strong_status_without_strong_support")
    question = value.get("current_question")
    if not isinstance(question, dict):
        errors.append("current_question_not_object")
        declared: list[str] = []
    else:
        if question.get("type") not in prompts.QUESTION_TYPES:
            errors.append("bad_question_type")
        requirements = question.get("answer_requirements")
        if not isinstance(requirements, list):
            errors.append("answer_requirements_not_list")
            declared: list[tuple[str, str]] = []
        else:
            if len(requirements) > 5:
                errors.append("too_many_answer_requirements")
            declared = []
            for index, item in enumerate(requirements):
                if not isinstance(item, dict):
                    errors.append(f"answer_requirement_{index}_not_object")
                    continue
                requirement_key = item.get("requirement_key")
                requirement_type = item.get("id")
                if not isinstance(requirement_key, str) or not requirement_key.strip():
                    errors.append(f"answer_requirement_{index}_bad_key")
                if requirement_type not in prompts.ANSWER_REQUIREMENT_TYPES:
                    errors.append(f"answer_requirement_{index}_bad_type")
                if isinstance(requirement_key, str) and requirement_key.strip():
                    declared.append((requirement_key, requirement_type))
            declared_keys = [key for key, _ in declared]
            if len(declared_keys) != len(set(declared_keys)):
                errors.append("duplicate_answer_requirement_key")
        secondary = question.get("secondary_questions")
        if not isinstance(secondary, list) or len(secondary) > 2:
            errors.append("bad_secondary_questions")
    coverage = value.get("coverage")
    if not isinstance(coverage, dict) or not isinstance(coverage.get("requirements"), list):
        errors.append("coverage_requirements_not_list")
    else:
        covered: list[tuple[str, str]] = []
        for index, item in enumerate(coverage["requirements"]):
            if not isinstance(item, dict):
                errors.append(f"coverage_{index}_not_object")
                continue
            requirement_key = item.get("requirement_key")
            requirement_type = item.get("requirement_id")
            if not isinstance(requirement_key, str) or not requirement_key.strip():
                errors.append(f"coverage_{index}_bad_key")
            if requirement_type not in prompts.ANSWER_REQUIREMENT_TYPES:
                errors.append(f"coverage_{index}_bad_requirement_type")
            if isinstance(requirement_key, str) and requirement_key.strip():
                covered.append((requirement_key, requirement_type))
            if item.get("status") not in prompts.COVERAGE_STATUSES:
                errors.append(f"coverage_{index}_bad_status")
            if item.get("direction") not in prompts.COVERAGE_DIRECTIONS:
                errors.append(f"coverage_{index}_bad_direction")
            evidence = item.get("supporting_evidence")
            if not isinstance(evidence, list) or len(evidence) > 2:
                errors.append(f"coverage_{index}_bad_evidence_count")
        covered_keys = [key for key, _ in covered]
        if len(covered_keys) != len(set(covered_keys)):
            errors.append("duplicate_coverage_requirement_key")
        if covered != declared:
            errors.append("coverage_question_requirement_mismatch")
        if coverage.get("aggregate") not in prompts.COVERAGE_AGGREGATES:
            errors.append("bad_coverage_aggregate")
        if coverage.get("aggregate") == "sufficiently_answered" and any(
            item.get("status") != "sufficiently_addressed"
            for item in coverage["requirements"] if isinstance(item, dict)
        ):
            errors.append("sufficient_aggregate_with_incomplete_requirement")
    if is_first_step:
        if value.get("question_continuity") != "initial":
            errors.append("first_step_question_continuity_not_initial")
        change = value.get("assumption_change")
        if not isinstance(change, dict) or change.get("label") != "initial":
            errors.append("first_step_assumption_change_not_initial")
        if value.get("derived_transition") != "initial":
            errors.append("first_step_derived_transition_not_initial")
        previous = value.get("previous_order_update")
        if not isinstance(previous, dict) or previous.get("applicable") is not False:
            errors.append("first_step_previous_order_update_applicable")
    return errors


def as_set(value: Any, key: str) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item.get(key)) for item in value if isinstance(item, dict) and item.get(key)}


def jaccard(left: set[str], right: set[str]) -> float:
    return 1.0 if not (left | right) else len(left & right) / len(left | right)


def compare_step(direct: dict[str, Any] | None, recode: dict[str, Any] | None) -> dict[str, Any]:
    direct = direct or {}
    recode = recode or {}
    da = as_set(direct.get("assumptions"), "type")
    ra = as_set(recode.get("assumptions"), "type")
    dq = direct.get("current_question") if isinstance(direct.get("current_question"), dict) else {}
    rq = recode.get("current_question") if isinstance(recode.get("current_question"), dict) else {}
    dr = as_set(dq.get("answer_requirements"), "id")
    rr = as_set(rq.get("answer_requirements"), "id")
    dc = direct.get("coverage") if isinstance(direct.get("coverage"), dict) else {}
    rc = recode.get("coverage") if isinstance(recode.get("coverage"), dict) else {}
    return {
        "assumption_type_jaccard": round(jaccard(da, ra), 4),
        "direct_assumption_types": sorted(da),
        "recode_assumption_types": sorted(ra),
        "question_type_match": dq.get("type") == rq.get("type"),
        "direct_question_type": dq.get("type"),
        "recode_question_type": rq.get("type"),
        "requirement_jaccard": round(jaccard(dr, rr), 4),
        "direct_requirements": sorted(dr),
        "recode_requirements": sorted(rr),
        "aggregate_coverage_match": dc.get("aggregate") == rc.get("aggregate"),
        "direct_aggregate_coverage": dc.get("aggregate"),
        "recode_aggregate_coverage": rc.get("aggregate"),
        "transition_match": direct.get("derived_transition") == recode.get("derived_transition"),
    }


def run_patient(patient: dict[str, Any], *, model: str, frames: dict[str, pd.DataFrame],
                labmap: dict) -> dict[str, Any]:
    disease = patient["disease"]
    hadm_id = int(patient["hadm_id"])
    source = read_json(ROOT / patient["source_path"])
    record = load_masked_record(disease, hadm_id, frames, labmap)
    if len(record["decision_points"]) != len(source["steps"]):
        raise ValueError(
            f"step alignment mismatch {disease}/{hadm_id}: "
            f"masked={len(record['decision_points'])}, annotation={len(source['steps'])}"
        )
    direct_prior = None
    recode_prior = None
    steps = []
    for dp, source_step in zip(record["decision_points"], source["steps"], strict=True):
        if int(dp["step"]) != int(source_step["step"]):
            raise ValueError(f"step id mismatch {disease}/{hadm_id}")
        direct_user = prompts.build_direct_user(dp, record["baseline"], direct_prior)
        recode_user = prompts.build_recode_user(
            source_step["representative_ex_ante"], str(source_step.get("ordered", "")), recode_prior
        )
        direct_call = call_json(
            prompts.DIRECT_SYSTEM, direct_user, model=model, temperature=0.0, max_tokens=4000
        )
        recode_call = call_json(
            prompts.RECODE_SYSTEM, recode_user, model=model, temperature=0.0, max_tokens=4000
        )
        direct_value = direct_call.get("parsed")
        recode_value = recode_call.get("parsed")
        steps.append({
            "step": int(dp["step"]),
            "ordered": dp["ordered"],
            "direct_prompt_sha256": sha256(direct_user),
            "recode_prompt_sha256": sha256(recode_user),
            "direct": direct_call,
            "recode": recode_call,
            "direct_validation_errors": validate_output(direct_value, is_first_step=int(dp["step"]) == 1),
            "recode_validation_errors": validate_output(recode_value, is_first_step=int(dp["step"]) == 1),
            "comparison": compare_step(direct_value, recode_value),
        })
        direct_prior = direct_value
        recode_prior = recode_value
    return {
        "schema_version": "1.0.0-development",
        "disease_stratum_sampling_only": disease,
        "hadm_id": hadm_id,
        "source_path": patient["source_path"],
        "model": model,
        "causal_boundary": (
            "DIRECT=masked chart+order; RECODE=old ex-ante reconstruction+order; both exclude "
            "current result, later events, verification, deviation, ACR, and disease answer labels"
        ),
        "steps": steps,
    }


def summarize(manifest: dict[str, Any]) -> dict[str, Any]:
    steps = []
    queue = []
    for patient in manifest["patients"]:
        path = RESULTS / f"{patient['disease']}_{patient['hadm_id']}.json"
        if not path.exists():
            continue
        result = read_json(path)
        for step in result["steps"]:
            comparison = step["comparison"]
            steps.append(step)
            needs_review = (
                bool(step["direct_validation_errors"] or step["recode_validation_errors"])
                or comparison["assumption_type_jaccard"] < 0.5
                or not comparison["question_type_match"]
                or comparison["requirement_jaccard"] < 0.5
            )
            if needs_review:
                queue.append({
                    "coding_id": f"{result['disease_stratum_sampling_only']}:{result['hadm_id']}:s{step['step']}",
                    "reasons": {
                        "direct_validation_errors": step["direct_validation_errors"],
                        "recode_validation_errors": step["recode_validation_errors"],
                        "assumption_type_jaccard": comparison["assumption_type_jaccard"],
                        "question_type_match": comparison["question_type_match"],
                        "requirement_jaccard": comparison["requirement_jaccard"],
                    },
                })
    usage_by_arm = {}
    for arm in ("direct", "recode"):
        usages = [step[arm].get("usage") for step in steps]
        recorded = [usage for usage in usages if isinstance(usage, dict)]
        usage_by_arm[arm] = {
            "recorded_calls": len(recorded),
            "missing_usage_calls": len(usages) - len(recorded),
            "prompt_tokens": sum(int(usage.get("prompt_tokens") or 0) for usage in recorded),
            "completion_tokens": sum(
                int(usage.get("completion_tokens") or 0) for usage in recorded
            ),
            "cost_usd": round(sum(float(usage.get("cost") or 0) for usage in recorded), 6),
        }
    summary = {
        "schema_version": "1.0.0-development",
        "n_planned_patients": manifest["n_patients"],
        "n_completed_patients": sum(
            (RESULTS / f"{p['disease']}_{p['hadm_id']}.json").exists()
            for p in manifest["patients"]
        ),
        "n_completed_steps": len(steps),
        "direct_valid_steps": sum(not step["direct_validation_errors"] for step in steps),
        "recode_valid_steps": sum(not step["recode_validation_errors"] for step in steps),
        "mean_assumption_type_jaccard": round(mean(
            step["comparison"]["assumption_type_jaccard"] for step in steps
        ), 4) if steps else None,
        "question_type_match_rate": round(mean(
            step["comparison"]["question_type_match"] for step in steps
        ), 4) if steps else None,
        "mean_requirement_jaccard": round(mean(
            step["comparison"]["requirement_jaccard"] for step in steps
        ), 4) if steps else None,
        "aggregate_coverage_match_rate": round(mean(
            step["comparison"]["aggregate_coverage_match"] for step in steps
        ), 4) if steps else None,
        "usage_by_arm": usage_by_arm,
        "recorded_cost_usd": round(
            sum(arm_usage["cost_usd"] for arm_usage in usage_by_arm.values()), 6
        ),
        "cost_scope": (
            "successful calls retained in patient result files; abandoned or failed calls without "
            "a retained usage object are not included"
        ),
        "manual_adjudication_queue": queue,
        "interpretation_boundary": (
            "agreement measures framing dependence, not physician-intent truth or clinical correctness; "
            "coverage from DIRECT remains primary because RECODE lacks the full causally available chart"
        ),
    }
    write_json(RESULTS / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="run only first N selected patients")
    parser.add_argument("--force", action="store_true", help="overwrite completed patient outputs")
    args = parser.parse_args()

    assert BUILDER_ROOT == ROOT
    manifest = prepare_manifest()
    if args.prepare_only:
        print(
            f"prepared framework-check manifest: {manifest['n_patients']} patients, "
            f"{manifest['n_source_steps']} source steps"
        )
        return

    frames = {disease: pd.read_csv(ROOT / RAW[disease]) for disease in DISEASES}
    labmap = load_lab_map()
    RESULTS.mkdir(parents=True, exist_ok=True)
    patients = manifest["patients"][: args.limit]
    for index, patient in enumerate(patients, start=1):
        out = RESULTS / f"{patient['disease']}_{patient['hadm_id']}.json"
        if out.exists() and not args.force:
            print(f"[{index}/{len(patients)}] resume skip {out.name}")
            continue
        print(f"[{index}/{len(patients)}] {patient['disease']}/{patient['hadm_id']}")
        write_json(out, run_patient(patient, model=args.model, frames=frames, labmap=labmap))
        summarize(manifest)
    summary = summarize(manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
