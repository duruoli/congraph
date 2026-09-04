#!/usr/bin/env python3
"""Build an order-blinded development layer and quarantine unresolved leakage."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_aqc_input_leakage import (  # noqa: E402
    apply_reviewed_redactions,
    modality_name,
    score_sentence,
    sentence_sha256,
    split_sentences,
)
from scripts.build_masked_view import RAW, build_record, load_lab_map  # noqa: E402


SPLIT = ROOT / "data" / "aqc_development" / "split_manifest.json"
OUT = ROOT / "data" / "aqc_prediction" / "development_v1"
INPUTS = OUT / "preorder_inputs.jsonl"
LABELS = OUT / "labels.jsonl"
AUDIT = OUT / "leakage_audit.jsonl"
MANIFEST = OUT / "manifest.json"
SALT = "congraph-aqc-preorder-prediction-id-v1"
PROSPECTIVE_CUE = re.compile(
    r"\b(plan(?:ned|ning)?|order(?:ed|ing)?|obtain(?:ed|ing)?|recommend(?:ed|ing)?|"
    r"schedule(?:d|ing)?|pending|await(?:ing)?|will\s+(?:get|have|undergo)|"
    r"to\s+(?:get|obtain|undergo))\b",
    re.I,
)
MODALITY_MENTION = {
    "CT": re.compile(r"\b(?:CT|computed tomography|CAT scan)\b", re.I),
    "US": re.compile(r"\b(?:ultrasound|sonogram|sonography|RUQ US|pelvic US)\b", re.I),
    "MRI": re.compile(r"\b(?:MRI|magnetic resonance)\b", re.I),
    "MRCP": re.compile(r"\b(?:MRCP|cholangiopancreatography)\b", re.I),
    "CTU": re.compile(r"\b(?:CTU|CT urogram|CT urography)\b", re.I),
}


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def opaque_id(disease: str, hadm_id: int, step: int) -> str:
    value = f"{SALT}|{disease}|{hadm_id}|{step}"
    return "pred_" + hashlib.sha256(value.encode()).hexdigest()[:24]


def opaque_sequence_id(disease: str, hadm_id: int) -> str:
    value = f"{SALT}|sequence|{disease}|{hadm_id}"
    return "seq_" + hashlib.sha256(value.encode()).hexdigest()[:24]


def load_existing_reviews() -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    for path in sorted((ROOT / "data" / "aqc_direct").glob("*_leakage_review.json")):
        document = read_json(path)
        for row in document.get("reviews") or []:
            coding_id = row["coding_id"]
            if coding_id in reviews:
                old = reviews[coding_id]
                old_redaction = old.get("history_redactions", old.get("history_redaction"))
                new_redaction = row.get("history_redactions", row.get("history_redaction"))
                if (
                    old.get("decision") != row.get("decision")
                    or old.get("sentence_sha256", old.get("sentence_sha256s"))
                    != row.get("sentence_sha256", row.get("sentence_sha256s"))
                    or old_redaction != new_redaction
                ):
                    raise ValueError(f"conflicting existing leakage reviews: {coding_id}")
                continue
            reviews[coding_id] = row
    return reviews


def load_prediction_reviews() -> dict[str, dict[str, Any]]:
    path = OUT / "leakage_review.json"
    if not path.exists():
        return {}
    rows = read_json(path).get("reviews") or []
    reviews = {row["coding_id"]: row for row in rows}
    if len(reviews) != len(rows):
        raise ValueError("duplicate coding_id in prediction leakage review")
    return reviews


def prospective_candidates(history: str, target_modality: str) -> list[str]:
    pattern = MODALITY_MENTION.get(target_modality)
    if pattern is None:
        return []
    return [
        sentence for sentence in split_sentences(history)
        if pattern.search(sentence) and PROSPECTIVE_CUE.search(sentence)
    ]


def detailed_modality(value: str) -> str:
    return {"Ultrasound": "US"}.get(value, value)


def primary_modality(value: str) -> str:
    return {"CTU": "CT", "MRI": "MR", "MRCP": "MR"}.get(value, value)


def apply_prediction_specific_redactions(
    history: str, disease: str, hadm_id: int, reviews: dict[str, dict[str, Any]]
) -> tuple[str, list[dict[str, str]]]:
    filtered = re.sub(r"\s+", " ", history).strip()
    applied = []
    prefix = f"{disease}:{hadm_id}:s"
    for coding_id, review in reviews.items():
        if not coding_id.startswith(prefix) or review.get("decision") != "confirmed_target_revealing_text":
            continue
        redaction = review.get("history_redaction")
        if not isinstance(redaction, dict):
            raise ValueError(f"missing prediction-specific redaction: {coding_id}")
        exact_text = redaction.get("exact_text")
        expected_hash = redaction.get("sentence_sha256")
        if not isinstance(exact_text, str) or sentence_sha256(exact_text) != expected_hash:
            raise ValueError(f"invalid prediction-specific redaction binding: {coding_id}")
        if filtered.count(exact_text) != 1:
            raise ValueError(f"prediction-specific redaction must match once: {coding_id}")
        replacement = redaction.get("replacement", "")
        filtered = filtered.replace(exact_text, replacement, 1)
        applied.append({"coding_id": coding_id, **redaction})
    return re.sub(r"\s+", " ", filtered).strip(), applied


def main() -> None:
    split = read_json(SPLIT)
    patients = [row for row in split["patients"] if row["partition"] == "development"]
    if len(patients) != 235 or any(row["partition"] != "development" for row in patients):
        raise AssertionError("prediction layer must contain exactly the development patients")
    frames = {disease: pd.read_csv(ROOT / path) for disease, path in RAW.items()}
    labmap = load_lab_map()
    existing_reviews = load_existing_reviews()
    prediction_reviews = load_prediction_reviews()
    result_reviews = dict(existing_reviews)
    for coding_id, review in prediction_reviews.items():
        if review.get("decision") == "confirmed_target_revealing_text":
            continue
        old = result_reviews.get(coding_id)
        if old is not None and old.get("decision") != review.get("decision"):
            raise ValueError(f"conflicting result-review decision: {coding_id}")
        result_reviews[coding_id] = review
    inputs: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    patient_ids: set[str] = set()

    for patient in patients:
        disease = patient["disease"]
        hadm_id = int(patient["hadm_id"])
        row = frames[disease][frames[disease]["hadm_id"] == hadm_id]
        if row.empty:
            raise ValueError(f"missing raw patient {disease}:{hadm_id}")
        record = build_record(disease, hadm_id, row.iloc[0], labmap)
        if len(record["decision_points"]) != int(patient["n_steps"]):
            raise AssertionError(f"step mismatch for {disease}:{hadm_id}")
        raw_history = record["baseline"]["patient_history"]
        filtered_history, applied = apply_reviewed_redactions(
            raw_history, disease, hadm_id, result_reviews
        )
        filtered_history, prediction_applied = apply_prediction_specific_redactions(
            filtered_history, disease, hadm_id, prediction_reviews
        )
        applied = [*applied, *prediction_applied]
        patient_key = f"{disease}:{hadm_id}"
        patient_ids.add(patient_key)
        previous_modality: str | None = None
        for dp in record["decision_points"]:
            step = int(dp["step"])
            coding_id = f"{patient_key}:s{step}"
            prediction_id = opaque_id(disease, hadm_id, step)
            target_modality_raw = modality_name(dp["ordered"])
            target_modality = detailed_modality(target_modality_raw)
            result_candidates = [
                sentence for sentence in split_sentences(filtered_history)
                if score_sentence(sentence, dp["masked_result_of_this_test"], target_modality_raw)["candidate"]
            ]
            review = result_reviews.get(coding_id)
            result_resolved = not result_candidates or (
                review is not None
                and review.get("decision") in {
                    "confirmed_current_result_leak", "cleared_prior_or_external_study"
                }
            )
            prospective = prospective_candidates(filtered_history, target_modality)
            audit_rows.append({
                "prediction_id": prediction_id,
                "coding_id": coding_id,
                "target_modality_for_audit_only": target_modality,
                "result_restatement_candidates": result_candidates,
                "existing_result_review_decision": (review or {}).get("decision"),
                "result_restatement_resolved": result_resolved,
                "prospective_order_mention_candidates": prospective,
                "prospective_order_mention_resolved": not prospective,
                "applied_existing_redactions": applied,
                "blocking": not result_resolved or bool(prospective),
            })
            baseline = dict(record["baseline"])
            baseline["patient_history"] = filtered_history
            inputs.append({
                "prediction_id": prediction_id,
                "sequence_id": opaque_sequence_id(disease, hadm_id),
                "step_index": step,
                "baseline": baseline,
                "visible_prior_imaging": dp["visible_prior_imaging"],
            })
            relation = "initial"
            if step > 1:
                relation = "repeat" if target_modality == previous_modality else "switch"
            labels.append({
                "prediction_id": prediction_id,
                "patient_group_id": patient_key,
                "disease_stratum": disease,
                "step_index": step,
                "next_modality_primary": primary_modality(target_modality),
                "next_modality_detailed": target_modality,
                "observed_action_relation": relation,
                "previous_modality_family": previous_modality,
            })
            previous_modality = target_modality

    if len(inputs) != 433 or len(labels) != 433 or len({row["prediction_id"] for row in inputs}) != 433:
        raise AssertionError("unexpected prediction-layer cardinality")
    blocked = [row for row in audit_rows if row["blocking"]]
    result_unresolved = [row for row in audit_rows if not row["result_restatement_resolved"]]
    prospective_unresolved = [row for row in audit_rows if not row["prospective_order_mention_resolved"]]
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(INPUTS, inputs)
    write_jsonl(LABELS, labels)
    write_jsonl(AUDIT, audit_rows)
    write_json(MANIFEST, {
        "schema_version": "1.0.0-preorder-prediction-layer",
        "status": "blocked_pending_leakage_review" if blocked else "analysis_ready",
        "partition": "development_only",
        "final_test_used": False,
        "acr_used": False,
        "current_order_present_in_inputs": False,
        "current_or_future_results_present_by_construction": False,
        "input_file": str(INPUTS.relative_to(ROOT)),
        "label_file_kept_separate": str(LABELS.relative_to(ROOT)),
        "audit_file": str(AUDIT.relative_to(ROOT)),
        "counts": {
            "patients": len(patient_ids),
            "decision_points": len(inputs),
            "noninitial_decision_points": sum(row["step_index"] > 1 for row in labels),
            "blocked_decision_points": len(blocked),
            "unresolved_result_restatement_steps": len(result_unresolved),
            "unresolved_prospective_order_mention_steps": len(prospective_unresolved),
            "next_modality_primary": dict(sorted(Counter(row["next_modality_primary"] for row in labels).items())),
            "next_modality_detailed": dict(sorted(Counter(row["next_modality_detailed"] for row in labels).items())),
            "noninitial_action_relation": dict(sorted(Counter(
                row["observed_action_relation"] for row in labels if row["step_index"] > 1
            ).items())),
        },
        "release_rule": (
            "Do not annotate or model until every result-restatement and prospective-order candidate "
            "is adjudicated and all confirmed target-revealing text is removed by exact hash-bound redaction."
        ),
        "identifiability_boundary": (
            "The layer supports next modality among observed decisions and repeat-versus-switch among "
            "noninitial observed decisions. It does not identify continue-versus-stop."
        ),
    })
    print(json.dumps(read_json(MANIFEST)["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
