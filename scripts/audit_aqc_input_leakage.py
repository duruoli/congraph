"""Preflight audit for current-result restatements inside the HPI baseline.

The structured radiology mask hides the current report, but the derived
``Patient History`` field can restate a pre-admission ED study after it was
performed.  This audit compares HPI sentences with each currently masked
report.  It is deliberately high-recall: candidates require manual review.
A confirmed sentence can be removed in memory only when the review binds its
exact text and SHA-256; the raw source file is never changed.
"""
from __future__ import annotations

import argparse
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

from scripts.build_masked_view import RAW, build_record, load_lab_map  # noqa: E402

SCHEMA_VERSION = "1.0.0-aqc-input-leakage-preflight"
ALGORITHM_VERSION = "1.1.0-hpi-current-report-overlap-reviewed-redaction"

STOPWORDS = {
    "about", "after", "again", "also", "and", "been", "before", "being", "between",
    "current", "during", "from", "given", "having", "history", "into", "noted",
    "patient", "prior", "report", "reported", "showed", "shows", "study", "that",
    "their", "there", "these", "they", "this", "those", "underwent", "were", "with",
    "without", "the", "abd", "abdomen", "abdominal", "pelvis", "finding", "findings", "examination",
}

RESULT_CUE = re.compile(
    r"\b(show(?:ed|s)?|demonstrat(?:e|ed|es)|reveal(?:ed|s)?|found|noted|"
    r"prelim(?:inary)?\s+read|read\s+(?:as|showed)|evidence\s+of|"
    r"normal|abnormal|dilat(?:ed|ion)|distend(?:ed|sion)|stone(?:s)?|sludge|"
    r"fluid|thicken(?:ed|ing)|collection|mass|abscess|obstruction)\b",
    re.I,
)

MODALITY_PATTERNS = {
    "Ultrasound": re.compile(r"\b(ultrasound|sonograph\w*|u\s*/?\s*s|ruq\s+us)\b", re.I),
    "CT": re.compile(r"\b(ct|computed\s+tomography|cat\s+scan)\b", re.I),
    "MRI": re.compile(r"\b(mri|mr\s+abdomen|magnetic\s+resonance)\b", re.I),
    "MRCP": re.compile(r"\b(mrcp|mr\s+cholangiopancreatography)\b", re.I),
    "CTU": re.compile(r"\b(ctu|ct\s+urogram|ct\s+urography)\b", re.I),
}

# Specific concepts improve recall for short HPI summaries such as
# "normal pancreas on CT prelim read" while leaving final judgment to humans.
CONCEPTS = {
    "abscess", "appendix", "biliary", "bowel", "cholecystostomy", "collection",
    "duct", "gallbladder", "hepatic", "ileus", "inflammation", "lymph", "murphy",
    "necrotic", "obstruction", "pancreas", "pericholecystic", "porta", "sludge",
    "stone", "stones", "stranding", "thickening", "vasculitis",
}

REVIEW_DECISIONS = {
    "confirmed_current_result_leak",
    "cleared_prior_or_external_study",
    "unclear_rebuild_or_exclude",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z][a-z0-9]+|\d+(?:\.\d+)?", normalize(text))
        if len(token) >= 3 and token not in STOPWORDS
    }


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", normalize(text)))


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+|[\r\n]+", str(text))
    return [re.sub(r"\s+", " ", piece).strip() for piece in pieces if piece.strip()]


def modality_name(ordered: str) -> str:
    first = str(ordered).split(" ", 1)[0]
    return first if first in MODALITY_PATTERNS else ""


def sentence_sha256(sentence: str) -> str:
    return hashlib.sha256(sentence.encode("utf-8")).hexdigest()


def score_sentence(sentence: str, report: str, modality: str) -> dict[str, Any]:
    sentence_tokens = tokens(sentence)
    report_tokens = tokens(report)
    overlap = sorted(sentence_tokens & report_tokens)
    coverage = len(overlap) / len(sentence_tokens) if sentence_tokens else 0.0
    shared_numbers = sorted(numbers(sentence) & numbers(report))
    shared_concepts = sorted(set(overlap) & CONCEPTS)
    modality_match = bool(MODALITY_PATTERNS.get(modality, re.compile(r"$^" )).search(sentence))
    result_cue = bool(RESULT_CUE.search(sentence))
    candidate = bool(
        modality_match
        and result_cue
        and (
            (len(overlap) >= 3 and coverage >= 0.30)
            or (len(overlap) >= 2 and coverage >= 0.45)
            or len(shared_concepts) >= 2
            or (shared_numbers and shared_concepts)
        )
    )
    return {
        "candidate": candidate,
        "modality_match": modality_match,
        "result_cue": result_cue,
        "token_coverage_against_current_report": round(coverage, 4),
        "shared_tokens": overlap,
        "shared_numbers": shared_numbers,
        "shared_concepts": shared_concepts,
    }


def load_reviews(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    value = read_json(path)
    if not isinstance(value, dict) or value.get("algorithm_version") != ALGORITHM_VERSION:
        raise ValueError(f"review algorithm_version must equal {ALGORITHM_VERSION}")
    rows = value.get("reviews") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise ValueError("review file must contain a reviews list")
    reviews: dict[str, dict[str, Any]] = {}
    for row in rows:
        coding_id = row.get("coding_id") if isinstance(row, dict) else None
        decision = row.get("decision") if isinstance(row, dict) else None
        if not isinstance(coding_id, str) or decision not in REVIEW_DECISIONS:
            raise ValueError("each review needs coding_id and a supported decision")
        if coding_id in reviews:
            raise ValueError(f"duplicate review coding_id: {coding_id}")
        reviews[coding_id] = row
    return reviews


def review_redactions(review: dict[str, Any]) -> list[dict[str, str]]:
    value = review.get("history_redactions", review.get("history_redaction", []))
    rows = value if isinstance(value, list) else [value]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("history_redactions must be an object or list of objects")
    return rows


def apply_reviewed_redactions(
    history: str,
    disease: str,
    hadm_id: int,
    reviews: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    """Apply exact, hash-bound confirmed redactions for one patient in memory."""
    redactions: dict[str, dict[str, str]] = {}
    prefix = f"{disease}:{hadm_id}:s"
    for coding_id, review in reviews.items():
        if not coding_id.startswith(prefix) or review.get("decision") != "confirmed_current_result_leak":
            continue
        for row in review_redactions(review):
            exact_text = row.get("exact_text")
            expected_hash = row.get("sentence_sha256")
            replacement = row.get("replacement", "")
            if not isinstance(exact_text, str) or not exact_text:
                raise ValueError(f"confirmed review lacks exact_text redaction: {coding_id}")
            if expected_hash != sentence_sha256(exact_text):
                raise ValueError(f"redaction SHA-256 mismatch: {coding_id}")
            if not isinstance(replacement, str):
                raise ValueError(f"redaction replacement must be a string: {coding_id}")
            old = redactions.get(exact_text)
            if old is not None and old["replacement"] != replacement:
                raise ValueError(f"conflicting redactions for the same HPI sentence: {coding_id}")
            redactions[exact_text] = {
                "coding_id": coding_id,
                "exact_text": exact_text,
                "sentence_sha256": expected_hash,
                "replacement": replacement,
            }
    # Candidate sentences are whitespace-normalized, so establish the same
    # deterministic representation before requiring an exact-once match.
    filtered = re.sub(r"\s+", " ", history).strip()
    applied: list[dict[str, str]] = []
    for exact_text, row in redactions.items():
        if filtered.count(exact_text) != 1:
            raise ValueError(
                f"redaction text must occur exactly once in raw HPI: {row['coding_id']}"
            )
        filtered = filtered.replace(exact_text, row["replacement"], 1)
        applied.append(row)
    filtered = re.sub(r"\s+", " ", filtered).strip()
    return filtered, applied


def audit_manifest(manifest_path: Path, review_path: Path | None = None) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    patients = manifest.get("patients") if isinstance(manifest, dict) else None
    if not isinstance(patients, list):
        raise ValueError("manifest must contain a patients list")
    reviews = load_reviews(review_path)
    if review_path is not None:
        review_document = read_json(review_path)
        bound_manifest = review_document.get("manifest")
        if not isinstance(bound_manifest, str):
            raise ValueError("review file must bind a manifest path")
        bound_path = Path(bound_manifest)
        if not bound_path.is_absolute():
            bound_path = ROOT / bound_path
        if bound_path.resolve() != manifest_path.resolve():
            raise ValueError("review file is bound to a different manifest")
    frames = {disease: pd.read_csv(ROOT / path) for disease, path in RAW.items()}
    labmap = load_lab_map()
    steps: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    candidate_hashes: dict[str, set[str]] = {}
    histories: dict[tuple[str, int], str] = {}
    reports_by_id: dict[str, str] = {}

    for patient in patients:
        disease = str(patient["disease"])
        hadm_id = int(patient["hadm_id"])
        row = frames[disease][frames[disease]["hadm_id"] == hadm_id]
        if row.empty:
            raise ValueError(f"raw record not found: {disease}/{hadm_id}")
        record = build_record(disease, hadm_id, row.iloc[0], labmap)
        history = record["baseline"]["patient_history"]
        histories[(disease, hadm_id)] = history
        for dp in record["decision_points"]:
            coding_id = f"{disease}:{hadm_id}:s{dp['step']}"
            reports_by_id[coding_id] = str(dp["masked_result_of_this_test"])
            modality = modality_name(str(dp["ordered"]))
            candidates = []
            for sentence in split_sentences(history):
                score = score_sentence(sentence, str(dp["masked_result_of_this_test"]), modality)
                if score.pop("candidate"):
                    candidates.append({
                        "sentence": sentence,
                        "sentence_sha256": sentence_sha256(sentence),
                        **score,
                    })
            if candidates:
                candidate_ids.add(coding_id)
                candidate_hashes[coding_id] = {
                    candidate["sentence_sha256"] for candidate in candidates
                }
            review = reviews.get(coding_id)
            steps.append({
                "coding_id": coding_id,
                "ordered": dp["ordered"],
                "n_visible_prior_imaging": len(dp.get("visible_prior_imaging") or []),
                "n_candidates": len(candidates),
                "candidates": candidates,
                "manual_review": review,
            })

    unused_reviews = sorted(set(reviews) - {row["coding_id"] for row in steps})
    if unused_reviews:
        raise ValueError(f"review file contains coding IDs outside the manifest: {unused_reviews}")
    noncandidate_reviews = sorted(set(reviews) - candidate_ids)
    if noncandidate_reviews:
        raise ValueError(f"review file contains steps not flagged by this algorithm: {noncandidate_reviews}")
    for coding_id, review in reviews.items():
        reviewed_hashes = review.get("sentence_sha256s")
        if reviewed_hashes is None and isinstance(review.get("sentence_sha256"), str):
            reviewed_hashes = [review["sentence_sha256"]]
        if not isinstance(reviewed_hashes, list) or set(reviewed_hashes) != candidate_hashes[coding_id]:
            raise ValueError(
                f"review sentence hashes do not match current candidates for {coding_id}"
            )
        if review.get("decision") == "confirmed_current_result_leak":
            redaction_hashes = {
                row.get("sentence_sha256") for row in review_redactions(review)
            }
            if redaction_hashes != candidate_hashes[coding_id]:
                raise ValueError(
                    f"confirmed review redactions do not cover its candidates: {coding_id}"
                )

    filtered_histories: dict[tuple[str, int], str] = {}
    applied_by_patient: dict[tuple[str, int], list[dict[str, str]]] = {}
    for (disease, hadm_id), history in histories.items():
        filtered, applied = apply_reviewed_redactions(history, disease, hadm_id, reviews)
        filtered_histories[(disease, hadm_id)] = filtered
        applied_by_patient[(disease, hadm_id)] = applied

    effective_unresolved: list[str] = []
    for step in steps:
        disease, hadm_text, _ = step["coding_id"].split(":", 2)
        patient_key = (disease, int(hadm_text))
        filtered = filtered_histories[patient_key]
        modality = modality_name(str(step["ordered"]))
        remaining = [
            sentence for sentence in split_sentences(filtered)
            if score_sentence(sentence, reports_by_id[step["coding_id"]], modality)["candidate"]
        ]
        step["effective_history_sha256"] = sentence_sha256(filtered)
        step["applied_history_redactions"] = applied_by_patient[patient_key]
        step["effective_candidate_sentences"] = remaining
        decision = (step.get("manual_review") or {}).get("decision")
        if remaining and decision != "cleared_prior_or_external_study":
            effective_unresolved.append(step["coding_id"])
    unreviewed = sorted(candidate_ids - set(reviews))
    confirmed = sorted(
        coding_id for coding_id in candidate_ids
        if reviews.get(coding_id, {}).get("decision") == "confirmed_current_result_leak"
    )
    unclear = sorted(
        coding_id for coding_id in candidate_ids
        if reviews.get(coding_id, {}).get("decision") == "unclear_rebuild_or_exclude"
    )
    cleared = sorted(
        coding_id for coding_id in candidate_ids
        if reviews.get(coding_id, {}).get("decision") == "cleared_prior_or_external_study"
    )
    resolved = sorted(set(confirmed) - set(effective_unresolved))
    unresolved_confirmed = sorted(set(confirmed) & set(effective_unresolved))
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "manifest": str(manifest_path),
        "review": str(review_path) if review_path else None,
        "n_patients": len(patients),
        "n_steps": len(steps),
        "n_candidate_steps": len(candidate_ids),
        "n_unreviewed_candidate_steps": len(unreviewed),
        "n_confirmed_leaks": len(confirmed),
        "n_resolved_by_redaction": len(resolved),
        "n_unresolved_confirmed_leaks": len(unresolved_confirmed),
        "n_unclear_rebuild_or_exclude": len(unclear),
        "n_cleared_prior_or_external": len(cleared),
        "blocking": bool(unreviewed or unclear or effective_unresolved),
        "unreviewed_candidate_steps": unreviewed,
        "confirmed_leaks": confirmed,
        "resolved_by_redaction": resolved,
        "unresolved_confirmed_leaks": unresolved_confirmed,
        "effective_unresolved_candidate_steps": sorted(set(effective_unresolved)),
        "unclear_rebuild_or_exclude": unclear,
        "cleared_prior_or_external": cleared,
        "steps": steps,
        "interpretation": (
            "High-recall screening only. A candidate is not automatically a leak. "
            "Confirmed candidates require exact reviewed redaction; unclear cases must be "
            "rebuilt or excluded before annotation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-blocking", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    review_path = None
    if args.review:
        review_path = args.review if args.review.is_absolute() else ROOT / args.review
    report = audit_manifest(manifest_path, review_path)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output_path = args.output if args.output.is_absolute() else ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text, end="")
    if args.fail_on_blocking and report["blocking"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
