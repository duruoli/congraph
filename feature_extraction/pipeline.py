"""pipeline.py — Orchestrates algo + LLM extraction into a feature-vector sequence.

Public API
----------
extract_patient_steps(row, client, *, run_llm=True) → list[ExtractionStep]

    Given one raw CSV row (dict), returns an ordered list of ExtractionStep
    objects.  Each step holds a complete, cumulative feature dict that can be
    fed directly into ClinicalSession or the rubric graph.

    Step 0 : HPI + Physical Exam + Basic Labs
    Step k  : Step k-1 features  +  findings from the k-th radiology report

Usage
-----
    import csv
    from openai import OpenAI
    from feature_extraction.pipeline import extract_patient_steps

    client = OpenAI(api_key="sk-...")

    with open("raw_data/appendicitis_hadm_info_first_diag.csv") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    steps = extract_patient_steps(row, client)
    for s in steps:
        print(s.step_label, "→", s.features["pain_location"])
"""

from __future__ import annotations

import copy
import sys
import os
from dataclasses import dataclass, field
from typing import Optional

# Allow imports from the workspace root (parent of feature_extraction/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from feature_schema import default_features
from feature_extraction.algo_extractor import (
    extract_algo_features,
    extract_radiology_tests,
)
from feature_extraction.llm_extractor import (
    extract_step0_llm,
    extract_imaging_llm,
)


# ---------------------------------------------------------------------------
# Data class for one extraction step
# ---------------------------------------------------------------------------

@dataclass
class ExtractionStep:
    """
    Snapshot of the cumulative feature dict at one clinical step.

    Attributes
    ----------
    step_index : int
        0 for the initial step (HPI + PE + Labs); 1, 2, … for each
        subsequent radiology report.
    step_label : str
        Human-readable description, e.g. "Step 1 — CT: CT ABD & PELVIS".
    test_key : str | None
        The VALID_TESTS string added at this step (None for step 0).
    note_id : str
        MIMIC Note ID for imaging steps; empty string for step 0.
    features : dict
        Complete, cumulative feature dict (all 53 keys from feature_schema).
    """
    step_index: int
    step_label: str
    test_key:   Optional[str]
    note_id:    str
    features:   dict = field(repr=False)


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def _merge_imaging(base: dict, updates: dict) -> None:
    """
    Merge imaging feature updates into base in-place.

    Booleans are OR-combined (a positive finding once seen stays True).
    Numeric fields (CTSI_score) take the maximum seen so far.
    """
    for key, val in updates.items():
        if isinstance(val, bool):
            base[key] = base.get(key, False) or val
        elif isinstance(val, float):
            base[key] = max(base.get(key, 0.0), val)
        else:
            base[key] = val


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_patient_steps(
    row: dict,
    client: OpenAI,
    *,
    run_llm: bool = True,
) -> list[ExtractionStep]:
    """
    Build the full feature-vector sequence for one patient CSV row.

    Parameters
    ----------
    row : dict
        A single CSV row as produced by csv.DictReader.
    client : OpenAI
        Initialised OpenAI client instance.
    run_llm : bool
        If False, skip all LLM calls and return algo-only feature vectors
        (useful for fast unit tests or cost-estimation runs).

    Returns
    -------
    list[ExtractionStep]
        Ordered list: step 0 (initial) followed by one entry per radiology
        report in the chronological order stored in the CSV.
    """
    # ── Initialise from schema defaults ──────────────────────────────────────
    features = default_features()

    # ── Algorithmic extraction: labs + vitals + tests_done ───────────────────
    algo_feats = extract_algo_features(row)
    features.update(algo_feats)

    # ── LLM extraction: HPI + Physical Examination ───────────────────────────
    if run_llm:
        llm_feats = extract_step0_llm(
            client,
            patient_history=str(row.get("Patient History", "") or ""),
            physical_exam  =str(row.get("Physical Examination", "") or ""),
        )
        features.update(llm_feats)

    # ── Step 0 snapshot ───────────────────────────────────────────────────────
    steps: list[ExtractionStep] = [
        ExtractionStep(
            step_index=0,
            step_label="Step 0 — HPI + Physical Exam + Basic Labs",
            test_key=None,
            note_id="",
            features=copy.deepcopy(features),
        )
    ]

    # ── Steps 1+: iterate over radiology reports ─────────────────────────────
    radiology_entries = extract_radiology_tests(row)

    for i, entry in enumerate(radiology_entries, start=1):
        modality  = entry["modality"]
        exam_name = entry["exam_name"]
        test_key  = entry["test_key"]
        report    = entry["report"]
        note_id   = entry["note_id"]

        # Update tests_done (append if not already recorded)
        if test_key and test_key not in features["tests_done"]:
            features["tests_done"].append(test_key)

        # LLM extraction for this imaging report
        if run_llm and report.strip():
            img_feats = extract_imaging_llm(client, report, modality, exam_name)
            _merge_imaging(features, img_feats)

        steps.append(ExtractionStep(
            step_index=i,
            step_label=f"Step {i} — {modality}: {exam_name}",
            test_key=test_key,
            note_id=note_id,
            features=copy.deepcopy(features),
        ))

    return steps


# ---------------------------------------------------------------------------
# Convenience: process an entire CSV file
# ---------------------------------------------------------------------------

def extract_all_patients(
    csv_path: str,
    client: OpenAI,
    *,
    run_llm: bool = True,
    limit: Optional[int] = None,
) -> dict[str, list[ExtractionStep]]:
    """
    Process every row in a CSV file and return a mapping of
    hadm_id → list[ExtractionStep].

    Parameters
    ----------
    csv_path : str
        Path to one of the four disease CSVs.
    client : OpenAI
        Initialised OpenAI client.
    run_llm : bool
        Toggle LLM extraction (default True).
    limit : int | None
        If set, process at most this many patients (useful for testing).

    Returns
    -------
    dict[str, list[ExtractionStep]]
        Keys are hadm_id strings; values are the step sequences.
    """
    import csv

    results: dict[str, list[ExtractionStep]] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            hadm_id = str(row.get("hadm_id", i))
            results[hadm_id] = extract_patient_steps(row, client, run_llm=run_llm)

    return results
