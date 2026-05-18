"""
llm_tau_calibrator.py

Use GPT-4o to identify the earliest step in a patient's diagnostic trajectory
at which a clinician would feel confident enough to commit to a primary diagnosis.

The entropy at that chosen step is computed via pipeline.diagnosis_distribution
(running the rubric traversal on the step's feature dict).  Averaging over a
cohort yields a calibrated tau.

Data source
-----------
Feature trajectories are extracted from results/<disease>_features.json
(one file per disease, produced by the feature-extraction pipeline).
Use `load_cohort_from_json()` to read them.

Diseases
--------
appendicitis | cholecystitis | diverticulitis | pancreatitis
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

# Pipeline imports — used only for entropy computation on the chosen step.
from pipeline.traversal_engine import run_full_traversal
from pipeline.diagnosis_distribution import compute_distribution

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIAGNOSES = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")

SYSTEM_PROMPT = """\
You are an experienced abdominal medicine clinician.
You will be shown a patient's diagnostic workup sequence —
the features known at each step as tests are completed.
The possible diagnoses are: appendicitis, cholecystitis,
diverticulitis, and pancreatitis.

Your task: identify the earliest step at which you would feel
confident enough to commit to a primary diagnosis.
"Confident" means the leading diagnosis is sufficiently supported
that further testing would be for confirmation only,
not for disambiguation between competing diagnoses.

Respond ONLY with valid JSON, no explanation outside the JSON.\
"""


# ---------------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------------

def _entropy_from_features(features: dict) -> float:
    """
    Compute H(P(d | features)) using the rubric traversal + diagnosis_distribution.
    Returns entropy in nats.  Returns log(4) (≈ 1.386) on error.
    """
    try:
        full_result = run_full_traversal(features)
        dist = compute_distribution(full_result, features)
        probs = dist.probabilities
        h = -sum(p * math.log(p) for p in probs.values() if p > 0)
        return h
    except Exception as exc:
        print(f"[llm_tau_calibrator] Entropy computation error: {exc}")
        return math.log(len(DIAGNOSES))


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(
    patient_trajectory: list[dict],
    test_sequence: list[str],
) -> str:
    """
    Build the user prompt showing per-step feature *diffs* (new or changed
    features only, compared to the previous step).

    patient_trajectory : cumulative feature dicts, one per step
    test_sequence      : human-readable label for each step (step 0 = "baseline")
    """
    lines: list[str] = []
    prev_features: dict = {}

    for step_idx, (features, test_name) in enumerate(
        zip(patient_trajectory, test_sequence)
    ):
        if step_idx == 0:
            label = "Step 0 (baseline)"
            # Show all non-False / non-zero / non-null features at baseline
            relevant = {
                k: v for k, v in features.items()
                if k != "tests_done" and v not in (False, 0, 0.0, None, "")
            }
            feat_str = (
                ", ".join(f"{k}={v}" for k, v in relevant.items())
                or "(no notable findings)"
            )
        else:
            label = f"Step {step_idx} (after {test_name})"
            new_keys = {k: v for k, v in features.items() if k not in prev_features}
            changed_keys = {
                k: v
                for k, v in features.items()
                if k in prev_features and prev_features[k] != v and k != "tests_done"
            }
            diff = {**new_keys, **changed_keys}
            relevant_diff = {
                k: v for k, v in diff.items()
                if v not in (False, 0, 0.0, None, "")
            }
            feat_str = (
                ", ".join(f"{k}={v}" for k, v in relevant_diff.items())
                or "(no new significant findings)"
            )

        lines.append(f"{label}: {feat_str}")
        prev_features = dict(features)

    lines.append(
        "\nFor each step above, decide whether you would already feel confident "
        "enough to commit to a primary diagnosis at that point.\n"
        "Respond with JSON in exactly this format:\n"
        "{\n"
        '  "confident_step": <int>,\n'
        '  "primary_diagnosis": "<diagnosis>",\n'
        '  "reasoning": "<one sentence>"\n'
        "}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def get_llm_confident_step(
    patient_trajectory: list[dict],
    test_sequence: list[str],
) -> Optional[dict]:
    """
    Ask GPT-4o to identify the earliest step at which it is confident enough
    to commit to a primary diagnosis, then compute the diagnosis entropy at
    that step via the rubric pipeline.

    Parameters
    ----------
    patient_trajectory : list[dict]
        Cumulative feature dictionaries — one per step, reflecting the full
        known feature state *after* each test.
    test_sequence : list[str]
        Human-readable name of each step (step 0 is typically "baseline").

    Returns
    -------
    dict with keys:
        "confident_step"    : int   — 0-indexed step the LLM chose
        "primary_diagnosis" : str   — one of the four diagnoses
        "reasoning"         : str   — one-sentence justification
        "tau"               : float — H(P(d|features)) at the chosen step
    Returns None on API/parse error or out-of-range step.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    user_prompt = _build_user_prompt(patient_trajectory, test_sequence)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        print(f"[llm_tau_calibrator] OpenAI API error: {exc}")
        return None

    raw_text = response.choices[0].message.content

    # --- Parse JSON ---
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        print(f"[llm_tau_calibrator] JSON parse error: {exc}")
        print(f"  Raw response: {raw_text!r}")
        return None

    # --- Validate confident_step ---
    confident_step = parsed.get("confident_step")
    n_steps = len(patient_trajectory)

    if not isinstance(confident_step, int):
        print(
            f"[llm_tau_calibrator] 'confident_step' is not an int: {confident_step!r}"
        )
        return None

    if not (0 <= confident_step < n_steps):
        print(
            f"[llm_tau_calibrator] 'confident_step' {confident_step} out of range "
            f"[0, {n_steps - 1}]."
        )
        return None

    # --- Compute entropy at the chosen step only ---
    tau = _entropy_from_features(patient_trajectory[confident_step])

    return {
        "confident_step":    confident_step,
        "primary_diagnosis": parsed.get("primary_diagnosis", ""),
        "reasoning":         parsed.get("reasoning", ""),
        "tau":               tau,
    }


# ---------------------------------------------------------------------------
# Cohort calibration
# ---------------------------------------------------------------------------

def calibrate_tau_on_cohort(
    trajectories: list[list[dict]],
    test_sequences: list[list[str]],
) -> dict:
    """
    Run get_llm_confident_step over a cohort and compute aggregate tau stats.

    Parameters
    ----------
    trajectories   : list of per-patient cumulative feature trajectories
    test_sequences : list of per-patient test-name sequences (same length)

    Returns
    -------
    dict with keys:
        "mean_tau"    : float
        "std_tau"     : float
        "per_patient" : list[dict | None]  — one entry per patient
    """
    per_patient: list[Optional[dict]] = []

    for idx, (traj, tests) in enumerate(zip(trajectories, test_sequences)):
        print(
            f"[llm_tau_calibrator] Patient {idx + 1}/{len(trajectories)} ..."
        )
        result = get_llm_confident_step(traj, tests)
        per_patient.append(result)

    valid_taus = [r["tau"] for r in per_patient if r is not None]

    if valid_taus:
        mean_tau = float(np.mean(valid_taus))
        std_tau  = float(np.std(valid_taus, ddof=1) if len(valid_taus) > 1 else 0.0)
    else:
        print("[llm_tau_calibrator] Warning: no valid results — returning NaN taus.")
        mean_tau = float("nan")
        std_tau  = float("nan")

    return {
        "mean_tau":    mean_tau,
        "std_tau":     std_tau,
        "per_patient": per_patient,
    }


# ---------------------------------------------------------------------------
# Data loader  (reads results/<disease>_features.json)
# ---------------------------------------------------------------------------

def load_cohort_from_json(
    disease: str,
    results_dir: str | Path = "results",
    max_patients: Optional[int] = None,
) -> tuple[list[list[dict]], list[list[str]]]:
    """
    Load patient trajectories from results/<disease>_features.json.

    Each patient record is a list of step dicts with keys:
        step_index, step_label, test_key, note_id, features

    Parameters
    ----------
    disease      : one of appendicitis | cholecystitis | diverticulitis | pancreatitis
    results_dir  : path to the results directory (default: "results")
    max_patients : if set, load at most this many patients

    Returns
    -------
    (trajectories, test_sequences)
        trajectories   — list of per-patient list[dict] (cumulative features)
        test_sequences — list of per-patient list[str]  (human-readable step labels)
    """
    path = Path(results_dir) / f"{disease}_features.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    patients_dict: dict = raw["results"]
    patient_ids = list(patients_dict.keys())
    if max_patients is not None:
        patient_ids = patient_ids[:max_patients]

    trajectories: list[list[dict]] = []
    test_sequences: list[list[str]] = []

    for pid in patient_ids:
        steps: list[dict] = patients_dict[pid]
        # Sort by step_index just to be safe
        steps = sorted(steps, key=lambda s: s["step_index"])

        traj = [s["features"] for s in steps]
        # Step 0: "baseline", subsequent steps: step_label (e.g. "CT ABD & PELVIS…")
        labels = []
        for s in steps:
            if s["step_index"] == 0:
                labels.append("baseline")
            else:
                labels.append(s.get("step_label") or s.get("test_key") or f"step_{s['step_index']}")
        trajectories.append(traj)
        test_sequences.append(labels)

    return trajectories, test_sequences


# ---------------------------------------------------------------------------
# Mock test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Construct a tiny 3-step trajectory matching the real feature schema ---
    _base = {
        "pain_location": "RLQ",
        "pain_migration_to_RLQ": False,
        "epigastric_radiating_to_back": False,
        "bowel_habit_change": False,
        "symptom_duration_over_72h": False,
        "anorexia": True,
        "nausea_vomiting": True,
        "alcohol_history": False,
        "gallstone_history": False,
        "prior_diverticular_disease": False,
        "murphys_sign": False,
        "RLQ_tenderness": True,
        "rebound_tenderness": False,
        "RUQ_mass": False,
        "fever_temp_ge_37_3": True,
        "fever_temp_ge_38": False,
        "peritoneal_signs": False,
        "WBC_gt_10k": False,
        "WBC_gt_18k": False,
        "left_shift": False,
        "CRP_elevated": False,
        "lipase_ge_3xULN": False,
        "bilirubin_elevated": False,
        "LFTs_elevated": False,
        "US_appendix_inflamed": False,
        "US_gallstones": False,
        "CT_appendicitis_positive": False,
        "CT_cholecystitis_positive": False,
        "CT_diverticulitis_confirmed": False,
        "CT_pancreatitis_positive": False,
        "tests_done": ["Lab_Panel"],
    }

    _after_cbc = {**_base, "WBC_gt_10k": True, "CRP_elevated": True,
                  "tests_done": ["Lab_Panel", "CBC"]}

    _after_ct  = {**_after_cbc, "CT_appendicitis_positive": True,
                  "tests_done": ["Lab_Panel", "CBC", "CT_Abdomen"]}

    mock_trajectory   = [_base, _after_cbc, _after_ct]
    mock_test_sequence = ["baseline", "CBC", "CT Abdomen"]

    print("=" * 60)
    print("Mock test: get_llm_confident_step")
    print("=" * 60)

    result = get_llm_confident_step(
        patient_trajectory=mock_trajectory,
        test_sequence=mock_test_sequence,
    )

    if result is not None:
        print(json.dumps(result, indent=2))
    else:
        print("get_llm_confident_step returned None (check errors above).")
