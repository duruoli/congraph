"""demo_pipeline.py  —  Step 9: End-to-end pipeline validation

Two modes:
  1. SCENARIO SWEEP (default)
       Runs every toy-patient scenario step-by-step through the full pipeline.
       Asserts that the primary diagnosis matches the scenario's disease at the
       final step (step3), and that the terminal node matches EXPECTED_TERMINALS.

  2. NARRATIVE DEMO  (--demo  or  --demo C1)
       Plays one patient through the session in a story-like fashion,
       printing full AssessmentState reports at each step.

Usage
-----
    python demo_pipeline.py              # scenario sweep (all 6 patients)
    python demo_pipeline.py --demo       # narrative demo (C1 + P2)
    python demo_pipeline.py --demo C1    # single patient narrative
    python demo_pipeline.py --demo P2
    python demo_pipeline.py --demo D2
"""

from __future__ import annotations

import sys

from clinical_session import ClinicalSession
from toy_patients import SCENARIOS, EXPECTED_TERMINALS
from diagnosis_distribution import compute_distribution
from traversal_engine import run_full_traversal


# ---------------------------------------------------------------------------
# Scenario → expected primary diagnosis
# ---------------------------------------------------------------------------

_EXPECTED_PRIMARY: dict[str, str] = {
    "C1_cholecystitis_mild":            "cholecystitis",
    "C2_cholecystitis_moderate":        "cholecystitis",
    "P1_pancreatitis_mild":             "pancreatitis",
    "P2_pancreatitis_severe":           "pancreatitis",
    "D1_diverticulitis_uncomplicated":  "diverticulitis",
    "D2_diverticulitis_hinchey2":       "diverticulitis",
}

_SCENARIO_DISEASE: dict[str, str] = {
    "C1_cholecystitis_mild":            "cholecystitis",
    "C2_cholecystitis_moderate":        "cholecystitis",
    "P1_pancreatitis_mild":             "pancreatitis",
    "P2_pancreatitis_severe":           "pancreatitis",
    "D1_diverticulitis_uncomplicated":  "diverticulitis",
    "D2_diverticulitis_hinchey2":       "diverticulitis",
}


# ---------------------------------------------------------------------------
# Mode 1: Scenario Sweep
# ---------------------------------------------------------------------------

def run_sweep() -> bool:
    """
    For each scenario × step, assert:
      (a) terminal node matches EXPECTED_TERMINALS
      (b) at step3, primary diagnosis matches expected disease

    Returns True if all assertions pass.
    """
    total = passed = 0
    failures: list[str] = []

    for scenario, steps in SCENARIOS.items():
        disease  = _SCENARIO_DISEASE[scenario]
        expected_steps = EXPECTED_TERMINALS.get(scenario, {})

        for step, features in steps.items():
            total += 1
            session = ClinicalSession(features)
            state   = session.assess()

            # (a) Terminal node
            actual_terminal  = state.traversal.diseases[disease].terminal_node
            expected_terminal = expected_steps.get(step)
            ok_terminal = (actual_terminal == expected_terminal)

            # (b) Primary diagnosis at step3 (final step)
            ok_primary = True
            if step == "step3":
                ok_primary = (state.primary_diagnosis == _EXPECTED_PRIMARY[scenario])

            ok = ok_terminal and ok_primary
            icon = "PASS" if ok else "FAIL"

            label = f"{scenario} / {step}"
            detail = (
                f"terminal: {actual_terminal!r}=={expected_terminal!r}"
                + (f"  primary: {state.primary_diagnosis!r}" if step == "step3" else "")
            )
            print(f"  [{icon}] {label:<45s}  {detail}")
            print(f"          dist: {state.distribution.summary_line()}")

            if ok:
                passed += 1
            else:
                failures.append(label)

    print(f"\n{'─'*64}")
    print(f"  Results: {passed}/{total} passed")
    if failures:
        print(f"  Failed : {', '.join(failures)}")
    print(f"{'─'*64}\n")
    return passed == total


# ---------------------------------------------------------------------------
# Mode 2: Narrative Demo
# ---------------------------------------------------------------------------

_DEMO_PATIENTS = {
    "C1": ("C1_cholecystitis_mild",
           "C1 — 55-year-old woman, Grade I Cholecystitis"),
    "C2": ("C2_cholecystitis_moderate",
           "C2 — 68-year-old man, Grade II Cholecystitis"),
    "P1": ("P1_pancreatitis_mild",
           "P1 — 42-year-old man, Mild Acute Pancreatitis"),
    "P2": ("P2_pancreatitis_severe",
           "P2 — 71-year-old man, Severe Acute Pancreatitis"),
    "D1": ("D1_diverticulitis_uncomplicated",
           "D1 — 58-year-old man, Uncomplicated Diverticulitis"),
    "D2": ("D2_diverticulitis_hinchey2",
           "D2 — 72-year-old woman, Hinchey II Diverticulitis"),
}

_STEP_LABEL = {
    "step0": "Step 0 — HPI + Physical Examination only",
    "step1": "Step 1 — + Basic Labs (Lab_Panel)",
    "step2": "Step 2 — + Abdominal Ultrasound",
    "step3": "Step 3 — + CT Abdomen/Pelvis",
}


def run_demo(patient_key: str = "C1") -> None:
    key = patient_key.upper()
    if key not in _DEMO_PATIENTS:
        print(f"Unknown patient key {key!r}. Choose from: {list(_DEMO_PATIENTS)}")
        return

    scenario_name, title = _DEMO_PATIENTS[key]
    steps = SCENARIOS[scenario_name]

    print(f"\n{'█'*64}")
    print(f"  PATIENT DEMO: {title}")
    print(f"{'█'*64}")

    for step_name, features in steps.items():
        print(f"\n  {'─'*62}")
        print(f"  {_STEP_LABEL[step_name]}")
        print(f"  {'─'*62}")
        session = ClinicalSession(features)
        state   = session.assess()
        state.print_report()
        print(f"  Summary: {state.summary()}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--demo" in args:
        idx = args.index("--demo")
        patient_keys = args[idx + 1:]
        if not patient_keys:
            # Default: show C1 and P2 (one cholecystitis + one pancreatitis)
            run_demo("C1")
            run_demo("P2")
        else:
            for key in patient_keys:
                run_demo(key)
    else:
        success = run_sweep()
        sys.exit(0 if success else 1)
