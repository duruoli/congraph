"""test_extraction.py — Quick smoke-test: run feature extraction on 5 patients.

Usage
-----
    conda run -n congraph python test_extraction.py [--disease appendicitis|cholecystitis|diverticulitis|pancreatitis] [--n 5] [--no-llm]

Requires OPENAI_API_KEY in environment (or pass via --api-key).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import textwrap
from pathlib import Path

# ── ensure workspace root on path ──────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from openai import OpenAI
from feature_extraction.pipeline import extract_patient_steps, ExtractionStep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Feature extraction smoke-test")
    p.add_argument(
        "--disease",
        default="cholecystitis",
        choices=["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"],
        help="Which disease CSV to use (default: cholecystitis)",
    )
    p.add_argument("--n", type=int, default=5, help="Number of patients to process (default 5)")
    p.add_argument("--api-key", default=None, help="OpenAI API key (overrides env var)")
    p.add_argument("--no-llm", action="store_true", help="Skip LLM calls (algo-only mode)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_BOOL_TRUE  = "✓"
_BOOL_FALSE = "·"

_GROUP_LABELS = {
    "Pain": [
        "pain_location",
        "pain_migration_to_RLQ",
        "epigastric_radiating_to_back",
        "bowel_habit_change",
        "symptom_duration_over_72h",
    ],
    "HPI symptoms": [
        "anorexia",
        "nausea_vomiting",
        "alcohol_history",
        "gallstone_history",
        "prior_diverticular_disease",
    ],
    "Physical exam": [
        "murphys_sign",
        "RLQ_tenderness",
        "rebound_tenderness",
        "RUQ_mass",
        "fever_temp_ge_37_3",
        "fever_temp_ge_38",
        "peritoneal_signs",
        "impaired_mental_status",
    ],
    "Labs": [
        "WBC_gt_10k",
        "WBC_gt_18k",
        "left_shift",
        "CRP_elevated",
        "CRP_gt_200",
        "lipase_ge_3xULN",
        "BUN_gt_25",
        "bilirubin_elevated",
        "LFTs_elevated",
        "creatinine_elevated",
        "beta_hCG_positive",
        "SIRS_criteria_ge_2",
    ],
    "Demographics": [
        "age_gt_60",
        "is_female_reproductive_age",
    ],
    "Ultrasound": [
        "US_appendix_inflamed",
        "US_gallstones",
        "US_GB_wall_thickening",
        "US_pericholecystic_fluid",
        "US_sonographic_murphys",
    ],
    "CT": [
        "CT_appendicitis_positive",
        "CT_perforation_abscess",
        "CT_cholecystitis_positive",
        "CT_GB_severe_findings",
        "CT_diverticulitis_confirmed",
        "CT_diverticulitis_complicated",
        "CT_phlegmon",
        "CT_abscess_lt_3cm",
        "CT_abscess_ge_3cm",
        "CT_purulent_peritonitis",
        "CT_fecal_peritonitis",
        "CT_pancreatitis_positive",
        "CTSI_score",
    ],
    "Other imaging": [
        "cholecystitis_additional_imaging_positive",
        "pleural_effusion_on_imaging",
    ],
    "Organ dysfunction": [
        "has_organ_dysfunction",
        "organ_failure_transient",
        "organ_failure_persistent",
        "local_complications_pancreatitis",
    ],
}


def _fmt_val(v) -> str:
    if isinstance(v, bool):
        return _BOOL_TRUE if v else _BOOL_FALSE
    if isinstance(v, float):
        return f"{v:.1f}"
    if isinstance(v, list):
        return ", ".join(v) if v else "(none)"
    return str(v)


def print_step(step: ExtractionStep) -> None:
    f = step.features
    print(f"\n  ── {step.step_label} ──")
    print(f"     tests_done : {_fmt_val(f['tests_done'])}")
    for group, keys in _GROUP_LABELS.items():
        active = []
        for k in keys:
            v = f.get(k)
            if v is None:
                continue
            # For booleans, only print True entries; for others always print
            if isinstance(v, bool):
                if v:
                    active.append(f"{k}={_BOOL_TRUE}")
            elif isinstance(v, float) and v == 0.0:
                pass
            else:
                active.append(f"{k}={_fmt_val(v)}")
        if active:
            line = "  ".join(active)
            wrapped = textwrap.fill(line, width=90, subsequent_indent=" " * 22)
            print(f"     [{group:>16s}] {wrapped}")


def print_patient(hadm_id: str, steps: list[ExtractionStep], discharge_dx: str) -> None:
    print("\n" + "=" * 90)
    print(f"  Patient: {hadm_id}   Discharge Dx: {discharge_dx[:80]}")
    print(f"  Steps: {len(steps)}  (0 = HPI/PE/Labs, 1+ = each radiology report)")
    for step in steps:
        print_step(step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.no_llm:
        print("ERROR: No OpenAI API key found.\n"
              "  Set OPENAI_API_KEY environment variable, or pass --api-key <key>,\n"
              "  or run with --no-llm to skip LLM calls.")
        sys.exit(1)

    client = OpenAI(api_key=api_key) if api_key else None

    csv_path = os.path.join(
        str(_REPO_ROOT),
        "data",
        "raw_data",
        f"{args.disease}_hadm_info_first_diag.csv",
    )
    print(f"\nLoading: {csv_path}")
    print(f"Mode   : {'algo-only (no LLM)' if args.no_llm else 'full LLM (gpt-4o)'}")
    print(f"Patients: {args.n}\n")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if i >= args.n:
                break
            rows.append(row)

    for i, row in enumerate(rows, 1):
        hadm_id = str(row.get("hadm_id", f"row_{i}"))
        discharge_dx = str(row.get("Discharge Diagnosis", ""))
        print(f"\n[{i}/{len(rows)}] Processing hadm_id={hadm_id} ...", flush=True)

        try:
            steps = extract_patient_steps(
                row,
                client,
                run_llm=not args.no_llm,
            )
            print_patient(hadm_id, steps, discharge_dx)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 90)
    print("Done.")


if __name__ == "__main__":
    main()
