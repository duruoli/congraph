"""run_feature_extraction.py — Full-scale feature extraction for all diseases.

Processes one or all four disease CSVs with the feature extraction pipeline
and saves the results as JSON files (one per disease).

Usage
-----
    python run_feature_extraction.py [OPTIONS]

Options
-------
    --disease   {appendicitis,cholecystitis,diverticulitis,pancreatitis,all}
                Which disease to process. Default: all
    --output-dir PATH
                Directory to write result JSON files. Default: results/feature_extraction
    --limit INT
                Max patients per disease (omit for all). Useful for testing.
    --no-llm    Skip LLM calls; run algo-only extraction.
    --api-key   OpenAI API key (falls back to OPENAI_API_KEY env var).

Output
------
    <output-dir>/<disease>_features.json
        A JSON object mapping hadm_id → list of step dicts, where each step dict
        contains: step_index, step_label, test_key, note_id, features.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from feature_extraction.pipeline import extract_patient_steps, ExtractionStep

DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def step_to_dict(step: ExtractionStep) -> dict:
    return {
        "step_index": step.step_index,
        "step_label": step.step_label,
        "test_key":   step.test_key,
        "note_id":    step.note_id,
        "features":   step.features,
    }


# ---------------------------------------------------------------------------
# Per-disease processing
# ---------------------------------------------------------------------------

def process_disease(
    disease: str,
    client: OpenAI | None,
    output_dir: Path,
    *,
    run_llm: bool = True,
    limit: int | None = None,
) -> None:
    csv_path = os.path.join(RAW_DATA_DIR, f"{disease}_hadm_info_first_diag.csv")
    out_path = output_dir / f"{disease}_features.json"

    print(f"\n{'=' * 70}")
    print(f"  Disease : {disease}")
    print(f"  Input   : {csv_path}")
    print(f"  Output  : {out_path}")
    print(f"  Mode    : {'algo-only' if not run_llm else 'LLM (gpt-4o)'}")
    if limit is not None:
        print(f"  Limit   : {limit} patients")
    print(f"{'=' * 70}")

    if not os.path.isfile(csv_path):
        print(f"  [SKIP] CSV not found: {csv_path}")
        return

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if limit is not None:
        rows = rows[:limit]

    total = len(rows)
    results: dict[str, list[dict]] = {}
    errors:  dict[str, str]        = {}

    t0 = time.time()
    for i, row in enumerate(rows, 1):
        hadm_id = str(row.get("hadm_id", f"row_{i}"))
        elapsed = time.time() - t0
        eta_str = ""
        if i > 1:
            avg = elapsed / (i - 1)
            eta  = avg * (total - i + 1)
            eta_str = f"  ETA {eta/60:.1f} min"
        print(f"  [{i:>4}/{total}] hadm_id={hadm_id}{eta_str}", flush=True)

        try:
            steps = extract_patient_steps(row, client, run_llm=run_llm)
            results[hadm_id] = [step_to_dict(s) for s in steps]
        except Exception as exc:
            print(f"           ERROR: {exc}")
            errors[hadm_id] = str(exc)

    # Write results
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": results, "errors": errors}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    duration = time.time() - t0
    print(f"\n  Done: {len(results)} patients saved, {len(errors)} errors.")
    print(f"  Wall time: {duration/60:.1f} min  →  {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full-scale feature extraction for all diseases."
    )
    p.add_argument(
        "--disease",
        default="all",
        choices=DISEASES + ["all"],
        help="Disease to process (default: all)",
    )
    p.add_argument(
        "--output-dir",
        default="results/feature_extraction",
        help="Output directory for JSON result files (default: results/feature_extraction)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max patients per disease (default: no limit)",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM calls; run algo-only extraction",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (overrides OPENAI_API_KEY env var)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_llm = not args.no_llm

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if run_llm and not api_key:
        print(
            "ERROR: No OpenAI API key found.\n"
            "  Set OPENAI_API_KEY environment variable, pass --api-key <key>,\n"
            "  or run with --no-llm to skip LLM calls."
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key) if (run_llm and api_key) else None

    output_dir = Path(args.output_dir)
    diseases   = DISEASES if args.disease == "all" else [args.disease]

    print("=" * 70)
    print("  Feature Extraction Run")
    print(f"  Diseases   : {', '.join(diseases)}")
    print(f"  Output dir : {output_dir}")
    print(f"  Mode       : {'algo-only' if not run_llm else 'LLM (gpt-4o)'}")
    print("=" * 70)

    overall_t0 = time.time()
    for disease in diseases:
        process_disease(
            disease,
            client,
            output_dir,
            run_llm=run_llm,
            limit=args.limit,
        )

    total_min = (time.time() - overall_t0) / 60
    print(f"\n{'=' * 70}")
    print(f"  All done.  Total wall time: {total_min:.1f} min")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
