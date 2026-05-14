"""
run_tau_calibration.py

Batch runner: calibrate tau on all four diseases using GPT-4o.

Usage
-----
  python run_tau_calibration.py [--max-patients N] [--output PATH]

Defaults: --max-patients 50  --output results/tau_calibration.json
Set --max-patients 0 to run the full cohort.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from llm_tau_calibrator import calibrate_tau_on_cohort, load_cohort_from_json

DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-patients", type=int, default=50,
        help="Patients per disease (0 = all). Default: 50",
    )
    parser.add_argument(
        "--output", type=str, default="results/tau_calibration.json",
        help="Where to save the JSON results.",
    )
    args = parser.parse_args()

    max_p: int | None = args.max_patients if args.max_patients > 0 else None
    out_path = Path(args.output)

    all_results: dict = {}
    summary_rows: list[dict] = []

    for disease in DISEASES:
        print(f"\n{'='*60}")
        print(f"  Disease: {disease}  (max_patients={max_p or 'ALL'})")
        print(f"{'='*60}")

        trajs, seqs = load_cohort_from_json(disease, max_patients=max_p)
        print(f"  Loaded {len(trajs)} patients.")

        t0 = time.time()
        report = calibrate_tau_on_cohort(trajs, seqs)
        elapsed = time.time() - t0

        n_valid = sum(1 for r in report["per_patient"] if r is not None)
        all_results[disease] = report
        summary_rows.append({
            "disease":    disease,
            "n_patients": len(trajs),
            "n_valid":    n_valid,
            "mean_tau":   round(report["mean_tau"], 4),
            "std_tau":    round(report["std_tau"],  4),
            "elapsed_s":  round(elapsed, 1),
        })

        print(
            f"\n  → mean_tau={report['mean_tau']:.4f}  "
            f"std={report['std_tau']:.4f}  "
            f"valid={n_valid}/{len(trajs)}  "
            f"time={elapsed:.0f}s"
        )

    # ------------------------------------------------------------------ save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved → {out_path}")

    # ---------------------------------------------------------------- summary
    print("\n" + "="*60)
    print(f"  {'Disease':<20} {'N':>5} {'Valid':>5} {'mean_tau':>9} {'std_tau':>8} {'time(s)':>8}")
    print("  " + "-"*56)
    for r in summary_rows:
        print(
            f"  {r['disease']:<20} {r['n_patients']:>5} {r['n_valid']:>5} "
            f"{r['mean_tau']:>9.4f} {r['std_tau']:>8.4f} {r['elapsed_s']:>8.1f}"
        )

    # Overall mean_tau (weighted by n_valid)
    total_valid = sum(r["n_valid"] for r in summary_rows)
    weighted_tau = sum(
        all_results[r["disease"]]["mean_tau"] * r["n_valid"]
        for r in summary_rows
        if r["n_valid"] > 0
    ) / max(total_valid, 1)
    print(f"\n  Overall weighted mean_tau = {weighted_tau:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
