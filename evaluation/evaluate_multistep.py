"""evaluate_multistep.py — Fixed-τ cohort evaluation using multi-step feature simulation.

Mirrors evaluate.py but uses the FeatureSimulator (KNN feature-level simulation)
instead of the 1-step KNN counterfactual.

Strategies evaluated
--------------------
  actual_real : real test sequence + real features          (ceiling)
  actual_sim  : real test sequence + KNN-simulated features
  bfs_sim     : BFS-recommended tests + simulated features
  ig_sim      : IG re-ranked tests  + simulated features
  random_sim  : random test order   + simulated features

Output CSV columns
------------------
  strategy, n_patients, accuracy, mean_n_tests, mean_cost, mean_burden, pct_stopped

Usage
-----
  python evaluation/evaluate_multistep.py
  python evaluation/evaluate_multistep.py --alpha 0.5 --n-test 300
  python evaluation/evaluate_multistep.py --csv results/evaluation_summary_multistep.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import csv
import random
from dataclasses import dataclass

from evaluation.knn_feature_eval import (
    SIM_STRATEGIES,
    DISEASE_FILES,
    load_all,
    run_multistep_alpha_sweep,
)

# ── Calibrated stopping threshold (matches evaluate.py) ─────────────────────
CALIBRATED_TAU: float = 0.7461

RESULTS_DIR = _ROOT / "results"


# ---------------------------------------------------------------------------
# Output dataclass (same fields as evaluate.py CohortSummary)
# ---------------------------------------------------------------------------

@dataclass
class CohortSummary:
    strategy:     str
    n_patients:   int
    accuracy:     float
    mean_n_tests: float
    mean_cost:    float
    mean_burden:  float
    pct_stopped:  float


# ---------------------------------------------------------------------------
# Display labels (mirrors evaluate.py _STRATEGY_LABEL)
# ---------------------------------------------------------------------------

_STRATEGY_LABEL: dict[str, str] = {
    "actual_real": "Real order + real features    [ceiling]",
    "actual_sim":  "Real order + simulated features",
    "bfs_sim":     "Rubric-guided + simulated features",
    "ig_sim":      "Efficiency-optimised + simulated features",
    "random_sim":  "Random + simulated features",
}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(stats: list[CohortSummary], tau: float = CALIBRATED_TAU) -> None:
    n = stats[0].n_patients if stats else 0
    W = 94
    print(f"\n{'═'*W}")
    print(f"  MULTI-STEP FEATURE-SIM EVALUATION   τ = {tau:.4f}   n = {n} patients")
    print(f"{'═'*W}")
    print(
        f"  {'Strategy':<44}  {'Accuracy':>8}  {'Tests':>6}  "
        f"{'Cost':>8}  {'Burden':>8}  {'Stopped%':>9}"
    )
    print(f"  {'─'*90}")
    for s in stats:
        label = _STRATEGY_LABEL.get(s.strategy, s.strategy)
        print(
            f"  {label:<44}  {s.accuracy:>8.1%}  {s.mean_n_tests:>6.2f}  "
            f"{s.mean_cost:>8.2f}  {s.mean_burden:>8.2f}  {s.pct_stopped:>8.1%}"
        )
    print(f"{'═'*W}\n")


def save_summary_csv(stats: list[CohortSummary], path: str) -> None:
    rows = [
        {
            "strategy":     s.strategy,
            "n_patients":   s.n_patients,
            "accuracy":     round(s.accuracy,     4),
            "mean_n_tests": round(s.mean_n_tests, 4),
            "mean_cost":    round(s.mean_cost,    4),
            "mean_burden":  round(s.mean_burden,  4),
            "pct_stopped":  round(s.pct_stopped,  4),
        }
        for s in stats
    ]
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha",        type=float, default=0.5,
                   help="IG recommender weight (default: 0.5).")
    p.add_argument("--tau",          type=float, default=CALIBRATED_TAU,
                   help=f"Stopping threshold (default: {CALIBRATED_TAU}).")
    p.add_argument("--n-test",       type=int,   default=300)
    p.add_argument("--n-train",      type=int,   default=None)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--min-tests",    type=int,   default=2,
                   help="Min imaging tests per patient (default: 2).")
    p.add_argument("--k",            type=int,   default=15)
    p.add_argument("--random-seeds", type=int,   default=3)
    p.add_argument("--csv",          type=str,
                   default=str(RESULTS_DIR / "evaluation_summary_multistep.csv"))
    p.add_argument("--quiet",        action="store_true")
    args    = p.parse_args()
    verbose = not args.quiet

    if verbose:
        print("Loading patient data…")
    all_patients = load_all()
    flat: list[tuple[str, str, list[dict]]] = [
        (disease, pid, steps)
        for disease, patients in all_patients.items()
        for pid, steps in patients.items()
    ]

    min_steps     = args.min_tests + 1
    flat_filtered = [(d, pid, s) for d, pid, s in flat if len(s) >= min_steps]
    if verbose:
        print(
            f"  After filter (≥{args.min_tests} tests): "
            f"{len(flat_filtered)}/{len(flat)} patients"
        )

    random.seed(args.seed)
    random.shuffle(flat_filtered)

    n_test  = min(args.n_test, len(flat_filtered))
    n_train = args.n_train or (len(flat_filtered) - n_test)
    n_train = min(n_train, len(flat_filtered) - n_test)

    test_patients  = flat_filtered[:n_test]
    train_patients = flat_filtered[n_test: n_test + n_train]

    if verbose:
        print(f"  Train: {len(train_patients)}  |  Test: {n_test}")
        print(f"\nRunning multi-step evaluation  τ={args.tau:.4f}  α={args.alpha:.2f}…\n")

    # run_multistep_alpha_sweep handles all fitting internally
    results_by_alpha = run_multistep_alpha_sweep(
        test_patients  = test_patients,
        train_patients = train_patients,
        tau_values     = [args.tau],
        alpha_values   = [args.alpha],
        k_knn          = args.k,
        random_seeds   = args.random_seeds,
        verbose        = verbose,
    )

    results = results_by_alpha[args.alpha]

    # Build CohortSummary for each strategy at the fixed tau
    summaries: list[CohortSummary] = []
    for strategy in SIM_STRATEGIES:
        st_list = [s for s in results.stats[strategy] if abs(s.tau - args.tau) < 1e-9]
        if not st_list:
            continue
        st = st_list[0]
        summaries.append(CohortSummary(
            strategy     = strategy,
            n_patients   = st.n_patients,
            accuracy     = st.accuracy,
            mean_n_tests = st.mean_stop_tests,
            mean_cost    = st.mean_cost,
            mean_burden  = st.mean_burden,
            pct_stopped  = st.pct_early_stop,
        ))

    print_summary(summaries, tau=args.tau)
    save_summary_csv(summaries, args.csv)


if __name__ == "__main__":
    main()
