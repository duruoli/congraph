"""evaluate.py — Unified fixed-τ cohort evaluation.

Stopping threshold fixed at CALIBRATED_TAU = 0.7461 (LLM-selected optimal).

Output: one row per strategy
  strategy | accuracy | mean_n_tests | mean_cost | mean_burden | pct_stopped

Usage
-----
  python evaluation/evaluate.py
  python evaluation/evaluate.py --alpha 0.5 --n-test 300
  python evaluation/evaluate.py --csv results/evaluation_summary.csv
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

from knn.empirical_scorer import EmpiricalScorer
from knn.entropy_reducer import EntropyReducer
from knn.ig_recommender import IGRecommender
import pipeline.diagnosis_distribution as _dd
from evaluation.knn_onestep_eval import (
    evaluate_patient as _eval_patient_full,
    KNN_STRATEGIES,
    DISEASE_FILES,
    load_all,
)
from evaluation.evaluation_metrics import PatientStopResult, aggregate_stats

# ── Calibrated stopping threshold (LLM-selected) ────────────────────────────
CALIBRATED_TAU: float = 0.7461

RESULTS_DIR = _ROOT / "results"


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class CohortSummary:
    """Aggregated metrics for one strategy across all patients at CALIBRATED_TAU."""
    strategy:     str
    n_patients:   int
    accuracy:     float
    mean_n_tests: float
    mean_cost:    float
    mean_burden:  float
    pct_stopped:  float   # fraction of patients that met τ before exhausting trajectory


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------

def evaluate_patient(
    disease:      str,
    steps:        list[dict],
    reducer:      EntropyReducer,
    ig_rec:       IGRecommender,
    tau:          float = CALIBRATED_TAU,
    random_seeds: int   = 5,
) -> dict[str, PatientStopResult]:
    """Single-τ wrapper — returns {strategy: PatientStopResult}."""
    raw = _eval_patient_full(disease, steps, reducer, ig_rec, [tau], random_seeds)
    return {s: d[tau] for s, d in raw.items() if tau in d}


def evaluate_cohort(
    test_patients:  list[tuple[str, str, list[dict]]],
    train_patients: list[tuple[str, str, list[dict]]],
    alpha:          float = 0.5,
    tau:            float = CALIBRATED_TAU,
    k:              int   = 15,
    random_seeds:   int   = 5,
    verbose:        bool  = True,
) -> list[CohortSummary]:
    """
    Fit KNN models on train_patients, evaluate test_patients at a fixed τ.

    Returns one CohortSummary per strategy, in KNN_STRATEGIES order.
    """
    if verbose:
        print("  Fitting EntropyReducer…", flush=True)
    train_dict: dict[str, dict[str, list[dict]]] = {d: {} for d in DISEASE_FILES}
    for disease, pid, steps in train_patients:
        train_dict[disease][pid] = steps
    reducer = EntropyReducer(k=k)
    reducer.fit(train_dict)
    if verbose:
        print(f"    → {reducer.n_transitions} transitions", flush=True)

    if verbose:
        print("  Fitting EmpiricalScorer…", flush=True)
    train_pairs = [(steps[0]["features"], disease) for disease, pid, steps in train_patients]
    emp_scorer  = EmpiricalScorer(k=k)
    emp_scorer.fit(train_pairs)
    _dd.set_empirical_scorer(emp_scorer)

    ig_rec = IGRecommender(reducer, alpha=alpha)

    accum: dict[str, list[PatientStopResult]] = {s: [] for s in KNN_STRATEGIES}
    total = len(test_patients)
    for idx, (disease, pid, steps) in enumerate(test_patients, 1):
        if verbose and idx % 50 == 0:
            print(f"    {idx}/{total} patients…", flush=True)
        per_patient = evaluate_patient(disease, steps, reducer, ig_rec, tau, random_seeds)
        for strategy, psr in per_patient.items():
            accum[strategy].append(psr)

    _dd.clear_empirical_scorer()

    summaries: list[CohortSummary] = []
    for strategy in KNN_STRATEGIES:
        results = accum[strategy]
        if not results:
            continue
        st = aggregate_stats(results, tau=tau, strategy=strategy)
        summaries.append(CohortSummary(
            strategy     = strategy,
            n_patients   = st.n_patients,
            accuracy     = st.accuracy,
            mean_n_tests = st.mean_stop_tests,
            mean_cost    = st.mean_cost,
            mean_burden  = st.mean_burden,
            pct_stopped  = st.pct_early_stop,
        ))
    return summaries


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_STRATEGY_LABEL: dict[str, str] = {
    "actual":      "Real order (KNN-estimated)",
    "actual_real": "Real order (actual features)  [ceiling]",
    "bfs":         "Rubric-guided order",
    "ig":          "Efficiency-optimized order",
    "random":      "Random order",
}


def print_summary(stats: list[CohortSummary], tau: float = CALIBRATED_TAU) -> None:
    n = stats[0].n_patients if stats else 0
    W = 90
    print(f"\n{'═'*W}")
    print(f"  EVALUATION SUMMARY   τ = {tau:.4f}   n = {n} patients")
    print(f"{'═'*W}")
    print(
        f"  {'Strategy':<38}  {'Accuracy':>8}  {'Tests':>6}  "
        f"{'Cost':>8}  {'Burden':>8}  {'Stopped%':>9}"
    )
    print(f"  {'─'*86}")
    for s in stats:
        label = _STRATEGY_LABEL.get(s.strategy, s.strategy)
        print(
            f"  {label:<38}  {s.accuracy:>8.1%}  {s.mean_n_tests:>6.2f}  "
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
    p = argparse.ArgumentParser(description="Fixed-τ cohort evaluation.")
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
                   default=str(RESULTS_DIR / "evaluation_summary.csv"))
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
        print(f"\nRunning evaluate_cohort  τ={args.tau:.4f}  α={args.alpha:.2f}…\n")

    stats = evaluate_cohort(
        test_patients  = test_patients,
        train_patients = train_patients,
        alpha          = args.alpha,
        tau            = args.tau,
        k              = args.k,
        random_seeds   = args.random_seeds,
        verbose        = verbose,
    )

    print_summary(stats, tau=args.tau)
    save_summary_csv(stats, args.csv)


if __name__ == "__main__":
    main()
