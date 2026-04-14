"""speed_accuracy_eval.py — Speed-Accuracy tradeoff evaluation framework

Core idea
---------
Given a stopping threshold τ (entropy bits):
  "Stop ordering more tests once H(diagnosis_distribution) < τ"

For a range of τ values we measure:
  - Speed    : average number of tests needed before stopping
  - Accuracy : fraction of patients whose primary_diagnosis at the stopping
               point equals the ground-truth disease

Sweeping τ ∈ [0, 2] produces a **speed-accuracy frontier curve**.
Four test-ordering strategies are compared:

  actual   — the order tests were actually performed in the medical record
  random   — random permutation (averaged over multiple seeds)
  bfs      — greedy: at each step pick the BFS top-1 from remaining tests
  ig       — greedy: at each step pick the IG re-ranked top-1 from remaining

Test-reordering simulation
--------------------------
Since patient feature dicts are *cumulative* (each step already merges all
prior evidence), re-ordering tests is non-trivial.  We solve this by
extracting a per-test **delta** — the feature changes introduced specifically
by that test — and then composing deltas in the desired order.

Boolean features are monotonically increasing (once True, stays True), so
deltas compose independently.  The resulting intermediate feature snapshots
are valid inputs to the pipeline.

Usage
-----
  python speed_accuracy_eval.py                         # full sweep, all diseases
  python speed_accuracy_eval.py --n-train 800 --n-test 200
  python speed_accuracy_eval.py --alpha 0.8             # weight IG more heavily
  python speed_accuracy_eval.py --tau-min 0.2 --tau-max 1.6 --tau-steps 15
  python speed_accuracy_eval.py --csv results/speed_accuracy.csv
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# ── project imports ───────────────────────────────────────────────────────────
from feature_schema import default_features, VALID_TESTS
from clinical_session import ClinicalSession
from diagnosis_distribution import DiagnosisDistribution, compute_distribution
from traversal_engine import run_full_traversal
from entropy_reducer import EntropyReducer, distribution_entropy
from ig_recommender import IGRecommender
import diagnosis_distribution as _dd
from empirical_scorer import EmpiricalScorer

RESULTS_DIR = Path(__file__).parent / "results"
DISEASE_FILES: dict[str, Path] = {
    "appendicitis":   RESULTS_DIR / "appendicitis_features.json",
    "cholecystitis":  RESULTS_DIR / "cholecystitis_features.json",
    "diverticulitis": RESULTS_DIR / "diverticulitis_features.json",
    "pancreatitis":   RESULTS_DIR / "pancreatitis_features.json",
}

STRATEGIES = ("actual", "random", "bfs", "ig")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all() -> dict[str, dict[str, list[dict]]]:
    all_patients: dict[str, dict[str, list[dict]]] = {}
    for disease, path in DISEASE_FILES.items():
        with open(path, encoding="utf-8") as f:
            all_patients[disease] = json.load(f)["results"]
    return all_patients


# ---------------------------------------------------------------------------
# Per-test delta extraction
# ---------------------------------------------------------------------------

def extract_test_deltas(steps: list[dict]) -> dict[str, dict]:
    """
    For each test in the patient's trajectory, compute the feature *delta*
    introduced by that test (changes relative to the preceding step).

    Returns
    -------
    {test_key: {feature_key: new_value, ...}}
        Only keys whose values *changed* between the two consecutive steps
        are included.  `tests_done` is excluded (managed separately).
    """
    defaults = default_features()
    deltas: dict[str, dict] = {}

    for i in range(len(steps) - 1):
        test_key = steps[i + 1].get("test_key")
        if not test_key:
            continue

        feat_before = steps[i]["features"]
        feat_after  = steps[i + 1]["features"]

        delta: dict = {}
        for key, val_after in feat_after.items():
            if key == "tests_done":
                continue
            val_before = feat_before.get(key, defaults.get(key))
            if val_after != val_before:
                delta[key] = val_after

        deltas[test_key] = delta

    return deltas


def apply_delta(features: dict, test_key: str, delta: dict) -> dict:
    """
    Return a new feature dict that reflects *features* + the evidence added
    by *test_key* (as captured in *delta*).
    """
    new_feat = dict(features)
    for key, val in delta.items():
        new_feat[key] = val
    done = list(new_feat.get("tests_done", []))
    if test_key not in done:
        done.append(test_key)
    new_feat["tests_done"] = done
    return new_feat


# ---------------------------------------------------------------------------
# Pipeline runner (thin wrapper)
# ---------------------------------------------------------------------------

def run_pipeline(features: dict) -> DiagnosisDistribution:
    """Run traversal + distribution on *features*, return DiagnosisDistribution."""
    traversal = run_full_traversal(features)
    return compute_distribution(traversal, features)


# ---------------------------------------------------------------------------
# Ordering strategy functions
# ---------------------------------------------------------------------------

OrderingFn = Callable[[dict, list[str]], list[str]]
"""
Signature: (current_features, remaining_tests) → ordered list of tests

Only the FIRST element of the returned list is used as the next test to do.
"""


def make_bfs_ordering(session_class=ClinicalSession) -> OrderingFn:
    """Return an ordering function that always picks the BFS top-1 next."""
    def _fn(features: dict, remaining: list[str]) -> list[str]:
        session = session_class(features)
        state   = session.assess()
        ranked  = [r.test for r in state.recommendations if r.test in remaining]
        unranked = [t for t in remaining if t not in ranked]
        return ranked + unranked
    return _fn


def make_ig_ordering(ig_rec: IGRecommender) -> OrderingFn:
    """Return an ordering function that picks the IG re-ranked top-1 next."""
    def _fn(features: dict, remaining: list[str]) -> list[str]:
        session = ClinicalSession(features)
        state   = session.assess()
        # Filter BFS recs to remaining candidate tests
        filtered = [r for r in state.recommendations if r.test in remaining]
        if not filtered:
            # No BFS candidates among remaining → fall back to IG-only scoring
            from entropy_reducer import EntropyReducer
            ig_scores = ig_rec.reducer.test_entropy_scores(features, remaining)
            return sorted(remaining, key=lambda t: ig_scores.get(t, 0.0), reverse=True)
        reranked = ig_rec.rerank(filtered, features)
        ranked   = [r.test for r in reranked]
        unranked = [t for t in remaining if t not in ranked]
        return ranked + unranked
    return _fn


# ---------------------------------------------------------------------------
# Single-patient trajectory simulation
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryResult:
    """Outcome of simulating one patient under one strategy at one τ."""
    steps_used:   int    # number of tests done before stopping (or all tests)
    stopped_early: bool  # True if H < τ was reached before all tests done
    correct:      bool   # primary_diagnosis == ground_truth at stop point
    final_H:      float  # entropy at stop point


def simulate_patient(
    base_features: dict,
    test_deltas: dict[str, dict],
    ordered_tests: list[str],   # pre-computed ordering for this strategy
    ground_truth: str,
    tau: float,
) -> TrajectoryResult:
    """
    Simulate doing tests in *ordered_tests* order, stopping when H < τ.

    Parameters
    ----------
    base_features  : features at step 0 (HPI + PE, no imaging/lab tests)
    test_deltas    : {test_key: delta_dict}, from extract_test_deltas
    ordered_tests  : the sequence of tests to apply (strategy-specific)
    ground_truth   : true disease label
    tau            : stopping threshold on entropy (bits)
    """
    features = dict(base_features)
    for i, test in enumerate(ordered_tests):
        delta = test_deltas.get(test, {})
        features = apply_delta(features, test, delta)

        dist  = run_pipeline(features)
        H     = distribution_entropy(dist)

        if H < tau:
            return TrajectoryResult(
                steps_used    = i + 1,
                stopped_early = True,
                correct       = dist.primary == ground_truth,
                final_H       = H,
            )

    # Never triggered early stopping — used all tests
    dist = run_pipeline(features)
    H    = distribution_entropy(dist)
    return TrajectoryResult(
        steps_used    = len(ordered_tests),
        stopped_early = False,
        correct       = dist.primary == ground_truth,
        final_H       = H,
    )


# ---------------------------------------------------------------------------
# Build ordering for each strategy (called once per patient, before τ sweep)
# ---------------------------------------------------------------------------

def build_orderings(
    base_features: dict,
    test_deltas: dict[str, dict],
    actual_order: list[str],
    bfs_fn: OrderingFn,
    ig_fn: OrderingFn,
    random_seeds: int = 5,
) -> dict[str, list[list[str]]]:
    """
    Pre-compute test orderings for all strategies for one patient.

    Returns
    -------
    {strategy: [ordering_1, ...]}
    "random" has *random_seeds* orderings (will be averaged over).
    All others have exactly one.
    """
    # actual
    orderings: dict[str, list[list[str]]] = {
        "actual": [actual_order],
    }

    # random (multiple seeds)
    rng_orderings: list[list[str]] = []
    for seed in range(random_seeds):
        perm = list(actual_order)
        random.Random(seed).shuffle(perm)
        rng_orderings.append(perm)
    orderings["random"] = rng_orderings

    # BFS greedy
    orderings["bfs"] = [_greedy_order(base_features, test_deltas, actual_order, bfs_fn)]

    # IG greedy
    orderings["ig"] = [_greedy_order(base_features, test_deltas, actual_order, ig_fn)]

    return orderings


def _greedy_order(
    base_features: dict,
    test_deltas: dict[str, dict],
    available_tests: list[str],
    ordering_fn: OrderingFn,
) -> list[str]:
    """Greedily build a test sequence by asking the ordering function at each step."""
    features  = dict(base_features)
    remaining = list(available_tests)
    result    = []

    while remaining:
        ranked = ordering_fn(features, remaining)
        # Pick top-1 (that is still available)
        next_test = ranked[0] if ranked else remaining[0]
        result.append(next_test)
        remaining.remove(next_test)
        delta    = test_deltas.get(next_test, {})
        features = apply_delta(features, next_test, delta)

    return result


# ---------------------------------------------------------------------------
# Full evaluation sweep
# ---------------------------------------------------------------------------

@dataclass
class StrategyStats:
    """Aggregated statistics for one strategy at one τ."""
    tau:           float
    strategy:      str
    n_patients:    int
    mean_steps:    float    # average tests used
    accuracy:      float    # fraction correct at stop
    pct_early_stop: float   # fraction that stopped before exhausting all tests


@dataclass
class EvalResults:
    """Complete sweep results across all τ values and strategies."""
    tau_values: list[float]
    stats: dict[str, list[StrategyStats]]   # strategy → [stats per τ]


def run_eval(
    all_patients: dict[str, dict[str, list[dict]]],
    train_diseases: dict[str, dict[str, list[dict]]],
    test_diseases: dict[str, dict[str, list[dict]]],
    tau_values: list[float],
    k_knn: int = 15,
    alpha: float = 0.5,
    random_seeds: int = 5,
    verbose: bool = True,
) -> EvalResults:
    """
    Run the full speed-accuracy sweep.

    Parameters
    ----------
    all_patients    : full dataset (used for EntropyReducer fitting)
    train_diseases  : training split for EmpiricalScorer
    test_diseases   : test split for evaluation
    tau_values      : list of stopping thresholds to sweep
    k_knn           : K for KNN in EntropyReducer
    alpha           : IG weight in IGRecommender (0=BFS, 1=pure IG)
    random_seeds    : number of random orderings to average over
    verbose         : print progress
    """
    # ── Build EntropyReducer on training data ─────────────────────────────
    if verbose:
        print("  Fitting EntropyReducer…", flush=True)
    reducer = EntropyReducer(k=k_knn)
    reducer.fit(train_diseases)

    # ── Build EmpiricalScorer on training data ─────────────────────────────
    if verbose:
        print("  Fitting EmpiricalScorer…", flush=True)
    train_pairs = [
        (steps[0]["features"], disease)
        for disease, patients in train_diseases.items()
        for steps in patients.values()
    ]
    emp_scorer = EmpiricalScorer(k=k_knn)
    emp_scorer.fit(train_pairs)
    _dd.set_empirical_scorer(emp_scorer)

    ig_rec  = IGRecommender(reducer, alpha=alpha)
    bfs_fn  = make_bfs_ordering()
    ig_fn   = make_ig_ordering(ig_rec)

    # ── Accumulate per-τ stats ─────────────────────────────────────────────
    # structure: {strategy: {tau: [TrajectoryResult, ...]}}
    accum: dict[str, dict[float, list[TrajectoryResult]]] = {
        s: defaultdict(list) for s in STRATEGIES
    }

    total = sum(len(p) for p in test_diseases.values())
    done  = 0

    for disease, patients in test_diseases.items():
        for pid, steps in patients.items():
            done += 1
            if verbose and done % 50 == 0:
                print(f"    {done}/{total} patients…", flush=True)

            if len(steps) < 2:
                continue  # only one step → no test to recommend

            base_features = steps[0]["features"]
            test_deltas   = extract_test_deltas(steps)
            actual_order  = [
                s["test_key"] for s in steps[1:]
                if s.get("test_key") and s["test_key"] in test_deltas
            ]
            if not actual_order:
                continue

            # Pre-compute orderings (expensive: runs pipeline greedily)
            orderings = build_orderings(
                base_features, test_deltas, actual_order,
                bfs_fn, ig_fn, random_seeds=random_seeds,
            )

            # Sweep τ for each ordering
            for tau in tau_values:
                for strategy, ordering_list in orderings.items():
                    results_for_strategy = [
                        simulate_patient(
                            base_features, test_deltas, ordering,
                            disease, tau,
                        )
                        for ordering in ordering_list
                    ]
                    # For multi-seed strategies (random), average
                    avg = _avg_results(results_for_strategy)
                    accum[strategy][tau].append(avg)

    # ── Aggregate ─────────────────────────────────────────────────────────
    stats: dict[str, list[StrategyStats]] = {s: [] for s in STRATEGIES}
    for strategy in STRATEGIES:
        for tau in tau_values:
            results = accum[strategy][tau]
            if not results:
                continue
            stats[strategy].append(StrategyStats(
                tau            = tau,
                strategy       = strategy,
                n_patients     = len(results),
                mean_steps     = float(np.mean([r.steps_used for r in results])),
                accuracy       = float(np.mean([r.correct for r in results])),
                pct_early_stop = float(np.mean([r.stopped_early for r in results])),
            ))

    _dd.clear_empirical_scorer()
    return EvalResults(tau_values=tau_values, stats=stats)


def _avg_results(results: list[TrajectoryResult]) -> TrajectoryResult:
    """Average a list of TrajectoryResults (used for random multi-seed)."""
    if len(results) == 1:
        return results[0]
    return TrajectoryResult(
        steps_used    = int(round(np.mean([r.steps_used for r in results]))),
        stopped_early = bool(np.mean([r.stopped_early for r in results]) >= 0.5),
        correct       = bool(np.mean([r.correct for r in results]) >= 0.5),
        final_H       = float(np.mean([r.final_H for r in results])),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results: EvalResults) -> None:
    """Print a formatted speed-accuracy table for each strategy."""
    print(f"\n{'═'*76}")
    print(f"  SPEED-ACCURACY SWEEP   (τ = entropy stopping threshold in bits)")
    print(f"{'═'*76}")
    print(f"  {'τ':>5}  {'Strategy':<10}  {'Accuracy':>8}  {'Mean tests':>10}  {'Early stop%':>11}")
    print(f"  {'─'*64}")

    for tau in results.tau_values:
        for strategy in STRATEGIES:
            st_list = [s for s in results.stats[strategy] if abs(s.tau - tau) < 1e-9]
            if not st_list:
                continue
            st = st_list[0]
            print(
                f"  {tau:>5.2f}  {strategy:<10}  "
                f"{st.accuracy:>8.1%}  "
                f"{st.mean_steps:>10.2f}  "
                f"{st.pct_early_stop:>10.1%}"
            )
        print()

    # Summary: at τ = 1.0 (moderate confidence threshold)
    print(f"{'─'*76}")
    print(f"  SUMMARY at τ = 1.0 bit")
    print(f"  {'─'*64}")
    for strategy in STRATEGIES:
        st_list = [s for s in results.stats[strategy] if abs(s.tau - 1.0) < 1e-9]
        if not st_list:
            continue
        st = st_list[0]
        print(
            f"  {strategy:<10}  acc={st.accuracy:.1%}  "
            f"avg_tests={st.mean_steps:.2f}  "
            f"early_stop={st.pct_early_stop:.1%}"
        )
    print(f"{'═'*76}\n")


def save_csv(results: EvalResults, path: str) -> None:
    """Save full results to a CSV file for external plotting."""
    import csv
    rows = []
    for strategy in STRATEGIES:
        for st in results.stats[strategy]:
            rows.append({
                "tau":          st.tau,
                "strategy":     st.strategy,
                "n_patients":   st.n_patients,
                "accuracy":     st.accuracy,
                "mean_steps":   st.mean_steps,
                "pct_early_stop": st.pct_early_stop,
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Speed-accuracy evaluation.")
    p.add_argument("--n-train",    type=int,   default=None,
                   help="Number of training patients (default: all minus n_test).")
    p.add_argument("--n-test",     type=int,   default=200,
                   help="Number of test patients (default: 200).")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--k",          type=int,   default=15,
                   help="KNN neighbors for EntropyReducer and EmpiricalScorer.")
    p.add_argument("--alpha",      type=float, default=0.5,
                   help="IG weight in IGRecommender (0=BFS, 1=pure IG).")
    p.add_argument("--tau-min",    type=float, default=0.2)
    p.add_argument("--tau-max",    type=float, default=1.8)
    p.add_argument("--tau-steps",  type=int,   default=9,
                   help="Number of τ values to evaluate.")
    p.add_argument("--random-seeds", type=int, default=3,
                   help="Seeds for random ordering baseline.")
    p.add_argument("--csv",        type=str,   default=None,
                   help="Optional path to save CSV results.")
    p.add_argument("--quiet",      action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    verbose = not args.quiet

    # ── Load data ──────────────────────────────────────────────────────────
    if verbose:
        print("Loading patient data…")
    all_patients = load_all()

    # Flatten to list for splitting
    flat: list[tuple[str, str, list[dict]]] = [
        (disease, pid, steps)
        for disease, patients in all_patients.items()
        for pid, steps in patients.items()
    ]
    random.seed(args.seed)
    random.shuffle(flat)

    n_test  = min(args.n_test, len(flat))
    n_train = args.n_train or (len(flat) - n_test)
    n_train = min(n_train, len(flat) - n_test)

    test_flat  = flat[:n_test]
    train_flat = flat[n_test: n_test + n_train]

    # Rebuild dicts
    def _to_dict(items: list[tuple]) -> dict[str, dict[str, list[dict]]]:
        result: dict[str, dict[str, list[dict]]] = {d: {} for d in DISEASE_FILES}
        for disease, pid, steps in items:
            result[disease][pid] = steps
        return result

    train_diseases = _to_dict(train_flat)
    test_diseases  = _to_dict(test_flat)

    if verbose:
        print(f"  Train: {len(train_flat)} patients  |  Test: {n_test} patients")

    # ── Tau grid ──────────────────────────────────────────────────────────
    tau_values = list(np.linspace(args.tau_min, args.tau_max, args.tau_steps))

    # ── Run evaluation ────────────────────────────────────────────────────
    if verbose:
        print(f"Running sweep over {len(tau_values)} τ values…")
    results = run_eval(
        all_patients   = all_patients,
        train_diseases = train_diseases,
        test_diseases  = test_diseases,
        tau_values     = tau_values,
        k_knn          = args.k,
        alpha          = args.alpha,
        random_seeds   = args.random_seeds,
        verbose        = verbose,
    )

    print_results(results)

    if args.csv:
        save_csv(results, args.csv)


if __name__ == "__main__":
    main()
