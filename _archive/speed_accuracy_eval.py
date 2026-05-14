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
Four test-ordering strategies are compared per run:

  actual   — the order tests were actually performed in the medical record
  random   — random permutation (averaged over multiple seeds)
  bfs      — greedy: at each step pick the BFS top-1 from remaining tests
  ig_α     — greedy: IG re-ranked top-1, for the current value of α
             α=0 → pure BFS weight, α=1 → pure IG weight

Both τ and α are swept, producing one figure per α value.

Filter
------
Only patients with ≥ min_tests imaging tests (default 2, i.e. 3+ steps)
are included so that test re-ordering is meaningful.

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
  python speed_accuracy_eval.py                              # full sweep
  python speed_accuracy_eval.py --n-train 800 --n-test 200
  python speed_accuracy_eval.py --min-tests 2               # only 2+ test patients
  python speed_accuracy_eval.py --alphas 0.0,0.5,1.0        # multiple alpha values
  python speed_accuracy_eval.py --tau-min 0.2 --tau-max 1.6 --tau-steps 15
  python speed_accuracy_eval.py --plot --plot-dir results/figs
  python speed_accuracy_eval.py --csv results/speed_accuracy.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ── project imports ───────────────────────────────────────────────────────────
from pipeline.feature_schema import default_features, VALID_TESTS
from pipeline.clinical_session import ClinicalSession
from pipeline.diagnosis_distribution import DiagnosisDistribution, compute_distribution
from pipeline.traversal_engine import run_full_traversal
from knn.entropy_reducer import EntropyReducer, distribution_entropy
from knn.ig_recommender import IGRecommender
import pipeline.diagnosis_distribution as _dd
from knn.empirical_scorer import EmpiricalScorer

RESULTS_DIR = _REPO_ROOT / "results"
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
        Only keys whose values *changed* between consecutive steps are included.
        `tests_done` is excluded (managed separately by apply_delta).
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
    """Return a new feature dict that reflects features + the evidence added by test_key."""
    new_feat = dict(features)
    for key, val in delta.items():
        new_feat[key] = val
    done = list(new_feat.get("tests_done", []))
    if test_key not in done:
        done.append(test_key)
    new_feat["tests_done"] = done
    return new_feat


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(features: dict) -> DiagnosisDistribution:
    """Run traversal + distribution on features, return DiagnosisDistribution."""
    traversal = run_full_traversal(features)
    return compute_distribution(traversal, features)


# ---------------------------------------------------------------------------
# Ordering strategy functions
# ---------------------------------------------------------------------------

OrderingFn = Callable[[dict, list[str]], list[str]]


def make_bfs_ordering() -> OrderingFn:
    def _fn(features: dict, remaining: list[str]) -> list[str]:
        session = ClinicalSession(features)
        state   = session.assess()
        ranked  = [r.test for r in state.recommendations if r.test in remaining]
        unranked = [t for t in remaining if t not in ranked]
        return ranked + unranked
    return _fn


def make_ig_ordering(ig_rec: IGRecommender) -> OrderingFn:
    def _fn(features: dict, remaining: list[str]) -> list[str]:
        session  = ClinicalSession(features)
        state    = session.assess()
        filtered = [r for r in state.recommendations if r.test in remaining]
        if not filtered:
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
    steps_used:    float   # float to support seed-averaging in _avg_results
    stopped_early: bool
    correct:       float   # float [0,1] to allow seed-averaging without majority-vote bias
    final_H:       float


def simulate_patient(
    base_features: dict,
    test_deltas: dict[str, dict],
    ordered_tests: list[str],
    ground_truth: str,
    tau: float,
) -> TrajectoryResult:
    features = dict(base_features)
    for i, test in enumerate(ordered_tests):
        delta    = test_deltas.get(test, {})
        features = apply_delta(features, test, delta)

        dist = run_pipeline(features)
        H    = distribution_entropy(dist)

        if H < tau:
            return TrajectoryResult(
                steps_used    = float(i + 1),
                stopped_early = True,
                correct       = float(dist.primary == ground_truth),
                final_H       = H,
            )

    dist = run_pipeline(features)
    H    = distribution_entropy(dist)
    return TrajectoryResult(
        steps_used    = float(len(ordered_tests)),
        stopped_early = False,
        correct       = float(dist.primary == ground_truth),
        final_H       = H,
    )


# ---------------------------------------------------------------------------
# Build greedy ordering for each strategy
# ---------------------------------------------------------------------------

def build_orderings(
    base_features: dict,
    test_deltas: dict[str, dict],
    actual_order: list[str],
    bfs_fn: OrderingFn,
    ig_fn: OrderingFn,
    random_seeds: int = 5,
) -> dict[str, list[list[str]]]:
    orderings: dict[str, list[list[str]]] = {
        "actual": [actual_order],
    }

    rng_orderings: list[list[str]] = []
    for seed in range(random_seeds):
        perm = list(actual_order)
        random.Random(seed).shuffle(perm)
        rng_orderings.append(perm)
    orderings["random"] = rng_orderings

    orderings["bfs"] = [_greedy_order(base_features, test_deltas, actual_order, bfs_fn)]
    orderings["ig"]  = [_greedy_order(base_features, test_deltas, actual_order, ig_fn)]

    return orderings


def _greedy_order(
    base_features: dict,
    test_deltas: dict[str, dict],
    available_tests: list[str],
    ordering_fn: OrderingFn,
) -> list[str]:
    features  = dict(base_features)
    remaining = list(available_tests)
    result    = []

    while remaining:
        ranked    = ordering_fn(features, remaining)
        next_test = ranked[0] if ranked else remaining[0]
        result.append(next_test)
        remaining.remove(next_test)
        features = apply_delta(features, next_test, test_deltas.get(next_test, {}))

    return result


# ---------------------------------------------------------------------------
# Aggregated statistics dataclass
# ---------------------------------------------------------------------------

@dataclass
class StrategyStats:
    tau:            float
    strategy:       str
    n_patients:     int
    mean_steps:     float
    accuracy:       float
    pct_early_stop: float


@dataclass
class EvalResults:
    tau_values: list[float]
    alpha:      float
    stats:      dict[str, list[StrategyStats]]   # strategy → [stats per τ]
    n_patients: int


# ---------------------------------------------------------------------------
# Core evaluation  (accepts pre-fitted components for multi-alpha efficiency)
# ---------------------------------------------------------------------------

def run_eval(
    test_patients:  list[tuple[str, str, list[dict]]],
    tau_values:     list[float],
    reducer:        EntropyReducer,
    ig_rec:         IGRecommender,
    alpha:          float,
    random_seeds:   int = 5,
    verbose:        bool = True,
) -> EvalResults:
    """
    Run speed-accuracy sweep for one alpha value over pre-split test patients.

    Parameters
    ----------
    test_patients  : list of (disease, patient_id, steps)
    tau_values     : stopping thresholds to evaluate
    reducer        : fitted EntropyReducer
    ig_rec         : IGRecommender constructed with the desired alpha
    alpha          : the alpha value (stored in EvalResults for labelling)
    random_seeds   : seeds for the random ordering baseline
    verbose        : print patient progress
    """
    bfs_fn = make_bfs_ordering()
    ig_fn  = make_ig_ordering(ig_rec)

    accum: dict[str, dict[float, list[TrajectoryResult]]] = {
        s: defaultdict(list) for s in STRATEGIES
    }

    total = len(test_patients)
    for done_count, (disease, pid, steps) in enumerate(test_patients, 1):
        if verbose and done_count % 50 == 0:
            print(f"    {done_count}/{total} patients…", flush=True)

        base_features = steps[0]["features"]
        test_deltas   = extract_test_deltas(steps)
        actual_order  = [
            s["test_key"] for s in steps[1:]
            if s.get("test_key") and s["test_key"] in test_deltas
        ]
        if len(actual_order) < 2:
            continue  # skip if fewer than 2 usable tests (shouldn't happen post-filter)

        orderings = build_orderings(
            base_features, test_deltas, actual_order,
            bfs_fn, ig_fn, random_seeds=random_seeds,
        )

        for tau in tau_values:
            for strategy, ordering_list in orderings.items():
                results = [
                    simulate_patient(base_features, test_deltas, o, disease, tau)
                    for o in ordering_list
                ]
                accum[strategy][tau].append(_avg_results(results))

    stats: dict[str, list[StrategyStats]] = {s: [] for s in STRATEGIES}
    n_actual = 0
    for strategy in STRATEGIES:
        for tau in tau_values:
            results = accum[strategy][tau]
            if not results:
                continue
            if strategy == "actual":
                n_actual = len(results)
            stats[strategy].append(StrategyStats(
                tau            = tau,
                strategy       = strategy,
                n_patients     = len(results),
                mean_steps     = float(np.mean([r.steps_used for r in results])),
                accuracy       = float(np.mean([r.correct for r in results])),
                pct_early_stop = float(np.mean([r.stopped_early for r in results])),
            ))

    return EvalResults(tau_values=tau_values, alpha=alpha, stats=stats, n_patients=n_actual)


def _avg_results(results: list[TrajectoryResult]) -> TrajectoryResult:
    if len(results) == 1:
        return results[0]
    # Average correctness as a float so that multi-seed random baseline is not
    # inflated by majority-vote; the outer np.mean across patients then gives
    # the true expected accuracy.
    return TrajectoryResult(
        steps_used    = float(np.mean([r.steps_used for r in results])),
        stopped_early = bool(np.mean([r.stopped_early for r in results]) >= 0.5),
        correct       = float(np.mean([r.correct for r in results])),
        final_H       = float(np.mean([r.final_H for r in results])),
    )


# ---------------------------------------------------------------------------
# Multi-alpha sweep  (fits components once, sweeps alpha)
# ---------------------------------------------------------------------------

def run_alpha_sweep(
    all_patients:    dict[str, dict[str, list[dict]]],
    test_patients:   list[tuple[str, str, list[dict]]],
    train_patients:  list[tuple[str, str, list[dict]]],
    tau_values:      list[float],
    alpha_values:    list[float],
    k_knn:           int = 15,
    random_seeds:    int = 5,
    verbose:         bool = True,
) -> dict[float, EvalResults]:
    """
    Fit EntropyReducer + EmpiricalScorer once, then sweep over alpha values.

    Returns {alpha: EvalResults}
    """
    # ── Fit EntropyReducer on training data ──────────────────────────────
    if verbose:
        print("  Fitting EntropyReducer on training data…", flush=True)
    train_dict: dict[str, dict[str, list[dict]]] = {d: {} for d in DISEASE_FILES}
    for disease, pid, steps in train_patients:
        train_dict[disease][pid] = steps

    reducer = EntropyReducer(k=k_knn)
    reducer.fit(train_dict)
    if verbose:
        print(f"    → {reducer.n_transitions} transitions in corpus", flush=True)

    # ── Fit EmpiricalScorer on training data ──────────────────────────────
    if verbose:
        print("  Fitting EmpiricalScorer on training data…", flush=True)
    train_pairs = [
        (steps[0]["features"], disease)
        for disease, pid, steps in train_patients
    ]
    emp_scorer = EmpiricalScorer(k=k_knn)
    emp_scorer.fit(train_pairs)
    _dd.set_empirical_scorer(emp_scorer)

    # ── Sweep alpha ────────────────────────────────────────────────────────
    results_by_alpha: dict[float, EvalResults] = {}

    for alpha in alpha_values:
        if verbose:
            print(f"\n  ── α = {alpha:.2f}  ({len(test_patients)} test patients) ──", flush=True)
        ig_rec = IGRecommender(reducer, alpha=alpha)
        results = run_eval(
            test_patients = test_patients,
            tau_values    = tau_values,
            reducer       = reducer,
            ig_rec        = ig_rec,
            alpha         = alpha,
            random_seeds  = random_seeds,
            verbose       = verbose,
        )
        results_by_alpha[alpha] = results

    _dd.clear_empirical_scorer()
    return results_by_alpha


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results: EvalResults) -> None:
    alpha = results.alpha
    print(f"\n{'═'*76}")
    print(f"  SPEED-ACCURACY SWEEP   α={alpha:.2f}   n={results.n_patients} patients")
    print(f"  (τ = entropy stopping threshold in bits)")
    print(f"{'═'*76}")
    print(f"  {'τ':>5}  {'Strategy':<10}  {'Accuracy':>8}  {'Mean tests':>10}  {'Early stop%':>11}")
    print(f"  {'─'*64}")

    for tau in results.tau_values:
        for strategy in STRATEGIES:
            st_list = [s for s in results.stats[strategy] if abs(s.tau - tau) < 1e-9]
            if not st_list:
                continue
            st = st_list[0]
            label = f"ig(α={alpha:.2f})" if strategy == "ig" else strategy
            print(
                f"  {tau:>5.2f}  {label:<14}  "
                f"{st.accuracy:>8.1%}  "
                f"{st.mean_steps:>10.2f}  "
                f"{st.pct_early_stop:>10.1%}"
            )
        print()

    print(f"{'═'*76}\n")


def save_csv(results_by_alpha: dict[float, EvalResults], path: str) -> None:
    import csv
    rows = []
    for alpha, results in results_by_alpha.items():
        for strategy in STRATEGIES:
            for st in results.stats[strategy]:
                rows.append({
                    "alpha":         alpha,
                    "tau":           st.tau,
                    "strategy":      st.strategy,
                    "n_patients":    st.n_patients,
                    "accuracy":      st.accuracy,
                    "mean_steps":    st.mean_steps,
                    "pct_early_stop": st.pct_early_stop,
                })
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved to {path}")


# ---------------------------------------------------------------------------
# Plotting  (one figure per alpha)
# ---------------------------------------------------------------------------

_STRATEGY_STYLE: dict[str, dict] = {
    "actual": dict(color="#2196F3", marker="o",  label="Actual order",   lw=2.0, zorder=4),
    "random": dict(color="#9E9E9E", marker="s",  label="Random order",   lw=1.5, zorder=2, ls="--"),
    "bfs":    dict(color="#F44336", marker="^",  label="BFS order",      lw=2.0, zorder=3),
    "ig":     dict(color="#4CAF50", marker="D",  label="IG re-ranked",   lw=2.5, zorder=5),
}


def plot_results(
    results_by_alpha: dict[float, EvalResults],
    out_dir: str,
    title_suffix: str = "",
) -> None:
    """Generate one PNG per alpha value in out_dir."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  matplotlib not installed — skipping plots.")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for alpha, results in sorted(results_by_alpha.items()):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        for strategy in STRATEGIES:
            style = dict(_STRATEGY_STYLE[strategy])
            if strategy == "ig":
                style["label"] = f"IG re-ranked (α={alpha:.2f})"
            st_list = sorted(results.stats[strategy], key=lambda s: s.tau)
            if not st_list:
                continue
            tau_arr = [s.tau for s in st_list]
            acc_arr = [s.accuracy for s in st_list]
            spd_arr = [s.mean_steps for s in st_list]

            ax1.plot(tau_arr, acc_arr, **style)
            ax2.plot(tau_arr, spd_arr, **style)

        n = results.n_patients
        fig.suptitle(
            f"Speed-Accuracy Analysis — {n} test patients  (2+ tests only)\n"
            f"ConGraph diagnostic pipeline  |  τ sweep  |  α={alpha:.2f}"
            + (f"  |  {title_suffix}" if title_suffix else ""),
            fontsize=12, fontweight="bold",
        )

        # ── top panel: accuracy ────────────────────────────────────────────
        ax1.set_title(f"Accuracy vs τ  |  fixed IG α={alpha:.2f}", fontsize=10)
        ax1.set_ylabel("Diagnostic accuracy", fontsize=10)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax1.invert_xaxis()      # high τ (easy stop) on left → low τ (strict) on right
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ── bottom panel: speed ────────────────────────────────────────────
        ax2.set_title(f"Mean Tests vs τ  |  fixed IG α={alpha:.2f}", fontsize=10)
        ax2.set_xlabel("Stopping threshold τ (bits)", fontsize=10)
        ax2.set_ylabel("Mean tests ordered", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = Path(out_dir) / f"speed_accuracy_alpha_{alpha:.2f}.png"
        fig.savefig(str(fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Combined figure: all alphas on one plot (for the IG curve only)
# ---------------------------------------------------------------------------

def plot_ig_alpha_comparison(
    results_by_alpha: dict[float, EvalResults],
    out_dir: str,
) -> None:
    """
    Single figure with all α values overlaid (IG curve only), plus fixed
    actual/random/bfs baselines from the first alpha run.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import matplotlib
    except ImportError:
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sorted_alphas = sorted(results_by_alpha)
    cmap = matplotlib.colormaps.get_cmap("RdYlGn").resampled(len(sorted_alphas))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    # Plot baseline strategies from the first alpha (they don't change with alpha)
    first_results = results_by_alpha[sorted_alphas[0]]
    for strategy in ("actual", "random", "bfs"):
        style = dict(_STRATEGY_STYLE[strategy])
        st_list = sorted(first_results.stats[strategy], key=lambda s: s.tau)
        if not st_list:
            continue
        tau_arr = [s.tau for s in st_list]
        ax1.plot(tau_arr, [s.accuracy for s in st_list], **style)
        ax2.plot(tau_arr, [s.mean_steps for s in st_list], **style)

    # Plot IG curve for each alpha
    for i, alpha in enumerate(sorted_alphas):
        color  = cmap(i)
        results = results_by_alpha[alpha]
        st_list = sorted(results.stats["ig"], key=lambda s: s.tau)
        if not st_list:
            continue
        tau_arr = [s.tau for s in st_list]
        lw = 2.0 + 0.5 * i  # thicker for higher alpha
        ax1.plot(tau_arr, [s.accuracy for s in st_list],
                 color=color, marker="D", lw=lw, label=f"IG α={alpha:.2f}", zorder=5)
        ax2.plot(tau_arr, [s.mean_steps for s in st_list],
                 color=color, marker="D", lw=lw, label=f"IG α={alpha:.2f}", zorder=5)

    n = first_results.n_patients
    fig.suptitle(
        f"Speed-Accuracy: IG α comparison — {n} test patients (2+ tests)\n"
        "ConGraph diagnostic pipeline  |  τ sweep",
        fontsize=12, fontweight="bold",
    )

    ax1.set_title("Accuracy vs τ  |  all α values", fontsize=10)
    ax1.set_ylabel("Diagnostic accuracy", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax1.invert_xaxis()
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Mean Tests vs τ  |  all α values", fontsize=10)
    ax2.set_xlabel("Stopping threshold τ (bits)", fontsize=10)
    ax2.set_ylabel("Mean tests ordered", fontsize=10)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = Path(out_dir) / "speed_accuracy_alpha_comparison.png"
    fig.savefig(str(fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Speed-accuracy evaluation with α sweep.")
    p.add_argument("--n-train",      type=int,   default=None,
                   help="Training patients (default: all minus n-test).")
    p.add_argument("--n-test",       type=int,   default=300,
                   help="Test patients (default: 300).")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--min-tests",    type=int,   default=2,
                   help="Min imaging tests per patient (default: 2, i.e. 3+ steps).")
    p.add_argument("--k",            type=int,   default=15)
    p.add_argument("--alphas",       type=str,   default="0.0,0.25,0.5,0.75,1.0",
                   help="Comma-separated α values to sweep (default: 0.0,0.25,0.5,0.75,1.0).")
    p.add_argument("--tau-min",      type=float, default=0.1)
    p.add_argument("--tau-max",      type=float, default=1.9)
    p.add_argument("--tau-steps",    type=int,   default=19,
                   help="Number of τ values (default: 19).")
    p.add_argument("--random-seeds", type=int,   default=3)
    p.add_argument("--csv",          type=str,   default=None)
    p.add_argument("--plot",         action="store_true",
                   help="Generate PNG figures.")
    p.add_argument("--plot-dir",     type=str,   default="results/figs",
                   help="Directory for output figures (default: results/figs).")
    p.add_argument("--quiet",        action="store_true")
    return p.parse_args()


def main() -> None:
    args    = _parse_args()
    verbose = not args.quiet

    # ── Load & filter ──────────────────────────────────────────────────────
    if verbose:
        print("Loading patient data…")
    all_patients = load_all()

    flat: list[tuple[str, str, list[dict]]] = [
        (disease, pid, steps)
        for disease, patients in all_patients.items()
        for pid, steps in patients.items()
    ]

    # Filter: keep only patients with >= min_tests imaging tests (i.e. min_tests+1 steps)
    min_steps = args.min_tests + 1
    flat_filtered = [(d, pid, s) for d, pid, s in flat if len(s) >= min_steps]
    if verbose:
        print(
            f"  After filter (≥{args.min_tests} tests): "
            f"{len(flat_filtered)}/{len(flat)} patients"
        )

    # ── Train/test split ───────────────────────────────────────────────────
    random.seed(args.seed)
    random.shuffle(flat_filtered)

    n_test  = min(args.n_test, len(flat_filtered))
    n_train = args.n_train or (len(flat_filtered) - n_test)
    n_train = min(n_train, len(flat_filtered) - n_test)

    test_patients  = flat_filtered[:n_test]
    train_patients = flat_filtered[n_test: n_test + n_train]

    # Also build full all_patients dict for reducer
    all_dict = all_patients

    if verbose:
        print(f"  Train: {len(train_patients)}  |  Test: {n_test}")

    # ── Parse alpha values ──────────────────────────────────────────────────
    alpha_values = [float(a.strip()) for a in args.alphas.split(",")]

    # ── Tau grid ────────────────────────────────────────────────────────────
    tau_values = list(np.linspace(args.tau_min, args.tau_max, args.tau_steps))

    # ── Run alpha sweep ─────────────────────────────────────────────────────
    if verbose:
        print(f"\nRunning α sweep: {alpha_values}")
        print(f"τ grid: {args.tau_steps} values in [{args.tau_min}, {args.tau_max}]")

    results_by_alpha = run_alpha_sweep(
        all_patients   = all_dict,
        test_patients  = test_patients,
        train_patients = train_patients,
        tau_values     = tau_values,
        alpha_values   = alpha_values,
        k_knn          = args.k,
        random_seeds   = args.random_seeds,
        verbose        = verbose,
    )

    # ── Print ────────────────────────────────────────────────────────────────
    for alpha, results in sorted(results_by_alpha.items()):
        print_results(results)

    # ── CSV ──────────────────────────────────────────────────────────────────
    if args.csv:
        save_csv(results_by_alpha, args.csv)

    # ── Plots ────────────────────────────────────────────────────────────────
    if args.plot:
        print(f"\nGenerating plots → {args.plot_dir}")
        plot_results(results_by_alpha, args.plot_dir)
        plot_ig_alpha_comparison(results_by_alpha, args.plot_dir)


if __name__ == "__main__":
    main()
