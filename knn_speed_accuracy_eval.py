"""knn_speed_accuracy_eval.py — 1-step KNN Counterfactual Speed-Accuracy Framework

Motivation
----------
The original speed_accuracy_eval.py is constrained to re-ordering tests the
patient *actually* did.  Strategies like BFS or IG can recommend any pending
test, but the simulation can only evaluate tests with known feature deltas.

This framework removes that limitation via **1-step KNN counterfactual
estimation**: instead of applying the real feature delta, we look up K similar
training patients who performed the recommended test and use their inverse-
distance-weighted outcomes to estimate:
  • ΔH     — expected entropy reduction after the test
  • P̂(d)  — expected post-test diagnosis distribution

Only one KNN step is taken per evaluation point (no chaining), so estimation
error does not accumulate.

Sample unit
-----------
(patient, step k): the feature state at step k of the actual trajectory.

For each (patient, step k):
  1. features  = steps[k]["features"]  (tests_done already contains k tests)
  2. H_current = entropy(run_pipeline(features))
  3. pending   = VALID_TESTS − tests_done
  4. For each strategy S: X_S = top-1 recommendation from pending tests
  5. (ΔH, P̂) = reducer.test_knn_outcomes(features, [X_S])
  6. If H_current − ΔH < τ → mark (patient, k) as stopping point for S

Metrics at threshold τ
----------------------
  Speed    = mean over patients of (number of tests done at min stopping step)
  Accuracy = mean over patients of [argmax(P̂) == ground_truth at stopping step]

Usage
-----
  python knn_speed_accuracy_eval.py
  python knn_speed_accuracy_eval.py --n-train 800 --n-test 200
  python knn_speed_accuracy_eval.py --alphas 0.0,0.5,1.0
  python knn_speed_accuracy_eval.py --plot --plot-dir results/figs
  python knn_speed_accuracy_eval.py --csv results/knn_speed_accuracy.csv
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from feature_schema import VALID_TESTS
from clinical_session import ClinicalSession
from diagnosis_distribution import compute_distribution
from traversal_engine import run_full_traversal
from entropy_reducer import EntropyReducer, distribution_entropy, DISEASES
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

KNN_STRATEGIES = ("actual", "actual_real", "random", "bfs", "ig")

_UNIFORM_DIST: dict[str, float] = {d: 1.0 / len(DISEASES) for d in DISEASES}


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
# Pipeline helper
# ---------------------------------------------------------------------------

def _run_pipeline_dist(features: dict):
    """Return DiagnosisDistribution for the given features."""
    traversal = run_full_traversal(features)
    return compute_distribution(traversal, features)


def _pending_tests(features: dict) -> list[str]:
    done = set(features.get("tests_done", []))
    return [t for t in VALID_TESTS if t not in done]


# ---------------------------------------------------------------------------
# Per-strategy top-1 recommendation helpers
# ---------------------------------------------------------------------------

def _bfs_top1(features: dict, pending: list[str]) -> Optional[str]:
    if not pending:
        return None
    session = ClinicalSession(features)
    state   = session.assess()
    ranked  = [r.test for r in state.recommendations if r.test in pending]
    return ranked[0] if ranked else pending[0]


def _ig_top1(
    features: dict,
    pending: list[str],
    ig_rec: IGRecommender,
) -> Optional[str]:
    if not pending:
        return None
    session  = ClinicalSession(features)
    state    = session.assess()
    filtered = [r for r in state.recommendations if r.test in pending]
    if filtered:
        reranked = ig_rec.rerank(filtered, features)
        return reranked[0].test if reranked else pending[0]
    # BFS found nothing relevant in pending.
    # At α=0 there is no IG signal, so fall back identically to BFS (pending[0]).
    # At α>0 use KNN entropy scores to pick the most informative test.
    if ig_rec.alpha == 0.0:
        return pending[0]
    ig_scores = ig_rec.reducer.test_entropy_scores(features, pending)
    return max(pending, key=lambda t: ig_scores.get(t, 0.0))


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PatientStopResult:
    """KNN-estimated stopping result for one patient / strategy / τ."""
    stop_tests:     int    # tests done at the stopping step
    correct:        bool   # estimated primary diagnosis == ground truth
    ever_stopped:   bool   # False if no step met the τ condition (used full traj)


@dataclass
class CalibrationRecord:
    """One (patient, step k) KNN-estimate vs real-outcome comparison."""
    disease:       str
    step_k:        int
    test_key:      str
    knn_delta_H:   float             # KNN-estimated entropy reduction
    real_delta_H:  float             # actual H_k − H_{k+1}
    knn_top1:      str               # argmax of KNN-estimated post-test dist
    real_top1:     str               # argmax of real step k+1 dist
    l1_dist:       float             # L1 distance between KNN and real diag_dist


# ---------------------------------------------------------------------------
# Per-patient evaluation
# ---------------------------------------------------------------------------

def evaluate_patient(
    disease:      str,
    steps:        list[dict],
    reducer:      EntropyReducer,
    ig_rec:       IGRecommender,
    tau_values:   list[float],
    random_seeds: int = 5,
) -> dict[str, dict[float, PatientStopResult]]:
    """
    Evaluate all strategies × τ values for one patient.

    Returns
    -------
    {strategy: {tau: PatientStopResult}}
    Empty dict if the patient has fewer than 2 steps.
    """
    n_steps = len(steps)
    if n_steps < 2:
        return {}

    ground_truth = disease

    # ── Step 1: pre-compute per-step context ──────────────────────────────────
    step_ctx: list[dict] = []
    for k in range(n_steps):
        features = steps[k]["features"]
        dist     = _run_pipeline_dist(features)
        H        = distribution_entropy(dist)
        pending  = _pending_tests(features)
        step_ctx.append({
            "features": features,
            "H":        H,
            "dist":     dist,
            "pending":  pending,
        })

    # ── Step 2: pre-compute recommendations + KNN outcomes per step ──────────
    # step_outcomes[(k, strategy)] = {"delta_H": float, "diag_dist": dict} | None
    step_outcomes: dict[tuple[int, str], Optional[dict]] = {}

    for k in range(n_steps - 1):   # step 0 … T-2; must have a "next step" slot
        features = step_ctx[k]["features"]
        pending  = step_ctx[k]["pending"]

        if not pending:
            break   # no more tests can be recommended from this step onward

        # Collect all tests that any strategy might recommend at this step
        tests_to_query: set[str] = set()

        actual_next = steps[k + 1].get("test_key")
        if actual_next and actual_next in VALID_TESTS:
            tests_to_query.add(actual_next)

        bfs_next = _bfs_top1(features, pending)
        if bfs_next:
            tests_to_query.add(bfs_next)

        ig_next = _ig_top1(features, pending, ig_rec)
        if ig_next:
            tests_to_query.add(ig_next)

        rng_nexts = [
            random.Random(seed * 10_000 + k).choice(pending)
            for seed in range(random_seeds)
        ]
        tests_to_query.update(rng_nexts)

        # Single batched KNN call for all candidate tests
        knn = reducer.test_knn_outcomes(features, list(tests_to_query))

        # actual (KNN-estimated)
        if actual_next and actual_next in VALID_TESTS:
            step_outcomes[(k, "actual")] = knn.get(actual_next)
        else:
            step_outcomes[(k, "actual")] = None

        # actual_real: real observed outcome from step k+1 (no KNN)
        real_delta_H = step_ctx[k]["H"] - step_ctx[k + 1]["H"]
        real_diag    = dict(step_ctx[k + 1]["dist"].probabilities)
        step_outcomes[(k, "actual_real")] = {"delta_H": real_delta_H, "diag_dist": real_diag}

        # bfs
        step_outcomes[(k, "bfs")] = knn.get(bfs_next) if bfs_next else None

        # ig
        step_outcomes[(k, "ig")] = knn.get(ig_next) if ig_next else None

        # random — store per-seed outcomes independently (NOT averaged).
        # Averaging at the step level creates an oracle ensemble (the mixed
        # diag_dist is more stable than any single test result).  Instead we
        # keep each seed's outcome separate and average PatientStopResult at
        # the patient level, mirroring speed_accuracy_eval.py's approach.
        for s, rng_test in enumerate(rng_nexts):
            step_outcomes[(k, "random", s)] = knn.get(rng_test)

    # Fallback distribution: actual final step's computed distribution
    final_dist = dict(step_ctx[-1]["dist"].probabilities)

    # ── Step 3: find stopping step for each strategy × τ ─────────────────────
    results: dict[str, dict[float, PatientStopResult]] = {
        s: {} for s in KNN_STRATEGIES
    }

    # For actual_real the final fallback uses the real last-step distribution
    # (same as final_dist already computed below)

    def _one_stop(outcome_key_seq: list, tau: float) -> PatientStopResult:
        """
        Simulate one trajectory given a per-step outcome key sequence.

        outcome_key_seq[k] is the key into step_outcomes for step k.
        """
        stop_k    = None
        stop_diag = None

        for k, okey in enumerate(outcome_key_seq):
            H_current = step_ctx[k]["H"]
            outcome   = step_outcomes.get(okey)
            if outcome is None:
                continue
            if H_current - outcome["delta_H"] < tau:
                stop_k    = k
                stop_diag = outcome["diag_dist"]
                break

        if stop_k is None:
            _st = len(step_ctx[-1]["features"].get("tests_done", []))
            _sd = final_dist
            _ev = False
        else:
            _st = len(step_ctx[stop_k]["features"].get("tests_done", []))
            _sd = stop_diag
            _ev = True

        _pred = max(_sd, key=lambda d: _sd[d])  # type: ignore[arg-type]
        return PatientStopResult(
            stop_tests   = _st,
            correct      = _pred == ground_truth,
            ever_stopped = _ev,
        )

    # Pre-build per-step key sequences for each strategy
    det_keys: dict[str, list] = {
        s: [(k, s) for k in range(n_steps - 1)]
        for s in ("actual", "actual_real", "bfs", "ig")
    }
    rng_keys_per_seed: list[list] = [
        [(k, "random", s) for k in range(n_steps - 1)]
        for s in range(random_seeds)
    ]

    for tau in tau_values:
        # Deterministic strategies
        for strategy in ("actual", "actual_real", "bfs", "ig"):
            results[strategy][tau] = _one_stop(det_keys[strategy], tau)

        # Random: each seed runs an independent trajectory; average at patient level
        seed_psrs = [_one_stop(rng_keys_per_seed[s], tau) for s in range(random_seeds)]
        results["random"][tau] = PatientStopResult(
            stop_tests   = int(round(float(np.mean([r.stop_tests   for r in seed_psrs])))),
            correct      = float(np.mean([r.correct      for r in seed_psrs])) >= 0.5,
            ever_stopped = float(np.mean([r.ever_stopped for r in seed_psrs])) >= 0.5,
        )

    return results


# ---------------------------------------------------------------------------
# KNN calibration: estimate vs real per (patient, step k)
# ---------------------------------------------------------------------------

def collect_patient_calibration(
    disease: str,
    steps:   list[dict],
    reducer: EntropyReducer,
) -> list[CalibrationRecord]:
    """
    For each consecutive step pair (k, k+1) in the patient's trajectory,
    compare the KNN-estimated outcome for the actual test against the real
    observed outcome.

    Returns a list of CalibrationRecord, one per evaluable (k, actual_next)
    pair.  Empty if the patient has fewer than 2 steps.
    """
    n_steps = len(steps)
    if n_steps < 2:
        return []

    # Pre-compute actual pipeline context for every step
    step_ctx: list[dict] = []
    for k in range(n_steps):
        features = steps[k]["features"]
        dist     = _run_pipeline_dist(features)
        H        = distribution_entropy(dist)
        step_ctx.append({"features": features, "H": H, "dist": dist})

    records: list[CalibrationRecord] = []

    for k in range(n_steps - 1):
        actual_next = steps[k + 1].get("test_key")
        if not actual_next or actual_next not in VALID_TESTS:
            continue

        # KNN estimate
        knn = reducer.test_knn_outcomes(step_ctx[k]["features"], [actual_next])
        outcome = knn.get(actual_next)
        if outcome is None:
            continue

        knn_delta_H  = outcome["delta_H"]
        knn_diag     = outcome["diag_dist"]

        # Real outcome
        real_delta_H = step_ctx[k]["H"] - step_ctx[k + 1]["H"]
        real_diag    = dict(step_ctx[k + 1]["dist"].probabilities)

        knn_top1  = max(knn_diag,  key=lambda d: knn_diag[d])   # type: ignore[arg-type]
        real_top1 = max(real_diag, key=lambda d: real_diag[d])  # type: ignore[arg-type]

        l1 = sum(abs(knn_diag.get(d, 0.0) - real_diag.get(d, 0.0)) for d in DISEASES)

        records.append(CalibrationRecord(
            disease      = disease,
            step_k       = k,
            test_key     = actual_next,
            knn_delta_H  = knn_delta_H,
            real_delta_H = real_delta_H,
            knn_top1     = knn_top1,
            real_top1    = real_top1,
            l1_dist      = l1,
        ))

    return records


def run_knn_calibration(
    test_patients:  list[tuple[str, str, list[dict]]],
    reducer:        EntropyReducer,
    verbose:        bool = True,
) -> list[CalibrationRecord]:
    """
    Collect calibration records for all test patients and return them.
    Prints a summary report if verbose=True.
    """
    all_records: list[CalibrationRecord] = []
    total = len(test_patients)
    for idx, (disease, pid, steps) in enumerate(test_patients, 1):
        if verbose and idx % 50 == 0:
            print(f"    {idx}/{total} patients…", flush=True)
        all_records.extend(collect_patient_calibration(disease, steps, reducer))

    if verbose:
        print_calibration_report(all_records)
    return all_records


def print_calibration_report(records: list[CalibrationRecord]) -> None:
    """Print a human-readable KNN calibration summary."""
    if not records:
        print("  No calibration records.")
        return

    n = len(records)
    dH_errors   = [abs(r.knn_delta_H - r.real_delta_H) for r in records]
    dH_biases   = [r.knn_delta_H - r.real_delta_H       for r in records]
    l1_dists    = [r.l1_dist                             for r in records]
    top1_agree  = [r.knn_top1 == r.real_top1             for r in records]

    mae_dH   = float(np.mean(dH_errors))
    rmse_dH  = float(np.sqrt(np.mean([e ** 2 for e in dH_errors])))
    bias_dH  = float(np.mean(dH_biases))
    mean_l1  = float(np.mean(l1_dists))
    top1_pct = float(np.mean(top1_agree))

    # Pearson r between knn and real delta_H
    knn_dH_arr  = np.array([r.knn_delta_H  for r in records])
    real_dH_arr = np.array([r.real_delta_H for r in records])
    if knn_dH_arr.std() > 1e-9 and real_dH_arr.std() > 1e-9:
        corr_dH = float(np.corrcoef(knn_dH_arr, real_dH_arr)[0, 1])
    else:
        corr_dH = float("nan")

    print(f"\n{'═'*68}")
    print(f"  KNN CALIBRATION REPORT   ({n} transition records)")
    print(f"{'═'*68}")
    print(f"  ΔH  MAE          : {mae_dH:+.4f} bits")
    print(f"  ΔH  RMSE         : {rmse_dH:.4f} bits")
    print(f"  ΔH  bias (knn−real): {bias_dH:+.4f} bits  "
          f"({'over' if bias_dH > 0 else 'under'}-estimates reduction)")
    print(f"  ΔH  Pearson r    : {corr_dH:.3f}")
    print(f"  diag_dist L1     : {mean_l1:.4f}  (max=2.0 for 4-class)")
    print(f"  argmax agreement : {top1_pct:.1%}  ({sum(top1_agree)}/{n})")

    # Per-disease breakdown
    diseases_seen = sorted({r.disease for r in records})
    print(f"\n  {'Disease':<16}  {'n':>5}  {'MAE ΔH':>8}  {'bias ΔH':>9}  "
          f"{'L1':>6}  {'top1 agree':>10}")
    print(f"  {'─'*64}")
    for d in diseases_seen:
        sub = [r for r in records if r.disease == d]
        s_n    = len(sub)
        s_mae  = float(np.mean([abs(r.knn_delta_H - r.real_delta_H) for r in sub]))
        s_bias = float(np.mean([r.knn_delta_H - r.real_delta_H       for r in sub]))
        s_l1   = float(np.mean([r.l1_dist                             for r in sub]))
        s_top1 = float(np.mean([r.knn_top1 == r.real_top1            for r in sub]))
        print(f"  {d:<16}  {s_n:>5}  {s_mae:>8.4f}  {s_bias:>+9.4f}  "
              f"{s_l1:>6.4f}  {s_top1:>9.1%}")

    # Per-step-index breakdown (k=0,1,2,3+)
    print(f"\n  {'Step k':<8}  {'n':>5}  {'MAE ΔH':>8}  {'L1':>6}  {'top1 agree':>10}")
    print(f"  {'─'*44}")
    for k_label, pred in [("0", lambda r: r.step_k == 0),
                           ("1", lambda r: r.step_k == 1),
                           ("2", lambda r: r.step_k == 2),
                           ("3+", lambda r: r.step_k >= 3)]:
        sub = [r for r in records if pred(r)]
        if not sub:
            continue
        s_n    = len(sub)
        s_mae  = float(np.mean([abs(r.knn_delta_H - r.real_delta_H) for r in sub]))
        s_l1   = float(np.mean([r.l1_dist                            for r in sub]))
        s_top1 = float(np.mean([r.knn_top1 == r.real_top1           for r in sub]))
        print(f"  {k_label:<8}  {s_n:>5}  {s_mae:>8.4f}  {s_l1:>6.4f}  {s_top1:>9.1%}")

    print(f"{'═'*68}\n")


# ---------------------------------------------------------------------------
# Aggregated statistics
# ---------------------------------------------------------------------------

@dataclass
class KNNStrategyStats:
    tau:             float
    strategy:        str
    n_patients:      int
    mean_stop_tests: float
    accuracy:        float
    pct_early_stop:  float


@dataclass
class KNNEvalResults:
    tau_values: list[float]
    alpha:      float
    stats:      dict[str, list[KNNStrategyStats]]
    n_patients: int


# ---------------------------------------------------------------------------
# Core evaluation  (one alpha value)
# ---------------------------------------------------------------------------

def run_knn_eval(
    test_patients:  list[tuple[str, str, list[dict]]],
    tau_values:     list[float],
    reducer:        EntropyReducer,
    ig_rec:         IGRecommender,
    alpha:          float,
    random_seeds:   int = 5,
    verbose:        bool = True,
) -> KNNEvalResults:
    """
    Run the KNN speed-accuracy sweep for one alpha value.

    Parameters
    ----------
    test_patients : list of (disease, patient_id, steps)
    tau_values    : stopping thresholds to evaluate
    reducer       : fitted EntropyReducer (with diag_dist_after in corpus)
    ig_rec        : IGRecommender for the desired alpha
    alpha         : stored in KNNEvalResults for labelling
    random_seeds  : seeds used to average the random strategy
    verbose       : print progress
    """
    # Accumulate per-patient results
    accum: dict[str, dict[float, list[PatientStopResult]]] = {
        s: defaultdict(list) for s in KNN_STRATEGIES
    }

    total = len(test_patients)
    for idx, (disease, pid, steps) in enumerate(test_patients, 1):
        if verbose and idx % 50 == 0:
            print(f"    {idx}/{total} patients…", flush=True)

        patient_results = evaluate_patient(
            disease      = disease,
            steps        = steps,
            reducer      = reducer,
            ig_rec       = ig_rec,
            tau_values   = tau_values,
            random_seeds = random_seeds,
        )
        if not patient_results:
            continue

        for strategy, tau_dict in patient_results.items():
            for tau, res in tau_dict.items():
                accum[strategy][tau].append(res)

    # Aggregate
    stats: dict[str, list[KNNStrategyStats]] = {s: [] for s in KNN_STRATEGIES}
    n_patients = 0
    for strategy in KNN_STRATEGIES:
        for tau in tau_values:
            results = accum[strategy][tau]
            if not results:
                continue
            if strategy == "actual":
                n_patients = len(results)
            stats[strategy].append(KNNStrategyStats(
                tau             = tau,
                strategy        = strategy,
                n_patients      = len(results),
                mean_stop_tests = float(np.mean([r.stop_tests for r in results])),
                accuracy        = float(np.mean([r.correct for r in results])),
                pct_early_stop  = float(np.mean([r.ever_stopped for r in results])),
            ))

    return KNNEvalResults(
        tau_values = tau_values,
        alpha      = alpha,
        stats      = stats,
        n_patients = n_patients,
    )


# ---------------------------------------------------------------------------
# Multi-alpha sweep
# ---------------------------------------------------------------------------

def run_knn_alpha_sweep(
    test_patients:  list[tuple[str, str, list[dict]]],
    train_patients: list[tuple[str, str, list[dict]]],
    tau_values:     list[float],
    alpha_values:   list[float],
    k_knn:          int  = 15,
    random_seeds:   int  = 5,
    verbose:        bool = True,
) -> dict[float, KNNEvalResults]:
    """
    Fit EntropyReducer + EmpiricalScorer once on training data, then sweep α.

    Returns {alpha: KNNEvalResults}
    """
    # ── Fit EntropyReducer ───────────────────────────────────────────────────
    if verbose:
        print("  Fitting EntropyReducer on training data…", flush=True)
    train_dict: dict[str, dict[str, list[dict]]] = {d: {} for d in DISEASE_FILES}
    for disease, pid, steps in train_patients:
        train_dict[disease][pid] = steps

    reducer = EntropyReducer(k=k_knn)
    reducer.fit(train_dict)
    if verbose:
        print(f"    → {reducer.n_transitions} transitions in corpus", flush=True)

    # ── Fit EmpiricalScorer ──────────────────────────────────────────────────
    if verbose:
        print("  Fitting EmpiricalScorer on training data…", flush=True)
    train_pairs = [
        (steps[0]["features"], disease)
        for disease, pid, steps in train_patients
    ]
    emp_scorer = EmpiricalScorer(k=k_knn)
    emp_scorer.fit(train_pairs)
    _dd.set_empirical_scorer(emp_scorer)

    # ── Alpha sweep ──────────────────────────────────────────────────────────
    results_by_alpha: dict[float, KNNEvalResults] = {}

    for alpha in alpha_values:
        if verbose:
            print(
                f"\n  ── α = {alpha:.2f}  ({len(test_patients)} test patients) ──",
                flush=True,
            )
        ig_rec = IGRecommender(reducer, alpha=alpha)
        results = run_knn_eval(
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

def print_knn_results(results: KNNEvalResults) -> None:
    alpha = results.alpha
    print(f"\n{'═'*76}")
    print(f"  KNN SPEED-ACCURACY SWEEP   α={alpha:.2f}   n={results.n_patients} patients")
    print(f"  (τ = entropy stopping threshold in bits; speed = tests done at stop)")
    print(f"{'═'*76}")
    print(f"  {'τ':>5}  {'Strategy':<14}  {'Accuracy':>8}  {'Mean tests':>10}  {'Early stop%':>11}")
    print(f"  {'─'*66}")

    for tau in results.tau_values:
        for strategy in KNN_STRATEGIES:
            st_list = [s for s in results.stats[strategy] if abs(s.tau - tau) < 1e-9]
            if not st_list:
                continue
            st    = st_list[0]
            if strategy == "ig":
                label = f"ig(α={alpha:.2f})"
            elif strategy == "actual_real":
                label = "actual(real)"
            else:
                label = strategy
            print(
                f"  {tau:>5.2f}  {label:<14}  "
                f"{st.accuracy:>8.1%}  "
                f"{st.mean_stop_tests:>10.2f}  "
                f"{st.pct_early_stop:>10.1%}"
            )
        print()

    print(f"{'═'*76}\n")


def save_knn_csv(results_by_alpha: dict[float, KNNEvalResults], path: str) -> None:
    import csv
    rows = []
    for alpha, results in results_by_alpha.items():
        for strategy in KNN_STRATEGIES:
            for st in results.stats[strategy]:
                rows.append({
                    "alpha":           alpha,
                    "tau":             st.tau,
                    "strategy":        st.strategy,
                    "n_patients":      st.n_patients,
                    "accuracy":        st.accuracy,
                    "mean_stop_tests": st.mean_stop_tests,
                    "pct_early_stop":  st.pct_early_stop,
                })
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved to {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_STRATEGY_STYLE: dict[str, dict] = {
    "actual":      dict(color="#90CAF9", marker="o", label="Actual order (KNN est.)", lw=1.8, zorder=4, ls="--"),
    "actual_real": dict(color="#1565C0", marker="o", label="Actual order (real)",     lw=2.2, zorder=6),
    "random":      dict(color="#9E9E9E", marker="s", label="Random order",            lw=1.5, zorder=2, ls="--"),
    "bfs":         dict(color="#F44336", marker="^", label="BFS order",               lw=2.0, zorder=3),
    "ig":          dict(color="#4CAF50", marker="D", label="IG re-ranked",            lw=2.5, zorder=5),
}


def plot_knn_results(
    results_by_alpha: dict[float, KNNEvalResults],
    out_dir: str,
    title_suffix: str = "",
) -> None:
    """Generate one PNG per alpha value."""
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

        for strategy in KNN_STRATEGIES:
            style = dict(_STRATEGY_STYLE[strategy])
            if strategy == "ig":
                style["label"] = f"IG re-ranked (α={alpha:.2f})"
            st_list = sorted(results.stats[strategy], key=lambda s: s.tau)
            if not st_list:
                continue
            tau_arr  = [s.tau             for s in st_list]
            acc_arr  = [s.accuracy        for s in st_list]
            spd_arr  = [s.mean_stop_tests for s in st_list]

            ax1.plot(tau_arr, acc_arr, **style)
            ax2.plot(tau_arr, spd_arr, **style)

        n = results.n_patients
        fig.suptitle(
            f"KNN Speed-Accuracy (1-step counterfactual) — {n} test patients\n"
            f"ConGraph  |  τ sweep  |  α={alpha:.2f}"
            + (f"  |  {title_suffix}" if title_suffix else ""),
            fontsize=12, fontweight="bold",
        )

        ax1.set_title(f"Accuracy vs τ  |  α={alpha:.2f}", fontsize=10)
        ax1.set_ylabel("Diagnostic accuracy", fontsize=10)
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"{y:.0%}")
        )
        ax1.invert_xaxis()
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.set_title(f"Mean Tests at Stop vs τ  |  α={alpha:.2f}", fontsize=10)
        ax2.set_xlabel("Stopping threshold τ (bits)", fontsize=10)
        ax2.set_ylabel("Mean tests done at stop", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = Path(out_dir) / f"knn_speed_accuracy_alpha_{alpha:.2f}.png"
        fig.savefig(str(fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_knn_ig_comparison(
    results_by_alpha: dict[float, KNNEvalResults],
    out_dir: str,
) -> None:
    """Single figure comparing all α values (IG curve only) plus fixed baselines."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sorted_alphas = sorted(results_by_alpha)
    cmap = matplotlib.colormaps.get_cmap("RdYlGn").resampled(len(sorted_alphas))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    first = results_by_alpha[sorted_alphas[0]]
    for strategy in ("actual", "actual_real", "random", "bfs"):
        style   = dict(_STRATEGY_STYLE[strategy])
        st_list = sorted(first.stats[strategy], key=lambda s: s.tau)
        if not st_list:
            continue
        tau_arr = [s.tau for s in st_list]
        ax1.plot(tau_arr, [s.accuracy        for s in st_list], **style)
        ax2.plot(tau_arr, [s.mean_stop_tests for s in st_list], **style)

    for i, alpha in enumerate(sorted_alphas):
        color   = cmap(i)
        results = results_by_alpha[alpha]
        st_list = sorted(results.stats["ig"], key=lambda s: s.tau)
        if not st_list:
            continue
        tau_arr = [s.tau for s in st_list]
        lw = 2.0 + 0.4 * i
        ax1.plot(tau_arr, [s.accuracy        for s in st_list],
                 color=color, marker="D", lw=lw, label=f"IG α={alpha:.2f}", zorder=5)
        ax2.plot(tau_arr, [s.mean_stop_tests for s in st_list],
                 color=color, marker="D", lw=lw, label=f"IG α={alpha:.2f}", zorder=5)

    n = first.n_patients
    fig.suptitle(
        f"KNN Speed-Accuracy: IG α comparison — {n} test patients\n"
        "ConGraph  |  1-step counterfactual  |  τ sweep",
        fontsize=12, fontweight="bold",
    )

    ax1.set_title("Accuracy vs τ  |  all α values", fontsize=10)
    ax1.set_ylabel("Diagnostic accuracy", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax1.invert_xaxis()
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Mean Tests at Stop vs τ  |  all α values", fontsize=10)
    ax2.set_xlabel("Stopping threshold τ (bits)", fontsize=10)
    ax2.set_ylabel("Mean tests done at stop", fontsize=10)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = Path(out_dir) / "knn_speed_accuracy_alpha_comparison.png"
    fig.savefig(str(fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KNN counterfactual speed-accuracy evaluation with α sweep."
    )
    p.add_argument("--n-train",      type=int,   default=None)
    p.add_argument("--n-test",       type=int,   default=300)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--min-tests",    type=int,   default=2,
                   help="Min imaging tests per patient (default: 2).")
    p.add_argument("--k",            type=int,   default=15)
    p.add_argument("--alphas",       type=str,   default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--tau-min",      type=float, default=0.1)
    p.add_argument("--tau-max",      type=float, default=1.9)
    p.add_argument("--tau-steps",    type=int,   default=19)
    p.add_argument("--random-seeds", type=int,   default=3)
    p.add_argument("--csv",          type=str,   default=None)
    p.add_argument("--plot",         action="store_true")
    p.add_argument("--plot-dir",     type=str,   default="results/figs")
    p.add_argument("--calibration",  action="store_true",
                   help="Run KNN calibration report (estimate vs real outcome).")
    p.add_argument("--quiet",        action="store_true")
    return p.parse_args()


def main() -> None:
    args    = _parse_args()
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

    # ── Optional: KNN calibration report ─────────────────────────────────────
    if args.calibration:
        if verbose:
            print("\nFitting EntropyReducer for calibration…")
        train_dict: dict[str, dict[str, list[dict]]] = {d: {} for d in DISEASE_FILES}
        for disease, pid, steps in train_patients:
            train_dict[disease][pid] = steps
        cal_reducer = EntropyReducer(k=args.k)
        cal_reducer.fit(train_dict)
        if verbose:
            print(f"  → {cal_reducer.n_transitions} transitions")
            print(f"\nRunning KNN calibration on {len(test_patients)} test patients…")
        run_knn_calibration(test_patients, cal_reducer, verbose=verbose)

    # ── Main α sweep ──────────────────────────────────────────────────────────
    alpha_values = [float(a.strip()) for a in args.alphas.split(",")]
    tau_values   = list(np.linspace(args.tau_min, args.tau_max, args.tau_steps))

    if verbose:
        print(f"\nRunning KNN α sweep: {alpha_values}")
        print(f"τ grid: {args.tau_steps} values in [{args.tau_min}, {args.tau_max}]")

    results_by_alpha = run_knn_alpha_sweep(
        test_patients  = test_patients,
        train_patients = train_patients,
        tau_values     = tau_values,
        alpha_values   = alpha_values,
        k_knn          = args.k,
        random_seeds   = args.random_seeds,
        verbose        = verbose,
    )

    for alpha, results in sorted(results_by_alpha.items()):
        print_knn_results(results)

    if args.csv:
        save_knn_csv(results_by_alpha, args.csv)

    if args.plot:
        print(f"\nGenerating plots → {args.plot_dir}")
        plot_knn_results(results_by_alpha, args.plot_dir)
        plot_knn_ig_comparison(results_by_alpha, args.plot_dir)


if __name__ == "__main__":
    main()
