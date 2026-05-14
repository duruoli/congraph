"""knn_feature_eval.py — KNN Feature-Simulation Speed-Accuracy Framework

This module evaluates the FeatureSimulator approach, which differs from the
1-step KNN counterfactual in knn_speed_accuracy_eval.py in two key ways:

  1. Simulation unit: instead of predicting (ΔH, diag_dist) per test, it
     predicts a full feature dict, which can be re-fed into the pipeline.

  2. Multi-step chaining: because the simulation output is a feature dict,
     it can be used as the input to the next simulation step, enabling
     arbitrarily long counterfactual trajectories:
         features_0 → simulate(T1) → features_1 → simulate(T2) → features_2 → …

Evaluation modes
----------------
  --calibration    (1-step) Compare simulated vs real features_after for each
                   (patient, step k) transition.  Reports per-feature boolean
                   accuracy, encoded L1 distance, and diagnosis accuracy.

  --multistep      Full multi-step speed-accuracy sweep.  For each patient
                   and strategy, simulate an entire test sequence using
                   FeatureSimulator; record (n_tests_at_stop, correct_dx) at
                   each entropy threshold τ.

Strategies
----------
  actual_real  : real test sequence + real features              (ceiling)
  actual_sim   : real test sequence + simulated features         (simulation quality reference)
  bfs_sim      : BFS-recommended tests + simulated features
  ig_sim       : IG-recommended tests  + simulated features
  random_sim   : random tests          + simulated features

Usage
-----
  python knn_feature_eval.py --calibration
  python knn_feature_eval.py --multistep
  python knn_feature_eval.py --calibration --multistep --n-test 300 --plot --plot-dir results/figs
  python knn_feature_eval.py --alphas 0.0,0.5,1.0 --tau-min 0.2 --tau-max 1.8
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline.feature_schema import VALID_TESTS
from knn.empirical_scorer import EmpiricalScorer, encode_features
from pipeline.clinical_session import ClinicalSession
from pipeline.traversal_engine import run_full_traversal
from pipeline.diagnosis_distribution import compute_distribution
from knn.entropy_reducer import EntropyReducer, distribution_entropy, DISEASES
from knn.ig_recommender import IGRecommender
from knn.feature_simulator import FeatureSimulator
import pipeline.diagnosis_distribution as _dd
from evaluation.evaluation_metrics import (
    PatientStopResult,
    StrategyStats,
    aggregate_stats,
    compute_patient_metrics,
)


# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DISEASE_FILES: dict[str, Path] = {
    "appendicitis":   RESULTS_DIR / "appendicitis_features.json",
    "cholecystitis":  RESULTS_DIR / "cholecystitis_features.json",
    "diverticulitis": RESULTS_DIR / "diverticulitis_features.json",
    "pancreatitis":   RESULTS_DIR / "pancreatitis_features.json",
}

# Strategy names used throughout (real-test strategies first for display order)
SIM_STRATEGIES = ("actual_real", "actual_sim", "bfs_sim", "ig_sim", "random_sim")

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
# Pipeline helpers
# ---------------------------------------------------------------------------

def _run_pipeline_dist(features: dict):
    traversal = run_full_traversal(features)
    return compute_distribution(traversal, features)


def _pending_tests(features: dict) -> list[str]:
    done = set(features.get("tests_done", []))
    return [t for t in VALID_TESTS if t not in done]


def _entropy_from_features(features: dict) -> float:
    dist = _run_pipeline_dist(features)
    return distribution_entropy(dist)


def _primary_dx(features: dict) -> str:
    dist = _run_pipeline_dist(features)
    return dist.ranked[0]


# ---------------------------------------------------------------------------
# Per-strategy test selection helpers
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
    if ig_rec.alpha == 0.0:
        return pending[0]
    ig_scores = ig_rec.reducer.test_entropy_scores(features, pending)
    return max(pending, key=lambda t: ig_scores.get(t, 0.0))


# ============================================================================
# MODE 1: 1-STEP CALIBRATION
# ============================================================================

@dataclass
class FeatureCalibrationRecord:
    """
    Compare simulated vs real features_after for one (patient, step k) pair.
    """
    disease:          str
    step_k:           int
    test_key:         str

    # Feature-level accuracy
    bool_accuracy:    float   # fraction of boolean keys that match
    pain_loc_match:   bool    # whether pain_location is correctly simulated
    vec_l1:           float   # L1 distance between simulated and real encoded vectors

    # Downstream diagnosis quality
    sim_top1:         str     # argmax of diag distribution from simulated features
    real_top1:        str     # argmax of diag distribution from real features_after
    top1_match:       bool    # sim_top1 == real_top1
    sim_correct:      bool    # sim_top1 == ground-truth disease
    real_correct:     bool    # real_top1 == ground-truth disease

    # Entropy quality
    sim_H:            float   # entropy of distribution from simulated features
    real_H:           float   # entropy of distribution from real features_after
    delta_H_sim:      float   # H_before − sim_H  (estimated reduction)
    delta_H_real:     float   # H_before − real_H (actual reduction)


def collect_feature_calibration(
    disease: str,
    steps:   list[dict],
    simulator: FeatureSimulator,
) -> list[FeatureCalibrationRecord]:
    """
    For each consecutive step pair (k → k+1), simulate features_after from
    features_before and compare against the actual features_after.
    """
    from knn.empirical_scorer import _BOOL_KEYS

    n_steps = len(steps)
    if n_steps < 2:
        return []

    records: list[FeatureCalibrationRecord] = []

    # Pre-compute per-step context
    step_ctx = []
    for k in range(n_steps):
        features = steps[k]["features"]
        dist     = _run_pipeline_dist(features)
        H        = distribution_entropy(dist)
        step_ctx.append({"features": features, "H": H, "dist": dist})

    for k in range(n_steps - 1):
        actual_next = steps[k + 1].get("test_key")
        if not actual_next or actual_next not in VALID_TESTS:
            continue

        features_before = step_ctx[k]["features"]
        features_after  = step_ctx[k + 1]["features"]
        H_before        = step_ctx[k]["H"]

        # Simulate
        sim_features = simulator.simulate_features(features_before, actual_next)
        if sim_features is None:
            continue

        # Feature-level comparison: boolean keys
        bool_matches = [
            bool(sim_features.get(key, False)) == bool(features_after.get(key, False))
            for key in _BOOL_KEYS
        ]
        bool_acc = float(sum(bool_matches)) / len(_BOOL_KEYS) if _BOOL_KEYS else 1.0

        pain_loc_match = (
            sim_features.get("pain_location") == features_after.get("pain_location")
        )

        # Encoded vector L1
        vec_sim  = encode_features(sim_features)
        vec_real = encode_features(features_after)
        vec_l1   = float(np.sum(np.abs(vec_sim - vec_real)))

        # Downstream diagnosis
        sim_dist  = _run_pipeline_dist(sim_features)
        real_dist = step_ctx[k + 1]["dist"]

        sim_top1  = sim_dist.ranked[0]
        real_top1 = real_dist.ranked[0]

        sim_H  = distribution_entropy(sim_dist)
        real_H = step_ctx[k + 1]["H"]

        records.append(FeatureCalibrationRecord(
            disease        = disease,
            step_k         = k,
            test_key       = actual_next,
            bool_accuracy  = bool_acc,
            pain_loc_match = pain_loc_match,
            vec_l1         = vec_l1,
            sim_top1       = sim_top1,
            real_top1      = real_top1,
            top1_match     = sim_top1 == real_top1,
            sim_correct    = sim_top1 == disease,
            real_correct   = real_top1 == disease,
            sim_H          = sim_H,
            real_H         = real_H,
            delta_H_sim    = H_before - sim_H,
            delta_H_real   = H_before - real_H,
        ))

    return records


def run_feature_calibration(
    test_patients: list[tuple[str, str, list[dict]]],
    simulator:     FeatureSimulator,
    verbose:       bool = True,
) -> list[FeatureCalibrationRecord]:
    """Collect calibration records for all test patients."""
    all_records: list[FeatureCalibrationRecord] = []
    total = len(test_patients)
    for idx, (disease, pid, steps) in enumerate(test_patients, 1):
        if verbose and idx % 50 == 0:
            print(f"    {idx}/{total} patients…", flush=True)
        all_records.extend(collect_feature_calibration(disease, steps, simulator))

    if verbose:
        print_calibration_report(all_records)
    return all_records


def print_calibration_report(records: list[FeatureCalibrationRecord]) -> None:
    """Human-readable feature simulation calibration report."""
    if not records:
        print("  No calibration records.")
        return

    n = len(records)

    bool_acc_arr   = np.array([r.bool_accuracy   for r in records])
    pain_arr       = np.array([r.pain_loc_match   for r in records], dtype=float)
    vec_l1_arr     = np.array([r.vec_l1           for r in records])
    top1_match_arr = np.array([r.top1_match       for r in records], dtype=float)
    sim_corr_arr   = np.array([r.sim_correct      for r in records], dtype=float)
    real_corr_arr  = np.array([r.real_correct     for r in records], dtype=float)

    sim_H_arr   = np.array([r.sim_H  for r in records])
    real_H_arr  = np.array([r.real_H for r in records])
    dH_sim_arr  = np.array([r.delta_H_sim  for r in records])
    dH_real_arr = np.array([r.delta_H_real for r in records])

    dH_mae  = float(np.mean(np.abs(dH_sim_arr - dH_real_arr)))
    dH_bias = float(np.mean(dH_sim_arr - dH_real_arr))
    if dH_sim_arr.std() > 1e-9 and dH_real_arr.std() > 1e-9:
        dH_corr = float(np.corrcoef(dH_sim_arr, dH_real_arr)[0, 1])
    else:
        dH_corr = float("nan")

    W = 80
    print(f"\n{'═'*W}")
    print(f"  FEATURE SIMULATION CALIBRATION REPORT   ({n} transition records)")
    print(f"{'═'*W}")

    print(f"\n  ── Feature-level accuracy ───────────────────────────────────────────────")
    print(f"  Boolean accuracy (avg over all bool keys): {float(np.mean(bool_acc_arr)):.3f}")
    print(f"  pain_location accuracy                   : {float(np.mean(pain_arr)):.1%}")
    print(f"  Encoded vector L1 distance (mean)        : {float(np.mean(vec_l1_arr)):.4f}")

    print(f"\n  ── Downstream diagnosis quality ─────────────────────────────────────────")
    print(f"  Diag argmax agreement (sim == real)      : {float(np.mean(top1_match_arr)):.1%}  ({int(sum(top1_match_arr))}/{n})")
    print(f"  Simulated features → correct disease     : {float(np.mean(sim_corr_arr)):.1%}")
    print(f"  Real features_after → correct disease    : {float(np.mean(real_corr_arr)):.1%}  (reference)")

    print(f"\n  ── Entropy quality ──────────────────────────────────────────────────────")
    print(f"  ΔH (sim) vs ΔH (real)  MAE  : {dH_mae:.4f} bits")
    print(f"  ΔH bias (sim − real)        : {dH_bias:+.4f} bits")
    print(f"  ΔH Pearson r                : {dH_corr:.3f}")
    print(f"  Mean H (simulated features) : {float(np.mean(sim_H_arr)):.4f} bits")
    print(f"  Mean H (real features_after): {float(np.mean(real_H_arr)):.4f} bits  (reference)")

    # Per-disease breakdown
    diseases_seen = sorted({r.disease for r in records})
    print(f"\n  ── Per-disease breakdown ────────────────────────────────────────────────")
    print(f"  {'Disease':<16}  {'n':>5}  {'bool_acc':>8}  {'pain':>5}  "
          f"{'vec_L1':>7}  {'top1_match':>10}  {'sim_corr':>9}  {'dH_mae':>7}")
    print(f"  {'─'*80}")
    for d in diseases_seen:
        sub = [r for r in records if r.disease == d]
        sn    = len(sub)
        s_ba  = float(np.mean([r.bool_accuracy   for r in sub]))
        s_pm  = float(np.mean([r.pain_loc_match   for r in sub]))
        s_l1  = float(np.mean([r.vec_l1           for r in sub]))
        s_t1  = float(np.mean([r.top1_match       for r in sub]))
        s_sc  = float(np.mean([r.sim_correct      for r in sub]))
        s_dH  = float(np.mean([abs(r.delta_H_sim - r.delta_H_real) for r in sub]))
        print(f"  {d:<16}  {sn:>5}  {s_ba:>8.3f}  {s_pm:>5.1%}  "
              f"{s_l1:>7.4f}  {s_t1:>10.1%}  {s_sc:>9.1%}  {s_dH:>7.4f}")

    # Per-test breakdown
    tests_seen = sorted({r.test_key for r in records})
    print(f"\n  ── Per-test breakdown ───────────────────────────────────────────────────")
    print(f"  {'Test':<28}  {'n':>5}  {'bool_acc':>8}  {'vec_L1':>7}  "
          f"{'top1_match':>10}  {'dH_mae':>7}")
    print(f"  {'─'*72}")
    for t in tests_seen:
        sub = [r for r in records if r.test_key == t]
        sn   = len(sub)
        s_ba = float(np.mean([r.bool_accuracy for r in sub]))
        s_l1 = float(np.mean([r.vec_l1        for r in sub]))
        s_t1 = float(np.mean([r.top1_match    for r in sub]))
        s_dH = float(np.mean([abs(r.delta_H_sim - r.delta_H_real) for r in sub]))
        print(f"  {t:<28}  {sn:>5}  {s_ba:>8.3f}  {s_l1:>7.4f}  "
              f"{s_t1:>10.1%}  {s_dH:>7.4f}")

    print(f"{'═'*W}\n")


# ============================================================================
# MODE 2: MULTI-STEP SPEED-ACCURACY EVALUATION
# ============================================================================

def _simulate_multistep(
    initial_features:    dict,
    actual_test_sequence: list[str],   # actual test_keys for the patient (may be empty)
    strategy:             str,
    simulator:            FeatureSimulator,
    ig_rec:               IGRecommender,
    tau_values:           list[float],
    ground_truth:         str,
    random_seed:          int = 0,
    max_steps:            int = 10,
) -> dict[float, PatientStopResult]:
    """
    Simulate a multi-step trajectory for one patient under one strategy.

    The simulation loop:
        current_features = initial_features
        for step in 0, 1, 2, …:
            H = entropy(pipeline(current_features))
            if H < tau → stop
            if no more pending tests → stop
            pick next test per strategy
            current_features = simulator.simulate_features(current_features, test)
            (if strategy == "actual_real", use real features instead)

    Returns {tau: PatientStopResult}.
    """
    results: dict[float, PatientStopResult] = {}

    # Build sequence of (features, test_done_this_step)
    # We eagerly simulate the full trajectory first, then apply τ thresholds.
    states: list[dict] = [initial_features]   # index 0 = initial (no test yet)

    current = initial_features
    actual_idx = 0    # pointer into actual_test_sequence

    for step in range(max_steps):
        pending = _pending_tests(current)
        if not pending:
            break

        # Pick next test
        if strategy == "actual_real":
            if actual_idx >= len(actual_test_sequence):
                break
            test = actual_test_sequence[actual_idx]
            actual_idx += 1
            if test not in pending:
                break
            # For actual_real, use real features: just look up the next step
            # This is handled externally; here we advance to the real state.
            # (We break here and handle actual_real separately below.)
            break

        elif strategy == "actual_sim":
            if actual_idx >= len(actual_test_sequence):
                break
            test = actual_test_sequence[actual_idx]
            actual_idx += 1
            if test not in pending:
                break

        elif strategy == "bfs_sim":
            test = _bfs_top1(current, pending)
            if test is None:
                break

        elif strategy == "ig_sim":
            test = _ig_top1(current, pending, ig_rec)
            if test is None:
                break

        elif strategy == "random_sim":
            rng  = random.Random(random_seed * 10_000 + step)
            test = rng.choice(pending)

        else:
            break

        # ── Simulate next state ───────────────────────────────────────────────
        # If the simulator lacks KNN coverage for the chosen test, try other
        # pending tests before giving up.  This prevents random_sim (and
        # occasionally bfs/ig) from terminating early due to corpus gaps rather
        # than genuine trajectory exhaustion.
        # actual_sim is excluded: it must follow the real test sequence.
        next_features = simulator.simulate_features(current, test)

        if next_features is None and strategy in ("bfs_sim", "ig_sim", "random_sim"):
            fallback = [t for t in pending if t != test]
            if strategy == "random_sim":
                # Keep randomness: shuffle fallback with a deterministic seed
                rng_fb = random.Random(random_seed * 10_000 + step + 99_999)
                rng_fb.shuffle(fallback)
            for fb_test in fallback:
                next_features = simulator.simulate_features(current, fb_test)
                if next_features is not None:
                    break   # found a simulatable alternative

        if next_features is None:
            break   # no pending test could be simulated → end trajectory

        states.append(next_features)
        current = next_features

    # Apply τ thresholds to the simulated trajectory
    for tau in tau_values:
        stop_k    = None
        stop_feat = None
        for k, feat in enumerate(states):
            H = _entropy_from_features(feat)
            if H < tau:
                stop_k    = k
                stop_feat = feat
                break

        if stop_k is None:
            stop_feat = states[-1]
            ever_stopped = False
        else:
            ever_stopped = True

        results[tau] = compute_patient_metrics(
            stop_features     = stop_feat,
            predicted_disease = _primary_dx(stop_feat),
            gt_disease        = ground_truth,
            ever_stopped      = ever_stopped,
        )

    return results


def evaluate_patient_multistep(
    disease:              str,
    steps:                list[dict],
    simulator:            FeatureSimulator,
    ig_rec:               IGRecommender,
    tau_values:           list[float],
    random_seeds:         int = 5,
) -> dict[str, dict[float, PatientStopResult]]:
    """
    Evaluate all strategies × τ values for one patient using multi-step simulation.

    Returns {strategy: {tau: PatientStopResult}}.
    Empty dict if the patient has fewer than 2 steps.
    """
    n_steps = len(steps)
    if n_steps < 2:
        return {}

    ground_truth         = disease
    initial_features     = steps[0]["features"]
    actual_test_sequence = [
        steps[k]["test_key"]
        for k in range(1, n_steps)
        if steps[k].get("test_key") and steps[k]["test_key"] in VALID_TESTS
    ]

    results: dict[str, dict[float, PatientStopResult]] = {s: {} for s in SIM_STRATEGIES}

    # ── actual_real: real test sequence + real features (no simulation) ────────
    real_states = [steps[k]["features"] for k in range(n_steps)]
    for tau in tau_values:
        stop_k    = None
        stop_feat = None
        for k, feat in enumerate(real_states):
            H = _entropy_from_features(feat)
            if H < tau:
                stop_k    = k
                stop_feat = feat
                break
        if stop_k is None:
            stop_feat    = real_states[-1]
            ever_stopped = False
        else:
            ever_stopped = True
        results["actual_real"][tau] = compute_patient_metrics(
            stop_features     = stop_feat,
            predicted_disease = _primary_dx(stop_feat),
            gt_disease        = ground_truth,
            ever_stopped      = ever_stopped,
        )

    # ── Simulated strategies ───────────────────────────────────────────────────
    for strategy in ("actual_sim", "bfs_sim", "ig_sim"):
        res = _simulate_multistep(
            initial_features      = initial_features,
            actual_test_sequence  = actual_test_sequence,
            strategy              = strategy,
            simulator             = simulator,
            ig_rec                = ig_rec,
            tau_values            = tau_values,
            ground_truth          = ground_truth,
        )
        results[strategy] = res

    # ── random_sim: average over multiple seeds ───────────────────────────────
    seed_results: list[dict[float, PatientStopResult]] = []
    for seed in range(random_seeds):
        res = _simulate_multistep(
            initial_features      = initial_features,
            actual_test_sequence  = actual_test_sequence,
            strategy              = "random_sim",
            simulator             = simulator,
            ig_rec                = ig_rec,
            tau_values            = tau_values,
            ground_truth          = ground_truth,
            random_seed           = seed,
        )
        seed_results.append(res)

    for tau in tau_values:
        psrs = [sr[tau] for sr in seed_results if tau in sr]
        if not psrs:
            continue
        results["random_sim"][tau] = PatientStopResult(
            stop_tests   = int(round(float(np.mean([r.stop_tests   for r in psrs])))),
            correct      = float(np.mean([r.correct      for r in psrs])) >= 0.5,
            ever_stopped = float(np.mean([r.ever_stopped for r in psrs])) >= 0.5,
            cost         = float(np.mean([r.cost         for r in psrs])),
            burden       = float(np.mean([r.burden       for r in psrs])),
        )

    return results


# ---------------------------------------------------------------------------
# Aggregated stats
# ---------------------------------------------------------------------------

@dataclass
class MultiStepEvalResults:
    tau_values: list[float]
    alpha:      float
    stats:      dict[str, list[StrategyStats]]
    n_patients: int


def run_multistep_eval(
    test_patients: list[tuple[str, str, list[dict]]],
    tau_values:    list[float],
    simulator:     FeatureSimulator,
    ig_rec:        IGRecommender,
    alpha:         float,
    random_seeds:  int = 5,
    verbose:       bool = True,
) -> MultiStepEvalResults:
    """Run the multi-step speed-accuracy sweep for one alpha value."""
    accum: dict[str, dict[float, list[PatientStopResult]]] = {
        s: defaultdict(list) for s in SIM_STRATEGIES
    }

    total = len(test_patients)
    for idx, (disease, pid, steps) in enumerate(test_patients, 1):
        if verbose and idx % 50 == 0:
            print(f"    {idx}/{total} patients…", flush=True)

        patient_results = evaluate_patient_multistep(
            disease      = disease,
            steps        = steps,
            simulator    = simulator,
            ig_rec       = ig_rec,
            tau_values   = tau_values,
            random_seeds = random_seeds,
        )
        if not patient_results:
            continue

        for strategy, tau_dict in patient_results.items():
            for tau, res in tau_dict.items():
                accum[strategy][tau].append(res)

    stats: dict[str, list[StrategyStats]] = {s: [] for s in SIM_STRATEGIES}
    n_patients = 0
    for strategy in SIM_STRATEGIES:
        for tau in tau_values:
            r_list = accum[strategy][tau]
            if not r_list:
                continue
            if strategy == "actual_real":
                n_patients = len(r_list)
            stats[strategy].append(aggregate_stats(r_list, tau=tau, strategy=strategy))

    return MultiStepEvalResults(
        tau_values = tau_values,
        alpha      = alpha,
        stats      = stats,
        n_patients = n_patients,
    )


def run_multistep_alpha_sweep(
    test_patients:  list[tuple[str, str, list[dict]]],
    train_patients: list[tuple[str, str, list[dict]]],
    tau_values:     list[float],
    alpha_values:   list[float],
    k_knn:          int  = 15,
    random_seeds:   int  = 5,
    verbose:        bool = True,
) -> dict[float, MultiStepEvalResults]:
    """Fit FeatureSimulator + EntropyReducer + EmpiricalScorer once, then sweep α."""

    if verbose:
        print("  Fitting FeatureSimulator on training data…", flush=True)
    train_dict: dict[str, dict[str, list[dict]]] = {d: {} for d in DISEASE_FILES}
    for disease, pid, steps in train_patients:
        train_dict[disease][pid] = steps

    simulator = FeatureSimulator(k=k_knn)
    simulator.fit(train_dict)
    if verbose:
        summary = simulator.corpus_summary()
        print(f"    → {summary['n_transitions']} transitions in corpus", flush=True)

    # EntropyReducer is still needed for IG recommender
    if verbose:
        print("  Fitting EntropyReducer on training data…", flush=True)
    reducer = EntropyReducer(k=k_knn)
    reducer.fit(train_dict)

    # EmpiricalScorer for diagnosis distribution improvement
    if verbose:
        print("  Fitting EmpiricalScorer on training data…", flush=True)
    train_pairs = [(steps[0]["features"], disease) for disease, pid, steps in train_patients]
    emp_scorer  = EmpiricalScorer(k=k_knn)
    emp_scorer.fit(train_pairs)
    _dd.set_empirical_scorer(emp_scorer)

    results_by_alpha: dict[float, MultiStepEvalResults] = {}
    for alpha in alpha_values:
        if verbose:
            print(
                f"\n  ── α = {alpha:.2f}  ({len(test_patients)} test patients) ──",
                flush=True,
            )
        ig_rec = IGRecommender(reducer, alpha=alpha)
        results = run_multistep_eval(
            test_patients = test_patients,
            tau_values    = tau_values,
            simulator     = simulator,
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

_STRATEGY_LABEL: dict[str, str] = {
    "actual_real": "Real order + real features  (ceiling)",
    "actual_sim":  "Real order + simulated features",
    "bfs_sim":     "Rubric-guided + simulated features",
    "ig_sim":      "Efficiency-opt + simulated features",
    "random_sim":  "Random + simulated features",
}


def print_multistep_results(results: MultiStepEvalResults) -> None:
    alpha = results.alpha
    print(f"\n{'═'*80}")
    print(f"  MULTI-STEP FEATURE-SIM SPEED-ACCURACY   α={alpha:.2f}   n={results.n_patients} patients")
    print(f"  (τ = entropy stopping threshold bits; speed = tests done at stop; sim=KNN-simulated features)")
    print(f"{'═'*80}")
    print(f"  {'τ':>5}  {'Strategy':<38}  {'Accuracy':>8}  {'Mean tests':>10}  {'Early stop%':>11}")
    print(f"  {'─'*78}")

    for tau in results.tau_values:
        for strategy in SIM_STRATEGIES:
            st_list = [s for s in results.stats[strategy] if abs(s.tau - tau) < 1e-9]
            if not st_list:
                continue
            st = st_list[0]
            if strategy == "ig_sim":
                label = f"Efficiency-opt (α={alpha:.2f}) + sim features"
            else:
                label = _STRATEGY_LABEL.get(strategy, strategy)
            print(
                f"  {tau:>5.2f}  {label:<38}  "
                f"{st.accuracy:>8.1%}  "
                f"{st.mean_stop_tests:>10.2f}  "
                f"{st.pct_early_stop:>10.1%}"
            )
        print()

    print(f"{'═'*80}\n")


def save_multistep_csv(results_by_alpha: dict[float, MultiStepEvalResults], path: str) -> None:
    import csv
    rows = []
    for alpha, results in results_by_alpha.items():
        for strategy in SIM_STRATEGIES:
            for st in results.stats[strategy]:
                rows.append({
                    "alpha":           alpha,
                    "tau":             st.tau,
                    "strategy":        st.strategy,
                    "n_patients":      st.n_patients,
                    "accuracy":        st.accuracy,
                    "mean_stop_tests": st.mean_stop_tests,
                    "mean_cost":       st.mean_cost,
                    "mean_burden":     st.mean_burden,
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
    "actual_real": dict(color="#1565C0", marker="o", label="Real order + real features (ceiling)", lw=2.2, zorder=6),
    "actual_sim":  dict(color="#90CAF9", marker="o", label="Real order + simulated features",      lw=1.8, zorder=4, ls="--"),
    "random_sim":  dict(color="#9E9E9E", marker="s", label="Random + simulated features",          lw=1.5, zorder=2, ls="--"),
    "bfs_sim":     dict(color="#F44336", marker="^", label="Rubric-guided + simulated features",   lw=2.0, zorder=3),
    "ig_sim":      dict(color="#4CAF50", marker="D", label="Efficiency-opt + simulated features",  lw=2.5, zorder=5),
}


def plot_multistep_results(
    results_by_alpha: dict[float, MultiStepEvalResults],
    out_dir:          str,
) -> None:
    """Generate accuracy and mean-tests plots for each alpha value."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  matplotlib not installed — skipping plots.")
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for alpha, results in sorted(results_by_alpha.items()):
        a = f"{alpha:.2f}"
        n = results.n_patients
        title_base = (
            f"Multi-Step Feature-Sim Speed-Accuracy — {n} patients  |  α = {alpha:.2f}"
        )

        for metric, ylabel, fname, as_pct in [
            ("accuracy",        "Diagnostic accuracy",     f"feat_sim_accuracy_alpha_{a}.png",   True),
            ("mean_stop_tests", "Mean tests done at stop", f"feat_sim_tests_alpha_{a}.png",       False),
        ]:
            fig, ax = plt.subplots(figsize=(9, 5))
            for strategy in SIM_STRATEGIES:
                style = dict(_STRATEGY_STYLE[strategy])
                if strategy == "ig_sim":
                    style["label"] = f"Efficiency-opt (α={alpha:.2f}) + sim features"
                st_list = sorted(results.stats[strategy], key=lambda s: s.tau)
                if not st_list:
                    continue
                tau_arr = [s.tau              for s in st_list]
                val_arr = [getattr(s, metric) for s in st_list]
                ax.plot(tau_arr, val_arr, **style)

            ax.set_title(title_base, fontsize=10, fontweight="bold")
            ax.set_xlabel("Maximum tolerated uncertainty τ (bits)", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.invert_xaxis()
            if as_pct:
                ax.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda y, _: f"{y:.0%}")
                )
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            fpath = out / fname
            fig.savefig(str(fpath), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {fpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KNN feature-simulation speed-accuracy evaluation."
    )
    p.add_argument("--n-train",       type=int,   default=None)
    p.add_argument("--n-test",        type=int,   default=300)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--min-tests",     type=int,   default=2,
                   help="Min imaging tests per patient (default: 2).")
    p.add_argument("--k",             type=int,   default=15)
    p.add_argument("--alphas",        type=str,   default="0.0,0.5,1.0")
    p.add_argument("--tau-min",       type=float, default=0.1)
    p.add_argument("--tau-max",       type=float, default=1.9)
    p.add_argument("--tau-steps",     type=int,   default=19)
    p.add_argument("--random-seeds",  type=int,   default=3)
    p.add_argument("--calibration",   action="store_true",
                   help="Run 1-step feature calibration report.")
    p.add_argument("--multistep",     action="store_true",
                   help="Run multi-step speed-accuracy sweep (default if neither flag set).")
    p.add_argument("--csv",           type=str,   default=None)
    p.add_argument("--plot",          action="store_true")
    p.add_argument("--plot-dir",      type=str,   default="results/figs")
    p.add_argument("--quiet",         action="store_true")
    return p.parse_args()


def main() -> None:
    args    = _parse_args()
    verbose = not args.quiet

    # Default: run multistep if neither mode flag is given
    run_cal      = args.calibration
    run_multistep = args.multistep or (not args.calibration and not args.multistep)

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
    train_patients = flat_filtered[n_test : n_test + n_train]

    if verbose:
        print(f"  Train: {len(train_patients)}  |  Test: {n_test}")

    # ── Calibration mode ─────────────────────────────────────────────────────
    if run_cal:
        if verbose:
            print("\nFitting FeatureSimulator for calibration…")
        train_dict: dict[str, dict[str, list[dict]]] = {d: {} for d in DISEASE_FILES}
        for disease, pid, steps in train_patients:
            train_dict[disease][pid] = steps
        cal_simulator = FeatureSimulator(k=args.k)
        cal_simulator.fit(train_dict)
        if verbose:
            print(f"  → {cal_simulator.n_transitions} transitions")
            print(f"\nRunning feature calibration on {n_test} test patients…")
        run_feature_calibration(test_patients, cal_simulator, verbose=verbose)

    # ── Multi-step sweep ──────────────────────────────────────────────────────
    if run_multistep:
        alpha_values = [float(a.strip()) for a in args.alphas.split(",")]
        tau_values   = list(np.linspace(args.tau_min, args.tau_max, args.tau_steps))

        if verbose:
            print(f"\nRunning multi-step feature-sim α sweep: {alpha_values}")
            print(f"τ grid: {args.tau_steps} values in [{args.tau_min}, {args.tau_max}]")

        results_by_alpha = run_multistep_alpha_sweep(
            test_patients  = test_patients,
            train_patients = train_patients,
            tau_values     = tau_values,
            alpha_values   = alpha_values,
            k_knn          = args.k,
            random_seeds   = args.random_seeds,
            verbose        = verbose,
        )

        for alpha, results in sorted(results_by_alpha.items()):
            print_multistep_results(results)

        if args.csv:
            save_multistep_csv(results_by_alpha, args.csv)

        if args.plot:
            print(f"\nGenerating plots → {args.plot_dir}")
            plot_multistep_results(results_by_alpha, args.plot_dir)


if __name__ == "__main__":
    main()
