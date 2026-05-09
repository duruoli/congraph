"""knn_cost_burden_eval.py — KNN Speed / Accuracy / Cost / Burden Framework

Extends knn_speed_accuracy_eval.py with two additional per-patient metrics:

  Cost          — additive sum of TEST_COST[t] over tests_done at stopping step.
  Patient burden— additive sum of TEST_BURDEN[t] over tests_done at stopping step.

Both lookup tables live in test_burden_cost.py (1–10 relative scale per test).
Lower cost / burden at a given accuracy is better.

New output columns (CSV, console, plots)
-----------------------------------------
  mean_cost   — mean cumulative cost at stopping step across patients
  mean_burden — mean cumulative burden at stopping step across patients

New plot files (per alpha, per comparison)
------------------------------------------
  knn_cost_alpha_{a}.png / knn_cost_alpha_{a}_deployable.png
  knn_burden_alpha_{a}.png / knn_burden_alpha_{a}_deployable.png
  knn_cost_alpha_comparison.png / knn_cost_alpha_comparison_deployable.png
  knn_burden_alpha_comparison.png / knn_burden_alpha_comparison_deployable.png

Everything else (KNN counterfactual logic, alpha sweep, calibration) is
identical to knn_speed_accuracy_eval.py.

Usage
-----
  python knn_cost_burden_eval.py
  python knn_cost_burden_eval.py --alphas 0.0,0.5,1.0 --plot --plot-dir results/figs/cb
  python knn_cost_burden_eval.py --csv results/knn_cost_burden.csv
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
from test_burden_cost import TEST_COST, TEST_BURDEN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entropy_from_dict(dist: dict[str, float]) -> float:
    H = 0.0
    for p in dist.values():
        if p > 1e-12:
            H -= p * math.log2(p)
    return H


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
# Pipeline helpers
# ---------------------------------------------------------------------------

def _run_pipeline_dist(features: dict):
    traversal = run_full_traversal(features)
    return compute_distribution(traversal, features)


def _pending_tests(features: dict) -> list[str]:
    done = set(features.get("tests_done", []))
    return [t for t in VALID_TESTS if t not in done]


# ---------------------------------------------------------------------------
# Per-strategy top-1 recommendation helpers (unchanged from knn_speed_accuracy)
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
    pending:  list[str],
    ig_rec:   IGRecommender,
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


# ---------------------------------------------------------------------------
# Result dataclasses — extended with cost and burden
# ---------------------------------------------------------------------------

@dataclass
class PatientStopResult:
    """KNN-estimated stopping result for one patient / strategy / τ."""
    stop_tests:     int    # tests done at the stopping step
    correct:        bool   # estimated primary diagnosis == ground truth
    ever_stopped:   bool   # False if no step met the τ condition (used full traj)
    cost:           float  # sum of TEST_COST[t] over tests_done at stopping step
    burden:         float  # sum of TEST_BURDEN[t] over tests_done at stopping step


@dataclass
class CalibrationRecord:
    """One (patient, step k) KNN-estimate vs real-outcome comparison."""
    disease:               str
    step_k:                int
    test_key:              str
    knn_delta_H:           float
    knn_delta_H_from_dist: float
    real_delta_H:          float
    knn_top1:              str
    real_top1:             str
    l1_dist:               float


# ---------------------------------------------------------------------------
# Cost / burden accumulation helper
# ---------------------------------------------------------------------------

def _compute_cost_burden(tests_done: list[str]) -> tuple[float, float]:
    """Return (cumulative_cost, cumulative_burden) for a list of completed tests."""
    cost   = sum(TEST_COST.get(t,   0.0) for t in tests_done)
    burden = sum(TEST_BURDEN.get(t, 0.0) for t in tests_done)
    return cost, burden


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

    Returns {strategy: {tau: PatientStopResult}}.
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
    step_outcomes: dict[tuple, Optional[dict]] = {}

    for k in range(n_steps - 1):
        features = step_ctx[k]["features"]
        pending  = step_ctx[k]["pending"]

        if not pending:
            break

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

        knn = reducer.test_knn_outcomes(features, list(tests_to_query))

        if actual_next and actual_next in VALID_TESTS:
            step_outcomes[(k, "actual")] = knn.get(actual_next)
        else:
            step_outcomes[(k, "actual")] = None

        real_delta_H = step_ctx[k]["H"] - step_ctx[k + 1]["H"]
        real_diag    = dict(step_ctx[k + 1]["dist"].probabilities)
        step_outcomes[(k, "actual_real")] = {"delta_H": real_delta_H, "diag_dist": real_diag}

        step_outcomes[(k, "bfs")] = knn.get(bfs_next) if bfs_next else None
        step_outcomes[(k, "ig")]  = knn.get(ig_next)  if ig_next  else None

        for s, rng_test in enumerate(rng_nexts):
            step_outcomes[(k, "random", s)] = knn.get(rng_test)

    final_dist = dict(step_ctx[-1]["dist"].probabilities)

    # ── Step 3: find stopping step for each strategy × τ ─────────────────────
    results: dict[str, dict[float, PatientStopResult]] = {
        s: {} for s in KNN_STRATEGIES
    }

    def _one_stop(outcome_key_seq: list, tau: float) -> PatientStopResult:
        stop_k    = None
        stop_diag = None

        for k, *okey_rest in outcome_key_seq:
            okey      = (k, *okey_rest)
            H_current = step_ctx[k]["H"]
            outcome   = step_outcomes.get(okey)
            if outcome is None:
                continue
            if H_current - outcome["delta_H"] < tau:
                stop_k    = k
                stop_diag = outcome["diag_dist"]
                break

        if stop_k is None:
            _tests_done = step_ctx[-1]["features"].get("tests_done", [])
            _sd         = final_dist
            _ev         = False
        else:
            _tests_done = step_ctx[stop_k]["features"].get("tests_done", [])
            _sd         = stop_diag
            _ev         = True

        _st             = len(_tests_done)
        _pred           = max(_sd, key=lambda d: _sd[d])  # type: ignore[arg-type]
        _cost, _burden  = _compute_cost_burden(_tests_done)

        return PatientStopResult(
            stop_tests   = _st,
            correct      = _pred == ground_truth,
            ever_stopped = _ev,
            cost         = _cost,
            burden       = _burden,
        )

    # Build per-step key sequences for deterministic strategies
    det_keys: dict[str, list] = {
        s: [(k, s) for k in range(n_steps - 1)]
        for s in ("actual", "actual_real", "bfs", "ig")
    }
    rng_keys_per_seed: list[list] = [
        [(k, "random", s) for k in range(n_steps - 1)]
        for s in range(random_seeds)
    ]

    for tau in tau_values:
        for strategy in ("actual", "actual_real", "bfs", "ig"):
            results[strategy][tau] = _one_stop(det_keys[strategy], tau)

        seed_psrs = [_one_stop(rng_keys_per_seed[s], tau) for s in range(random_seeds)]
        results["random"][tau] = PatientStopResult(
            stop_tests   = int(round(float(np.mean([r.stop_tests   for r in seed_psrs])))),
            correct      = float(np.mean([r.correct      for r in seed_psrs])) >= 0.5,
            ever_stopped = float(np.mean([r.ever_stopped for r in seed_psrs])) >= 0.5,
            cost         = float(np.mean([r.cost         for r in seed_psrs])),
            burden       = float(np.mean([r.burden       for r in seed_psrs])),
        )

    return results


# ---------------------------------------------------------------------------
# KNN calibration (unchanged from knn_speed_accuracy_eval)
# ---------------------------------------------------------------------------

def collect_patient_calibration(
    disease: str,
    steps:   list[dict],
    reducer: EntropyReducer,
) -> list[CalibrationRecord]:
    n_steps = len(steps)
    if n_steps < 2:
        return []

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

        knn = reducer.test_knn_outcomes(step_ctx[k]["features"], [actual_next])
        outcome = knn.get(actual_next)
        if outcome is None:
            continue

        knn_delta_H           = outcome["delta_H"]
        knn_diag              = outcome["diag_dist"]
        H_current             = step_ctx[k]["H"]
        knn_delta_H_from_dist = H_current - _entropy_from_dict(knn_diag)
        real_delta_H          = H_current - step_ctx[k + 1]["H"]
        real_diag             = dict(step_ctx[k + 1]["dist"].probabilities)

        knn_top1  = max(knn_diag,  key=lambda d: knn_diag[d])   # type: ignore[arg-type]
        real_top1 = max(real_diag, key=lambda d: real_diag[d])  # type: ignore[arg-type]
        l1        = sum(abs(knn_diag.get(d, 0.0) - real_diag.get(d, 0.0)) for d in DISEASES)

        records.append(CalibrationRecord(
            disease               = disease,
            step_k                = k,
            test_key              = actual_next,
            knn_delta_H           = knn_delta_H,
            knn_delta_H_from_dist = knn_delta_H_from_dist,
            real_delta_H          = real_delta_H,
            knn_top1              = knn_top1,
            real_top1             = real_top1,
            l1_dist               = l1,
        ))

    return records


def run_knn_calibration(
    test_patients: list[tuple[str, str, list[dict]]],
    reducer:       EntropyReducer,
    verbose:       bool = True,
) -> list[CalibrationRecord]:
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
    if not records:
        print("  No calibration records.")
        return

    n            = len(records)
    knn_dH_arr   = np.array([r.knn_delta_H           for r in records])
    fd_dH_arr    = np.array([r.knn_delta_H_from_dist  for r in records])
    real_dH_arr  = np.array([r.real_delta_H           for r in records])

    internal_diff  = knn_dH_arr - fd_dH_arr
    internal_mae   = float(np.mean(np.abs(internal_diff)))
    internal_bias  = float(np.mean(internal_diff))
    internal_rmse  = float(np.sqrt(np.mean(internal_diff ** 2)))
    internal_corr  = (
        float(np.corrcoef(knn_dH_arr, fd_dH_arr)[0, 1])
        if knn_dH_arr.std() > 1e-9 and fd_dH_arr.std() > 1e-9
        else float("nan")
    )

    mae_dH   = float(np.mean(np.abs(knn_dH_arr - real_dH_arr)))
    rmse_dH  = float(np.sqrt(np.mean((knn_dH_arr - real_dH_arr) ** 2)))
    bias_dH  = float(np.mean(knn_dH_arr - real_dH_arr))
    corr_dH  = (
        float(np.corrcoef(knn_dH_arr, real_dH_arr)[0, 1])
        if knn_dH_arr.std() > 1e-9 and real_dH_arr.std() > 1e-9
        else float("nan")
    )

    mae_fd   = float(np.mean(np.abs(fd_dH_arr - real_dH_arr)))
    rmse_fd  = float(np.sqrt(np.mean((fd_dH_arr - real_dH_arr) ** 2)))
    bias_fd  = float(np.mean(fd_dH_arr - real_dH_arr))
    corr_fd  = (
        float(np.corrcoef(fd_dH_arr, real_dH_arr)[0, 1])
        if fd_dH_arr.std() > 1e-9 and real_dH_arr.std() > 1e-9
        else float("nan")
    )

    l1_dists   = [r.l1_dist                   for r in records]
    top1_agree = [r.knn_top1 == r.real_top1   for r in records]
    mean_l1    = float(np.mean(l1_dists))
    top1_pct   = float(np.mean(top1_agree))

    W = 80
    print(f"\n{'═'*W}")
    print(f"  KNN CALIBRATION REPORT   ({n} transition records)")
    print(f"{'═'*W}")
    print(f"\n  ── Internal discrepancy: direct ΔH  vs  ΔH derived from P̂(d) ──────────")
    print(f"  MAE  (direct − from_dist) : {internal_mae:.4f} bits")
    print(f"  RMSE (direct − from_dist) : {internal_rmse:.4f} bits")
    print(f"  Bias (direct − from_dist) : {internal_bias:+.4f} bits  "
          f"({'direct larger' if internal_bias > 0 else 'from_dist larger'})")
    print(f"  Pearson r (direct, from_dist): {internal_corr:.3f}")
    print(f"\n  ── Accuracy vs real ΔH ──────────────────────────────────────────────────")
    print(f"  {'Metric':<30}  {'direct KNN ΔH':>14}  {'from_dist ΔH':>13}")
    print(f"  {'─'*62}")
    print(f"  {'MAE':<30}  {mae_dH:>14.4f}  {mae_fd:>13.4f}  bits")
    print(f"  {'RMSE':<30}  {rmse_dH:>14.4f}  {rmse_fd:>13.4f}  bits")
    print(f"  {'Bias (knn − real)':<30}  {bias_dH:>+14.4f}  {bias_fd:>+13.4f}  bits")
    print(f"  {'Pearson r with real ΔH':<30}  {corr_dH:>14.3f}  {corr_fd:>13.3f}")
    print(f"\n  ── Diagnosis distribution quality ───────────────────────────────────────")
    print(f"  diag_dist L1     : {mean_l1:.4f}  (max=2.0 for 4-class)")
    print(f"  argmax agreement : {top1_pct:.1%}  ({sum(top1_agree)}/{n})")

    diseases_seen = sorted({r.disease for r in records})
    print(f"\n  ── Per-disease breakdown ────────────────────────────────────────────────")
    print(f"  {'Disease':<16}  {'n':>5}  {'direct MAE':>10}  {'fd MAE':>8}  "
          f"{'int. MAE':>9}  {'L1':>6}  {'top1%':>6}")
    print(f"  {'─'*72}")
    for d in diseases_seen:
        sub   = [r for r in records if r.disease == d]
        s_n   = len(sub)
        s_dir = float(np.mean([abs(r.knn_delta_H - r.real_delta_H)          for r in sub]))
        s_fd  = float(np.mean([abs(r.knn_delta_H_from_dist - r.real_delta_H) for r in sub]))
        s_int = float(np.mean([abs(r.knn_delta_H - r.knn_delta_H_from_dist)  for r in sub]))
        s_l1  = float(np.mean([r.l1_dist                                     for r in sub]))
        s_t1  = float(np.mean([r.knn_top1 == r.real_top1                    for r in sub]))
        print(f"  {d:<16}  {s_n:>5}  {s_dir:>10.4f}  {s_fd:>8.4f}  "
              f"{s_int:>9.4f}  {s_l1:>6.4f}  {s_t1:>5.1%}")

    print(f"{'═'*W}\n")


# ---------------------------------------------------------------------------
# Aggregated statistics — extended with mean_cost and mean_burden
# ---------------------------------------------------------------------------

@dataclass
class KNNStrategyStats:
    tau:             float
    strategy:        str
    n_patients:      int
    mean_stop_tests: float
    accuracy:        float
    pct_early_stop:  float
    mean_cost:       float   # mean cumulative TEST_COST at stopping step
    mean_burden:     float   # mean cumulative TEST_BURDEN at stopping step


@dataclass
class KNNEvalResults:
    tau_values: list[float]
    alpha:      float
    stats:      dict[str, list[KNNStrategyStats]]
    n_patients: int


# ---------------------------------------------------------------------------
# Core evaluation (one alpha value)
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
                mean_stop_tests = float(np.mean([r.stop_tests   for r in results])),
                accuracy        = float(np.mean([r.correct       for r in results])),
                pct_early_stop  = float(np.mean([r.ever_stopped  for r in results])),
                mean_cost       = float(np.mean([r.cost          for r in results])),
                mean_burden     = float(np.mean([r.burden        for r in results])),
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
    """Fit EntropyReducer + EmpiricalScorer once on training data, then sweep α."""
    if verbose:
        print("  Fitting EntropyReducer on training data…", flush=True)
    train_dict: dict[str, dict[str, list[dict]]] = {d: {} for d in DISEASE_FILES}
    for disease, pid, steps in train_patients:
        train_dict[disease][pid] = steps

    reducer = EntropyReducer(k=k_knn)
    reducer.fit(train_dict)
    if verbose:
        print(f"    → {reducer.n_transitions} transitions in corpus", flush=True)

    if verbose:
        print("  Fitting EmpiricalScorer on training data…", flush=True)
    train_pairs = [
        (steps[0]["features"], disease)
        for disease, pid, steps in train_patients
    ]
    emp_scorer = EmpiricalScorer(k=k_knn)
    emp_scorer.fit(train_pairs)
    _dd.set_empirical_scorer(emp_scorer)

    results_by_alpha: dict[float, KNNEvalResults] = {}

    for alpha in alpha_values:
        if verbose:
            print(
                f"\n  ── α = {alpha:.2f}  ({len(test_patients)} test patients) ──",
                flush=True,
            )
        ig_rec = IGRecommender(reducer, alpha=alpha)
        results_by_alpha[alpha] = run_knn_eval(
            test_patients = test_patients,
            tau_values    = tau_values,
            reducer       = reducer,
            ig_rec        = ig_rec,
            alpha         = alpha,
            random_seeds  = random_seeds,
            verbose       = verbose,
        )

    _dd.clear_empirical_scorer()
    return results_by_alpha


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_knn_results(results: KNNEvalResults) -> None:
    alpha = results.alpha
    print(f"\n{'═'*90}")
    print(f"  KNN SPEED / ACCURACY / COST / BURDEN   α={alpha:.2f}   n={results.n_patients} patients")
    print(f"  (τ = entropy stopping threshold in bits; cost & burden are additive sums)")
    print(f"{'═'*90}")
    print(
        f"  {'τ':>5}  {'Strategy':<14}  {'Accuracy':>8}  {'Tests':>6}  "
        f"{'Cost':>8}  {'Burden':>8}  {'Early%':>7}"
    )
    print(f"  {'─'*76}")

    for tau in results.tau_values:
        for strategy in KNN_STRATEGIES:
            st_list = [s for s in results.stats[strategy] if abs(s.tau - tau) < 1e-9]
            if not st_list:
                continue
            st = st_list[0]
            if strategy == "ig":
                label = f"ig(α={alpha:.2f})"
            elif strategy == "actual_real":
                label = "actual(real)"
            else:
                label = strategy
            print(
                f"  {tau:>5.2f}  {label:<14}  "
                f"{st.accuracy:>8.1%}  "
                f"{st.mean_stop_tests:>6.2f}  "
                f"{st.mean_cost:>8.2f}  "
                f"{st.mean_burden:>8.2f}  "
                f"{st.pct_early_stop:>6.1%}"
            )
        print()

    print(f"{'═'*90}\n")


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
                    "mean_cost":       st.mean_cost,
                    "mean_burden":     st.mean_burden,
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
    "actual":      dict(color="#90CAF9", marker="o", label="Real clinical order (KNN est.)", lw=1.8, zorder=4, ls="--"),
    "actual_real": dict(color="#1565C0", marker="o", label="Real clinical order (actual)",   lw=2.2, zorder=6),
    "random":      dict(color="#9E9E9E", marker="s", label="Random order",                   lw=1.5, zorder=2, ls="--"),
    "bfs":         dict(color="#F44336", marker="^", label="Rubric-guided order",            lw=2.0, zorder=3),
    "ig":          dict(color="#4CAF50", marker="D", label="Efficiency-optimized order",     lw=2.5, zorder=5),
}

_X_LABEL = "Maximum tolerated uncertainty τ (bits)"

# Metric definitions: (attr_name, y-axis label, is_pct)
# Used to build fname from a stem prefix passed per call-site.
_METRIC_DEFS: list[tuple[str, str, bool]] = [
    ("accuracy",        "Diagnostic accuracy",                    True),
    ("mean_stop_tests", "Mean tests done at stop",                False),
    ("mean_cost",       "Mean cumulative cost (1–10 per test)",   False),
    ("mean_burden",     "Mean cumulative burden (1–10 per test)", False),
]


def _find_best_accuracy(
    tau_arr:           list[float],
    acc_arr:           list[float],
    random_acc_by_tau: dict[float, float] | None = None,
) -> tuple[float, float] | None:
    if not acc_arr:
        return None
    if random_acc_by_tau:
        eligible = [
            i for i, (tau, acc) in enumerate(zip(tau_arr, acc_arr))
            if acc > random_acc_by_tau.get(tau, -1.0)
        ]
    else:
        eligible = list(range(len(acc_arr)))
    if not eligible:
        return None
    best_idx = max(eligible, key=lambda i: acc_arr[i])
    return tau_arr[best_idx], acc_arr[best_idx]


def _draw_peak_markers(ax, peaks: list[tuple[float, float, str]]) -> None:
    for best_tau, best_acc, color in peaks:
        ax.plot(best_tau, best_acc, marker="*", markersize=14, color=color, zorder=20, linestyle="none")
        ax.annotate(
            f"({best_tau:.2f}, {best_acc:.1%})",
            xy=(best_tau, best_acc),
            xytext=(4, 6), textcoords="offset points",
            fontsize=9, color=color, fontweight="bold",
            ha="left", va="bottom", zorder=21,
        )


def _save_strategy_figs(
    results:         KNNEvalResults,
    alpha:           float,
    strategy_subset: list[str],
    out_dir:         Path,
    fname_stems:     dict[str, str],   # {metric_attr: filename_without_extension}
    group_title:     str,
    title_suffix:    str,
    plt,
    mticker,
) -> None:
    """Save one figure per metric for the given strategy subset.

    fname_stems maps metric attribute name → output filename (no .png).
    E.g. {"accuracy": "knn_accuracy_alpha_0.50", "mean_cost": "knn_cost_alpha_0.50", ...}
    """
    n          = results.n_patients
    base_title = (
        f"Strategy Evaluation — {n} patients  |  α = {alpha:.2f}  |  {group_title}"
        + (f"\n{title_suffix}" if title_suffix else "")
    )

    _rnd = sorted(results.stats.get("random", []), key=lambda s: s.tau)
    random_acc_by_tau: dict[float, float] = {s.tau: s.accuracy for s in _rnd}

    for metric, ylabel, as_pct in _METRIC_DEFS:
        fname = fname_stems.get(metric)
        if not fname:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        peaks: list[tuple[float, float, str]] = []

        for strategy in strategy_subset:
            style = dict(_STRATEGY_STYLE[strategy])
            if strategy == "ig":
                style["label"] = f"Efficiency-optimized order (α={alpha:.2f})"
            st_list = sorted(results.stats[strategy], key=lambda s: s.tau)
            if not st_list:
                continue
            tau_arr = [s.tau              for s in st_list]
            val_arr = [getattr(s, metric) for s in st_list]
            ax.plot(tau_arr, val_arr, **style)
            if metric == "accuracy" and strategy != "random":
                pt = _find_best_accuracy(tau_arr, val_arr, random_acc_by_tau)
                if pt:
                    peaks.append((*pt, style["color"]))

        _draw_peak_markers(ax, peaks)
        ax.set_title(base_title, fontsize=10, fontweight="bold")
        ax.set_xlabel(_X_LABEL, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.invert_xaxis()
        if as_pct:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fpath = out_dir / f"{fname}.png"
        fig.savefig(str(fpath), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fpath}")


def plot_knn_results(
    results_by_alpha: dict[float, KNNEvalResults],
    out_dir: str,
    title_suffix: str = "",
) -> None:
    """For each alpha generate 8 PNGs: 4 metrics × (all + deployable-only)."""
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

    deployable = [s for s in KNN_STRATEGIES if s != "actual_real"]

    for alpha, results in sorted(results_by_alpha.items()):
        a = f"{alpha:.2f}"

        all_stems = {
            "accuracy":        f"knn_accuracy_alpha_{a}",
            "mean_stop_tests": f"knn_tests_alpha_{a}",
            "mean_cost":       f"knn_cost_alpha_{a}",
            "mean_burden":     f"knn_burden_alpha_{a}",
        }
        dep_stems = {k: v + "_deployable" for k, v in all_stems.items()}

        _save_strategy_figs(
            results, alpha,
            strategy_subset = list(KNN_STRATEGIES),
            out_dir         = out,
            fname_stems     = all_stems,
            group_title     = "All strategies  (real clinical order = reference ceiling)",
            title_suffix    = title_suffix,
            plt=plt, mticker=mticker,
        )
        _save_strategy_figs(
            results, alpha,
            strategy_subset = deployable,
            out_dir         = out,
            fname_stems     = dep_stems,
            group_title     = "Deployable strategies only",
            title_suffix    = title_suffix,
            plt=plt, mticker=mticker,
        )


def plot_knn_ig_comparison(
    results_by_alpha: dict[float, KNNEvalResults],
    out_dir: str,
) -> None:
    """Compare all α values for the efficiency-optimized curve + fixed baselines.

    Generates 8 PNGs: 4 metrics × (all + deployable-only).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sorted_alphas = sorted(results_by_alpha)
    cmap  = matplotlib.colormaps.get_cmap("RdYlGn").resampled(len(sorted_alphas))
    first = results_by_alpha[sorted_alphas[0]]
    n     = first.n_patients

    base_title = (
        f"Strategy Evaluation: efficiency-optimized order — α comparison\n"
        f"{n} test patients  |  1-step counterfactual"
    )

    all_baselines        = ("actual", "actual_real", "random", "bfs")
    deployable_baselines = ("actual", "random", "bfs")

    _rnd_first = sorted(first.stats.get("random", []), key=lambda s: s.tau)
    random_acc_by_tau_cmp: dict[float, float] = {s.tau: s.accuracy for s in _rnd_first}

    for tag, baselines, extra_title in [
        ("",            all_baselines,        "All strategies  (real clinical order = reference ceiling)"),
        ("_deployable", deployable_baselines, "Deployable strategies only"),
    ]:
        for metric, ylabel, as_pct in _METRIC_DEFS:
            fname_map = {
                "accuracy":        "knn_accuracy_alpha_comparison",
                "mean_stop_tests": "knn_tests_alpha_comparison",
                "mean_cost":       "knn_cost_alpha_comparison",
                "mean_burden":     "knn_burden_alpha_comparison",
            }
            fname_stem = fname_map[metric] + tag

            fig, ax = plt.subplots(figsize=(11, 6))
            peaks: list[tuple[float, float, str]] = []

            for strategy in baselines:
                style   = dict(_STRATEGY_STYLE[strategy])
                st_list = sorted(first.stats[strategy], key=lambda s: s.tau)
                if not st_list:
                    continue
                tau_arr = [s.tau              for s in st_list]
                val_arr = [getattr(s, metric) for s in st_list]
                ax.plot(tau_arr, val_arr, **style)
                if metric == "accuracy" and strategy != "random":
                    pt = _find_best_accuracy(tau_arr, val_arr, random_acc_by_tau_cmp)
                    if pt:
                        peaks.append((*pt, style["color"]))

            for i, alpha in enumerate(sorted_alphas):
                color   = cmap(i)
                res     = results_by_alpha[alpha]
                st_list = sorted(res.stats["ig"], key=lambda s: s.tau)
                if not st_list:
                    continue
                tau_arr = [s.tau              for s in st_list]
                val_arr = [getattr(s, metric) for s in st_list]
                lw = 2.0 + 0.4 * i
                ax.plot(tau_arr, val_arr, color=color, marker="D", lw=lw,
                        label=f"Efficiency-optimized order (α={alpha:.2f})", zorder=5)
                if metric == "accuracy":
                    pt = _find_best_accuracy(tau_arr, val_arr, random_acc_by_tau_cmp)
                    if pt:
                        peaks.append((*pt, color))

            _draw_peak_markers(ax, peaks)
            ax.set_title(base_title + f"\n{extra_title}", fontsize=11, fontweight="bold")
            ax.set_xlabel(_X_LABEL, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.invert_xaxis()
            if as_pct:
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            fpath = out / f"{fname_stem}.png"
            fig.savefig(str(fpath), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {fpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KNN counterfactual speed / accuracy / cost / burden evaluation."
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
    p.add_argument("--plot-dir",     type=str,   default="results/figs/cb")
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
        print(f"\nCost lookup  : { {k: int(v) for k, v in TEST_COST.items()} }")
        print(f"Burden lookup: { {k: int(v) for k, v in TEST_BURDEN.items()} }\n")

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
