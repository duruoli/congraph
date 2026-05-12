"""evaluation_metrics.py — Shared per-patient and per-strategy evaluation metrics.

Single source of truth for the 4 metrics used across all KNN evaluators:
  accuracy, n_tests, cost, burden

Two dataclasses + three helpers; no pipeline / KNN imports, so this module
is dependency-free aside from numpy + the static cost/burden lookup.

Used by knn_feature_eval.py (multi-step) and knn_cost_burden_eval.py (1-step).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from test_burden_cost import TEST_COST, TEST_BURDEN


# ---------------------------------------------------------------------------
# Per-patient stop result
# ---------------------------------------------------------------------------

@dataclass
class PatientStopResult:
    """Per-patient stopping outcome under one strategy / one τ threshold."""
    stop_tests:   int    # tests done at the stopping step
    correct:      bool   # primary dx at stop == ground truth
    ever_stopped: bool   # False if no step met the τ condition (used full traj)
    cost:         float  # sum of TEST_COST[t] over tests_done at stopping step
    burden:       float  # sum of TEST_BURDEN[t] over tests_done at stopping step


# ---------------------------------------------------------------------------
# Per-strategy aggregated stats
# ---------------------------------------------------------------------------

@dataclass
class StrategyStats:
    """Aggregated metrics for one strategy at one τ across all patients."""
    tau:             float
    strategy:        str
    n_patients:      int
    mean_stop_tests: float
    accuracy:        float
    pct_early_stop:  float
    mean_cost:       float
    mean_burden:     float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_cost_burden(tests_done: list[str]) -> tuple[float, float]:
    """Cumulative (cost, burden) for a list of completed tests."""
    cost   = sum(TEST_COST.get(t,   0.0) for t in tests_done)
    burden = sum(TEST_BURDEN.get(t, 0.0) for t in tests_done)
    return cost, burden


def compute_patient_metrics(
    stop_features:     dict,
    predicted_disease: str,
    gt_disease:        str,
    ever_stopped:      bool = True,
) -> PatientStopResult:
    """
    Build a PatientStopResult from one patient's stopping state.

    Parameters
    ----------
    stop_features     : feature dict at the stopping step (uses tests_done)
    predicted_disease : argmax of the diagnosis distribution at stop
    gt_disease        : ground-truth disease label
    ever_stopped      : True iff τ was met at some step, False if trajectory
                        was exhausted without ever reaching the threshold
    """
    tests_done   = stop_features.get("tests_done", [])
    cost, burden = compute_cost_burden(tests_done)
    return PatientStopResult(
        stop_tests   = len(tests_done),
        correct      = predicted_disease == gt_disease,
        ever_stopped = ever_stopped,
        cost         = cost,
        burden       = burden,
    )


def aggregate_stats(
    results:  list[PatientStopResult],
    tau:      float,
    strategy: str,
) -> StrategyStats:
    """Mean-aggregate per-patient results into a StrategyStats row."""
    return StrategyStats(
        tau             = tau,
        strategy        = strategy,
        n_patients      = len(results),
        mean_stop_tests = float(np.mean([r.stop_tests   for r in results])),
        accuracy        = float(np.mean([r.correct      for r in results])),
        pct_early_stop  = float(np.mean([r.ever_stopped for r in results])),
        mean_cost       = float(np.mean([r.cost         for r in results])),
        mean_burden     = float(np.mean([r.burden       for r in results])),
    )
