"""evaluation_metrics.py — Shared per-patient and per-strategy evaluation metrics.

Single source of truth for the 4 metrics used across all KNN evaluators:
  accuracy, n_tests, cost (USD, absolute), burden (1–10 relative scale)

Two dataclasses + three helpers; no pipeline / KNN imports, so this module
is dependency-free aside from numpy + the static cost/burden lookup.

cost   — absolute USD (CMS 2025 midpoints). Imaging uses static TEST_COST.
          Lab_Panel uses ``lab_panel_cost_usd(features['lab_itemids'])`` when
          MIMIC itemids are present; otherwise the bundle midpoint.
burden — relative 1–10 scale grounded in published radiology / nuclear
          medicine literature (see test_burden_cost.py for per-test citations).

Used by knn_feature_eval.py (multi-step) and knn_onestep_eval.py (1-step).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluation.test_burden_cost import TEST_BURDEN, TEST_COST, lab_panel_cost_usd


# ---------------------------------------------------------------------------
# Per-patient stop result
# ---------------------------------------------------------------------------

@dataclass
class PatientStopResult:
    """Per-patient stopping outcome under one strategy / one τ threshold."""
    stop_tests:   int    # tests done at the stopping step
    correct:      bool   # primary dx at stop == ground truth
    ever_stopped: bool   # False if no step met the τ condition (used full traj)
    cost:         float  # cumulative USD cost of tests_done (CMS 2025 absolute)
    burden:       float  # cumulative patient burden score (1–10 scale, summed)


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

def compute_cost_burden(
    tests_done: list[str],
    *,
    lab_itemids: list[str] | None = None,
) -> tuple[float, float]:
    """Cumulative (cost, burden). Lab_Panel cost is patient-specific when lab_itemids is set."""
    cost = 0.0
    for t in tests_done:
        if t == "Lab_Panel":
            cost += lab_panel_cost_usd(lab_itemids or [])
        else:
            cost += TEST_COST.get(t, 0.0)
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
    lab_ids      = stop_features.get("lab_itemids")
    lab_list     = lab_ids if isinstance(lab_ids, list) else None
    cost, burden = compute_cost_burden(tests_done, lab_itemids=lab_list)
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
