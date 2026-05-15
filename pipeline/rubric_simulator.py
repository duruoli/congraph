"""rubric_simulator.py — Rubric-guided test sequence simulation per patient.

Algorithm
---------
For each patient starting from step-0 features (HPI + PE, no tests done yet):

  Phase 1 — Triage routing
    Run the main triage graph.  If Lab_Panel is not yet done the routing edges
    are all gated, so the triage graph will expose "Lab_Panel" as a pending test.
    Simulate Lab_Panel via FeatureSimulator, then re-run triage.  Repeat until
    a disease sub-rubric is selected or the graph can no longer advance.

    When multiple routing conditions fire simultaneously (e.g. RUQ pain +
    elevated lipase activates both cholecystitis and pancreatitis routes),
    break the tie by running a quick sub-rubric traversal on all candidates
    and picking the one with the highest conditional_triggers score.

  Phase 2 — Sub-rubric loop
    Traverse the selected sub-rubric graph with the current feature dict.

      • If a terminal node is reached → stop (termination_reason = terminal_type).
      • If no pending tests remain → stop ("no_pending_tests").
      • Otherwise take the first pending test, simulate it via FeatureSimulator,
        update features, and repeat.

    KNN simulation note: booleans are discretised by thresholding at 0.5 inside
    decode_features (empirical_scorer.py), so the simulated feature dict fed
    back into the rubric graph is always a valid binary feature state.

Safety cap: max_steps (default 10) limits the total number of simulated tests
regardless of phase.

Output per patient
------------------
  RubricSimResult
    test_sequence         : ordered list of test keys recommended and simulated
    intermediate_features : simulated feature dict after each test (one per test)
    selected_subrubric    : disease sub-rubric chosen by triage, or None
    terminal_node         : rubric node id where simulation ended, or None
    termination_reason    : one of
        "terminal_confirmed"  — leaf node confirms the disease
        "terminal_excluded"   — leaf node excludes the disease
        "terminal_low_risk"   — leaf node flags low-risk (not definitive exclusion)
        "max_steps"           — safety cap exhausted
        "no_pending_tests"    — sub-rubric graph blocked, no test can advance it
        "simulation_failed"   — FeatureSimulator returned None for all pending tests
        "no_subrubric"        — triage completed but no disease route was activated
                                (extended differential or all routing conditions failed)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from pipeline.rubric_graph import DISEASE_GRAPHS, TRIAGE_GRAPH
from pipeline.traversal_engine import traverse_graph, TraversalResult
from knn.feature_simulator import FeatureSimulator


# ---------------------------------------------------------------------------
# Triage routing map  (mirrors traversal_engine._ROUTE_TO_DISEASE)
# ---------------------------------------------------------------------------

_TRIAGE_ROUTE_MAP: dict[str, str] = {
    "ROUTE_APPENDICITIS":   "appendicitis",
    "ROUTE_CHOLECYSTITIS":  "cholecystitis",
    "ROUTE_DIVERTICULITIS": "diverticulitis",
    "ROUTE_PANCREATITIS":   "pancreatitis",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RubricSimResult:
    """Rubric-simulated test sequence result for one patient."""

    test_sequence:          list[str]         # tests simulated, in order
    intermediate_features:  list[dict]        # feature dict after each test
    selected_subrubric:     Optional[str]     # sub-rubric chosen by triage
    terminal_node:          Optional[str]     # rubric node id at termination
    termination_reason:     str               # see module docstring


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _activated_diseases(triage_result: TraversalResult) -> list[str]:
    """Return diseases for all triage routing nodes that became frontier_leaf."""
    return [
        disease
        for route_nid, disease in _TRIAGE_ROUTE_MAP.items()
        if (
            triage_result.node_statuses.get(route_nid) is not None
            and triage_result.node_statuses[route_nid].status == "frontier_leaf"
        )
    ]


def _pick_subrubric(activated: list[str], current_features: dict) -> str:
    """
    Pick one sub-rubric from the activated list.

    Single activation → return immediately.
    Multiple activations → traverse each sub-rubric and return the one with
    the highest conditional_triggers; ties broken alphabetically for
    determinism.
    """
    if len(activated) == 1:
        return activated[0]

    scores = {
        disease: traverse_graph(
            DISEASE_GRAPHS[disease], current_features
        ).conditional_triggers
        for disease in activated
    }
    return max(activated, key=lambda d: (scores[d], d))


def _try_simulate(
    current: dict,
    pending_tests: list[str],
    simulator: FeatureSimulator,
) -> tuple[Optional[str], Optional[dict]]:
    """
    Try each test in pending_tests in order.  Return (test_key, new_features)
    for the first one that the simulator can handle; (None, None) if all fail.
    """
    for test in pending_tests:
        new_features = simulator.simulate_features(current, test)
        if new_features is not None:
            return test, new_features
    return None, None


# ---------------------------------------------------------------------------
# Core simulation function
# ---------------------------------------------------------------------------

def simulate_patient_rubric(
    initial_features: dict,
    simulator: FeatureSimulator,
    max_steps: int = 10,
    force_subrubric: Optional[str] = None,
) -> RubricSimResult:
    """
    Generate a rubric-simulated test sequence for one patient.

    Parameters
    ----------
    initial_features : feature dict at step 0 (HPI + PE; typically no tests done)
    simulator        : fitted FeatureSimulator instance
    max_steps        : maximum total tests to simulate (safety cap)
    force_subrubric  : if given, skip triage entirely and start directly in this
                       sub-rubric (oracle / ground-truth mode).  Must be one of
                       "appendicitis", "cholecystitis", "diverticulitis",
                       "pancreatitis".

    Returns
    -------
    RubricSimResult
    """
    current                = initial_features
    test_sequence:  list[str]  = []
    inter_features: list[dict] = []

    # Oracle mode: bypass triage, lock sub-rubric immediately.
    selected_subrubric: Optional[str] = force_subrubric

    for _step in range(max_steps):

        # ── Phase 1: Determine sub-rubric via triage ──────────────────────────
        if selected_subrubric is None:
            triage_result = traverse_graph(TRIAGE_GRAPH, current)
            activated     = _activated_diseases(triage_result)

            if activated:
                # Lock in the sub-rubric (possibly breaking a tie).
                # No `continue` — fall through to Phase 2 in this same iteration.
                selected_subrubric = _pick_subrubric(activated, current)

            else:
                # Triage routing has not fired yet.
                pending = triage_result.pending_tests
                if not pending:
                    # All required tests done but no disease route activated
                    # → atypical / extended differential presentation.
                    return RubricSimResult(
                        test_sequence         = test_sequence,
                        intermediate_features = inter_features,
                        selected_subrubric    = None,
                        terminal_node         = None,
                        termination_reason    = "no_subrubric",
                    )

                # Simulate the first simulatable triage pending test.
                next_test, new_features = _try_simulate(current, pending, simulator)
                if new_features is None:
                    return RubricSimResult(
                        test_sequence         = test_sequence,
                        intermediate_features = inter_features,
                        selected_subrubric    = None,
                        terminal_node         = None,
                        termination_reason    = "simulation_failed",
                    )

                test_sequence.append(next_test)
                inter_features.append(new_features)
                current = new_features
                continue   # re-evaluate triage with updated features

        # ── Phase 2: Traverse sub-rubric and recommend next test ──────────────
        sub_result = traverse_graph(DISEASE_GRAPHS[selected_subrubric], current)

        # Terminal node reached → done
        if sub_result.terminal_node is not None:
            return RubricSimResult(
                test_sequence         = test_sequence,
                intermediate_features = inter_features,
                selected_subrubric    = selected_subrubric,
                terminal_node         = sub_result.terminal_node,
                termination_reason    = sub_result.terminal_type,
            )

        # No pending tests → graph cannot advance further
        if not sub_result.pending_tests:
            return RubricSimResult(
                test_sequence         = test_sequence,
                intermediate_features = inter_features,
                selected_subrubric    = selected_subrubric,
                terminal_node         = None,
                termination_reason    = "no_pending_tests",
            )

        # Simulate the first simulatable pending test.
        next_test, new_features = _try_simulate(
            current, sub_result.pending_tests, simulator
        )
        if new_features is None:
            return RubricSimResult(
                test_sequence         = test_sequence,
                intermediate_features = inter_features,
                selected_subrubric    = selected_subrubric,
                terminal_node         = None,
                termination_reason    = "simulation_failed",
            )

        test_sequence.append(next_test)
        inter_features.append(new_features)
        current = new_features
        # Loop continues: Phase 2 re-traverses sub-rubric with updated features.

    # Safety cap exhausted
    return RubricSimResult(
        test_sequence         = test_sequence,
        intermediate_features = inter_features,
        selected_subrubric    = selected_subrubric,
        terminal_node         = None,
        termination_reason    = "max_steps",
    )


# ---------------------------------------------------------------------------
# Cohort runner
# ---------------------------------------------------------------------------

def simulate_cohort_rubric(
    all_patients:  dict[str, dict[str, list[dict]]],
    simulator:     FeatureSimulator,
    max_steps:     int  = 10,
    oracle_routing: bool = False,
    verbose:       bool = False,
) -> dict[str, dict[str, RubricSimResult]]:
    """
    Run rubric simulation for every patient in the cohort.

    Parameters
    ----------
    all_patients   : {disease: {patient_id: [step_dict, ...]}}
                     Uses only step 0 ("features" key) for each patient.
    simulator      : fitted FeatureSimulator instance
    max_steps      : per-patient step cap
    oracle_routing : if True, skip triage and force the ground-truth sub-rubric
                     for each patient.  Useful as a ceiling / ablation mode.
    verbose        : print progress every 100 patients

    Returns
    -------
    {disease: {patient_id: RubricSimResult}}
    """
    results: dict[str, dict[str, RubricSimResult]] = {}
    total = sum(len(pats) for pats in all_patients.values())
    done  = 0

    for disease, patients in all_patients.items():
        results[disease] = {}
        for pid, steps in patients.items():
            if not steps:
                continue
            initial_features = steps[0]["features"]
            results[disease][pid] = simulate_patient_rubric(
                initial_features = initial_features,
                simulator        = simulator,
                max_steps        = max_steps,
                force_subrubric  = disease if oracle_routing else None,
            )
            done += 1
            if verbose and done % 100 == 0:
                print(f"  {done}/{total} patients simulated…", flush=True)

    if verbose:
        print(f"  {done}/{total} patients simulated — done.", flush=True)

    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_rubric_simulations(
    sim_results: dict[str, dict[str, RubricSimResult]],
) -> dict:
    """
    Aggregate statistics over all simulated patients.

    Returns
    -------
    dict with keys:
        n_patients               : total patients simulated
        n_correct_subrubric      : patients routed to their ground-truth sub-rubric
        pct_correct_subrubric    : fraction correctly routed
        termination_counts       : {reason: count}
        mean_test_sequence_length: average number of tests in simulation
        subrubric_routing        : {disease: {selected_subrubric: count}}
    """
    n_patients = 0
    n_correct  = 0
    term_counts: Counter = Counter()
    seq_lengths: list[int] = []
    routing: dict[str, Counter] = {}

    for disease, patients in sim_results.items():
        routing[disease] = Counter()
        for _pid, res in patients.items():
            n_patients += 1
            if res.selected_subrubric == disease:
                n_correct += 1
            term_counts[res.termination_reason] += 1
            seq_lengths.append(len(res.test_sequence))
            routing[disease][res.selected_subrubric or "(none)"] += 1

    return {
        "n_patients":                n_patients,
        "n_correct_subrubric":       n_correct,
        "pct_correct_subrubric":     n_correct / n_patients if n_patients else 0.0,
        "termination_counts":        dict(term_counts),
        "mean_test_sequence_length": (
            sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0.0
        ),
        "subrubric_routing":         {d: dict(c) for d, c in routing.items()},
    }


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def print_rubric_sim_summary(
    sim_results: dict[str, dict[str, RubricSimResult]],
) -> None:
    """Print a compact summary of cohort rubric simulation results."""
    stats = summarize_rubric_simulations(sim_results)
    n  = stats["n_patients"]
    W  = 70

    print(f"\n{'═'*W}")
    print(f"  RUBRIC SIMULATION SUMMARY   ({n} patients)")
    print(f"{'═'*W}")
    print(
        f"  Sub-rubric routing accuracy : "
        f"{stats['n_correct_subrubric']}/{n} "
        f"({stats['pct_correct_subrubric']:.1%})"
    )
    print(
        f"  Mean test sequence length   : "
        f"{stats['mean_test_sequence_length']:.2f}"
    )

    print(f"\n  Termination reasons:")
    for reason, count in sorted(
        stats["termination_counts"].items(), key=lambda x: -x[1]
    ):
        print(f"    {reason:<28}  {count:>6}  ({count/n:.1%})")

    print(f"\n  Sub-rubric routing breakdown:")
    diseases = sorted(stats["subrubric_routing"])
    for disease in diseases:
        row = stats["subrubric_routing"][disease]
        total_d = sum(row.values())
        print(f"    {disease} (n={total_d}):")
        for routed, cnt in sorted(row.items(), key=lambda x: -x[1]):
            marker = " ✓" if routed == disease else ""
            print(f"      → {routed:<20}  {cnt:>5}  ({cnt/total_d:.1%}){marker}")

    print(f"{'═'*W}\n")
