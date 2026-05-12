"""feature_simulator.py — KNN-based feature vector simulation

Simulates the post-test feature state by computing an inverse-distance-weighted
average of feature *deltas* (Δ = vec_after − vec_before) from similar training
transitions that performed the same test.

    simulated_state ≈ encode(query) + Σ_i  w_i × Δ_i  /  Σ_i w_i

Why deltas instead of raw averages of vec_after
------------------------------------------------
Averaging vec_after directly would overwrite the patient's already-known
findings with neighbour averages.  Using deltas preserves the current state
and only injects the *change* signal from neighbours, so:
  • Features already True in query stay True (monotonicity is natural).
  • Tests done in prior steps are not erased.
  • Only findings that the new test plausibly reveals are updated.

Advantage over EntropyReducer
------------------------------
EntropyReducer predicts (ΔH, diag_dist_after) — useful for a single decision
but those outputs are not feature vectors and cannot be chained.
FeatureSimulator returns a complete feature dict, which can be fed back in as
the query for the next step, enabling multi-step counterfactual trajectories:

    features_0 → simulate(test_1) → features_1
               → simulate(test_2) → features_2  → ...

Usage
-----
    simulator = FeatureSimulator(k=15)
    simulator.fit(all_patients)

    # 1-step simulation
    features_1 = simulator.simulate_features(current_features, "CT_Abdomen")

    # Multi-step trajectory (returns list of simulated dicts, one per test)
    trajectory = simulator.simulate_trajectory(
        initial_features,
        test_sequence=["Lab_Panel", "Ultrasound_Abdomen", "CT_Abdomen"],
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from empirical_scorer import encode_features, decode_features
from feature_schema import VALID_TESTS


# ---------------------------------------------------------------------------
# Corpus entry
# ---------------------------------------------------------------------------

@dataclass
class _FeatureTransitionEntry:
    """One consecutive-step transition stored in the FeatureSimulator corpus."""
    vec_before: np.ndarray   # encode(features_before), shape (dim,)
    test_key:   str          # test performed at this transition
    delta_vec:  np.ndarray   # encode(features_after) − encode(features_before)
    disease:    str          # ground-truth disease label


# ---------------------------------------------------------------------------
# FeatureSimulator
# ---------------------------------------------------------------------------

class FeatureSimulator:
    """
    KNN-based feature state simulator.

    Parameters
    ----------
    k         : number of nearest neighbours to consider per query
    min_count : minimum number of neighbours that performed the queried test;
                if fewer, simulate_features returns None
    """

    def __init__(self, k: int = 15, min_count: int = 1) -> None:
        self.k = k
        self.min_count = min_count
        self._corpus: list[_FeatureTransitionEntry] = []
        self._corpus_matrix: Optional[np.ndarray] = None   # shape (N, dim)

    # ── fitting ──────────────────────────────────────────────────────────────

    def fit(
        self,
        all_patients: dict[str, dict[str, list[dict]]],
    ) -> "FeatureSimulator":
        """
        Build the transition corpus from multi-disease patient data.

        Parameters
        ----------
        all_patients : {disease: {patient_id: [step_dict, ...]}}
            Each step_dict must have "features" and optionally "test_key".
            step_dict["test_key"] is None (or absent) for the first (HPI/PE) step.
        """
        entries: list[_FeatureTransitionEntry] = []

        for disease, patients in all_patients.items():
            for pid, steps in patients.items():
                for i in range(len(steps) - 1):
                    test_key = steps[i + 1].get("test_key")
                    if not test_key:
                        continue

                    vec_before = encode_features(steps[i]["features"])
                    vec_after  = encode_features(steps[i + 1]["features"])

                    entries.append(_FeatureTransitionEntry(
                        vec_before = vec_before,
                        test_key   = test_key,
                        delta_vec  = vec_after - vec_before,
                        disease    = disease,
                    ))

        self._corpus = entries
        if entries:
            self._corpus_matrix = np.stack(
                [e.vec_before for e in entries], axis=0
            )  # (N, dim)
        return self

    # ── 1-step simulation ────────────────────────────────────────────────────

    def simulate_features(
        self,
        query_features: dict,
        test_key: str,
    ) -> Optional[dict]:
        """
        Simulate the feature state after performing test_key from query_features.

        Algorithm
        ---------
        1. Encode query_features → q.
        2. Find K nearest corpus entries by L2(q, vec_before).
        3. Among K neighbours, keep those where entry.test_key == test_key.
        4. Compute inverse-distance-weighted mean delta:
               Δ̄ = Σ w_i Δ_i  /  Σ w_i
        5. Decode  q + Δ̄  back to a feature dict.
        6. Enforce monotonicity (booleans True in query stay True) and
           set tests_done = query["tests_done"] ∪ {test_key}.

        Returns None if fewer than min_count neighbours performed test_key.
        """
        if self._corpus_matrix is None or not self._corpus:
            return None

        q     = encode_features(query_features)
        dists = np.linalg.norm(self._corpus_matrix - q, axis=1)

        k      = min(self.k, len(self._corpus))
        nn_idx = np.argpartition(dists, k - 1)[:k]

        neighbors      = [self._corpus[i] for i in nn_idx]
        neighbor_dists = dists[nn_idx]
        # Inverse-distance weights; +ε avoids division by zero for exact matches.
        neighbor_weights = 1.0 / (neighbor_dists + 1e-6)

        # Keep only neighbours that performed test_key.
        rel_nb = [nb for nb in neighbors           if nb.test_key == test_key]
        rel_w  = [w  for nb, w in zip(neighbors, neighbor_weights)
                  if nb.test_key == test_key]

        if len(rel_nb) < self.min_count:
            return None

        w_sum     = sum(rel_w)
        avg_delta = sum(w * nb.delta_vec for w, nb in zip(rel_w, rel_nb)) / w_sum

        # Apply delta to query vector and decode.
        # base_features enforces monotonicity for booleans.
        simulated = decode_features(q + avg_delta, base_features=query_features)

        # Override tests_done: guarantee monotonic growth.
        prior_done = list(query_features.get("tests_done", []))
        if test_key not in prior_done:
            simulated["tests_done"] = prior_done + [test_key]
        else:
            simulated["tests_done"] = prior_done

        return simulated

    # ── multi-step simulation ────────────────────────────────────────────────

    def simulate_trajectory(
        self,
        initial_features: dict,
        test_sequence: list[str],
    ) -> list[dict]:
        """
        Chain simulate_features across a sequence of tests.

        Returns a list of simulated feature dicts, one per test.  The list
        is shorter than test_sequence if simulation fails at any step
        (no neighbours found for the requested test at the current state).

        Example
        -------
        trajectory = simulator.simulate_trajectory(
            initial_features,
            ["Lab_Panel", "Ultrasound_Abdomen", "CT_Abdomen"],
        )
        # trajectory[0] = simulated features after Lab_Panel
        # trajectory[1] = simulated features after Ultrasound_Abdomen
        # trajectory[2] = simulated features after CT_Abdomen
        """
        trajectory: list[dict] = []
        current = initial_features

        for test in test_sequence:
            next_features = self.simulate_features(current, test)
            if next_features is None:
                break
            trajectory.append(next_features)
            current = next_features

        return trajectory

    # ── diagnostics ──────────────────────────────────────────────────────────

    @property
    def n_transitions(self) -> int:
        """Total number of step transitions in the corpus."""
        return len(self._corpus)

    def corpus_summary(self) -> dict:
        """Return basic statistics about the training corpus."""
        if not self._corpus:
            return {"n_transitions": 0}

        by_test: dict[str, int] = {}
        for e in self._corpus:
            by_test[e.test_key] = by_test.get(e.test_key, 0) + 1

        return {
            "n_transitions": len(self._corpus),
            "per_test_count": dict(sorted(by_test.items())),
        }
