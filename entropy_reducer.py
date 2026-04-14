"""entropy_reducer.py — KNN-conditioned empirical entropy reduction

For each candidate test t, estimates:

    EIG(t | current_state) ≈ E[H_before − H_after | test=t, features≈query]

using KNN over a training corpus of consecutive patient step transitions.

Algorithm
---------
fit(all_patients):
    For each consecutive pair (step_i → step_{i+1}) across all training patients:
        1. Run full pipeline on features_before → DiagnosisDistribution → H_before
        2. Run full pipeline on features_after  → DiagnosisDistribution → H_after
        3. Record (encode(features_before), test_key, H_before − H_after, disease)

    The encoded features_before vectors form a searchable corpus.

test_entropy_scores(query_features, candidate_tests):
    1. Find K nearest corpus entries by L2(encode(query_features), vec_before)
    2. For each candidate test t:
       - Filter neighbors where transition test_key == t
       - Return mean entropy reduction (0.0 if no neighbor did test t)

Entropy convention
------------------
Shannon entropy in bits over 4 diseases, H ∈ [0, 2].
  H = 2.0  → completely uniform (maximum uncertainty)
  H = 0.0  → certain (one disease at 100%)
Positive ΔH = H_before − H_after means test reduced uncertainty (desirable).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from empirical_scorer import encode_features
from traversal_engine import run_full_traversal
from diagnosis_distribution import compute_distribution, DiagnosisDistribution


# ---------------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------------

def distribution_entropy(dist: DiagnosisDistribution) -> float:
    """Shannon entropy (bits) of a DiagnosisDistribution over 4 diseases."""
    H = 0.0
    for d in dist.ranked:
        p = dist.prob(d)
        if p > 1e-12:
            H -= p * math.log2(p)
    return H


# ---------------------------------------------------------------------------
# Corpus entry
# ---------------------------------------------------------------------------

@dataclass
class _TransitionEntry:
    """One consecutive-step transition from the training corpus."""
    vec_before: np.ndarray   # encoded features_before, shape (dim,)
    test_key:   str          # test that was done at step i+1
    delta_H:    float        # H_before − H_after  (positive = test helped)
    disease:    str          # ground-truth disease label


# ---------------------------------------------------------------------------
# EntropyReducer
# ---------------------------------------------------------------------------

class EntropyReducer:
    """
    KNN-conditioned empirical entropy reduction estimator.

    Parameters
    ----------
    k          : number of nearest neighbors to consider per query
    min_count  : minimum number of neighbors that did test t to return a
                 score; if fewer, fall back to 0.0 (no estimate)
    """

    def __init__(self, k: int = 15, min_count: int = 1) -> None:
        self.k = k
        self.min_count = min_count
        self._corpus: list[_TransitionEntry] = []
        self._corpus_matrix: Optional[np.ndarray] = None  # shape (N, dim)

    # ── fitting ──────────────────────────────────────────────────────────────

    def fit(
        self,
        all_patients: dict[str, dict[str, list[dict]]],
    ) -> "EntropyReducer":
        """
        Build the transition corpus from multi-disease patient data.

        Parameters
        ----------
        all_patients : {disease: {patient_id: [step_dict, ...]}}
            Each step_dict must have keys "features" and "test_key".
            step_dict["test_key"] is None for the first (HPI/PE) step.
        """
        entries: list[_TransitionEntry] = []

        for disease, patients in all_patients.items():
            for pid, steps in patients.items():
                for i in range(len(steps) - 1):
                    test_key = steps[i + 1].get("test_key")
                    if not test_key:
                        continue  # skip if the next step has no associated test

                    features_before = steps[i]["features"]
                    features_after  = steps[i + 1]["features"]

                    H_before = self._pipeline_entropy(features_before)
                    H_after  = self._pipeline_entropy(features_after)

                    entries.append(_TransitionEntry(
                        vec_before = encode_features(features_before),
                        test_key   = test_key,
                        delta_H    = H_before - H_after,
                        disease    = disease,
                    ))

        self._corpus = entries
        if entries:
            self._corpus_matrix = np.stack(
                [e.vec_before for e in entries], axis=0
            )  # shape (N, dim)
        return self

    # ── query ─────────────────────────────────────────────────────────────────

    def test_entropy_scores(
        self,
        query_features: dict,
        candidate_tests: list[str],
    ) -> dict[str, float]:
        """
        Return expected entropy reduction for each candidate test, given the
        current patient state (query_features).

        Parameters
        ----------
        query_features  : current patient feature dict
        candidate_tests : tests to score (typically the BFS pending_tests list)

        Returns
        -------
        dict[test_name → expected ΔH]
            Positive values mean the test is expected to reduce uncertainty.
            0.0 is returned when no neighbors performed that test.
        """
        if self._corpus_matrix is None or len(self._corpus) == 0:
            return {t: 0.0 for t in candidate_tests}

        q = encode_features(query_features)
        dists = np.linalg.norm(self._corpus_matrix - q, axis=1)

        k = min(self.k, len(self._corpus))
        nn_idx = np.argpartition(dists, k - 1)[:k]
        neighbors = [self._corpus[i] for i in nn_idx]

        scores: dict[str, float] = {}
        for test in candidate_tests:
            relevant = [nb for nb in neighbors if nb.test_key == test]
            if len(relevant) < self.min_count:
                scores[test] = 0.0
            else:
                scores[test] = float(np.mean([nb.delta_H for nb in relevant]))

        return scores

    # ── diagnostics ──────────────────────────────────────────────────────────

    @property
    def n_transitions(self) -> int:
        """Total number of step transitions in the corpus."""
        return len(self._corpus)

    def corpus_summary(self) -> dict[str, object]:
        """Return basic statistics about the training corpus."""
        if not self._corpus:
            return {"n_transitions": 0}

        by_test: dict[str, list[float]] = {}
        for e in self._corpus:
            by_test.setdefault(e.test_key, []).append(e.delta_H)

        return {
            "n_transitions": len(self._corpus),
            "per_test": {
                t: {
                    "count":   len(vals),
                    "mean_dH": float(np.mean(vals)),
                    "std_dH":  float(np.std(vals)),
                }
                for t, vals in sorted(by_test.items())
            },
        }

    # ── internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _pipeline_entropy(features: dict) -> float:
        """Run the traversal + distribution pipeline and return H in bits."""
        traversal = run_full_traversal(features)
        dist      = compute_distribution(traversal, features)
        return distribution_entropy(dist)


# ---------------------------------------------------------------------------
# Quick test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path

    RESULTS_DIR = Path(__file__).parent / "results"
    DISEASE_FILES = {
        "appendicitis":   RESULTS_DIR / "appendicitis_features.json",
        "cholecystitis":  RESULTS_DIR / "cholecystitis_features.json",
        "diverticulitis": RESULTS_DIR / "diverticulitis_features.json",
        "pancreatitis":   RESULTS_DIR / "pancreatitis_features.json",
    }

    print("Loading patient data…")
    all_patients: dict[str, dict[str, list[dict]]] = {}
    for disease, path in DISEASE_FILES.items():
        with open(path, encoding="utf-8") as f:
            all_patients[disease] = json.load(f)["results"]

    print("Fitting EntropyReducer…")
    reducer = EntropyReducer(k=15)
    reducer.fit(all_patients)

    summary = reducer.corpus_summary()
    print(f"\nCorpus: {summary['n_transitions']} transitions")
    print("\nPer-test global entropy reduction (unconditional mean):")
    print(f"  {'Test':<28s}  {'count':>6}  {'mean ΔH':>8}  {'std ΔH':>7}")
    print(f"  {'─'*60}")
    for test, stats in summary["per_test"].items():
        print(
            f"  {test:<28s}  {stats['count']:>6}  "
            f"{stats['mean_dH']:>8.4f}  {stats['std_dH']:>7.4f}"
        )

    # Demo: score tests for a sample patient's first step
    first_disease = next(iter(all_patients))
    first_pid     = next(iter(all_patients[first_disease]))
    first_step    = all_patients[first_disease][first_pid][0]
    features      = first_step["features"]

    from feature_schema import VALID_TESTS
    print(f"\nKNN-conditioned scores for patient {first_pid} ({first_disease}) step 0:")
    knn_scores = reducer.test_entropy_scores(features, list(VALID_TESTS))
    for test, score in sorted(knn_scores.items(), key=lambda x: -x[1]):
        bar = "█" * max(0, int(score * 20))
        print(f"  {test:<28s}  ΔH={score:+.4f}  {bar}")
