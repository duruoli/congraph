"""ig_recommender.py — Information-Gain re-ranked test recommender

Two-stage ranking pipeline:
  Stage 1 (BFS structural rank)  — existing recommender.py
      relevance(t) = Σ_d  P(d) × I(t ∈ pending_tests(d))
      Tells us which tests are structurally needed by the rubric graphs.

  Stage 2 (Empirical entropy-reduction re-rank)  — this module
      ig_score(t) = E[H_before − H_after | test=t, current_state≈query]
                    estimated via KNN over the training corpus.
      Tells us which of the structurally-needed tests will most efficiently
      reduce diagnostic uncertainty given the current patient state.

Final score (weighted combination):
      final_score(t) = α × ig_score(t) + (1 − α) × relevance(t)

  α = 1.0  → pure information-gain ordering
  α = 0.0  → pure BFS structural ordering (original recommender behaviour)
  α = 0.5  → balanced (default)

Usage
-----
    from entropy_reducer import EntropyReducer
    from ig_recommender import IGRecommender

    reducer = EntropyReducer(k=15)
    reducer.fit(train_patients)

    ig_rec = IGRecommender(reducer, alpha=0.5)
    reranked = ig_rec.rerank(bfs_recommendations, current_features)
"""

from __future__ import annotations

from dataclasses import dataclass

from recommender import TestRecommendation
from entropy_reducer import EntropyReducer


# ---------------------------------------------------------------------------
# Enhanced recommendation dataclass (extends the BFS one with IG scores)
# ---------------------------------------------------------------------------

@dataclass
class IGTestRecommendation:
    """
    Re-ranked test recommendation with both BFS and IG scores exposed.

    Attributes
    ----------
    test          : test name
    bfs_relevance : original BFS structural relevance score
    ig_score      : KNN-conditioned expected entropy reduction (ΔH bits)
    final_score   : weighted combination used for final ranking
    alpha         : IG weight used (for reproducibility)
    advances      : diseases advanced by this test (from BFS stage)
    """
    test:          str
    bfs_relevance: float
    ig_score:      float
    final_score:   float
    alpha:         float
    advances:      list[str]


# ---------------------------------------------------------------------------
# IGRecommender
# ---------------------------------------------------------------------------

class IGRecommender:
    """
    Wraps an EntropyReducer to re-rank a BFS recommendation list.

    Parameters
    ----------
    reducer : a fitted EntropyReducer instance
    alpha   : weight on the IG score component [0, 1]
              0.0 → pure BFS order
              1.0 → pure IG order
              0.5 → balanced (default)
    normalise : if True, independently normalise bfs_relevance and ig_score
                to [0, 1] before combining (recommended when alpha != 0 or 1)
    """

    def __init__(
        self,
        reducer: EntropyReducer,
        alpha: float = 0.5,
        normalise: bool = True,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.reducer   = reducer
        self.alpha     = alpha
        self.normalise = normalise

    # ── main entry point ──────────────────────────────────────────────────────

    def rerank(
        self,
        bfs_recs: list[TestRecommendation],
        current_features: dict,
    ) -> list[IGTestRecommendation]:
        """
        Re-rank a BFS recommendation list using KNN-conditioned IG scores.

        Parameters
        ----------
        bfs_recs         : output of recommender.recommend_tests()
        current_features : current patient feature dict (query for KNN)

        Returns
        -------
        List of IGTestRecommendation sorted by final_score descending.
        """
        if not bfs_recs:
            return []

        candidate_tests = [r.test for r in bfs_recs]

        # Stage 2: KNN-conditioned entropy reduction scores
        ig_raw = self.reducer.test_entropy_scores(current_features, candidate_tests)

        # Collect raw scores
        bfs_vals = [r.relevance for r in bfs_recs]
        ig_vals  = [ig_raw.get(r.test, 0.0) for r in bfs_recs]

        # Normalise each component to [0, 1] independently
        if self.normalise:
            bfs_norm = _normalise(bfs_vals)
            ig_norm  = _normalise(ig_vals)
        else:
            bfs_norm = bfs_vals
            ig_norm  = ig_vals

        results: list[IGTestRecommendation] = []
        for rec, bfs_n, ig_n, ig_r in zip(bfs_recs, bfs_norm, ig_norm, ig_vals):
            final = self.alpha * ig_n + (1.0 - self.alpha) * bfs_n
            results.append(IGTestRecommendation(
                test          = rec.test,
                bfs_relevance = rec.relevance,
                ig_score      = ig_r,          # raw ΔH in bits
                final_score   = final,
                alpha         = self.alpha,
                advances      = rec.advances,
            ))

        results.sort(key=lambda r: r.final_score, reverse=True)
        return results

    # ── convenience: pure IG ranking (alpha = 1.0) ───────────────────────────

    def pure_ig_rank(
        self,
        candidate_tests: list[str],
        current_features: dict,
    ) -> list[tuple[str, float]]:
        """
        Rank arbitrary tests purely by KNN-conditioned IG score (no BFS weight).

        Returns list of (test_name, ig_score) sorted by ig_score descending.
        """
        scores = self.reducer.test_entropy_scores(current_features, candidate_tests)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_ig_recommendations(recs: list[IGTestRecommendation]) -> str:
    """Human-readable re-ranked recommendation summary."""
    if not recs:
        return "  (no further tests needed)"
    lines = []
    for i, r in enumerate(recs, 1):
        advances_str = ", ".join(r.advances)
        lines.append(
            f"  {i}. {r.test:<28s}  "
            f"final={r.final_score:.3f}  "
            f"(bfs={r.bfs_relevance:.2f}  ig_ΔH={r.ig_score:+.4f})  "
            f"advances=[{advances_str}]"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(values: list[float]) -> list[float]:
    """Min-max normalise a list to [0, 1]. Returns as-is if all equal."""
    mn, mx = min(values), max(values)
    if mx - mn < 1e-12:
        return [0.5] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path

    from clinical_session import ClinicalSession
    from entropy_reducer import EntropyReducer

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

    ig_rec = IGRecommender(reducer, alpha=0.5)

    # Pick first patient from each disease and show BFS vs re-ranked recs
    for disease in list(DISEASE_FILES)[:2]:
        patients = all_patients[disease]
        pid   = next(iter(patients))
        steps = patients[pid]
        step  = steps[0]   # first step (HPI + PE + basic labs)

        features = step["features"]
        session  = ClinicalSession(features)
        state    = session.assess()

        print(f"\n{'═'*70}")
        print(f"  {disease.upper()}  |  patient {pid}  |  step 0")
        print(f"{'═'*70}")

        print("\n  BFS ranking (original recommender):")
        for i, r in enumerate(state.recommendations, 1):
            print(f"    {i}. {r.test:<28s}  relevance={r.relevance:.2f}")

        reranked = ig_rec.rerank(state.recommendations, features)
        print("\n  IG re-ranked (alpha=0.5):")
        print(format_ig_recommendations(reranked))
