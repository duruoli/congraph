"""recommender.py  —  Step 7: Next-Test Recommender

Given a FullTraversalResult and a DiagnosisDistribution, ranks candidate
diagnostic tests by their expected clinical value.

Scoring model for each candidate test t:

  relevance(t) = Σ_d  P(d) × I(t ∈ pending_tests(d))  × (1 - resolved(d))

  i.e. sum over diseases of:
    probability-weighted indicator that test t is needed to advance
    disease d's rubric, excluding diseases already confirmed or excluded.

  Ties are broken by the order each test first appears (depth-first,
  so tests for the highest-ranked disease come first).

Tests already in features["tests_done"] are filtered out.

Output: list[TestRecommendation] sorted by relevance descending.
"""

from __future__ import annotations

from dataclasses import dataclass

from feature_schema import VALID_TESTS
from traversal_engine import FullTraversalResult
from diagnosis_distribution import DiagnosisDistribution


@dataclass
class TestRecommendation:
    """One ranked test recommendation."""
    test: str            # e.g. "Ultrasound_Abdomen"
    relevance: float     # weighted sum of P(d) for diseases needing this test
    # Which diseases would be advanced by this test (sorted by P descending)
    advances: list[str]


def recommend_tests(
    full_result: FullTraversalResult,
    distribution: DiagnosisDistribution,
    features: dict | None = None,
) -> list[TestRecommendation]:
    """
    Return a ranked list of next tests to order.

    Parameters
    ----------
    full_result  : output of run_full_traversal
    distribution : output of compute_distribution
    features     : raw feature dict (used to filter already-done tests)
    """
    done: set[str] = set((features or {}).get("tests_done", []))

    # Collect per-test relevance across all unresolved diseases
    relevance: dict[str, float] = {}
    advances:  dict[str, list[str]] = {}

    for name in distribution.ranked:            # highest-prob first (for tie-breaking)
        result = full_result.diseases[name]
        if result.terminal_node is not None:    # resolved: skip
            continue

        p = distribution.prob(name)
        for test in result.pending_tests:
            if test in done:
                continue
            relevance[test] = relevance.get(test, 0.0) + p
            if test not in advances:
                advances[test] = []
            advances[test].append(name)

    # Sort by relevance desc, preserve encounter order for ties
    ranked_tests = sorted(relevance, key=lambda t: relevance[t], reverse=True)

    return [
        TestRecommendation(
            test=t,
            relevance=relevance[t],
            advances=advances[t],
        )
        for t in ranked_tests
    ]


def format_recommendations(recs: list[TestRecommendation]) -> str:
    """Human-readable recommendation summary."""
    if not recs:
        return "  (no further tests needed — all active diagnoses resolved)"
    lines = []
    for i, r in enumerate(recs, 1):
        advances_str = ", ".join(r.advances)
        lines.append(
            f"  {i}. {r.test:<28s}  "
            f"relevance={r.relevance:.2f}  "
            f"advances=[{advances_str}]"
        )
    return "\n".join(lines)
