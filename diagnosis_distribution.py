"""diagnosis_distribution.py  —  Step 6: Diagnosis Probability Distribution

Converts a FullTraversalResult into a probability distribution P(d | features)
over the four diagnoses: appendicitis · cholecystitis · diverticulitis · pancreatitis.

Scoring model (three additive components, summed then softmax'd):

  rubric_score(d)   — derived purely from traversal evidence
      • confirmed terminal  : +CONFIRMED_BONUS   (strong evidence)
      • triage activated    : +TRIAGE_BONUS       (routing evidence)
      • traversal depth     : +depth × DEPTH_SCALE, capped at DEPTH_CAP
      • excluded terminal   : EXCLUDED_SCORE      (near-zero after softmax)

  mimic_score(d)    — stub returning 0.0 for all diseases
      Will be replaced with MIMIC feature-frequency likelihood ratios
      once the MIMIC retrieval pipeline (Step 5) is connected.

  log_prior(d)      — uniform log-prior (log 0.25) for all diseases;
      can be swapped for population base-rate priors later.

P(d) = softmax( rubric_score + mimic_score + log_prior )

Design: this module is intentionally decoupled from the traversal engine.
The traversal produces TraversalResult objects; this module interprets them
as probabilistic evidence.  They can evolve independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from empirical_scorer import EmpiricalScorer

from traversal_engine import FullTraversalResult, TraversalResult


# ---------------------------------------------------------------------------
# Scoring hyper-parameters  (adjust here; values tuned on toy patients)
# ---------------------------------------------------------------------------

CONFIRMED_BONUS: float = 6.0   # Terminal confirmed reached
TRIAGE_BONUS:    float = 2.0   # Disease activated by triage routing
DEPTH_SCALE:     float = 0.5   # Score per rubric node traversed
DEPTH_CAP:       float = 3.0   # Max depth contribution (prevents depth dominating)
EXCLUDED_SCORE:  float = -12.0 # Effectively zeros out excluded diseases (exp≈0)


# ---------------------------------------------------------------------------
# Empirical scorer — module-level singleton (set via set_empirical_scorer)
# ---------------------------------------------------------------------------

_empirical_scorer: Optional["EmpiricalScorer"] = None


def set_empirical_scorer(scorer: "EmpiricalScorer") -> None:
    """
    Inject a fitted EmpiricalScorer so that empirical_score() returns real
    KNN-derived log-probabilities instead of 0.0.

    Call this once before running the pipeline (e.g. from real_pipeline.py
    after training the scorer on the held-out training split).
    """
    global _empirical_scorer
    _empirical_scorer = scorer


def clear_empirical_scorer() -> None:
    """Remove the empirical scorer (revert to uniform stub)."""
    global _empirical_scorer
    _empirical_scorer = None


def empirical_score(disease: str, features: dict) -> float:
    """
    Return the empirical component of the scoring model.

    If an EmpiricalScorer has been injected via set_empirical_scorer(), this
    delegates to its KNN log-probability estimate:
        log P(disease | K nearest neighbors in training corpus)

    Otherwise returns 0.0 (uniform — rubric score is the sole signal).
    """
    if _empirical_scorer is not None and _empirical_scorer.is_fitted:
        return _empirical_scorer.score(disease, features)
    return 0.0


# ---------------------------------------------------------------------------
# Rubric scoring
# ---------------------------------------------------------------------------

def _rubric_score(result: TraversalResult) -> float:
    """Compute a scalar evidence score from one disease's traversal result."""
    if result.excluded:
        return EXCLUDED_SCORE

    score = 0.0
    if result.triage_activated:
        score += TRIAGE_BONUS
    score += min(result.depth * DEPTH_SCALE, DEPTH_CAP)
    if result.confirmed:
        score += CONFIRMED_BONUS
    return score


# ---------------------------------------------------------------------------
# Softmax helper
# ---------------------------------------------------------------------------

def _softmax(scores: dict[str, float]) -> dict[str, float]:
    """Numerically stable softmax over a score dict."""
    max_s = max(scores.values())
    exps = {d: math.exp(s - max_s) for d, s in scores.items()}
    total = sum(exps.values())
    return {d: e / total for d, e in exps.items()}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisDistribution:
    """
    Probability distribution over the four diagnoses.

    Attributes
    ----------
    probabilities   : dict[disease -> P(d)]  (sum to 1.0)
    raw_scores      : dict[disease -> float] (before softmax)
    ranked          : diseases in descending probability order
    primary         : highest-probability disease
    resolved        : diseases with a confirmed OR excluded terminal
    """
    probabilities: dict[str, float]
    raw_scores:    dict[str, float]
    ranked:        list[str]
    primary:       str
    resolved:      dict[str, Optional[str]]   # disease -> terminal_node | None

    def prob(self, disease: str) -> float:
        """Return P(disease), or 0.0 if unknown."""
        return self.probabilities.get(disease, 0.0)

    def top(self, n: int = 2) -> list[tuple[str, float]]:
        """Return the top-n (disease, probability) pairs."""
        return [(d, self.probabilities[d]) for d in self.ranked[:n]]

    def summary_line(self) -> str:
        parts = [
            f"{d}={self.probabilities[d]:.1%}"
            for d in self.ranked
        ]
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_distribution(
    full_result: FullTraversalResult,
    features: dict | None = None,
) -> DiagnosisDistribution:
    """
    Compute P(d | features) from a FullTraversalResult.

    Parameters
    ----------
    full_result : result of traversal_engine.run_full_traversal(features)
    features    : the raw feature dict (needed for the MIMIC stub hook;
                  can be None until Step 5 is connected)
    """
    _features = features or {}
    _LOG_PRIOR = math.log(0.25)   # uniform prior over 4 diseases

    raw_scores: dict[str, float] = {}
    for name, result in full_result.diseases.items():
        raw_scores[name] = (
            _rubric_score(result)
            + empirical_score(name, _features)
            + _LOG_PRIOR
        )

    probs = _softmax(raw_scores)
    ranked = sorted(probs, key=lambda d: probs[d], reverse=True)
    primary = ranked[0]

    resolved: dict[str, Optional[str]] = {
        name: result.terminal_node
        for name, result in full_result.diseases.items()
        if result.terminal_node is not None
    }

    return DiagnosisDistribution(
        probabilities=probs,
        raw_scores=raw_scores,
        ranked=ranked,
        primary=primary,
        resolved=resolved,
    )
