"""diagnosis_distribution.py  —  Step 6: Diagnosis Probability Distribution

Converts a FullTraversalResult into a probability distribution P(d | features)
over the four diagnoses: appendicitis · cholecystitis · diverticulitis · pancreatitis.

Scoring model (three additive components in log-space, summed then softmax'd):

  sub_rubric_score(d)  — evidence from traversal of disease sub-rubric
      • excluded terminal (imaging-confirmed) : EXCLUDED_SCORE    (hard, -12)
      • low_risk terminal (clinical only)     : LOW_RISK_SCORE    (soft,  -1.5)
      • conditional edges triggered           : (triggers/E) × DEPTH_CAP
        — normalised by total conditional edges in graph; "always" hops excluded
      • confirmed terminal                    : +CONFIRMED_BONUS

  empirical_score(d)   — KNN log-probability from EmpiricalScorer
      Returns 0.0 until an EmpiricalScorer is injected via set_empirical_scorer().

  triage_log_prior(d)  — log P(disease | triage routing), replaces uniform prior
      Triage activated k diseases with odds ratio TRIAGE_PRIOR_RATIO vs rest.
      Degrades gracefully to uniform log(0.25) when k=0 or k=4.
      Never reaches 0, keeping all diseases recoverable (preserves flip).

P(d) = softmax( sub_rubric_score + empirical_score + triage_log_prior )

Design rationale
----------------
  1. Graduated exclusion: low-risk terminals use LOW_RISK_SCORE=-1.5
     (≈Alvarado≤3 PPV ~10%); imaging-confirmed exclusions use EXCLUDED_SCORE=-12.
     All terminal_excluded nodes require imaging, so no soft-exclusion tier needed.
  2. Conditional-trigger depth: only patient-data-driven edge firings count toward
     depth, removing the structural advantage of graphs with many "always" hops.
  2. Depth normalisation: (depth / graph_size) × DEPTH_CAP removes structural
     bias from graphs of different sizes.
  3. Triage as log-prior: triage evidence is modelled as a prior probability
     distribution, keeping it conceptually separate from sub-rubric likelihood.
     The prior odds ratio TRIAGE_PRIOR_RATIO is conservative (default 2) so
     strong sub-rubric evidence can always flip the initial hypothesis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from knn.empirical_scorer import EmpiricalScorer

from pipeline.traversal_engine import FullTraversalResult, TraversalResult
from pipeline.rubric_graph import ALWAYS_EDGE_CONDITION, DISEASE_GRAPHS


# ---------------------------------------------------------------------------
# Scoring hyper-parameters  (adjust here)
# ---------------------------------------------------------------------------

CONFIRMED_BONUS:     float = 6.0   # terminal_confirmed reached
DEPTH_CAP:           float = 3.0   # max contribution from normalised conditional triggers
EXCLUDED_SCORE:      float = -12.0 # hard exclusion (imaging-confirmed; all terminal_excluded require imaging)
LOW_RISK_SCORE:      float = -1.5  # low-risk terminal: clinical suspicion low but not excluded
                                   # (Alvarado ≤3 PPV ~10%; TG18 A/B criteria not met at step-0)
TRIAGE_PRIOR_RATIO:  float = 2.0   # prior odds ratio: activated vs non-activated

# Component weights — scale each term before summing.
# sub_rubric_score spans ~[-12, 9] (range ≈ 21), while empirical_score and
# triage_log_prior are proper log-probabilities with range ≈ [-4, 0] and
# ≈ [-2, -0.7] respectively.  W_RUBRIC ≈ 0.2 brings the rubric span in line
# with the empirical component so no single term dominates the final softmax.
W_RUBRIC:    float = 1.0   # weight for _sub_rubric_score
W_EMPIRICAL: float = 1.0   # weight for empirical_score (KNN log-prob)
W_TRIAGE:    float = 1.0   # weight for triage log-prior

# Conditional edge counts per disease graph for trigger normalisation.
# Only edges whose condition is NOT ALWAYS_EDGE_CONDITION count.
_GRAPH_COND_EDGE_COUNTS: dict[str, int] = {
    name: sum(1 for e in graph.edges if e.condition is not ALWAYS_EDGE_CONDITION)
    for name, graph in DISEASE_GRAPHS.items()
}


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
# Sub-rubric scoring (pure traversal evidence, no triage)
# ---------------------------------------------------------------------------

def _sub_rubric_score(result: TraversalResult) -> float:
    """
    Compute a scalar evidence score from one disease's sub-rubric traversal.

    Terminal handling:
      terminal_low_risk  → LOW_RISK_SCORE (-1.5): clinical evidence suggests low
                           probability but this is NOT a definitive exclusion.
                           Disease remains in differential; triage can flip it.
      terminal_excluded  → EXCLUDED_SCORE (-12): always imaging-confirmed
                           (all terminal_excluded nodes require imaging).

    Evidence depth uses conditional_triggers — the number of conditional edges
    (condition is not ALWAYS_EDGE_CONDITION) whose conditions evaluated to True.
    Structural "always" hops are excluded so graphs cannot score higher merely
    by having more unconditional routing edges (e.g. diverticulitis's
    "pain != LLQ → uncertain → CT" catch-all path no longer inflates its score).
    """
    if result.low_risk:
        return LOW_RISK_SCORE

    if result.excluded:
        return EXCLUDED_SCORE

    cond_total = _GRAPH_COND_EDGE_COUNTS.get(result.disease, 1)
    norm_triggers = result.conditional_triggers / cond_total
    score = norm_triggers * DEPTH_CAP
    if result.confirmed:
        score += CONFIRMED_BONUS
    return score


# ---------------------------------------------------------------------------
# Triage log-prior
# ---------------------------------------------------------------------------

def _triage_log_prior(disease: str, full_result: FullTraversalResult) -> float:
    """
    Return log P(disease | triage routing).

    Triage activating k diseases shifts their prior to p_act = r × p_not,
    where r = TRIAGE_PRIOR_RATIO and p_act + p_not normalise to 1.
    Falls back to uniform log(0.25) when k=0 or k=4.
    """
    n_activated = sum(1 for r in full_result.diseases.values() if r.triage_activated)
    if n_activated == 0 or n_activated == 4:
        return math.log(0.25)

    r = TRIAGE_PRIOR_RATIO
    # p_not × (k×r + (4-k)) = 1  →  solve for p_not
    p_not = 1.0 / (n_activated * r + (4 - n_activated))
    p_act = r * p_not

    activated = full_result.diseases[disease].triage_activated
    return math.log(p_act if activated else p_not)


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

    raw_scores: dict[str, float] = {}
    for name, result in full_result.diseases.items():
        raw_scores[name] = (
            W_RUBRIC    * _sub_rubric_score(result)
            + W_EMPIRICAL * empirical_score(name, _features)
            + W_TRIAGE    * _triage_log_prior(name, full_result)
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
