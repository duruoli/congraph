"""failure_analyzer.py — Attribution layer for the rubric optimizer.

Given a list of PatientRecord, analyses failures at two granularities:

  1. Feature fingerprint  — for each boolean clinical feature, how much more /
     less common is it in failure patients vs correctly-diagnosed ones?
     Gap < 0 → feature absent in failures (rubric might require it but dataset
                doesn't have it → condition too strict).
     Gap > 0 → feature present in failures but doesn't help (rubric might be
                over-weighting it or the wrong disease benefits more).

  2. Edge trigger rate    — for each conditional edge in the disease sub-rubric,
     what fraction of failure vs correct patients actually fired the condition?
     Large negative gap → edge under-triggers for failure patients, suggesting
     the condition is too strict or a synonym feature is missing.

The auto-generated attribution_hypothesis summarises the top findings in plain
English, ready to be included in the LLM prompt.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_PKG  = Path(__file__).parent
_ROOT = _PKG.parent
for _p in [str(_ROOT), str(_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.rubric_graph import DISEASE_GRAPHS, ALWAYS_EDGE_CONDITION
from trial_runner import PatientRecord, DISEASES


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FeatureGap:
    """Discriminative power of one boolean feature for one disease's failures."""
    feature:           str
    rate_in_failures:  float   # fraction of failure patients with feature=True
    rate_in_correct:   float   # fraction of correctly diagnosed patients with feature=True
    gap:               float   # rate_in_failures − rate_in_correct
                               # negative → feature absent in failures


@dataclass
class EdgeTriggerGap:
    """How much more/less an edge condition fires in failure vs correct patients."""
    source:            str
    target:            str
    edge_label:        str
    rate_in_failures:  float
    rate_in_correct:   float
    gap:               float   # negative → under-triggers in failures


@dataclass
class FailureAnalysis:
    """Full attribution report for failures of one disease."""
    disease:                  str
    n_failures:               int
    n_correct:                int

    most_confused_as:         str              # most common wrong prediction
    confused_as_counts:       dict[str, int]   # all wrong prediction counts

    mean_trig_frac_failures:  float   # mean (conditional_triggers / total_edges)
    mean_trig_frac_correct:   float

    feature_gaps:             list[FeatureGap]      # sorted by |gap| desc
    edge_gaps:                list[EdgeTriggerGap]  # sorted by |gap| desc

    attribution_hypothesis:   str    # auto-generated plain-English summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bool_feature_keys(records: list[PatientRecord]) -> list[str]:
    """Return sorted list of boolean feature keys present in any record."""
    keys: set[str] = set()
    for r in records:
        keys.update(
            k for k, v in r.features.items()
            if isinstance(v, bool)
        )
    return sorted(keys)


def _edge_trigger_rates(
    features_list: list[dict],
    disease:       str,
) -> dict[str, float]:
    """
    For each conditional edge in the disease sub-rubric, compute the fraction
    of patients in features_list that fire that edge's condition.

    Key format: "{source}→{target}".
    """
    graph = DISEASE_GRAPHS[disease]
    rates: dict[str, float] = {}
    n = len(features_list)
    if n == 0:
        return rates

    for edge in graph.edges:
        if edge.condition is ALWAYS_EDGE_CONDITION:
            continue
        key = f"{edge.source}→{edge.target}"
        try:
            count = sum(1 for f in features_list if bool(edge.condition(f)))
        except Exception:
            count = 0
        rates[key] = count / n

    return rates


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def analyze_failures(
    records: list[PatientRecord],
    disease: str,
    top_n:   int = 10,
) -> FailureAnalysis:
    """
    Compute feature-level and edge-level attribution for *disease* failures.

    Parameters
    ----------
    records : all PatientRecord from a trial run
    disease : the disease to analyse (only patients with true_disease==disease are used)
    top_n   : how many top discriminative features / edges to include
    """
    disease_records = [r for r in records if r.true_disease == disease]
    failures = [r for r in disease_records if not r.is_correct]
    correct  = [r for r in disease_records if r.is_correct]

    if not failures:
        return FailureAnalysis(
            disease=disease, n_failures=0, n_correct=len(correct),
            most_confused_as="n/a", confused_as_counts={},
            mean_trig_frac_failures=0.0, mean_trig_frac_correct=0.0,
            feature_gaps=[], edge_gaps=[],
            attribution_hypothesis="No failures — this disease is classified perfectly.",
        )

    # ── Confusion distribution ─────────────────────────────────────────────
    confused_counts: dict[str, int] = {}
    for r in failures:
        confused_counts[r.predicted] = confused_counts.get(r.predicted, 0) + 1
    most_confused_as = max(confused_counts, key=lambda d: confused_counts[d])

    # ── Mean rubric activation (trigger fraction) ──────────────────────────
    def _trig_frac(r: PatientRecord) -> float:
        ts = r.traversal.get(disease)
        return ts.trigger_fraction if ts else 0.0

    mean_fail = sum(_trig_frac(r) for r in failures) / len(failures)
    mean_corr = sum(_trig_frac(r) for r in correct) / len(correct) if correct else 0.0

    # ── Feature fingerprint ────────────────────────────────────────────────
    bool_keys = _bool_feature_keys(disease_records)
    feature_gaps: list[FeatureGap] = []

    for key in bool_keys:
        fail_rate = sum(
            1 for r in failures if bool(r.features.get(key, False))
        ) / len(failures)
        corr_rate = (
            sum(1 for r in correct if bool(r.features.get(key, False))) / len(correct)
            if correct else 0.0
        )
        gap = fail_rate - corr_rate
        if abs(gap) > 0.01:   # skip near-zero gaps to reduce noise
            feature_gaps.append(FeatureGap(
                feature=key,
                rate_in_failures=fail_rate,
                rate_in_correct=corr_rate,
                gap=gap,
            ))

    feature_gaps.sort(key=lambda fg: abs(fg.gap), reverse=True)
    feature_gaps = feature_gaps[:top_n]

    # ── Edge trigger rate fingerprint ──────────────────────────────────────
    fail_features = [r.features for r in failures]
    corr_features = [r.features for r in correct]

    fail_rates = _edge_trigger_rates(fail_features, disease)
    corr_rates = _edge_trigger_rates(corr_features, disease)

    graph = DISEASE_GRAPHS[disease]
    edge_gaps: list[EdgeTriggerGap] = []
    for edge in graph.edges:
        if edge.condition is ALWAYS_EDGE_CONDITION:
            continue
        key = f"{edge.source}→{edge.target}"
        fr  = fail_rates.get(key, 0.0)
        cr  = corr_rates.get(key, 0.0)
        gap = fr - cr
        if abs(gap) > 0.01:
            edge_gaps.append(EdgeTriggerGap(
                source=edge.source,
                target=edge.target,
                edge_label=edge.label,
                rate_in_failures=fr,
                rate_in_correct=cr,
                gap=gap,
            ))

    edge_gaps.sort(key=lambda eg: abs(eg.gap), reverse=True)
    edge_gaps = edge_gaps[:top_n]

    # ── Auto attribution hypothesis ────────────────────────────────────────
    parts: list[str] = []

    if edge_gaps:
        worst_edge = edge_gaps[0]
        direction  = "under-triggers" if worst_edge.gap < 0 else "over-triggers"
        parts.append(
            f"Edge '{worst_edge.edge_label}' ({worst_edge.source}→{worst_edge.target}) "
            f"{direction} in failure patients "
            f"(failure rate: {worst_edge.rate_in_failures:.0%} vs "
            f"correct: {worst_edge.rate_in_correct:.0%})."
        )

    if feature_gaps:
        top_feat  = feature_gaps[0]
        direction = "ABSENT" if top_feat.gap < 0 else "PRESENT"
        parts.append(
            f"Feature '{top_feat.feature}' is systematically {direction} "
            f"in failure cases "
            f"(failures: {top_feat.rate_in_failures:.0%}, "
            f"correct: {top_feat.rate_in_correct:.0%})."
        )

    if mean_fail < mean_corr - 0.08:
        parts.append(
            f"Overall rubric activation is low for failures "
            f"(mean {mean_fail:.1%} vs {mean_corr:.1%} for correct), "
            f"suggesting missing or too-strict conditions."
        )

    if most_confused_as:
        parts.append(
            f"Most failures are misclassified as {most_confused_as} "
            f"({confused_counts[most_confused_as]}/{len(failures)} cases)."
        )

    if not parts:
        parts.append("No strongly discriminating factor found; inspect full report.")

    return FailureAnalysis(
        disease                 = disease,
        n_failures              = len(failures),
        n_correct               = len(correct),
        most_confused_as        = most_confused_as,
        confused_as_counts      = confused_counts,
        mean_trig_frac_failures = mean_fail,
        mean_trig_frac_correct  = mean_corr,
        feature_gaps            = feature_gaps,
        edge_gaps               = edge_gaps,
        attribution_hypothesis  = " ".join(parts),
    )
