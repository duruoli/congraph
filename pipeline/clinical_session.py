"""clinical_session.py  —  Step 8: Integration Layer

ClinicalSession is the single entry point for the full pipeline:

    feature_dict  →  [Traversal]  →  [Distribution]  →  [Recommender]
                          ↓                  ↓                  ↓
                    FullTraversalResult  DiagnosisDistribution  list[TestRec]
                          └──────────────────┴──────────────────┘
                                           AssessmentState

Usage
-----
    session = ClinicalSession()

    # HPI + Physical Exam
    session.update(pain_location="RUQ", murphys_sign=True, nausea_vomiting=True)
    state = session.assess()
    state.print_report()

    # Basic Labs come back
    session.add_test("Lab_Panel", WBC_gt_10k=True, CRP_elevated=True)
    state = session.assess()
    state.print_report()

    # Ultrasound
    session.add_test("Ultrasound_Abdomen",
                     US_gallstones=True, US_GB_wall_thickening=True)
    state = session.assess()
    state.print_report()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pipeline.feature_schema import default_features, VALID_TESTS
from pipeline.traversal_engine import FullTraversalResult, run_full_traversal
from pipeline.diagnosis_distribution import DiagnosisDistribution, compute_distribution
from pipeline.recommender import TestRecommendation, recommend_tests, format_recommendations

# Boolean features where positive evidence should never be retracted by a later
# test that simply doesn't mention the finding.  Once True, stays True.
# 一旦某项检查确认为阳性，后续检查不得将其覆写为阴性。
_STICKY_TRUE_FEATURES: frozenset[str] = frozenset({
    "has_organ_dysfunction",
    "organ_failure_transient",
    "organ_failure_persistent",
    "local_complications_pancreatitis",
})


# ---------------------------------------------------------------------------
# AssessmentState — snapshot of the full pipeline output
# ---------------------------------------------------------------------------

@dataclass
class AssessmentState:
    """Immutable snapshot produced by ClinicalSession.assess()."""

    features:       dict
    traversal:      FullTraversalResult
    distribution:   DiagnosisDistribution
    recommendations: list[TestRecommendation]

    # ── convenience properties ────────────────────────────────────────────

    @property
    def primary_diagnosis(self) -> str:
        return self.distribution.primary

    @property
    def tests_done(self) -> list[str]:
        return list(self.features.get("tests_done", []))

    @property
    def confirmed_diagnoses(self) -> dict[str, str]:
        """Return {disease: terminal_node} for confirmed diseases."""
        return {
            d: r.terminal_node
            for d, r in self.traversal.diseases.items()
            if r.confirmed
        }

    @property
    def excluded_diagnoses(self) -> dict[str, str]:
        """Return {disease: terminal_node} for excluded diseases."""
        return {
            d: r.terminal_node
            for d, r in self.traversal.diseases.items()
            if r.excluded
        }

    # ── report ────────────────────────────────────────────────────────────

    def print_report(self) -> None:
        """Print a full human-readable assessment report."""
        _bar = "═" * 64

        print(f"\n{_bar}")
        print(f"  CLINICAL ASSESSMENT")
        print(f"  Tests done : {self.tests_done or '(none)'}")
        print(f"{_bar}")

        # Probability distribution
        print("\n  DIAGNOSIS DISTRIBUTION")
        print(f"  {'─'*60}")
        for d in self.distribution.ranked:
            p = self.distribution.prob(d)
            r = self.traversal.diseases[d]
            bar = "█" * int(p * 30)
            tag = ""
            if r.confirmed:
                tag = f"  ✓ CONFIRMED [{r.terminal_node}]"
            elif r.excluded:
                tag = f"  ✗ excluded  [{r.terminal_node}]"
            elif r.triage_activated:
                tag = "  ◀ triage"
            print(f"  {d:16s} {p:5.1%}  {bar:<30s}{tag}")

        # Rubric positions
        print(f"\n  RUBRIC POSITIONS")
        print(f"  {'─'*60}")
        for d in self.distribution.ranked:
            r = self.traversal.diseases[d]
            frontier_labels = []
            for nid in r.frontier:
                ns = r.node_statuses[nid]
                icon = {"pending": "⏳", "blocked": "✗",
                        "frontier_leaf": "◉"}.get(ns.status, "?")
                extra = f" ← {ns.missing_tests}" if ns.missing_tests else ""
                frontier_labels.append(f"{icon}{nid}{extra}")
            front_str = "  |  ".join(frontier_labels) or "(none)"
            print(f"  {d:16s}  depth={r.depth}  frontier: {front_str}")

        # Recommendations
        print(f"\n  NEXT TESTS RECOMMENDED")
        print(f"  {'─'*60}")
        print(format_recommendations(self.recommendations))

        print(f"\n{_bar}\n")

    def summary(self) -> str:
        """One-line summary string."""
        top = self.distribution.top(2)
        top_str = "  ".join(f"{d}={p:.0%}" for d, p in top)
        recs = [r.test for r in self.recommendations[:2]]
        return (
            f"[{', '.join(self.tests_done) or 'HPI/PE only'}]  "
            f"P: {top_str}  |  next: {recs}"
        )


# ---------------------------------------------------------------------------
# ClinicalSession — stateful session manager
# ---------------------------------------------------------------------------

class ClinicalSession:
    """
    Manages a single patient encounter.

    State is held in a feature dict (see feature_schema.default_features).
    Call update() or add_test() to add evidence, then assess() to run the
    full pipeline and get an AssessmentState snapshot.
    """

    def __init__(self, initial_features: dict | None = None) -> None:
        self._features: dict = (
            dict(initial_features) if initial_features else default_features()
        )

    # ── feature updates ───────────────────────────────────────────────────

    def update(self, **kwargs) -> "ClinicalSession":
        """
        Update one or more feature fields.

        Usage:
            session.update(pain_location="RUQ", murphys_sign=True)
        """
        for key, value in kwargs.items():
            if key == "tests_done":
                # Extend, don't replace
                existing = set(self._features.get("tests_done", []))
                existing.update(value if isinstance(value, list) else [value])
                self._features["tests_done"] = list(existing)
            else:
                self._features[key] = value
        return self   # fluent API

    def add_test(self, test_name: str, **results) -> "ClinicalSession":
        """
        Record that a test was completed and update its result features.

        Usage:
            session.add_test("Lab_Panel", WBC_gt_10k=True, CRP_elevated=True)
            session.add_test("Ultrasound_Abdomen",
                             US_gallstones=True, US_GB_wall_thickening=True)
        """
        if test_name not in VALID_TESTS:
            raise ValueError(
                f"Unknown test {test_name!r}. Valid tests: {VALID_TESTS}"
            )
        done = list(self._features.get("tests_done", []))
        if test_name not in done:
            done.append(test_name)
        self._features["tests_done"] = done
        for key, value in results.items():
            if key in _STICKY_TRUE_FEATURES and isinstance(value, bool):
                # Positive evidence is cumulative: True from any source wins.
                self._features[key] = self._features.get(key, False) or value
            else:
                self._features[key] = value
        return self   # fluent API

    # ── pipeline ─────────────────────────────────────────────────────────

    def assess(self) -> AssessmentState:
        """Run the full pipeline and return an AssessmentState snapshot."""
        traversal     = run_full_traversal(self._features)
        distribution  = compute_distribution(traversal, self._features)
        recs          = recommend_tests(traversal, distribution, self._features)
        return AssessmentState(
            features        = dict(self._features),
            traversal       = traversal,
            distribution    = distribution,
            recommendations = recs,
        )

    # ── snapshot / restore ────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return a deep copy of the current feature dict."""
        import copy
        return copy.deepcopy(self._features)

    def load_snapshot(self, features: dict) -> "ClinicalSession":
        """Restore session from a previously saved snapshot."""
        import copy
        self._features = copy.deepcopy(features)
        return self

    @property
    def features(self) -> dict:
        """Read-only view of current features."""
        return dict(self._features)
