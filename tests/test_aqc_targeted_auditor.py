from __future__ import annotations

import unittest

from scripts.aqc_algorithmic_auditor import apply_operations
from scripts.aqc_targeted_auditor import _temporal_operations, temporal_candidate


def annotation(*, aggregate: str = "partially_answered") -> dict:
    return {
        "current_question": {
            "primary": "Have the pancreatic collections progressed since the prior CT?",
            "secondary_questions": [],
            "answer_requirements": [
                {
                    "requirement_key": "req_extent",
                    "id": "severity_or_extent",
                    "dimension": "Current collection size",
                }
            ],
        },
        "coverage": {
            "requirements": [
                {
                    "requirement_key": "req_extent",
                    "requirement_id": "severity_or_extent",
                    "status": "sufficiently_addressed",
                }
            ],
            "aggregate": aggregate,
            "aggregate_reason": "Original reason.",
        },
    }


def response(decision: str) -> dict:
    return {
        "decision": decision,
        "reason": "The comparison with the prior CT is independently necessary.",
        "revised_primary": None,
        "revised_secondary_questions": None,
        "temporal_requirement": None,
        "temporal_coverage": None,
    }


class TemporalAuditorTests(unittest.TestCase):
    def test_overlay_supports_minimal_array_append(self) -> None:
        repaired = apply_operations(
            {"items": [{"id": "existing"}]},
            [{"op": "add", "path": "/items/-", "value": {"id": "new"}}],
        )
        self.assertEqual(repaired["items"], [{"id": "existing"}, {"id": "new"}])

    def test_required_adds_typed_requirement_and_keeps_compatible_aggregate(self) -> None:
        source = annotation()
        model = response("interval_comparison_required")
        model["temporal_requirement"] = {
            "requirement_key": "req_time",
            "id": "temporal_course_or_response",
            "other_proposed_dimension": "",
            "dimension": "Change in collection size compared with the prior CT",
            "why_required": "The question asks whether the collection progressed.",
        }
        model["temporal_coverage"] = {
            "requirement_key": "req_time",
            "requirement_id": "temporal_course_or_response",
            "status": "unaddressed",
            "direction": "no_direction",
            "supporting_evidence": [],
            "reason": "The prior CT supplies a baseline but not the later change.",
        }

        operations, errors = _temporal_operations(source, model)
        repaired = apply_operations(source, operations)

        self.assertEqual(errors, [])
        self.assertIsNone(temporal_candidate(repaired))
        self.assertEqual(repaired["coverage"]["aggregate"], "partially_answered")

    def test_aggregate_is_derived_when_addition_invalidates_original(self) -> None:
        source = annotation(aggregate="sufficiently_answered")
        model = response("interval_comparison_required")
        model["temporal_requirement"] = {
            "requirement_key": "req_time",
            "id": "temporal_course_or_response",
            "other_proposed_dimension": "",
            "dimension": "Change compared with the prior CT",
            "why_required": "Progression is the clinical question.",
        }
        model["temporal_coverage"] = {
            "requirement_key": "req_time",
            "requirement_id": "temporal_course_or_response",
            "status": "unaddressed",
            "direction": "no_direction",
            "supporting_evidence": [],
            "reason": "Only the baseline is available.",
        }

        operations, errors = _temporal_operations(source, model)
        repaired = apply_operations(source, operations)

        self.assertEqual(errors, [])
        self.assertEqual(repaired["coverage"]["aggregate"], "partially_answered")
        self.assertIn("deterministically updated", repaired["coverage"]["aggregate_reason"])

    def test_current_state_decision_requires_and_closes_wording_repair(self) -> None:
        source = annotation()
        model = response("current_state_question_only")
        model["reason"] = "Only current collection size is needed."
        model["revised_primary"] = "What is the current size of the pancreatic collections?"

        operations, errors = _temporal_operations(source, model)
        repaired = apply_operations(source, operations)

        self.assertEqual(errors, [])
        self.assertIsNone(temporal_candidate(repaired))

    def test_current_state_decision_removes_existing_typed_requirement(self) -> None:
        source = annotation()
        source["current_question"]["primary"] = "What is the current collection size?"
        source["current_question"]["answer_requirements"].append({
            "requirement_key": "req_time",
            "id": "temporal_course_or_response",
            "dimension": "Interval change",
        })
        source["coverage"]["requirements"].append({
            "requirement_key": "req_time",
            "requirement_id": "temporal_course_or_response",
            "status": "unaddressed",
        })
        model = response("current_state_question_only")
        model["reason"] = "No comparison is needed to answer the current-state question."

        operations, errors = _temporal_operations(source, model)
        repaired = apply_operations(source, operations)

        self.assertEqual(errors, [])
        self.assertIsNone(temporal_candidate(repaired))
        self.assertEqual(len(repaired["current_question"]["answer_requirements"]), 1)
        self.assertEqual(len(repaired["coverage"]["requirements"]), 1)

    def test_ambiguous_aligned_decision_is_rejected(self) -> None:
        model = response("aligned")
        _, errors = _temporal_operations(annotation(), model)
        self.assertEqual(errors, ["targeted_temporal_bad_decision"])


if __name__ == "__main__":
    unittest.main()
