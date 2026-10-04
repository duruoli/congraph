"""Checks for temporal and grouping risks in the IBD action-step selector."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_ibd_action_steps import make_steps  # noqa: E402


def event(event_id, source, at, precision="timestamp", available_at=None, detail=None, end=None):
    return {
        "subject_id": 1, "hadm_id": 2, "event_id": event_id,
        "source_table": source, "source_key": event_id,
        "event_at": at, "event_end_exclusive": end,
        "time_precision": precision, "available_at": available_at,
        "availability_precision": "timestamp" if available_at else "unknown",
        "detail": detail or {},
    }


class ActionStepTests(unittest.TestCase):
    def test_result_does_not_create_a_second_action_step(self):
        events = [
            event("exam", "radiology", "2020-01-01T08:00:00", available_at="2020-01-01T11:00:00",
                  detail={"exam_name": "CT ABD & PELVIS"}),
            event("rx", "prescriptions", "2020-01-01T09:00:00",
                  detail={"drug": "Prednisone"}),
        ]
        _, steps = make_steps(events)
        self.assertEqual([s["event_at"] for s in steps], ["2020-01-01T08:00:00", "2020-01-01T09:00:00"])
        self.assertFalse(steps[0]["strict_asof_action_evidence"])

    def test_same_time_labs_group_and_date_procedure_keeps_interval(self):
        events = [
            event("a", "labs", "2020-01-01T06:00:00", available_at="2020-01-01T07:00:00"),
            event("b", "labs", "2020-01-01T06:00:00", available_at="2020-01-01T07:05:00"),
            event("c", "procedures", "2020-01-01T00:00:00", "date",
                  detail={"long_title": "Colonoscopy"}, end="2020-01-02T00:00:00"),
        ]
        _, steps = make_steps(events)
        self.assertEqual(len(steps), 2)
        lab = next(s for s in steps if s["time_precision"] == "timestamp")
        procedure = next(s for s in steps if s["time_precision"] == "date")
        self.assertEqual(set(lab["source_event_ids"]), {"a", "b"})
        self.assertEqual(procedure["event_end_exclusive"], "2020-01-02T00:00:00")
        self.assertEqual(procedure["order_certainty"], "date_interval_only")

    def test_discharge_and_retro_note_are_not_action_anchors(self):
        events = [
            event("discharge", "cohort", "2020-01-02T10:00:00", detail={"anchor": "discharge"}),
            event("note", "discharge_notes", "2020-01-02T10:00:00", detail={"note_type": "DS"}),
            event("start", "cohort", "2020-01-01T10:00:00", available_at="2020-01-01T10:00:00",
                  detail={"anchor": "admission"}),
        ]
        candidates, steps = make_steps(events)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["selection_categories"], ["care_transition"])


if __name__ == "__main__":
    unittest.main()
