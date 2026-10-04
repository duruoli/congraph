"""The report arrival, rather than the exam, anchors an information step."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_ibd_information_steps import make_information_steps, trigger_reason  # noqa: E402


class InformationStepTests(unittest.TestCase):
    def test_cmv_viral_load_triggers_but_ldh_does_not(self):
        base = {"source_table": "labs", "detail": {"label": "Cytomegalovirus Viral Load"}}
        self.assertEqual(trigger_reason(base), "selected_lab_result_available")
        base["detail"]["label"] = "Lactate Dehydrogenase (LD)"
        self.assertIsNone(trigger_reason(base))

    def test_report_arrival_is_step_cutoff(self):
        events = [
            {"event_id": "exam", "subject_id": 1, "hadm_id": 2,
             "source_table": "radiology", "source_key": "report-1",
             "event_at": "2020-01-01T08:00:00", "available_at": "2020-01-01T11:00:00",
             "detail": {"exam_name": "CT ABDOMEN"}},
        ]
        checkpoints = [
            {"subject_id": 1, "hadm_ids": [2], "step_index": 0,
             "available_at": "2020-01-01T11:00:00", "new_event_ids": ["exam"],
             "cumulative_available_event_count": 1},
        ]
        steps = make_information_steps(events, checkpoints)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["evidence_cutoff_at"], "2020-01-01T11:00:00")
        self.assertEqual(steps[0]["source_references"][0]["event_at"], "2020-01-01T08:00:00")

    def test_routine_lab_checkpoint_is_retained_only_in_source(self):
        event = {"event_id": "glucose", "subject_id": 1, "hadm_id": 2,
                 "source_table": "labs", "source_key": "3",
                 "event_at": "2020-01-01T08:00:00", "available_at": "2020-01-01T09:00:00",
                 "detail": {"label": "Glucose"}}
        checkpoint = {"subject_id": 1, "hadm_ids": [2], "step_index": 0,
                      "available_at": "2020-01-01T09:00:00", "new_event_ids": ["glucose"],
                      "cumulative_available_event_count": 1}
        self.assertEqual(make_information_steps([event], [checkpoint]), [])

    def test_skipped_checkpoint_is_in_next_step_delta(self):
        routine = {"event_id": "glucose", "subject_id": 1, "hadm_id": 2,
                   "source_table": "labs", "source_key": "3",
                   "event_at": "2020-01-01T08:00:00", "available_at": "2020-01-01T09:00:00",
                   "detail": {"label": "Glucose"}}
        report = {"event_id": "ct", "subject_id": 1, "hadm_id": 2,
                  "source_table": "radiology", "source_key": "4",
                  "event_at": "2020-01-01T08:30:00", "available_at": "2020-01-01T11:00:00",
                  "detail": {"exam_name": "CT ABDOMEN"}}
        checkpoints = [
            {"subject_id": 1, "hadm_ids": [2], "step_index": 0,
             "available_at": "2020-01-01T09:00:00", "new_event_ids": ["glucose"],
             "cumulative_available_event_count": 1},
            {"subject_id": 1, "hadm_ids": [2], "step_index": 1,
             "available_at": "2020-01-01T11:00:00", "new_event_ids": ["ct"],
             "cumulative_available_event_count": 2},
        ]
        steps = make_information_steps([routine, report], checkpoints)
        self.assertEqual(steps[0]["trigger_event_ids"], ["ct"])
        self.assertEqual(steps[0]["new_since_previous_step_event_ids"], ["glucose", "ct"])


if __name__ == "__main__":
    unittest.main()
