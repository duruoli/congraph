"""Temporal and target-separation checks for IBD decision candidates."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_ibd_decision_steps import build_decision_candidates  # noqa: E402


def event(event_id, source, at, available_at, *, subject=1, detail=None, precision="timestamp"):
    return {
        "event_id": event_id, "subject_id": subject, "hadm_id": 2,
        "source_table": source, "source_key": event_id, "event_at": at,
        "available_at": available_at, "availability_precision": precision,
        "detail": detail or {},
    }


def group(actions, at="2020-01-01T10:00:00"):
    return {
        "action_group_id": "group1", "subject_id": 1, "hadm_id": 2,
        "time_precision": "timestamp", "event_at": at, "actions": actions,
    }


class DecisionCandidateTests(unittest.TestCase):
    def test_pre_action_evidence_excludes_target_later_equal_and_other_patient(self):
        events = [
            event("admission", "cohort", "2020-01-01T07:00:00", "2020-01-01T07:00:00",
                  detail={"anchor": "admission"}),
            event("prior_lab", "labs", "2020-01-01T08:00:00", "2020-01-01T09:00:00"),
            event("equal", "labs", "2020-01-01T08:00:00", "2020-01-01T10:00:00"),
            event("late", "labs", "2020-01-01T08:00:00", "2020-01-01T11:00:00"),
            event("prior_report", "radiology", "2020-01-01T07:30:00", "2020-01-01T09:30:00"),
            event("target", "radiology", "2020-01-01T10:00:00", "2020-01-01T12:00:00"),
            event("other_patient", "labs", "2020-01-01T08:00:00", "2020-01-01T09:00:00", subject=3),
            event("discharge", "discharge_notes", "2020-01-01T08:00:00", "2020-01-01T09:00:00"),
        ]
        actions = [{"event_id": "target", "source_table": "radiology", "source_key": "target",
                    "category": "gi_imaging", "detail": {"exam_name": "CT ABDOMEN"}}]
        decisions, outcomes, report = build_decision_candidates(events, [group(actions)])
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["pre_action_proxy_evidence_event_ids"],
                         ["admission", "prior_lab", "prior_report"])
        self.assertEqual(decisions[0]["action_proxy_overlap_evidence_event_ids"], ["equal"])
        self.assertEqual(outcomes[0]["target_action_event_ids"], ["target"])
        self.assertFalse(decisions[0]["pre_order_evidence_certified"])
        self.assertEqual(report["candidate_count"], 1)

    def test_routine_lab_does_not_seed_candidate(self):
        lab = {"event_id": "lab", "source_table": "labs", "category": "lab_collection"}
        decisions, _, _ = build_decision_candidates([], [group([lab])])
        self.assertEqual(decisions, [])

    def test_date_only_procedure_uses_prior_days_and_marks_same_day_uncertain(self):
        events = [
            event("prior", "labs", "2020-01-01T08:00:00", "2020-01-01T18:00:00"),
            event("prior_date", "microbiology", "2020-01-01T00:00:00",
                  "2020-01-01T00:00:00", precision="date"),
            event("prior_date_with_noon_collection", "microbiology", "2020-01-01T12:00:00",
                  "2020-01-01T00:00:00", precision="date"),
            event("same_day", "labs", "2020-01-02T08:00:00", "2020-01-02T10:00:00"),
            event("procedure", "procedures", "2020-01-02T00:00:00", None),
        ]
        procedure = {"event_id": "procedure", "source_table": "procedures",
                     "source_key": "procedure", "category": "gi_or_supportive_procedure",
                     "detail": {"long_title": "Colectomy"}}
        action = group([procedure], "2020-01-02T00:00:00")
        action["time_precision"] = "date"
        action["event_end_exclusive"] = "2020-01-03T00:00:00"
        decisions, _, _ = build_decision_candidates(events, [action])
        self.assertEqual(decisions[0]["pre_action_proxy_evidence_event_ids"],
                         ["prior_date", "prior_date_with_noon_collection", "prior"])
        self.assertEqual(decisions[0]["action_proxy_overlap_evidence_event_ids"], ["same_day"])
        self.assertEqual(decisions[0]["boundary_precision"], "date")

    def test_repeated_prescription_is_flagged_without_losing_candidate(self):
        events = [
            event("first", "prescriptions", "2020-01-01T10:00:00", None),
            event("second", "prescriptions", "2020-01-02T10:00:00", None),
        ]
        actions = []
        for index, (event_id, at) in enumerate([
            ("first", "2020-01-01T10:00:00"), ("second", "2020-01-02T10:00:00")
        ]):
            action = group([{"event_id": event_id, "source_table": "prescriptions",
                             "source_key": event_id, "category": "steroid_or_aminosalicylate_rx",
                             "detail": {"drug": "Prednisone", "route": "PO"}}], at)
            action["action_group_id"] = f"group{index}"
            actions.append(action)
        decisions, _, report = build_decision_candidates(events, actions)
        self.assertEqual([row["prescription_record_role"] for row in decisions], [
            "first_recorded_start_for_drug_and_route_in_admission",
            "repeat_same_drug_and_route_in_admission",
        ])
        self.assertEqual(report["lower_priority_repeat_prescription_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
