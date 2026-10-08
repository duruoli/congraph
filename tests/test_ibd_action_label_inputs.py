"""Check report-section selection and action-time evidence boundaries."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_ibd_action_label_inputs import make_packet, report_section  # noqa: E402


def event(event_id, source, at, available_at=None, detail=None, end=None, precision="timestamp"):
    return {
        "event_id": event_id, "source_key": event_id, "source_table": source,
        "event_at": at, "event_end_exclusive": end, "time_precision": precision,
        "available_at": available_at, "availability_precision": "timestamp" if available_at else "unknown",
        "detail": detail or {},
    }


class LabelInputTests(unittest.TestCase):
    def test_report_sections_do_not_merge_findings_into_indication(self):
        note = "INDICATION: assess colonic dilation\nFINDINGS: later observation\nIMPRESSION: final conclusion"
        self.assertEqual(report_section(note, ("INDICATION",), 100)["text"], "assess colonic dilation")

    def test_target_result_is_hidden_and_later_prior_report_is_excluded(self):
        action = {**event("target", "radiology", "2020-01-02T12:00:00",
                          detail={"exam_name": "ABDOMEN (SUPINE ONLY)"}),
                  "action_kind": "imaging_exam"}
        step = {"action_group_id": "group", "event_at": "2020-01-02T12:00:00",
                "time_precision": "timestamp", "event_end_exclusive": None, "actions": [action]}
        events = [
            event("prior", "radiology", "2020-01-01T09:00:00", "2020-01-01T10:00:00",
                  {"exam_name": "CT ABD"}),
            event("not_yet_reported", "radiology", "2020-01-02T08:00:00", "2020-01-02T13:00:00",
                  {"exam_name": "CT ABD"}),
        ]
        notes = {
            "target": "INDICATION: assess dilation\nIMPRESSION: target answer",
            "prior": "INDICATION: prior question\nIMPRESSION: prior answer",
            "not_yet_reported": "IMPRESSION: unavailable answer",
        }
        packet = make_packet(step, action, events, {}, notes, {})
        self.assertEqual(packet["target_report_indication"]["text"], "assess dilation")
        self.assertEqual([r["event_id"] for r in packet["prior_available_reports"]], ["prior"])
        self.assertNotIn("target answer", str(packet))
        self.assertNotIn("unavailable answer", str(packet))

    def test_date_only_procedure_excludes_same_day_evidence(self):
        action = {**event("proc", "procedures", "2020-01-02T00:00:00",
                          detail={"long_title": "Colonoscopy"},
                          end="2020-01-03T00:00:00", precision="date"),
                  "action_kind": "billed_procedure_date"}
        step = {"action_group_id": "group", "event_at": "2020-01-02T00:00:00",
                "time_precision": "date", "event_end_exclusive": "2020-01-03T00:00:00", "actions": [action]}
        same_day = event("same_day", "radiology", "2020-01-02T09:00:00",
                         "2020-01-02T10:00:00", {"exam_name": "CT ABD"})
        packet = make_packet(step, action, [same_day], {}, {"same_day": "IMPRESSION: result"}, {})
        self.assertEqual(packet["evidence_cutoff_rule"], "strictly_before_action_date")
        self.assertEqual(packet["prior_available_reports"], [])


if __name__ == "__main__":
    unittest.main()
