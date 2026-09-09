from __future__ import annotations

import unittest

from experiments.aqc_acr_bridge.prompts import LLM_DIMENSIONS, output_contract
from scripts.run_aqc_acr_bridge_pilot import parse_step_id, validate_extraction


class BridgePilotRunnerTests(unittest.TestCase):
    def test_parse_step_id(self) -> None:
        self.assertEqual(
            parse_step_id("appendicitis:20123918:s2"),
            ("appendicitis", 20123918, 2),
        )

    def test_validates_exact_evidence(self) -> None:
        parsed = output_contract()
        for dimension in LLM_DIMENSIONS:
            parsed["patient_context"][dimension] = []
        parsed["patient_context"]["symptom_state"] = [{
            "symptom": "pain",
            "site": "right_lower_quadrant",
            "state": "present",
            "evidence": [{
                "source": "history",
                "support": "Right lower quadrant pain",
            }],
        }]
        parsed["other_proposed_dimension"] = []
        validation = validate_extraction(
            parsed,
            {"patient_history": "Right lower quadrant pain.", "physical_examination": ""},
            [],
        )
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertEqual(validation["item_count"], 1)

    def test_rejects_nonverbatim_evidence(self) -> None:
        parsed = output_contract()
        for dimension in LLM_DIMENSIONS:
            parsed["patient_context"][dimension] = []
        parsed["patient_context"]["diagnostic_state"] = [{
            "condition": "appendicitis",
            "status": "suspected",
            "role": "primary_diagnosis",
            "evidence": [{"source": "history", "support": "suspected appendicitis"}],
        }]
        parsed["other_proposed_dimension"] = []
        validation = validate_extraction(
            parsed,
            {"patient_history": "Abdominal pain.", "physical_examination": ""},
            [],
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any("exact source substring" in error for error in validation["errors"]))


if __name__ == "__main__":
    unittest.main()
