from __future__ import annotations

import unittest

from experiments.aqc_acr_bridge.prompts import (
    LLM_DIMENSIONS,
    SYSTEM,
    build_user,
    output_contract,
)


class PatientContextPromptTests(unittest.TestCase):
    def test_contract_uses_predicate_specific_arguments(self) -> None:
        contract = output_contract()
        context = contract["patient_context"]

        self.assertEqual(list(context), LLM_DIMENSIONS)
        self.assertEqual(
            set(context["symptom_state"][0]),
            {"symptom", "site", "state", "evidence"},
        )
        self.assertEqual(
            set(context["aggregate_assessment"][0]),
            {"assessment", "target", "state", "evidence"},
        )
        serialized = str(contract)
        for removed_field in (
            "temporality",
            "assertion_status",
            "epistemic_source",
            "logic_note",
            "reasoning",
        ):
            self.assertNotIn(removed_field, serialized)

    def test_all_non_stage_dimensions_remain_open_to_llm(self) -> None:
        context = output_contract()["patient_context"]
        self.assertIn("lab_finding_state", context)
        self.assertIn("test_history", context)
        self.assertNotIn("imaging_stage", context)
        self.assertIn("sign_state", context)
        self.assertIn("not documented is unknown", SYSTEM)

    def test_user_prompt_omits_algorithmic_inputs_and_is_order_blinded(self) -> None:
        baseline = {
            "patient_history": "Right lower quadrant pain.",
            "physical_examination": "HR 101. Tender in the right lower quadrant.",
            "laboratory_tests": "White Blood Cells: 13.3 K/uL",
        }
        prior = [{
            "modality": "Radiograph",
            "region": "Chest",
            "exam": "Chest PA and lateral",
            "role": "context",
            "report": "No focal consolidation.",
        }]
        prompt = build_user(baseline, prior)

        self.assertIn("Right lower quadrant pain.", prompt)
        self.assertIn("No focal consolidation.", prompt)
        self.assertNotIn("role=context", prompt)
        self.assertIn("[captured_vital]", prompt)
        self.assertNotIn("alg_lab_51301", prompt)
        self.assertNotIn("HR 101", prompt)
        self.assertNotIn("White Blood Cells: 13.3 K/uL", prompt)
        self.assertNotIn("masked_result_of_this_test", prompt)


if __name__ == "__main__":
    unittest.main()
