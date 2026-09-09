from __future__ import annotations

import json
import unittest

from experiments.aqc_acr_bridge.algorithmic_predicates import (
    extract_algorithmic_predicates,
    extract_labelled_vital_predicates,
    extract_lab_predicates,
    extract_test_history,
    mask_labelled_vitals,
)


class AlgorithmicPredicateTests(unittest.TestCase):
    def test_labs_use_reference_ranges_and_keep_specimen_provenance(self) -> None:
        values = json.dumps({
            "50956": "5502.0 IU/L",
            "50931": "181.0 mg/dL",
            "51478": "NEG.",
            "hold": "HOLD. DISCARD GREATER THAN 4 HOURS OLD.",
        })
        lower = json.dumps({"50956": 0.0, "50931": 70.0})
        upper = json.dumps({"50956": 60.0, "50931": 100.0})
        metadata = {
            "50956": {"label": "Lipase", "fluid": "Blood", "category": "Chemistry"},
            "50931": {"label": "Glucose", "fluid": "Blood", "category": "Chemistry"},
            "51478": {"label": "Glucose", "fluid": "Urine", "category": "Hematology"},
        }

        predicates = extract_lab_predicates(values, lower, upper, metadata)

        self.assertEqual(len(predicates), 3)
        by_id = {item["id"]: item for item in predicates}
        self.assertEqual(by_id["alg_lab_50956"]["analyte"], "lipase")
        self.assertEqual(by_id["alg_lab_50956"]["state"], "high")
        self.assertEqual(by_id["alg_lab_50931"]["state"], "high")
        self.assertEqual(by_id["alg_lab_51478"]["state"], "negative")
        self.assertEqual(by_id["alg_lab_50931"]["evidence"][0]["fluid"], "Blood")
        self.assertEqual(by_id["alg_lab_51478"]["evidence"][0]["fluid"], "Urine")
        self.assertNotIn("temporality", by_id["alg_lab_50956"])
        self.assertNotIn("change", by_id["alg_lab_50956"])

    def test_vitals_require_explicit_labels_and_take_last_match(self) -> None:
        labelled = (
            "Vitals: T99.4 F, HR 101, BP 102/43, RR 17, POx 94% 5L NC. "
            "Later HR: 88."
        )
        predicates = extract_labelled_vital_predicates(labelled)
        by_sign = {item["sign"]: item for item in predicates}

        self.assertEqual(by_sign["temperature"]["state"], "99.4_F")
        self.assertEqual(by_sign["heart_rate"]["state"], "88_bpm")
        self.assertEqual(by_sign["blood_pressure"]["state"], "102/43_mmHg")
        self.assertEqual(by_sign["respiratory_rate"]["state"], "17_per_min")
        self.assertEqual(by_sign["oxygen_saturation"]["state"], "94_percent")
        self.assertEqual(
            extract_labelled_vital_predicates("Vitals: 98.5 67 101/70 18 97%"),
            [],
        )
        masked = mask_labelled_vitals(labelled)
        self.assertNotIn("HR 101", masked)
        self.assertNotIn("HR: 88", masked)
        self.assertIn("[captured_vital]", masked)

    def test_test_history_keeps_latest_study_per_test(self) -> None:
        priors = [
            {
                "modality": "Radiograph",
                "region": "Chest",
                "exam": "Chest PA and lateral",
                "role": "context",
            },
            {
                "modality": "Ultrasound",
                "region": "Abdomen",
                "exam": "US complete",
                "role": "decision",
            },
            {
                "modality": "CT",
                "region": "Abdomen",
                "exam": "CT with contrast",
                "role": "decision",
            },
            {
                "modality": "Ultrasound",
                "region": "Abdomen",
                "exam": "US limited",
                "role": "decision",
            },
        ]

        history = extract_test_history(priors)

        self.assertEqual(
            [item["test"] for item in history],
            ["radiograph", "ct", "ultrasound"],
        )
        self.assertEqual(history[-1]["id"], "alg_test_ultrasound")
        self.assertEqual(
            history[-1]["status"],
            "completed_before_current_decision",
        )
        self.assertEqual(history[-1]["evidence"][0]["prior_imaging_index"], 4)
    def test_context_imaging_remains_in_test_history(self) -> None:
        context = [{
            "modality": "Radiograph",
            "region": "Chest",
            "exam": "Chest PA and lateral",
            "role": "context",
        }]
        self.assertEqual(extract_test_history(context)[0]["test"], "radiograph")

    def test_combined_output_contains_only_algorithmic_predicate_types(self) -> None:
        row = {
            "Laboratory Tests": json.dumps({"51301": "13.3 K/uL"}),
            "Reference Range Lower": json.dumps({"51301": 4.0}),
            "Reference Range Upper": json.dumps({"51301": 10.0}),
            "Physical Examination": "Temp: 98.6 HR: 72",
        }
        metadata = {
            "51301": {
                "label": "White Blood Cells",
                "fluid": "Blood",
                "category": "Hematology",
            }
        }
        decision = {"visible_prior_imaging": []}

        output = extract_algorithmic_predicates(row, decision, metadata)

        self.assertEqual(
            set(output),
            {
                "schema_version",
                "lab_finding_state",
                "sign_state",
                "test_history",
            },
        )
        self.assertEqual(
            output["lab_finding_state"][0]["analyte"],
            "white_blood_cell_count",
        )


if __name__ == "__main__":
    unittest.main()
