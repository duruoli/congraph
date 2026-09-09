from __future__ import annotations

import unittest

from scripts.map_manual_patient_context_to_acr import load_registry, semantic_links


class ManualPatientContextToAcrMappingTests(unittest.TestCase):
    def test_compiled_registry_has_78_instances(self) -> None:
        self.assertEqual(len(load_registry()), 78)

    def test_pregnancy_maps_to_exact_instance_with_narrower_detail(self) -> None:
        links = semantic_links(
            "appendicitis:20123918:s2",
            "patient_attribute",
            {"attribute": "pregnancy", "state": "g1p0_at_23_weeks_5_days"},
        )
        self.assertEqual(
            [(row["acr_predicate_instance_id"], row["relation"]) for row in links],
            [("acr_21_v3:c01", "patient_value_narrower")],
        )

    def test_established_pancreatitis_preserves_certainty_relations(self) -> None:
        links = semantic_links(
            "pancreatitis:25133113:s6",
            "diagnostic_state",
            {
                "condition": "acute_pancreatitis",
                "status": "established",
                "role": "primary_diagnosis",
            },
        )
        relations = {
            row["acr_predicate_instance_id"]: row["relation"] for row in links
        }
        self.assertEqual(relations["acr_126_v1:c01"], "patient_value_narrower")
        self.assertEqual(relations["acr_126_v3:c01"], "exact_or_equivalent")
        self.assertEqual(relations["acr_126_v5:c01"], "patient_value_broader")

    def test_absent_collection_contradicts_known_collection(self) -> None:
        links = semantic_links(
            "pancreatitis:25133113:s6",
            "imaging_finding_state",
            {
                "finding": "pancreatic_or_peripancreatic_collection",
                "site": "pancreas",
                "state": "absent",
            },
        )
        self.assertEqual(
            [(row["acr_predicate_instance_id"], row["relation"]) for row in links],
            [("acr_126_v6:c02", "contradicted")],
        )

    def test_unrepresented_item_has_no_semantic_link(self) -> None:
        self.assertEqual(
            semantic_links(
                "appendicitis:20123918:s2",
                "symptom_state",
                {"symptom": "dysuria", "site": "urinary_tract", "state": "absent"},
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
