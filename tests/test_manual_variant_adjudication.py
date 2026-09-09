from __future__ import annotations

import unittest

from scripts.adjudicate_manual_patient_context_variants import (
    ACR_AUDIT,
    DEFAULT_INPUT,
    adjudicate_file,
    read_json,
    validate_document,
)


class ManualVariantAdjudicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.acr = read_json(ACR_AUDIT)

    def document(self, name: str):
        return adjudicate_file(DEFAULT_INPUT / name, self.acr)

    def test_every_case_has_all_variants_and_predicates(self) -> None:
        for path in sorted(DEFAULT_INPUT.glob("*.json")):
            document = adjudicate_file(path, self.acr)
            validate_document(document)
            self.assertEqual(len(document["variants"]), 17)
            self.assertEqual(
                sum(len(row["predicate_matrix"]) for row in document["variants"]), 78
            )

    def test_direct_support_and_missingness_are_distinct(self) -> None:
        document = self.document("appendicitis_20123918_s2.json")
        variants = {row["variant_key"]: row for row in document["variants"]}
        v3 = {row["condition_id"]: row for row in variants["acr_21_v3"]["predicate_matrix"]}
        self.assertEqual(v3["c01"]["status"], "supported")
        self.assertEqual(v3["c02"]["status"], "supported")
        self.assertEqual(v3["c03"]["status"], "unknown")
        self.assertEqual(v3["c04"]["status"], "unknown")
        self.assertEqual(v3["c05"]["status"], "unknown")

    def test_unlinked_temporal_comparison_is_manually_adjudicated(self) -> None:
        document = self.document("pancreatitis_20720063_s3.json")
        variants = {row["variant_key"]: row for row in document["variants"]}
        v3 = {row["condition_id"]: row for row in variants["acr_126_v3"]["predicate_matrix"]}
        self.assertEqual(v3["c08"]["status"], "contradicted")
        self.assertEqual(v3["c08"]["adjudication"]["method"], "manual_evidence_review")

    def test_crohn_history_does_not_support_an_active_alternative_role(self) -> None:
        document = self.document("appendicitis_20123918_s2.json")
        variant = next(row for row in document["variants"] if row["variant_key"] == "acr_126_v2")
        c06 = next(row for row in variant["predicate_matrix"] if row["condition_id"] == "c06")
        self.assertEqual(c06["status"], "unknown")
        self.assertEqual(c06["adjudication"]["method"], "manual_evidence_review")

    def test_cross_topic_exact_matches_are_retained(self) -> None:
        document = self.document("cholecystitis_20334898_s2.json")
        self.assertEqual(document["reviewed_correspondence"]["label"], "multiple")
        candidates = {row["variant_key"]: row for row in document["candidate_variants"]}
        self.assertEqual(candidates["acr_132_v2"]["signature_evaluation"]["overall_signature_status"], "satisfied")
        self.assertEqual(candidates["acr_21_v1"]["signature_evaluation"]["overall_signature_status"], "satisfied")

    def test_variant_six_group_preserves_one_or_more_logic(self) -> None:
        document = self.document("pancreatitis_20720063_s3.json")
        variant = next(row for row in document["variants"] if row["variant_key"] == "acr_126_v6")
        self.assertEqual(variant["signature_evaluation"]["computed_groups"][0]["group_id"], "g01")
        self.assertEqual(variant["signature_evaluation"]["computed_groups"][0]["status"], "supported")
        self.assertEqual(variant["signature_evaluation"]["overall_signature_status"], "contradicted")

    def test_painless_cholecystitis_is_out_of_scope(self) -> None:
        document = self.document("cholecystitis_20660601_s2.json")
        self.assertEqual(document["reviewed_correspondence"]["label"], "out_of_scope")
        ruq = [row for row in document["variants"] if row["topic"] == "Right Upper Quadrant Pain"]
        self.assertTrue(all(row["signature_evaluation"]["overall_signature_status"] == "contradicted" for row in ruq))


if __name__ == "__main__":
    unittest.main()
