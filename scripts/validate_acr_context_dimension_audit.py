#!/usr/bin/env python3
"""Validate the 50-row ACR Context-value dimension audit against the corpus."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "acr_normative" / "acr_topics.json"
AUDIT = ROOT / "data" / "aqc_acr_bridge" / "acr_context_value_dimension_audit_v1.csv"

EXPECTED_COLUMNS = [
    "row_id",
    "native_value",
    "source_variant_ids",
    "legacy_dimension",
    "epistemic_kind",
    "dimension",
    "derivation_or_subtype",
    "logical_note",
    "classification_rationale",
    "boundary_note",
]
DIMENSION_KIND = {
    "symptoms": "factual",
    "signs_and_labs": "factual",
    "patient_characteristics": "factual",
    "disease_timing": "factual",
    "prior_test": "factual",
    "encounter_stage": "factual",
    "imaging_stage": "factual",
    "diagnosis": "inferential",
    "severity_or_complication": "inferential",
    "evidence_interpretation": "inferential",
}
EXPECTED_COUNTS = {
    "symptoms": 8,
    "signs_and_labs": 12,
    "patient_characteristics": 1,
    "disease_timing": 4,
    "prior_test": 1,
    "encounter_stage": 1,
    "imaging_stage": 2,
    "diagnosis": 7,
    "severity_or_complication": 8,
    "evidence_interpretation": 6,
}


def corpus_values() -> dict[str, set[str]]:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    uses: dict[str, set[str]] = defaultdict(set)
    for topic in corpus["topics"]:
        for variant in topic["variants"]:
            variant_id = f"acr_{topic['topic_id']}_v{variant['variant_id']}"
            context = variant["context"]
            value_lists = [
                context["clinical_state"]["presentation"],
                context["clinical_state"]["condition"],
                context["clinical_state"]["severity_or_complication"],
                context["imaging_history"]["prior_test"],
                context["imaging_history"]["prior_result"],
                context["modifiers"]["population"],
                context["modifiers"]["timing"],
                context["modifiers"]["constraints_or_confounders"],
                context["decision_stage"]["encounter_status"],
            ]
            stage = context["decision_stage"]["imaging_stage"]
            if stage != "unspecified":
                value_lists.append([stage])
            for values in value_lists:
                for value in values:
                    uses[value].add(variant_id)
    return uses


def main() -> None:
    with AUDIT.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == EXPECTED_COLUMNS, reader.fieldnames
        rows = list(reader)

    assert len(rows) == 50, len(rows)
    assert len({row["row_id"] for row in rows}) == 50
    assert len({row["native_value"] for row in rows}) == 50
    assert all(row["logical_note"] for row in rows)
    assert all(row["classification_rationale"] for row in rows)
    assert all(row["boundary_note"] for row in rows)

    uses = corpus_values()
    audited_values = {row["native_value"] for row in rows}
    assert audited_values == set(uses), {
        "missing_from_audit": sorted(set(uses) - audited_values),
        "not_in_corpus": sorted(audited_values - set(uses)),
    }

    for row in rows:
        dimension = row["dimension"]
        assert dimension in DIMENSION_KIND, dimension
        assert row["epistemic_kind"] == DIMENSION_KIND[dimension], row
        audited_sources = set(row["source_variant_ids"].split(";"))
        assert audited_sources == uses[row["native_value"]], {
            "value": row["native_value"],
            "audit": sorted(audited_sources),
            "corpus": sorted(uses[row["native_value"]]),
        }

    counts = Counter(row["dimension"] for row in rows)
    assert counts == Counter(EXPECTED_COUNTS), counts
    kind_counts = Counter(row["epistemic_kind"] for row in rows)
    assert kind_counts == {"factual": 29, "inferential": 21}, kind_counts
    print("PASS: 50 ACR Context values; 29 factual and 21 inferential across 10 dimensions.")


if __name__ == "__main__":
    main()
