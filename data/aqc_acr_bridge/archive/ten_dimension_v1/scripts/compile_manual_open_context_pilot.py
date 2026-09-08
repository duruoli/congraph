#!/usr/bin/env python3
"""Validate and compile the hand-authored 12-step open-context pilot.

The TSV is the auditable manual source.  This script verifies every quoted
span against the blinded input and expands the compact rows into the exact
Stage-1 prompt contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.aqc_acr_bridge.prompts import (
    ALL_DIMENSIONS,
    EPISTEMIC_SOURCES,
    FACTUAL_DIMENSIONS,
    INFERENTIAL_DIMENSIONS,
)


DEFAULT_DIR = ROOT / "data/aqc_acr_bridge/pilot_v1/open_context_manual_v1"
ASSERTION_STATUSES = {
    "affirmed", "negated", "suspected", "established", "challenged",
    "excluded", "equivocal", "unclear",
}
TEMPORALITIES = {"current", "historical", "trajectory", "unclear"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_DIR)
    return parser.parse_args()


def split_refs(value: str) -> list[str]:
    return [part for part in value.split("|") if part]


def evidence_source(record: dict[str, Any], section: str, prior_index: int | None) -> str:
    baseline = record["baseline"]
    if section == "history":
        return baseline["patient_history"]
    if section == "physical_examination":
        return baseline["physical_examination"]
    if section == "laboratory_tests":
        return baseline["laboratory_tests"]
    if section == "prior_imaging":
        if prior_index is None:
            raise ValueError("prior_imaging evidence requires prior_imaging_index")
        return record["visible_prior_imaging"][prior_index - 1]["report"]
    raise ValueError(f"unknown evidence section: {section}")


def main() -> None:
    args = parse_args()
    pilot_dir = args.pilot_dir.resolve()
    inputs = json.loads((pilot_dir / "inputs.json").read_text())
    case_map = json.loads((pilot_dir / "case_map.json").read_text())
    notes = json.loads((pilot_dir / "manual_case_notes_v1.json").read_text())
    records = {row["case_id"]: row["visible_preorder_record"] for row in inputs["cases"]}
    mapping = {row["case_id"]: row["step_id"] for row in case_map["cases"]}

    rows_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    tsv_path = pilot_dir / "manual_context_items_v1.tsv"
    with tsv_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_fields = {
            "case_id", "local_key", "epistemic_kind", "dimension", "value_native",
            "assertion_status", "temporality", "epistemic_source", "section",
            "prior_imaging_index", "quote", "based_on", "applies_to", "logic_note",
            "reasoning",
        }
        if set(reader.fieldnames or []) != expected_fields:
            raise ValueError(f"unexpected TSV fields: {reader.fieldnames}")
        for row in reader:
            rows_by_case[row["case_id"]].append(row)

    if set(rows_by_case) != set(records):
        raise ValueError("manual TSV case IDs do not exactly match blinded inputs")
    if set(notes) != set(records):
        raise ValueError("manual notes case IDs do not exactly match blinded inputs")

    out_rows: list[dict[str, Any]] = []
    total_items = 0
    for case_id in sorted(records):
        source_rows = rows_by_case[case_id]
        key_to_id = {row["local_key"]: f"ctx_{index:02d}" for index, row in enumerate(source_rows, 1)}
        key_to_kind = {row["local_key"]: row["epistemic_kind"] for row in source_rows}
        if len(key_to_id) != len(source_rows):
            raise ValueError(f"duplicate local_key in {case_id}")

        annotation: dict[str, Any] = {
            "schema_version": "1.0.0-open-patient-context",
            "factual_context": {dimension: [] for dimension in FACTUAL_DIMENSIONS},
            "inferential_context": {dimension: [] for dimension in INFERENTIAL_DIMENSIONS},
            "additional_dimension_outside_acr_schema": notes[case_id].get("additional", []),
            "latent_or_unidentifiable": notes[case_id].get("latent", []),
            "extraction_note": notes[case_id].get("extraction_note", ""),
        }
        for row in source_rows:
            kind = row["epistemic_kind"]
            dimension = row["dimension"]
            if dimension not in ALL_DIMENSIONS:
                raise ValueError(f"unknown dimension {dimension} in {case_id}")
            if kind == "factual" and dimension not in FACTUAL_DIMENSIONS:
                raise ValueError(f"kind/dimension mismatch in {case_id}:{row['local_key']}")
            if kind == "inferential" and dimension not in INFERENTIAL_DIMENSIONS:
                raise ValueError(f"kind/dimension mismatch in {case_id}:{row['local_key']}")
            if row["epistemic_source"] not in EPISTEMIC_SOURCES:
                raise ValueError(f"unknown source in {case_id}:{row['local_key']}")
            if row["assertion_status"] not in ASSERTION_STATUSES:
                raise ValueError(f"unknown assertion status in {case_id}:{row['local_key']}")
            if row["temporality"] not in TEMPORALITIES:
                raise ValueError(f"unknown temporality in {case_id}:{row['local_key']}")
            if row["epistemic_source"] in {"deterministic_derivation", "reconstructed_judgment"} and not row["reasoning"]:
                raise ValueError(f"derived/reconstructed item lacks reasoning in {case_id}:{row['local_key']}")

            prior_index = int(row["prior_imaging_index"]) if row["prior_imaging_index"] else None
            source = evidence_source(records[case_id], row["section"], prior_index)
            if row["quote"] not in source:
                raise ValueError(
                    f"non-verbatim quote in {case_id}:{row['local_key']}: {row['quote']!r}"
                )
            for ref in split_refs(row["based_on"]) + split_refs(row["applies_to"]):
                if ref not in key_to_id:
                    raise ValueError(f"unknown item reference {ref} in {case_id}:{row['local_key']}")
            for ref in split_refs(row["based_on"]):
                if key_to_kind[ref] != "factual":
                    raise ValueError(f"based_on must reference a factual item: {case_id}:{row['local_key']} -> {ref}")

            item = {
                "item_id": key_to_id[row["local_key"]],
                "value_native": row["value_native"],
                "assertion_status": row["assertion_status"],
                "temporality": row["temporality"],
                "epistemic_source": row["epistemic_source"],
                "evidence_spans": [{
                    "section": row["section"],
                    "prior_imaging_index": prior_index,
                    "quote": row["quote"],
                }],
                "based_on_item_ids": [key_to_id[x] for x in split_refs(row["based_on"])],
                "applies_to_item_ids": [key_to_id[x] for x in split_refs(row["applies_to"])],
                "logic_note": row["logic_note"] or "atomic",
                "reasoning": row["reasoning"],
            }
            annotation[f"{kind}_context"][dimension].append(item)
            total_items += 1

        out_rows.append({
            "case_id": case_id,
            "step_id": mapping[case_id],
            "annotation_status": "manual_contaminated_calibration_v1",
            "annotation": annotation,
        })

    output_path = pilot_dir / "manual_extractions_v1.jsonl"
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out_rows)
    )
    print(f"PASS: {len(out_rows)} decision steps, {total_items} manual context items")
    print(output_path)


if __name__ == "__main__":
    main()
