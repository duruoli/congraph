#!/usr/bin/env python3
"""Prepare anonymous, order-blinded inputs for the 12-step bridge pilot."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_masked_view import RAW, build_record, load_lab_map  # noqa: E402


MANIFEST = ROOT / "data" / "aqc_acr_bridge" / "pilot_v1" / "sample_manifest.json"
OUT = ROOT / "data" / "aqc_acr_bridge" / "pilot_v1" / "open_context_manual_v1"
SALT = "congraph-open-context-manual-pilot-v1"


def parse_step_id(step_id: str) -> tuple[str, int, int]:
    disease, hadm_text, step_text = step_id.split(":")
    return disease, int(hadm_text), int(step_text.removeprefix("s"))


def anonymous_id(step_id: str) -> str:
    digest = hashlib.sha256(f"{SALT}|{step_id}".encode()).hexdigest()[:10]
    return f"openctx_{digest}"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    labmap = load_lab_map()
    frames: dict[str, pd.DataFrame] = {}
    prepared = []
    case_map = []

    for selected in manifest["decision_steps"]:
        step_id = selected["step_id"]
        disease, hadm_id, step_index = parse_step_id(step_id)
        if disease not in frames:
            frames[disease] = pd.read_csv(ROOT / RAW[disease])
        rows = frames[disease][frames[disease]["hadm_id"] == hadm_id]
        assert len(rows) == 1, step_id
        record = build_record(disease, hadm_id, rows.iloc[0], labmap)
        decision_point = next(
            item for item in record["decision_points"] if int(item["step"]) == step_index
        )
        case_id = anonymous_id(step_id)
        prepared.append({
            "case_id": case_id,
            "visible_preorder_record": {
                "baseline": record["baseline"],
                "visible_prior_imaging": decision_point["visible_prior_imaging"],
            },
        })
        case_map.append({"case_id": case_id, "step_id": step_id})

    prepared.sort(key=lambda item: item["case_id"])
    case_map.sort(key=lambda item: item["case_id"])
    assert len(prepared) == len(case_map) == 12
    assert len({item["case_id"] for item in prepared}) == 12

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inputs.json").write_text(
        json.dumps({
            "schema_version": "1.0.0-open-context-manual-input",
            "status": "contaminated_method_calibration_only",
            "blinding": {
                "included": ["baseline", "resulted_prior_imaging"],
                "excluded": [
                    "disease_sampling_label",
                    "selection_tags",
                    "current_order",
                    "current_result",
                    "later_events",
                    "A/Q/C",
                    "ACR_context_values_variants_actions_and_ratings",
                ],
            },
            "cases": prepared,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "case_map.json").write_text(
        json.dumps({
            "schema_version": "1.0.0-open-context-case-map",
            "warning": "Do not expose this mapping during blinded annotation.",
            "cases": case_map,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote 12 anonymous inputs to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
