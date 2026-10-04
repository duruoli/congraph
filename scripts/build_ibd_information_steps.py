#!/usr/bin/env python3
"""Select 20-patient information-update steps at result/report availability.

These are the primary candidate anchors for time-correct eight-axis state labels.
The separate action-step file remains a record of performed/prescribed actions.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "runs/ibd_event_timeline_pilot_20"
DEFAULT_OUTPUT = ROOT / "runs/ibd_information_steps_pilot_20"
POLICY_VERSION = "ibd-information-step-selection/1.0"

# A deliberately narrow lab trigger. Other lab results are still available as
# context at each later step and remain in availability_checkpoints.jsonl.
LAB_TRIGGER = re.compile(
    r"c.reactive protein|calprotectin|sedimentation rate|albumin|"
    r"\blactate\b(?!\s+dehydrogenase)|cytomegalovirus viral load|cmv viral load", re.I,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def trigger_reason(event: dict[str, Any]) -> str | None:
    source = event["source_table"]
    if source == "cohort" and event["detail"].get("anchor") == "admission":
        return "admission"
    if source == "radiology":
        return "radiology_report_available"
    if source == "microbiology":
        return "microbiology_result_update"
    if source == "labs" and LAB_TRIGGER.search(str(event["detail"].get("label") or "")):
        return "selected_lab_result_available"
    return None


def make_information_steps(
    events: list[dict[str, Any]], checkpoints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {event["event_id"]: event for event in events}
    if len(by_id) != len(events):
        raise ValueError("Duplicate event IDs")
    steps = []
    index: Counter[int] = Counter()
    pending_event_ids: dict[int, list[str]] = defaultdict(list)
    previous_cutoff: dict[int, str] = {}
    for checkpoint in sorted(checkpoints, key=lambda c: (c["subject_id"], c["available_at"], c["step_index"])):
        new_events = [by_id[event_id] for event_id in checkpoint["new_event_ids"]]
        if any(e["available_at"] != checkpoint["available_at"] for e in new_events):
            raise ValueError("Checkpoint availability time disagrees with event ledger")
        subject_id = checkpoint["subject_id"]
        pending_event_ids[subject_id].extend(checkpoint["new_event_ids"])
        triggers = [(event, trigger_reason(event)) for event in new_events]
        triggers = [(event, reason) for event, reason in triggers if reason]
        if not triggers:
            continue
        steps.append({
            "subject_id": subject_id,
            "hadm_ids": checkpoint["hadm_ids"],
            "step_index": index[subject_id],
            "step_kind": "information_available_candidate",
            "evidence_cutoff_at": checkpoint["available_at"],
            "previous_step_cutoff_at": previous_cutoff.get(subject_id),
            "source_checkpoint_index": checkpoint["step_index"],
            "trigger_reasons": sorted({reason for _, reason in triggers}),
            "trigger_event_ids": sorted(event["event_id"] for event, _ in triggers),
            "all_new_available_event_ids": checkpoint["new_event_ids"],
            "new_since_previous_step_event_ids": pending_event_ids[subject_id],
            "cumulative_available_event_count": checkpoint["cumulative_available_event_count"],
            "source_references": [
                {"event_id": event["event_id"], "source_table": event["source_table"],
                 "source_key": event["source_key"], "event_at": event["event_at"],
                 "available_at": event["available_at"], "reason": reason}
                for event, reason in triggers
            ],
        })
        index[subject_id] += 1
        previous_cutoff[subject_id] = checkpoint["available_at"]
        pending_event_ids[subject_id] = []
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    events = read_jsonl(args.input_dir / "events.jsonl")
    checkpoints = read_jsonl(args.input_dir / "availability_checkpoints.jsonl")
    steps = make_information_steps(events, checkpoints)
    patients = {p["subject_id"] for p in manifest["patients"]}
    if {step["subject_id"] for step in steps} != patients:
        raise ValueError("Information steps do not cover all manifest patients")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "information_steps.jsonl").open("w") as handle:
        for step in steps:
            handle.write(json.dumps(step, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "policy_version": POLICY_VERSION,
        "patient_count": len(patients),
        "source_checkpoint_count": len(checkpoints),
        "selected_information_step_count": len(steps),
        "selected_steps_per_patient": dict(sorted(Counter(s["subject_id"] for s in steps).items())),
        "trigger_counts": dict(sorted(Counter(r for s in steps for r in s["trigger_reasons"]).items())),
        "limitations": [
            "The step time is result/report availability, not test order or specimen collection.",
            "Microbiology storetime may be the last update rather than initial result availability.",
            "The narrow lab trigger is provisional; omitted checkpoints remain in the source file for audit.",
            "A result being available does not prove a clinician reviewed it or acted on it.",
            "Prescription and procedure actions require a separate temporal review before combining with strict as-of labels.",
        ],
    }
    (args.output_dir / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
