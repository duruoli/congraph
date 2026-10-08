#!/usr/bin/env python3
"""Build local, reviewable MIMIC IBD action-label prompts; never call an LLM.

Default: radiology actions in the 20-patient action-step pilot. Output stays
under gitignored runs/. This is the first, action-time scene view; the later
retrospective-at-action phase reconstruction is a separate proposed pass.

Run: /opt/anaconda3/bin/python3.12 scripts/build_ibd_action_label_inputs.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
import yaml


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw_data/ibd_mimiciv_3_1"
DEFAULT_LEDGER = ROOT / "runs/ibd_event_timeline_pilot_20/events.jsonl"
DEFAULT_STEPS = ROOT / "runs/ibd_action_steps_pilot_20/action_steps.jsonl"
DEFAULT_OUTPUT = ROOT / "runs/ibd_action_label_inputs_pilot_20/radiology_scene_prompts.jsonl"
DEFINITIONS = ROOT / "IBD_MIMIC_ACTION_LABEL_PROMPT_DEFINITIONS_DRAFT.md"
VOCABULARY = ROOT / "unified_axes_v1.yml"
SECTION = re.compile(r"^\s*([A-Z][A-Z /\-]{2,40})\s*:\s*(.*)$")
LAB_RE = re.compile(
    r"c.reactive protein|calprotectin|sedimentation rate|albumin|lactate|"
    r"hemoglobin|white blood cell", re.I,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def report_section(note: str, names: tuple[str, ...], limit: int) -> dict[str, Any] | None:
    """Extract a named report section, without falling back to full report text."""
    sections: dict[str, list[str]] = defaultdict(list)
    current: str | None = None
    for line in note.splitlines():
        match = SECTION.match(line)
        if match:
            current = match.group(1).strip().upper()
            line = match.group(2)
        if current:
            sections[current].append(line.strip())
    for name in names:
        body = " ".join(" ".join(sections.get(name, [])).split())
        if body:
            return {"section": name, "text": body[:limit], "truncated": len(body) > limit}
    return None


def raw_action(action: dict[str, Any]) -> dict[str, Any]:
    source = action["source_table"]
    detail = action["detail"]
    fields = {
        "radiology": ("exam_name", "exam_code"),
        "labs": ("label", "itemid"),
        "microbiology": ("test_name", "specimen"),
        "prescriptions": ("drug", "route", "prescribed_stoptime"),
        "procedures": ("long_title", "icd_code", "icd_version"),
    }.get(source)
    if fields is None:
        raise ValueError(f"Unsupported target source: {source}")
    status = {
        "radiology": "exam_recorded_not_order_time",
        "labs": "specimen_collected_not_result_time",
        "microbiology": "specimen_collected_not_result_time",
        "prescriptions": "prescribed_start_not_administration",
        "procedures": "billed_procedure_date_only",
    }[source]
    main_field = {
        "radiology": "exam_name", "labs": "label", "microbiology": "test_name",
        "prescriptions": "drug", "procedures": "long_title",
    }[source]
    return {
        "event_id": action["event_id"], "source_table": source,
        "action_kind": action["action_kind"], "action_status": status,
        "action_text": detail.get(main_field),
        "raw_detail": {key: detail.get(key) for key in fields},
    }


def axis_values() -> dict[str, list[str]]:
    axes = yaml.safe_load(VOCABULARY.read_text(encoding="utf-8"))["axes"]
    return {axis["id"]: [value["id"] for value in axis["values"]] for axis in axes}


def load_source_rows(subject_hadm_ids: list[int]) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    predicate = ds.field("hadm_id").isin(subject_hadm_ids)
    reports = ds.dataset(RAW / "radiology", format="parquet").to_table(
        columns=["note_id", "text"], filter=predicate,
    ).to_pylist()
    labs = ds.dataset(RAW / "labs", format="parquet").to_table(
        columns=["labevent_id", "value", "valueuom"], filter=predicate,
    ).to_pylist()
    return (
        {str(row["note_id"]): row["text"] or "" for row in reports},
        {str(row["labevent_id"]): {"value": row["value"], "unit": row["valueuom"]} for row in labs},
    )


def strictly_earlier(event: dict[str, Any], action_at: str) -> bool:
    if event.get("time_precision") == "date":
        return bool(event.get("event_end_exclusive") and event["event_end_exclusive"] <= action_at)
    return bool(event.get("event_at") and event["event_at"] < action_at)


def make_packet(
    step: dict[str, Any], action: dict[str, Any], patient_events: list[dict[str, Any]],
    cohort: dict[str, Any], notes: dict[str, str], lab_values: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    at = step["event_at"]
    if at is None:
        raise ValueError("Actions without any time cannot establish a scene evidence cutoff")
    current_note = notes.get(action["source_key"], "") if action["source_table"] == "radiology" else ""
    indication = report_section(current_note, ("INDICATION", "CLINICAL INFORMATION", "HISTORY"), 1200)
    prior_reports = []
    prior_labs = []
    prior_actions = []
    for event in patient_events:
        source = event["source_table"]
        available = event.get("available_at")
        if source == "radiology" and available and event.get("availability_precision") == "timestamp" and available < at:
            note = notes.get(event["source_key"], "")
            prior_reports.append({
                "event_id": event["event_id"], "available_at": available,
                "exam_name": event["detail"].get("exam_name"),
                "indication": report_section(note, ("INDICATION", "CLINICAL INFORMATION", "HISTORY"), 400),
                "impression": report_section(note, ("IMPRESSION", "CONCLUSION"), 900),
            })
        elif source == "labs" and available and event.get("availability_precision") == "timestamp" and available < at:
            label = event["detail"].get("label") or ""
            if LAB_RE.search(label):
                prior_labs.append({
                    "event_id": event["event_id"], "available_at": available, "label": label,
                    **lab_values.get(event["source_key"], {"value": None, "unit": None}),
                })
        elif source in {"prescriptions", "procedures"} and strictly_earlier(event, at):
            prior_actions.append({
                "event_id": event["event_id"], "event_at": event["event_at"],
                "time_precision": event["time_precision"], "source_table": source,
                "action_status": "prescribed_start_not_administration" if source == "prescriptions" else "billed_procedure_date_only",
                "detail": {key: event["detail"].get(key) for key in
                           (("drug", "route") if source == "prescriptions" else ("long_title",))},
            })
    prior_reports.sort(key=lambda x: (x["available_at"], x["event_id"]))
    prior_labs.sort(key=lambda x: (x["available_at"], x["event_id"]))
    prior_actions.sort(key=lambda x: (x["event_at"], x["event_id"]))
    limits = {"prior_available_reports": 4, "prior_available_selected_labs": 10, "prior_actions": 8}
    all_context = {
        "prior_available_reports": prior_reports,
        "prior_available_selected_labs": prior_labs,
        "prior_actions": prior_actions,
    }
    context = {key: values[-limits[key]:] for key, values in all_context.items()}
    omitted = {key: len(values) - len(context[key]) for key, values in all_context.items()}
    other_actions = [raw_action(other) for other in step["actions"] if other["event_id"] != action["event_id"]]
    other_actions.sort(key=lambda item: (item["source_table"] == "labs", item["source_table"], item["event_id"]))
    admit, discharge = cohort.get("admittime"), cohort.get("dischtime")
    action_datetime = datetime.fromisoformat(at)
    within_admission = bool(admit and discharge and admit <= action_datetime < discharge)
    input_data = {
        "record_id": f"{step['action_group_id']}:{action['event_id']}",
        "action_group_id": step["action_group_id"], "action_at": at,
        "action_time_precision": step["time_precision"],
        "action_end_exclusive": step.get("event_end_exclusive"),
        "evidence_cutoff_rule": "strictly_before_action_date" if step["time_precision"] == "date" else "strictly_before_recorded_action_time",
        "patient_background": {
            "age": cohort.get("age"), "gender": cohort.get("gender"),
            "admittime": str(admit) if admit else None,
            "within_recorded_admission_interval": within_admission,
            "cohort_ibd_type_retrospective": cohort.get("ibd_type"),
        },
        "target_action": raw_action(action),
        "target_report_indication": indication,
        "target_report_indication_status": "later_report_proxy_not_proven_order_time" if indication else "not_available",
        "other_actions_in_same_recorded_group": other_actions[:8],
        "other_actions_omitted_count": max(0, len(other_actions) - 8),
        **context,
        "context_omitted_older_counts": omitted,
        "known_missing_inputs": [
            "No target report findings/impression in this scene pass.",
            "No full clinical progress notes or exact order times in current extract.",
            "No microbiology result values in this first input builder.",
        ],
    }
    return input_data


def format_prompt(definitions: str, values: dict[str, list[str]], packet: dict[str, Any]) -> str:
    output_shape = {
        "record_id": packet["record_id"],
        "action_type": "observation | intervention",
        "secondary_action_type": "observation | intervention | null",
        "requires_review": False,
        "action_type_basis": "brief evidence-based reason",
        "clinical_question": "specific question or null",
        "axes": {axis: {"status": "mapped | n_a | unknown | unmapped", "values": [
            {"id": "one YAML value", "assertion": "documented | inferred | suspected | historical | retrospective",
             "evidence_ids": ["event ID"], "basis": "brief reason"}
        ], "raw_concept": None} for axis in values},
    }
    return (
        definitions.strip() + "\n\n## 运行时 YAML 允许值\n"
        + json.dumps(values, ensure_ascii=False, indent=2)
        + "\n\n## 本次输入\n" + json.dumps(packet, ensure_ascii=False, indent=2, default=str)
        + "\n\n## 输出 JSON 结构示意\n" + json.dumps(output_shape, ensure_ascii=False, indent=2)
        + "\n只返回 JSON；不要复制结构示意中的占位值。\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--steps", type=Path, default=DEFAULT_STEPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-table", choices=["radiology", "labs", "microbiology", "prescriptions", "procedures"], default="radiology")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if ROOT / "runs" not in args.output.resolve().parents:
        parser.error("Prompt output with patient context must stay under gitignored runs/")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    events = read_jsonl(args.ledger)
    steps = read_jsonl(args.steps)
    selected = [(step, action) for step in steps for action in step["actions"]
                if action["source_table"] == args.source_table and step["event_at"] is not None]
    if args.limit is not None:
        selected = selected[:args.limit]
    subject_ids = {step["subject_id"] for step, _ in selected}
    by_subject: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event["subject_id"] in subject_ids:
            by_subject[event["subject_id"]].append(event)
    hadm_by_subject: dict[int, list[int]] = defaultdict(list)
    for event in events:
        if event["subject_id"] in subject_ids and event["hadm_id"] not in hadm_by_subject[event["subject_id"]]:
            hadm_by_subject[event["subject_id"]].append(event["hadm_id"])
    all_hadm = sorted({hadm for ids in hadm_by_subject.values() for hadm in ids})
    cohort_rows = ds.dataset(RAW / "cohort", format="parquet").to_table(
        columns=["hadm_id", "age", "gender", "admittime", "dischtime", "ibd_type"],
        filter=ds.field("hadm_id").isin(all_hadm),
    ).to_pylist() if all_hadm else []
    cohort = {row["hadm_id"]: row for row in cohort_rows}
    caches = {sid: load_source_rows(hadm_ids) for sid, hadm_ids in hadm_by_subject.items()}
    definitions = DEFINITIONS.read_text(encoding="utf-8")
    values = axis_values()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for step, action in selected:
            notes, lab_values = caches[step["subject_id"]]
            packet = make_packet(step, action, by_subject[step["subject_id"]],
                                 cohort[step["hadm_id"]], notes, lab_values)
            row = {"record_id": packet["record_id"], "input": packet,
                   "prompt": format_prompt(definitions, values, packet)}
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"output": str(args.output), "source_table": args.source_table,
                      "prompt_count": len(selected), "patient_count": len(subject_ids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
