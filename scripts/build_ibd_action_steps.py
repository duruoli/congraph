#!/usr/bin/env python3
"""Select reproducible action-step candidates from an existing IBD event ledger.

This is an action timeline, not an eight-axis annotation or a reconstruction of
clinician decisions. It does not infer orders from later reports. Run with:

    python3 scripts/build_ibd_action_steps.py

The default input is the fixed 20-patient pilot. The outputs under runs/ contain
no report text or lab values. The complete candidate file allows auditing every
action omitted from the selected step file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "runs/ibd_event_timeline_pilot_20"
DEFAULT_OUTPUT = ROOT / "runs/ibd_action_steps_pilot_20"
POLICY_VERSION = "ibd-action-step-selection/1.0"

# The narrow patterns select IBD/GI-oriented candidates. All other actions
# remain in candidate_actions.jsonl for review. Lab collections are all selected:
# the event ledger lacks order and specimen IDs needed for a narrower safe rule.
RX_PATTERNS = {
    "steroid_or_aminosalicylate_rx": r"mesalamine|sulfasalazine|balsalazide|olsalazine|predniso|methylpred|hydrocortisone|budesonide|dexamethasone",
    "biologic_or_immunomodulator_rx": r"infliximab|adalimumab|vedolizumab|ustekinumab|risankizumab|mirikizumab|guselkumab|golimumab|tofacitinib|upadacitinib|ozanimod|etrasimod|azathioprine|mercaptopurine|methotrexate|cyclosporin|tacrolimus",
    "antimicrobial_rx": r"metronidazole|ciprofloxacin|vancomycin|fidaxomicin|piperacillin.tazobactam|meropenem|ceftriaxone|fluconazole|ganciclovir",
    "nutrition_rx": r"parenteral nutrition|intralipid|fat emulsion|amino acid|tube feed|enteral nutrition",
}
IMAGING_PATTERN = re.compile(r"abd|pelvi|bowel|enterograph|colon|ileum|liver|gallbladder|barium|gi tract|gastro", re.I)
MICRO_PATTERN = re.compile(r"stool|fecal|clostrid|c\. difficile|campylobacter|yersinia|vibrio|e\.coli|giardia|cryptosporid|ova|parasite|cmv|cytomegalovirus", re.I)
PROCEDURE_PATTERN = re.compile(
    r"intestin|ileum|ileostom|col|rect|stoma|bowel|fistula|periton|abdomen|"
    r"endoscop|esophagogastroduodenoscop|nutritional|enteral|parenteral|"
    r"transfusion|drainage of peritoneal|colectomy|gastro", re.I,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def source_action(event: dict[str, Any]) -> tuple[str, str, bool] | None:
    source = event["source_table"]
    detail = event["detail"]
    if source == "cohort":
        if detail.get("anchor") == "admission":
            return "admission", "care_transition", True
        return None
    if source == "labs":
        return "specimen_collection", "lab_collection", True
    if source == "microbiology":
        description = " ".join(str(detail.get(k) or "") for k in ("test_name", "specimen"))
        selected = bool(MICRO_PATTERN.search(description))
        return "specimen_collection", "gi_microbiology" if selected else "other_microbiology", selected
    if source == "radiology":
        selected = bool(IMAGING_PATTERN.search(str(detail.get("exam_name") or "")))
        return "imaging_exam", "gi_imaging" if selected else "other_imaging", selected
    if source == "prescriptions":
        drug = str(detail.get("drug") or "")
        for category, pattern in RX_PATTERNS.items():
            if re.search(pattern, drug, re.I):
                return "prescribed_start", category, True
        return "prescribed_start", "other_prescription", False
    if source == "procedures":
        selected = bool(PROCEDURE_PATTERN.search(str(detail.get("long_title") or "")))
        return "billed_procedure_date", "gi_or_supportive_procedure" if selected else "other_procedure", selected
    return None


def action_key(event: dict[str, Any]) -> tuple[int, int, str, str]:
    # Same-time actions across sources form one step. Date-only rows stay in a
    # separate interval group and are never treated as preceding timed rows.
    return (event["subject_id"], event["hadm_id"], event["event_at"] or "unknown", event["time_precision"])


def make_steps(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if source_action(event) is not None:
            groups[action_key(event)].append(event)

    candidates: list[dict[str, Any]] = []
    for (subject_id, hadm_id, when, precision), members in groups.items():
        members.sort(key=lambda e: (e["source_table"], e["source_key"], e["event_id"]))
        classifications = [source_action(e) for e in members]
        assert all(c is not None for c in classifications)
        selected = any(c[2] for c in classifications if c is not None)
        keys = sorted(e["event_id"] for e in members)
        group_id = hashlib.sha256("|".join(keys).encode()).hexdigest()[:20]
        # An event's time of occurrence is not automatically its time of
        # documentation/availability. The latter is explicitly recorded here.
        candidate = {
            "action_group_id": group_id,
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "event_at": None if when == "unknown" else when,
            "event_end_exclusive": next((e["event_end_exclusive"] for e in members if e["event_end_exclusive"]), None),
            "time_precision": precision,
            "selected": selected,
            "selection_categories": sorted({c[1] for c in classifications if c and c[2]}),
            "action_kinds": sorted({c[0] for c in classifications if c}),
            "source_tables": sorted({e["source_table"] for e in members}),
            "source_event_ids": keys,
            "actions": [
                {
                    "event_id": e["event_id"],
                    "source_table": e["source_table"],
                    "source_key": e["source_key"],
                    "action_kind": classification[0],
                    "category": classification[1],
                    "selected_by_rule": classification[2],
                    "available_at": e["available_at"],
                    "availability_precision": e["availability_precision"],
                    "detail": e["detail"],
                }
                for e, classification in zip(members, classifications)
                if classification is not None
            ],
            "order_certainty": "date_interval_only" if precision == "date" else
                "unknown" if precision == "unknown" else "timestamp_observed",
            "strict_asof_action_evidence": all(
                e["available_at"] is not None and e["available_at"] <= when
                and e["availability_precision"] == "timestamp"
                for e in members if source_action(e)[2]
            ) if precision == "timestamp" else False,
        }
        candidates.append(candidate)

    candidates.sort(key=lambda c: (
        c["subject_id"], c["event_at"] or "9999-12-31T00:00:00",
        0 if c["time_precision"] == "date" else 1,
        c["hadm_id"], c["action_group_id"],
    ))
    selected_steps = []
    indices: Counter[int] = Counter()
    for candidate in candidates:
        if candidate["selected"]:
            step = dict(candidate)
            step["step_index"] = indices[step["subject_id"]]
            step["step_kind"] = "recorded_action_candidate_not_clinician_decision"
            indices[step["subject_id"]] += 1
            selected_steps.append(step)
    return candidates, selected_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    events = read_jsonl(args.input_dir / "events.jsonl")
    expected_subjects = {int(p["subject_id"]) for p in manifest["patients"]}
    if {e["subject_id"] for e in events} != expected_subjects:
        raise ValueError("Event subjects do not match the input manifest")
    candidates, selected = make_steps(events)
    selected_ids = {s["action_group_id"] for s in selected}
    if len(selected_ids) != len(selected):
        raise AssertionError("Duplicate selected action group IDs")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "candidate_actions.jsonl", candidates)
    write_jsonl(output / "action_steps.jsonl", selected)
    counts = Counter(category for row in selected for category in row["selection_categories"])
    report = {
        "policy_version": POLICY_VERSION,
        "input_manifest": str(args.input_dir / "manifest.json"),
        "patient_count": len(expected_subjects),
        "candidate_group_count": len(candidates),
        "selected_step_count": len(selected),
        "selected_steps_per_patient": dict(sorted(Counter(s["subject_id"] for s in selected).items())),
        "selected_categories": dict(sorted(counts.items())),
        "date_only_selected_count": sum(s["time_precision"] == "date" for s in selected),
        "unknown_time_selected_count": sum(s["time_precision"] == "unknown" for s in selected),
        "strict_asof_action_evidence_count": sum(s["strict_asof_action_evidence"] for s in selected),
        "limitations": [
            "Selected steps are recorded actions, not observed clinical decisions.",
            "Lab charttime denotes specimen time; a separate test-order time is not available.",
            "Radiology charttime denotes exam/chart time; report text is available only at storetime proxy.",
            "Prescription starttime is a prescribed start, not administration or exact order time; availability is unknown.",
            "Billed procedure dates are 24-hour intervals and do not establish within-day order.",
            "Discharge notes and retrospective diagnoses are excluded as action anchors.",
            "Selection keywords are provisional and must be reviewed before eight-axis annotation.",
        ],
    }
    (output / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
