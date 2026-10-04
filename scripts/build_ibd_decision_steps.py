#!/usr/bin/env python3
"""Build auditable pre-action decision *candidates* from an IBD event ledger.

The MIMIC extract has no order timestamp. Exam, specimen, and prescription-start
times are therefore proxies for a decision boundary, never observed decisions.
Evidence whose availability interval ends before the action proxy begins is
exposed. Overlapping availability remains separate. The subsequent recorded
action is kept in a separate outcome file.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from build_ibd_action_steps import make_steps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "runs/ibd_event_timeline_pilot_20"
DEFAULT_OUTPUT = ROOT / "runs/ibd_decision_steps_pilot_20"
POLICY_VERSION = "ibd-pre-action-decision-candidates/2.0"

# Generic lab panels and admission transitions are not sufficiently specific to
# seed this first decision pilot. Date-only procedures get a day interval.
TARGET_CATEGORIES = {
    "gi_imaging", "gi_microbiology", "steroid_or_aminosalicylate_rx",
    "biologic_or_immunomodulator_rx", "antimicrobial_rx", "nutrition_rx",
    "gi_or_supportive_procedure",
}
TARGET_SOURCES = {"radiology", "microbiology", "prescriptions", "procedures"}
EVIDENCE_SOURCES = {"cohort", "labs", "microbiology", "radiology"}
TIME_ROLE = {
    "radiology": "exam_time_proxy",
    "microbiology": "specimen_collection_proxy",
    "prescriptions": "prescription_start_proxy",
    "procedures": "billed_procedure_day_proxy",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def next_day(iso_timestamp: str) -> str:
    return (datetime.fromisoformat(iso_timestamp) + timedelta(days=1)).isoformat(timespec="seconds")


def evidence_relation(event: dict[str, Any], boundary_start: str, boundary_end: str) -> str | None:
    """Partial order of availability versus action proxy: before/overlap/after."""
    if event["source_table"] not in EVIDENCE_SOURCES or event["available_at"] is None:
        return None
    if event["source_table"] == "cohort" and event["detail"].get("anchor") != "admission":
        return None
    precision = event["availability_precision"]
    if precision not in {"timestamp", "date"}:
        return None
    start = event["available_at"]
    end = start if precision == "timestamp" else next_day(start)
    if event["event_at"] is not None and (
        event["event_at"] > start if precision == "timestamp" else event["event_at"] >= end
    ):
        return None  # inconsistent source chronology
    if precision == "timestamp" and start < boundary_start:
        return "before"
    if precision == "date" and end <= boundary_start:
        return "before"
    if (start > boundary_start if boundary_start == boundary_end else start >= boundary_end):
        return "after"
    return "overlap"


def build_decision_candidates(
    events: list[dict[str, Any]], action_steps: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_id = {event["event_id"]: event for event in events}
    if len(by_id) != len(events):
        raise ValueError("Duplicate ledger event IDs")
    by_patient: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_patient[event["subject_id"]].append(event)

    decisions: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    prior_prescriptions: dict[tuple[int, int], set[tuple[str, str]]] = defaultdict(set)
    for group in sorted(action_steps, key=lambda row: (
        row["subject_id"], row["event_at"] or "9999-12-31T00:00:00", row["action_group_id"]
    )):
        target_actions = [
            action for action in group["actions"]
            if action["category"] in TARGET_CATEGORIES
            and action["source_table"] in TARGET_SOURCES
        ]
        if not target_actions:
            continue
        if group["time_precision"] not in {"timestamp", "date"} or group["event_at"] is None:
            exclusions["imprecise_target_time"] += 1
            continue
        boundary = group["event_at"]
        boundary_end = (
            boundary if group["time_precision"] == "timestamp"
            else group.get("event_end_exclusive") or next_day(boundary)
        )
        subject_id = group["subject_id"]
        target_ids = {action["event_id"] for action in target_actions}
        if any(by_id[event_id]["subject_id"] != subject_id for event_id in target_ids):
            raise ValueError("Target action belongs to another patient")
        rx_actions = [action for action in target_actions if action["source_table"] == "prescriptions"]
        rx_keys = {
            (" ".join(str(action["detail"].get("drug") or "").casefold().split()),
             " ".join(str(action["detail"].get("route") or "").casefold().split()))
            for action in rx_actions
        }
        prior_rx = prior_prescriptions[(subject_id, group["hadm_id"])]
        if not rx_actions:
            prescription_record_role = "not_applicable"
        elif rx_keys <= prior_rx:
            prescription_record_role = "repeat_same_drug_and_route_in_admission"
        else:
            prescription_record_role = "first_recorded_start_for_drug_and_route_in_admission"
        prior_rx.update(rx_keys)
        evidence = sorted(
            (event for event in by_patient[subject_id]
             if event["event_id"] not in target_ids
             and evidence_relation(event, boundary, boundary_end) == "before"),
            key=lambda event: (event["available_at"], event["event_id"]),
        )
        overlapping = sorted(
            (event for event in by_patient[subject_id]
             if event["event_id"] not in target_ids
             and evidence_relation(event, boundary, boundary_end) == "overlap"),
            key=lambda event: (event["available_at"], event["event_id"]),
        )
        step_id = f"ibd_decision_{group['action_group_id']}"
        decisions.append({
            "step_id": step_id,
            "subject_id": subject_id,
            "hadm_id": group["hadm_id"],
            "step_kind": "pre_action_decision_candidate",
            "boundary_at": boundary,
            "boundary_end_exclusive": boundary_end if group["time_precision"] == "date" else None,
            "boundary_precision": group["time_precision"],
            "boundary_kind": "recorded_action_interval_proxy",
            "actual_order_time_observed": False,
            "pre_order_evidence_certified": False,
            "target_time_roles": sorted({TIME_ROLE[action["source_table"]] for action in target_actions}),
            "evidence_rule": "availability_interval_definitely_before_action_proxy_interval",
            "pre_action_proxy_evidence_event_ids": [event["event_id"] for event in evidence],
            "action_proxy_overlap_evidence_event_ids": [event["event_id"] for event in overlapping],
            "latest_evidence_available_at": evidence[-1]["available_at"] if evidence else None,
            "evidence_counts_by_source": dict(sorted(Counter(event["source_table"] for event in evidence).items())),
            "decision_question": None,
            "management_domain": None,
            "decision_status": "requires_clinical_review",
            "prescription_record_role": prescription_record_role,
            "review_priority": (
                "lower_repeat_prescription" if prescription_record_role == "repeat_same_drug_and_route_in_admission"
                else "standard"
            ),
            "observed_outcome_id": step_id,
        })
        outcomes.append({
            "step_id": step_id,
            "subject_id": subject_id,
            "hadm_id": group["hadm_id"],
            "recorded_action_at": boundary,
            "action_group_id": group["action_group_id"],
            "target_action_event_ids": sorted(target_ids),
            "target_actions": [
                {"event_id": action["event_id"], "source_table": action["source_table"],
                 "source_key": action["source_key"], "category": action["category"],
                 "detail": action["detail"]}
                for action in target_actions
            ],
        })

    decisions.sort(key=lambda row: (row["subject_id"], row["boundary_at"], row["step_id"]))
    outcomes.sort(key=lambda row: (row["subject_id"], row["recorded_action_at"], row["step_id"]))
    per_patient_index: Counter[int] = Counter()
    for row in decisions:
        row["step_index"] = per_patient_index[row["subject_id"]]
        per_patient_index[row["subject_id"]] += 1
    report = {
        "policy_version": POLICY_VERSION,
        "candidate_count": len(decisions),
        "patient_count": len({row["subject_id"] for row in decisions}),
        "target_categories": dict(sorted(Counter(
            action["category"] for row in outcomes for action in row["target_actions"]
        ).items())),
        "target_time_roles": dict(sorted(Counter(
            role for row in decisions for role in row["target_time_roles"]
        ).items())),
        "zero_evidence_candidates": sum(not row["pre_action_proxy_evidence_event_ids"] for row in decisions),
        "candidates_with_overlap_evidence": sum(bool(row["action_proxy_overlap_evidence_event_ids"]) for row in decisions),
        "lower_priority_repeat_prescription_candidates": sum(
            row["review_priority"] == "lower_repeat_prescription" for row in decisions
        ),
        "exclusions": dict(sorted(exclusions.items())),
        "limitations": [
            "These are action-conditioned candidate nodes, not verified clinician decisions.",
            "The actual order time is absent; evidence before the action proxy may still postdate the order.",
            "Date-only target actions are intervals; same-day evidence is overlapping, not ordered.",
            "Available evidence is not proof that a clinician saw or used it.",
            "No-action/stop opportunities cannot be identified reliably from this stream alone.",
            "Decision question and management domain require pre-action clinical review.",
            "Prescription start is not administration; indication is not inferred from drug name.",
            "A repeated same-drug/route prescription is flagged for lower review priority; dose changes are not represented in the ledger.",
            "Only IBD-coded admissions in the local extract are observable.",
        ],
    }
    return decisions, outcomes, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--actions", type=Path, help="Optional prebuilt action_steps.jsonl; otherwise derive groups from ledger")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    events = read_jsonl(args.ledger_dir / "events.jsonl")
    action_steps = read_jsonl(args.actions) if args.actions else make_steps(events)[1]
    decisions, outcomes, report = build_decision_candidates(events, action_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "decision_candidates.jsonl", decisions)
    write_jsonl(args.output_dir / "observed_outcomes.jsonl", outcomes)
    (args.output_dir / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
