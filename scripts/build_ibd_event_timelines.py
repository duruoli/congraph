#!/usr/bin/env python3
"""Build and validate a private, time-ordered IBD MIMIC-IV event pilot.

The output is an event ledger plus data-availability checkpoints, not an
inferred clinician decision sequence.
Run from any directory:

    python scripts/build_ibd_event_timelines.py
    python scripts/build_ibd_event_timelines.py --sample-size 100 \
        --exclude-manifest runs/ibd_event_timeline_pilot_20/manifest.json \
        --output-dir runs/ibd_event_timeline_pilot_100_additional

The default output is under runs/ (gitignored). No note text is copied. Every
event retains a source key so its full content can be fetched from the local
Parquet files by an authorized researcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/raw_data/ibd_mimiciv_3_1"
SCHEMA_VERSION = "ibd-event-timeline/1.0"
BASE_SAMPLE_DESIGN = {
    "crohn_repeat": 5,
    "uc_repeat": 5,
    "surgery_with_imaging": 5,
    "sparse": 4,
    "both_codes": 1,
}

SOURCE_COLUMNS = {
    "labs": ["subject_id", "hadm_id", "labevent_id", "charttime", "storetime", "itemid", "label", "flag"],
    "microbiology": ["subject_id", "hadm_id", "microevent_id", "chartdate", "charttime", "storedate", "storetime", "test_name", "spec_type_desc"],
    "prescriptions": ["subject_id", "hadm_id", "pharmacy_id", "starttime", "stoptime", "drug", "route"],
    "procedures": ["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version", "long_title"],
    "radiology": ["subject_id", "hadm_id", "note_id", "charttime", "storetime", "exam_name", "exam_code"],
    "discharge_notes": ["subject_id", "hadm_id", "note_id", "charttime", "storetime", "note_type"],
    "diagnoses": ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version", "long_title"],
}
ORDERED_SOURCES = tuple(name for name in SOURCE_COLUMNS if name != "diagnoses")
SURGERY_PATTERN = (
    r"colectomy|resection of (?:small|large) intestine|"
    r"ileostomy|proctocolectomy"
)


def load_table(source: Path, name: str, columns: list[str], hadm_ids: list[int] | None = None) -> pd.DataFrame:
    dataset = ds.dataset(source / name, format="parquet")
    predicate = ds.field("hadm_id").isin(hadm_ids) if hadm_ids is not None else None
    return dataset.to_table(columns=columns, filter=predicate).to_pandas()


def stamp(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat(timespec="seconds")


def day_window(value: Any) -> tuple[str | None, str | None]:
    if value is None or pd.isna(value):
        return None, None
    start = pd.Timestamp(value).normalize()
    return stamp(start), stamp(start + timedelta(days=1))


def clean(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def stable_rank(seed: int, group: str, subject_id: int) -> str:
    return hashlib.sha256(f"{seed}:{group}:{subject_id}".encode()).hexdigest()


def sample_design(sample_size: int) -> dict[str, int]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    exact = {name: count * sample_size / 20 for name, count in BASE_SAMPLE_DESIGN.items()}
    targets = {name: int(value) for name, value in exact.items()}
    remaining = sample_size - sum(targets.values())
    order = sorted(exact, key=lambda name: (-(exact[name] - targets[name]), list(BASE_SAMPLE_DESIGN).index(name)))
    for name in order[:remaining]:
        targets[name] += 1
    return targets


def choose_patients(
    source: Path, seed: int, sample_size: int, excluded_subject_ids: set[int],
) -> tuple[list[int], dict[int, str], pd.DataFrame, dict[str, int]]:
    cohort = load_table(source, "cohort", [
        "subject_id", "hadm_id", "admittime", "dischtime", "ibd_type",
        "ibd_admission_number", "is_primary_ibd_diagnosis",
    ])
    inputs = load_table(source, "admission_inputs", [
        "hadm_id", "has_patient_history", "has_physical_exam",
        "has_radiology", "discharge_note_count", "radiology_report_count",
    ])
    procedures = load_table(source, "procedures", ["hadm_id", "long_title"])
    surgical_hadm = set(procedures.loc[
        procedures["long_title"].str.contains(SURGERY_PATTERN, case=False, na=False, regex=True),
        "hadm_id",
    ].astype(int))
    cohort = cohort.merge(inputs, on="hadm_id", validate="one_to_one")
    cohort["surgical_imaging"] = cohort["hadm_id"].isin(surgical_hadm) & cohort["has_radiology"]
    by_patient = cohort.groupby("subject_id")
    candidates = {
        "crohn_repeat": [int(s) for s, g in by_patient if 2 <= len(g) <= 5 and (g.ibd_type == "crohn").any()],
        "uc_repeat": [int(s) for s, g in by_patient if 2 <= len(g) <= 5 and (g.ibd_type == "ulcerative_colitis").any()],
        "surgery_with_imaging": [int(s) for s, g in by_patient if len(g) <= 5 and g.surgical_imaging.any()],
        "sparse": [int(s) for s, g in by_patient if len(g) <= 5 and ((~g.has_radiology) | (g.discharge_note_count == 0)).any()],
        "both_codes": [int(s) for s, g in by_patient if len(g) <= 5 and (g.ibd_type == "both").any()],
    }
    targets = sample_design(sample_size)
    chosen: list[int] = []
    strata: dict[int, str] = {}
    for group, count in targets.items():
        pool = sorted(
            (s for s in candidates[group] if s not in strata and s not in excluded_subject_ids),
            key=lambda s: stable_rank(seed, group, s),
        )
        if len(pool) < count:
            raise ValueError(f"Insufficient distinct patients for {group}: {len(pool)} < {count}")
        for subject_id in pool[:count]:
            chosen.append(subject_id)
            strata[subject_id] = group
    if len(chosen) != sample_size or len(set(chosen)) != sample_size:
        raise AssertionError(f"The sample must contain {sample_size} distinct patients")
    selected = cohort[cohort.subject_id.isin(chosen)].copy()
    selected = selected.sort_values(["subject_id", "admittime", "hadm_id"])
    return chosen, strata, selected, targets


def event(
    subject_id: int, hadm_id: int, source_table: str, source_key: str,
    event_at: str | None, *, event_end_exclusive: str | None = None,
    time_precision: str = "timestamp", available_at: str | None = None,
    availability_precision: str = "unknown", detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "source_table": source_table,
        "source_key": source_key,
        "event_at": event_at,
        "event_end_exclusive": event_end_exclusive,
        "time_precision": time_precision,
        "available_at": available_at,
        "availability_precision": availability_precision,
        "detail": detail or {},
    }


def source_event(name: str, row: pd.Series, source_key: str) -> dict[str, Any]:
    sid, hadm = int(row.subject_id), int(row.hadm_id)
    if name == "labs":
        return event(sid, hadm, name, source_key, stamp(row.charttime),
            time_precision="timestamp" if pd.notna(row.charttime) else "unknown",
            available_at=stamp(row.storetime),
            availability_precision="timestamp" if pd.notna(row.storetime) else "unknown",
            detail={"itemid": clean(row.itemid), "label": clean(row.label), "flag": clean(row.flag)})
    if name == "microbiology":
        event_at = stamp(row.charttime)
        end = None
        precision = "timestamp"
        if event_at is None:
            event_at, end = day_window(row.chartdate)
            precision = "date" if event_at else "unknown"
        available_at = stamp(row.storetime)
        avail_precision = "timestamp" if available_at else "unknown"
        if available_at is None:
            available_at, _ = day_window(row.storedate)
            avail_precision = "date" if available_at else "unknown"
        return event(sid, hadm, name, source_key, event_at,
            event_end_exclusive=end, time_precision=precision,
            available_at=available_at, availability_precision=avail_precision,
            detail={"test_name": clean(row.test_name), "specimen": clean(row.spec_type_desc)})
    if name == "prescriptions":
        return event(sid, hadm, name, source_key, stamp(row.starttime),
            time_precision="timestamp" if pd.notna(row.starttime) else "unknown",
            detail={"drug": clean(row.drug), "route": clean(row.route),
                    "prescribed_stoptime": stamp(row.stoptime), "time_role": "prescribed_start_not_administration"})
    if name == "procedures":
        start, end = day_window(row.chartdate)
        return event(sid, hadm, name, source_key, start,
            event_end_exclusive=end, time_precision="date" if start else "unknown",
            detail={"icd_code": clean(row.icd_code), "icd_version": clean(row.icd_version),
                    "long_title": clean(row.long_title), "time_role": "billed_procedure_date_only"})
    if name == "radiology":
        return event(sid, hadm, name, source_key, stamp(row.charttime),
            time_precision="timestamp" if pd.notna(row.charttime) else "unknown",
            available_at=stamp(row.storetime),
            availability_precision="timestamp" if pd.notna(row.storetime) else "unknown",
            detail={"exam_name": clean(row.exam_name), "exam_code": clean(row.exam_code),
                    "content_retained": False})
    if name == "discharge_notes":
        return event(sid, hadm, name, source_key, stamp(row.charttime),
            time_precision="timestamp" if pd.notna(row.charttime) else "unknown",
            available_at=stamp(row.storetime),
            availability_precision="timestamp" if pd.notna(row.storetime) else "unknown",
            detail={"note_type": clean(row.note_type), "content_retained": False,
                    "time_role": "retrospective_discharge_summary"})
    raise ValueError(f"Unsupported source: {name}")


def row_key(name: str, row: pd.Series) -> str:
    if name == "labs":
        return str(int(row.labevent_id))
    if name == "microbiology":
        return str(int(row.microevent_id))
    if name == "prescriptions":
        # pharmacy_id groups components of one order and is not row-unique.
        return json.dumps(
            {"pharmacy_id": int(row.pharmacy_id), "starttime": stamp(row.starttime), "drug": str(row.drug)},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
    if name in {"radiology", "discharge_notes"}:
        return str(row.note_id)
    if name == "procedures":
        return f"{int(row.seq_num)}:{int(row.icd_version)}:{row.icd_code}:{row.chartdate}"
    raise ValueError(name)


def timeline_sort_key(e: dict[str, Any]) -> tuple[str, int, str, str]:
    # Date-only rows are displayed at that day's start but are explicitly NOT
    # asserted to precede timed rows on the same date.
    return (
        e["event_at"] or e["available_at"] or "9999-12-31T00:00:00",
        0 if e["time_precision"] == "date" else 1,
        e["source_table"], e["source_key"],
    )


def build(
    source: Path, seed: int, sample_size: int = 20,
    excluded_subject_ids: set[int] | None = None,
    exclude_manifest_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    excluded_subject_ids = excluded_subject_ids or set()
    selected_ids, strata, selected, targets = choose_patients(source, seed, sample_size, excluded_subject_ids)
    hadm_ids = sorted(int(h) for h in selected.hadm_id)
    ownership = {int(r.hadm_id): int(r.subject_id) for _, r in selected.iterrows()}

    admissions = []
    for _, row in selected.iterrows():
        admissions.append({
            "subject_id": int(row.subject_id), "hadm_id": int(row.hadm_id),
            "admittime": stamp(row.admittime), "dischtime": stamp(row.dischtime),
            "ibd_type_retrospective": str(row.ibd_type),
            "ibd_admission_number_in_extract": int(row.ibd_admission_number),
            "is_primary_ibd_diagnosis_retrospective": bool(row.is_primary_ibd_diagnosis),
            "has_patient_history_derived_from_discharge": bool(row.has_patient_history),
            "has_physical_exam_derived_from_discharge": bool(row.has_physical_exam),
            "sample_stratum": strata[int(row.subject_id)],
            "retrospective_diagnoses": [],
        })
    by_hadm = {a["hadm_id"]: a for a in admissions}
    diagnoses = load_table(source, "diagnoses", SOURCE_COLUMNS["diagnoses"], hadm_ids)
    for _, row in diagnoses.iterrows():
        h = int(row.hadm_id)
        if int(row.subject_id) != ownership[h]:
            raise AssertionError(f"subject/hadm mismatch in diagnoses: {h}")
        by_hadm[h]["retrospective_diagnoses"].append({
            "seq_num": int(row.seq_num), "icd_code": clean(row.icd_code),
            "icd_version": clean(row.icd_version), "long_title": clean(row.long_title),
        })
    for a in admissions:
        a["retrospective_diagnoses"].sort(key=lambda d: (d["seq_num"], d["icd_version"], d["icd_code"] or ""))

    events = []
    source_counts = {}
    for a in admissions:
        for kind, when in (("admission", a["admittime"]), ("discharge", a["dischtime"])):
            if when:
                events.append(event(a["subject_id"], a["hadm_id"], "cohort", kind, when,
                    available_at=when, availability_precision="timestamp",
                    detail={"anchor": kind}))
    source_counts["cohort_anchors"] = len(events)
    source_counts["diagnoses_retrospective"] = len(diagnoses)
    for name in ORDERED_SOURCES:
        table = load_table(source, name, SOURCE_COLUMNS[name], hadm_ids)
        source_counts[name] = len(table)
        seen_keys: Counter[tuple[int, str]] = Counter()
        for _, row in table.iterrows():
            h = int(row.hadm_id)
            if int(row.subject_id) != ownership[h]:
                raise AssertionError(f"subject/hadm mismatch in {name}: {h}")
            key = row_key(name, row)
            seen_keys[(h, key)] += 1
            if seen_keys[(h, key)] > 1:
                key = f"{key}#occurrence{seen_keys[(h, key)]}"
            events.append(source_event(name, row, key))
    events.sort(key=lambda e: (e["subject_id"], timeline_sort_key(e), e["hadm_id"]))
    patient_indices: Counter[int] = Counter()
    admission_indices: Counter[int] = Counter()
    for e in events:
        e["patient_event_index"] = patient_indices[e["subject_id"]]
        e["admission_event_index"] = admission_indices[e["hadm_id"]]
        patient_indices[e["subject_id"]] += 1
        admission_indices[e["hadm_id"]] += 1
        digest = hashlib.sha256(f"{e['subject_id']}:{e['hadm_id']}:{e['source_table']}:{e['source_key']}".encode()).hexdigest()
        e["event_id"] = digest[:20]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "seed": seed,
        "sample_size": sample_size,
        "sample_design": targets,
        "excluded_subject_count": len(excluded_subject_ids),
        "excluded_subject_ids_sha256": hashlib.sha256(
            ",".join(map(str, sorted(excluded_subject_ids))).encode()
        ).hexdigest(),
        "exclude_manifest": str(exclude_manifest_path) if exclude_manifest_path else None,
        "patients": [{"subject_id": sid, "sample_stratum": strata[sid]} for sid in selected_ids],
        "source_row_counts": source_counts,
        "time_policy": {
            "charttime": "event/charting time, not necessarily result availability",
            "storetime": "result/report availability proxy; null remains unknown",
            "microbiology_storetime": "last known result update; interim result availability is not reconstructed",
            "prescription_starttime": "prescribed start, not administration or exact order time",
            "procedure_chartdate": "event_at is the lower bound of a date interval, not a procedure time; within-day order unknown",
            "diagnoses_icd": "retrospective admission metadata; excluded from event order",
            "discharge_note": "retrospective text; excluded from any pre-discharge snapshot",
            "event_indices": "display ordering by interval lower bound only; overlapping intervals do not establish precedence",
            "unknown_event_time": "retained at the end of display order; chronological position is unknown",
        },
    }
    return admissions, events, manifest


def build_checkpoints(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group newly available test results by exact time, without inferring decisions."""
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        allowed = e["source_table"] in {"labs", "microbiology", "radiology"}
        allowed |= e["source_table"] == "cohort" and e["detail"].get("anchor") == "admission"
        if allowed and e["available_at"] and e["availability_precision"] == "timestamp":
            groups[(e["subject_id"], e["available_at"])].append(e)
    checkpoints = []
    indices: Counter[int] = Counter()
    known_counts: Counter[int] = Counter()
    for (subject_id, available_at), rows in sorted(groups.items()):
        rows.sort(key=lambda e: (e["source_table"], e["event_id"]))
        known_counts[subject_id] += len(rows)
        checkpoints.append({
            "subject_id": subject_id,
            "step_index": indices[subject_id],
            "step_kind": "data_available_not_clinician_decision",
            "available_at": available_at,
            "hadm_ids": sorted({e["hadm_id"] for e in rows}),
            "new_event_ids": [e["event_id"] for e in rows],
            "new_event_count": len(rows),
            "cumulative_available_event_count": known_counts[subject_id],
            "new_events_by_source": dict(sorted(Counter(e["source_table"] for e in rows).items())),
        })
        indices[subject_id] += 1
    return checkpoints


def validate_checkpoints(
    events: list[dict[str, Any]], checkpoints: list[dict[str, Any]], expected_patient_count: int,
) -> dict[str, Any]:
    eligible = {
        e["event_id"] for e in events
        if e["available_at"] and e["availability_precision"] == "timestamp"
        and (e["source_table"] in {"labs", "microbiology", "radiology"}
             or (e["source_table"] == "cohort" and e["detail"].get("anchor") == "admission"))
    }
    event_by_id = {e["event_id"]: e for e in events}
    errors = []
    seen: list[str] = []
    per_patient: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for step in checkpoints:
        per_patient[step["subject_id"]].append(step)
        new_ids = step["new_event_ids"]
        if not new_ids or step["new_event_count"] != len(new_ids):
            errors.append(f"empty or miscounted checkpoint: {step['subject_id']}:{step['step_index']}")
        for event_id in new_ids:
            e = event_by_id.get(event_id)
            if e is None or e["available_at"] != step["available_at"] or e["subject_id"] != step["subject_id"]:
                errors.append(f"invalid checkpoint event link: {event_id}")
            seen.append(event_id)
    if len(seen) != len(set(seen)) or set(seen) != eligible:
        errors.append("checkpoints do not partition all and only eligible available events")
    for sid, rows in per_patient.items():
        if [r["step_index"] for r in rows] != list(range(len(rows))):
            errors.append(f"nonconsecutive checkpoint indices: {sid}")
        if [r["available_at"] for r in rows] != sorted(r["available_at"] for r in rows):
            errors.append(f"checkpoints out of availability order: {sid}")
        if rows[-1]["cumulative_available_event_count"] != sum(r["new_event_count"] for r in rows):
            errors.append(f"incorrect cumulative count: {sid}")
    if len(per_patient) != expected_patient_count:
        errors.append(f"not all {expected_patient_count} patients have availability checkpoints")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "checkpoint_count": len(checkpoints),
        "available_event_count": len(eligible),
    }


def validate(admissions: list[dict[str, Any]], events: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: Counter[str] = Counter()
    warnings_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    def warn(kind: str, source: str) -> None:
        warnings[kind] += 1
        warnings_by_source[source][kind] += 1

    patients = {a["subject_id"] for a in admissions}
    expected_patient_count = manifest["sample_size"]
    if len(patients) != expected_patient_count or len(manifest["patients"]) != expected_patient_count:
        errors.append(f"sample is not {expected_patient_count} patients")
    by_hadm = {a["hadm_id"]: a for a in admissions}
    if len(by_hadm) != len(admissions):
        errors.append("duplicate admission IDs")
    ids = [e["event_id"] for e in events]
    if len(ids) != len(set(ids)):
        errors.append("duplicate event IDs")
    expected = sum(manifest["source_row_counts"][s] for s in ORDERED_SOURCES) + manifest["source_row_counts"]["cohort_anchors"]
    if len(events) != expected:
        errors.append(f"event count {len(events)} != source row count {expected}")
    if sum(len(a["retrospective_diagnoses"]) for a in admissions) != manifest["source_row_counts"]["diagnoses_retrospective"]:
        errors.append("retrospective diagnosis count mismatch")
    source_counts = Counter(e["source_table"] for e in events)
    for name in ORDERED_SOURCES:
        if source_counts[name] != manifest["source_row_counts"][name]:
            errors.append(f"{name} count mismatch")
    per_patient: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        h = e["hadm_id"]
        if h not in by_hadm or e["subject_id"] != by_hadm[h]["subject_id"]:
            errors.append(f"invalid event owner: {e['event_id']}")
            continue
        per_patient[e["subject_id"]].append(e)
        t, end, available = e["event_at"], e["event_end_exclusive"], e["available_at"]
        if e["time_precision"] == "date":
            if not t or not end or pd.Timestamp(end) - pd.Timestamp(t) != timedelta(days=1):
                errors.append(f"invalid date interval: {e['event_id']}")
            warn("date_only_event_order_within_day_unknown", e["source_table"])
        elif end is not None:
            errors.append(f"unexpected event interval: {e['event_id']}")
        if t is None:
            warn("missing_event_time", e["source_table"])
        if available is None:
            warn("unknown_availability_time", e["source_table"])
        elif t and e["time_precision"] == "timestamp" and e["availability_precision"] == "timestamp" and available < t:
            warn("available_before_charttime", e["source_table"])
        admission = by_hadm[h]
        if t and e["time_precision"] == "date":
            if end and end <= admission["admittime"]:
                warn("definitely_pre_admission_date_event", e["source_table"])
            elif t < admission["admittime"] < end:
                warn("date_event_overlaps_admission_boundary", e["source_table"])
            if admission["dischtime"]:
                if t >= admission["dischtime"]:
                    warn("definitely_post_discharge_date_event", e["source_table"])
                elif t < admission["dischtime"] < end:
                    warn("date_event_overlaps_discharge_boundary", e["source_table"])
        else:
            if t and t < admission["admittime"]:
                warn("pre_admission_event", e["source_table"])
            if t and admission["dischtime"] and t > admission["dischtime"]:
                warn("post_discharge_event", e["source_table"])
        if available and admission["dischtime"] and available > admission["dischtime"]:
            warn("availability_after_discharge", e["source_table"])
    for sid, rows in per_patient.items():
        if rows != sorted(rows, key=lambda e: (timeline_sort_key(e), e["hadm_id"])):
            errors.append(f"events not chronologically sorted for subject {sid}")
        if [e["patient_event_index"] for e in rows] != list(range(len(rows))):
            errors.append(f"patient event indices are not consecutive for subject {sid}")
    per_admission: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        per_admission[e["hadm_id"]].append(e)
    for hadm, rows in per_admission.items():
        if [e["admission_event_index"] for e in rows] != list(range(len(rows))):
            errors.append(f"admission event indices are not consecutive for admission {hadm}")
    if len(per_patient) != expected_patient_count:
        errors.append("not all sampled patients have events")
    for sid in patients:
        adm = sorted((a for a in admissions if a["subject_id"] == sid), key=lambda a: (a["admittime"], a["hadm_id"]))
        numbers = [a["ibd_admission_number_in_extract"] for a in adm]
        if numbers != sorted(numbers):
            warn("cohort_admission_number_not_time_sorted", "cohort")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": dict(sorted(warnings.items())),
        "warnings_by_source": {source: dict(sorted(counts.items())) for source, counts in sorted(warnings_by_source.items())},
        "patient_count": len(patients),
        "admission_count": len(admissions),
        "event_count": len(events),
        "events_by_source": dict(sorted(source_counts.items())),
        "retrospective_diagnosis_rows": sum(len(a["retrospective_diagnoses"]) for a in admissions),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def preview(admissions: list[dict[str, Any]], events: list[dict[str, Any]], checkpoints: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    by_patient: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_hadm: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        by_patient[e["subject_id"]].append(e)
        by_hadm[e["hadm_id"]].append(e)
    checkpoint_counts = Counter(step["subject_id"] for step in checkpoints)
    lines = [
        f"# IBD event timeline pilot: {manifest['sample_size']} patients",
        "",
        "Private MIMIC-derived preview. Relative hours are measured from each admission's admittime.",
        "Date-only procedures are shown as 24-hour intervals; their position among same-day timed events is unknown.",
        "Availability checkpoints group admission anchors and newly available lab, microbiology, and radiology results; they are not clinician decisions.",
        "No clinical note text is copied into the output.",
        "",
    ]
    for i, item in enumerate(manifest["patients"], 1):
        sid = item["subject_id"]
        episodes = sorted((a for a in admissions if a["subject_id"] == sid), key=lambda a: a["admittime"])
        lines.append(f"## Case {i:02d} — {item['sample_stratum']} ({len(episodes)} IBD-coded admissions, {len(by_patient[sid])} events, {checkpoint_counts[sid]} availability checkpoints)")
        lines.append("")
        for j, a in enumerate(episodes, 1):
            rows = by_hadm[a["hadm_id"]]
            counts = Counter(e["source_table"] for e in rows)
            lines.append(f"- Admission {j}: retrospective cohort type `{a['ibd_type_retrospective']}`; "
                         f"{len(rows)} events; sources " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
            base = pd.Timestamp(a["admittime"])
            highlights = [e for e in rows if e["source_table"] in {"cohort", "radiology", "procedures", "discharge_notes"}]
            for source_name in ("labs", "microbiology", "prescriptions"):
                first = next((e for e in rows if e["source_table"] == source_name), None)
                if first is not None:
                    highlights.append(first)
            highlights.sort(key=timeline_sort_key)
            for e in highlights[:12]:
                t = e["event_at"]
                relative = f"{(pd.Timestamp(t) - base).total_seconds() / 3600:+.1f}h" if t else "time unknown"
                label = (e["detail"].get("exam_name") or e["detail"].get("long_title") or
                         e["detail"].get("label") or e["detail"].get("test_name") or
                         e["detail"].get("drug") or e["detail"].get("anchor") or e["source_table"])
                suffix = ""
                if e["time_precision"] == "date":
                    upper = (pd.Timestamp(e["event_end_exclusive"]) - base).total_seconds() / 3600
                    relative = f"[{relative}, {upper:+.1f}h)"
                    suffix = " [date only; order within interval unknown]"
                if e["available_at"] and e["available_at"] != t:
                    availability_note = "available" if e["availability_precision"] == "timestamp" else "availability date starts"
                    suffix += f" [{availability_note} {(pd.Timestamp(e['available_at']) - base).total_seconds() / 3600:+.1f}h]"
                lines.append(f"  - {relative}: {e['source_table']} — {label}{suffix}")
            if len(highlights) > 12:
                lines.append(f"  - … {len(highlights) - 12} further anchor/report/procedure events; see events.jsonl")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--exclude-manifest", type=Path,
                        help="Manifest from an earlier run whose patients must be excluded")
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    source = args.source.resolve()
    output = (args.output_dir or ROOT / f"runs/ibd_event_timeline_pilot_{args.sample_size}").resolve()
    exclude_path = args.exclude_manifest.resolve() if args.exclude_manifest else None
    excluded_subject_ids: set[int] = set()
    if exclude_path:
        if not exclude_path.is_file():
            parser.error(f"Exclusion manifest not found: {exclude_path}")
        excluded_subject_ids = {
            int(item["subject_id"])
            for item in json.loads(exclude_path.read_text(encoding="utf-8"))["patients"]
        }
    if not (source / "cohort").is_dir():
        parser.error(f"MIMIC source not found: {source}")
    if output == source or source in output.parents:
        parser.error("Output must not be inside the raw data directory")
    admissions, events, manifest = build(
        source, args.seed, args.sample_size, excluded_subject_ids, exclude_path,
    )
    if any(item["subject_id"] in excluded_subject_ids for item in manifest["patients"]):
        raise RuntimeError("New sample overlaps the exclusion manifest")
    report = validate(admissions, events, manifest)
    checkpoints = build_checkpoints(events)
    checkpoint_report = validate_checkpoints(events, checkpoints, args.sample_size)
    report.update({"availability_checkpoints": checkpoint_report})
    if report["status"] != "pass" or checkpoint_report["status"] != "pass":
        raise RuntimeError(f"Timeline validation failed: {report['errors'] + checkpoint_report['errors']}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    write_jsonl(output / "admissions.jsonl", admissions)
    write_jsonl(output / "events.jsonl", events)
    write_jsonl(output / "availability_checkpoints.jsonl", checkpoints)
    write_json(output / "validation_report.json", report)
    (output / "timeline_preview.md").write_text(preview(admissions, events, checkpoints, manifest), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
