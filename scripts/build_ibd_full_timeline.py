#!/usr/bin/env python3
"""Build a patient-ordered event ledger for all locally extracted IBD admissions.

The ledger preserves recorded time separately from charted/event time. Date-only
events represent a whole day, and their order relative to timed events on that
day is unknown. Billing diagnoses have no event time and are not put on the line.
No clinical note text is copied. Output is private MIMIC-derived data under runs/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw_data/ibd_mimiciv_3_1"
OUTPUT = ROOT / "runs/ibd_full_timeline"
SOURCES = {
    "labs": ["labevent_id", "charttime", "storetime", "itemid", "label", "value", "valuenum", "valueuom", "flag"],
    "microbiology": ["microevent_id", "chartdate", "charttime", "storedate", "storetime", "test_name", "spec_type_desc", "org_name", "ab_name", "interpretation"],
    "prescriptions": ["pharmacy_id", "starttime", "stoptime", "drug", "route", "dose_val_rx", "dose_unit_rx"],
    "procedures": ["seq_num", "chartdate", "icd_code", "icd_version", "long_title"],
    "radiology": ["note_id", "charttime", "storetime", "exam_name", "exam_code"],
    "discharge_notes": ["note_id", "charttime", "storetime", "note_type"],
    "services": ["transfertime", "prev_service", "curr_service"],
    "poe": ["poe_id", "poe_seq", "ordertime", "order_type", "order_subtype", "transaction_type", "order_status", "discontinue_of_poe_id", "discontinued_by_poe_id"],
}
DETAILS = {
    "labs": ["labevent_id", "itemid", "label", "value", "valuenum", "valueuom", "flag"],
    "microbiology": ["microevent_id", "test_name", "spec_type_desc", "org_name", "ab_name", "interpretation"],
    "prescriptions": ["pharmacy_id", "drug", "route", "dose_val_rx", "dose_unit_rx"],
    "procedures": ["seq_num", "icd_code", "icd_version", "long_title"],
    "radiology": ["note_id", "exam_name", "exam_code"],
    "discharge_notes": ["note_id", "note_type"],
    "services": ["prev_service", "curr_service"],
    "poe": ["poe_id", "poe_seq", "order_type", "order_subtype", "transaction_type", "order_status", "discontinue_of_poe_id", "discontinued_by_poe_id"],
}
SCHEMA = pa.schema([
    ("subject_id", pa.int64()), ("hadm_id", pa.int64()),
    ("patient_event_index", pa.int64()), ("admission_event_index", pa.int64()),
    ("event_id", pa.string()), ("source_table", pa.string()),
    ("source_row_ordinal", pa.int64()), ("event_kind", pa.string()),
    ("event_at", pa.timestamp("us")), ("event_end_at", pa.timestamp("us")),
    ("event_end_kind", pa.string()), ("time_precision", pa.string()),
    ("recorded_at", pa.timestamp("us")), ("recorded_precision", pa.string()),
    ("detail_json", pa.string()),
])


def iso(value):
    return value.isoformat(sep=" ") if value is not None else None


def day(value):
    if value is None:
        return None, None
    start = datetime(value.year, value.month, value.day)
    return iso(start), iso(start + timedelta(days=1))


def detail_value(value):
    if isinstance(value, (datetime,)):
        return iso(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def event_fields(name, row):
    """Return occurrence, end, end kind, precision, record, record precision, kind."""
    if name == "labs":
        return iso(row["charttime"]), None, None, "minute", iso(row["storetime"]), "minute" if row["storetime"] else "unknown", "lab_measurement"
    if name == "microbiology":
        at, end = (iso(row["charttime"]), None) if row["charttime"] else day(row["chartdate"])
        record, _ = (iso(row["storetime"]), None) if row["storetime"] else day(row["storedate"])
        return at, end, "exclusive_date_bound" if end else None, "minute" if row["charttime"] else "date" if at else "unknown", record, "minute" if row["storetime"] else "date" if record else "unknown", "microbiology_observation"
    if name == "prescriptions":
        return iso(row["starttime"]), iso(row["stoptime"]), "prescribed_stop" if row["stoptime"] else None, "minute" if row["starttime"] else "unknown", None, "unknown", "prescribed_period"
    if name == "procedures":
        at, end = day(row["chartdate"])
        return at, end, "exclusive_date_bound" if end else None, "date" if at else "unknown", None, "unknown", "coded_procedure_date"
    if name in {"radiology", "discharge_notes"}:
        kind = "radiology_report" if name == "radiology" else "retrospective_discharge_note"
        return iso(row["charttime"]), None, None, "minute" if row["charttime"] else "unknown", iso(row["storetime"]), "minute" if row["storetime"] else "unknown", kind
    if name == "services":
        return iso(row["transfertime"]), None, None, "minute" if row["transfertime"] else "unknown", None, "unknown", "service_assignment_or_change"
    if name == "poe":
        return iso(row["ordertime"]), None, None, "minute" if row["ordertime"] else "unknown", None, "unknown", "provider_order"
    raise ValueError(name)


def batches(folder, columns, batch_size):
    dataset = ds.dataset(folder, format="parquet")
    for fragment in sorted(dataset.get_fragments(), key=lambda f: f.path):
        yield from fragment.to_batches(columns=columns, batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=50000)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output_dir.resolve()
    if output == source or source in output.parents:
        parser.error("Output must be outside the raw source directory")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if (output / "events.parquet").exists():
        parser.error("Output events.parquet already exists; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    db_path = output / "timeline_build.sqlite"
    if db_path.exists():
        parser.error(f"Temporary build database already exists: {db_path}")

    cohort = ds.dataset(source / "cohort", format="parquet").to_table(columns=[
        "subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
    ])
    owners = {int(r["hadm_id"]): int(r["subject_id"]) for r in cohort.to_pylist()}
    if len(owners) != len(cohort):
        raise ValueError("Cohort hadm_id is not unique")
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("CREATE TABLE events (subject_id INTEGER, hadm_id INTEGER, event_id TEXT, source_table TEXT, source_row_ordinal INTEGER, event_kind TEXT, event_at TEXT, event_end_at TEXT, event_end_kind TEXT, time_precision TEXT, recorded_at TEXT, recorded_precision TEXT, detail_json TEXT)")
    counts = Counter()
    missing = Counter()

    def insert(rows):
        con.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    anchors = []
    for row in cohort.to_pylist():
        sid, hadm = int(row["subject_id"]), int(row["hadm_id"])
        for kind, field in (("admission", "admittime"), ("discharge", "dischtime"), ("in_hospital_death", "deathtime")):
            at = iso(row[field])
            if at is None:
                continue
            key = f"cohort:{hadm}:{kind}"
            anchors.append((sid, hadm, hashlib.sha256(key.encode()).hexdigest()[:24], "cohort", None, kind,
                            at, None, None, "minute", at, "minute", "{}"))
            counts[f"cohort_{kind}"] += 1
    insert(anchors)
    con.commit()

    for name, extra_columns in SOURCES.items():
        ordinal = 0
        for batch in batches(source / name, ["subject_id", "hadm_id", *extra_columns], args.batch_size):
            rows = []
            for row in batch.to_pylist():
                sid, hadm = int(row["subject_id"]), int(row["hadm_id"])
                if owners.get(hadm) != sid:
                    raise ValueError(f"Source {name} contains a row outside the cohort")
                at, end, end_kind, precision, recorded, recorded_precision, kind = event_fields(name, row)
                if at is None:
                    missing[name] += 1
                key = f"{name}:{ordinal}"
                detail = {k: detail_value(row[k]) for k in DETAILS[name] if row[k] is not None}
                rows.append((sid, hadm, hashlib.sha256(key.encode()).hexdigest()[:24], name, ordinal,
                             kind, at, end, end_kind, precision, recorded, recorded_precision,
                             json.dumps(detail, ensure_ascii=False, sort_keys=True, default=str)))
                ordinal += 1
            insert(rows)
            counts[name] += len(rows)
            con.commit()
        print(f"{name}: {counts[name]:,} rows", flush=True)

    con.execute("CREATE INDEX event_order ON events(subject_id, event_at, hadm_id, source_table, source_row_ordinal)")
    con.commit()
    writer = pq.ParquetWriter(output / "events.parquet", SCHEMA, compression="zstd")
    fields = [field.name for field in SCHEMA]
    patient_index = Counter()
    admission_index = Counter()
    output_count = 0
    try:
        cursor = con.execute("SELECT * FROM events ORDER BY subject_id, event_at IS NULL, event_at, hadm_id, source_table, source_row_ordinal")
        while rows := cursor.fetchmany(args.batch_size):
            records = []
            for sid, hadm, event_id, source_table, ordinal, kind, at, end, end_kind, precision, recorded, recorded_precision, detail in rows:
                records.append((sid, hadm, patient_index[sid], admission_index[hadm], event_id, source_table,
                                ordinal, kind, datetime.fromisoformat(at) if at else None,
                                datetime.fromisoformat(end) if end else None, end_kind, precision,
                                datetime.fromisoformat(recorded) if recorded else None,
                                recorded_precision, detail))
                patient_index[sid] += 1
                admission_index[hadm] += 1
            writer.write_table(pa.Table.from_pylist([dict(zip(fields, row)) for row in records], schema=SCHEMA))
            output_count += len(records)
    finally:
        writer.close()
        con.close()
    expected = sum(counts.values())
    if output_count != expected:
        raise ValueError(f"Output row mismatch: {output_count} != {expected}")
    db_path.unlink()
    report = {
        "schema_version": "ibd-full-timeline/1.0",
        "patient_count": len(set(owners.values())), "admission_count": len(owners),
        "event_count": output_count, "events_by_source": dict(sorted(counts.items())),
        "missing_event_time_by_source": dict(sorted(missing.items())),
        "time_policy": {
            "date_only": "A date denotes [00:00 that day, 00:00 next day); display order within that day is unknown.",
            "charttime": "Charted observation time; may be rounded or entered later.",
            "storetime": "Validation/storage time, not necessarily first clinical availability.",
            "prescription": "starttime/stoptime describe a prescribed period, not confirmed administration.",
            "poe": "ordertime is an order timestamp, not proof the action occurred.",
            "services": "transfertime marks service assignment/change, not proof of a diagnostic change.",
            "diagnoses_icd": "Untimed retrospective admission labels; deliberately excluded from the event ledger.",
            "sorting": "patient_event_index is display order by event_at, with date-only events at their lower bound; it does not imply within-day precedence.",
        },
    }
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
