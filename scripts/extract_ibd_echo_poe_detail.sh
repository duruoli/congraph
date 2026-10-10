#!/usr/bin/env bash
# Extract admission-matched MIMIC-IV-ECHO metadata/measurements and POE details.
set -euo pipefail

project="congraph-mimic-ibd"
dataset="ibd_extract"
bucket="congraph-mimic-ibd-export-duruoli-2026"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_root="${repo_root}/data/raw_data/ibd_mimiciv_3_1"

run_sql() {
  bq query --project_id="$project" --use_legacy_sql=false \
    --maximum_bytes_billed=15000000000 "$@"
}

# Echo studies do not carry hadm_id. Attach only those performed within an IBD
# cohort admission; keep all original rows/fields and add the matched hadm_id.
run_sql <<'SQL'
CREATE TABLE `congraph-mimic-ibd.ibd_extract.echo_measurements` AS
SELECT c.hadm_id, m.*
FROM `physionet-data.mimiciv_echo.structured_measurement` AS m
JOIN `congraph-mimic-ibd.ibd_extract.cohort` AS c
  ON m.subject_id = c.subject_id
 AND m.measurement_datetime >= c.admittime
 AND m.measurement_datetime < c.dischtime;
SQL

run_sql <<'SQL'
CREATE TABLE `congraph-mimic-ibd.ibd_extract.echo_studies` AS
SELECT c.hadm_id, s.*
FROM `physionet-data.mimiciv_echo.echo_study_list` AS s
JOIN `congraph-mimic-ibd.ibd_extract.cohort` AS c
  ON s.subject_id = c.subject_id
 AND s.study_datetime >= c.admittime
 AND s.study_datetime < c.dischtime;
SQL

# This is a file index, not the DICOM image files themselves.
run_sql <<'SQL'
CREATE TABLE `congraph-mimic-ibd.ibd_extract.echo_record_list` AS
SELECT s.hadm_id, r.*
FROM `physionet-data.mimiciv_echo.echo_record_list` AS r
JOIN `congraph-mimic-ibd.ibd_extract.echo_studies` AS s
  ON r.subject_id = s.subject_id AND r.study_id = s.study_id;
SQL

# poe_detail has no hadm_id or event timestamp; inherit both from the exact POE.
run_sql <<'SQL'
CREATE TABLE `congraph-mimic-ibd.ibd_extract.poe_detail` AS
SELECT p.hadm_id, p.ordertime, p.order_type, p.order_subtype,
       p.transaction_type, d.*
FROM `physionet-data.mimiciv_3_1_hosp.poe_detail` AS d
JOIN `congraph-mimic-ibd.ibd_extract.poe` AS p
  ON d.subject_id = p.subject_id
 AND d.poe_id = p.poe_id
 AND d.poe_seq = p.poe_seq;
SQL

for table in echo_measurements echo_studies echo_record_list poe_detail; do
  expected="$(bq query --project_id="$project" --use_legacy_sql=false \
    --quiet --format=csv "SELECT COUNT(*) AS n FROM \`${project}.${dataset}.${table}\`" \
    | tail -n 1 | tr -d '\r')"
  if [[ ! "$expected" =~ ^[0-9]+$ ]]; then
    echo "Could not read BigQuery row count for ${table}: ${expected}" >&2
    exit 1
  fi

  export_prefix="${table}/$(date -u +%Y%m%dT%H%M%S)-$$"
  export_uri="gs://${bucket}/${export_prefix}/part-*.parquet"
  run_sql <<SQL
EXPORT DATA OPTIONS (
  uri = '${export_uri}',
  format = 'PARQUET',
  compression = 'SNAPPY',
  overwrite = FALSE
) AS
SELECT * FROM \`${project}.${dataset}.${table}\`;
SQL

  scratch="$(mktemp -d "${local_root}/.${table}.download.XXXXXX")"
  gcloud storage rsync --recursive "gs://${bucket}/${export_prefix}" "$scratch"

  python - "$scratch" "$expected" <<'PY'
from pathlib import Path
import sys
import pyarrow.parquet as pq

folder, expected = Path(sys.argv[1]), int(sys.argv[2])
files = sorted(folder.glob("part-*.parquet"))
actual = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
if not files or actual != expected:
    raise SystemExit(f"Validation failed: {folder}: {actual} rows, expected {expected}")
print(f"Validated {folder.name}: {actual:,} rows in {len(files)} file(s)")
PY

  target="${local_root}/${table}"
  if [[ -e "$target" ]]; then
    echo "Validated new export in ${scratch}, but target already exists: ${target}" >&2
    echo "Preserving both; review before replacing either copy." >&2
    exit 1
  fi
  mv "$scratch" "$target"
  echo "Saved ${table}: ${expected} rows to ${target}; GCS export retained at ${export_uri}"
done
