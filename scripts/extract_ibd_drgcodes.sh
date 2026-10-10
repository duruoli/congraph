#!/usr/bin/env bash
# Extract all DRG rows for the existing IBD admission cohort.
set -euo pipefail

project="congraph-mimic-ibd"
dataset="ibd_extract"
bucket="congraph-mimic-ibd-export-duruoli-2026"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_root="${repo_root}/data/raw_data/ibd_mimiciv_3_1"
target="${local_root}/drgcodes"

if [[ -e "$target" ]]; then
  echo "Target already exists; preserving it: ${target}" >&2
  exit 1
fi

bq query --project_id="$project" --use_legacy_sql=false \
  --maximum_bytes_billed=2000000000 <<'SQL'
CREATE TABLE `congraph-mimic-ibd.ibd_extract.drgcodes` AS
SELECT d.*
FROM `physionet-data.mimiciv_3_1_hosp.drgcodes` AS d
JOIN `congraph-mimic-ibd.ibd_extract.cohort` AS c
  USING (subject_id, hadm_id);
SQL

expected="$(bq query --project_id="$project" --use_legacy_sql=false \
  --quiet --format=csv \
  'SELECT COUNT(*) AS n FROM `congraph-mimic-ibd.ibd_extract.drgcodes`' \
  | tail -n 1 | tr -d '\r')"
if [[ ! "$expected" =~ ^[0-9]+$ ]]; then
  echo "Could not read BigQuery row count: ${expected}" >&2
  exit 1
fi

export_prefix="drgcodes/$(date -u +%Y%m%dT%H%M%S)-$$"
export_uri="gs://${bucket}/${export_prefix}/part-*.parquet"
bq query --project_id="$project" --use_legacy_sql=false \
  --maximum_bytes_billed=2000000000 <<SQL
EXPORT DATA OPTIONS (
  uri = '${export_uri}',
  format = 'PARQUET',
  compression = 'SNAPPY',
  overwrite = FALSE
) AS
SELECT * FROM \`congraph-mimic-ibd.ibd_extract.drgcodes\`;
SQL

scratch="$(mktemp -d "${local_root}/.drgcodes.download.XXXXXX")"
gcloud storage rsync --recursive "gs://${bucket}/${export_prefix}" "$scratch"

python - "$scratch" "$expected" <<'PY'
from pathlib import Path
import sys
import pyarrow.parquet as pq

folder, expected = Path(sys.argv[1]), int(sys.argv[2])
files = sorted(folder.glob("part-*.parquet"))
actual = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
if not files or actual != expected:
    raise SystemExit(f"Validation failed: {actual} rows, expected {expected}")
print(f"Validated drgcodes: {actual:,} rows in {len(files)} file(s)")
PY

mv "$scratch" "$target"
echo "Saved ${expected} DRG rows to ${target}; GCS export retained at ${export_uri}"
