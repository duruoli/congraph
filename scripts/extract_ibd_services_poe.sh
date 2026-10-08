#!/usr/bin/env bash
set -euo pipefail

project="congraph-mimic-ibd"
dataset="ibd_extract"
bucket="congraph-mimic-ibd-export-duruoli-2026"
local_root="data/raw_data/ibd_mimiciv_3_1"

for table in services poe; do
  bq query --project_id="$project" --use_legacy_sql=false <<SQL
CREATE OR REPLACE TABLE \`${project}.${dataset}.${table}\` AS
SELECT source.*
FROM \`physionet-data.mimiciv_3_1_hosp.${table}\` AS source
JOIN \`${project}.${dataset}.cohort\` AS cohort
  USING (subject_id, hadm_id);
SQL

  bq query --project_id="$project" --use_legacy_sql=false <<SQL
EXPORT DATA OPTIONS (
  uri = 'gs://${bucket}/${table}/part-*.parquet',
  format = 'PARQUET',
  compression = 'SNAPPY',
  overwrite = TRUE
) AS
SELECT * FROM \`${project}.${dataset}.${table}\`;
SQL

  mkdir -p "${local_root}/${table}"
  gcloud storage rsync --recursive \
    "gs://${bucket}/${table}" "${local_root}/${table}"

  python - "${local_root}/${table}" <<'PY'
from pathlib import Path
import sys
import pyarrow.parquet as pq

folder = Path(sys.argv[1])
files = sorted(folder.glob("part-*.parquet"))
if not files or sum(pq.ParquetFile(path).metadata.num_rows for path in files) == 0:
    raise SystemExit(f"No readable data in {folder}; GCS export retained")
PY

  gcloud storage rm "gs://${bucket}/${table}/part-*.parquet"
done
