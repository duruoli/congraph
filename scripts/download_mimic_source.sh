#!/usr/bin/env bash
# Download the MIMIC-IV source tables needed to attach real charttime/admittime to the
# annotated radiology decision steps (see mimic_supp_data.md + experiments/annotation/timing.py).
#
# Targets data/raw_data/mimic_source/ — the dir SourceTables.load(--source-dir) expects.
# Files are .gz; pandas.read_csv reads them transparently, no need to gunzip.
#
# Requires a PhysioNet account with signed DUAs for BOTH projects:
#   - MIMIC-IV v3.1          (admissions, procedures_icd, d_icd_procedures)
#   - MIMIC-IV-Note v2.2     (radiology — text reports live here, NOT in the core db)
#
# Needs an interactive password prompt, so run it from the Claude Code session with `!`:
#     ! bash scripts/download_mimic_source.sh <physionet_username>
# (default username below is cherishbeing; override by passing one as $1)

set -euo pipefail

USER_NAME="${1:-cherishbeing}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw_data/mimic_source"
mkdir -p "$DEST"

MIMIC="https://physionet.org/files/mimiciv/3.1/hosp"
NOTE="https://physionet.org/files/mimic-iv-note/2.2/note"

# file -> URL
declare -a FILES=(
  "$MIMIC/admissions.csv.gz"        # hadm_id, admittime, dischtime, edregtime, edouttime
  "$MIMIC/procedures_icd.csv.gz"    # hadm_id, chartdate, icd_code, icd_version (intervention gate)
  "$MIMIC/d_icd_procedures.csv.gz"  # icd_code, icd_version, long_title (procedure titles)
  "$NOTE/radiology.csv.gz"          # note_id, hadm_id, charttime, text  (largest file)
)

echo "[download] user=$USER_NAME  dest=$DEST"
echo "[download] you will be asked for your PhysioNet password ONCE per file."
echo

for url in "${FILES[@]}"; do
  fname="$(basename "$url")"
  echo "==> $fname"
  wget -c --user "$USER_NAME" --ask-password -P "$DEST" "$url"
  echo
done

echo "[download] done. Files in $DEST:"
ls -lh "$DEST"
