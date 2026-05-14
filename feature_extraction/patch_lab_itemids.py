"""patch_lab_itemids.py

Patch cached feature JSON files to inject `lab_itemids` extracted from
the raw CSV `Laboratory Tests` column into Step 0 of the features array.
"""

from __future__ import annotations

import argparse
import json
import csv
import sys
from pathlib import Path

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_extraction.algo_extractor import _parse_lab_json

# ── constants ─────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
RAW_DIR     = REPO_ROOT / "data" / "raw_data"

DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]


def _load_csv_rows(disease: str) -> dict[str, dict]:
    """Return {hadm_id: row_dict} from the raw CSV."""
    csv_path = RAW_DIR / f"{disease}_hadm_info_first_diag.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rows: dict[str, dict] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            hadm_id = str(row.get("hadm_id", "")).strip()
            if hadm_id:
                rows[hadm_id] = row
    return rows


def patch_disease(disease: str, dry_run: bool = False) -> None:
    json_path = RESULTS_DIR / f"{disease}_features.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return

    print(f"  Loading {json_path} ...", end=" ", flush=True)
    with json_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    print("done")

    csv_rows = _load_csv_rows(disease)
    print(f"  CSV loaded: {len(csv_rows)} patients")

    results: dict = data.get("results", {})
    n_patched = 0
    n_missing = 0
    n_step_changes = 0

    for hadm_id, steps in results.items():
        row = csv_rows.get(str(hadm_id))
        if row is None:
            n_missing += 1
            new_lab_itemids = []
        else:
            labs_raw = str(row.get("Laboratory Tests", "") or "")
            labs = _parse_lab_json(labs_raw)
            try:
                new_lab_itemids = sorted(labs.keys(), key=lambda x: int(str(x)))
            except (ValueError, TypeError):
                new_lab_itemids = sorted(map(str, labs.keys()))

        # Update lab_itemids only in the first step (algo step) or all steps?
        # Following patch_pe_features strategy: we overwrite all steps with the freshly re-extracted values
        for step in steps:
            features: dict = step.get("features", {})
            old_val = features.get("lab_itemids", [])
            if old_val != new_lab_itemids:
                features["lab_itemids"] = new_lab_itemids
                n_step_changes += 1

        n_patched += 1

    print(f"  Patients patched : {n_patched}  |  missing CSV rows : {n_missing}")
    print(f"  Step-level changes: {n_step_changes}")

    if dry_run:
        print("  [DRY RUN] No file written.")
    else:
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        print(f"  Written: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch cached feature JSONs with lab_itemids from raw CSV"
    )
    parser.add_argument(
        "--disease",
        choices=DISEASES + ["all"],
        default="all",
        help="Which disease JSON to patch (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files",
    )
    args = parser.parse_args()

    diseases = DISEASES if args.disease == "all" else [args.disease]

    for disease in diseases:
        print(f"\n{'='*60}")
        print(f"Disease: {disease}")
        patch_disease(disease, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
