"""patch_pe_features.py

Patch cached feature JSON files to include newly added PE features without
re-running the full (expensive) extraction pipeline.

New features added:
  - RUQ_tenderness  (from PE text; RUQ/right upper quadrant, negation-aware)
  - murphys_sign    (updated regex: now also catches inspiratory arrest /
                     catches breath paraphrases)

For each disease JSON the script:
  1. Loads the cached results/<disease>_features.json
  2. Loads the corresponding raw_data/<disease>_hadm_info_first_diag.csv
  3. For every hadm_id, calls extract_pe_signs() on the PE text column
  4. For every ExtractionStep in that hadm_id's list, merges the new/updated
     PE keys into features (all steps share the same physical exam baseline)
  5. Writes the patched JSON back in-place

Usage
-----
    python patch_pe_features.py              # patches all 4 diseases
    python patch_pe_features.py --disease cholecystitis
    python patch_pe_features.py --disease cholecystitis --dry-run
"""

from __future__ import annotations

import argparse
import json
import csv
import sys
from pathlib import Path

# ── local imports ─────────────────────────────────────────────────────────────
# Run from the repo root so that the feature_extraction package resolves.
sys.path.insert(0, str(Path(__file__).parent))
from feature_extraction.algo_extractor import extract_pe_signs  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
RESULTS_DIR = REPO_ROOT / "results"
RAW_DIR = REPO_ROOT / "raw_data"

DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]

# PE features that we update (superset of changed features; safe to refresh all)
PE_KEYS = {
    "murphys_sign",
    "RUQ_tenderness",
    "RLQ_tenderness",
    "rebound_tenderness",
    "RUQ_mass",
    "peritoneal_signs",
    "impaired_mental_status",
}


def load_csv_pe(disease: str) -> dict[str, str]:
    """Return {hadm_id: pe_text} from the raw CSV for the given disease."""
    csv_path = RAW_DIR / f"{disease}_hadm_info_first_diag.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    pe_map: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            hadm_id = str(row.get("hadm_id", "")).strip()
            pe_text = str(row.get("Physical Examination", "") or "")
            if hadm_id:
                pe_map[hadm_id] = pe_text
    return pe_map


def patch_disease(disease: str, dry_run: bool = False) -> None:
    json_path = RESULTS_DIR / f"{disease}_features.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return

    print(f"  Loading {json_path} ...", end=" ", flush=True)
    with json_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    print("done")

    pe_map = load_csv_pe(disease)
    print(f"  CSV PE text loaded for {len(pe_map)} hadm_ids")

    results: dict = data.get("results", {})
    n_patched = 0
    n_missing = 0
    n_changed = 0

    for hadm_id, steps in results.items():
        pe_text = pe_map.get(str(hadm_id), "")
        if not pe_text:
            n_missing += 1
            # Still inject defaults so the key exists
            new_pe = {k: False for k in PE_KEYS}
        else:
            new_pe = extract_pe_signs(pe_text)
            # Keep only the keys in PE_KEYS (don't touch impaired_mental_status
            # etc. if they were set differently via another source, but here we
            # want to refresh all of them consistently)
            new_pe = {k: new_pe[k] for k in PE_KEYS if k in new_pe}

        for step in steps:
            features: dict = step.get("features", {})
            # Track whether anything actually changes
            before = {k: features.get(k) for k in PE_KEYS}
            features.update(new_pe)
            after = {k: features.get(k) for k in PE_KEYS}
            if before != after:
                n_changed += 1

        n_patched += 1

    print(f"  Patched {n_patched} patients | {n_missing} with empty PE text | "
          f"{n_changed} step-level feature changes")

    if dry_run:
        print("  [DRY RUN] No file written.")
    else:
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        print(f"  Written: {json_path}")


def report_coverage(disease: str) -> None:
    """Print quick coverage stats for RUQ_tenderness and murphys_sign."""
    json_path = RESULTS_DIR / f"{disease}_features.json"
    if not json_path.exists():
        return
    with json_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    results = data.get("results", {})

    total = len(results)
    murphys = ruq_tend = group_a = 0
    for steps in results.values():
        # Use step 0 features as the ground-truth HPI+PE baseline
        step0_features = next(
            (s["features"] for s in steps if s["step_index"] == 0), {}
        )
        if step0_features.get("murphys_sign"):
            murphys += 1
        if step0_features.get("RUQ_tenderness"):
            ruq_tend += 1
        if step0_features.get("murphys_sign") or step0_features.get("RUQ_tenderness") \
                or step0_features.get("RUQ_mass"):
            group_a += 1

    print(f"\n  [{disease}] n={total}")
    print(f"    murphys_sign positive : {murphys:4d} ({100*murphys/total:.1f}%)")
    print(f"    RUQ_tenderness positive: {ruq_tend:4d} ({100*ruq_tend/total:.1f}%)")
    print(f"    TG18 Group A (any)    : {group_a:4d} ({100*group_a/total:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch cached PE features in JSON results")
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
    parser.add_argument(
        "--report",
        action="store_true",
        help="After patching, print coverage stats for key features",
    )
    args = parser.parse_args()

    diseases = DISEASES if args.disease == "all" else [args.disease]

    for disease in diseases:
        print(f"\n{'='*60}")
        print(f"Disease: {disease}")
        patch_disease(disease, dry_run=args.dry_run)

    if args.report:
        print(f"\n{'='*60}")
        print("Coverage report (post-patch):")
        for disease in diseases:
            report_coverage(disease)


if __name__ == "__main__":
    main()
