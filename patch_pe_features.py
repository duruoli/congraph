"""patch_pe_features.py

Patch cached feature JSON files to re-apply algorithmic (PE + HPI) extraction
without re-running the full (expensive) LLM pipeline.

Changes applied by the current version
---------------------------------------
Round 1 (original):
  - RUQ_tenderness  (from PE text; RUQ/right upper quadrant, negation-aware)
  - murphys_sign    (updated regex: now also catches inspiratory arrest /
                     catches breath paraphrases)

Round 2 (current):
  - PE signs (all): pre-window negation bleed fix — "no X, no Y, positive Z"
    was incorrectly negating Z; now clipped at last comma/semicolon/colon.
    Affects: murphys_sign, RUQ_tenderness, RLQ_tenderness, rebound_tenderness,
             RUQ_mass, peritoneal_signs.
  - HPI symptom flags: same negation bleed fix applied to comma-delimited lists
    in HPI text.  Affects: bowel_habit_change, anorexia, nausea_vomiting,
    alcohol_history, gallstone_history, prior_diverticular_disease.
  - fever_reported_in_hpi (NEW key): HPI narrative fever flag — True when the
    HPI positively mentions "fever"/"febrile" or a quantified Tmax ≥ 38 °C.
    Supplements fever_temp_ge_38 (admission PE vitals) for TG18 Group B.

Strategy
--------
PE signs and HPI flags are identical across all ExtractionSteps for a given
patient (LLM imaging extraction only writes imaging keys, never PE/HPI keys).
Therefore we simply overwrite all steps with the freshly re-extracted values —
no need to separate "imaging contribution" from "algo contribution".

Usage
-----
    python patch_pe_features.py              # patches all 4 diseases
    python patch_pe_features.py --disease cholecystitis
    python patch_pe_features.py --disease cholecystitis --dry-run
    python patch_pe_features.py --disease cholecystitis --report
"""

from __future__ import annotations

import argparse
import json
import csv
import sys
from pathlib import Path

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from feature_extraction.algo_extractor import (  # noqa: E402
    extract_pe_signs,
    extract_hpi_features,
)

# ── constants ─────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent
RESULTS_DIR = REPO_ROOT / "results"
RAW_DIR     = REPO_ROOT / "raw_data"

DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]

# Keys refreshed from PE text (negation bleed fix)
PE_KEYS: frozenset[str] = frozenset({
    "murphys_sign",
    "RUQ_tenderness",
    "RLQ_tenderness",
    "rebound_tenderness",
    "RUQ_mass",
    "peritoneal_signs",
    "impaired_mental_status",
})

# Keys refreshed from HPI text (negation bleed fix + new fever_reported_in_hpi)
HPI_KEYS: frozenset[str] = frozenset({
    "bowel_habit_change",
    "anorexia",
    "nausea_vomiting",
    "alcohol_history",
    "gallstone_history",
    "prior_diverticular_disease",
    "fever_reported_in_hpi",       # NEW — added in Round 2
})

ALL_ALGO_KEYS = PE_KEYS | HPI_KEYS


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
    key_delta: dict[str, int] = {}   # per-key change count

    for hadm_id, steps in results.items():
        row = csv_rows.get(str(hadm_id))
        if row is None:
            n_missing += 1
            new_algo = {k: False for k in ALL_ALGO_KEYS}
        else:
            pe_text  = str(row.get("Physical Examination", "") or "")
            hpi_text = str(row.get("Patient History", "") or "")

            new_pe  = {k: v for k, v in extract_pe_signs(pe_text).items()   if k in PE_KEYS}
            new_hpi = {k: v for k, v in extract_hpi_features(hpi_text).items() if k in HPI_KEYS}
            new_algo = {**new_pe, **new_hpi}

            # Ensure new key always present even when extractor doesn't emit it
            new_algo.setdefault("fever_reported_in_hpi", False)

        for step in steps:
            features: dict = step.get("features", {})
            for key, new_val in new_algo.items():
                old_val = features.get(key)
                if old_val != new_val:
                    features[key] = new_val
                    n_step_changes += 1
                    key_delta[key] = key_delta.get(key, 0) + 1
                elif key not in features:
                    # Key absent (fever_reported_in_hpi in old JSON) — inject default
                    features[key] = new_val

        n_patched += 1

    print(f"  Patients patched : {n_patched}  |  missing CSV rows : {n_missing}")
    print(f"  Step-level changes: {n_step_changes}")
    if key_delta:
        print("  Per-key changes (step-level counts):")
        for k, cnt in sorted(key_delta.items(), key=lambda x: -x[1]):
            print(f"    {k:40s}: {cnt}")

    if dry_run:
        print("  [DRY RUN] No file written.")
    else:
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        print(f"  Written: {json_path}")


def report_coverage(disease: str) -> None:
    """Print coverage stats for key TG18 Group A/B features."""
    json_path = RESULTS_DIR / f"{disease}_features.json"
    if not json_path.exists():
        return
    with json_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    results = data.get("results", {})
    total = len(results)
    if total == 0:
        return

    counts: dict[str, int] = {}
    check_keys = [
        "murphys_sign", "RUQ_tenderness", "RUQ_mass",
        "fever_temp_ge_38", "fever_reported_in_hpi",
        "WBC_gt_10k", "CRP_elevated",
    ]
    group_a = group_b = both_zero = 0
    for steps in results.values():
        f = next((s["features"] for s in steps if s["step_index"] == 0), {})
        for k in check_keys:
            if f.get(k):
                counts[k] = counts.get(k, 0) + 1
        a = f.get("murphys_sign") or f.get("RUQ_tenderness") or f.get("RUQ_mass")
        b = (f.get("fever_temp_ge_38") or f.get("fever_reported_in_hpi")
             or f.get("CRP_elevated") or f.get("WBC_gt_10k"))
        if a: group_a += 1
        if b: group_b += 1
        if not a and not b: both_zero += 1

    print(f"\n  [{disease}]  n={total}")
    for k in check_keys:
        c = counts.get(k, 0)
        print(f"    {k:40s}: {c:4d} ({100*c/total:.1f}%)")
    print(f"    {'TG18 Group A (any local sign)':40s}: {group_a:4d} ({100*group_a/total:.1f}%)")
    print(f"    {'TG18 Group B (any systemic sign)':40s}: {group_b:4d} ({100*group_b/total:.1f}%)")
    print(f"    {'Both A=0 AND B=0 (not suspected)':40s}: {both_zero:4d} ({100*both_zero/total:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-apply algo (PE + HPI) extraction to cached feature JSONs"
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
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print coverage stats for key features after patching",
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
