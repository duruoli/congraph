"""Select a small, untouched development-only pilot for final DIRECT workflow validation."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_aqc_discovery_sample import load_timing, profile_development, stable_hash

DEV = ROOT / "data" / "aqc_development"
FRAMEWORK = ROOT / "data" / "aqc_framework_check" / "sample_manifest.json"
OUT = ROOT / "data" / "aqc_direct" / "pilot_manifest.json"
SALT = "congraph-aqc-direct-pilot-v1"
REQUIRED_DISEASES = {"appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    split = read_json(DEV / "split_manifest.json")
    discovery = read_json(DEV / "development_sample_manifest.json")
    framework = read_json(FRAMEWORK)
    excluded = {
        (p["disease"], int(p["hadm_id"]))
        for batch in discovery["batches"] for p in batch["patients"]
    } | {
        (p["disease"], int(p["hadm_id"])) for p in framework["patients"]
    }
    final_test = {
        (p["disease"], int(p["hadm_id"]))
        for p in split["patients"] if p["partition"] == "final_test"
    }
    timing = load_timing()
    candidates = []
    for row in split["patients"]:
        key = (row["disease"], int(row["hadm_id"]))
        if row["partition"] != "development" or key in excluded:
            continue
        profile = profile_development(ROOT / row["source_path"], timing)
        profile["pilot_hash"] = stable_hash(f"{SALT}|{key[0]}|{key[1]}")
        candidates.append(profile)

    # Six cases: first guarantee all four disease strata, then add the cases that
    # contribute the most missing structural features. Hashes make ties stable.
    selected: list[dict[str, Any]] = []
    covered: set[str] = set()
    priority_flags = {"repeat", "modality_switch"}

    def score(row: dict[str, Any]) -> tuple[int, int, int, str]:
        flags = {k for k, v in row["structure_flags"].items() if v}
        missing_priority = len(priority_flags - {
            k for item in selected for k, v in item["structure_flags"].items() if v
        })
        contribution = len(set(row["selection_features"]) - covered)
        hits = len(flags & priority_flags) if missing_priority else 0
        # Prefer a single-step case until one is present, then richer trajectories.
        length_need = int(not any(item["n_steps"] == 1 for item in selected) and row["n_steps"] == 1)
        return (-length_need, -hits, -contribution, row["pilot_hash"])

    for disease in sorted(REQUIRED_DISEASES):
        pool = [r for r in candidates if r["disease"] == disease]
        best = sorted(pool, key=score)[0]
        selected.append(best)
        covered.update(best["selection_features"])

    while len(selected) < 6:
        used = {(r["disease"], r["hadm_id"]) for r in selected}
        pool = [r for r in candidates if (r["disease"], r["hadm_id"]) not in used]
        best = sorted(pool, key=score)[0]
        selected.append(best)
        covered.update(best["selection_features"])

    selected_keys = {(r["disease"], r["hadm_id"]) for r in selected}
    assert len(selected_keys) == 6
    assert {r["disease"] for r in selected} == REQUIRED_DISEASES
    assert not selected_keys & excluded
    assert not selected_keys & final_test
    assert any(r["n_steps"] == 1 for r in selected)
    assert any(r["n_steps"] > 1 for r in selected)
    assert any(r["structure_flags"]["repeat"] for r in selected)
    assert any(r["structure_flags"]["modality_switch"] for r in selected)

    manifest = {
        "schema_version": "1.0.0-development",
        "purpose": "minimal final-DIRECT prompt/model pilot",
        "source_partition": "development only",
        "selection_salt": SALT,
        "exclusions": {
            "codebook_discovery_patients": 48,
            "framework_check_patients": 16,
            "final_test": "all patients",
        },
        "selection_requirements": [
            "six untouched development patients",
            "all four disease sampling strata",
            "single-step and multi-step trajectories",
            "at least one repeat trajectory",
            "at least one modality-switch trajectory",
        ],
        "n_patients": 6,
        "n_steps": sum(r["n_steps"] for r in selected),
        "patients": selected,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: 6 patients/{manifest['n_steps']} steps")


if __name__ == "__main__":
    main()
