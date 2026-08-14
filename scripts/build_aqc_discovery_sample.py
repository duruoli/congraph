"""Build the frozen Track-B discovery sample without altering source annotations.

The source corpus is the rubric-free, order-aware Mode-A reconstruction corpus in
``results/annotation_experiment``.  Selection is deliberately fixed (rather than
random) so the qualitative open coding is reproducible and reviewable.

This script copies the assumption-bearing source fields verbatim and attaches
provisional *open* type codes.  It does not create final A/Q/C labels and never
loads the ACR normative corpus.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "aqc_development"

# Four trajectories per disease.  The set deliberately includes single- and
# multi-step workups, repeats/switches, broad searches, disease confirmation,
# etiology, severity, complications, and intervention/device follow-up.
SAMPLE: list[tuple[str, int, str]] = [
    ("appendicitis", 23202997, "results/annotation_experiment/appendicitis_23202997.json"),
    ("appendicitis", 21543797, "results/annotation_experiment/batch/appendicitis_21543797.json"),
    ("appendicitis", 28722849, "results/annotation_experiment/batch/appendicitis_28722849.json"),
    ("appendicitis", 29794234, "results/annotation_experiment/batch/appendicitis_29794234.json"),
    ("cholecystitis", 29573603, "results/annotation_experiment/cholecystitis_29573603.json"),
    ("cholecystitis", 25217286, "results/annotation_experiment/batch/cholecystitis_25217286.json"),
    ("cholecystitis", 25290083, "results/annotation_experiment/batch/cholecystitis_25290083.json"),
    ("cholecystitis", 28137567, "results/annotation_experiment/batch/cholecystitis_28137567.json"),
    ("diverticulitis", 26371704, "results/annotation_experiment/diverticulitis_26371704.json"),
    ("diverticulitis", 27675389, "results/annotation_experiment/diverticulitis_27675389.json"),
    ("diverticulitis", 22429578, "results/annotation_experiment/batch/diverticulitis_22429578.json"),
    ("diverticulitis", 24910158, "results/annotation_experiment/batch/diverticulitis_24910158.json"),
    ("pancreatitis", 21282967, "results/annotation_experiment/pancreatitis_21282967.json"),
    ("pancreatitis", 20418179, "results/annotation_experiment/batch/pancreatitis_20418179.json"),
    ("pancreatitis", 20720063, "results/annotation_experiment/batch/pancreatitis_20720063.json"),
    ("pancreatitis", 24449784, "results/annotation_experiment/batch/pancreatitis_24449784.json"),
]

# Manual first-cycle open codes, one list per decision step.  These are text-level
# codes over the copied source material, not frozen ontology labels.  A later
# paired pilot must atomize propositions and assign status per proposition.
OPEN_CODES: dict[tuple[str, int], list[list[str]]] = {
    ("appendicitis", 23202997): [
        ["disease_or_finding_identity", "alternative_source"],
        ["syndrome_or_source_frame", "alternative_source"],
        ["complication", "severity_extent_or_course"],
        ["severity_extent_or_course", "complication"],
    ],
    ("appendicitis", 21543797): [
        ["syndrome_or_source_frame", "alternative_source", "complication"],
    ],
    ("appendicitis", 28722849): [
        ["disease_or_finding_identity", "alternative_source"],
        ["disease_or_finding_identity", "complication", "alternative_source"],
    ],
    ("appendicitis", 29794234): [
        ["disease_or_finding_identity", "alternative_source"],
        ["alternative_source", "disease_or_finding_identity"],
        ["disease_or_finding_identity", "complication", "severity_extent_or_course"],
    ],
    ("cholecystitis", 29573603): [
        ["complication", "disease_or_finding_identity", "intervention_or_device_state"],
        ["disease_or_finding_identity", "complication", "intervention_or_device_state"],
        ["severity_extent_or_course", "complication"],
        ["severity_extent_or_course", "complication"],
        ["severity_extent_or_course", "complication", "intervention_or_device_state"],
        ["severity_extent_or_course", "complication", "alternative_source"],
    ],
    ("cholecystitis", 25217286): [
        ["disease_or_finding_identity", "alternative_source"],
        ["alternative_source", "syndrome_or_source_frame"],
    ],
    ("cholecystitis", 25290083): [
        ["disease_or_finding_identity", "severity_extent_or_course", "alternative_source"],
        ["disease_or_finding_identity", "severity_extent_or_course"],
    ],
    ("cholecystitis", 28137567): [
        ["intervention_or_device_state", "severity_extent_or_course", "complication"],
        ["etiology_or_mechanism", "intervention_or_device_state"],
        ["complication", "intervention_or_device_state", "alternative_source"],
        ["disease_or_finding_identity", "complication", "intervention_or_device_state"],
    ],
    ("diverticulitis", 26371704): [
        ["syndrome_or_source_frame", "disease_or_finding_identity", "complication"],
        ["severity_extent_or_course", "complication", "intervention_or_device_state"],
        ["alternative_source", "disease_or_finding_identity"],
        ["severity_extent_or_course", "complication", "intervention_or_device_state"],
    ],
    ("diverticulitis", 27675389): [
        ["disease_or_finding_identity", "alternative_source"],
        ["severity_extent_or_course", "complication"],
        ["severity_extent_or_course", "complication", "alternative_source"],
    ],
    ("diverticulitis", 22429578): [
        ["disease_or_finding_identity", "alternative_source"],
        ["alternative_source", "disease_or_finding_identity"],
    ],
    ("diverticulitis", 24910158): [
        ["disease_or_finding_identity", "alternative_source"],
        ["disease_or_finding_identity", "alternative_source"],
    ],
    ("pancreatitis", 21282967): [
        ["syndrome_or_source_frame", "etiology_or_mechanism", "alternative_source"],
        ["syndrome_or_source_frame", "alternative_source", "complication"],
        ["severity_extent_or_course", "complication"],
        ["severity_extent_or_course", "complication", "alternative_source"],
        ["etiology_or_mechanism", "alternative_source"],
        ["etiology_or_mechanism", "disease_or_finding_identity"],
        ["severity_extent_or_course", "etiology_or_mechanism", "complication"],
    ],
    ("pancreatitis", 20418179): [
        ["disease_or_finding_identity", "etiology_or_mechanism", "alternative_source"],
        ["severity_extent_or_course", "complication", "alternative_source"],
    ],
    ("pancreatitis", 20720063): [
        ["disease_or_finding_identity", "severity_extent_or_course", "etiology_or_mechanism", "complication"],
        ["etiology_or_mechanism"],
        ["etiology_or_mechanism", "disease_or_finding_identity"],
    ],
    ("pancreatitis", 24449784): [
        ["disease_or_finding_identity", "alternative_source", "syndrome_or_source_frame"],
    ],
}


def load_source(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "steps" not in data:
        raise ValueError(f"not an annotation trajectory: {relative_path}")
    return data


def profile_available_corpus() -> dict[str, Any]:
    """Profile unique root-pilot + batch trajectories available for discovery."""
    paths = sorted((ROOT / "results" / "annotation_experiment").glob("*.json"))
    paths += sorted((ROOT / "results" / "annotation_experiment" / "batch").glob("*.json"))
    seen: set[tuple[str, int]] = set()
    disease_counts: Counter[str] = Counter()
    n_steps = 0
    for path in paths:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not {"disease", "hadm_id", "steps"}.issubset(data):
            continue
        identity = (str(data["disease"]), int(data["hadm_id"]))
        if identity in seen:
            continue
        seen.add(identity)
        disease_counts[identity[0]] += 1
        n_steps += len(data["steps"])
    return {
        "n_unique_trajectories": len(seen),
        "n_decision_steps": n_steps,
        "trajectory_counts_by_disease": dict(sorted(disease_counts.items())),
        "search_scope": [
            "results/annotation_experiment/*.json",
            "results/annotation_experiment/batch/*.json",
        ],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()
    code_counts: Counter[str] = Counter()
    verification_counts: Counter[str] = Counter()
    disease_counts: Counter[str] = Counter()

    for disease, hadm_id, source_path in SAMPLE:
        source = load_source(source_path)
        if (source.get("disease"), source.get("hadm_id")) != (disease, hadm_id):
            raise ValueError(f"identity mismatch in {source_path}")
        codes_by_step = OPEN_CODES[(disease, hadm_id)]
        if len(codes_by_step) != len(source["steps"]):
            raise ValueError(f"open-code length mismatch for {disease} {hadm_id}")
        disease_counts[disease] += 1

        for step, codes in zip(source["steps"], codes_by_step, strict=True):
            ex_ante = step.get("representative_ex_ante") or {}
            verification = step.get("verification") or {}
            role = str(ex_ante.get("action_role", ""))
            verification_label = str(verification.get("verification", ""))
            role_counts[role] += 1
            verification_counts[verification_label] += 1
            code_counts.update(codes)
            rows.append({
                "sample_id": f"{disease}:{hadm_id}:s{step['step']}",
                "disease_stratum": disease,
                "hadm_id": hadm_id,
                "step": step["step"],
                "trajectory_length": len(source["steps"]),
                "source_path": source_path,
                "ordered": step.get("ordered", ""),
                "n_visible_prior": step.get("n_visible_prior", 0),
                "verbatim_assumption_material": {
                    "differential": ex_ante.get("differential", {}),
                    "other_hypothesis": ex_ante.get("other_hypothesis", ""),
                    "information_gap": ex_ante.get("information_gap", ""),
                    "expected_finding": ex_ante.get("expected_finding", ""),
                    "reasoning": ex_ante.get("reasoning", ""),
                },
                "source_action_role": role,
                "open_type_codes": codes,
                "coding_scope": "text-level first-cycle code; not a frozen A/Q/C label",
            })

    output_path = OUT_DIR / "discovery_open_coding.jsonl"
    with output_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": "0.1.0-provisional",
        "source_boundary": "schema-free empirical annotations only; ACR not loaded",
        "selection_method": "fixed maximum-variation purposive sample",
        "available_corpus_profile": profile_available_corpus(),
        "selection_criteria": [
            "exactly four trajectories per disease stratum",
            "include single-step and multi-step trajectories",
            "include repeat/switch, broad differential, rule-in/out, severity, complication, and intervention follow-up",
            "retain confirmed, disconfirmed, and uninformative local verification outcomes",
        ],
        "n_trajectories": len(SAMPLE),
        "n_steps": len(rows),
        "trajectory_counts_by_disease": dict(sorted(disease_counts.items())),
        "step_counts_by_action_role": dict(sorted(role_counts.items())),
        "step_counts_by_verification": dict(sorted(verification_counts.items())),
        "open_code_assignments": dict(sorted(code_counts.items())),
        "trajectory_sources": [
            {"disease": disease, "hadm_id": hadm_id, "source_path": source_path}
            for disease, hadm_id, source_path in SAMPLE
        ],
        "output": str(output_path.relative_to(ROOT)),
    }
    (OUT_DIR / "discovery_sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"wrote {len(rows)} steps from {len(SAMPLE)} trajectories to {output_path}")


if __name__ == "__main__":
    main()
