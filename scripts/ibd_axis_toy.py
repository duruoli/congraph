#!/usr/bin/env python3
"""Render a private, manually annotated IBD radiology trajectory toy experiment.

The case configuration and output belong under data/raw_data/ (gitignored), since
they describe one MIMIC patient. This script never prints source note text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import pyarrow.dataset as ds
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw_data/ibd_mimiciv_3_1"
AXES = ROOT / "unified_axes_v1.yml"
WEIGHTS = {
    "disease": 4,
    "management_domain": 3,
    "clinical_phase": 2,
    "diagnostic_modality": 2,
    "intervention_class": 2,
    "setting": 1,
    "population": 1,
    "anatomy": 1,
}


def load_vocabulary() -> dict[str, dict]:
    raw = yaml.safe_load(AXES.read_text(encoding="utf-8"))
    return {axis["id"]: axis for axis in raw["axes"]}


def validate_axes(axes: dict, vocabulary: dict[str, dict]) -> None:
    if set(axes) != set(vocabulary):
        raise ValueError(f"Eight axes required: missing={set(vocabulary)-set(axes)}, extra={set(axes)-set(vocabulary)}")
    for axis, values in axes.items():
        if values is None:  # Unknown: evidence is insufficient.
            continue
        if values == "n_a":  # Explicitly inactive, never a stand-in for unknown.
            if not vocabulary[axis]["allows_na"]:
                raise ValueError(f"{axis} does not permit n_a")
            continue
        if not isinstance(values, list) or not values:
            raise ValueError(f"{axis}: expected nonempty list, n_a, or null")
        allowed = {item["id"] for item in vocabulary[axis]["values"]}
        if len(values) != len(set(values)) or any(value not in allowed for value in values):
            raise ValueError(f"{axis}: invalid or duplicate value in {values}")


def load_case(hadm_id: int) -> tuple[dict, list[dict]]:
    cohort = ds.dataset(DATA / "cohort", format="parquet").to_table(
        filter=ds.field("hadm_id") == hadm_id
    ).to_pylist()
    if len(cohort) != 1:
        raise ValueError(f"Expected exactly one cohort row, got {len(cohort)}")
    reports = ds.dataset(DATA / "radiology", format="parquet").to_table(
        filter=ds.field("hadm_id") == hadm_id
    ).to_pylist()
    rr = sorted(
        (row for row in reports if row["note_type"] == "RR"),
        key=lambda row: (row["charttime"], row["note_id"]),
    )
    return cohort[0], rr


def load_article_index(jats_path: Path) -> tuple[dict[str, set[str]], dict[str, str], dict[str, str]]:
    root = ET.parse(jats_path).getroot()
    paragraphs = {
        p.get("id"): " ".join("".join(p.itertext()).split())
        for p in root.findall(".//p") if p.get("id")
    }
    labels: dict[str, set[str]] = defaultdict(set)
    for meta in root.findall(".//custom-meta[@specific-use='nigel-axis-assignment']"):
        name = meta.findtext("meta-name")
        detail = meta.findtext("meta-value") or ""
        if "evidence=" not in detail:
            continue
        for pointer in detail.split("evidence=", 1)[1].split(","):
            if pointer in paragraphs:
                labels[pointer].add(name)
    envelope = {
        meta.findtext("meta-name"): meta.findtext("meta-value") or ""
        for meta in root.findall(".//custom-meta-group[@specific-use='nigel-document-envelope']/custom-meta")
    }
    return labels, paragraphs, envelope


def format_axis(value: list[str] | str | None) -> str:
    if value is None:
        return "unknown"
    if value == "n_a":
        return "n_a"
    return ", ".join(value)


def format_phase(axes: dict, assertion: str) -> str:
    value = format_axis(axes["clinical_phase"])
    return f"{value} ({assertion})" if assertion else value


def observed_modality(exam_name: str) -> tuple[str | None, str]:
    upper = exam_name.upper()
    if "MR ENTEROGRAPH" in upper or re.search(r"\bMRI?\b", upper):
        return "cross_sectional_imaging_mri", "mapped"
    if re.search(r"\bCT\b", upper):
        return "cross_sectional_imaging_ct", "mapped"
    if "ABDOMEN" in upper and ("SUPINE" in upper or "ERECT" in upper):
        return None, "unmapped: abdominal radiograph is absent from unified_axes_v1"
    return None, "unmapped or ambiguous exam"


def top_matches(axes: dict, labels: dict[str, set[str]], limit: int = 3) -> list[tuple[int, str, list[str]]]:
    query = {axis: set(values) for axis, values in axes.items() if isinstance(values, list)}
    scored = []
    for paragraph_id, paragraph_labels in labels.items():
        exact = [
            f"{axis}:{value}"
            for axis, values in query.items()
            for value in values
            if value != "all" and f"{axis}:{value}" in paragraph_labels
        ]
        if exact:
            score = sum(WEIGHTS[item.split(":", 1)[0]] for item in exact)
            scored.append((score, paragraph_id, sorted(exact)))
    return sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jats", type=Path, help="Nigel labeled JATS file (defaults to the sole matching file in the workspace root)")
    args = parser.parse_args()
    if DATA not in args.config.resolve().parents or DATA not in args.out.resolve().parents:
        parser.error("Case configuration and report must stay under the gitignored IBD raw-data directory")

    candidates = [args.jats] if args.jats else list(ROOT.glob("*corpus.labeled.current.jats.xml"))
    if len(candidates) != 1 or not candidates[0].is_file():
        parser.error("Specify exactly one existing labeled JATS with --jats")
    jats_path = candidates[0]

    config = json.loads(args.config.read_text(encoding="utf-8"))
    jats_sha256 = hashlib.sha256(jats_path.read_bytes()).hexdigest()
    source_review = config["source_review"]
    if source_review["jats_sha256"] != jats_sha256:
        raise ValueError("Manual source review is for a different JATS version")
    vocabulary = load_vocabulary()
    cohort, reports = load_case(config["hadm_id"])
    annotations = config["steps"]
    if len(annotations) != len(reports):
        raise ValueError("Every RR report must have exactly one step annotation")
    article_labels, paragraphs, envelope = load_article_index(jats_path)
    modality_terms = re.compile(r"\b(radiograph|imaging|computed tomography|CT scan|MRI|MR enterography)\b", re.I)
    modality_mentions = [pid for pid, text in paragraphs.items() if modality_terms.search(text)]

    lines = [
        "# IBD eight-axis radiology toy trajectory (local, restricted)",
        "",
        f"Case alias: `{config['alias']}`. One admission, {len(reports)} routine radiology reports.",
        "Patient identifiers and verbatim note text are omitted. Dates below are days since admission.",
        "This illustrative case was selected for an explicit UC imaging question and a multi-step sequence; it is not a representative sample.",
        "",
        "## Interpretation rules",
        "",
        "- The primary card masks the **entire target report**, including its indication. The observed exam is an outcome, shown separately.",
        "- Prior report content is eligible only if `storetime < target charttime`; exact order times are unavailable, so this is an upper bound on potentially available information, not proven pre-order availability.",
        "- A second, retrospective card uses the target report's indication as a proxy for the clinical question. It must not be used as a prospective input.",
        "- These are manual toy annotations. Contemporaneous labs and clinical notes have not yet been integrated.",
        "- `unknown` means insufficient evidence; `n_a` means this imaging decision does not activate that axis.",
        "- Cohort ICD disease type is an outcome label, not proof that disease was known at an earlier decision.",
        "",
        "## Target-report-masked eight-axis cards",
        "",
        "| Step / day | Disease | Management domain | Intervention class | Diagnostic modality (pre-action) | Clinical phase (assertion) | Population | Setting | Anatomy | Observed next exam |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    records = []
    for index, (annotation, report) in enumerate(zip(annotations, reports, strict=True), 1):
        if annotation["exam_expect"] != report["exam_name"]:
            raise ValueError(f"Step {index}: exam changed; annotations need review")
        axes = annotation["pre_action_axes"]
        validate_axes(axes, vocabulary)
        validate_axes(annotation["axes"], vocabulary)
        days = (report["charttime"] - cohort["admittime"]).total_seconds() / 86400
        available_prior = [
            prior for prior in reports[: index - 1]
            if prior["storetime"] is not None and prior["storetime"] < report["charttime"]
        ]
        mapped, mapping_status = observed_modality(report["exam_name"])
        matches = top_matches(axes, article_labels)
        cells = [f"{index} / {days:.1f}"]
        cells += [
            format_phase(axes, annotation.get("pre_action_phase_assertion", ""))
            if axis == "clinical_phase" else format_axis(axes[axis])
            for axis in vocabulary
        ]
        cells.append(report["exam_name"])
        lines.append("| " + " | ".join(cells) + " |")
        records.append((index, annotation, report, available_prior, mapped, mapping_status, matches))

    lines += [
        "",
        "## What the target report's indication adds retrospectively",
        "",
        "| Step | Management domain with target indication | Clinical phase with target indication |",
        "|---|---|---|",
    ]
    for index, annotation, *_ in records:
        retrospective = annotation["axes"]
        lines.append(
            f"| {index} | {format_axis(retrospective['management_domain'])} | "
            f"{format_phase(retrospective, annotation.get('phase_assertion', ''))} |"
        )

    lines += ["", "## Evidence and retrieval by step", ""]
    for index, annotation, report, available_prior, mapped, mapping_status, matches in records:
        lines += [
            f"### Step {index}: {report['exam_name']}",
            "",
            f"- Target-report-masked basis: {annotation['pre_action_basis']}",
            f"- Retrospective clinical question (target-report indication proxy): {annotation['clinical_question']}",
            f"- Retrospective interpretation basis: {annotation['axis_basis']}",
            f"- Prior reports available by storetime: {len(available_prior)} / {index - 1}.",
            f"- Observed action modality: `{mapped or 'unmapped'}` ({mapping_status}).",
            "- Top thematic matches in the available Nigel JATS (weighted exact-axis overlap; not recommendations): "
            + ("; ".join(f"`{pid}` score={score} ({', '.join(shared)})" for score, pid, shared in matches) if matches else "none"),
            "- Deviation status: " + (
                "**not assessable**; the manually reviewed source has no next-radiology-action recommendation."
                if not source_review["has_radiology_action_recommendations"]
                else "**not yet adjudicated**; applicability and recommendation extraction require review."
            ),
            "",
        ]

    lines += [
        "## Source and vocabulary audit",
        "",
        f"- Nigel JATS SHA-256: `{jats_sha256}`.",
        f"- Unified axis YAML SHA-256: `{hashlib.sha256(AXES.read_bytes()).hexdigest()}`.",
        f"- JATS envelope `document_type={envelope.get('document_type', 'missing')}`; the article body identifies it as a review. Verify this metadata before source-type filtering.",
        f"- Paragraphs mentioning radiography, CT, MRI, or imaging by literal term search: {len(modality_mentions)}. This is a screening check, not a formal recommendation extraction.",
        f"- Manual source review: {source_review['note']}",
        "- Five observed abdominal radiograph actions have no corresponding `diagnostic_modality` value in `unified_axes_v1.yml`; keep their raw action names instead of assigning CT/MRI or `n_a`.",
        "- This toy establishes shared-vocabulary indexing and its abstention behavior. It cannot measure clinical deviation until an applicable, time-valid action recommendation is supplied.",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote restricted report: {args.out}")
    print(f"Steps={len(records)}; prior-report availability={[len(row[3]) for row in records]}; JATS radiology-term paragraphs={len(modality_mentions)}")


if __name__ == "__main__":
    main()
