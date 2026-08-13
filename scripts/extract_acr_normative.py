#!/usr/bin/env python3
"""Extract faithful ACR topic tables and narrative rationale with provenance.

The extraction deliberately mirrors ACR's own variant/procedure organization.  It
does not contain patient-level A/Q/C fields or infer a diagnostic pathway.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pdfplumber


TOPICS = (
    {
        "slug": "right_lower_quadrant_pain",
        "topic_id": 21,
        "title": "Right Lower Quadrant Pain",
        "topic_version": "Revised 2022",
    },
    {
        "slug": "right_upper_quadrant_pain",
        "topic_id": 132,
        "title": "Right Upper Quadrant Pain",
        "topic_version": "Revised 2022",
    },
    {
        "slug": "left_lower_quadrant_pain",
        "topic_id": 20,
        "title": "Left Lower Quadrant Pain",
        "topic_version": "Revised 2023",
    },
    {
        "slug": "acute_pancreatitis",
        "topic_id": 126,
        "title": "Acute Pancreatitis",
        "topic_version": "New 2019",
    },
)

# Two procedure strings are visually split by intervening table columns in the
# generated appendix PDF, defeating normalized text search.  These page values
# were checked against the rendered pages and are kept explicit/auditable.
APPENDIX_PAGE_OVERRIDES = {
    (21, 3, "MRI abdomen and pelvis without and with IV contrast"): 10,
    (126, 1, "CT abdomen and pelvis without IV contrast"): 2,
}

EXPLICIT_ACTION_RELATIONSHIPS = {
    (21, 3): ("equivalent_alternatives", (
        "US abdomen", "MRI abdomen and pelvis without IV contrast",
    )),
    (132, 1): ("equivalent_alternatives", (
        "US abdomen", "CT abdomen with IV contrast",
    )),
    (132, 3): ("equivalent_alternatives", (
        "MRI abdomen without and with IV contrast with MRCP",
        "CT abdomen with IV contrast", "MRI abdomen without IV contrast with MRCP",
    )),
    (132, 4): ("equivalent_alternatives", (
        "MRI abdomen without and with IV contrast with MRCP",
        "CT abdomen with IV contrast", "HIDA scan",
    )),
    (126, 2): ("complementary", (
        "CT abdomen and pelvis with IV contrast",
        "MRI abdomen without and with IV contrast with MRCP",
    )),
    (126, 3): ("complementary", (
        "CT abdomen and pelvis with IV contrast",
        "MRI abdomen without and with IV contrast with MRCP",
    )),
    (126, 4): ("complementary", (
        "CT abdomen and pelvis with IV contrast",
        "MRI abdomen without and with IV contrast with MRCP",
    )),
    (126, 6): ("complementary", (
        "CT abdomen and pelvis with IV contrast",
        "MRI abdomen without and with IV contrast with MRCP",
    )),
}

BASE = "https://acsearch.acr.org/list"


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class AppendixParser(HTMLParser):
    """Small purpose-built parser for the stable ACR appendix table markup."""

    def __init__(self) -> None:
        super().__init__()
        self.variants: list[dict[str, Any]] = []
        self._variant: dict[str, Any] | None = None
        self._row_class = ""
        self._cell_class: str | None = None
        self._cell_parts: list[str] = []
        self._cells: list[tuple[str, str]] = []
        self._in_h5 = False
        self._h5_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "h5":
            self._in_h5 = True
            self._h5_parts = []
        elif tag == "tr":
            self._row_class = attrs_dict.get("class", "") or ""
            self._cells = []
        elif tag == "td" and self._row_class:
            self._cell_class = attrs_dict.get("class", "") or ""
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_h5:
            self._h5_parts.append(data)
        if self._cell_class is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "h5" and self._in_h5:
            text = clean("".join(self._h5_parts))
            match = re.match(r"Variant\s+(\d+):\s*(.*)", text)
            if match:
                self._variant = {
                    "variant_id": int(match.group(1)),
                    "variant_text": match.group(2),
                    "actions": [],
                }
                self.variants.append(self._variant)
            self._in_h5 = False
        elif tag == "td" and self._cell_class is not None:
            self._cells.append((self._cell_class, clean("".join(self._cell_parts))))
            self._cell_class = None
            self._cell_parts = []
        elif tag == "tr" and self._row_class:
            if self._variant is not None and "procedurestr" in self._row_class:
                values: dict[str, str] = {}
                finals: list[int] = []
                for cls, value in self._cells:
                    if "Procedure" in cls:
                        values["procedure"] = value
                    elif "ratingResult" in cls:
                        values["appropriateness_category"] = value
                    elif cls == "SOE":
                        values["strength_of_evidence"] = value.removesuffix("References").strip()
                    elif cls == "RRL":
                        values["adult_rrl"] = value
                    elif cls == "PedRRL":
                        values["pediatric_rrl"] = value or None
                    elif "MedianRating" in cls:
                        values["median_rating"] = value
                    elif re.search(r"(^|\s)rating(\s|$)", cls):
                        values["rating"] = value
                    elif "finaltabulationsTds" in cls:
                        finals.append(int(value))
                values["rating"] = int(values["rating"])
                median = values["median_rating"]
                values["median_rating"] = int(median) if median.isdigit() else median
                values["final_tabulations"] = {str(i): n for i, n in enumerate(finals, 1)}
                values["evidence_references"] = []
                self._variant["actions"].append(values)
            elif self._variant is not None and "reftbltr" in self._row_class:
                cells = [value for _, value in self._cells]
                if len(cells) >= 2 and self._variant["actions"]:
                    ref_match = re.match(r"(\d+)\s*\(([-\d]+)\)", cells[0])
                    if ref_match:
                        pmid = ref_match.group(2)
                        self._variant["actions"][-1]["evidence_references"].append(
                            {
                                "reference_number": int(ref_match.group(1)),
                                "pmid_as_printed": pmid,
                                "study_quality_as_printed": cells[1],
                            }
                        )
            self._row_class = ""
            self._cells = []


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "congraph-acr-extraction/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def pdf_pages(path: Path, include_left_column: bool = False) -> list[str]:
    with pdfplumber.open(path) as pdf:
        pages: list[str] = []
        for page in pdf.pages:
            full = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            if include_left_column:
                left = page.crop((0, 0, page.width * 0.26, page.height)).extract_text(
                    x_tolerance=2, y_tolerance=3
                ) or ""
                full += "\n[[LEFT COLUMN]]\n" + left
            pages.append(full)
        return pages


def find_appendix_page(pages: list[str], variant_id: int, procedure: str) -> int | None:
    variant_re = re.compile(rf"Variant\s+{variant_id}:\s")
    starts = [i for i, page in enumerate(pages) if variant_re.search(page)]
    if not starts:
        return None
    start = starts[0]
    next_starts = [i for i, page in enumerate(pages[start + 1 :], start + 1) if re.search(r"Variant\s+\d+:\s", page)]
    # ACR often begins the next variant near the bottom of the same PDF page on
    # which the prior variant's last actions appear, so include that boundary page.
    end = (next_starts[0] + 1) if next_starts else len(pages)
    needle = clean(procedure).lower()
    for index in range(start, end):
        haystack = clean(pages[index]).lower()
        if needle in haystack:
            return index + 1
        # PDF line wrapping can split procedure names; normalized alphanumerics is safer.
        compact_needle = re.sub(r"\W", "", needle)
        compact_haystack = re.sub(r"\W", "", haystack)
        if compact_needle in compact_haystack:
            return index + 1
    return None


def strip_page_artifacts(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        if line.startswith("ACR Appropriateness Criteria®"):
            continue
        if re.match(r"^(Right Lower Quadrant Pain|Right Upper Quadrant Pain|Left Lower Quadrant Pain|Acute Pancreatitis)\s+\d+$", line):
            continue
        kept.append(line)
    return "\n".join(kept)


def narrative_sections(pages: list[str], variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return verbatim-ish PDF text blocks under ACR's procedure-family headings.

    Whitespace caused by PDF line wrapping is normalized, but wording is not paraphrased.
    Page boundaries and the ACR heading are retained as provenance.
    """
    tagged: list[str] = []
    for page_number, page in enumerate(pages, 1):
        tagged.append(f"\n[[PAGE:{page_number}]]\n{strip_page_artifacts(page)}")
    text = "".join(tagged)
    discussion = text.find("Discussion of Procedures by Variant")
    summary = text.find("Summary of Recommendations")
    if discussion < 0 or summary < 0:
        raise ValueError("Could not find narrative discussion/summary boundaries")
    prior_pages = re.findall(r"\[\[PAGE:(\d+)\]\]", text[:discussion])
    page_anchor = f"[[PAGE:{prior_pages[-1]}]]\n" if prior_pages else ""
    body = page_anchor + text[discussion:summary]
    heading_re = re.compile(r"(?m)^([A-H])\. ([^\n]+)$")
    headings = list(heading_re.finditer(body))
    result: list[dict[str, Any]] = []
    current_variant = None
    for index, match in enumerate(headings):
        prefix = body[: match.start()]
        variant_matches = list(re.finditer(r"Variant\s+(\d+):", prefix))
        if variant_matches:
            current_variant = int(variant_matches[-1].group(1))
        if current_variant is None:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        raw = body[match.end() : end]
        # Remove repeated variant heading immediately before the next procedure while keeping prose.
        raw = re.sub(r"\nVariant\s+\d+:.*?(?=\n[A-H]\. |\Z)", "", raw, flags=re.S)
        page_numbers = [int(n) for n in re.findall(r"\[\[PAGE:(\d+)\]\]", raw)]
        before_pages = re.findall(r"\[\[PAGE:(\d+)\]\]", body[: match.start()])
        heading_page = int(before_pages[-1]) if before_pages else 1
        all_pages = sorted(set([heading_page, *page_numbers]))
        content = re.sub(r"\[\[PAGE:\d+\]\]", " ", raw)
        content = clean(content)
        if content:
            result.append(
                {
                    "rationale_id": f"v{current_variant}_{match.group(1).lower()}",
                    "variant_id": current_variant,
                    "acr_section_heading": f"{match.group(1)}. {clean(match.group(2))}",
                    "text": content,
                    "page_start": min(all_pages),
                    "page_end": max(all_pages),
                }
            )
    return result


def summary_recommendations(pages: list[str]) -> dict[int, dict[str, Any]]:
    tagged = "".join(f"\n[[PAGE:{i}]]\n{strip_page_artifacts(page)}" for i, page in enumerate(pages, 1))
    start = tagged.find("Summary of Recommendations")
    end = tagged.find("Supporting Documents", start)
    if start < 0 or end < 0:
        raise ValueError("Could not find summary recommendation boundaries")
    prior_pages = re.findall(r"\[\[PAGE:(\d+)\]\]", tagged[:start])
    anchor = f"[[PAGE:{prior_pages[-1]}]]\n" if prior_pages else ""
    section = anchor + tagged[start:end]
    bullets = list(re.finditer(r"•\s*Variant\s+(\d+):", section))
    result: dict[int, dict[str, Any]] = {}
    for index, match in enumerate(bullets):
        block_end = bullets[index + 1].start() if index + 1 < len(bullets) else len(section)
        block = section[match.start():block_end]
        page_numbers = [int(x) for x in re.findall(r"\[\[PAGE:(\d+)\]\]", section[:block_end])]
        page_start_candidates = [int(x) for x in re.findall(r"\[\[PAGE:(\d+)\]\]", section[:match.start()])]
        page_start = page_start_candidates[-1] if page_start_candidates else page_numbers[0]
        pages_inside = [int(x) for x in re.findall(r"\[\[PAGE:(\d+)\]\]", block)]
        text = clean(re.sub(r"\[\[PAGE:\d+\]\]", " ", block))
        result[int(match.group(1))] = {
            "text": text,
            "page_start": page_start,
            "page_end": max([page_start, *pages_inside]),
        }
    return result


def family(procedure: str) -> str:
    p = procedure.lower()
    if p.startswith("ct pelvis"):
        return "ct_pelvis"
    if p.startswith("ct "):
        return "ct_abdomen_pelvis" if "pelvis" in p else "ct_abdomen"
    if p.startswith("mri "):
        return "mri_abdomen_pelvis" if "pelvis" in p else "mri_abdomen"
    if "duplex doppler" in p:
        return "us_duplex_doppler_abdomen"
    if "with iv contrast" in p and p.startswith("us "):
        return "us_abdomen_with_iv_contrast"
    if p.startswith("us abdomen"):
        return "us_abdomen"
    if p.startswith("us pelvis"):
        return "us_pelvis"
    if "nuclear medicine" in p or "hida" in p:
        return "nuclear_medicine_gallbladder"
    if p.startswith("radiography"):
        return "radiography"
    if p.startswith("fluoroscopy contrast"):
        return "fluoroscopy_contrast_enema"
    if p.startswith("fluoroscopy cystography"):
        return "fluoroscopy_cystography"
    if p.startswith("image-guided cholecystostomy"):
        return "image_guided_cholecystostomy"
    if p.startswith("wbc scan"):
        return "wbc_scan"
    return re.sub(r"[^a-z0-9]+", "_", p).strip("_")


def action_components(procedure: str) -> dict[str, Any]:
    """Decompose only tokens explicitly present in the ACR procedure wording."""
    p = procedure.lower()
    fam = family(procedure)
    if p.startswith("ct "):
        modality = "CT"
    elif p.startswith("mri "):
        modality = "MRI"
    elif p.startswith("us "):
        modality = "US"
    elif p.startswith("radiography"):
        modality = "Radiography"
    elif p.startswith("fluoroscopy"):
        modality = "Fluoroscopy"
    elif p.startswith("wbc scan"):
        modality = "WBC scan"
    elif p.startswith("hida") or p.startswith("nuclear medicine"):
        modality = "Nuclear medicine"
    elif p.startswith("image-guided"):
        modality = "Image-guided procedure"
    else:
        modality = procedure.split()[0]
    regions = [
        label for token, label in (
            ("abdomen", "abdomen"), ("pelvis", "pelvis"),
            ("gallbladder", "gallbladder"), ("bladder", "bladder"),
        ) if token in p
    ]
    protocol: list[str] = []
    contrast_tokens = (
        ["without and with IV contrast"] if "without and with iv contrast" in p
        else ["without IV contrast"] if "without iv contrast" in p
        else ["with IV contrast"] if "with iv contrast" in p
        else []
    )
    for token in (
        *contrast_tokens, "with MRCP", "duplex Doppler", "transabdominal",
        "transvaginal", "bladder contrast", "contrast enema",
    ):
        if token.lower() in p:
            protocol.append(token)
    return {
        "family": fam,
        "modality": modality,
        "body_region_or_target": regions,
        "protocol_terms": protocol,
        "procedure_role": "image-guided intervention" if fam == "image_guided_cholecystostomy" else "diagnostic imaging",
    }


def heading_family(heading: str) -> str:
    h = heading.lower()
    if "ct pelvis with bladder" in h:
        return "ct_pelvis"
    if "ct abdomen" in h:
        return "ct_abdomen_pelvis" if "pelvis" in h else "ct_abdomen"
    if "mri abdomen and pelvis" in h:
        return "mri_abdomen_pelvis"
    if "mri abdomen" in h:
        return "mri_abdomen"
    if "duplex doppler" in h:
        return "us_duplex_doppler_abdomen"
    if "us abdomen with iv contrast" in h:
        return "us_abdomen_with_iv_contrast"
    if "us abdomen" in h:
        return "us_abdomen"
    if "us pelvis" in h:
        return "us_pelvis"
    if "nuclear medicine" in h:
        return "nuclear_medicine_gallbladder"
    if "radiography" in h:
        return "radiography"
    if "contrast-enhanced enema" in h or "contrast enema" in h:
        return "fluoroscopy_contrast_enema"
    if "fluoroscopy cystography" in h:
        return "fluoroscopy_cystography"
    if "image-guided" in h or "biopsy liver" in h:
        return "image_guided_cholecystostomy"
    if "wbc scan" in h:
        return "wbc_scan"
    return "unknown"


def infer_context(variant_text: str) -> dict[str, Any]:
    """Organize literal ACR variant phrases into the reviewed four-part context."""
    low = variant_text.lower()
    terms: dict[str, list[str]] = {}
    phrase_map = {
        "presentation": ["right lower quadrant pain", "right upper quadrant pain", "left lower quadrant pain", "epigastric pain", "atypical signs and symptoms", "fever", "leukocytosis", "elevated WBC count", "no fever", "no high white blood cell (WBC) count", "increased amylase and lipase", "equivocal amylase and lipase values", "continued abdominal pain", "early satiety", "nausea", "vomiting", "signs of infection"],
        "condition": ["suspected appendicitis", "suspected biliary disease", "suspected acalculous cholecystitis", "suspected diverticulitis", "suspected acute pancreatitis", "acute pancreatitis", "known necrotizing pancreatitis", "known pancreatic or peripancreatic fluid collections", "unknown etiology"],
        "population": ["pregnant woman"],
        "timing": ["less than 48 to 72 hours after symptom onset", "greater than 48 to 72 hours after onset of symptoms", "greater than 7 to 21 days after onset of symptoms", "greater than 4 weeks after symptom onset"],
        "severity_or_complication": ["suspected complication(s) of diverticulitis", "critically ill", "systemic inflammatory response syndrome (SIRS)", "severe clinical scores", "continued SIRS", "significant deterioration in clinical status", "abrupt decrease in hemoglobin or hematocrit", "hypotension", "tachycardia", "tachypnea", "abrupt change in fever curve", "increase in white blood cells"],
        "constraints_or_confounders": ["possibly confounded by acute kidney injury or chronic kidney disease", "when diagnoses other than pancreatitis may be possible (bowel perforation, bowel ischemia, etc.)"],
    }
    for category in phrase_map:
        terms[category] = []
    occupied: list[tuple[int, int]] = []
    candidates = sorted(
        ((phrase, category) for category, phrases in phrase_map.items() for phrase in phrases),
        key=lambda item: len(item[0]), reverse=True,
    )
    for phrase, category in candidates:
        for match in re.finditer(re.escape(phrase.lower()), low):
            if not any(match.start() < end and match.end() > start for start, end in occupied):
                terms[category].append(phrase)
                occupied.append(match.span())
                break
    prior_phrase = "negative or equivocal ultrasound"
    if prior_phrase in low:
        imaging_history = {
            "prior_test": ["ultrasound"],
            "prior_result": ["negative or equivocal"],
            "source_phrases": [prior_phrase],
        }
    else:
        imaging_history = {"prior_test": [], "prior_result": [], "source_phrases": []}
    if "next imaging study" in low:
        imaging_stage = "next"
        stage_source_phrase = "next imaging study"
    elif "initial imaging" in low:
        imaging_stage = "initial"
        stage_source_phrase = "initial imaging"
    else:
        imaging_stage = "unspecified"
        stage_source_phrase = None
    encounter_status = ["first time presentation"] if "first time presentation" in low else []
    return {
        "clinical_state": {
            "presentation": terms["presentation"],
            "condition": terms["condition"],
            "severity_or_complication": terms["severity_or_complication"],
        },
        "imaging_history": imaging_history,
        "modifiers": {
            "population": terms["population"],
            "timing": terms["timing"],
            "constraints_or_confounders": terms["constraints_or_confounders"],
        },
        "decision_stage": {
            "imaging_stage": imaging_stage,
            "encounter_status": encounter_status,
            "source_phrase": stage_source_phrase,
        },
    }


def build(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    sources = output / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    acquired = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    old_manifest_path = sources / "manifest.json"
    if old_manifest_path.exists():
        old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        old_acquired = {
            (item["topic_id"], item["kind"], item["file"]): item["acquired_at_utc"]
            for item in old_manifest.get("sources", [])
        }
    else:
        old_acquired = {}
    corpus: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []

    for topic in TOPICS:
        slug, tid = topic["slug"], topic["topic_id"]
        urls = {
            "narrative": f"{BASE}/TopicNarrativePdf?topicId={tid}",
            "appendix": f"{BASE}/GenerateAppendixPDF?PanelName=Gastrointestinal&TopicId={tid}",
            "appendix_html": f"{BASE}/GetAppendix?PanelName=Gastrointestinal&TopicId={tid}",
            "evidence_table": f"{BASE}/DownloadPdf?TopicId={tid}&TopicName={topic['title'].replace(' ', '%20')}",
            "evidence_html": f"{BASE}/GetEvidence?TopicId={tid}&TopicName={topic['title'].replace(' ', '+')}",
        }
        files = {
            "narrative": sources / f"{slug}_narrative.pdf",
            "appendix": sources / f"{slug}_appendix.pdf",
            "appendix_html": sources / f"{slug}_appendix.html",
            "evidence_table": sources / f"{slug}_evidence_table.pdf",
            "evidence_html": sources / f"{slug}_evidence_table.html",
        }
        for kind, path in files.items():
            downloaded = args.refresh or not path.exists()
            if downloaded:
                download(urls[kind], path)
            relative_file = str(path.relative_to(output))
            source_manifest.append(
                {
                    "topic_id": tid, "kind": kind, "url": urls[kind],
                    "file": relative_file, "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "acquired_at_utc": acquired if downloaded else old_acquired.get(
                        (tid, kind, relative_file), acquired
                    ),
                }
            )

        parser = AppendixParser()
        parser.feed(files["appendix_html"].read_text(encoding="utf-8"))
        appendix_pages = pdf_pages(files["appendix"], include_left_column=True)
        narrative_page_text = pdf_pages(files["narrative"])
        rationales = narrative_sections(narrative_page_text, parser.variants)
        summaries = summary_recommendations(narrative_page_text)
        rationale_lookup = {
            (r["variant_id"], heading_family(r["acr_section_heading"])): r["rationale_id"]
            for r in rationales
        }

        for variant in parser.variants:
            variant["context"] = infer_context(variant["variant_text"])
            variant["action_relationships"] = []
            variant["provenance"] = {
                "source_kind": "ACR appendix",
                "source_file": f"sources/{slug}_appendix.pdf",
                "source_url": urls["appendix"],
                "locator": f"Variant {variant['variant_id']}",
            }
            for position, action in enumerate(variant["actions"], 1):
                action["action_id"] = f"acr_{tid}_v{variant['variant_id']}_a{position:02d}"
                action["final_rating"] = action.pop("rating")
                action["action_family"] = family(action["procedure"])
                action["action_components"] = action_components(action["procedure"])
                rationale_id = rationale_lookup.get((variant["variant_id"], action["action_family"]))
                action["rationale_ids"] = [rationale_id] if rationale_id else []
                page = APPENDIX_PAGE_OVERRIDES.get(
                    (tid, variant["variant_id"], action["procedure"]),
                    find_appendix_page(appendix_pages, variant["variant_id"], action["procedure"]),
                )
                action["provenance"] = {
                    "source_kind": "ACR appendix",
                    "source_file": f"sources/{slug}_appendix.pdf",
                    "source_url": urls["appendix"],
                    "page": page,
                    "locator": f"Variant {variant['variant_id']} / {action['procedure']}",
                    "html_source_file": f"sources/{slug}_appendix.html",
                    "html_locator": f"h5 Variant {variant['variant_id']} / tr.procedurestr / td.Procedure exact text: {action['procedure']}",
                }
                action_rows.append(
                    {
                        "topic_id": tid, "topic_slug": slug, "topic_title": topic["title"],
                        "topic_version": topic["topic_version"],
                        "variant_id": variant["variant_id"], "variant_text": variant["variant_text"],
                        "context": variant["context"], **action,
                    }
                )

            relation_spec = EXPLICIT_ACTION_RELATIONSHIPS.get((tid, variant["variant_id"]))
            if relation_spec:
                relationship, procedure_names = relation_spec
                action_by_procedure = {a["procedure"]: a["action_id"] for a in variant["actions"]}
                summary = summaries[variant["variant_id"]]
                variant["action_relationships"].append({
                    "relationship": relationship,
                    "procedure_names": list(procedure_names),
                    "action_ids": [action_by_procedure[name] for name in procedure_names],
                    "source_text": summary["text"],
                    "provenance": {
                        "source_kind": "ACR narrative summary",
                        "source_file": f"sources/{slug}_narrative.pdf",
                        "source_url": urls["narrative"],
                        "page_start": summary["page_start"],
                        "page_end": summary["page_end"],
                        "locator": f"Summary of Recommendations / Variant {variant['variant_id']}",
                    },
                })

        for rationale in rationales:
            rationale["provenance"] = {
                "source_kind": "ACR narrative",
                "source_file": f"sources/{slug}_narrative.pdf",
                "source_url": urls["narrative"],
                "page_start": rationale.pop("page_start"),
                "page_end": rationale.pop("page_end"),
                "locator": f"Variant {rationale['variant_id']} / {rationale['acr_section_heading']}",
            }

        corpus.append(
            {
                **topic,
                "official_urls": urls,
                "source_files": {key: f"sources/{path.name}" for key, path in files.items()},
                "variants": parser.variants,
                "rationales": rationales,
            }
        )

    (output / "acr_topics.json").write_text(
        json.dumps({
            "schema_version": "1.1.0",
            "ranking_policy": {
                "primary_metric": "final_rating",
                "direction": "higher_is_more_appropriate",
                "range": [1, 9],
                "tie_policy": "Preserve ties; do not infer a unique path from equal ratings.",
                "non_ranking_fields": ["appropriateness_category", "strength_of_evidence", "median_rating", "final_tabulations"],
            },
            "topics": corpus,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "acr_actions.jsonl").open("w", encoding="utf-8") as stream:
        for row in action_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (sources / "manifest.json").write_text(
        json.dumps({"generated_at_utc": acquired, "sources": source_manifest}, indent=2) + "\n",
        encoding="utf-8",
    )

    contexts: dict[str, set[str]] = {}
    for topic in corpus:
        for variant in topic["variants"]:
            context = variant["context"]
            for parent in ("clinical_state", "imaging_history", "modifiers", "decision_stage"):
                for category, value in context[parent].items():
                    if isinstance(value, list):
                        contexts.setdefault(f"{parent}.{category}", set()).update(value)
                    elif value is not None:
                        contexts.setdefault(f"{parent}.{category}", set()).add(str(value))
    vocab = {
        "method": "Reviewed four-part context induced from exact phrases in the 17 ACR variants; not A/Q/C.",
        "context_structure": {
            "clinical_state": ["presentation", "condition", "severity_or_complication"],
            "imaging_history": ["prior_test", "prior_result", "source_phrases"],
            "modifiers": ["population", "timing", "constraints_or_confounders"],
            "decision_stage": ["imaging_stage", "encounter_status", "source_phrase"],
        },
        "context_vocabulary": {key: sorted(values, key=str.lower) for key, values in sorted(contexts.items())},
        "action_families": dict(sorted(Counter(row["action_family"] for row in action_rows).items())),
        "procedure_wording": sorted({row["procedure"] for row in action_rows}, key=str.lower),
        "action_ranking": {
            "primary_metric": "final_rating",
            "direction": "higher_is_more_appropriate",
            "range": [1, 9],
            "tie_policy": "Preserve ties; do not infer a unique path from equal ratings.",
        },
        "appropriateness_categories": sorted({row["appropriateness_category"] for row in action_rows}),
        "strength_of_evidence_values": sorted({row["strength_of_evidence"] for row in action_rows}),
    }
    (output / "native_vocabulary.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "topics": len(corpus),
        "variants": sum(len(t["variants"]) for t in corpus),
        "actions": len(action_rows),
        "rationales": sum(len(t["rationales"]) for t in corpus),
        "unmapped_rationale_actions": sum(not row["rationale_ids"] for row in action_rows),
        "missing_appendix_pages": sum(row["provenance"]["page"] is None for row in action_rows),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/acr_normative"))
    parser.add_argument("--refresh", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
