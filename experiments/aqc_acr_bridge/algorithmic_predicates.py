"""Deterministic patient-predicate extraction for the AQC--ACR bridge.

This module handles only evidence that is already structured, or can be parsed
with a high-confidence labelled pattern:

- laboratory values and reference ranges;
- labelled vital signs in the physical-examination text;
- resulted prior-imaging metadata, including non-decision context studies;

It does not call an LLM, infer longitudinal change, or expose the current
imaging order/result. Narrative HPI, examination findings, and imaging-report
findings are intentionally left to the later text-extraction step.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0.0-algorithmic-patient-predicates"
DEFAULT_LAB_MAP = (
    Path(__file__).resolve().parents[2] / "data" / "raw_data" / "lab_test_mapping.csv"
)

_LEADING_NUMBER = re.compile(r"^\s*(?:[<>]=?\s*)?([-+]?\d+(?:\.\d+)?)")
_NON_RESULT = re.compile(
    r"^(?:hold\b|discard\b|not\s*done\b|not\s*performed\b|cancel(?:led|ed)?\b)",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[a-z0-9]+")

_ANALYTE_ALIASES = {
    "white blood cells": "white_blood_cell_count",
    "white blood cell count": "white_blood_cell_count",
    "wbc": "white_blood_cell_count",
    "platelet count": "platelet_count",
    "red blood cells": "red_blood_cell_count",
}

_VITAL_PATTERNS = {
    "temperature": re.compile(
        r"(?<![A-Za-z])(?:Temp(?:erature)?|Tc|T)\s*:?[ ]*"
        r"(?P<value>\d{2,3}(?:\.\d+)?)\s*(?P<unit>°?\s*[FC])?\b",
        re.IGNORECASE,
    ),
    "heart_rate": re.compile(
        r"\b(?:HR|heart\s*rate|pulse)\s*:?[ ]*(?P<value>\d{2,3})\b",
        re.IGNORECASE,
    ),
    "blood_pressure": re.compile(
        r"\b(?:BP|blood\s*pressure)\s*:?[ ]*"
        r"(?P<value>\d{2,3}\s*/\s*\d{2,3})\b",
        re.IGNORECASE,
    ),
    "respiratory_rate": re.compile(
        r"\b(?:RR|resp(?:iratory)?(?:\s*rate)?)\s*:?[ ]*(?P<value>\d{1,2})\b",
        re.IGNORECASE,
    ),
    "oxygen_saturation": re.compile(
        r"\b(?:SpO2|SaO2|POx|O2\s*sat(?:uration)?)\s*:?[ ]*"
        r"(?P<value>\d{2,3})\s*%?",
        re.IGNORECASE,
    ),
}


def load_lab_metadata(path: Path = DEFAULT_LAB_MAP) -> dict[str, dict[str, str]]:
    """Return itemid-indexed lab label, fluid, and category metadata."""
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            str(row["itemid"]).removesuffix(".0"): {
                "label": str(row.get("label") or "").strip(),
                "fluid": str(row.get("fluid") or "").strip(),
                "category": str(row.get("category") or "").strip(),
            }
            for row in csv.DictReader(handle)
            if str(row.get("itemid") or "").strip()
        }


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if not isinstance(value, str) or value.strip().lower() in {"", "nan", "none"}:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): item for key, item in parsed.items()}


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    match = _LEADING_NUMBER.search(str(value))
    return float(match.group(1)) if match else None


def _reference(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and math.isfinite(number) else None


def _lab_state(raw_value: Any, lower: Any, upper: Any) -> str:
    text = str(raw_value).strip()
    number = _number(raw_value)
    low = _reference(lower)
    high = _reference(upper)
    if number is not None and (low is not None or high is not None):
        if low is not None and number < low:
            return "low"
        if high is not None and number > high:
            return "high"
        return "normal"

    folded = text.casefold()
    if re.match(r"^(?:neg(?:ative)?|not detected|no growth)\b", folded):
        return "negative"
    if re.match(r"^(?:none|absent)\b", folded):
        return "absent"
    if re.match(r"^(?:pos(?:itive)?|present)\b", folded) or "growth" in folded:
        return "positive"
    if re.match(r"^(?:normal|within normal)\b", folded):
        return "normal"
    return "reported"


def _identifier(text: str) -> str:
    return "_".join(_TOKEN.findall(text.casefold())) or "unknown"


def _analyte(label: str, itemid: str) -> str:
    folded = re.sub(r"\s+", " ", label.strip().casefold())
    if folded in _ANALYTE_ALIASES:
        return _ANALYTE_ALIASES[folded]
    return _identifier(label) if label else f"lab_{itemid}"


def extract_lab_predicates(
    laboratory_tests: Any,
    reference_range_lower: Any,
    reference_range_upper: Any,
    lab_metadata: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Convert result-bearing lab JSON entries to ``lab_finding_state`` items."""
    values = _json_object(laboratory_tests)
    lower = _json_object(reference_range_lower)
    upper = _json_object(reference_range_upper)
    predicates = []
    for itemid, raw_value in values.items():
        raw_text = str(raw_value).strip()
        if not raw_text or _NON_RESULT.match(raw_text):
            continue
        metadata = dict(lab_metadata.get(str(itemid), {}))
        label = str(metadata.get("label") or "").strip()
        low = _reference(lower.get(itemid))
        high = _reference(upper.get(itemid))
        predicates.append({
            "id": f"alg_lab_{_identifier(str(itemid))}",
            "analyte": _analyte(label, str(itemid)),
            "state": _lab_state(raw_value, low, high),
            "evidence": [{
                "source": "laboratory_tests",
                "itemid": str(itemid),
                "label": label or None,
                "fluid": metadata.get("fluid") or None,
                "category": metadata.get("category") or None,
                "raw_value": raw_text,
                "reference_low": low,
                "reference_high": high,
            }],
        })
    return predicates


def _vital_state(sign: str, match: re.Match[str]) -> str:
    raw_value = re.sub(r"\s+", "", match.group("value"))
    if sign == "temperature":
        unit_group = match.groupdict().get("unit") or ""
        unit = re.sub(r"[^FC]", "", unit_group.upper())
        if not unit:
            unit = "F" if float(raw_value) > 50 else "C"
        return f"{raw_value}_{unit}"
    suffix = {
        "heart_rate": "bpm",
        "blood_pressure": "mmHg",
        "respiratory_rate": "per_min",
        "oxygen_saturation": "percent",
    }[sign]
    return f"{raw_value}_{suffix}"


def extract_labelled_vital_predicates(physical_examination: Any) -> list[dict[str, Any]]:
    """Extract only explicitly labelled vitals, taking the last match per sign."""
    text = str(physical_examination or "")
    predicates = []
    for sign, pattern in _VITAL_PATTERNS.items():
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        match = matches[-1]
        predicates.append({
            "id": f"alg_sign_{sign}",
            "sign": sign,
            "state": _vital_state(sign, match),
            "evidence": [{
                "source": "physical_examination",
                "quote": match.group(0).strip(),
            }],
        })
    return predicates


def mask_labelled_vitals(physical_examination: Any) -> str:
    """Hide vital spans already handled by the deterministic extractor."""
    text = str(physical_examination or "")
    for pattern in _VITAL_PATTERNS.values():
        text = pattern.sub("[captured_vital]", text)
    return text


def _test_name(item: Mapping[str, Any]) -> str:
    modality = _identifier(str(item.get("modality") or "test"))
    aliases = {
        "computed_tomography": "ct",
        "ultrasound": "ultrasound",
        "magnetic_resonance_imaging": "mri",
    }
    return aliases.get(modality, modality)


def extract_test_history(
    visible_prior_imaging: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Retain the latest visible prior study for each normalized test identity."""
    latest: dict[str, tuple[int, Mapping[str, Any]]] = {}
    priors = list(visible_prior_imaging or [])
    for index, item in enumerate(priors, start=1):
        latest[_test_name(item)] = (index, item)

    predicates = []
    for test, (index, item) in sorted(latest.items(), key=lambda pair: pair[1][0]):
        predicates.append({
            "id": f"alg_test_{test}",
            "test": test,
            "status": "completed_before_current_decision",
            "evidence": [{
                "source": "prior_imaging_metadata",
                "prior_imaging_index": index,
                "modality": item.get("modality"),
                "region": item.get("region"),
                "exam": item.get("exam"),
                "role": item.get("role"),
            }],
        })
    return predicates


def extract_algorithmic_predicates(
    raw_row: Mapping[str, Any],
    decision_point: Mapping[str, Any],
    lab_metadata: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Extract the three predicate types covered by the deterministic path."""
    visible_prior = decision_point.get("visible_prior_imaging") or []
    return {
        "schema_version": SCHEMA_VERSION,
        "lab_finding_state": extract_lab_predicates(
            raw_row.get("Laboratory Tests"),
            raw_row.get("Reference Range Lower"),
            raw_row.get("Reference Range Upper"),
            lab_metadata,
        ),
        "sign_state": extract_labelled_vital_predicates(
            raw_row.get("Physical Examination")
        ),
        "test_history": extract_test_history(visible_prior),
    }
