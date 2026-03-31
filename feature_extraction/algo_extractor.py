"""algo_extractor.py — Algorithmic feature extraction from structured CSV data.

Extracts all features that can be derived without LLM:
  - Lab thresholds  (from Laboratory Tests JSON + Reference Range JSONs)
  - Temperature / HR / RR  (regex on Physical Examination free text)
  - SIRS criteria  (derived from labs + vitals)
  - tests_done     (Lab_Panel always present; Radiology modalities parsed separately)

The Radiology JSON list is also parsed here to produce an ordered list of
imaging entries for the pipeline to iterate over.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Lab item ID sets  (MIMIC-IV itemids)
# ---------------------------------------------------------------------------

_WBC_IDS    = {"51301", "51300"}          # White Blood Cell count  (K/uL)
_BAND_IDS   = {"51144", "51146"}          # Bands %
_LIPASE_IDS = {"50956"}                   # Lipase  (IU/L)
_BUN_IDS    = {"51006", "52647"}          # Blood Urea Nitrogen  (mg/dL)
_BILI_IDS   = {"50885"}                   # Bilirubin, Total  (mg/dL)
_ALT_IDS    = {"50861"}                   # Alanine Aminotransferase
_AST_IDS    = {"50878"}                   # Aspartate Aminotransferase
_ALP_IDS    = {"50863"}                   # Alkaline Phosphatase
_GGT_IDS    = {"50927"}                   # Gamma-Glutamyltransferase
_CREAT_IDS  = {"50912", "52024", "52546"} # Creatinine  (mg/dL)
_CRP_IDS    = {"50889"}                   # C-Reactive Protein  (mg/L)
_HCG_IDS    = {"51085", "52720"}          # beta-hCG  (qualitative text)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _parse_lab_json(raw: str) -> dict[str, str]:
    """Parse the Laboratory Tests column (JSON string) → {itemid: value_str}."""
    if not raw or str(raw).strip() in ("", "nan"):
        return {}
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return {}


def _parse_ref_json(raw: str) -> dict[str, Optional[float]]:
    """Parse Reference Range Lower/Upper column → {itemid: float | None}."""
    if not raw or str(raw).strip() in ("", "nan"):
        return {}
    try:
        raw_dict = json.loads(raw)
    except Exception:
        try:
            raw_dict = ast.literal_eval(raw)
        except Exception:
            return {}
    result: dict[str, Optional[float]] = {}
    for k, v in raw_dict.items():
        if v is None or (isinstance(v, float) and v != v):  # NaN
            result[k] = None
        else:
            try:
                result[k] = float(v)
            except Exception:
                result[k] = None
    return result


# ---------------------------------------------------------------------------
# Lab value lookup helpers
# ---------------------------------------------------------------------------

def _get_numeric(labs: dict[str, str], item_ids: set[str]) -> Optional[float]:
    """Return the first numeric value found for any of the given item IDs."""
    for iid in item_ids:
        raw = labs.get(iid)
        if raw is None:
            continue
        m = re.search(r"[-+]?\d+\.?\d*", str(raw))
        if m:
            return float(m.group())
    return None


def _get_ref_upper(
    refs_upper: dict[str, Optional[float]], item_ids: set[str]
) -> Optional[float]:
    """Return the reference range upper limit for the first matching item ID."""
    for iid in item_ids:
        v = refs_upper.get(iid)
        if v is not None:
            return v
    return None


def _get_text(labs: dict[str, str], item_ids: set[str]) -> Optional[str]:
    """Return the raw text value for the first matching item ID."""
    for iid in item_ids:
        v = labs.get(iid)
        if v is not None:
            return str(v).strip()
    return None


# ---------------------------------------------------------------------------
# Vitals extraction via regex (Physical Examination free text)
# ---------------------------------------------------------------------------

def _extract_temp_f(pe_text: str) -> Optional[float]:
    """
    Extract body temperature from PE text and normalise to Fahrenheit.

    Handles labelled patterns ("Temp: 98.6", "T: 99.1") and unlabelled
    vital-sign sequences ("98.1 95 124/61 22 96%").
    Distinguishes °F from °C by magnitude (>50 → Fahrenheit).
    """
    # Labelled: Temp / Temperature / T followed by number
    m = re.search(
        r"\b(?:Temp(?:erature)?|T)[:\s]+(\d{2,3}(?:\.\d{1,2})?)",
        pe_text,
        re.IGNORECASE,
    )
    if m:
        t = float(m.group(1))
        return t if t > 50 else t * 9 / 5 + 32   # convert °C → °F if needed

    # Unlabelled vitals sequence: e.g. "98.1 95 124/61 22 96%"
    m2 = re.search(
        r"\b(9\d\.\d)\s+\d{2,3}\s+\d{2,3}/\d{2,3}\s+\d{1,2}",
        pe_text,
    )
    if m2:
        return float(m2.group(1))

    return None


def _extract_hr(pe_text: str) -> Optional[float]:
    """Extract heart rate (bpm) from PE text."""
    m = re.search(r"\b(?:HR|Heart\s*Rate|Pulse)[:\s]+(\d{2,3})", pe_text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _extract_rr(pe_text: str) -> Optional[float]:
    """Extract respiratory rate (breaths/min) from PE text."""
    m = re.search(
        r"\b(?:RR|Resp(?:iratory)?(?:\s*Rate)?)[:\s]+(\d{1,2})",
        pe_text,
        re.IGNORECASE,
    )
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# SIRS computation
# ---------------------------------------------------------------------------

def _compute_sirs(
    temp_f: Optional[float],
    hr: Optional[float],
    rr: Optional[float],
    wbc: Optional[float],
    bands: Optional[float],
) -> bool:
    """
    Return True if ≥ 2 SIRS criteria are met.

    Criteria:
      1. Temp < 36°C or > 38°C
      2. HR > 90
      3. RR > 20
      4. WBC < 4 K/uL or > 12 K/uL  OR  Bands > 10 %
    """
    count = 0
    if temp_f is not None:
        temp_c = (temp_f - 32) * 5 / 9
        if temp_c < 36.0 or temp_c > 38.0:
            count += 1
    if hr is not None and hr > 90:
        count += 1
    if rr is not None and rr > 20:
        count += 1
    if wbc is not None and (wbc < 4.0 or wbc > 12.0):
        count += 1
    elif bands is not None and bands > 10.0:
        count += 1
    return count >= 2


# ---------------------------------------------------------------------------
# Radiology modality → VALID_TESTS key
# ---------------------------------------------------------------------------

def _modality_to_test(modality: str, exam_name: str, region: str) -> Optional[str]:
    """Map a radiology note's Modality/Exam to a VALID_TESTS string."""
    mod  = modality.lower().strip()
    name = exam_name.lower()
    reg  = region.lower()

    if mod == "ultrasound":
        return "Ultrasound_Abdomen"
    if mod == "ct":
        return "CT_Abdomen"
    if mod == "mri":
        return "MRCP_Abdomen" if "mrcp" in name else "MRI_Abdomen"
    if mod in ("radiograph", "x-ray", "xray"):
        return "Radiograph_Chest" if ("chest" in name or "chest" in reg) else None
    if "hida" in mod or "hida" in name:
        return "HIDA_Scan"
    if "mrcp" in mod or "mrcp" in name:
        return "MRCP_Abdomen"
    return None


# ---------------------------------------------------------------------------
# Main extraction functions
# ---------------------------------------------------------------------------

def extract_algo_features(row: dict) -> dict:
    """
    Extract all algorithmically-derivable features from one CSV row.

    Args:
        row: dict matching the raw CSV column names.

    Returns:
        Partial feature dict populated only with algorithmically derived keys.
        Keys absent here will keep their default_features() values in the pipeline.
    """
    labs      = _parse_lab_json(row.get("Laboratory Tests", ""))
    ref_upper = _parse_ref_json(row.get("Reference Range Upper", ""))
    pe_text   = str(row.get("Physical Examination", "") or "")

    features: dict = {}

    # ── WBC ──────────────────────────────────────────────────────────────────
    wbc = _get_numeric(labs, _WBC_IDS)
    features["WBC_gt_10k"] = wbc is not None and wbc > 10.0
    features["WBC_gt_18k"] = wbc is not None and wbc > 18.0

    # ── Bands / Left Shift ───────────────────────────────────────────────────
    bands       = _get_numeric(labs, _BAND_IDS)
    bands_upper = _get_ref_upper(ref_upper, _BAND_IDS) or 2.0
    features["left_shift"] = bands is not None and bands > bands_upper

    # ── Lipase ───────────────────────────────────────────────────────────────
    lipase       = _get_numeric(labs, _LIPASE_IDS)
    lipase_upper = _get_ref_upper(ref_upper, _LIPASE_IDS)
    if lipase is not None and lipase_upper and lipase_upper > 0:
        features["lipase_ge_3xULN"] = lipase >= 3.0 * lipase_upper
    else:
        features["lipase_ge_3xULN"] = False

    # ── BUN ──────────────────────────────────────────────────────────────────
    bun = _get_numeric(labs, _BUN_IDS)
    features["BUN_gt_25"] = bun is not None and bun > 25.0

    # ── Bilirubin ────────────────────────────────────────────────────────────
    bili       = _get_numeric(labs, _BILI_IDS)
    bili_upper = _get_ref_upper(ref_upper, _BILI_IDS)
    features["bilirubin_elevated"] = (
        bili is not None and bili_upper is not None and bili > bili_upper
    )

    # ── LFTs (any one elevated counts) ───────────────────────────────────────
    lft_pairs = [
        (_get_numeric(labs, _ALT_IDS), _get_ref_upper(ref_upper, _ALT_IDS)),
        (_get_numeric(labs, _AST_IDS), _get_ref_upper(ref_upper, _AST_IDS)),
        (_get_numeric(labs, _ALP_IDS), _get_ref_upper(ref_upper, _ALP_IDS)),
        (_get_numeric(labs, _GGT_IDS), _get_ref_upper(ref_upper, _GGT_IDS)),
    ]
    features["LFTs_elevated"] = any(
        val is not None and upper is not None and val > upper
        for val, upper in lft_pairs
    )

    # ── Creatinine ───────────────────────────────────────────────────────────
    creat       = _get_numeric(labs, _CREAT_IDS)
    creat_upper = _get_ref_upper(ref_upper, _CREAT_IDS)
    features["creatinine_elevated"] = (
        creat is not None and creat_upper is not None and creat > creat_upper
    )

    # ── Organ dysfunction (lab-derived) ──────────────────────────────────────
    # Creatinine ≥ 2.0 mg/dL  → renal organ failure  (TG18 Grade III / Modified Marshall)
    # Bilirubin  ≥ 2.0 mg/dL  → hepatic organ failure (TG18 Grade III)
    # This is a conservative floor: CT/LLM extraction may upgrade to True later;
    # once set True here it must not be reset False by a subsequent update
    # (see ClinicalSession.add_test sticky-OR logic).
    features["has_organ_dysfunction"] = (
        (creat is not None and creat >= 2.0)
        or (bili  is not None and bili  >= 2.0)
    )

    # ── CRP ──────────────────────────────────────────────────────────────────
    crp       = _get_numeric(labs, _CRP_IDS)
    crp_upper = _get_ref_upper(ref_upper, _CRP_IDS)
    if crp is not None:
        # Fall back to 10 mg/L as conventional upper-normal if ref missing
        features["CRP_elevated"] = crp > (crp_upper if crp_upper is not None else 10.0)
        features["CRP_gt_200"]   = crp > 200.0
    else:
        features["CRP_elevated"] = False
        features["CRP_gt_200"]   = False

    # ── beta-hCG (qualitative) ────────────────────────────────────────────────
    hcg_text = _get_text(labs, _HCG_IDS)
    if hcg_text is not None:
        low = hcg_text.lower()
        # Match "NEGATIVE" as a standalone result word; anything else (including
        # explicit "POSITIVE" or a numeric titre) is treated as positive.
        # We use word-boundary matching to avoid false hits like "POSITIVES".
        is_negative = bool(re.search(r"\bneg(?:ative)?\b", low))
        is_positive = bool(re.search(r"\bpos(?:itive)?\b", low))
        features["beta_hCG_positive"] = is_positive or (
            not is_negative and bool(re.search(r"\d+\.?\d*\s*(?:miu|mIU|IU)?", hcg_text))
        )
    else:
        features["beta_hCG_positive"] = False

    # ── Temperature / Fever ──────────────────────────────────────────────────
    temp_f = _extract_temp_f(pe_text)
    if temp_f is not None:
        temp_c = (temp_f - 32) * 5 / 9
        features["fever_temp_ge_37_3"] = temp_c >= 37.3
        features["fever_temp_ge_38"]   = temp_c >= 38.0
    else:
        features["fever_temp_ge_37_3"] = False
        features["fever_temp_ge_38"]   = False

    # ── SIRS ─────────────────────────────────────────────────────────────────
    hr = _extract_hr(pe_text)
    rr = _extract_rr(pe_text)
    features["SIRS_criteria_ge_2"] = _compute_sirs(temp_f, hr, rr, wbc, bands)

    # ── tests_done ────────────────────────────────────────────────────────────
    features["tests_done"] = ["Lab_Panel"] if labs else []

    return features


def extract_radiology_tests(row: dict) -> list[dict]:
    """
    Parse the Radiology column and return an ordered list of imaging entries.

    Each entry dict contains:
        note_id   (str)           — MIMIC note identifier
        modality  (str)           — raw Modality string
        region    (str)           — raw Region string
        exam_name (str)           — Exam Name
        report    (str)           — full report text (for LLM extraction)
        test_key  (str | None)    — mapped VALID_TESTS key; None if unmappable

    The list preserves the original ordering from MIMIC (chronological).
    """
    raw = str(row.get("Radiology", "") or "")
    if raw.strip() in ("", "[]", "nan"):
        return []

    try:
        reports = json.loads(raw)
    except Exception:
        try:
            reports = ast.literal_eval(raw)
        except Exception:
            return []

    result = []
    for r in reports:
        if not isinstance(r, dict):
            continue
        modality  = r.get("Modality", "")
        exam_name = r.get("Exam Name", "")
        region    = r.get("Region", "")
        result.append({
            "note_id":  r.get("Note ID", ""),
            "modality": modality,
            "region":   region,
            "exam_name": exam_name,
            "report":   r.get("Report", ""),
            "test_key": _modality_to_test(modality, exam_name, region),
        })

    return result
