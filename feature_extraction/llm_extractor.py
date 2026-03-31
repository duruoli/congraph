"""llm_extractor.py — LLM-based feature extraction using OpenAI GPT-4o.

Two public functions:
  extract_step0_llm(client, patient_history, physical_exam)
      → dict of HPI + PE features

  extract_imaging_llm(client, report, modality, exam_name)
      → dict of imaging features for one radiology report
"""

from __future__ import annotations

import json
import time
from typing import Optional

from openai import OpenAI

from feature_extraction.prompts import (
    STEP0_SYSTEM,
    STEP0_USER_TEMPLATE,
    IMAGING_SYSTEM,
    US_USER_TEMPLATE,
    CT_USER_TEMPLATE,
    HIDA_MRCP_MRI_USER_TEMPLATE,
    CHEST_USER_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MODEL       = "gpt-4o"
_TEMPERATURE = 0.0
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds; doubles on each retry

_VALID_PAIN_LOCATIONS = frozenset({
    "RLQ", "RUQ", "LLQ", "LUQ",
    "Epigastric", "Periumbilical", "Pelvic",
    "General_Abdomen", "Other",
})


# ---------------------------------------------------------------------------
# Internal chat helper
# ---------------------------------------------------------------------------

def _chat(client: OpenAI, system: str, user: str) -> dict:
    """
    Send a chat request in JSON mode and return the parsed response dict.
    Retries on transient errors with exponential back-off.
    """
    delay = _RETRY_DELAY
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=_TEMPERATURE,
            )
            raw = response.choices[0].message.content
            return json.loads(raw)

        except json.JSONDecodeError as exc:
            if attempt == _MAX_RETRIES - 1:
                raise RuntimeError(
                    f"GPT-4o returned non-JSON output: {exc}"
                ) from exc

        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            # Retry on rate-limit or transient network errors
            time.sleep(delay)
            delay *= 2

    return {}


# ---------------------------------------------------------------------------
# Type-coercion helpers
# ---------------------------------------------------------------------------

def _safe_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val)
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return False


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Step 0: HPI + Physical Examination
# ---------------------------------------------------------------------------

_STEP0_BOOL_KEYS: tuple[str, ...] = (
    "pain_migration_to_RLQ",
    "epigastric_radiating_to_back",
    "bowel_habit_change",
    "symptom_duration_over_72h",
    "anorexia",
    "nausea_vomiting",
    "alcohol_history",
    "gallstone_history",
    "prior_diverticular_disease",
    "murphys_sign",
    "RLQ_tenderness",
    "rebound_tenderness",
    "RUQ_mass",
    "peritoneal_signs",
    "impaired_mental_status",
    "age_gt_60",
    "is_female_reproductive_age",
)


def extract_step0_llm(
    client: OpenAI,
    patient_history: str,
    physical_exam: str,
) -> dict:
    """
    Extract HPI + Physical Exam features via GPT-4o.

    Args:
        client:          Initialised OpenAI client.
        patient_history: Raw Patient History / HPI text from the CSV.
        physical_exam:   Raw Physical Examination text from the CSV.

    Returns:
        Partial feature dict with keys:
          pain_location, all _STEP0_BOOL_KEYS
    """
    user = STEP0_USER_TEMPLATE.format(
        patient_history=patient_history or "(not available)",
        physical_exam=physical_exam   or "(not available)",
    )
    raw = _chat(client, STEP0_SYSTEM, user)

    features: dict = {}

    # pain_location — categorical, validated against allowed values
    loc = raw.get("pain_location", "Other")
    features["pain_location"] = loc if loc in _VALID_PAIN_LOCATIONS else "Other"

    # All remaining boolean features
    for key in _STEP0_BOOL_KEYS:
        features[key] = _safe_bool(raw.get(key, False))

    return features


# ---------------------------------------------------------------------------
# Imaging: per-radiology-report
# ---------------------------------------------------------------------------

_US_BOOL_KEYS: tuple[str, ...] = (
    "US_appendix_inflamed",
    "US_gallstones",
    "US_GB_wall_thickening",
    "US_pericholecystic_fluid",
    "US_sonographic_murphys",
    "pleural_effusion_on_imaging",
)

_CT_BOOL_KEYS: tuple[str, ...] = (
    "CT_appendicitis_positive",
    "CT_perforation_abscess",
    "CT_cholecystitis_positive",
    "CT_GB_severe_findings",
    "CT_diverticulitis_confirmed",
    "CT_diverticulitis_complicated",
    "CT_phlegmon",
    "CT_abscess_lt_3cm",
    "CT_abscess_ge_3cm",
    "CT_purulent_peritonitis",
    "CT_fecal_peritonitis",
    "CT_pancreatitis_positive",
    "pleural_effusion_on_imaging",
    "cholecystitis_additional_imaging_positive",
    "has_organ_dysfunction",
    "local_complications_pancreatitis",
    "organ_failure_transient",
    "organ_failure_persistent",
)

_OTHER_IMG_BOOL_KEYS: tuple[str, ...] = (
    "cholecystitis_additional_imaging_positive",
    "pleural_effusion_on_imaging",
)

_CHEST_BOOL_KEYS: tuple[str, ...] = ("pleural_effusion_on_imaging",)


def extract_imaging_llm(
    client: OpenAI,
    report: str,
    modality: str,
    exam_name: str = "",
) -> dict:
    """
    Extract imaging features from one radiology report via GPT-4o.

    Selects the appropriate prompt template based on modality.

    Args:
        client:    Initialised OpenAI client.
        report:    Full text of the radiology report.
        modality:  Modality string from the Radiology JSON
                   (e.g. "Ultrasound", "CT", "Radiograph").
        exam_name: Exam Name string (used to identify MRCP within MRI).

    Returns:
        Partial feature dict with only the keys relevant to the modality.
    """
    mod = modality.lower().strip()
    features: dict = {}

    if mod == "ultrasound":
        user = US_USER_TEMPLATE.format(report=report)
        raw  = _chat(client, IMAGING_SYSTEM, user)
        for key in _US_BOOL_KEYS:
            features[key] = _safe_bool(raw.get(key, False))

    elif mod == "ct":
        user = CT_USER_TEMPLATE.format(report=report)
        raw  = _chat(client, IMAGING_SYSTEM, user)
        for key in _CT_BOOL_KEYS:
            features[key] = _safe_bool(raw.get(key, False))
        features["CTSI_score"] = _safe_float(raw.get("CTSI_score", 0.0))

    elif mod in ("radiograph", "x-ray", "xray"):
        user = CHEST_USER_TEMPLATE.format(report=report)
        raw  = _chat(client, IMAGING_SYSTEM, user)
        for key in _CHEST_BOOL_KEYS:
            features[key] = _safe_bool(raw.get(key, False))

    else:
        # HIDA, MRCP, MRI
        user = HIDA_MRCP_MRI_USER_TEMPLATE.format(
            modality=modality, report=report
        )
        raw = _chat(client, IMAGING_SYSTEM, user)
        for key in _OTHER_IMG_BOOL_KEYS:
            features[key] = _safe_bool(raw.get(key, False))

    return features
