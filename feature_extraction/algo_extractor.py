"""algo_extractor.py — Algorithmic feature extraction from structured CSV data.

Extracts all features that can be derived without LLM:
  - HPI features   (pain location, symptom flags, duration — regex + keyword)
  - PE signs       (Murphy's, RLQ tenderness, rebound, peritoneal, mental status)
  - Demographics   (age, sex — from CSV columns, text parsing, or GFR age-group proxy)
  - Lab thresholds (from Laboratory Tests JSON + Reference Range JSONs)
  - Temperature / HR / RR  (regex on Physical Examination free text)
  - SIRS criteria  (derived from labs + vitals)
  - tests_done     (Lab_Panel when Laboratory Tests JSON non-empty; Radiology modalities parsed separately)
  - lab_itemids    (MIMIC itemid keys from Laboratory Tests JSON, for lab cost attribution)

The Radiology JSON list is also parsed here to produce an ordered list of
imaging entries for the pipeline to iterate over.

Design principles
-----------------
* Pain location uses an **ordered priority list** (specific quadrant → diffuse)
  so a note mentioning both "RLQ" and "diffuse" returns "RLQ".
* All sign extraction uses **bi-directional negation** — checks for denial words
  both BEFORE ("no Murphy's sign") and AFTER ("Murphy's sign: negative") the
  matched span to avoid false positives.
* Pain migration is only flagged when explicit **migration language** (migrated,
  moved, shifted, started X now Y) accompanies RLQ as the destination.
* Symptom duration performs **numeric + verbal arithmetic** so "4 days", "four
  days", "one week", "> 72 hours", etc. all correctly evaluate to True.
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
# Negation-aware sign matching  (HPI + PE free text)
# ---------------------------------------------------------------------------

# Words that negate a finding when they appear BEFORE the matched span.
_NEG_PRE = re.compile(
    r"\bno\b|\bnot\b|\bwithout\b|\bdenies\b|\bdenied\b"
    r"|\bnegative\b|\babsent\b|\bfails?\s+to\b",
    re.IGNORECASE,
)
# Words that negate a finding when they appear anywhere in the post window.
# Unanchored so it catches "murphy's sign [is] negative" even when "sign"
# sits between the match end and the negation word.
_NEG_POST = re.compile(
    r"\b(?:negative|absent|not\s+(?:present|appreciated|elicited|found|demonstrated|noted))\b",
    re.IGNORECASE,
)


def _sign_positive(
    text: str,
    pattern: str,
    pre_win: int = 45,
    post_win: int = 35,
) -> bool:
    """
    Return True if `pattern` matches in `text` AND the finding is not negated.

    Operates **sentence by sentence** to prevent negation from a previous
    clause contaminating the next finding.  E.g. in:

        "Murphy's sign: negative.  RLQ tenderness present."

    the "negative" belongs to Murphy's and must not suppress RLQ_tenderness.

    Within each sentence negation is checked in both directions:
      • PRE_WIN chars before the match  → "no Murphy's sign"
      • POST_WIN chars after the match  → "Murphy's sign [is] negative"

    Two-tier pre-window negation
    ----------------------------
    MIMIC PE notes use comma-separated multi-item clauses, e.g.:
        "CV: RRR, no m/g/r  Lungs: CTAB, no w/r/r  Abd: soft, tender RUQ"

    We distinguish two negation classes:

    1. List-spanning negations ("denies / denied"):
       These can govern an enumerated list — "denies nausea, vomiting, fever"
       — so the check window extends back to the last section boundary
       (colon / semicolon), but does NOT cross it.

    2. Clause-local negations ("no / not / without / negative / absent"):
       These negate only the immediately following phrase, so the window is
       clipped at the nearest comma / semicolon / colon.  This prevents
       "no w/r/r, tender RUQ" from mis-negating "RUQ" due to "no w/r/r".
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        for m in re.finditer(pattern, sent, re.IGNORECASE):
            # Full sentence prefix up to the match (used for list-spanning checks)
            full_pre = sent[: m.start()]
            raw_pre  = sent[max(0, m.start() - pre_win): m.start()]
            post     = sent[m.end(): min(len(sent), m.end() + post_win)]

            # ── Tier 1: list-spanning "denies / denied" ──────────────────────
            # "denies nausea, vomiting, fever" — "denies" can be far before the
            # matched term so we scan the *full* sentence prefix, but clip at
            # the last colon/semicolon (section boundary) so negation does not
            # bleed across PE sections ("Lungs: no w/r/r; Abd: tender RUQ").
            major_bdry   = max(full_pre.rfind(";"), full_pre.rfind(":"))
            span_pre     = full_pre[major_bdry + 1:] if major_bdry >= 0 else full_pre
            if re.search(r"\bdenies\b|\bdenied\b", span_pre, re.IGNORECASE):
                continue  # negated by list-spanning denial

            # Also catch parenthetical negative notation "(-) fever" common in
            # MIMIC ROS sections.
            if re.search(r"\(\s*[-–]\s*\)", raw_pre):
                continue

            # ── Tier 2: clause-local negations ───────────────────────────────
            # Clip at the nearest comma/semicolon/colon (within pre_win window)
            # to avoid bleed from prior list items:
            #   "no w/r/r, no rebound, tender RUQ" → clip at last comma
            #   → immediate clause is "tender " → no negation → correct.
            all_bdry = max(raw_pre.rfind(","), raw_pre.rfind(";"), raw_pre.rfind(":"))
            clipped  = raw_pre[all_bdry + 1:] if all_bdry >= 0 else raw_pre
            if re.search(
                r"\bno\b|\bnot\b|\bwithout\b|\bnegative\b|\babsent\b|\bfails?\s+to\b",
                clipped,
                re.IGNORECASE,
            ):
                continue  # negated in immediate clause

            # ── Post-window: "murphy's sign [is] negative" ───────────────────
            if _NEG_POST.search(post):
                continue

            return True
    return False


# ---------------------------------------------------------------------------
# Pain location  (ordered: specific quadrant before diffuse)
# ---------------------------------------------------------------------------

_LOCATION_PATTERNS: list[tuple[str, str]] = [
    ("RUQ",             r"\bruq\b|right\s+upper\s+quadrant"),
    ("RLQ",             r"\brlq\b|right\s+lower\s+quadrant|\bmcburney\b"),
    ("LLQ",             r"\bllq\b|left\s+lower\s+quadrant"),
    ("LUQ",             r"\bluq\b|left\s+upper\s+quadrant"),
    ("Epigastric",      r"\bepigastric\b|\bepigastrium\b"),
    ("Periumbilical",   r"\bperiumbilical\b|peri[-\s]?umbilical\b"
                        r"|umbilical\s+(?:area|region|pain|discomfort)\b"),
    ("Pelvic",          r"\bpelvic\b|\bsuprapubic\b"),
    ("General_Abdomen", r"\bdiffuse\b|generalized\s+abdom|pan[-\s]?abdominal"),
]


def _extract_pain_location(text: str) -> str:
    """
    Return the primary pain location using ordered keyword matching.
    Checks specific quadrants first; falls back to 'Other'.
    """
    t = text.lower()
    for loc, pattern in _LOCATION_PATTERNS:
        if re.search(pattern, t):
            return loc
    return "Other"


# ---------------------------------------------------------------------------
# Symptom duration  (> 72 hours / > 3 days)
# ---------------------------------------------------------------------------

_WORD_TO_INT = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}


def _symptom_over_72h(text: str) -> bool:
    """
    Return True when the text explicitly states a duration > 72 h / > 3 days.

    Handles:
      - Numeric days  ("4 days", "5-6 days" → lower-bound check)
      - Written-out numbers  ("four days", "five days")
      - Qualifiers  ("over 3 days", "more than 3 days", "> 72 hours")
      - Weeks / months  (always > 72 h)
      - "several days" / "multiple days"
    """
    t = text.lower()

    # "X days" — take the first (lower-bound) number in a range
    for m in re.finditer(r"\b(\d+)\s*(?:-\s*\d+\s*)?days?\b", t):
        if int(m.group(1)) >= 4:
            return True

    # Written-out cardinal + "days"
    for word, val in _WORD_TO_INT.items():
        if val >= 4 and re.search(rf"\b{word}\s+days?\b", t):
            return True

    # Explicit qualifier: "over / more than / > 3 days"
    if re.search(r"(?:over|more\s+than|greater\s+than|>)\s*3\s+days?", t):
        return True

    # "X hours" with X > 72
    for m in re.finditer(r"\b(\d+)\s*(?:-\s*\d+\s*)?hours?\b", t):
        if int(m.group(1)) > 72:
            return True

    # "over / more than / > 72 hours"
    if re.search(r"(?:over|more\s+than|greater\s+than|>)\s*72\s+hours?", t):
        return True

    # Weeks or months → always > 72 h
    if re.search(r"\bweeks?\b|\bmonths?\b", t):
        return True

    # "several days" / "multiple days"
    if re.search(r"\bseveral\s+days?\b|\bmultiple\s+days?\b", t):
        return True

    return False


# ---------------------------------------------------------------------------
# HPI feature extraction
# ---------------------------------------------------------------------------

def extract_hpi_features(hpi: str) -> dict:
    """
    Extract all HPI-derived binary features and pain location from free text.

    Uses ordered keyword matching for pain location and negation-aware regex
    for all symptom flags.

    Key subtleties
    --------------
    pain_migration_to_RLQ
        Requires **explicit migration language** (migrated, moved, shifted,
        started X → now Y) with RLQ as the destination.  A note that merely
        mentions both "epigastric" and "RLQ" does NOT trigger this flag.

    epigastric_radiating_to_back
        Requires the co-occurrence of (a) epigastric, (b) radiating/radiation,
        AND (c) back anywhere in the text.

    symptom_duration_over_72h
        Performs numeric + verbal duration arithmetic; only True when duration
        is unambiguously > 72 h / > 3 days.

    gallstone_history / prior_diverticular_disease
        In MIMIC HPI text these are almost exclusively prior diagnoses, so a
        plain keyword match (with negation guard) is sufficient.
    """
    t = hpi.lower()

    # Quantified HPI temperature ≥ 38 °C / 100.4 °F — more specific than bare
    # "fever" keyword; extracted separately to support downstream feature logic.
    def _hpi_has_quantified_fever(text: str) -> bool:
        # "Tmax / Tm / temp 38.5", "fever to 101.2", "temperature of 38"
        for m in re.finditer(
            r"(?:tmax|tm|temp(?:erature)?)[:\s]+(?:of\s+)?(\d{2,3}(?:\.\d{1,2})?)",
            text, re.IGNORECASE,
        ):
            t_raw = float(m.group(1))
            t_c   = (t_raw - 32) * 5 / 9 if t_raw > 50 else t_raw
            if t_c >= 38.0:
                return True
        # "fever to 100.4 / 101 / ..."
        for m in re.finditer(r"fever\s+(?:to|of)\s+(1\d{2}(?:\.\d)?)", text, re.IGNORECASE):
            t_raw = float(m.group(1))
            t_c   = (t_raw - 32) * 5 / 9 if t_raw > 50 else t_raw
            if t_c >= 38.0:
                return True
        return False

    return {
        "pain_location": _extract_pain_location(hpi),

        # Pain started elsewhere and migrated to RLQ (appendicitis hallmark)
        "pain_migration_to_RLQ": bool(
            # "migrated to RLQ / right lower quadrant"
            re.search(
                r"migrat\w+\s+(?:to\s+)?(?:rlq\b|right\s+lower\s+quadrant)", t
            )
            # "started/began/originated … moved/shifted/now [in] RLQ"
            or re.search(
                r"(?:started|began|origin\w+)\b.{0,50}"
                r"(?:now|moved?|shifted?|progressed?)\b.{0,40}"
                r"(?:rlq\b|right\s+lower\s+quadrant)",
                t,
            )
            # "periumbilical/epigastric … migrated/moved … RLQ"
            or re.search(
                r"(?:periumbilical|epigastric|umbilical)\b.{0,60}"
                r"(?:migrat\w+|moved?\s+to|shifted?\s+to)\b.{0,40}"
                r"(?:rlq\b|right\s+lower\s+quadrant)",
                t,
            )
        ),

        # Epigastric pain radiating to the back (classic pancreatitis)
        # Requires all three elements: epigastric, radiating language, and back
        "epigastric_radiating_to_back": bool(
            re.search(r"\bepigastric\b", t)
            and re.search(r"\bradiating?\b|\bradiation\b|\bradiates?\b", t)
            and re.search(r"\bback\b|\bdorsum\b|\bdorsal\b|\binterscapular\b", t)
        ),

        # Bowel habit change (diverticulitis / colonic pathology) — negation-aware
        # Using _sign_positive to avoid false positives from "no diarrhea",
        # "denied melena", "no changes in bowel habits", etc.
        "bowel_habit_change": _sign_positive(
            t,
            r"\bdiarrhea\b|\bdiarrhoea\b|\bconstipation\b|\bloose\s+stool[s]?\b"
            r"|\bbowel\s+habit\w*\s+change\b|\bchange\s+in\s+bowel\w*\b"
            r"|\bbloody\s+stool[s]?\b|\bmelena\b|\bhematochezia\b"
            r"|\balter\w+\s+bowel\b|\brectal\s+bleed\w*\b",
        ),

        "symptom_duration_over_72h": _symptom_over_72h(hpi),

        # Anorexia / loss of appetite — negation-aware
        "anorexia": _sign_positive(
            t,
            r"\banorexia\b|\bloss\s+of\s+appetite\b|\bpoor\s+appetite\b"
            r"|\bnot\s+(?:eating|tolerating\s+(?:food|oral|po))\b"
            r"|\bdecreased\s+(?:appetite|oral\s+intake|po\s+intake)\b",
        ),

        # Nausea and/or vomiting — negation-aware
        "nausea_vomiting": _sign_positive(
            t, r"\bnausea\b|\bvomit\w+\b|\bemesis\b|\bn/v\b"
        ),

        # Significant alcohol use history — negation-aware
        # "social drinker" intentionally excluded (not "significant")
        "alcohol_history": _sign_positive(
            t,
            r"\balcohol\b|\betoh\b|\bethanol\b|\balcoholic\b"
            r"|\bheav\w+\s+drink\w*\b|\bbinge\s+drink\w*\b"
            r"|\bdrinks?\s+(?:heav\w+|daily|regularly)\b",
        ),

        # Prior known gallstones / cholelithiasis — negation-aware
        "gallstone_history": _sign_positive(
            t,
            r"\bgallstones?\b|\bcholelithiasis\b"
            r"|\bcholedocholithiasis\b|\bgallbladder\s+stones?\b",
        ),

        # Prior diverticular disease — negation-aware
        "prior_diverticular_disease": _sign_positive(
            t,
            r"\bdiverticulosis\b|\bdiverticulitis\b|\bdiverticular\s+disease\b",
        ),

        # Fever reported in HPI narrative (TG18 Group B supplement).
        # Many cholecystitis patients arrive already afebrile on admission (PE
        # temp normal) but describe fever at home or en route.  This flag
        # captures that history even when the admission vital is < 38 °C.
        # Two tiers:
        #   1. Keyword: "fever" / "febrile" present and not negated.
        #   2. Quantified: Tmax / temp ≥ 38 °C explicitly stated in HPI.
        "fever_reported_in_hpi": (
            _sign_positive(t, r"\bfever(?:s|ed|ish)?\b|\bfebrile\b")
            or _hpi_has_quantified_fever(t)
        ),
    }


# ---------------------------------------------------------------------------
# Physical Examination sign extraction
# ---------------------------------------------------------------------------

def extract_pe_signs(pe_text: str) -> dict:
    """
    Extract PE-derived clinical signs using negation-aware regex.

    Vitals (fever, HR, RR) are handled separately by the existing helpers and
    are NOT duplicated here.

    Key subtleties
    --------------
    murphys_sign
        Checks pre-negation ("no Murphy's") AND post-negation ("Murphy's sign
        negative" / "Murphy's sign: absent").  The sign is positive only when
        both checks pass.

    RLQ_tenderness
        Includes Rovsing's sign, psoas sign, obturator sign, and McBurney's
        point tenderness as alternative RLQ indicators.

    rebound_tenderness
        Requires "rebound tenderness" (not bare "rebound" which can appear in
        other clinical contexts such as "rebound hypertension").

    peritoneal_signs
        Flags involuntary guarding, rigidity, or explicit peritoneal sign
        language.  Overlaps intentionally with rebound_tenderness (both may
        be True for the same patient).

    RUQ_mass
        Requires BOTH an RUQ locator AND a mass/palpable/fullness keyword;
        this avoids triggering on "RUQ tenderness" alone.
    """
    t = pe_text.lower()
    return {
        # Murphy's sign positive on palpation, including descriptive paraphrases:
        # "inspiratory arrest on RUQ palpation", "catches breath on deep palpation"
        "murphys_sign": (
            _sign_positive(t, r"murphy'?s?\s*(?:sign)?")
            or _sign_positive(
                t,
                r"inspiratory\s+arrest\b|\bcatches?\s+(?:her\s+|his\s+|their\s+)?breath",
            )
        ),

        # RUQ tenderness — TG18 Group A extension.
        # In PE text, "RUQ" / "right upper quadrant" mentioned without negation
        # almost always denotes tenderness ("TTP RUQ", "tender to palpation RUQ").
        # Analogous structure to RLQ_tenderness.
        "RUQ_tenderness": _sign_positive(
            t, r"\bruq\b|\bright\s+upper\s+quadrant\b"
        ),

        # RLQ tenderness — also flags Rovsing's, psoas, obturator, McBurney
        "RLQ_tenderness": (
            _sign_positive(t, r"\brlq\b|\bright\s+lower\s+quadrant\b")
            or _sign_positive(
                t,
                r"\brovsing'?s?\b|\bpsoas\s+sign\b"
                r"|\bobturator\s+sign\b|\bmcburney'?s?\b",
            )
        ),

        # Rebound tenderness (Blumberg's sign)
        # Use "rebound tenderness" to avoid false hits on "rebound" alone
        "rebound_tenderness": _sign_positive(
            t, r"\brebound\s+tenderness\b|\bblumberg'?s?\b"
        ),

        # Palpable RUQ mass (Courvoisier, distended GB, hepatic mass)
        "RUQ_mass": bool(
            re.search(r"\bruq\b|\bright\s+upper\s+quadrant\b", t)
            and _sign_positive(
                t, r"\bmass\b|\bfullness\b|\binduration\b|\bpalpable\b"
            )
        ),

        # Peritoneal signs: involuntary guarding, rigidity, or explicit label
        "peritoneal_signs": (
            _sign_positive(t, r"\bguarding\b|\brigidity\b")
            or _sign_positive(t, r"\bperitoneal\s+sign[s]?\b|\bperitonism\b")
        ),

        # Impaired mental status: confusion, somnolence, GCS < 15
        "impaired_mental_status": bool(re.search(
            r"\bconfus(?:ed|ion)\b|\bsomnolent\b|\bsomnolence\b"
            r"|\baltered\s+(?:mental\s+status|sensorium|mentation|consciousness|loc)\b"
            r"|\blethargi(?:c|a)\b|\bdisoriented\b|\bobtunded\b|\bunresponsive\b"
            r"|\bgcs\s*[<≤:=]?\s*(?:1[0-4]|[1-9])\b",
            t,
        )),
    }


# ---------------------------------------------------------------------------
# Demographics extraction
# ---------------------------------------------------------------------------

def _extract_demographics(row: dict) -> dict:
    """
    Attempt to extract age_gt_60 and is_female_reproductive_age.

    Strategy (in priority order):
      1. Explicit "Age" / "Gender" / "Sex" columns in the CSV row.
      2. Regex parsing of the Patient History text:
           - age: "X-year-old" or "X year old" (MIMIC sometimes leaves this)
           - sex: "female"/"woman"/"she" vs "male"/"man"/"he"
      3. GFR age-group proxy: MIMIC-IV eGFR lab reports embed an "age group
         XX-YY" bracket in the value string (e.g. ">=60 [age group 50-60]").
         The midpoint of that bracket is used as an age estimate.
         This is the primary recovery path when MIMIC redacts ages as '___'.
      4. Conservative default (False) when age remains unknown.

    Note: sex can usually be recovered from HPI pronouns even when age is
    fully redacted.  age_gt_60 will be False whenever no proxy succeeds.
    """
    age: Optional[float] = None
    sex: Optional[str]   = None

    # --- 1. Explicit CSV columns ---
    for col in ("Age", "age", "AGE"):
        raw = str(row.get(col, "")).strip()
        if raw and raw not in ("nan", "None", ""):
            try:
                age = float(raw)
                break
            except ValueError:
                pass

    for col in ("Gender", "gender", "Sex", "sex"):
        raw = str(row.get(col, "")).strip().upper()
        if raw in ("F", "FEMALE"):
            sex = "F"
            break
        if raw in ("M", "MALE"):
            sex = "M"
            break

    # --- 2. Fallback: parse Patient History text ---
    hpi = str(row.get("Patient History", "") or "")

    if age is None:
        m = re.search(r"\b(\d{1,3})\s*[-\s]?\s*year[-\s]?old\b", hpi, re.IGNORECASE)
        if m:
            candidate = float(m.group(1))
            if 0 < candidate < 120:
                age = candidate

    if sex is None:
        if re.search(r"\bfemale\b|\bwoman\b|\bshe\b|\bher\b", hpi, re.IGNORECASE):
            sex = "F"
        elif re.search(r"\bmale\b|\bman\b|\bhe\b|\bhis\b", hpi, re.IGNORECASE):
            sex = "M"

    # --- 3. GFR age-group proxy (MIMIC-IV primary recovery path) ---
    # eGFR reports in MIMIC-IV embed brackets like ">=60 [age group 50-60]"
    # or "age group 40-50" directly in the value string.  Searching the raw
    # Laboratory Tests JSON text (before parsing) is sufficient to find them.
    if age is None:
        lab_raw = str(row.get("Laboratory Tests", "") or "")
        m = re.search(r"\bage\s+group\s+(\d+)[-–]\s*(\d+)", lab_raw, re.IGNORECASE)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            mid = (lo + hi) / 2.0
            if 0 < mid < 120:
                age = mid  # midpoint estimate — sufficient for >60 threshold

    age_gt_60 = age is not None and age > 60
    is_female_repro = sex == "F" and age is not None and 15.0 <= age <= 50.0

    return {
        "age_gt_60": age_gt_60,
        "is_female_reproductive_age": is_female_repro,
    }


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
    hpi       = str(row.get("Patient History", "") or "")

    features: dict = {}

    # ── HPI features (pain profile + symptom flags) ───────────────────────────
    features.update(extract_hpi_features(hpi))

    # ── pain_location fallback: if HPI has no specific quadrant, try PE text ──
    # e.g. "abdominal pain" in HPI but "TTP RUQ" in PE → use "RUQ"
    if features.get("pain_location") == "Other":
        pe_loc = _extract_pain_location(pe_text)
        if pe_loc != "Other":
            features["pain_location"] = pe_loc

    # ── PE signs (Murphy's, RLQ, rebound, peritoneal, mental status) ──────────
    features.update(extract_pe_signs(pe_text))

    # ── Demographics (age, sex) ───────────────────────────────────────────────
    features.update(_extract_demographics(row))

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

    # ── tests_done + lab provenance (MIMIC itemids) ────────────────────────────
    features["tests_done"] = ["Lab_Panel"] if labs else []
    # Stable order for reproducibility / diff-friendly JSON dumps
    try:
        features["lab_itemids"] = sorted(labs.keys(), key=lambda x: int(str(x)))
    except (ValueError, TypeError):
        features["lab_itemids"] = sorted(map(str, labs.keys()))

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
