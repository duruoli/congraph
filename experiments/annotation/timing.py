"""
Timing pipeline: split radiology decision steps into PRE-intervention (diagnostic)
vs POST-intervention (monitoring), using MIMIC-IV source timestamps.

WHY
----
The derived `*_hadm_info_first_diag.{csv,pkl}` aggregates all radiology of a stay and
drops every per-test `charttime`. Decision steps are therefore ordered only by the
`RR-N` note sequence, and a scan taken AFTER a therapeutic intervention (e.g. a CT the
day after an appendectomy, an US after ERCP/sphincterotomy, imaging around a
percutaneous drain) is indistinguishable from a pre-diagnosis test. Those post-
intervention scans are *monitoring*, not diagnostic test-selection, and contaminate the
deviation analysis (they inflate the "redundant / over-imaging / stale-anatomy"
disconfirmed cells — see llm_annotation_analysis.md DD-1 / FD-2 / FD-3).

This module reconstructs the true timeline by joining three MIMIC-IV source tables
(see mimic_supp_data.md) on the `note_id` that the derived data already preserves
inside each Radiology record ("Note ID": "<subject_id>-RR-<n>"), and labels every
decision step with a `timing_role`:

    pre_admission              charttime <  admittime           (ED / outpatient; purest diagnostic)
    post_admission_diagnostic  admittime <= charttime < first_intervention   (admitted, still diagnosing)
    same_day_as_intervention   charttime.date() == first_intervention_date   (ambiguous; flag, keep)
    post_intervention          charttime.date() >  first_intervention_date   (MONITORING -> exclude)

Only HARD therapeutic interventions gate (appendectomy, cholecystectomy,
cholecystostomy/GB-drainage, therapeutic ERCP, percutaneous abdominal drainage, bowel
resection). Pure-diagnostic acts (diagnostic ERCP, cholangiogram, EGD, EUS) are tracked
as SOFT markers but do NOT close the diagnostic window.

This file is pure logic + source loaders. Orchestration / CLI: scripts/build_timing_table.py
It runs TODAY in degraded mode (intervention typing from the titles already in the
derived data); the `charttime` join activates the moment the source CSVs are present.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# 1. Intervention classification
#    Calibrated against the real procedure titles in the 4 derived datasets.
#    Primary signal = title regex (present in derived data AND joinable from
#    d_icd_procedures); ICD9 code set kept for precision once source arrives.
# --------------------------------------------------------------------------- #

# type -> compiled title regex. HARD types close the diagnostic window.
_TITLE_PATTERNS: dict[str, re.Pattern] = {
    "appendectomy": re.compile(r"appendectomy|resection of appendix", re.I),
    "cholecystectomy": re.compile(r"cholecystectomy|resection of gallbladder", re.I),
    # percutaneous aspiration / drainage / cholecystostomy of the gallbladder
    "cholecystostomy": re.compile(
        r"cholecystostomy|aspiration of gallbladder|drainage of gallbladder", re.I
    ),
    # THERAPEUTIC biliary endoscopy (sphincterotomy / stone removal / stent / dilation)
    "ercp_therapeutic": re.compile(
        r"sphincterotomy|papillotomy"
        r"|removal of stone.*biliary|extirpation of matter from common bile duct"
        r"|insertion of stent.*(bile duct|pancreatic duct)"
        r"|dilation of common bile duct",
        re.I,
    ),
    "percutaneous_drainage": re.compile(
        r"percutaneous abdominal drainage|drainage of appendiceal abscess"
        r"|drainage of (peritoneal|pelvic) cavity",
        re.I,
    ),
    "bowel_resection": re.compile(
        r"sigmoidectomy|(resection|excision) of sigmoid colon|colectomy|colostomy|hartmann",
        re.I,
    ),
}

# Diagnostic-only acts: recorded but do NOT gate (window stays open).
_SOFT_TITLE = re.compile(
    r"\[ercp\]|retrograde cholangiopancreatography"      # diagnostic ERCP (no therapy verb)
    r"|cholangiogram|fluoroscopy of bile ducts"           # diagnostic cholangiography
    r"|esophagogastroduodenoscopy|\begd\b"                # EGD
    r"|diagnostic ultrasound|inspection of"               # EUS / endoscopic inspection
    r"|other endoscopy of small intestine",
    re.I,
)

# ICD-9 procedure codes for the HARD types (used when source procedures_icd arrives;
# titles remain primary). Stored as code-prefix sets.
_ICD9_HARD: dict[str, set[str]] = {
    "appendectomy": {"47.0", "47.01", "47.09", "47.2"},
    "cholecystectomy": {"51.22", "51.23", "51.24"},
    "cholecystostomy": {"51.01", "51.02", "51.03", "51.04"},
    "ercp_therapeutic": {"51.84", "51.85", "51.86", "51.87", "51.88", "52.93", "52.94", "52.97"},
    "percutaneous_drainage": {"54.91", "54.0"},
    "bowel_resection": {"45.7", "45.71", "45.72", "45.73", "45.74", "45.75", "45.76",
                         "45.79", "45.8", "45.81", "45.82", "45.83", "46.1", "46.13",
                         "48.5", "48.6"},
}

HARD_TYPES = frozenset(_TITLE_PATTERNS)          # all defined types are HARD gates
# index-organ map: which intervention type kills which disease's primary scan target
DEAD_PREMISE_ORGAN = {
    "appendectomy": "appendix",
    "cholecystectomy": "gallbladder",
    "cholecystostomy": "gallbladder",
    "bowel_resection": "colon",
}


def classify_procedure(title: Optional[str] = None,
                       icd9: Optional[str] = None) -> Optional[str]:
    """Return the HARD intervention type for a procedure, or None.

    Title is primary (always available in derived data). icd9 is a fallback/cross-check
    once the source procedures_icd table is joined. Returns None for diagnostic-only or
    unrelated procedures (catheters, ventilation, nutrition, EGD, diagnostic ERCP ...).
    """
    if title and not _SOFT_TITLE.search(title):
        for typ, pat in _TITLE_PATTERNS.items():
            if pat.search(title):
                return typ
    if icd9:
        code = str(icd9).strip()
        for typ, codes in _ICD9_HARD.items():
            if code in codes or any(code.startswith(c) for c in codes if c.endswith(".") or len(c) <= 4):
                # only honour the ICD9 path if the title did not already veto it as soft
                if not (title and _SOFT_TITLE.search(title)):
                    return typ
    return None


def is_soft_procedure(title: Optional[str]) -> bool:
    return bool(title and _SOFT_TITLE.search(title))


def _iter_titles(cell) -> list[str]:
    """Derived `Procedures ICD9 Title` / `... ICD10 Title` is a stringified list."""
    if not isinstance(cell, str) or cell.strip() in ("", "[]", "nan", "None"):
        return []
    try:
        v = ast.literal_eval(cell)
    except Exception:
        return [cell]
    return [str(x).strip() for x in v] if isinstance(v, (list, tuple)) else [str(v).strip()]


def interventions_from_titles(row: pd.Series) -> dict[str, list[str]]:
    """All HARD intervention types present in a derived-data row's procedure titles.

    Returns {type: [matching title, ...]}. Works TODAY (no source download needed) but
    carries no timing — used for the degraded self-test and for the dead-premise check.
    """
    found: dict[str, list[str]] = {}
    for col in ("Procedures ICD9 Title", "Procedures ICD10 Title"):
        for t in _iter_titles(row.get(col)):
            typ = classify_procedure(title=t)
            if typ:
                found.setdefault(typ, []).append(t)
    return found


# --------------------------------------------------------------------------- #
# 2. Source-table loaders  (schema per mimic_supp_data.md)
#    Files are not present until the PhysioNet DUA clears; loaders validate
#    columns and fail loudly so the pipeline is correct the day they land.
# --------------------------------------------------------------------------- #

@dataclass
class SourceTables:
    """Lazy holder for the three MIMIC-IV source tables.

    admissions   : hosp/admissions.csv[.gz]            -> hadm_id, admittime, dischtime, edregtime, edouttime
    radiology    : mimic-iv-note/radiology.csv[.gz]    -> note_id, hadm_id, charttime, text
    procedures   : hosp/procedures_icd.csv[.gz]        -> hadm_id, seq_num, chartdate, icd_code, icd_version
    d_procedures : hosp/d_icd_procedures.csv[.gz]      -> icd_code, icd_version, long_title   (optional, for titles)
    """
    admissions: pd.DataFrame
    radiology: pd.DataFrame
    procedures: pd.DataFrame
    d_procedures: Optional[pd.DataFrame] = None
    _ct_by_note: Optional[dict] = field(default=None, repr=False, compare=False)

    @staticmethod
    def _read(path: Path, required: set[str]) -> pd.DataFrame:
        df = pd.read_csv(path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns {sorted(missing)}; got {list(df.columns)}")
        return df

    @classmethod
    def load(cls, source_dir: str | Path) -> "SourceTables":
        d = Path(source_dir)

        def find(*names: str) -> Path:
            for n in names:
                for cand in (d / n, d / (n + ".gz")):
                    if cand.exists():
                        return cand
            raise FileNotFoundError(f"none of {names} (.gz ok) under {d}")

        adm = cls._read(find("admissions.csv"),
                        {"hadm_id", "admittime", "dischtime"})
        rad = cls._read(find("radiology.csv"),
                        {"note_id", "hadm_id", "charttime"})
        proc = cls._read(find("procedures_icd.csv"),
                         {"hadm_id", "chartdate", "icd_code", "icd_version"})
        dproc = None
        try:
            dproc = cls._read(find("d_icd_procedures.csv"),
                              {"icd_code", "icd_version", "long_title"})
        except FileNotFoundError:
            pass

        for col in ("admittime", "dischtime", "edregtime", "edouttime"):
            if col in adm:
                adm[col] = pd.to_datetime(adm[col], errors="coerce")
        rad["charttime"] = pd.to_datetime(rad["charttime"], errors="coerce")
        proc["chartdate"] = pd.to_datetime(proc["chartdate"], errors="coerce")
        return cls(adm, rad, proc, dproc)

    # -- per-hadm accessors ------------------------------------------------- #
    def admittime(self, hadm: int) -> Optional[pd.Timestamp]:
        r = self.admissions.loc[self.admissions.hadm_id == hadm, "admittime"]
        return r.iloc[0] if len(r) else None

    def charttime_by_note(self, hadm: int) -> dict[str, pd.Timestamp]:
        sub = self.radiology[self.radiology.hadm_id == hadm]
        return dict(zip(sub.note_id.astype(str), sub.charttime))

    def charttime_for_note(self, note_id: str) -> Optional[pd.Timestamp]:
        """charttime for a note by its GLOBAL note_id, ignoring hadm_id.

        ED / outpatient radiology often carries hadm_id=NULL in the source table, so a
        per-hadm filter (charttime_by_note) silently drops exactly the pre-admission
        diagnostic scans we care about. The derived data already pinned each Radiology
        record to its stay via its note_id ("<subject_id>-RR-<n>"), and note_id is unique
        across the whole table, so we look the timestamp up directly by that key.
        """
        if self._ct_by_note is None:
            self._ct_by_note = dict(zip(self.radiology.note_id.astype(str),
                                        self.radiology.charttime))
        return self._ct_by_note.get(str(note_id))

    def first_intervention(self, hadm: int) -> tuple[Optional[pd.Timestamp], Optional[str]]:
        """(date, type) of the earliest HARD therapeutic intervention, by chartdate."""
        sub = self.procedures[self.procedures.hadm_id == hadm].copy()
        if sub.empty:
            return None, None
        titles = None
        if self.d_procedures is not None:
            titles = self.d_procedures.set_index(["icd_code", "icd_version"])["long_title"].to_dict()
        best_date, best_type = None, None
        for _, p in sub.iterrows():
            title = titles.get((p.icd_code, p.icd_version)) if titles else None
            typ = classify_procedure(title=title, icd9=p.icd_code if p.icd_version == 9 else None)
            if typ and pd.notna(p.chartdate):
                if best_date is None or p.chartdate < best_date:
                    best_date, best_type = p.chartdate, typ
        return best_date, best_type


# --------------------------------------------------------------------------- #
# 3. Timing role
# --------------------------------------------------------------------------- #

PRE_ADMISSION = "pre_admission"
POST_ADMIT_DIAGNOSTIC = "post_admission_diagnostic"
SAME_DAY_AS_INTERVENTION = "same_day_as_intervention"
POST_INTERVENTION = "post_intervention"
UNKNOWN = "unknown"  # charttime not joinable yet

# roles that should be EXCLUDED from deviation analysis
MONITORING_ROLES = frozenset({POST_INTERVENTION})


def timing_role(charttime: Optional[pd.Timestamp],
                admittime: Optional[pd.Timestamp],
                first_intervention_date: Optional[pd.Timestamp]) -> str:
    if charttime is None or pd.isna(charttime):
        return UNKNOWN
    if first_intervention_date is not None and pd.notna(first_intervention_date):
        cd, fid = charttime.normalize(), first_intervention_date.normalize()
        if cd > fid:
            return POST_INTERVENTION
        if cd == fid:
            return SAME_DAY_AS_INTERVENTION
    if admittime is not None and pd.notna(admittime) and charttime < admittime:
        return PRE_ADMISSION
    return POST_ADMIT_DIAGNOSTIC
