"""test_burden_cost.py — Absolute cost (USD) and relative patient-burden lookup.

Covers every modality in VALID_TESTS:
  Lab_Panel, Ultrasound_Abdomen, CT_Abdomen, HIDA_Scan,
  MRCP_Abdomen, Radiograph_Chest, MRI_Abdomen

═══════════════════════════════════════════════════════════════════════════════
COST  —  Absolute USD values (CMS 2025 non-facility national payment amounts)
═══════════════════════════════════════════════════════════════════════════════
Values are midpoints of the 2025 CMS Medicare Physician Fee Schedule
non-facility national allowed amounts.

  Radiograph_Chest    CPT 71046   range $55–80    → midpoint $67
  Lab_Panel           sum of CLFS 2024 NLA  → $49.00
      (CBC 85025: $7.77, CMP 80053: $10.56, Lipase 83690: $6.89,
       UA 81003: $2.25, CRP 86140: $5.18, hCG 84703: $7.52, Venipuncture 36415: $8.83)
  Ultrasound_Abdomen  CPT 76705   range $140–200  → midpoint $170
  HIDA_Scan           CPT 78226   range $450–650  → midpoint $550
  CT_Abdomen          CPT 74177   range $620–950  → midpoint $785
  MRCP_Abdomen        CPT 74183   range $950–1500 → midpoint $1225
  MRI_Abdomen         CPT 74183   range $1050–1700→ midpoint $1375

Reference
---------
  Centers for Medicare & Medicaid Services (2025). *Physician Fee Schedule —
  Non-Facility National Payment Amounts*. Retrieved from
  https://www.cms.gov/medicare/payment/fee-schedules/physician

═══════════════════════════════════════════════════════════════════════════════
BURDEN  —  Relative 1–10 scale (higher = more burdensome to the patient)
═══════════════════════════════════════════════════════════════════════════════
Scale semantics
---------------
  1  — minimal   : seconds, no radiation, no IV access required
  2  — mild      : brief discomfort (e.g. venipuncture)
  3  — low       : mild probe pressure, no radiation
  4  — moderate  : IV contrast risk + meaningful radiation, 5–10 min scan
  5  — mod-high  : prolonged scan (30–75 min), claustrophobia, breath-holds
  6  — high      : radiotracer IV injection + 2–4 h wait + repeated imaging
  7+ — reserved for invasive procedures not in current test set

Per-test burden ratings and supporting literature
-------------------------------------------------
  Radiograph_Chest   → 1
    30-second standing exposure; effective dose ~0.1 mSv (≈ 10 days of
    background radiation). Classified as "trivial" patient burden.
    [Wall et al., *Radiology* 2011; ICRP Publication 103, 2007]

  Ultrasound_Abdomen → 1
    Non-invasive, no ionising radiation, no IV access; mild probe pressure
    only. Regarded as the lowest-burden abdominal imaging modality.
    [ACR–AIUM–SPR–SRU Practice Parameter for Abdominal Ultrasound, 2022]

  Lab_Panel          → 2
    Standard venipuncture; brief sting, minor bruising risk (~10–15 min).
    [Witting et al., *Ann Emerg Med* 2005; WHO Phlebotomy Guidelines, 2010]

  CT_Abdomen         → 4
    Effective dose ~8–10 mSv (≈ 3 years background radiation); IV iodinated
    contrast carries allergy risk (~0.6% minor, ~0.04% severe reactions).
    Scan time 5–10 min.
    [Brenner & Hall, *NEJM* 2007; ACR Manual on Contrast Media, 2023]

  MRCP_Abdomen       → 5
    30–45 min scan; claustrophobia reported in ~5–10% of patients; repeated
    breath-holds required; no ionising radiation but IV gadolinium optional.
    [Dill et al., *J Magn Reson Imaging* 2008; Enders et al., *Eur Radiol*
    2011 (claustrophobia prevalence)]

  MRI_Abdomen        → 5
    45–75 min scan; same claustrophobia and breath-hold burden as MRCP;
    IV gadolinium contrast common (nephrogenic systemic fibrosis risk at
    eGFR < 30). Scored identically to MRCP.
    [Dill et al., *J Magn Reson Imaging* 2008; ACR Manual on Contrast
    Media, 2023]

  HIDA_Scan          → 6
    IV radiotracer (99mTc-IDA); patient waits 2–4 h for hepatobiliary
    clearance with repeated gamma-camera acquisitions; effective dose
    ~2.5–3 mSv from the tracer plus extended facility time (~4 h total).
    [ACR–SPR Practice Parameter for Hepatobiliary Scintigraphy, 2022;
    Shah et al., *J Nucl Med* 2016]
"""

from __future__ import annotations

import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Burden: relative 1–10 scale (see docstring for per-test literature support)
# ---------------------------------------------------------------------------
TEST_BURDEN: dict[str, float] = {
    "Radiograph_Chest":    1.0,   # trivial: 30-sec, ~0.1 mSv
    "Ultrasound_Abdomen":  1.0,   # non-invasive, no radiation
    "Lab_Panel":           2.0,   # venipuncture: brief sting
    "CT_Abdomen":          4.0,   # IV contrast risk + ~8–10 mSv radiation
    "MRCP_Abdomen":        5.0,   # 30–45 min scan, claustrophobia
    "MRI_Abdomen":         5.0,   # 45–75 min scan, optional IV gadolinium
    "HIDA_Scan":           6.0,   # radiotracer IV + 2–4 h wait
}

# ---------------------------------------------------------------------------
# Cost loading from data/raw_data/cost_mapping.csv
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CSV_PATH = _REPO_ROOT / "data" / "raw_data" / "cost_mapping.csv"

TEST_COST: dict[str, float] = {"Lab_Panel": 49.0}  # default fallback
LAB_COMPONENT_USD: dict[str, float] = {}
_ITEMID_TO_COMPONENT: dict[str, str] = {}
UNMAPPED_LAB_ITEM_USD: float = 12.0
VENIPUNCTURE_FEE: float = 8.83

if _CSV_PATH.exists():
    with _CSV_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            itype = row["item_type"]
            iid = row["item_id"]
            cost_str = row["cost_usd"]
            cost = float(cost_str) if cost_str else 0.0
            
            if itype == "imaging":
                TEST_COST[iid] = cost
            elif itype == "lab_bundle":
                LAB_COMPONENT_USD[iid] = cost
            elif itype == "lab_itemid":
                _ITEMID_TO_COMPONENT[iid] = row["category"]
            elif itype == "fee":
                if iid == "venipuncture":
                    VENIPUNCTURE_FEE = cost
                elif iid == "unmapped_lab":
                    UNMAPPED_LAB_ITEM_USD = cost


def lab_panel_cost_usd(item_ids: list[str] | tuple[str, ...] | None) -> float:
    """
    Estimate Lab_Panel USD from MIMIC Laboratory Tests JSON keys.

    When ``item_ids`` is missing or empty, returns the legacy bundle midpoint
    (``TEST_COST['Lab_Panel']``) so counterfactuals without lab provenance
    stay comparable to the old behaviour.
    """
    if not item_ids:
        return float(TEST_COST["Lab_Panel"])

    ids = {str(x).strip() for x in item_ids if str(x).strip()}
    if not ids:
        return float(TEST_COST["Lab_Panel"])

    components: set[str] = set()
    for iid in ids:
        comp = _ITEMID_TO_COMPONENT.get(iid)
        if comp:
            components.add(comp)
    cost = sum(LAB_COMPONENT_USD.get(c, 0.0) for c in components)

    mapped = {i for i in ids if i in _ITEMID_TO_COMPONENT}
    unmapped_n = len(ids - mapped)
    cost += float(unmapped_n) * UNMAPPED_LAB_ITEM_USD

    # Add venipuncture flat fee if any blood tests are ordered.
    # We assume 'urinalysis' is urine and does not require venipuncture. All other
    # mapped components and any unmapped tests are assumed to be blood draws.
    blood_components = components - {"urinalysis"}
    if blood_components or unmapped_n > 0:
        cost += VENIPUNCTURE_FEE

    return float(cost)
