"""test_burden_cost.py — Relative cost and patient burden lookup table.

Covers every modality in VALID_TESTS:
  Lab_Panel, Ultrasound_Abdomen, CT_Abdomen, HIDA_Scan,
  MRCP_Abdomen, Radiograph_Chest, MRI_Abdomen

Scores use a 1–10 relative scale (higher = more expensive / more burdensome).
Values were estimated with LLM reasoning and spot-checked against 2025 CMS
Medicare Physician Fee Schedule national non-facility rates.

LLM prompt used to derive initial estimates
-------------------------------------------
  "For each of the following abdominal diagnostic tests, rate on a 1–10 scale:
   (A) relative cost to the healthcare system (1 = cheapest, 10 = most expensive)
   (B) patient burden — discomfort, time, invasiveness, or radiation risk
       (1 = minimal, 10 = highly burdensome).
   Tests: Lab Panel (CBC+CMP+Lipase+UA+CRP+β-hCG), Abdominal Ultrasound,
   Abdominal CT with IV contrast, HIDA Scan, MRCP, Chest X-ray, MRI Abdomen."

CMS 2025 non-facility approximate rates (used for spot-check ordering)
-----------------------------------------------------------------------
  Radiograph_Chest    CPT 71046   ~$55–80
  Lab_Panel           CPT bundle  ~$90–170  (CBC 85025, CMP 80053, Lipase 83690,
                                             UA 81003, CRP 86140, hCG 84703)
  Ultrasound_Abdomen  CPT 76705   ~$140–200
  HIDA_Scan           CPT 78226   ~$450–650
  CT_Abdomen          CPT 74177   ~$620–950
  MRCP_Abdomen        CPT 74183   ~$950–1500
  MRI_Abdomen         CPT 74183   ~$1050–1700  (full abdomen, may include MRCP)

Burden scale semantics
----------------------
  1  — minimal: no pain, no radiation, takes seconds
  2  — mild: brief discomfort (e.g. blood draw)
  3  — low: mild pressure or probe contact, no radiation
  4  — moderate: IV contrast risk, meaningful radiation, or ~5–10 min scan
  5  — moderate-high: prolonged scan (~30–45 min), claustrophobia, breath-holds
  6  — high: radiotracer IV + extended wait + repeated imaging
  7+ — reserved for invasive procedures (not in current test set)

Note: values are intentionally coarse (integer 1–10). Exact values are
refinable; relative ordering is sufficient for v1 evaluation.
"""

from __future__ import annotations

# Relative cost on a 1–10 scale (higher = more expensive)
TEST_COST: dict[str, float] = {
    "Radiograph_Chest":    1.0,   # CMS ~$55–80
    "Lab_Panel":           2.0,   # CMS ~$90–170 bundled
    "Ultrasound_Abdomen":  3.0,   # CMS ~$140–200
    "HIDA_Scan":           6.0,   # CMS ~$450–650; nuclear medicine facility overhead
    "CT_Abdomen":          7.0,   # CMS ~$620–950; most commonly ordered abdominal CT
    "MRCP_Abdomen":        8.0,   # CMS ~$950–1500; MR biliary focused
    "MRI_Abdomen":         9.0,   # CMS ~$1050–1700; full MRI abdomen/pelvis
}

# Relative patient burden on a 1–10 scale (higher = more burdensome)
TEST_BURDEN: dict[str, float] = {
    "Radiograph_Chest":    1.0,   # 30-sec standing pose; radiation ~0.1 mSv
    "Ultrasound_Abdomen":  1.0,   # non-invasive, no radiation, mild probe pressure
    "Lab_Panel":           2.0,   # venipuncture: brief sting, minor bruising risk
    "CT_Abdomen":          4.0,   # IV contrast risk, radiation ~8–10 mSv, 5–10 min
    "MRCP_Abdomen":        5.0,   # 30–45 min, loud, claustrophobia, breath-holds
    "MRI_Abdomen":         5.0,   # 45–75 min, loud, claustrophobia, possible IV contrast
    "HIDA_Scan":           6.0,   # radiotracer IV + 2–4 h wait + repeated gamma imaging
}
