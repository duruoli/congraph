"""toy_patients.py

Static "script" patients for pipeline demonstration and traversal testing.
静态剧本患者，用于pipeline演示和traversal engine测试。

Each patient is defined as 4 progressive snapshots of a feature dict,
simulating incremental information acquisition in a real clinical encounter:

  step0  — HPI + Physical Examination only        (no tests done)
  step1  — step0  +  Basic Labs                   (Lab_Panel completed)
  step2  — step1  +  Abdominal Ultrasound          (Ultrasound_Abdomen completed)
  step3  — step2  +  CT Abdomen/Pelvis             (CT_Abdomen completed)

Each step is a complete, self-contained feature dict (not a delta).
每个step是完整的特征字典（非增量diff）。

Diseases covered (2 patients each):
  - Cholecystitis  :  C1 Grade I (mild)  ·  C2 Grade II (moderate)
  - Pancreatitis   :  P1 Mild            ·  P2 Severe
  - Diverticulitis :  D1 Uncomplicated   ·  D2 Hinchey II (complicated)

EXPECTED_TERMINALS records the rubric-graph terminal node each patient should
reach at steps where a terminal is reachable; None means the traversal is
still "in progress" (required tests not yet done).
"""

from __future__ import annotations

from feature_schema import default_features


# ---------------------------------------------------------------------------
# Internal helper  内部辅助函数
# ---------------------------------------------------------------------------

def _build(overrides: dict) -> dict:
    """Return a default feature dict updated with the given overrides."""
    f = default_features()
    f.update(overrides)
    return f


# ===========================================================================
# CHOLECYSTITIS
# ===========================================================================

# ---------------------------------------------------------------------------
# C1 — Grade I Mild Acute Calculous Cholecystitis
# 55-year-old woman.  Classic RUQ pain after a fatty meal, Murphy's sign,
# moderate leukocytosis.  US confirms gallstones + wall thickening.
# No organ dysfunction and no Grade-II local complication criteria → GRADE_I.
# ---------------------------------------------------------------------------

_C1_HPI = dict(
    pain_location="RUQ",
    murphys_sign=True,          # positive on physical exam
    nausea_vomiting=True,
    gallstone_history=True,
)

C1_STEPS: dict[str, dict] = {
    # ── step0: HPI + PE only ────────────────────────────────────────────────
    "step0": _build({
        **_C1_HPI,
        "tests_done": [],
    }),

    # ── step1: + Basic Labs ─────────────────────────────────────────────────
    # WBC 13,500  CRP 42 mg/L
    # Triage gate: RUQ + murphys_sign + WBC_gt_10k → ROUTE_CHOLECYSTITIS
    "step1": _build({
        **_C1_HPI,
        "WBC_gt_10k": True,
        "CRP_elevated": True,
        "tests_done": ["Lab_Panel"],
    }),

    # ── step2: + Abdominal Ultrasound ───────────────────────────────────────
    # US: gallstones present, GB wall 6 mm (>4 mm), sonographic Murphy's sign.
    # TG18 C-group positive → definite diagnosis → SEVERITY_GRADING.
    # No Grade-II criteria (WBC <18k, no mass, symptoms <72h) → GRADE_I.
    "step2": _build({
        **_C1_HPI,
        "WBC_gt_10k": True,
        "CRP_elevated": True,
        "US_gallstones": True,
        "US_GB_wall_thickening": True,
        "US_sonographic_murphys": True,
        "tests_done": ["Lab_Panel", "Ultrasound_Abdomen"],
    }),

    # ── step3: + CT Abdomen ─────────────────────────────────────────────────
    # CT confirms cholecystitis; no gangrenous / emphysematous findings.
    # Severity unchanged: GRADE_I.
    "step3": _build({
        **_C1_HPI,
        "WBC_gt_10k": True,
        "CRP_elevated": True,
        "US_gallstones": True,
        "US_GB_wall_thickening": True,
        "US_sonographic_murphys": True,
        "CT_cholecystitis_positive": True,
        "CT_GB_severe_findings": False,
        "tests_done": ["Lab_Panel", "Ultrasound_Abdomen", "CT_Abdomen"],
    }),
}

# ---------------------------------------------------------------------------
# C2 — Grade II Moderate Acute Calculous Cholecystitis
# 68-year-old man.  4-day RUQ pain (symptom duration >72h), fever 38.6°C,
# WBC 22,000 (>18k), pericholecystic fluid on US.
# Grade-II criteria met (WBC_gt_18k + symptom_duration_over_72h), no organ
# dysfunction → GRADE_II.
# ---------------------------------------------------------------------------

_C2_HPI = dict(
    pain_location="RUQ",
    murphys_sign=True,
    nausea_vomiting=True,
    gallstone_history=True,
    symptom_duration_over_72h=True,     # 4-day history
)

C2_STEPS: dict[str, dict] = {
    "step0": _build({
        **_C2_HPI,
        "tests_done": [],
    }),

    # WBC 22,000; CRP 118 mg/L; Temp 38.6 °C
    # TG18 B-group: fever_temp_ge_38 + CRP_elevated + WBC_gt_10k (all three)
    "step1": _build({
        **_C2_HPI,
        "fever_temp_ge_38": True,
        "WBC_gt_10k": True,
        "WBC_gt_18k": True,             # Grade-II local criterion
        "CRP_elevated": True,
        "tests_done": ["Lab_Panel"],
    }),

    # US: gallstones, wall thickening 8 mm, pericholecystic fluid.
    # TG18 C positive → definite diagnosis.
    # Grade-II: WBC_gt_18k=True AND symptom_duration_over_72h=True → GRADE_II.
    "step2": _build({
        **_C2_HPI,
        "fever_temp_ge_38": True,
        "WBC_gt_10k": True,
        "WBC_gt_18k": True,
        "CRP_elevated": True,
        "US_gallstones": True,
        "US_GB_wall_thickening": True,
        "US_pericholecystic_fluid": True,
        "tests_done": ["Lab_Panel", "Ultrasound_Abdomen"],
    }),

    # CT: confirms cholecystitis; no gangrenous/emphysematous changes.
    # Still GRADE_II (no organ dysfunction).
    "step3": _build({
        **_C2_HPI,
        "fever_temp_ge_38": True,
        "WBC_gt_10k": True,
        "WBC_gt_18k": True,
        "CRP_elevated": True,
        "US_gallstones": True,
        "US_GB_wall_thickening": True,
        "US_pericholecystic_fluid": True,
        "CT_cholecystitis_positive": True,
        "CT_GB_severe_findings": False,
        "tests_done": ["Lab_Panel", "Ultrasound_Abdomen", "CT_Abdomen"],
    }),
}


# ===========================================================================
# PANCREATITIS
# ===========================================================================

# ---------------------------------------------------------------------------
# P1 — Mild Acute Pancreatitis (Alcohol-induced)
# 42-year-old man.  Binge drinking two days ago.  Sudden epigastric pain
# radiating to back, nausea/vomiting.  Lipase 1,850 U/L (≥3× ULN).
# BISAP = 0.  No organ failure, no local complications → MILD_AP.
# ---------------------------------------------------------------------------

_P1_HPI = dict(
    pain_location="Epigastric",
    epigastric_radiating_to_back=True,
    nausea_vomiting=True,
    alcohol_history=True,
)

P1_STEPS: dict[str, dict] = {
    "step0": _build({
        **_P1_HPI,
        "tests_done": [],
    }),

    # Lipase 1,850 U/L (≥3×ULN).
    # Triage: Epigastric + lipase_ge_3xULN → ROUTE_PANCREATITIS.
    # Revised Atlanta criteria met: ①epigastric + ②lipase ≥2 → CONFIRMED (pending US).
    "step1": _build({
        **_P1_HPI,
        "lipase_ge_3xULN": True,
        "tests_done": ["Lab_Panel"],
    }),

    # US: no gallstones (alcohol etiology).
    # CONFIRMED node satisfied (US done).
    # BISAP = 0 (no BUN↑, no SIRS, no age>60, no pleural effusion) → ATLANTA_LOW.
    # No organ failure, no local complications → MILD_AP.
    "step2": _build({
        **_P1_HPI,
        "lipase_ge_3xULN": True,
        "US_gallstones": False,
        "tests_done": ["Lab_Panel", "Ultrasound_Abdomen"],
    }),

    # CT: pancreatic edema, minimal peripancreatic fluid; CTSI = 2.
    # No necrosis; no organ failure → still MILD_AP.
    "step3": _build({
        **_P1_HPI,
        "lipase_ge_3xULN": True,
        "US_gallstones": False,
        "CT_pancreatitis_positive": True,
        "CTSI_score": 2.0,
        "tests_done": ["Lab_Panel", "Ultrasound_Abdomen", "CT_Abdomen"],
    }),
}

# ---------------------------------------------------------------------------
# P2 — Severe Acute Pancreatitis (Gallstone-induced)
# 71-year-old man.  Prior gallstone history.  Sudden severe epigastric pain,
# profound nausea.  Labs: lipase 3,200 U/L, BUN 32, SIRS ×2 criteria.
# BISAP ≥ 3 (BUN + SIRS + age>60 + pleural effusion = 4).
# Develops persistent respiratory failure → SEVERE_AP.
# ---------------------------------------------------------------------------

_P2_HPI = dict(
    pain_location="Epigastric",
    epigastric_radiating_to_back=True,
    nausea_vomiting=True,
    gallstone_history=True,
    age_gt_60=True,                 # 71 years old
)

P2_STEPS: dict[str, dict] = {
    "step0": _build({
        **_P2_HPI,
        "tests_done": [],
    }),

    # Lipase 3,200 U/L; BUN 32 mg/dL; SIRS: Temp 38.9 + HR 108 + WBC 14k.
    # Triage: Epigastric + lipase_ge_3xULN → ROUTE_PANCREATITIS.
    # Atlanta: ①+② → CONFIRMED (pending US).
    # Provisional BISAP (without imaging): BUN=1 + SIRS=1 + age=1 = 3 → high-risk.
    "step1": _build({
        **_P2_HPI,
        "lipase_ge_3xULN": True,
        "BUN_gt_25": True,
        "SIRS_criteria_ge_2": True,
        "WBC_gt_10k": True,
        "fever_temp_ge_38": True,
        "creatinine_elevated": True,
        "tests_done": ["Lab_Panel"],
    }),

    # US: gallstones confirmed (biliary etiology).
    # Atlanta criteria: ①+②+③ = 3 → CONFIRMED node satisfied.
    # BISAP = 3 so far (pleural effusion not yet seen) → ATLANTA_HIGH (pending CT).
    "step2": _build({
        **_P2_HPI,
        "lipase_ge_3xULN": True,
        "BUN_gt_25": True,
        "SIRS_criteria_ge_2": True,
        "WBC_gt_10k": True,
        "fever_temp_ge_38": True,
        "creatinine_elevated": True,
        "US_gallstones": True,
        "tests_done": ["Lab_Panel", "Ultrasound_Abdomen"],
    }),

    # CT at 48h: extensive pancreatic necrosis, bilateral pleural effusions.
    # CTSI = 7.  Persistent respiratory failure develops (organ_failure_persistent).
    # BISAP = 4 (BUN + SIRS + age + pleural effusion).
    # ATLANTA_HIGH → ORGAN_FAILURE_ASSESS → persistent failure → CT_CTSI → SEVERE_AP.
    "step3": _build({
        **_P2_HPI,
        "lipase_ge_3xULN": True,
        "BUN_gt_25": True,
        "SIRS_criteria_ge_2": True,
        "WBC_gt_10k": True,
        "fever_temp_ge_38": True,
        "creatinine_elevated": True,
        "US_gallstones": True,
        "CT_pancreatitis_positive": True,
        "CTSI_score": 7.0,
        "pleural_effusion_on_imaging": True,
        "organ_failure_persistent": True,
        "tests_done": ["Lab_Panel", "Ultrasound_Abdomen", "CT_Abdomen"],
    }),
}


# ===========================================================================
# DIVERTICULITIS
# ===========================================================================

# ---------------------------------------------------------------------------
# D1 — Uncomplicated Acute Diverticulitis
# 62-year-old woman.  Known diverticular disease (prior colonoscopy).
# 2-day LLQ pain with change in bowel habits, low-grade fever 38.2°C,
# WBC 12,400, CRP 68 mg/L.  CT: sigmoid wall thickening + fat stranding,
# no abscess or perforation → UNCOMPLICATED.
# ---------------------------------------------------------------------------

_D1_HPI = dict(
    pain_location="LLQ",
    bowel_habit_change=True,
    prior_diverticular_disease=True,
    nausea_vomiting=True,
)

D1_STEPS: dict[str, dict] = {
    "step0": _build({
        **_D1_HPI,
        "tests_done": [],
    }),

    # Temp 38.2°C; WBC 12,400; CRP 68 mg/L (not >200).
    # Triage: LLQ + fever_temp_ge_38 + prior_diverticular_disease → ROUTE_DIVERTICULITIS.
    # Diverticulitis CLINICAL_ASSESSMENT:
    #   LLQ + fever_temp_ge_38 + CRP_elevated + no peritoneal_signs + no organ_dysfunction
    #   + no CRP_gt_200 → CLINICAL_DIAGNOSIS (then routes to CT anyway).
    "step1": _build({
        **_D1_HPI,
        "fever_temp_ge_38": True,
        "WBC_gt_10k": True,
        "CRP_elevated": True,
        "tests_done": ["Lab_Panel"],
    }),

    # US: non-diagnostic for diverticulitis (standard — US has low sensitivity
    # for sigmoid diverticulitis).  CT still required for confirmation.
    "step2": _build({
        **_D1_HPI,
        "fever_temp_ge_38": True,
        "WBC_gt_10k": True,
        "CRP_elevated": True,
        "tests_done": ["Lab_Panel", "Ultrasound_Abdomen"],
    }),

    # CT: sigmoid wall thickening + pericolic fat stranding; no abscess or free air.
    # CT_diverticulitis_confirmed=True, CT_diverticulitis_complicated=False → UNCOMPLICATED.
    "step3": _build({
        **_D1_HPI,
        "fever_temp_ge_38": True,
        "WBC_gt_10k": True,
        "CRP_elevated": True,
        "CT_diverticulitis_confirmed": True,
        "CT_diverticulitis_complicated": False,
        "tests_done": ["Lab_Panel", "Ultrasound_Abdomen", "CT_Abdomen"],
    }),
}

# ---------------------------------------------------------------------------
# D2 — Complicated Acute Diverticulitis, Hinchey II (Distant Abscess ≥ 3 cm)
# 75-year-old man.  5-day LLQ pain, known prior diverticular disease.
# Markedly elevated CRP 310 mg/L (>200 — suggests possible perforation),
# fever 38.9°C, WBC 19,500.  CT: pelvic abscess 4.5 cm → HINCHEY_II.
# ---------------------------------------------------------------------------

_D2_HPI = dict(
    pain_location="LLQ",
    bowel_habit_change=True,
    prior_diverticular_disease=True,
    nausea_vomiting=True,
    symptom_duration_over_72h=True,    # 5-day history
    age_gt_60=True,
)

D2_STEPS: dict[str, dict] = {
    "step0": _build({
        **_D2_HPI,
        "tests_done": [],
    }),

    # CRP 310 mg/L (>200); WBC 19,500; Temp 38.9°C.
    # Triage: LLQ + fever_temp_ge_38 + prior_diverticular_disease → ROUTE_DIVERTICULITIS.
    # Diverticulitis CLINICAL_ASSESSMENT:
    #   CRP_gt_200=True → high-risk route → CT_ABD_PELVIS directly (skips clinical diagnosis).
    "step1": _build({
        **_D2_HPI,
        "fever_temp_ge_38": True,
        "WBC_gt_10k": True,
        "CRP_elevated": True,
        "CRP_gt_200": True,             # key high-risk flag → direct CT pathway
        "tests_done": ["Lab_Panel"],
    }),

    # US: non-diagnostic; possible LLQ complex fluid but not characterised.
    # CT required for Hinchey classification.
    "step2": _build({
        **_D2_HPI,
        "fever_temp_ge_38": True,
        "WBC_gt_10k": True,
        "CRP_elevated": True,
        "CRP_gt_200": True,
        "tests_done": ["Lab_Panel", "Ultrasound_Abdomen"],
    }),

    # CT: sigmoid diverticulitis confirmed; distant pelvic abscess 4.5 cm.
    # CT_diverticulitis_complicated=True → HINCHEY_GRADING.
    # CT_abscess_ge_3cm=True (no purulent/fecal peritonitis) → HINCHEY_II.
    "step3": _build({
        **_D2_HPI,
        "fever_temp_ge_38": True,
        "WBC_gt_10k": True,
        "CRP_elevated": True,
        "CRP_gt_200": True,
        "CT_diverticulitis_confirmed": True,
        "CT_diverticulitis_complicated": True,
        "CT_abscess_ge_3cm": True,
        "tests_done": ["Lab_Panel", "Ultrasound_Abdomen", "CT_Abdomen"],
    }),
}


# ===========================================================================
# Master SCENARIOS dict  总场景字典
# ===========================================================================

SCENARIOS: dict[str, dict[str, dict]] = {
    "C1_cholecystitis_mild":        C1_STEPS,
    "C2_cholecystitis_moderate":    C2_STEPS,
    "P1_pancreatitis_mild":         P1_STEPS,
    "P2_pancreatitis_severe":       P2_STEPS,
    "D1_diverticulitis_uncomplicated": D1_STEPS,
    "D2_diverticulitis_hinchey2":   D2_STEPS,
}


# ===========================================================================
# Expected terminal nodes  预期终态节点（用于断言测试）
# ===========================================================================
# None  → traversal still in progress (required tests not yet completed)
# str   → rubric-graph terminal node id that the traversal should reach
#
# Key design decisions recorded here:
#   C1/C2 reach a terminal at step2 (US confirms TG18 C → severity grading)
#   P1    reaches MILD_AP at step2 (US satisfies CONFIRMED required_tests)
#   P2    cannot classify severity until CT at step3 (ATLANTA_HIGH needs CT)
#   D1/D2 cannot reach a terminal before CT (CT is the gold standard)

EXPECTED_TERMINALS: dict[str, dict[str, str | None]] = {
    "C1_cholecystitis_mild": {
        "step0": None,
        "step1": None,          # US not done yet; IMAGING_US node pending
        "step2": "GRADE_I",     # US positive + no Grade-II criteria
        "step3": "GRADE_I",
    },
    "C2_cholecystitis_moderate": {
        "step0": None,
        "step1": None,
        "step2": "GRADE_II",    # WBC_gt_18k + symptom_duration_over_72h
        "step3": "GRADE_II",
    },
    "P1_pancreatitis_mild": {
        "step0": None,
        "step1": None,          # CONFIRMED node requires Ultrasound_Abdomen
        "step2": "MILD_AP",     # BISAP=0 → ATLANTA_LOW → no organ failure
        "step3": "MILD_AP",
    },
    "P2_pancreatitis_severe": {
        "step0": None,
        "step1": None,          # CONFIRMED pending US
        "step2": None,          # ATLANTA_HIGH pending CT (required_tests=["CT_Abdomen"])
        "step3": "SEVERE_AP",   # persistent organ failure → CT_CTSI → SEVERE_AP
    },
    "D1_diverticulitis_uncomplicated": {
        "step0": None,
        "step1": None,          # CLINICAL_DIAGNOSIS reached but CT still needed
        "step2": None,          # CT not yet done
        "step3": "UNCOMPLICATED",
    },
    "D2_diverticulitis_hinchey2": {
        "step0": None,
        "step1": None,          # CT_ABD_PELVIS pending
        "step2": None,
        "step3": "HINCHEY_II",  # pelvic abscess ≥3cm, no peritonitis
    },
}
