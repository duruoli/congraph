"""feature_schema.py

Canonical feature dictionary schema for the abdominal diagnosis system.
全局特征字典的规范定义，供所有模块共享。

Each key corresponds to one clinical condition checked by the rubric graphs
in rubric_graph.py.  Values are typed as follows:
  - bool  : binary clinical flag  (True / False; default False)
  - str   : categorical value     (default as noted)
  - float : continuous score      (default 0.0)
  - list  : ordered collection    (default [])

Usage
-----
from feature_schema import default_features
features = default_features()
features["pain_location"] = "RUQ"
features["murphys_sign"] = True
features["tests_done"].append("Lab_Panel")
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Feature groups and their clinical sources
# ---------------------------------------------------------------------------

# ── Group 1: Pain characteristics  疼痛特征 (from HPI) ──────────────────────
# pain_location              来自HPI，疼痛主要部位
# pain_migration_to_RLQ      疼痛向右下腹转移（Alvarado ①）
# epigastric_radiating_to_back  上腹痛放射至背部（胰腺炎特征性表现）
# bowel_habit_change         排便习惯改变（憩室炎特征）
# symptom_duration_over_72h  症状持续>72h（TG18 Grade II 标准之一）

# ── Group 2: HPI symptom flags  主诉症状标志 (from HPI / intake) ─────────────
# anorexia                   厌食（Alvarado ②）
# nausea_vomiting            恶心/呕吐（Alvarado ③ / 通用）
# alcohol_history            饮酒史（胰腺炎病因）
# gallstone_history          胆石症既往史（胰腺炎/胆囊炎病因）
# prior_diverticular_disease 既往憩室病史（憩室炎诊断支持）
# fever_reported_in_hpi      HPI叙述中报告发热（TG18 B组补充）
#                            许多胆囊炎患者入院时体温已正常，但HPI描述曾有发热；
#                            keyword（fever/febrile）+ 量化体温≥38°C均可触发

# ── Group 3: Physical examination  体格检查 ──────────────────────────────────
# murphys_sign               Murphy征阳性（TG18 A组 / 主分流胆囊炎路由）
# RUQ_tenderness             右上腹压痛（TG18 A组扩展 / 吸气停顿等paraphrase）
# RLQ_tenderness             右下腹压痛（Alvarado ④）
# rebound_tenderness         反跳痛（Alvarado ⑤）
# RUQ_mass                   右上腹可触及包块（TG18 A组 + Grade II 标准之一）
# fever_temp_ge_37_3         体温≥37.3°C（Alvarado ⑥）
# fever_temp_ge_38           体温>38°C（TG18 B组）
# peritoneal_signs           腹膜刺激征（憩室炎高危因素）
# impaired_mental_status     意识障碍（Glasgow<15，BISAP ②）

# ── Group 4: Labs  实验室检查 ────────────────────────────────────────────────
# WBC_gt_10k                 白细胞>10,000（Alvarado ⑦ / TG18 B组）
# WBC_gt_18k                 白细胞>18,000（TG18 Grade II 标准之一）
# left_shift                 核左移（Alvarado ⑧，bands升高）
# CRP_elevated               CRP升高（TG18 B组）
# CRP_gt_200                 CRP>200 mg/L（憩室炎穿孔提示）
# lipase_ge_3xULN            Lipase≥3倍正常上限（胰腺炎诊断标准 + 主分流路由）
# BUN_gt_25                  BUN>25 mg/dL（BISAP ①）
# bilirubin_elevated         胆红素升高
# LFTs_elevated              肝功能指标升高（ALT/AST/ALP/GGT）
# creatinine_elevated        肌酐升高（肾功能）
# beta_hCG_positive          β-hCG阳性（育龄女性排除异位妊娠）
# SIRS_criteria_ge_2         SIRS满足≥2条（BISAP ③）
#                              体温<36或>38 / 心率>90 / 呼吸>20 / WBC<4k或>12k或Bands>10%

# ── Group 5: Demographics  人口学信息 ─────────────────────────────────────────
# age_gt_60                  年龄>60岁（BISAP ④）
# is_female_reproductive_age 育龄女性（需查β-hCG排除异位妊娠）

# ── Group 6: Ultrasound findings  超声影像 ──────────────────────────────────
# US_appendix_inflamed       超声示阑尾炎症（阑尾炎中风险确认）
# US_gallstones              超声示胆囊结石
# US_GB_wall_thickening      超声示胆囊壁增厚>4mm（TG18 C组）
# US_pericholecystic_fluid   超声示胆囊周围积液（TG18 C组）
# US_sonographic_murphys     超声下Murphy征阳性（TG18 C组）

# ── Group 7: CT findings  CT影像 ────────────────────────────────────────────
# CT_appendicitis_positive   CT确认阑尾炎
# CT_perforation_abscess     CT示穿孔/脓肿/蜂窝织炎（阑尾炎并发症）
# CT_cholecystitis_positive  CT示胆囊炎阳性
# CT_GB_severe_findings      CT示坏疽性/气肿性胆囊炎（TG18 Grade II 标准之一）
# CT_diverticulitis_confirmed  CT确认憩室炎（肠壁增厚+脂肪浸润）
# CT_diverticulitis_complicated  CT示憩室炎并发症（需Hinchey分级）
# CT_phlegmon                CT示蜂窝织炎（Hinchey Ia）
# CT_abscess_lt_3cm          CT示局限脓肿<3cm（Hinchey Ib）
# CT_abscess_ge_3cm          CT示远隔脓肿≥3cm（Hinchey II）
# CT_purulent_peritonitis    CT示化脓性腹膜炎（Hinchey III）
# CT_fecal_peritonitis       CT示粪性腹膜炎（Hinchey IV）
# CT_pancreatitis_positive   CT确认胰腺炎（炎症征象）
# CTSI_score                 CT严重度指数 0–10（Modified CTSI）

# ── Group 8: Other imaging  其他影像 ─────────────────────────────────────────
# cholecystitis_additional_imaging_positive
#                            HIDA/CT/MRCP追加影像阳性（胆囊炎超声不确定时）
# pleural_effusion_on_imaging  影像示胸腔积液（BISAP ⑤）

# ── Group 9: Organ dysfunction  器官功能障碍 ─────────────────────────────────
# has_organ_dysfunction      任一器官系统功能障碍（TG18 Grade III / 憩室炎高危）
# organ_failure_transient    短暂性器官衰竭<48h（胰腺炎中重度标准）
# organ_failure_persistent   持续性器官衰竭>48h（胰腺炎重度标准）
# local_complications_pancreatitis
#                            胰腺炎局部并发症（积液/坏死，中重度标准）

# ── Group 10: Test tracking  检查状态追踪 ────────────────────────────────────
# tests_done                 已完成检查列表（顺序记录）
#                            Valid values: "Lab_Panel" | "Ultrasound_Abdomen" |
#                            "CT_Abdomen" | "HIDA_Scan" | "MRCP_Abdomen" |
#                            "Radiograph_Chest" | "MRI_Abdomen"


def default_features() -> dict:
    """
    Return a feature dict with all keys initialised to their default values.
    返回所有字段初始化为默认值的特征字典。

    Call this at the start of a new patient session, then update fields
    incrementally as observations arrive.
    每次新病人就诊时调用此函数，随着检查结果到来逐步更新字段。
    """
    return {
        # ── Pain characteristics ──
        "pain_location": "Other",          # str: "RLQ"|"RUQ"|"LLQ"|"Epigastric"|"General_Abdomen"|"Other"
        "pain_migration_to_RLQ": False,
        "epigastric_radiating_to_back": False,
        "bowel_habit_change": False,
        "symptom_duration_over_72h": False,

        # ── HPI symptom flags ──
        "anorexia": False,
        "nausea_vomiting": False,
        "alcohol_history": False,
        "gallstone_history": False,
        "prior_diverticular_disease": False,
        "fever_reported_in_hpi": False,

        # ── Physical examination ──
        "murphys_sign": False,
        "RUQ_tenderness": False,
        "RLQ_tenderness": False,
        "rebound_tenderness": False,
        "RUQ_mass": False,
        "fever_temp_ge_37_3": False,
        "fever_temp_ge_38": False,
        "peritoneal_signs": False,
        "impaired_mental_status": False,

        # ── Labs ──
        "WBC_gt_10k": False,
        "WBC_gt_18k": False,
        "left_shift": False,
        "CRP_elevated": False,
        "CRP_gt_200": False,
        "lipase_ge_3xULN": False,
        "BUN_gt_25": False,
        "bilirubin_elevated": False,
        "LFTs_elevated": False,
        "creatinine_elevated": False,
        "beta_hCG_positive": False,
        "SIRS_criteria_ge_2": False,

        # ── Demographics ──
        "age_gt_60": False,
        "is_female_reproductive_age": False,

        # ── Ultrasound findings ──
        "US_appendix_inflamed": False,
        "US_gallstones": False,
        "US_GB_wall_thickening": False,
        "US_pericholecystic_fluid": False,
        "US_sonographic_murphys": False,

        # ── CT findings ──
        "CT_appendicitis_positive": False,
        "CT_perforation_abscess": False,
        "CT_cholecystitis_positive": False,
        "CT_GB_severe_findings": False,
        "CT_diverticulitis_confirmed": False,
        "CT_diverticulitis_complicated": False,
        "CT_phlegmon": False,
        "CT_abscess_lt_3cm": False,
        "CT_abscess_ge_3cm": False,
        "CT_purulent_peritonitis": False,
        "CT_fecal_peritonitis": False,
        "CT_pancreatitis_positive": False,
        "CTSI_score": 0.0,

        # ── Other imaging ──
        "cholecystitis_additional_imaging_positive": False,
        "pleural_effusion_on_imaging": False,

        # ── Organ dysfunction ──
        "has_organ_dysfunction": False,
        "organ_failure_transient": False,
        "organ_failure_persistent": False,
        "local_complications_pancreatitis": False,

        # ── Test tracking ──
        "tests_done": [],   # list[str]; append modality name after each test is completed
    }


# ---------------------------------------------------------------------------
# Allowed values for categorical / list fields
# ---------------------------------------------------------------------------

VALID_PAIN_LOCATIONS: tuple[str, ...] = (
    "RLQ",              # Right Lower Quadrant  右下腹
    "RUQ",              # Right Upper Quadrant  右上腹
    "LLQ",              # Left Lower Quadrant   左下腹
    "LUQ",              # Left Upper Quadrant   左上腹
    "Epigastric",       # Epigastric / 上腹部
    "Periumbilical",    # Periumbilical / 脐周
    "Pelvic",           # Pelvic / 盆腔
    "General_Abdomen",  # Diffuse / 弥漫性腹痛
    "Other",            # Unclear or unspecified / 不明确
)

VALID_TESTS: tuple[str, ...] = (
    "Lab_Panel",            # CBC · CMP · Lipase · UA · CRP · β-hCG
    "Ultrasound_Abdomen",   # 腹部超声
    "CT_Abdomen",           # CT腹盆腔（含或不含IV造影剂）
    "HIDA_Scan",            # 肝胆显像（胆囊功能性评估）
    "MRCP_Abdomen",         # 磁共振胰胆管造影
    "Radiograph_Chest",     # 胸片（胰腺炎BISAP胸腔积液评估）
    "MRI_Abdomen",          # MRI腹部
)
