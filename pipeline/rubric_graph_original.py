"""rubric_graph_original.py

Programmatic representation of 5 clinical diagnosis flowcharts for abdominal pain triage.
This is the **original / strict-guideline version** — cholecystitis uses TG18 verbatim:
  - Group A AND Group B both required before proceeding to imaging (strict A AND B gate)
  - Group B criteria: fever >38°C · elevated CRP · leukocytosis (no HPI-narrative supplement)

Compare with rubric_graph.py, which relaxes the cholecystitis gate to A OR B and adds
fever_reported_in_hpi to Group B to improve coverage on MIMIC-IV data.

Sources:
  - WSES Jerusalem Guidelines 2020          (Appendicitis)
  - Tokyo Guidelines TG18 (2018)            (Cholecystitis)
  - AAFP / AGA Clinical Practice Update 2021 (Diverticulitis)
  - Revised Atlanta Classification 2012 / AAFP (Pancreatitis)

Each RubricGraph contains:
  - nodes : dict[id -> RubricNode]   — clinical states in the flowchart
  - edges : list[RubricEdge]         — condition-gated directed transitions
  - root  : str                      — starting node id

Edge conditions are pure functions  (dict -> bool)  evaluated against a
patient "feature dict" whose keys are defined in feature_schema.py.
An edge is traversable when its condition returns True AND all required_tests
of the source node are present in features["tests_done"].

Usage example
-------------
from rubric_graph import ALL_GRAPHS, alvarado_score, bisap_score
graph = ALL_GRAPHS["pancreatitis"]
for edge in graph.edges:
    if edge.source == graph.root:
        can_traverse = edge.condition(patient_features)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Unconditional-edge sentinel  无条件边哨兵
# ---------------------------------------------------------------------------

def _always(_: dict) -> bool:
    """Sentinel for unconditional / always-traversable edges.

    Using a named function (rather than an anonymous lambda) lets the traversal
    engine detect always-edges via identity check:
        edge.condition is ALWAYS_EDGE_CONDITION
    """
    return True


# Public export so traversal_engine and diagnosis_distribution can import it.
ALWAYS_EDGE_CONDITION: Callable[[dict], bool] = _always


# ---------------------------------------------------------------------------
# Core data structures  核心数据结构
# ---------------------------------------------------------------------------

@dataclass
class RubricNode:
    """A node (clinical state) in the diagnostic flowchart."""

    id: str
    label: str
    # "start" | "assessment" | "decision" |
    # "terminal_confirmed" | "terminal_excluded" | "terminal_low_risk" | "routing"
    node_type: str
    # 到达/评估此节点需要先完成的检查（未完成时 → 状态标记为 pending）
    required_tests: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class RubricEdge:
    """A condition-gated directed edge between two nodes.

    condition: callable (feature_dict -> bool)
               Returns True when the transition is clinically valid.
               Defaults to ALWAYS_EDGE_CONDITION (unconditional / "always" edge).
               Use `edge.condition is ALWAYS_EDGE_CONDITION` to detect always-edges.
    """

    source: str
    target: str
    label: str
    # Default: unconditional (sentinel function, detectable by identity check)
    condition: Callable[[dict], bool] = _always


@dataclass
class RubricGraph:
    """Complete rubric graph for one disease (or the triage dispatcher)."""

    disease: str          # e.g. "cholecystitis" | "triage"
    nodes: dict[str, RubricNode]
    edges: list[RubricEdge]
    root: str             # id of the starting node


# ---------------------------------------------------------------------------
# Shared helper functions  公共辅助函数
# ---------------------------------------------------------------------------

def alvarado_score(f: dict) -> int:
    """
    Alvarado Score (0–10) for appendicitis risk stratification.
    阑尾炎风险分层评分（Alvarado评分）

    Score breakdown:
      ① Pain migration to RLQ     +1
      ② Anorexia                  +1
      ③ Nausea / Vomiting         +1
      ④ RLQ tenderness            +2
      ⑤ Rebound tenderness        +1
      ⑥ Temperature ≥ 37.3 °C    +1
      ⑦ WBC > 10,000              +2
      ⑧ Left shift (bands)        +1
    """
    return (
        int(bool(f.get("pain_migration_to_RLQ"))) * 1
        + int(bool(f.get("anorexia"))) * 1
        + int(bool(f.get("nausea_vomiting"))) * 1
        + int(bool(f.get("RLQ_tenderness"))) * 2
        + int(bool(f.get("rebound_tenderness"))) * 1
        + int(bool(f.get("fever_temp_ge_37_3"))) * 1
        + int(bool(f.get("WBC_gt_10k"))) * 2
        + int(bool(f.get("left_shift"))) * 1
    )


def bisap_score(f: dict) -> int:
    """
    BISAP Score (0–5) for pancreatitis severity prediction.
    胰腺炎重症预测评分（BISAP评分）

    Score breakdown (each item = 1 point):
      ① BUN > 25 mg/dL
      ② Impaired mental status (Glasgow < 15)
      ③ SIRS ≥ 2 criteria
      ④ Age > 60 years
      ⑤ Pleural effusion on imaging
    """
    return (
        int(bool(f.get("BUN_gt_25"))) * 1
        + int(bool(f.get("impaired_mental_status"))) * 1
        + int(bool(f.get("SIRS_criteria_ge_2"))) * 1
        + int(bool(f.get("age_gt_60"))) * 1
        + int(bool(f.get("pleural_effusion_on_imaging"))) * 1
    )


def _revised_atlanta_criteria_count(f: dict) -> int:
    """
    Count of Revised Atlanta diagnostic criteria met (0–3).
    满足的Revised Atlanta诊断标准条数

    ① Upper abdominal pain (epigastric or radiating to back)
    ② Lipase or Amylase ≥ 3× ULN
    ③ Characteristic imaging findings (CT / MRI / US)
    """
    return (
        int(bool(f.get("epigastric_radiating_to_back") or f.get("pain_location") == "Epigastric"))
        + int(bool(f.get("lipase_ge_3xULN")))
        + int(bool(f.get("CT_pancreatitis_positive") or f.get("US_gallstones")))
    )


# ---- Test-completion helpers  检查完成状态辅助函数 ----

def _done(f: dict, *tests: str) -> bool:
    """Return True if ALL specified tests are in features['tests_done']."""
    done_set = set(f.get("tests_done", []))
    return all(t in done_set for t in tests)


def _any_done(f: dict, *tests: str) -> bool:
    """Return True if AT LEAST ONE of the specified tests is in features['tests_done']."""
    done_set = set(f.get("tests_done", []))
    return any(t in done_set for t in tests)


# ---- TG18 sub-conditions for cholecystitis  胆囊炎TG18标准辅助函数 ----

def _tg18_group_a(f: dict) -> bool:
    """
    TG18 Group A (local inflammation signs): ≥ 1 criterion present.
    TG18 A组局部炎症体征（满足≥1项）

    Extended: RUQ_tenderness added as a proxy for Murphy's sign when the
    clinician documents "TTP RUQ" / "tender RUQ" rather than naming the sign.
    """
    return (
        f.get("murphys_sign", False)
        or f.get("RUQ_tenderness", False)
        or f.get("RUQ_mass", False)
    )


def _tg18_group_b(f: dict) -> bool:
    """
    TG18 Group B (systemic inflammation signs): ≥ 1 criterion present.
    TG18 B组全身炎症体征（满足≥1项）— strict TG18 criteria, no HPI-narrative supplement.

    Original TG18 Group B:
      ① Fever > 38 °C  (admission vitals)
      ② Elevated CRP
      ③ Leukocytosis
    """
    return (
        f.get("fever_temp_ge_38", False)
        or f.get("CRP_elevated", False)
        or f.get("WBC_gt_10k", False)
    )


def _tg18_suspected(f: dict) -> bool:
    """
    TG18 suspected diagnosis: Group A ≥1 AND Group B ≥1.
    TG18疑似诊断（A+B均满足≥1项）
    """
    return _tg18_group_a(f) and _tg18_group_b(f)


def _tg18_us_positive(f: dict) -> bool:
    """
    TG18 Group C (positive US imaging for cholecystitis).
    TG18 C组超声阳性
    """
    return (
        f.get("US_gallstones", False)
        or f.get("US_GB_wall_thickening", False)
        or f.get("US_pericholecystic_fluid", False)
        or f.get("US_sonographic_murphys", False)
    )


def _tg18_organ_dysfunction(f: dict) -> bool:
    """
    TG18 Grade III organ dysfunction: any of the six organ systems affected.
    TG18重度器官功能障碍判断（涵盖六大系统）

    Sources checked (in priority order):
      1. has_organ_dysfunction  — CT/LLM-confirmed imaging/clinical evidence
                                   OR lab-derived (creatinine/bilirubin ≥ 2.0)
                                   set by algo_extractor or imaging LLM
      2. impaired_mental_status — neurological dysfunction (GCS ≤ 13/confusion)
                                   extracted from PE at Step 0; satisfies the
                                   TG18 Grade III neurological criterion directly
    """
    return (
        f.get("has_organ_dysfunction", False)
        or f.get("impaired_mental_status", False)
    )


def _tg18_grade_ii_local(f: dict) -> bool:
    """
    TG18 Grade II local complication criteria (no organ dysfunction but ≥1):
    TG18中度（无器官功能障碍但存在以下任一局部并发症）
      - WBC > 18,000
      - Palpable tender RUQ mass
      - Symptom duration > 72h
      - Gangrenous or emphysematous GB on imaging
    """
    return (
        f.get("WBC_gt_18k", False)
        or f.get("RUQ_mass", False)
        or f.get("symptom_duration_over_72h", False)
        or f.get("CT_GB_severe_findings", False)
    )


# ---------------------------------------------------------------------------
# Graph 0: Main Triage  主分流图
# ---------------------------------------------------------------------------

def _build_triage_graph() -> RubricGraph:
    nodes = {
        "START": RubricNode(
            "START",
            "Abdominal Pain",
            "start",
            required_tests=[],
            description="Patient presents with abdominal pain / 腹痛就诊",
        ),
        "HPI_PE": RubricNode(
            "HPI_PE",
            "HPI + Physical Examination",
            "assessment",
            required_tests=[],
            description=(
                "Pain location / quality / onset / duration · fever · nausea · vomiting\n"
                "PMH · alcohol history · menstrual history (females)\n"
                "疼痛位置/性质/时间 · 发热 · 恶心呕吐 · 既往史 · 饮酒史 · 月经史（女）"
            ),
        ),
        "BASIC_LABS": RubricNode(
            "BASIC_LABS",
            "Basic Labs",
            "assessment",
            required_tests=["Lab_Panel"],
            description=(
                "CBC · CMP · Lipase · UA · β-hCG (reproductive-age females)\n"
                "基础实验室检查"
            ),
        ),
        "ROUTE_APPENDICITIS": RubricNode(
            "ROUTE_APPENDICITIS",
            "→ Appendicitis Sub-Rubric",
            "routing",
            description=(
                "RLQ pain + migration + anorexia/nausea + leukocytosis\n"
                "RLQ痛 + 转移痛 + 厌食恶心 + WBC升高"
            ),
        ),
        "ROUTE_CHOLECYSTITIS": RubricNode(
            "ROUTE_CHOLECYSTITIS",
            "→ Cholecystitis Sub-Rubric",
            "routing",
            description=(
                "RUQ pain + Murphy's sign + postprandial aggravation + WBC/CRP elevated\n"
                "RUQ痛 + Murphy征 + 餐后加重 + WBC/CRP升高"
            ),
        ),
        "ROUTE_DIVERTICULITIS": RubricNode(
            "ROUTE_DIVERTICULITIS",
            "→ Diverticulitis Sub-Rubric",
            "routing",
            description=(
                "LLQ pain + fever + bowel habit change + prior diverticular disease + WBC/CRP↑\n"
                "LLQ痛 + 发热 + 便习改变 + 既往憩室病史 + WBC/CRP升高"
            ),
        ),
        "ROUTE_PANCREATITIS": RubricNode(
            "ROUTE_PANCREATITIS",
            "→ Pancreatitis Sub-Rubric",
            "routing",
            description=(
                "Epigastric pain radiating to back + Lipase ≥3×ULN + alcohol/gallstone history\n"
                "上腹痛放射至背 + Lipase≥3×ULN + 饮酒史或胆石症史"
            ),
        ),
        "EXTENDED_DIFFERENTIAL": RubricNode(
            "EXTENDED_DIFFERENTIAL",
            "Extended Differential Workup",
            "assessment",
            description=(
                "Atypical or unclear presentation — further workup required\n"
                "上述特征均不典型，进一步检查鉴别诊断"
            ),
        ),
    }

    edges = [
        RubricEdge("START", "HPI_PE", "always"),
        RubricEdge("HPI_PE", "BASIC_LABS", "always"),

        # → Appendicitis: RLQ pain + ≥2 of {migration, anorexia/nausea, leukocytosis}
        # 阑尾炎分流：RLQ疼痛 + 至少满足2项附加特征
        RubricEdge(
            "BASIC_LABS", "ROUTE_APPENDICITIS",
            "RLQ pain + migration/anorexia/nausea + leukocytosis",
            condition=lambda f: (
                _done(f, "Lab_Panel")
                and f.get("pain_location") == "RLQ"
                and (
                    int(bool(f.get("pain_migration_to_RLQ")))
                    + int(bool(f.get("anorexia") or f.get("nausea_vomiting")))
                    + int(bool(f.get("WBC_gt_10k")))
                ) >= 2
            ),
        ),

        # → Cholecystitis: RUQ pain + (Murphy's sign OR RUQ tenderness) + WBC or CRP elevated
        # 胆囊炎分流：RUQ疼痛 + （Murphy征或RUQ压痛）+ 炎症指标升高
        # RUQ_tenderness is accepted as a proxy for Murphy's sign: clinicians
        # frequently document "TTP RUQ" without explicitly naming the sign.
        RubricEdge(
            "BASIC_LABS", "ROUTE_CHOLECYSTITIS",
            "RUQ pain + Murphy's sign or RUQ tenderness + WBC/CRP elevated",
            condition=lambda f: (
                _done(f, "Lab_Panel")
                and f.get("pain_location") == "RUQ"
                and (f.get("murphys_sign", False) or f.get("RUQ_tenderness", False))
                and (f.get("WBC_gt_10k", False) or f.get("CRP_elevated", False))
                and not f.get("lipase_ge_3xULN", False)   # lipase↑ → prefer pancreatitis route
            ),
        ),

        # → Diverticulitis: LLQ pain + fever + bowel changes or prior diverticular disease
        # 憩室炎分流：LLQ疼痛 + 发热 + 便习改变或既往憩室病史
        RubricEdge(
            "BASIC_LABS", "ROUTE_DIVERTICULITIS",
            "LLQ pain + fever + bowel changes / prior diverticular disease",
            condition=lambda f: (
                _done(f, "Lab_Panel")
                and f.get("pain_location") == "LLQ"
                and f.get("fever_temp_ge_38", False)
                and (
                    f.get("bowel_habit_change", False)
                    or f.get("prior_diverticular_disease", False)
                )
            ),
        ),

        # → Pancreatitis: lipase ≥3×ULN + epigastric or RUQ pain
        # 胰腺炎分流：Lipase≥3×ULN（高度特异性）+ 上腹/右上腹疼痛
        # RUQ is included because gallstone pancreatitis frequently presents
        # with RUQ pain (the stone obstructs the common bile duct) rather than
        # classic epigastric pain.  Lipase ≥3×ULN is the dominant gate.
        RubricEdge(
            "BASIC_LABS", "ROUTE_PANCREATITIS",
            "Lipase ≥3×ULN + epigastric or RUQ pain",
            condition=lambda f: (
                _done(f, "Lab_Panel")
                and f.get("lipase_ge_3xULN", False)
                and (
                    f.get("epigastric_radiating_to_back", False)
                    or f.get("pain_location") in ("Epigastric", "RUQ")
                )
            ),
        ),

        # → Extended differential: presentation does not fit any above pattern
        # 不典型：不符合任何主要分流特征
        RubricEdge(
            "BASIC_LABS", "EXTENDED_DIFFERENTIAL",
            "Atypical / unclear presentation",
            condition=lambda f: (
                _done(f, "Lab_Panel")
                and f.get("pain_location") not in ("RLQ", "RUQ", "LLQ", "Epigastric")
                and not f.get("lipase_ge_3xULN", False)
                and not f.get("murphys_sign", False)
                and not f.get("RUQ_tenderness", False)
            ),
        ),
    ]

    return RubricGraph(disease="triage", nodes=nodes, edges=edges, root="START")


# ---------------------------------------------------------------------------
# Graph 1: Appendicitis Sub-Rubric  阑尾炎诊断流程
# Source: WSES Jerusalem Guidelines 2020 · Alvarado Score
# ---------------------------------------------------------------------------

def _build_appendicitis_graph() -> RubricGraph:
    nodes = {
        "SUSPECTED": RubricNode(
            "SUSPECTED",
            "Suspected Appendicitis",
            "start",
            description="Routed from main triage / 来自主分流",
        ),
        "ALVARADO": RubricNode(
            "ALVARADO",
            "Alvarado Score Calculation",
            "assessment",
            required_tests=["Lab_Panel"],
            description=(
                "Requires: CBC with differential + physical examination\n"
                "Score 0–10:\n"
                "  ① Migration of pain to RLQ  +1\n"
                "  ② Anorexia                  +1\n"
                "  ③ Nausea / Vomiting         +1\n"
                "  ④ RLQ Tenderness            +2\n"
                "  ⑤ Rebound Tenderness        +1\n"
                "  ⑥ Temp ≥ 37.3 °C           +1\n"
                "  ⑦ WBC > 10,000              +2\n"
                "  ⑧ Left Shift (bands)        +1"
            ),
        ),
        "OBSERVE": RubricNode(
            "OBSERVE",
            "Observe / Discharge with Follow-up",
            "assessment",
            description="Low risk (Alvarado ≤3): outpatient monitoring acceptable / 低风险，观察或出院随访",
        ),
        "EXCLUDED_LOW": RubricNode(
            "EXCLUDED_LOW",
            "Appendicitis Low Risk (Outpatient Monitoring)",
            "terminal_low_risk",
            description=(
                "Alvarado ≤3 — low pre-test probability; outpatient monitoring acceptable.\n"
                "NOT a definitive exclusion: ~10% PPV at this score threshold.\n"
                "低风险（Alvarado≤3）：门诊随访即可，非影像确认排除"
            ),
        ),
        "US_ABDOMEN": RubricNode(
            "US_ABDOMEN",
            "Abdominal Ultrasound",
            "assessment",
            required_tests=["Ultrasound_Abdomen"],
            description=(
                "Preferred first-line for: children, pregnant women, young females\n"
                "中风险4–6分首选超声（儿童/孕妇/年轻女性优先）"
            ),
        ),
        "US_FINDINGS": RubricNode(
            "US_FINDINGS",
            "US Findings — Appendicitis Confirmed by Ultrasound",
            "decision",
            required_tests=[],  # US already done; arrived via US positive edge
            description=(
                "Appendicitis confirmed on US; assess for complications visible on US\n"
                "CT not required when US is diagnostic (WSES 2020)\n"
                "超声确认阑尾炎，评估超声下并发症征象；超声确诊时无需CT（WSES 2020）"
            ),
        ),
        "CT_ABD_MID": RubricNode(
            "CT_ABD_MID",
            "CT Abdomen/Pelvis with IV Contrast (after non-diagnostic US)",
            "assessment",
            required_tests=[],   # CT_FINDINGS is the sole CT gate; this node is semantic only
            description=(
                "After non-diagnostic US in intermediate-risk patients\n"
                "Sensitivity >95% for appendicitis / 超声不确定后进行CT（敏感性>95%）"
            ),
        ),
        "CT_ABD_HIGH": RubricNode(
            "CT_ABD_HIGH",
            "CT Abdomen/Pelvis or Direct Surgical Consult (High Risk)",
            "assessment",
            required_tests=[],   # CT_FINDINGS is the sole CT gate; this node is semantic only
            description=(
                "High risk (Alvarado 7–10)\n"
                "Males < 40y may proceed directly to surgical consult without CT\n"
                "高风险直接CT或手术会诊（<40岁男性可直接手术会诊）"
            ),
        ),
        "CT_FINDINGS": RubricNode(
            "CT_FINDINGS",
            "CT Findings Evaluation",
            "decision",
            required_tests=["CT_Abdomen"],
            description="Interpret CT for appendicitis confirmation / CT结果判断",
        ),
        "EXCLUDED_CT": RubricNode(
            "EXCLUDED_CT",
            "Appendicitis Excluded / Alternative Diagnosis",
            "terminal_excluded",
            description="CT negative for appendicitis / CT阴性，排除阑尾炎",
        ),
        "UNCOMPLICATED": RubricNode(
            "UNCOMPLICATED",
            "Uncomplicated Appendicitis",
            "terminal_confirmed",
            description=(
                "Appendicitis confirmed (US or CT); no perforation / abscess / phlegmon\n"
                "→ Surgical consult: laparoscopic appendectomy vs antibiotics\n"
                "非复杂性阑尾炎（超声或CT确认，无穿孔/脓肿/蜂窝织炎）— 手术 vs 抗生素治疗"
            ),
        ),
        "COMPLICATED": RubricNode(
            "COMPLICATED",
            "Complicated Appendicitis",
            "terminal_confirmed",
            description=(
                "Appendicitis confirmed (US or CT) with perforation / abscess / phlegmon\n"
                "→ Surgical consult required\n"
                "复杂性阑尾炎（超声或CT见穿孔/脓肿/蜂窝织炎）— 外科会诊"
            ),
        ),
    }

    edges = [
        RubricEdge("SUSPECTED", "ALVARADO", "always"),

        # Low risk: Alvarado ≤ 3  低风险
        RubricEdge(
            "ALVARADO", "OBSERVE",
            "Score ≤3 — Low Risk",
            condition=lambda f: _done(f, "Lab_Panel") and alvarado_score(f) <= 3,
        ),
        RubricEdge("OBSERVE", "EXCLUDED_LOW", "always"),

        # Intermediate risk: Alvarado 4–6 → Ultrasound  中风险
        RubricEdge(
            "ALVARADO", "US_ABDOMEN",
            "Score 4–6 — Intermediate Risk → Ultrasound",
            condition=lambda f: _done(f, "Lab_Panel") and 4 <= alvarado_score(f) <= 6,
        ),

        # US positive: appendix visualized and inflamed → US_FINDINGS (no CT needed)
        # 超声阳性 → 超声结果判断（WSES 2020：超声确诊时无需CT）
        RubricEdge(
            "US_ABDOMEN", "US_FINDINGS",
            "US positive — appendix visualized, inflamed",
            condition=lambda f: (
                _done(f, "Ultrasound_Abdomen") and f.get("US_appendix_inflamed", False)
            ),
        ),

        # US_FINDINGS → UNCOMPLICATED: no perforation/abscess visible on US
        # 超声未见穿孔/脓肿 → 非复杂性阑尾炎，直接手术会诊
        RubricEdge(
            "US_FINDINGS", "UNCOMPLICATED",
            "US positive — no perforation/abscess on US",
            condition=lambda f: not f.get("US_perforation_abscess", False),
        ),

        # US_FINDINGS → COMPLICATED: perforation/abscess visible on US
        # 超声见穿孔/脓肿 → 复杂性阑尾炎
        RubricEdge(
            "US_FINDINGS", "COMPLICATED",
            "US positive — perforation/abscess on US",
            condition=lambda f: f.get("US_perforation_abscess", False),
        ),

        # US negative / non-diagnostic → CT  超声阴性/不确定 → CT
        RubricEdge(
            "US_ABDOMEN", "CT_ABD_MID",
            "US negative or non-diagnostic",
            condition=lambda f: (
                _done(f, "Ultrasound_Abdomen") and not f.get("US_appendix_inflamed", False)
            ),
        ),
        RubricEdge("CT_ABD_MID", "CT_FINDINGS", "always"),

        # High risk: Alvarado 7–10 → CT  高风险
        RubricEdge(
            "ALVARADO", "CT_ABD_HIGH",
            "Score 7–10 — High Risk → CT or Surgical Consult",
            condition=lambda f: _done(f, "Lab_Panel") and alvarado_score(f) >= 7,
        ),
        RubricEdge("CT_ABD_HIGH", "CT_FINDINGS", "always"),

        # CT interpretation  CT结果判断
        RubricEdge(
            "CT_FINDINGS", "EXCLUDED_CT",
            "CT negative",
            condition=lambda f: (
                _done(f, "CT_Abdomen") and not f.get("CT_appendicitis_positive", False)
            ),
        ),
        RubricEdge(
            "CT_FINDINGS", "UNCOMPLICATED",
            "CT positive — no perforation / abscess",
            condition=lambda f: (
                _done(f, "CT_Abdomen")
                and f.get("CT_appendicitis_positive", False)
                and not f.get("CT_perforation_abscess", False)
            ),
        ),
        RubricEdge(
            "CT_FINDINGS", "COMPLICATED",
            "CT positive — perforation / abscess / phlegmon",
            condition=lambda f: (
                _done(f, "CT_Abdomen")
                and f.get("CT_appendicitis_positive", False)
                and f.get("CT_perforation_abscess", False)
            ),
        ),
    ]

    return RubricGraph(
        disease="appendicitis", nodes=nodes, edges=edges, root="SUSPECTED"
    )


# ---------------------------------------------------------------------------
# Graph 2: Cholecystitis Sub-Rubric  胆囊炎诊断流程
# Source: Tokyo Guidelines 2018 (TG18) — Diagnosis criteria + Severity grading
# ---------------------------------------------------------------------------

def _build_cholecystitis_graph() -> RubricGraph:
    nodes = {
        "SUSPECTED": RubricNode(
            "SUSPECTED",
            "Suspected Cholecystitis",
            "start",
            description="Routed from main triage / 来自主分流",
        ),
        # TG18 Group A and B run in parallel from "SUSPECTED"
        # TG18 A组和B组从"SUSPECTED"并行开始评估
        "GROUP_A": RubricNode(
            "GROUP_A",
            "TG18 Group A — Local Inflammation Signs",
            "assessment",
            required_tests=[],  # 体格检查，无需实验室
            description=(
                "Physical examination:\n"
                "  ① Murphy's sign positive\n"
                "  ② RUQ pain / tenderness / palpable mass\n"
                "局部炎症体征（体格检查）"
            ),
        ),
        "GROUP_B": RubricNode(
            "GROUP_B",
            "TG18 Group B — Systemic Inflammation Signs",
            "assessment",
            required_tests=["Lab_Panel"],
            description=(
                "Labs + vitals + HPI narrative:\n"
                "  ① Fever > 38 °C (admission PE vitals)\n"
                "  ② Fever reported in HPI (home/pre-admission fever history)\n"
                "  ③ CRP elevated\n"
                "  ④ WBC elevated (leukocytosis)\n"
                "全身炎症体征（体格检查 + CBC + CRP + HPI发热史）\n"
                "注：MIMIC入院体温常已退热，HPI发热叙述作为补充B组依据"
            ),
        ),
        "AB_DECISION": RubricNode(
            "AB_DECISION",
            "Both Group A AND Group B Met?",
            "decision",
            required_tests=["Lab_Panel"],
            description=(
                "TG18 strict suspected-diagnosis gate (original TG18)\n"
                "疑似诊断门控：A组≥1项 AND B组≥1项（严格TG18原版标准）"
            ),
        ),
        "NOT_SUSPECTED_CLINICAL": RubricNode(
            "NOT_SUSPECTED_CLINICAL",
            "Cholecystitis Not Suspected (Clinical Only)",
            "terminal_low_risk",
            description=(
                "TG18 A AND B not both met — suspected diagnosis not established.\n"
                "Includes: A-only, B-only, or neither. No imaging indicated per TG18.\n"
                "NOT imaging-confirmed exclusion; clinical suspicion insufficient.\n"
                "A+B未同时满足（含仅A、仅B、或均未满足）：TG18原版不进入影像学，非影像排除"
            ),
        ),
        "NOT_SUSPECTED_IMAGING": RubricNode(
            "NOT_SUSPECTED_IMAGING",
            "Cholecystitis Excluded (Imaging Confirmed)",
            "terminal_excluded",
            description=(
                "Additional imaging (HIDA/CT/MRCP) negative — diagnosis excluded.\n"
                "影像学阴性确认排除胆囊炎"
            ),
        ),
        "IMAGING_US": RubricNode(
            "IMAGING_US",
            "Abdominal Ultrasound (TG18 Group C)",
            "assessment",
            required_tests=["Ultrasound_Abdomen"],
            description=(
                "First-line imaging:\n"
                "  - GB wall thickening > 4mm\n"
                "  - Pericholecystic fluid\n"
                "  - Sonographic Murphy's sign\n"
                "  - Gallstones\n"
                "腹部超声首选（胆囊壁增厚/胆囊周围积液/超声Murphy征/胆结石）"
            ),
        ),
        "ADDITIONAL_IMAGING": RubricNode(
            "ADDITIONAL_IMAGING",
            "Additional Imaging (US inconclusive)",
            "assessment",
            # No required_tests: CT, HIDA, or MRCP are all valid alternatives.
            # Edge conditions use _any_done() to check which was performed.
            # 不强制要求特定检查：CT / HIDA / MRCP 均可，由 edge 条件判断
            description=(
                "HIDA scan (functional; preferred for acalculous)\n"
                "OR CT Abdomen (anatomical detail)\n"
                "OR MRCP (if biliary stones suspected)\n"
                "Also check: LFTs · Bilirubin · GGT\n"
                "超声不确定时追加：HIDA扫描 / CT / MRCP + LFTs"
            ),
        ),
        "SEVERITY_GRADING": RubricNode(
            "SEVERITY_GRADING",
            "TG18 Severity Grading",
            "decision",
            description=(
                "Requires confirmed diagnosis (A+B+C)\n"
                "Assess organ dysfunction and local complications\n"
                "确定诊断后进行TG18严重度分级"
            ),
        ),
        "GRADE_I": RubricNode(
            "GRADE_I",
            "Grade I — Mild Cholecystitis",
            "terminal_confirmed",
            description=(
                "No organ dysfunction · mild GB disease\n"
                "轻度：无器官功能障碍，轻度胆囊炎症"
            ),
        ),
        "GRADE_II": RubricNode(
            "GRADE_II",
            "Grade II — Moderate Cholecystitis",
            "terminal_confirmed",
            description=(
                "No organ dysfunction + ≥1 local complication criterion:\n"
                "  - WBC > 18,000\n"
                "  - Palpable tender RUQ mass\n"
                "  - Symptom duration > 72h\n"
                "  - Gangrenous / emphysematous GB on imaging\n"
                "中度：无器官功能障碍但存在局部并发症"
            ),
        ),
        "GRADE_III": RubricNode(
            "GRADE_III",
            "Grade III — Severe Cholecystitis",
            "terminal_confirmed",
            description=(
                "≥1 organ system dysfunction:\n"
                "  cardiovascular · neurological · respiratory\n"
                "  renal · hepatic · hematologic\n"
                "重度：≥1项器官系统功能障碍"
            ),
        ),
    }

    edges = [
        # Both Group A and Group B assessments start immediately
        # A组和B组并行评估（无前提条件）
        RubricEdge("SUSPECTED", "GROUP_A", "always"),
        RubricEdge("SUSPECTED", "GROUP_B", "always"),

        # Group A → AB_DECISION (local signs can be assessed immediately)
        # A组评估完成后进入判断节点
        RubricEdge(
            "GROUP_A", "AB_DECISION",
            "Group A assessment complete",
        ),

        # Group B → AB_DECISION (requires Lab_Panel)
        # B组需要实验室结果
        RubricEdge(
            "GROUP_B", "AB_DECISION",
            "Group B assessment complete (Lab_Panel done)",
            condition=lambda f: _done(f, "Lab_Panel"),
        ),

        # AB_DECISION → Imaging: A ≥1 AND B ≥1 → TG18 suspected diagnosis (strict original)
        # 疑似诊断成立（A+B均满足）→ 进入影像学，TG18原版严格标准
        RubricEdge(
            "AB_DECISION", "IMAGING_US",
            "A ≥1 AND B ≥1 → Suspected — proceed to imaging (TG18 strict)",
            condition=lambda f: _tg18_suspected(f),
        ),

        # AB_DECISION → Not suspected: A AND B not both met (A-only, B-only, or neither)
        # TG18原版：未同时满足A+B → 不进入影像，包含仅A、仅B或均未满足
        RubricEdge(
            "AB_DECISION", "NOT_SUSPECTED_CLINICAL",
            "A AND B not both met → Not suspected (TG18 strict)",
            condition=lambda f: not _tg18_suspected(f),
        ),

        # Ultrasound positive → confirmed (A + B + C) → severity grading
        # 超声阳性 → 确定诊断 → 严重度分级
        RubricEdge(
            "IMAGING_US", "SEVERITY_GRADING",
            "US positive — definite diagnosis (A + B + C)",
            condition=lambda f: (
                _done(f, "Ultrasound_Abdomen") and _tg18_us_positive(f)
            ),
        ),

        # Ultrasound inconclusive / negative → additional imaging
        # 超声不确定/阴性 → 追加影像
        RubricEdge(
            "IMAGING_US", "ADDITIONAL_IMAGING",
            "US inconclusive or negative",
            condition=lambda f: (
                _done(f, "Ultrasound_Abdomen") and not _tg18_us_positive(f)
            ),
        ),

        # Additional imaging positive → severity grading
        # 追加影像阳性（HIDA/CT/MRCP）→ 确定诊断 → 严重度分级
        # CT_cholecystitis_positive is accepted directly so that CT used as
        # primary or secondary modality reaches SEVERITY_GRADING correctly.
        RubricEdge(
            "ADDITIONAL_IMAGING", "SEVERITY_GRADING",
            "Additional imaging positive for cholecystitis",
            condition=lambda f: (
                _any_done(f, "HIDA_Scan", "CT_Abdomen", "MRCP_Abdomen")
                and (
                    f.get("cholecystitis_additional_imaging_positive", False)
                    or f.get("CT_cholecystitis_positive", False)
                )
            ),
        ),

        # Additional imaging negative → imaging-confirmed exclusion
        # 追加影像阴性 → 影像确认排除
        RubricEdge(
            "ADDITIONAL_IMAGING", "NOT_SUSPECTED_IMAGING",
            "Additional imaging negative — diagnosis excluded by imaging",
            condition=lambda f: (
                _any_done(f, "HIDA_Scan", "CT_Abdomen", "MRCP_Abdomen")
                and not f.get("cholecystitis_additional_imaging_positive", False)
                and not f.get("CT_cholecystitis_positive", False)
                and not _tg18_us_positive(f)
            ),
        ),

        # Severity grading: Grade III (organ dysfunction — highest priority)
        # 重度（器官功能障碍）优先判断
        RubricEdge(
            "SEVERITY_GRADING", "GRADE_III",
            "≥1 organ system dysfunction",
            condition=lambda f: _tg18_organ_dysfunction(f),
        ),

        # Severity grading: Grade II (local complications, no organ dysfunction)
        # 中度（局部并发症，无器官功能障碍）
        RubricEdge(
            "SEVERITY_GRADING", "GRADE_II",
            "No organ dysfunction + local complication criteria",
            condition=lambda f: (
                not _tg18_organ_dysfunction(f)
                and _tg18_grade_ii_local(f)
            ),
        ),

        # Severity grading: Grade I (mild)
        # 轻度（无器官功能障碍，无局部并发症）
        RubricEdge(
            "SEVERITY_GRADING", "GRADE_I",
            "No organ dysfunction, no local complications",
            condition=lambda f: (
                not _tg18_organ_dysfunction(f)
                and not _tg18_grade_ii_local(f)
            ),
        ),
    ]

    return RubricGraph(
        disease="cholecystitis", nodes=nodes, edges=edges, root="SUSPECTED"
    )


# ---------------------------------------------------------------------------
# Graph 3: Diverticulitis Sub-Rubric  憩室炎诊断流程
# Source: AAFP Guidelines · AGA Clinical Practice Update 2021 · Hinchey Classification
# ---------------------------------------------------------------------------

def _build_diverticulitis_graph() -> RubricGraph:
    nodes = {
        "SUSPECTED": RubricNode(
            "SUSPECTED",
            "Suspected Diverticulitis",
            "start",
            description="Routed from main triage / 来自主分流",
        ),
        "LABS": RubricNode(
            "LABS",
            "Laboratory Workup",
            "assessment",
            required_tests=["Lab_Panel"],
            description=(
                "CBC (WBC) · BMP · CRP · UA · β-hCG (females)\n"
                "Note: WBC normal in ~45% of patients\n"
                "CRP > 200 mg/L suggests possible perforation\n"
                "实验室检查（WBC在45%患者中可正常；CRP>200提示可能穿孔）"
            ),
        ),
        "CLINICAL_ASSESSMENT": RubricNode(
            "CLINICAL_ASSESSMENT",
            "Clinical Assessment",
            "decision",
            description=(
                "Assess suspicion level and risk factors\n"
                "临床评估：怀疑程度 + 危险因素"
            ),
        ),
        "CLINICAL_DIAGNOSIS": RubricNode(
            "CLINICAL_DIAGNOSIS",
            "Clinical Diagnosis (tentative)",
            "assessment",
            description=(
                "High suspicion: LLQ + fever + CRP↑ + WBC↑, no high-risk features\n"
                "Accuracy only 40–65%; imaging confirmation strongly recommended\n"
                "高度怀疑可临床诊断，但准确率仅40-65%，建议影像确认"
            ),
        ),
        "CT_ABD_PELVIS": RubricNode(
            "CT_ABD_PELVIS",
            "CT Abdomen/Pelvis with IV Contrast",
            "assessment",
            required_tests=["CT_Abdomen"],
            description=(
                "Gold standard: sensitivity 94%, specificity 99%\n"
                "诊断金标准（敏感性94%，特异性99%）"
            ),
        ),
        "CT_NEGATIVE": RubricNode(
            "CT_NEGATIVE",
            "Diverticulitis Excluded / Alternative Diagnosis",
            "terminal_excluded",
            description="CT negative for diverticulitis / CT阴性，排除憩室炎",
        ),
        "UNCOMPLICATED": RubricNode(
            "UNCOMPLICATED",
            "Uncomplicated Diverticulitis",
            "terminal_confirmed",
            description=(
                "Wall thickening + pericolic fat stranding\n"
                "No abscess / perforation / fistula / obstruction\n"
                "非复杂性憩室炎 — 肠壁增厚+结肠周围脂肪浸润，无并发症"
            ),
        ),
        "HINCHEY_GRADING": RubricNode(
            "HINCHEY_GRADING",
            "Hinchey Classification",
            "decision",
            required_tests=["CT_Abdomen"],
            description=(
                "Classify complications by CT findings\n"
                "基于CT表现进行Hinchey并发症分级"
            ),
        ),
        "HINCHEY_IA": RubricNode(
            "HINCHEY_IA",
            "Hinchey Ia — Phlegmon / Pericolic Inflammation",
            "terminal_confirmed",
            description="Treatment: antibiotics / 治疗：抗生素",
        ),
        "HINCHEY_IB": RubricNode(
            "HINCHEY_IB",
            "Hinchey Ib — Localized Abscess < 3cm",
            "terminal_confirmed",
            description="Treatment: antibiotics ± percutaneous drainage / 治疗：抗生素 ± 穿刺引流",
        ),
        "HINCHEY_II": RubricNode(
            "HINCHEY_II",
            "Hinchey II — Distant / Pelvic Abscess ≥ 3cm",
            "terminal_confirmed",
            description="Treatment: interventional drainage + antibiotics / 治疗：介入穿刺引流 + 抗生素",
        ),
        "HINCHEY_III": RubricNode(
            "HINCHEY_III",
            "Hinchey III — Purulent Peritonitis",
            "terminal_confirmed",
            description="Treatment: emergency surgery / 治疗：急诊手术",
        ),
        "HINCHEY_IV": RubricNode(
            "HINCHEY_IV",
            "Hinchey IV — Fecal Peritonitis",
            "terminal_confirmed",
            description="Treatment: emergency surgery / 治疗：急诊手术",
        ),
    }

    edges = [
        RubricEdge("SUSPECTED", "LABS", "always"),
        RubricEdge("LABS", "CLINICAL_ASSESSMENT", "always"),

        # High clinical suspicion → tentative clinical diagnosis (then still do CT)
        # 高度怀疑 → 临床诊断（仍建议CT确认）
        RubricEdge(
            "CLINICAL_ASSESSMENT", "CLINICAL_DIAGNOSIS",
            "High suspicion: LLQ + fever + CRP↑/WBC↑, no high-risk features",
            condition=lambda f: (
                _done(f, "Lab_Panel")
                and f.get("pain_location") == "LLQ"
                and f.get("fever_temp_ge_38", False)
                and (f.get("CRP_elevated", False) or f.get("WBC_gt_10k", False))
                and not f.get("peritoneal_signs", False)
                and not _tg18_organ_dysfunction(f)
                and not f.get("CRP_gt_200", False)
            ),
        ),

        # Uncertain / suspected complications / high-risk → direct CT
        # 诊断不确定 / 怀疑并发症 / 高危因素 → 直接CT
        RubricEdge(
            "CLINICAL_ASSESSMENT", "CT_ABD_PELVIS",
            "Uncertain OR suspected complications OR high-risk features",
            condition=lambda f: (
                _done(f, "Lab_Panel")
                and (
                    f.get("peritoneal_signs", False)
                    or _tg18_organ_dysfunction(f)
                    or f.get("CRP_gt_200", False)
                    or (f.get("pain_location") != "LLQ")
                )
            ),
        ),

        # Clinical diagnosis still routes to CT for imaging confirmation
        # 临床诊断后仍需CT确认（准确率仅40-65%）
        RubricEdge("CLINICAL_DIAGNOSIS", "CT_ABD_PELVIS", "always"),

        # CT results
        RubricEdge(
            "CT_ABD_PELVIS", "CT_NEGATIVE",
            "CT negative — alternative diagnosis",
            condition=lambda f: (
                _done(f, "CT_Abdomen")
                and not f.get("CT_diverticulitis_confirmed", False)
                and not f.get("CT_diverticulitis_complicated", False)
            ),
        ),
        RubricEdge(
            "CT_ABD_PELVIS", "UNCOMPLICATED",
            "Wall thickening + fat stranding; no complications",
            condition=lambda f: (
                _done(f, "CT_Abdomen")
                and f.get("CT_diverticulitis_confirmed", False)
                and not f.get("CT_diverticulitis_complicated", False)
            ),
        ),
        RubricEdge(
            "CT_ABD_PELVIS", "HINCHEY_GRADING",
            "Complications present on CT",
            condition=lambda f: (
                _done(f, "CT_Abdomen")
                and f.get("CT_diverticulitis_complicated", False)
            ),
        ),

        # Hinchey grading (mutually exclusive based on CT findings)
        # Hinchey分级（基于CT表现互斥）
        RubricEdge(
            "HINCHEY_GRADING", "HINCHEY_IV",
            "Fecal peritonitis",
            # 粪性腹膜炎优先级最高，先判断
            condition=lambda f: f.get("CT_fecal_peritonitis", False),
        ),
        RubricEdge(
            "HINCHEY_GRADING", "HINCHEY_III",
            "Purulent peritonitis (no fecal)",
            condition=lambda f: (
                f.get("CT_purulent_peritonitis", False)
                and not f.get("CT_fecal_peritonitis", False)
            ),
        ),
        RubricEdge(
            "HINCHEY_GRADING", "HINCHEY_II",
            "Distant / pelvic abscess ≥ 3cm",
            condition=lambda f: (
                f.get("CT_abscess_ge_3cm", False)
                and not f.get("CT_purulent_peritonitis", False)
                and not f.get("CT_fecal_peritonitis", False)
            ),
        ),
        RubricEdge(
            "HINCHEY_GRADING", "HINCHEY_IB",
            "Localized abscess < 3cm",
            condition=lambda f: (
                f.get("CT_abscess_lt_3cm", False)
                and not f.get("CT_abscess_ge_3cm", False)
                and not f.get("CT_purulent_peritonitis", False)
                and not f.get("CT_fecal_peritonitis", False)
            ),
        ),
        RubricEdge(
            "HINCHEY_GRADING", "HINCHEY_IA",
            "Phlegmon / pericolic inflammation only",
            condition=lambda f: (
                f.get("CT_phlegmon", False)
                and not f.get("CT_abscess_lt_3cm", False)
                and not f.get("CT_abscess_ge_3cm", False)
                and not f.get("CT_purulent_peritonitis", False)
                and not f.get("CT_fecal_peritonitis", False)
            ),
        ),
    ]

    return RubricGraph(
        disease="diverticulitis", nodes=nodes, edges=edges, root="SUSPECTED"
    )


# ---------------------------------------------------------------------------
# Graph 4: Pancreatitis Sub-Rubric  胰腺炎诊断流程
# Source: Revised Atlanta Classification 2012 · AAFP · BISAP Score
# ---------------------------------------------------------------------------

def _build_pancreatitis_graph() -> RubricGraph:
    nodes = {
        "SUSPECTED": RubricNode(
            "SUSPECTED",
            "Suspected Pancreatitis",
            "start",
            description="Routed from main triage / 来自主分流",
        ),
        "DIAGNOSTIC_CRITERIA": RubricNode(
            "DIAGNOSTIC_CRITERIA",
            "Revised Atlanta Diagnostic Criteria",
            "decision",
            required_tests=["Lab_Panel"],
            description=(
                "≥2 of 3 criteria required for diagnosis:\n"
                "  ① Upper abdominal pain (epigastric, may radiate to back)\n"
                "  ② Lipase or Amylase ≥ 3× ULN\n"
                "  ③ Characteristic imaging findings (CT / MRI / US)\n"
                "满足≥2条即可确诊（满足1条时需影像学确认）"
            ),
        ),
        "CT_MRI_DIAGNOSTIC": RubricNode(
            "CT_MRI_DIAGNOSTIC",
            "CT or MRI Abdomen (for diagnostic uncertainty)",
            "assessment",
            required_tests=["CT_Abdomen"],
            description=(
                "Ordered when only 1 criterion is met\n"
                "Purpose: confirm pancreatitis and exclude other diagnoses\n"
                "仅满足1条标准时行CT/MRI排除其他病因"
            ),
        ),
        "EXCLUDED": RubricNode(
            "EXCLUDED",
            "Pancreatitis Excluded — Alternative Diagnosis",
            "terminal_excluded",
            description="Imaging does not support pancreatitis / 影像学不支持胰腺炎",
        ),
        "CONFIRMED": RubricNode(
            "CONFIRMED",
            "Acute Pancreatitis Confirmed",
            "assessment",
            required_tests=["Ultrasound_Abdomen"],
            description=(
                "Admit + IV fluids + pain management\n"
                "Abdominal US: assess for gallstones (biliary etiology is most common)\n"
                "Evaluate etiology: biliary / alcohol / hypertriglyceridemia / other\n"
                "确诊后住院治疗，腹部超声查胆石症（最常见病因）"
            ),
        ),
        "BISAP": RubricNode(
            "BISAP",
            "BISAP Score (within 24h of admission)",
            "assessment",
            required_tests=["Lab_Panel"],
            description=(
                "BISAP Score 0–5 (each item = 1 point):\n"
                "  ① BUN > 25 mg/dL\n"
                "  ② Impaired mental status (Glasgow < 15)\n"
                "  ③ SIRS ≥ 2 criteria (temp/HR/RR/WBC)\n"
                "  ④ Age > 60 years\n"
                "  ⑤ Pleural effusion on imaging\n"
                "入院24h内计算BISAP评分"
            ),
        ),
        "ATLANTA_LOW": RubricNode(
            "ATLANTA_LOW",
            "Revised Atlanta Assessment — Low Risk (BISAP 0–2)",
            "assessment",
            description=(
                "Predicted mortality < 2%\n"
                "Assess organ failure using Modified Marshall Score\n"
                "低风险（死亡率<2%），评估器官功能障碍"
            ),
        ),
        "ATLANTA_HIGH": RubricNode(
            "ATLANTA_HIGH",
            "Revised Atlanta Assessment — High Risk (BISAP ≥3)",
            "assessment",
            required_tests=[],
            description=(
                "Predicted mortality > 15%\n"
                "ICU monitoring\n"
                "Contrast-enhanced CT recommended at 48–72h for necrosis assessment\n"
                "(CT is a management recommendation, not a prerequisite for organ-failure assessment)\n"
                "高风险（死亡率>15%），ICU监护；CT推荐在48-72h后评估坏死，但不阻断器官衰竭评估"
            ),
        ),
        "ORGAN_FAILURE_ASSESS": RubricNode(
            "ORGAN_FAILURE_ASSESS",
            "Organ Failure Assessment (Modified Marshall Score)",
            "decision",
            description=(
                "Evaluate: cardiovascular, respiratory, renal organ systems\n"
                "Classify as: no failure / transient (<48h) / persistent (>48h)\n"
                "评估器官衰竭：无/短暂性(<48h)/持续性(>48h)"
            ),
        ),
        "MILD_AP": RubricNode(
            "MILD_AP",
            "Mild Acute Pancreatitis",
            "terminal_confirmed",
            description=(
                "No organ failure · no local complications · no systemic complications\n"
                "Typically resolves within 72h\n"
                "轻度：无器官衰竭，无局部/全身并发症，通常72h内自行恢复"
            ),
        ),
        "MODERATELY_SEVERE_AP": RubricNode(
            "MODERATELY_SEVERE_AP",
            "Moderately Severe Acute Pancreatitis",
            "terminal_confirmed",
            description=(
                "Transient organ failure < 48h (self-resolving)\n"
                "AND/OR local complications (peripancreatic fluid / necrosis)\n"
                "中重度：短暂性器官衰竭<48h 和/或 局部并发症（积液/坏死）"
            ),
        ),
        "CT_CTSI": RubricNode(
            "CT_CTSI",
            "Contrast-Enhanced CT Abdomen/Pelvis — CTSI Scoring (at 48–72h)",
            "assessment",
            required_tests=["CT_Abdomen"],
            description=(
                "Modified CT Severity Index (CTSI) 0–10:\n"
                "  Pancreatic inflammation + necrosis + extrapancreatic complications\n"
                "CT严重度指数（CTSI）0-10分：评估炎症范围+坏死程度+胰外并发症"
            ),
        ),
        "SEVERE_AP": RubricNode(
            "SEVERE_AP",
            "Severe Acute Pancreatitis",
            "terminal_confirmed",
            description=(
                "Persistent organ failure > 48h\n"
                "ICU management · infected necrosis debridement / drainage\n"
                "重度：持续性器官衰竭>48h，ICU管理，处理感染性坏死"
            ),
        ),
    }

    edges = [
        RubricEdge("SUSPECTED", "DIAGNOSTIC_CRITERIA", "always"),

        # Only 1 criterion met → CT/MRI for confirmation
        # 仅满足1条标准 → 影像学确认
        RubricEdge(
            "DIAGNOSTIC_CRITERIA", "CT_MRI_DIAGNOSTIC",
            "Only 1 criterion met — imaging required for confirmation",
            condition=lambda f: (
                _done(f, "Lab_Panel")
                and _revised_atlanta_criteria_count(f) == 1
            ),
        ),

        # ≥2 criteria met → confirmed diagnosis directly
        # 满足≥2条 → 直接确诊
        RubricEdge(
            "DIAGNOSTIC_CRITERIA", "CONFIRMED",
            "≥2 criteria met — Acute Pancreatitis confirmed",
            condition=lambda f: (
                _done(f, "Lab_Panel")
                and _revised_atlanta_criteria_count(f) >= 2
            ),
        ),

        # CT/MRI confirms pancreatitis
        # CT/MRI支持胰腺炎诊断
        RubricEdge(
            "CT_MRI_DIAGNOSTIC", "CONFIRMED",
            "Imaging confirms pancreatitis",
            condition=lambda f: (
                _done(f, "CT_Abdomen") and f.get("CT_pancreatitis_positive", False)
            ),
        ),

        # CT/MRI does not support pancreatitis
        # CT/MRI不支持胰腺炎
        RubricEdge(
            "CT_MRI_DIAGNOSTIC", "EXCLUDED",
            "Imaging does not support pancreatitis",
            condition=lambda f: (
                _done(f, "CT_Abdomen") and not f.get("CT_pancreatitis_positive", False)
            ),
        ),

        RubricEdge("CONFIRMED", "BISAP", "always"),

        # BISAP stratification
        # BISAP风险分层
        RubricEdge(
            "BISAP", "ATLANTA_LOW",
            "BISAP 0–2 — Low Risk (mortality <2%)",
            condition=lambda f: bisap_score(f) <= 2,
        ),
        RubricEdge(
            "BISAP", "ATLANTA_HIGH",
            "BISAP ≥3 — High Risk (mortality >15%)",
            condition=lambda f: bisap_score(f) >= 3,
        ),

        RubricEdge("ATLANTA_LOW", "ORGAN_FAILURE_ASSESS", "always"),
        RubricEdge("ATLANTA_HIGH", "ORGAN_FAILURE_ASSESS", "always"),

        # Organ failure assessment → severity classification
        # 器官功能障碍评估 → 严重度分级

        # Mild: no organ failure, no complications
        # 轻度：无器官衰竭，无并发症
        RubricEdge(
            "ORGAN_FAILURE_ASSESS", "MILD_AP",
            "No organ failure, no local/systemic complications",
            condition=lambda f: (
                not f.get("organ_failure_persistent", False)
                and not f.get("organ_failure_transient", False)
                and not f.get("local_complications_pancreatitis", False)
            ),
        ),

        # Moderately severe: transient failure OR local complications
        # 中重度：短暂性器官衰竭 或 局部并发症
        RubricEdge(
            "ORGAN_FAILURE_ASSESS", "MODERATELY_SEVERE_AP",
            "Transient organ failure <48h AND/OR local complications",
            condition=lambda f: (
                not f.get("organ_failure_persistent", False)
                and (
                    f.get("organ_failure_transient", False)
                    or f.get("local_complications_pancreatitis", False)
                )
            ),
        ),

        # Severe: persistent organ failure → CTSI CT
        # 重度：持续性器官衰竭 → CTSI评分CT
        RubricEdge(
            "ORGAN_FAILURE_ASSESS", "CT_CTSI",
            "Persistent organ failure >48h → CT for CTSI scoring",
            condition=lambda f: f.get("organ_failure_persistent", False),
        ),

        RubricEdge("CT_CTSI", "SEVERE_AP", "always"),
    ]

    return RubricGraph(
        disease="pancreatitis", nodes=nodes, edges=edges, root="SUSPECTED"
    )


# ---------------------------------------------------------------------------
# Build and export all graphs  构建并导出所有图
# ---------------------------------------------------------------------------

TRIAGE_GRAPH: RubricGraph = _build_triage_graph()
APPENDICITIS_GRAPH: RubricGraph = _build_appendicitis_graph()
CHOLECYSTITIS_GRAPH: RubricGraph = _build_cholecystitis_graph()
DIVERTICULITIS_GRAPH: RubricGraph = _build_diverticulitis_graph()
PANCREATITIS_GRAPH: RubricGraph = _build_pancreatitis_graph()

# All 5 graphs including triage  全部5张图（含主分流）
# NOTE: This module is the original/strict-TG18 version.
#       Import rubric_graph (without _original suffix) for the MIMIC-adapted relaxed version.
ALL_GRAPHS: dict[str, RubricGraph] = {
    "triage": TRIAGE_GRAPH,
    "appendicitis": APPENDICITIS_GRAPH,
    "cholecystitis": CHOLECYSTITIS_GRAPH,
    "diverticulitis": DIVERTICULITIS_GRAPH,
    "pancreatitis": PANCREATITIS_GRAPH,
}

# Disease-specific graphs only (excludes triage)  仅疾病子图（不含主分流）
DISEASE_GRAPHS: dict[str, RubricGraph] = {
    k: v for k, v in ALL_GRAPHS.items() if k != "triage"
}
