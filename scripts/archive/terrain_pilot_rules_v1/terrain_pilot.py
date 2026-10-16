#!/usr/bin/env python3
"""Pilot density terrain: facts only -> organ / question -> day-by-day density.

Only FACTS about the patient are deposited (abnormal labs, positive imaging
findings, positive microbiology). Decisions are left out: medications,
orders, consults, transfers and imaging indications. Transfers are drawn as
markers only, for comparison.

Weight of one fact (fixed in advance, never tuned to the doctor's actions):
  - first time this (source, finding) appears for the patient   -> 1.0
  - the same (source, finding) again on a later day             -> 0.5
  - the same finding on the same day, or a copied imaging line  -> 0
A finding that comes back through another source has a new (source, finding)
key and so counts 1.0.

Sea level = mean cumulative density of demographic peers on the same
hospital day: primary-IBD admissions, same sex, age within +/-5 years,
still in hospital on that day.

Outputs (restricted MIMIC-derived, stay under runs/):
  runs/terrain_pilot/facts.parquet, terrain_data.json, terrain.html (zh), terrain_en.html (en)
"""

from __future__ import annotations

import argparse
import warnings
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RAW = Path("data/raw_data/ibd_mimiciv_3_1")
OUT = Path("runs/terrain_pilot")
PILOT = [23605251, 25292961, 29942834]
AGE_CALIPER = 5
MAX_DAY = 14
DAYS = range(-1, MAX_DAY + 1)  # -1 = before admission

ORGANS = {
    "colon": "结肠", "smallbowel": "小肠", "peritoneum": "腹腔", "liver": "肝胆", "pancreas": "胰腺",
    "kidney": "肾 / 泌尿", "lung": "肺", "heart": "心脏 / 血管", "blood": "血液", "systemic": "全身",
}
QUESTIONS = {
    "active": "肠道炎症活动", "gi_infect": "肠道感染", "perf": "穿孔", "abscess": "脓肿 / 瘘",
    "obstruct": "梗阻 / 狭窄", "bleed": "出血 / 贫血", "sepsis": "全身感染", "liver": "肝胆异常",
    "kidney": "肾 / 泌尿问题", "thrombo": "血栓", "nutri": "营养 / 电解质", "cardiac": "心脏问题",
}

# ---- labs: label -> (organ, finding, questions). Unlisted labels are ignored.
# organ = where the test directly measures a state (white cells live in blood); questions = what it usually means.
# Glucose (steroid effect) and drug levels are traces of treatment, so they are left out.
LAB_GROUPS = [
    (r"^(Hemoglobin|Hematocrit|Red Blood Cells)$", "blood", "血红蛋白低", ["bleed"]),
    (r"^(Iron|Ferritin|Transferrin|Iron Binding Capacity, Total|Reticulocyte Count, Automated|Haptoglobin)$",
     "blood", "铁 / 溶血指标", ["bleed"]),
    (r"^Platelet Count$", "blood", "血小板异常", ["bleed"]),
    (r"^(INR\(PT\)|PT|PTT|Fibrinogen, Functional)$", "blood", "凝血异常", ["bleed"]),
    (r"^D-Dimer$", "heart", "D-二聚体高", ["thrombo"]),
    (r"^(C-Reactive Protein|Sedimentation Rate)$", "systemic", "炎症指标高", ["active"]),
    (r"(?i)calprotectin", "colon", "粪钙卫蛋白高", ["active"]),
    (r"^(White Blood Cells|Neutrophils|Absolute Neutrophil Count|Bands|Immature Granulocytes|Metamyelocytes|Myelocytes)$",
     "blood", "白细胞异常", ["sepsis"]),
    (r"^Lactate$", "systemic", "乳酸高", ["sepsis"]),
    (r"^(Creatinine|Urea Nitrogen)$", "kidney", "肾功能异常", ["kidney"]),
    (r"^(RBC|WBC|Protein|Bacteria|Leukocytes|Blood|Nitrite)$", "kidney", "尿检异常", ["kidney"]),  # fluid == Urine
    (r"^(Alanine Aminotransferase \(ALT\)|Asparate Aminotransferase \(AST\)|Alkaline Phosphatase|Bilirubin, Total|Bilirubin, Direct)$",
     "liver", "肝功能异常", ["liver"]),
    (r"^Lipase$", "pancreas", "脂肪酶高", []),
    (r"^(Albumin|Protein, Total|25-OH Vitamin D|Vitamin B12)$", "systemic", "营养指标低", ["nutri"]),
    (r"^(Potassium|Magnesium|Phosphate|Calcium, Total|Sodium|Chloride|Bicarbonate|Anion Gap)$",
     "systemic", "电解质异常", ["nutri"]),
    (r"^(Troponin T|NTproBNP)$", "heart", "心肌指标高", ["cardiac"]),
    (r"^(pO2|pCO2)$", "lung", "血气异常", []),
]

# ---- imaging: finding pattern -> (organ, finding, questions); matched on findings + impression only.
IMG_FINDINGS = [
    (r"colitis|proctitis|loss of (the )?haustra|(colon|colonic|rectal|rectum|sigmoid)[^.]{0,40}(thicken|inflam|hyperenhanc)|"
     r"(thicken|inflam)[^.]{0,40}(colon|rectum|sigmoid)", "colon", "结肠炎症", ["active"]),
    (r"megacolon|(colon|colonic)[^.]{0,30}(dilat|distension|distention)|distension of the colon", "colon", "结肠扩张",
     ["active", "obstruct"]),
    (r"ileitis|enteritis|(ileum|ileal|small bowel|jejun)[^.]{0,40}(thicken|inflam|hyperenhanc)", "smallbowel",
     "小肠炎症", ["active"]),
    (r"small bowel obstruction|\bSBO\b|dilated (loops of )?small bowel|small bowel[^.]{0,30}dilat|air-fluid levels",
     "smallbowel", "小肠梗阻", ["obstruct"]),
    (r"stricture", "smallbowel", "狭窄", ["obstruct"]),
    (r"fistula|sinus tract", "smallbowel", "瘘", ["abscess"]),
    (r"free (intraperitoneal )?air|pneumoperitoneum|perforat|extraluminal (contrast|air|gas)|pneumatosis",
     "peritoneum", "游离气 / 穿孔", ["perf"]),
    (r"abscess|(fluid )?collection", "peritoneum", "脓肿 / 积液", ["abscess"]),
    (r"ascites|free fluid", "peritoneum", "腹水", []),
    (r"steatosis|fatty liver|hepatomegaly|cirrho|biliary (ductal )?dilat|(intrahepatic|bile) duct[^.]{0,30}dilat|"
     r"cholangitis|cholelithiasis|gallstone|cholecystitis|sclerosing", "liver", "肝胆影像异常", ["liver"]),
    (r"pancreatitis|peripancreatic", "pancreas", "胰腺炎表现", []),
    (r"hydronephrosis|nephrolithiasis|renal (stone|calculus|calculi)|pyelonephritis", "kidney", "肾 / 尿路异常",
     ["kidney"]),
    (r"pneumonia|consolidation", "lung", "肺炎 / 实变", ["sepsis"]),
    (r"pleural effusion|atelectasis|pulmonary edema|opacit", "lung", "肺部其他异常", []),
    (r"(pulmonary )?embol|thromb|\bDVT\b|deep vein|\bclot\b", "heart", "血栓", ["thrombo"]),
    (r"pericardial effusion|cardiomegaly|reduced (systolic|ejection)|hypokinesis", "heart", "心脏影像异常",
     ["cardiac"]),
]
NEG_CUE = re.compile(r"\b(no|not|without|negative for|free of|resolved|resolution of|rather than|"
                     r"no evidence of|no definite|ruled out|absence of)\b", re.I)
# negation runs from its cue to the next scope breaker; list commas do not break it
BREAK = re.compile(r";|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\bwhich\b|\bexcept\b", re.I)
POST_NEG = re.compile(r"\b(no longer|has resolved|have resolved|resolved|not (seen|identified|evident|present)|"
                      r"(cannot|can not|could not|not) be excluded|not excluded|is not)\b", re.I)
HISTORY = re.compile(r"\b(known|history of|hx of|s/p|status post|prior|previous)\b", re.I)

# ---- microbiology: specimen -> (organ, finding, questions)
MICRO = [
    (r"^STOOL", "colon", "粪便病原阳性", ["gi_infect"]),
    (r"^BLOOD CULTURE", "blood", "血培养阳性", ["sepsis"]),
    (r"^URINE", "kidney", "尿培养阳性", ["kidney"]),
    (r"^(ABSCESS|PERITONEAL FLUID|TISSUE|FLUID,OTHER|SWAB)", "peritoneum", "腹腔 / 伤口培养阳性", ["abscess"]),
    (r"^BILE", "liver", "胆汁培养阳性", ["liver"]),
    (r"^(SPUTUM|BRONCHOALVEOLAR|BRONCHIAL|Mini-BAL)", "lung", "呼吸道培养阳性", ["sepsis"]),
    (r"^CATHETER TIP", "systemic", "导管培养阳性", ["sepsis"]),
]
POS = re.compile(r"\bpositive\b|\bdetected\b", re.I)
NEG = re.compile(r"negative|not detected|no growth", re.I)


def cohort():
    c = pd.read_parquet(RAW / "cohort")
    c = c[c.is_primary_ibd_diagnosis][["hadm_id", "gender", "age", "admittime", "dischtime"]].drop_duplicates("hadm_id")
    c["los_days"] = (c.dischtime - c.admittime).dt.total_seconds() / 86400
    return c


def lab_facts(hadms):
    l = pd.read_parquet(RAW / "labs", columns=["hadm_id", "label", "fluid", "charttime", "flag"])
    l = l[l.hadm_id.isin(hadms) & (l.flag == "abnormal")]
    rows = []
    for pat, organ, finding, qs in LAB_GROUPS:
        sel = l[l.label.str.contains(pat, regex=True)]
        urine = pat.startswith("^(RBC|WBC")
        sel = sel[sel.fluid.eq("Urine")] if urine else sel[sel.fluid.ne("Urine")]
        rows.append(sel.assign(organ=organ, finding=finding, questions=[qs] * len(sel), source="化验",
                               text=sel.label)[["hadm_id", "charttime", "organ", "finding", "questions", "source", "text"]])
    return pd.concat(rows).rename(columns={"charttime": "t"})


def report_sentences(text):
    m = re.search(r"FINDINGS?:(.*)", text, re.S | re.I)
    body = m.group(1) if m else text
    body = re.split(r"\n\s*(RECOMMENDATION|NOTIFICATION)S?:", body, flags=re.I)[0]
    body = re.sub(r"IMPRESSION:", " ", body, flags=re.I)
    parts = re.split(r"(?<=[.])\s+|\n\s*\d+\.\s+|\b(?=[A-Z][A-Z /&]{2,}:)", body)
    return [" ".join(s.split()) for s in parts if s and len(s.strip()) > 3]


def negated(s, m):
    before = BREAK.split(s[: m.start()])[-1]
    after = BREAK.split(s[m.end():])[0]
    return bool(NEG_CUE.search(before) or POST_NEG.search(after))


def image_facts(hadms):
    r = pd.read_parquet(RAW / "radiology", columns=["hadm_id", "charttime", "text"])
    r = r[r.hadm_id.isin(hadms)]
    pats = [(re.compile(p, re.I), o, f, q) for p, o, f, q in IMG_FINDINGS]
    rows = []
    for h, t, text in r.itertuples(index=False):
        for s in report_sentences(text or ""):
            for pat, organ, finding, qs in pats:
                if any(not negated(s, m) for m in pat.finditer(s)):
                    rows.append((h, t, organ, finding, qs, "影像", s, bool(HISTORY.search(s))))
    return pd.DataFrame(rows, columns=["hadm_id", "t", "organ", "finding", "questions", "source", "text", "history"])


def micro_facts(hadms):
    m = pd.read_parquet(RAW / "microbiology",
                        columns=["hadm_id", "charttime", "chartdate", "spec_type_desc", "test_name", "org_name", "comments"])
    m = m[m.hadm_id.isin(hadms)]
    com = m.comments.fillna("")
    pos = m.org_name.notna() | (com.str.contains(POS) & ~com.str.contains(NEG))
    m = m[pos].assign(t=lambda d: d.charttime.fillna(d.chartdate))
    rows = []
    for pat, organ, finding, qs in MICRO:
        sel = m[m.spec_type_desc.str.contains(pat, regex=True, na=False)]
        sel = sel.drop_duplicates(["hadm_id", "t", "spec_type_desc"])
        rows.append(sel.assign(organ=organ, finding=finding, questions=[qs] * len(sel), source="微生物",
                               text=sel.org_name.fillna(sel.test_name))[
            ["hadm_id", "t", "organ", "finding", "questions", "source", "text"]])
    return pd.concat(rows)


def weigh(facts, coh):
    f = facts.merge(coh[["hadm_id", "admittime"]], on="hadm_id")
    # day -1 = before admission (ED); 0 = first 24 h after admission
    f["day"] = ((f.t - f.admittime).dt.total_seconds() // 86400).clip(lower=-1).astype(int)
    f = f[f.day <= MAX_DAY].sort_values(["hadm_id", "t"]).reset_index(drop=True)
    f["key"] = f.source + "|" + f.organ + "|" + f.finding
    # same finding, same source, same day = one observation
    f = f.drop_duplicates(["hadm_id", "key", "day"]).copy()
    first = ~f.duplicated(["hadm_id", "key"])
    copied = f.source.eq("影像") & f.duplicated(["hadm_id", "text"])
    f["weight"] = np.where(copied, 0.0, np.where(first, 1.0, 0.5))
    return f


def grids(f, hadms, unit):
    """cumulative density per hadm x unit x day (0..MAX_DAY)."""
    if unit == "question":
        f = f.explode("questions").dropna(subset=["questions"]).rename(columns={"questions": "u"})
    else:
        f = f.rename(columns={"organ": "u"})
    keys = list(QUESTIONS if unit == "question" else ORGANS)
    daily = (f.groupby(["hadm_id", "u", "day"]).weight.sum().unstack("day")
             .reindex(columns=DAYS, fill_value=0).fillna(0))
    idx = pd.MultiIndex.from_product([hadms, keys], names=["hadm_id", "u"])
    return daily.reindex(idx, fill_value=0).cumsum(axis=1)


def sea_level(cum, coh, h):
    me = coh.set_index("hadm_id").loc[h]
    peers = coh[(coh.gender == me.gender) & ((coh.age - me.age).abs() <= AGE_CALIPER) & (coh.hadm_id != h)]
    out, n = {}, []
    sub = cum.loc[peers.hadm_id]
    los = peers.set_index("hadm_id").los_days
    for d in DAYS:
        alive = los[los >= d].index
        s = sub.loc[alive].groupby("u")[d]
        out[d] = s.mean()
        n.append(len(alive))
    sd = {d: sub.loc[los[los >= d].index].groupby("u")[d].std() for d in DAYS}
    return pd.DataFrame(out), pd.DataFrame(sd), n, len(peers)


# ---- doctor actions: drawn as lanes for comparison, never deposited as density
LANES = {"transfer": ("转科", "Transfer"), "imaging": ("影像检查", "Imaging"), "procedure": ("操作 / 手术", "Procedure"),
         "consult": ("会诊", "Consult"), "med": ("关键用药", "Key medication")}
MED_CLASSES = [  # (pattern, zh, en); a course starts when the same drug was not given in the previous 24 h
    (r"methylpred|prednis|hydrocortisone|budesonide|dexamethasone", "激素", "Steroid"),
    (r"infliximab|remicade|adalimumab|humira|vedolizumab|entyvio|ustekinumab|stelara|certolizumab|golimumab|"
     r"tofacitinib|upadacitinib|cyclosporin|tacrolimus|azathioprine|mercaptopurine|methotrexate", "免疫 / 生物制剂",
     "Immunomodulator / biologic"),
    (r"ciprofloxacin|metronidazole|vancomycin|piperacillin|cef|meropenem|ertapenem|ampicillin|levofloxacin|"
     r"fidaxomicin|clindamycin|gentamicin|ganciclovir|valganciclovir|fluconazole|micafungin|amoxicillin|"
     r"sulfamethoxazole|linezolid|daptomycin|azithromycin|doxycycline", "抗感染", "Anti-infective"),
    # heparin counts only when given IV (SC heparin = prophylaxis; flush / dwell = line care)
    (r"warfarin|apixaban|rivaroxaban|dabigatran|argatroban|bivalirudin|fondaparinux|enoxaparin \(treatment\)|"
     r"^heparin( sodium)?$", "抗凝", "Anticoagulant"),
    (r"norepinephrine|vasopressin|phenylephrine|epinephrine|dopamine", "升压药", "Vasopressor"),
    (r"packed red|red blood cell|platelet transfusion|fresh frozen", "输血", "Transfusion"),
]
SKIP_CONSULT = re.compile(r"Discharge Followup|Social Work|Nutrition|Physical Therapy|Case Manag", re.I)


def actions(h, admit):
    """doctor actions of one admission: [{t (days), lane, label}]"""
    out = []
    day = lambda t: round((t - admit).total_seconds() / 86400, 2)  # negative = before admission
    clock = lambda t: f"{t:%H:%M}"
    svc = pd.read_parquet(RAW / "services", filters=[("hadm_id", "=", h)])
    for r in svc[svc.prev_service.notna()].itertuples():
        out.append({"t": day(r.transfertime), "lane": "transfer", "clock": clock(r.transfertime),
                    "label": f"{r.prev_service} → {r.curr_service}"})
    rad = pd.read_parquet(RAW / "radiology", columns=["hadm_id", "charttime", "exam_name"], filters=[("hadm_id", "=", h)])
    out += [{"t": day(r.charttime), "clock": clock(r.charttime), "lane": "imaging", "label": r.exam_name} for r in rad.itertuples()]
    pr = pd.read_parquet(RAW / "procedures", filters=[("hadm_id", "=", h)])
    # date only: calendar-day offset, so a procedure on the admission date falls in day 1, not before admission
    out += [{"t": float((pd.Timestamp(r.chartdate).normalize() - admit.normalize()).days), "lane": "procedure", "label": r.long_title, "date_only": True}
            for r in pr.itertuples()]
    poe = pd.read_parquet(RAW / "poe", columns=["hadm_id", "ordertime", "order_type", "order_subtype"],
                          filters=[("hadm_id", "=", h)])
    con = poe[(poe.order_type == "Consults") & ~poe.order_subtype.fillna("").str.contains(SKIP_CONSULT)]
    out += [{"t": day(r.ordertime), "clock": clock(r.ordertime), "lane": "consult", "label": r.order_subtype} for r in con.itertuples()]
    rx = pd.read_parquet(RAW / "prescriptions", columns=["hadm_id", "starttime", "stoptime", "drug", "route"],
                         filters=[("hadm_id", "=", h)]).dropna(subset=["starttime"]).sort_values("starttime")
    for pat, zh, en in MED_CLASSES:
        sel = rx[rx.drug.str.contains(pat, case=False, regex=True, na=False)]
        sel = sel[~sel.drug.str.match(r"(?i)heparin") | sel.route.eq("IV")]
        for drug, g in sel.groupby(sel.drug.str.lower().str.replace(" sodium", "")):
            last = None
            for r in g.itertuples():
                if last is None or r.starttime - last > pd.Timedelta(hours=24):
                    out.append({"t": day(r.starttime), "clock": clock(r.starttime), "lane": "med", "cls": [zh, en],
                                "label": f"{r.drug}{' (' + r.route + ')' if r.route else ''}"})
                last = max(last or r.starttime, r.stoptime if pd.notna(r.stoptime) else r.starttime)
    return sorted(out, key=lambda a: a["t"])


SERVICES = {  # MIMIC service codes -> (zh, en)
    "MED": ("内科", "General Medicine"), "SURG": ("普外科", "General Surgery"), "VSURG": ("血管外科", "Vascular Surgery"),
    "CMED": ("心内科", "Cardiology"), "OMED": ("肿瘤内科", "Oncology"), "CSURG": ("心外科", "Cardiac Surgery"),
    "TSURG": ("胸外科", "Thoracic Surgery"), "NSURG": ("神经外科", "Neurosurgery"), "ORTHO": ("骨科", "Orthopaedics"),
    "GU": ("泌尿科", "Urology"), "GYN": ("妇科", "Gynaecology"), "NMED": ("神经内科", "Neurology"),
    "TRAUM": ("创伤外科", "Trauma Surgery"), "PSURG": ("整形外科", "Plastic Surgery"), "OBS": ("产科", "Obstetrics"),
    "ENT": ("耳鼻喉科", "ENT"), "PSYCH": ("精神科", "Psychiatry"),
}
EN = {
    # organs
    "结肠": "Colon", "小肠": "Small bowel", "腹腔": "Peritoneum", "肝胆": "Liver / biliary", "胰腺": "Pancreas",
    "肾 / 泌尿": "Kidney / urinary", "肺": "Lung", "心脏 / 血管": "Heart / vessels", "血液": "Blood", "全身": "Whole body",
    # questions
    "肠道炎症活动": "Active bowel inflammation", "肠道感染": "Gut infection", "穿孔": "Perforation",
    "脓肿 / 瘘": "Abscess / fistula", "梗阻 / 狭窄": "Obstruction / stricture", "出血 / 贫血": "Bleeding / anaemia",
    "全身感染": "Systemic infection", "肝胆异常": "Liver / biliary problem", "肾 / 泌尿问题": "Kidney / urinary problem",
    "血栓": "Thrombosis", "营养 / 电解质": "Nutrition / electrolytes", "心脏问题": "Cardiac problem",
    # sources
    "化验": "Lab", "影像": "Imaging", "微生物": "Micro",
    # findings
    "血红蛋白低": "Low haemoglobin", "铁 / 溶血指标": "Iron / haemolysis markers", "血小板异常": "Abnormal platelets",
    "凝血异常": "Abnormal clotting", "D-二聚体高": "High D-dimer", "炎症指标高": "High inflammation markers",
    "粪钙卫蛋白高": "High faecal calprotectin", "白细胞异常": "Abnormal white cells", "乳酸高": "High lactate",
    "肾功能异常": "Abnormal kidney function", "尿检异常": "Abnormal urinalysis", "肝功能异常": "Abnormal liver tests",
    "脂肪酶高": "High lipase", "营养指标低": "Low nutrition markers", "电解质异常": "Abnormal electrolytes",
    "心肌指标高": "High cardiac markers", "血气异常": "Abnormal blood gas", "结肠炎症": "Colon inflammation",
    "结肠扩张": "Colon dilatation", "小肠炎症": "Small-bowel inflammation", "小肠梗阻": "Small-bowel obstruction",
    "狭窄": "Stricture", "瘘": "Fistula", "游离气 / 穿孔": "Free air / perforation", "脓肿 / 积液": "Abscess / collection",
    "腹水": "Ascites", "肝胆影像异常": "Liver / biliary imaging finding", "胰腺炎表现": "Pancreatitis signs",
    "肾 / 尿路异常": "Kidney / urinary tract finding", "肺炎 / 实变": "Pneumonia / consolidation",
    "肺部其他异常": "Other lung finding", "心脏影像异常": "Cardiac imaging finding",
    "粪便病原阳性": "Positive stool pathogen", "血培养阳性": "Positive blood culture", "尿培养阳性": "Positive urine culture",
    "腹腔 / 伤口培养阳性": "Positive abdominal / wound culture", "胆汁培养阳性": "Positive bile culture",
    "呼吸道培养阳性": "Positive respiratory culture", "导管培养阳性": "Positive catheter culture",
}
UI = {
    "zh": """{
  doctitle: "病人密度地形", title: "病人密度地形（pilot）",
  intro: "只计“事实”：异常化验、影像阳性所见、微生物阳性，不计用药、医嘱、会诊。海平面 = 同性别、年龄 ±5 岁的 IBD 主诊断住院病人在同一住院日的平均累积密度。转科只作标记。",
  abs: "绝对密度", rel: "高出海平面", layer1: "第一层 · 器官", layer2: "第二层 · 被点亮的问题",
  qlegend: "点亮 = 累积密度高出同类病人平均 1 个标准差以上；数字是“本人 / 海平面”。",
  hmtitle: "时间 · 器官 × 住院日", hmlegend: "“前” = 入院前（急诊阶段）；第 1 天 = 入院后头 24 小时。只有日期的操作按日历日归天。⇄ = 当天有转科，鼠标停在上面看详情。斜线 = 已出院。",
  factstitle: "当天新沉淀的事实", nofacts: "这一天没有新事实。",
  leg_abs: "颜色越深 = 到这一天累积的事实越多。", leg_rel: "橙 = 高于同类病人平均，蓝 = 低于；数字 = 本人 − 海平面。",
  me: "本人", sea: "海平面", transfer: "转科", none: "无", case: "病例",
  actions: "医生行动（只作对照，不进密度）", actstitle: "当天医生行动", noacts: "这一天没有记录到的行动。",
  dateonly: "（只有日期）", dayfmt: d => `第 ${d} 天`, pre: "入院前", pre_short: "前",
  meta: P => `${P.gender === "F" ? "女" : "男"} · ${P.age} 岁 · 住院 ${P.los} 天 · 同类病人 ${P.n_peers} 人 · 转科：`
}""",
    "en": """{
  doctitle: "Patient Density Terrain", title: "Patient density terrain (pilot)",
  intro: "Facts only: abnormal labs, positive imaging findings, positive microbiology. Medications, orders and consults are not counted. Sea level = mean cumulative density of primary-IBD admissions of the same sex, age ±5 years, on the same hospital day. Service transfers are shown as markers only.",
  abs: "Absolute density", rel: "Above sea level", layer1: "Layer 1 · Organs", layer2: "Layer 2 · Problems lit up",
  qlegend: "Lit = cumulative density more than 1 SD above similar patients; numbers are “this patient / sea level”.",
  hmtitle: "Time · organ × hospital day", hmlegend: "“Pre” = before admission (ED); Day 1 = first 24 h after admission. Date-only procedures are placed by calendar day. ⇄ = service transfer that day; hover for details. Hatched = discharged.",
  factstitle: "New facts deposited this day", nofacts: "No new facts on this day.",
  leg_abs: "Darker = more facts accumulated by this day.", leg_rel: "Orange = above similar patients, blue = below; number = this patient − sea level.",
  me: "patient", sea: "sea level", transfer: "Transfer", none: "none", case: "Case",
  actions: "Doctor actions (for comparison, not counted in density)", actstitle: "Doctor actions this day",
  noacts: "No recorded actions on this day.", dateonly: "(date only)",
  dayfmt: d => `Day ${d}`, pre: "Pre-admission", pre_short: "Pre",
  meta: P => `${P.gender === "F" ? "Female" : "Male"} · ${P.age} y · stay ${P.los} days · ${P.n_peers} similar patients · Transfers: `
}""",
}


def localise(data, lang):
    """copy of data with display labels in one language; zh labels are the originals."""
    d = json.loads(json.dumps(data, ensure_ascii=False, default=str))
    tr = (lambda x: EN.get(x, x)) if lang == "en" else (lambda x: x)
    i = 1 if lang == "en" else 0
    d["organs"] = {k: tr(v) for k, v in d["organs"].items()}
    d["questions"] = {k: tr(v) for k, v in d["questions"].items()}
    d["lanes"] = {k: v[i] for k, v in LANES.items()}
    for p in d["patients"]:
        for a in p["actions"]:
            if a["lane"] == "med":
                a["label"] = f'{a.pop("cls")[i]}: {a["label"]}'
            elif a["lane"] == "transfer":
                f_, t_ = a["label"].split(" → ")
                a["label"] = f'{SERVICES.get(f_, (f_, f_))[i]} → {SERVICES.get(t_, (t_, t_))[i]}'
        for f in p["facts"]:
            f["finding"], f["source"] = tr(f["finding"]), tr(f["source"])
        for t in p["transfers"]:
            t["from_name"] = SERVICES.get(t["from"], (t["from"], t["from"]))[i]
            t["to_name"] = SERVICES.get(t["to"], (t["to"], t["to"]))[i]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hadm", type=int, nargs="*", default=PILOT)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    coh = cohort()
    hadms = coh.hadm_id.tolist()
    facts = pd.concat([lab_facts(hadms), image_facts(hadms), micro_facts(hadms)], ignore_index=True)
    facts["history"] = facts.history.astype("boolean").fillna(False).astype(bool)
    f = weigh(facts, coh)
    f.drop(columns=["text"]).to_parquet(OUT / "facts.parquet")

    svc = pd.read_parquet(RAW / "services")
    data = {"organs": ORGANS, "questions": QUESTIONS, "max_day": MAX_DAY, "patients": []}
    for unit in ("organ", "question"):
        cum = grids(f, hadms, unit)
        for h in args.hadm:
            if unit == "organ":
                me = coh.set_index("hadm_id").loc[h]
                tr = (svc[(svc.hadm_id == h) & svc.prev_service.notna()].merge(coh[["hadm_id", "admittime"]])
                      .sort_values("transfertime"))
                p = {"hadm_id": h, "gender": me.gender, "age": int(me.age), "los": round(me.los_days, 1),
                     "transfers": [{"day": round((r.transfertime - r.admittime).total_seconds() / 86400, 2),
                                    "time": f"{r.transfertime:%H:%M}", "from": r.prev_service, "to": r.curr_service}
                                   for r in tr.itertuples()],
                     "facts": [{"day": int(r.day), "organ": r.organ, "finding": r.finding, "source": r.source,
                                "weight": r.weight, "questions": list(r.questions), "history": bool(r.history)}
                               for r in f[f.hadm_id == h].itertuples()],
                     "actions": actions(h, me.admittime)}
                data["patients"].append(p)
            p = next(x for x in data["patients"] if x["hadm_id"] == h)
            mean, sd, n_alive, n_peers = sea_level(cum, coh, h)
            p[unit] = {"cum": cum.loc[h].round(2).to_dict(orient="index"),
                       "sea": mean.round(2).to_dict(orient="index"),
                       "sd": sd.fillna(0).round(2).to_dict(orient="index")}
            p["n_peers"], p["n_alive"] = n_peers, n_alive
    (OUT / "terrain_data.json").write_text(json.dumps(data, ensure_ascii=False, default=str))
    tpl = (Path(__file__).parent / "terrain_pilot_template.html").read_text()
    for lang, name in (("zh", "terrain.html"), ("en", "terrain_en.html")):
        blob = json.dumps(localise(data, lang), ensure_ascii=False, default=str)
        html = tpl.replace("__DATA__", blob).replace("__T__", UI[lang]).replace("__LANG__", lang)
        (OUT / name).write_text(html)
    print(f"facts: {len(f)} rows over {f.hadm_id.nunique()} admissions; "
          f"by source {f.source.value_counts().to_dict()}")
    for p in data["patients"]:
        print(p["hadm_id"], p["gender"], p["age"], "peers", p["n_peers"], "transfers", p["transfers"])


if __name__ == "__main__":
    main()
