#!/usr/bin/env python3
"""Condense one admission's materials.json into a transfer-centred reading sheet.

Input is the output of build_ibd_patient_material_timeline.py. Each kind of
record is kept in the form that suits it instead of being deleted:

- single events: imaging, microbiology, procedures, consults, admit/transfer
- numeric tracks: labs -> one trend line per test, change points only
- courses: prescriptions -> start / dose or route change / stop / restart

Cross-table duplicates (lab orders vs lab results, medication orders vs
prescriptions) are dropped from the timeline. Supportive care (fluids,
electrolytes, pain, nausea, flushes) is collapsed into its own section.
The output contains restricted MIMIC-derived text; write it under runs/.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

GAP = timedelta(hours=12)  # prescriptions further apart than this start a new course
JUMP = 0.25  # relative change that counts as a lab change point

SUPPORTIVE = re.compile(
    r"sodium chloride|^ns$|dextrose|ringer|flush|^bag$|^vial$|sterile water|^sw$|^potassium (chl|phos)|^magnesium|"
    r"calcium|phosph|insulin|acetaminophen|ondansetron|docusate|senna|bisacodyl|polyethylene|"
    r"hydromorphone|morphine|oxycodone|tramadol|fentanyl|lorazepam|zolpidem|melatonin|pantoprazole|"
    r"famotidine|metoclopramide|prochlorperazine|simethicone|nicotine|multivitamin|thiamine|folic|"
    r"lidocaine|glucagon|sucralfate|artificial tear|bacitracin|sarna|menthol|guaifenesin|influenza|"
    r"pneumococcal|ketorolac|ibuprofen|diphenhydramine|cepacol|nystatin|lactulose|famot", re.I)
PROPHYLAXIS = re.compile(r"heparin|enoxaparin", re.I)
# Usual home medicines that rarely bear on the admission's question.
CHRONIC = re.compile(
    r"statin|sartan|pril$|thiazide|amlodipine|levothyroxine|vitamin d|ranitidine|omeprazole|"
    r"calcium carbonate|multivitamin|finasteride|tamsulosin|sertraline|citalopram|escitalopram|"
    r"fluoxetine|gabapentin|allopurinol|metformin|glipizide|cholecalciferol|ferrous", re.I)
# Agents given only to make a test possible: these belong to observation.
TEST_AGENT = re.compile(r"tuberculin|gastroview|diatrizoate|contrast|iohexol|omnipaque|readi-cat", re.I)

GLOSS = {
    "methylprednisolone": "静脉激素", "prednisone": "口服激素", "hydrocortisone": "激素",
    "ciprofloxacin": "抗生素", "metronidazole": "抗生素", "vancomycin": "抗生素",
    "piperacillin": "抗生素", "ceftriaxone": "抗生素", "cefepime": "抗生素", "meropenem": "抗生素",
    "heparin": "抗凝（防/治血栓）", "enoxaparin": "抗凝", "warfarin": "口服抗凝", "alteplase": "溶栓药",
    "adalimumab": "生物制剂", "humira": "生物制剂", "infliximab": "生物制剂", "mesalamine": "IBD 基础药",
    "azathioprine": "免疫抑制", "mercaptopurine": "免疫抑制", "tacrolimus": "免疫抑制",
    "metoprolol": "控心率/降压", "lisinopril": "降压/护心", "furosemide": "利尿", "aspirin": "抗血小板",
}
EXAM_GLOSS = [
    (r"MRCP", "磁共振胆道成像"), (r"CTA", "血管 CT"), (r"\bCT\b", "CT"), (r"ART", "动脉检查"),
    (r"VEIN", "静脉超声"), (r"\bUS\b|ULTRASOUND", "超声"), (r"GASTROGRAF", "造影剂灌肠"),
    (r"CHEST", "胸片"), (r"ABDOMEN", "腹部 X 光"), (r"\bMR", "磁共振"),
]
POE_KEEP = {  # (order_type, subtype or None) -> category
    ("ADT orders", None): "协调", ("Consults", None): "协调",
    ("Radiology", None): "观察", ("Cardiology", None): "观察",
    ("Respiratory", "Mechanical Ventilation"): "干预", ("Blood Bank", "Red Cell Product Order"): "干预",
    ("Hemodialysis", None): "干预", ("TPN", None): "干预",
    ("Nutrition", "NPO/Diet for Procedure"): "干预", ("General Care", "Telemetry"): "观察",
}


def ts(value):
    return datetime.fromisoformat(value) if value else None


def rel(t, anchor):
    hours = (t - anchor).total_seconds() / 3600
    sign = "+" if hours >= 0 else "−"
    hours = abs(hours)
    return f"{sign}{hours:.1f}h" if hours < 48 else f"{sign}{hours / 24:.1f}d"


def gloss(name):
    low = name.lower()
    return next((g for k, g in GLOSS.items() if k in low), "")


def exam_gloss(name):
    return next((g for pattern, g in EXAM_GLOSS if re.search(pattern, name.upper())), "")


def section(text, *heads):
    for head in heads:
        found = re.search(rf"{head}:?\s*(.+?)(?:\n\s*\n[A-Z ]{{4,}}:|\Z)", text, re.S)
        if found:
            return " ".join(found.group(1).split())
    return ""


def late_flag(start, avail, anchor):
    if avail and start < anchor <= avail:
        return "⚠ 检查在转科前、结果在转科后才出"
    return ""


def lab_tracks(events, anchor):
    by_test = defaultdict(list)
    for e in events:
        m = e["material"]
        if m.get("valuenum") is None and not m.get("flag"):
            continue
        by_test[m["label"]].append((ts(m["charttime"]), ts(m["storetime"]), m.get("valuenum"),
                                    m.get("flag") is not None, m.get("valueuom") or ""))
    tracks, timeline, quiet = [], [], []
    for label, rows in by_test.items():
        rows.sort(key=lambda r: r[0])
        if not any(r[3] for r in rows):
            quiet.append(f"{label}×{len(rows)}")
            continue
        points, last = [], None
        for i, (t, avail, v, abn, uom) in enumerate(rows):
            reason = ""
            if last is None:
                reason = "首次"
            elif abn != last[3] and (v is None or last[2] in (None, 0) or abs(v - last[2]) / abs(last[2]) >= 0.1):
                reason = "转为异常" if abn else "恢复正常"
            elif v is not None and last[2] not in (None, 0) and abs(v - last[2]) / abs(last[2]) >= JUMP:
                reason = "明显变化"
            if reason:
                points.append((t, avail, v, abn, reason))
                last = (t, avail, v, abn)
                if reason in ("转为异常", "恢复正常", "明显变化"):
                    timeline.append((t, "观察·监测", f"化验 {label}: {v:g}{'*' if abn else ''} {uom}（{reason}）",
                                     late_flag(t, avail, anchor)))
        trend = " → ".join(f"{v:g}{'*' if a else ''}" for _, _, v, a, _ in points if v is not None)
        before = [r for r in rows if (r[1] or r[0]) < anchor]
        tracks.append((label, len(rows), trend, before[-1][2] if before else None, rows[-1][4]))
    return tracks, timeline, quiet


def courses(events):
    by_drug = defaultdict(list)
    for e in events:
        m = e["material"]
        start, stop = ts(m["starttime"]), ts(m["stoptime"])
        if start is None:
            continue
        by_drug[(m["drug"] or "").strip()].append((start, stop or start, m))
    result = []
    for drug, rows in by_drug.items():
        rows.sort(key=lambda r: r[0])
        current = None
        for start, stop, m in rows:
            dose = f"{m.get('dose_val_rx') or ''} {m.get('dose_unit_rx') or ''} {m.get('route') or ''}".strip()
            if current and start <= current["end"] + GAP:
                if dose != current["doses"][-1][1]:
                    current["doses"].append((start, dose))
                current["end"] = max(current["end"], stop)
                continue
            current = {"drug": drug, "start": start, "end": stop, "doses": [(start, dose)], "m": m,
                       "restart": bool(result and result[-1]["drug"] == drug)}
            result.append(current)
    for c in result:
        m = c["m"]
        supportive = m.get("drug_type") == "BASE" or bool(SUPPORTIVE.search(c["drug"]))
        if PROPHYLAXIS.search(c["drug"]) and (m.get("route") == "SC"):
            supportive = True
        c["kind"] = ("观察辅助" if TEST_AGENT.search(c["drug"]) else
                     "长期基础" if CHRONIC.search(c["drug"]) else
                     "支持维持" if supportive else "疾病")
        c["supportive"] = c["kind"] != "疾病"
    return result


def build(path: Path, anchor_index: int):
    data = json.loads(path.read_text())
    ev = data["events"]
    by = defaultdict(list)
    for e in ev:
        by[e["source"]].append(e)
    services = sorted(by["services"], key=lambda e: e["anchor"])
    transfers = [e for e in services if e["material"]["prev_service"]]
    anchor = ts(transfers[anchor_index]["anchor"])
    coh = data["cohort"]
    admit, disch = ts(coh["admittime"]), ts(coh["dischtime"])
    rows = []  # (time, category, text, note)

    for e in services:
        m = e["material"]
        rows.append((ts(e["anchor"]), "协调", f"**负责科室：{m['prev_service'] or '初始'} → {m['curr_service']}**", ""))
    rows.append((admit, "协调", "入院", ""))
    rows.append((disch, "协调", "出院", ""))

    detail = defaultdict(list)
    for e in by["poe_detail"]:
        m = e["material"]
        detail[m["poe_id"]].append(f"{m['field_name']}={m['field_value']}")
    seen = set()
    for e in by["poe"]:
        m = e["material"]
        key = (m["order_type"], m["order_subtype"])
        cat = POE_KEEP.get(key) or POE_KEEP.get((m["order_type"], None))
        if not cat or m["transaction_type"] != "New" or m["order_type"] == "ADT orders" and m["order_subtype"] == "Admit":
            continue
        t = ts(e["anchor"])
        sig = (t.replace(second=0), key)
        if sig in seen:
            continue
        seen.add(sig)
        label = f"开单：{m['order_type']} / {m['order_subtype'] or ''}".rstrip(" /")
        extra = "; ".join(detail.get(m["poe_id"], []))
        if m["order_subtype"] == "NPO/Diet for Procedure":
            extra = (extra + "; " if extra else "") + "为操作禁食（记录写明为后续操作）"
        rows.append((t, cat if cat != "观察" else "观察(开单)", label, extra))

    exams = defaultdict(int)
    for e in sorted(by["radiology"], key=lambda e: e["anchor"]):
        m = e["material"]
        t, avail = ts(m["charttime"]), ts(m["storetime"])
        text = m.get("text") or ""
        name = m["exam_name"]
        kind = "监测" if exams[name] else "诊断"
        exams[name] += 1
        ind = section(text, "INDICATION", "CLINICAL INDICATION", "HISTORY", "REASON FOR EXAM")
        imp = section(text, "IMPRESSION") or section(text, "FINDINGS")
        notif = section(text, "NOTIFICATION")
        body = (f"影像 {name}（{exam_gloss(name)}）<br>· 检查理由：{ind[:300] or '（无）'}<br>"
                f"· 结论：{imp[:400]}" + (f"<br>· 通知：{notif[:120]}" if notif else ""))
        note = late_flag(t, avail, anchor) or f"报告 {rel(avail, anchor)}" if avail else ""
        rows.append((t, f"观察·{kind}", body, note))

    micro = {}
    for e in by["microbiology"]:
        m = e["material"]
        key = (m["micro_specimen_id"], m["test_name"])
        t = ts(m["charttime"]) or ts(m["chartdate"])
        avail = ts(m["storetime"]) or ts(m["storedate"])
        entry = micro.setdefault(key, {"t": t, "avail": avail, "spec": m["spec_type_desc"], "org": set(), "c": ""})
        if m.get("org_name"):
            entry["org"].add(m["org_name"])
        entry["c"] = entry["c"] or (m.get("comments") or "").strip()
        if avail and (entry["avail"] is None or avail > entry["avail"]):
            entry["avail"] = avail
    for (_, test), x in micro.items():
        result = ("阳性：" + ", ".join(sorted(x["org"]))) if x["org"] else (x["c"][:80] or "无结果文字")
        rows.append((x["t"], "观察·诊断", f"微生物 {x['spec']} · {test}：{result}", late_flag(x["t"], x["avail"], anchor)))

    for e in by["procedures"]:
        m = e["material"]
        rows.append((ts(e["anchor"]), "操作(仅日期)", f"操作编码：{m['long_title']}", "只知日期，不知当天先后"))

    tracks, lab_rows, quiet = lab_tracks(by["labs"], anchor)
    rows += lab_rows

    meds = courses(by["prescriptions"])
    for c in meds:
        if c["supportive"]:
            continue
        g = gloss(c["drug"])
        name = f"{c['drug']}" + (f"（{g}）" if g else "")
        rows.append((c["start"], "干预·疾病", f"{'重新开始' if c['restart'] else '开始'}用药 {name}：{c['doses'][0][1]}", ""))
        for t, dose in c["doses"][1:]:
            rows.append((t, "干预·疾病", f"调整 {name} → {dose}", ""))
        if c["end"] < disch - timedelta(hours=6):
            rows.append((c["end"], "干预·疾病", f"停止 {name}", "处方计划停止时间"))

    rows.sort(key=lambda r: (r[0] or datetime.max))
    raw_n = sum(len(v) for k, v in by.items() if k not in ("diagnoses", "drgcodes"))
    out = [f"# 住院 {data['hadm_id']}：转科阅读表（压缩版）", "",
           f"{coh['gender']}，{coh['age']} 岁；队列 IBD 类型（事后编码）：{coh['ibd_type']}。"
           f"参照转科：**{transfers[anchor_index]['material']['prev_service']} → "
           f"{transfers[anchor_index]['material']['curr_service']}**，{anchor}。",
           f"原始记录 {raw_n} 条 → 时间线 {len(rows)} 条。处方是计划，不等于实际给药；"
           "检查以实施时间排序，报告出来的时间写在备注。`*` = 化验超出参考范围。", "",
           "## 时间线", "", "| 相对转科 | 时间 | 类别 | 内容 | 备注 |", "|---|---|---|---|---|"]
    for t, cat, text, note in rows:
        out.append(f"| {rel(t, anchor) if t else ''} | {t:%m-%d %H:%M} | {cat} | {text} | {note} |" if t else
                   f"| | | {cat} | {text} | {note} |")

    out += ["", "## 疾病相关用药疗程", "", "| 药物 | 开始 | 结束 | 剂量/途径变化 |", "|---|---|---|---|"]
    for c in sorted((c for c in meds if not c["supportive"]), key=lambda c: c["start"]):
        changes = " → ".join(d for _, d in c["doses"])
        out.append(f"| {c['drug']} {gloss(c['drug'])} | {rel(c['start'], anchor)} | {rel(c['end'], anchor)} | {changes} |")

    out += ["", "## 化验趋势（只列出现过异常的项目；只保留变化点）", "",
            "| 项目 | 测量次数 | 变化点 | 转科前最后一次 |", "|---|---|---|---|"]
    for label, n, trend, before, uom in sorted(tracks, key=lambda x: -x[1]):
        out.append(f"| {label} ({uom}) | {n} | {trend} | {before if before is not None else ''} |")
    out += ["", f"始终正常的化验（折叠）：{', '.join(sorted(quiet))}", "",
            "## 其他用药（折叠）：支持维持 / 平时长期用药 / 为检查而用", ""]
    for kind in ("支持维持", "长期基础", "观察辅助"):
        sup = defaultdict(list)
        for c in meds:
            if c["kind"] == kind:
                sup[c["drug"]].append(f"{rel(c['start'], anchor)}~{rel(c['end'], anchor)}")
        out.append(f"- **{kind}**：" + "; ".join(f"{d}（{', '.join(v)}）" for d, v in sorted(sup.items())))
    out += ["", "## 事后材料（只作对照，不作转科前证据）", ""]
    out += [f"- 诊断编码 {e['material']['seq_num']}：{e['material']['long_title']}"
            for e in sorted(by["diagnoses"], key=lambda e: e["material"]["seq_num"])[:6]]
    out.append("- 出院摘要：见 materials.json（事后叙述）")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path, nargs="+")
    parser.add_argument("--anchor-index", type=int, default=0, help="which service transfer to centre on")
    args = parser.parse_args()
    for path in args.materials:
        target = path.with_name("condensed.md")
        target.write_text(build(path, args.anchor_index))
        print(target)


if __name__ == "__main__":
    main()
