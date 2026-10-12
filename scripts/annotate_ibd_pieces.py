#!/usr/bin/env python3
"""Split one admission into annotated trace pieces and a problem-by-time grid.

Each piece gets three dimensions:
- nature: statement (事实陈述) / inference (推断) / question (疑问) / action (行动)
- voice: instrument (仪器) / reader (检查医生) / orderer (开单医生) / team (团队)
- time: hours relative to the reference transfer, plus result availability

Imaging reports are split into sentences: the indication speaks for the
ordering clinician, the impression for the reader. Nature is pre-filled by
wording rules and must be reviewed. Problems come from a hand-written
problems.json beside materials.json ({"problem": ["regex", ...]}). A piece
counts towards every problem it mentions; names starting with "_" are kept
in the listing but left out of the grid. Without the file every piece is
left unassigned, which is the starting point for writing one.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from condense_ibd_admission import courses, exam_gloss, lab_tracks, rel, section, ts

HEDGE = re.compile(r"\b(possible|possibly|likely|may|might|suggest|consistent with|compatible with|"
                   r"concern|presum|probabl|favor|could|cannot be excluded|differential|vs\.?|versus)\b", re.I)
ASK = re.compile(r"\?|\b(eval|evaluate|assess|r/o|rule out|query|please|look for)\b", re.I)
BINS = [(-1e9, -72, "<−3d"), (-72, -48, "−3~−2d"), (-48, -24, "−2~−1d"), (-24, -12, "−24~−12h"),
        (-12, 0, "−12~0h"), (0, 12, "0~12h"), (12, 24, "12~24h"), (24, 72, "+1~3d"), (72, 1e9, ">+3d")]
SHORT = {"事实陈述": "事", "推断": "推", "疑问": "问", "行动": "行"}


def sentences(text):
    parts = re.split(r"(?<=[.;])\s+|//|\s(?=\d\.\s)", text)
    return [" ".join(p.split()).strip(" .") for p in parts if len(p.strip(" .")) > 2]


def nature_of(text, voice):
    if voice == "开单医生" and ASK.search(text):
        return "疑问"
    return "推断" if HEDGE.search(text) else "事实陈述"


def pieces(data, anchor_index=0):
    by = defaultdict(list)
    for e in data["events"]:
        by[e["source"]].append(e)
    services = sorted(by["services"], key=lambda e: e["anchor"])
    transfers = [e for e in services if e["material"]["prev_service"]]
    out = []

    def add(t, nature, voice, source, text, avail=None):
        out.append({"t": t, "avail": avail, "nature": nature, "voice": voice, "source": source, "text": text})

    for e in services:
        m = e["material"]
        add(ts(e["anchor"]), "行动", "团队", "科室", f"负责科室 {m['prev_service'] or '初始'} → {m['curr_service']}")
    for e in by["radiology"]:
        m = e["material"]
        t, avail, text = ts(m["charttime"]), ts(m["storetime"]), m.get("text") or ""
        name = f"{m['exam_name']}（{exam_gloss(m['exam_name'])}）"
        add(t, "行动", "团队", "影像", f"做检查 {name}", avail)
        ind = section(text, "INDICATION", "CLINICAL INDICATION", "HISTORY", "REASON FOR EXAM", paragraph=True)
        for s in sentences(ind):
            add(t, nature_of(s, "开单医生"), "开单医生", "影像·检查理由", s, avail)
        for s in sentences(section(text, "IMPRESSION") or section(text, "FINDINGS"))[:8]:
            add(avail or t, nature_of(s, "检查医生"), "检查医生", "影像·结论", s, avail)
    seen = set()
    for e in by["poe"]:
        m = e["material"]
        kind = (m["order_type"], m["order_subtype"])
        if m["transaction_type"] != "New" or m["order_type"] not in ("ADT orders", "Consults", "Cardiology") \
                and kind not in (("Nutrition", "NPO/Diet for Procedure"), ("General Care", "Telemetry")):
            continue
        if kind == ("ADT orders", "Admit") or (ts(e["anchor"]).replace(second=0), kind) in seen:
            continue
        seen.add((ts(e["anchor"]).replace(second=0), kind))
        add(ts(e["anchor"]), "行动", "团队", "医嘱", f"开单 {m['order_type']} / {m['order_subtype'] or ''}".rstrip(" /"))
    for e in by["microbiology"]:
        m = e["material"]
        if m.get("org_name") or m.get("comments"):
            text = f"{m['spec_type_desc']} {m['test_name']}: {m.get('org_name') or (m.get('comments') or '').strip()[:80]}"
            add(ts(m["charttime"]) or ts(m["chartdate"]), "事实陈述", "仪器", "微生物", text,
                ts(m["storetime"]) or ts(m["storedate"]))
    for e in by["echo_measurements"]:
        m = e["material"]
        keep = {x["measurement"]: x for x in m["measurements"] if x["result"]}
        parts = [f"{keep[k]['measurement_description']}: {keep[k]['result']}{keep[k]['unit'] or ''}"
                 for k in ("lvef", "lvef_upper", "pericardial_effusion", "rv_function", "pfo_asd") if k in keep]
        if parts:
            # measurement time only; when the result reached the team is unknown
            add(ts(m["measurement_datetime"]), "事实陈述", "仪器", f"心超 {m['test_type'].upper()}", "; ".join(parts))
    for e in by["procedures"]:
        add(ts(e["anchor"]), "行动", "团队", "操作(仅日期)", e["material"]["long_title"])
    anchor = ts(transfers[anchor_index]["anchor"])
    _, lab_rows, _ = lab_tracks(by["labs"], anchor)
    for t, _, text, _ in lab_rows:
        add(t, "事实陈述", "仪器", "化验", text.replace("化验 ", ""))
    for c in courses(by["prescriptions"]):
        if c["kind"] != "疾病":
            continue
        add(c["start"], "行动", "团队", "用药", f"{'重新开始' if c['restart'] else '开始'} {c['drug']} {c['doses'][0][1]}")
        for t, dose in c["doses"][1:]:
            add(t, "行动", "团队", "用药", f"调整 {c['drug']} → {dose}")
    return sorted(out, key=lambda p: p["t"]), transfers


def assign(items, problems):
    compiled = [(name, re.compile("|".join(pats), re.I)) for name, pats in problems.items()]
    for p in items:
        p["problem"] = [name for name, rx in compiled if rx.search(p["text"])]


def grid(items, anchor):
    cells = defaultdict(lambda: defaultdict(int))
    for p in items:
        # a piece weighs in when it becomes available, not when the sample was taken
        hours = ((p["avail"] or p["t"]) - anchor).total_seconds() / 3600
        label = next(b for lo, hi, b in BINS if lo <= hours < hi)
        for name in p["problem"]:
            if not name.startswith("_"):
                cells[name][(label, p["nature"])] += 1
    return cells


def render(data, items, transfers, cells, problems, anchor_index=0):
    anchor = ts(transfers[anchor_index]["anchor"])
    names = [n for n in problems if not n.startswith("_")]
    tr = " / ".join(f"{e['material']['prev_service']}→{e['material']['curr_service']} {rel(ts(e['anchor']), anchor)}"
                    for e in transfers)
    out = [f"# 住院 {data['hadm_id']}：痕迹标注与问题地形", "",
           f"参照转科（0 点）：{tr}。每格：事=事实陈述 推=推断 问=疑问 行=行动；按信息**可得时间**计入。", "",
           "## 问题 × 时间", "", "| 问题 | " + " | ".join(b for _, _, b in BINS) + " |",
           "|---|" + "---|" * len(BINS)]
    for n in names:
        row = []
        for _, _, b in BINS:
            c = cells[n]
            row.append(" ".join(f"{SHORT[k]}{c[(b, k)]}" for k in SHORT if c[(b, k)]))
        out.append(f"| {n} | " + " | ".join(row) + " |")
    unassigned = sum(1 for p in items if not p["problem"])
    out += ["", f"未归类痕迹：{unassigned} 条（多为例行化验或无关用药）。", "",
            "## 逐条痕迹", "", "| 相对转科 | 问题 | 性质 | 来源 | 内容 |", "|---|---|---|---|---|"]
    for p in items:
        late = " ⚠结果在转科后" if p["avail"] and p["t"] < anchor <= p["avail"] else ""
        out.append(f"| {rel(p['t'], anchor)} | {'、'.join(p['problem'])} | {p['nature']} | {p['voice']}·{p['source']} | "
                   f"{p['text'][:220]}{late} |")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path, nargs="+")
    parser.add_argument("--anchor-index", type=int, default=0, help="which transfer is time zero")
    args = parser.parse_args()
    for path in args.materials:
        data = json.loads(path.read_text())
        items, transfers = pieces(data, args.anchor_index)
        pfile = path.with_name("problems.json")
        problems = json.loads(pfile.read_text()) if pfile.exists() else {}
        assign(items, problems)
        with path.with_name("pieces.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rel", "time", "available", "problem", "nature", "voice", "source", "text"])
            anchor = ts(transfers[args.anchor_index]["anchor"])
            for p in items:
                w.writerow([rel(p["t"], anchor), p["t"], p["avail"] or "", "|".join(p["problem"]), p["nature"],
                            p["voice"], p["source"], p["text"]])
        target = path.with_name("terrain.md")
        target.write_text(render(data, items, transfers, grid(items, anchor), problems, args.anchor_index))
        print(target, len(items), "pieces")


if __name__ == "__main__":
    main()
