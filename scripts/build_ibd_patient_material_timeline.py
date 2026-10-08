#!/usr/bin/env python3
"""Arrange every locally available source row for one IBD admission by time.

The outputs contain restricted MIMIC-derived material; keep them under runs/.
An event is shown once at its recorded anchor. Spanning prescriptions retain
their end time, while date-only records crossing a service boundary remain
unassigned to either service interval.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw_data/ibd_mimiciv_3_1"
TABLES = (
    "services", "labs", "microbiology", "poe", "prescriptions",
    "procedures", "radiology", "discharge_notes", "diagnoses",
)


def serialize(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()
    return value


def read_rows(folder: Path, hadm_id: int):
    dataset = ds.dataset(folder, format="parquet")
    return dataset.to_table(filter=ds.field("hadm_id") == hadm_id).to_pylist()


def start_end(name: str, row: dict):
    """Return display anchor, possible interval end, and temporal semantics."""
    if name == "services":
        return row["transfertime"], None, "service_assignment"
    if name == "labs":
        return row["charttime"], None, "charted_measurement"
    if name == "microbiology":
        if row["charttime"] is not None:
            return row["charttime"], None, "charted_microbiology"
        if row["chartdate"] is not None:
            start = datetime.combine(row["chartdate"].date(), time.min)
            return start, start + timedelta(days=1), "date_only_microbiology"
        return None, None, "untimed_microbiology"
    if name == "poe":
        return row["ordertime"], None, "order_placed"
    if name == "prescriptions":
        return row["starttime"], row["stoptime"], "prescribed_period"
    if name == "procedures":
        if row["chartdate"] is None:
            return None, None, "untimed_coded_procedure"
        start = datetime.combine(row["chartdate"], time.min)
        return start, start + timedelta(days=1), "date_only_coded_procedure"
    if name == "radiology":
        return row["charttime"], None, "charted_radiology_report"
    if name == "discharge_notes":
        return row["charttime"], None, "retrospective_discharge_note"
    return None, None, "untimed_retrospective_diagnosis"


def title(name: str, row: dict):
    if name == "services":
        return f"{row['prev_service'] or '初始'} → {row['curr_service']}"
    if name == "labs":
        return f"{row.get('label') or row.get('itemid')}: {row.get('value') or ''} {row.get('valueuom') or ''}".strip()
    if name == "microbiology":
        return f"{row.get('spec_type_desc') or ''} · {row.get('test_name') or ''} · {row.get('org_name') or ''}".strip(" ·")
    if name == "poe":
        return f"{row.get('order_type')} · {row.get('order_subtype') or '未细分'} · {row.get('transaction_type') or ''}"
    if name == "prescriptions":
        return f"{row.get('drug') or '未命名药物'} · {row.get('dose_val_rx') or ''} {row.get('dose_unit_rx') or ''}".strip()
    if name == "procedures":
        return row.get("long_title") or row.get("icd_code") or "编码操作"
    if name == "radiology":
        return row.get("exam_name") or "影像报告"
    if name == "discharge_notes":
        return "出院摘要（事后叙述）"
    return f"{row.get('icd_code')} · {row.get('long_title') or ''}"


def recorded_at(name: str, row: dict):
    if name in {"labs", "radiology", "discharge_notes"}:
        return row.get("storetime")
    if name == "microbiology":
        return row.get("storetime") or row.get("storedate")
    return None


def interval_label(start, end, service):
    if service == "before_admission":
        return f"入院前 · 至 {end}"
    if service == "after_discharge":
        return f"出院后 · 自 {start}"
    if service is None:
        return f"已入院、服务未记录 · {start} — {end}"
    return f"{service} 负责时段 · {start} — {end}"


def make_intervals(cohort, services):
    admit, discharge = cohort["admittime"], cohort["dischtime"]
    rows = sorted(services, key=lambda r: r["transfertime"])
    intervals = [{"id": "before_admission", "start": None, "end": admit, "service": "before_admission"}]
    cursor, current = admit, None
    for row in rows:
        at = row["transfertime"]
        if at < admit or at > discharge:
            continue
        if at > cursor:
            intervals.append({"id": f"segment_{len(intervals)}", "start": cursor, "end": at, "service": current})
        cursor, current = at, row["curr_service"]
    if cursor < discharge:
        intervals.append({"id": f"segment_{len(intervals)}", "start": cursor, "end": discharge, "service": current})
    intervals.append({"id": "after_discharge", "start": discharge, "end": None, "service": "after_discharge"})
    for item in intervals:
        item["label"] = interval_label(item["start"], item["end"], item["service"])
    return intervals


def assign(event, intervals):
    start, end = event["anchor"], event["end"]
    if start is None:
        return "untimed", []
    if end is None:
        overlaps = [i["id"] for i in intervals if
                    (i["start"] is None or start >= i["start"]) and
                    (i["end"] is None or start < i["end"])]
    else:
        overlaps = [i["id"] for i in intervals if
                    (i["end"] is None or start < i["end"]) and
                    (i["start"] is None or end > i["start"])]
    if event["temporal_kind"].startswith("date_only") and len(overlaps) > 1:
        return "boundary_uncertain", overlaps
    for item in intervals:
        if (item["start"] is None or start >= item["start"]) and (item["end"] is None or start < item["end"]):
            return item["id"], overlaps
    return "untimed", overlaps


def make_html(payload):
    events = payload["events"]
    groups = [(i["id"], i["label"]) for i in payload["intervals"]]
    groups += [("boundary_uncertain", "仅知日期，跨越服务变更时刻"), ("untimed", "没有可定位事件时间的回顾性材料")]
    styles = """body{font:15px/1.55 system-ui,sans-serif;max-width:1160px;margin:28px auto;padding:0 20px;color:#253044;background:#f7f8fb}h1,h2{color:#17233a}header,.card{background:white;border:1px solid #dfe4ec;border-radius:10px;padding:16px;margin:12px 0}header{position:sticky;top:0;z-index:2;box-shadow:0 2px 12px #0001}input{width:70%;padding:8px;border:1px solid #adb8c8;border-radius:6px}select{padding:8px}.meta{color:#5c6880;font-size:13px}.event{border-left:3px solid #617fbd;padding:8px 12px;margin:8px 0;background:#f8faff}.event summary{cursor:pointer}.event pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.45 ui-monospace,monospace;background:#fff;padding:12px;border:1px solid #e0e6ef}.badge{display:inline-block;background:#e5edfa;border-radius:4px;padding:1px 5px;margin-right:5px}.warn{background:#fff2ce;padding:9px;border-radius:6px}.count{font-size:13px;color:#526177}"""
    parts = ["<!doctype html><html lang='zh'><meta charset='utf-8'><title>住院材料时间轴</title>",
             f"<style>{styles}</style><header><h1>住院材料时间轴 · {payload['hadm_id']}</h1>",
             "<p class='meta'>按服务归属时段排列原始材料；展开可查看完整源记录。日期仅有日精度时，不推断当天先后。出院摘要是事后叙述。</p>",
             "<input id='query' placeholder='搜索材料标题、来源或内容'> <select id='source'><option value=''>全部来源</option>"]
    for name in sorted(payload["source_counts"]):
        parts.append(f"<option>{html.escape(name)}</option>")
    parts.append("</select><p id='visible' class='count'></p></header>")
    for group_id, label in groups:
        subset = [e for e in events if e["group"] == group_id]
        if not subset:
            continue
        parts.append(f"<section class='card'><h2>{html.escape(label)} <small class='count'>({len(subset)} 条)</small></h2>")
        for e in sorted(subset, key=lambda x: (x["anchor"] or datetime.max, x["source"], x["source_index"])):
            raw = json.dumps(e["material"], ensure_ascii=False, indent=2, default=serialize)
            search = f"{e['source']} {e['title']} {raw}".lower()
            metadata = f"{e['anchor'] or '无时间'} · {e['temporal_kind']}"
            if e["recorded_at"] is not None:
                metadata += f" · 记录／验证于 {e['recorded_at']}"
            if e["end"] is not None:
                metadata += f" · 至 {e['end']}（排他或处方停止时间）"
            if e["overlaps"] and len(e["overlaps"]) > 1:
                metadata += " · 跨时段：" + ", ".join(e["overlaps"])
            parts.append(f"<details class='event' data-source='{html.escape(e['source'],quote=True)}' data-search='{html.escape(search,quote=True)}'><summary><span class='badge'>{html.escape(e['source'])}</span> {html.escape(e['title'])}<div class='meta'>{html.escape(metadata)}</div></summary><pre>{html.escape(raw)}</pre></details>")
        parts.append("</section>")
    parts.append("<section class='card'><h2>住院背景与派生材料</h2><p class='meta'>这两类记录不是住院过程中的单点事件。cohort 含入出院边界和最终队列标签；admission_inputs 从其他来源提取文字，可能重复出院摘要。</p>")
    for label, material in [("cohort", payload["cohort"]), ("admission_inputs", payload["derived_admission_inputs"])]:
        raw = json.dumps(material, ensure_ascii=False, indent=2, default=serialize)
        parts.append(f"<details class='event'><summary>{html.escape(label)}</summary><pre>{html.escape(raw)}</pre></details>")
    parts.append("</section>")
    parts.append("""<script>const q=document.querySelector('#query'),s=document.querySelector('#source'),v=document.querySelector('#visible');function update(){let n=0;for(const e of document.querySelectorAll('.event')){let show=(!s.value||e.dataset.source===s.value)&&e.dataset.search.includes(q.value.toLowerCase());e.hidden=!show;if(show)n++}v.textContent=`显示 ${n} 条原始记录`;}q.addEventListener('input',update);s.addEventListener('change',update);update();</script></html>""")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hadm_id", type=int)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = (args.output_dir or ROOT / "runs" / "ibd_patient_material_timeline" / str(args.hadm_id)).resolve()
    if output == source or source in output.parents:
        parser.error("Output must be outside the raw source directory")
    cohort_rows = read_rows(source / "cohort", args.hadm_id)
    if len(cohort_rows) != 1:
        parser.error("Expected exactly one cohort row for hadm_id")
    cohort = cohort_rows[0]
    raw = {name: read_rows(source / name, args.hadm_id) for name in TABLES}
    intervals = make_intervals(cohort, raw["services"])
    events = []
    for name, rows in raw.items():
        for index, row in enumerate(rows):
            anchor, end, kind = start_end(name, row)
            event = {"source": name, "source_index": index, "title": title(name, row),
                     "anchor": anchor, "end": end, "recorded_at": recorded_at(name, row),
                     "temporal_kind": kind, "material": row}
            event["group"], event["overlaps"] = assign(event, intervals)
            events.append(event)
    # Derived admission_inputs repeat cohort data and extract text from the discharge
    # note. Keep it in the export as a clearly marked derivative, not another event.
    derived = read_rows(source / "admission_inputs", args.hadm_id)
    payload = {"schema_version": "ibd-patient-material-timeline/1.0", "hadm_id": args.hadm_id,
               "cohort": cohort, "intervals": intervals, "events": events,
               "derived_admission_inputs": derived,
               "source_counts": dict(Counter(e["source"] for e in events)),
               "scope": "Locally extracted tables for this admission; not the complete EHR."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "materials.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=serialize) + "\n")
    (output / "timeline.html").write_text(make_html(payload))
    print(json.dumps({"output": str(output), "events": len(events),
                      "source_counts": payload["source_counts"],
                      "group_counts": dict(Counter(e["group"] for e in events))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
