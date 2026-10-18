#!/usr/bin/env python3
"""Sameness ② organ and ③ problem, by hand-written tables (no LLM): scripts/organ_map.csv, scripts/problem_map.csv.

Input: imaging facts after ① (runs/sameness/v1/facts.jsonl) + lab / micro units (runs/terrain_units/word/units.parquet).
Each fact gets: organ, problem(s), status = 异常 (adds height) / 查了正常 (checked-normal layer).

Polarity comes from the extraction's per-fact finding_status, never from the tables: "no flow" is abnormal,
"no monophasic waveform" is normal. So both tables are written once per observation, without certainty.

Imaging organ order: obs_fixed[observation] > structure[anatomy] > structure[site word in observation]
                     > obs_default[observation] > 未定位
Problem: (observation, organ) row, else (observation, *) row. Labs by name (+ high / low), micro by specimen.
Nothing is weighted yet: counts only (facts, reports / days, sources).

  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/organ_problem.py [--hadm ...]
Outputs runs/sameness/v1/organ_problem/: facts.csv, summary.md
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

IMG = "runs/sameness/v1/facts.jsonl"
UNITS = "runs/terrain_units/word/units.parquet"
RAW = "data/raw_data/ibd_mimiciv_3_1"
OUT = "runs/sameness/v1/organ_problem"
PILOT = [23605251, 25292961, 29942834]
DROP_TYPES = {"technical", "device", "procedure"}


def tables():
    o = pd.read_csv("scripts/organ_map.csv")
    p = pd.read_csv("scripts/problem_map.csv")
    organ = {k: dict(zip(g.key, g.organ)) for k, g in o.groupby("kind")}
    return organ, p


def image_facts(hadms, organ, pm):
    d = pd.read_json(IMG, lines=True)
    d = d[d.hadm_id.isin(hadms) & ~d.obs_type.isin(DROP_TYPES) & (d.finding_status != "not_evaluated")].copy()
    st = organ["structure"]

    def pick_organ(r):
        if r.obs_core in organ["obs_fixed"]:
            return organ["obs_fixed"][r.obs_core]
        for s in (r.anat_structure, r.obs_site):
            if isinstance(s, str) and s:
                return st.get(s, "UNMAPPED:" + s)
        return organ["obs_default"].get(r.obs_core, "未定位")

    d["organ"] = d.apply(pick_organ, axis=1)
    img = pm[pm.source == "img"]
    exact = {(r.key, r.organ): r.problem for r in img.itertuples() if r.organ != "*"}
    star = {r.key: r.problem for r in img.itertuples() if r.organ == "*"}

    def pick_problem(r):
        if r.obs_type == "vague":
            return "无"
        return exact.get((r.obs_core, r.organ)) or star.get(r.obs_core) or "UNMAPPED"

    d["problem"] = d.apply(pick_problem, axis=1)
    d["status"] = d.finding_status.map({"abnormal": "异常", "equivocal": "异常", "normal": "查了正常"})
    d["what"] = d.anat_structure.fillna(d.obs_site).fillna("") + " · " + d.obs_core
    return d.assign(source="img", report=d.note_id)[
        ["hadm_id", "t", "source", "report", "what", "organ", "problem", "status", "quote"]]


def lab_micro_facts(hadms, organ, pm):
    u = pd.read_parquet(UNITS)
    u = u[u.hadm_id.isin(hadms) & u.source.isin(["lab", "micro"])].copy()
    u["key"] = u.name.where(u.source == "lab", u.name.str.split(" · ").str[0])

    def pick_organ(r):
        t = organ[r.source]
        if r.key in t:
            return t[r.key]
        if r.source == "lab" and r.key.endswith("(Urine)"):
            return t["* (Urine)"]
        return t["*"]

    u["organ"] = u.apply(pick_organ, axis=1)
    u = u[u.organ != "排除"]
    rows = pm[pm.source.isin(["lab", "micro"])]
    by_key = {k: g for k, g in rows.groupby(["source", "key"])}

    def pick_problem(r):
        g = by_key.get((r.source, r.key))
        if g is None:
            return "无"
        if r.status in ("normal", "negative"):
            return g.problem.iloc[0]  # checked-normal for the question this test answers
        hit = g[g.direction.isin([r.status, "*"])]
        return hit.problem.iloc[0] if len(hit) else "无"

    u["problem"] = u.apply(pick_problem, axis=1)
    u["what"] = u.name + " " + u.status
    u["status"] = u.status.map(lambda s: "查了正常" if s in ("normal", "negative") else "异常")
    u["report"] = u.source + ":" + u.t.astype(str)  # one draw / specimen time
    return u.assign(quote="")[["hadm_id", "t", "source", "report", "what", "organ", "problem", "status", "quote"]]


def transfers(hadms):
    s = pd.read_parquet(f"{RAW}/services", filters=[("hadm_id", "in", hadms)])
    s = s[s.prev_service.notna()].sort_values("transfertime")
    return {h: [(r.transfertime, f"{r.prev_service}→{r.curr_service}") for r in g.itertuples()]
            for h, g in s.groupby("hadm_id")}


def table(f, by):
    ab = f[f.status == "异常"]
    if not len(ab):
        return "（无）"
    g = ab.groupby(by).agg(facts=("what", "size"), reports=("report", "nunique"), days=("day", "nunique"),
                           sources=("source", lambda s: "+".join(sorted(set(s)))),
                           img=("source", lambda s: (s == "img").sum()))
    g["checked_normal"] = f[f.status == "查了正常"].groupby(by).size().reindex(g.index).fillna(0).astype(int)
    g = g.sort_values(["days", "reports"], ascending=False)
    lines = [f"| {by} | 异常 facts | 其中影像 | 报告/抽血次数 | 天数 | 来源 | 查了正常 |", "|---|---:|---:|---:|---:|---|---:|"]
    lines += [f"| {i} | {r.facts} | {r.img} | {r.reports} | {r.days} | {r.sources} | {r.checked_normal} |"
              for i, r in g.iterrows()]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hadm", type=int, nargs="*", default=PILOT)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    organ, pm = tables()
    f = pd.concat([image_facts(args.hadm, organ, pm), lab_micro_facts(args.hadm, organ, pm)], ignore_index=True)
    # a fact can answer two problems ("肿瘤 / 占位 + 梗阻 / 狭窄"): one row per problem
    f["problem"] = f.problem.str.split(r" \+ ")
    f = f.explode("problem")
    adm = pd.read_parquet(f"{RAW}/cohort", columns=["hadm_id", "admittime"]).drop_duplicates("hadm_id")
    f = f.merge(adm, on="hadm_id")
    f["t"] = pd.to_datetime(f.t)
    f["day"] = ((f.t - f.admittime).dt.total_seconds() // 86400).astype(int)
    f.drop(columns="admittime").to_csv(f"{OUT}/facts.csv", index=False)

    unm = f[f.organ.str.startswith("UNMAPPED") | (f.problem == "UNMAPPED")]
    tr = transfers(args.hadm)
    lines = ["# Sameness ② 器官 / ③ 问题 — 手工表，3 个 pilot", "",
             "表：`scripts/organ_map.csv`、`scripts/problem_map.csv`。只计数，不加权。"
             "“天数” = 出现异常的不同住院日数；“来源” = img / lab / micro。", "",
             f"未映射：{len(unm)} 条" + ("" if unm.empty else "：" + "; ".join(sorted(set(unm.what))[:30]))]
    for h in args.hadm:
        fh = f[f.hadm_id == h]
        adm_t = adm.set_index("hadm_id").admittime[h]
        cuts = tr.get(h, [])
        lines += ["", f"## {h}", "", "转科：" + "，".join(f"{lab}（第 {(t - adm_t).total_seconds() / 86400:.1f} 天）"
                                                        for t, lab in cuts)]
        for title, sub in [("整次住院", fh)] + [(f"{lab} 之前", fh[fh.t < t]) for t, lab in cuts]:
            lines += ["", f"### 问题 — {title}", "", table(sub, "problem"),
                      "", f"### 器官 — {title}", "", table(sub, "organ")]
    open(f"{OUT}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
