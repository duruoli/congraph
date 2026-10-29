#!/usr/bin/env python3
"""Abdominal Bayes env: coarse observations o from the standardized atomic facts (results/evidence_pieces_std/).

  s  one row per admission: the disease file it came from (4-way, mutually exclusive by construction).
  o  one row per admission x source unit x item = the MOST abnormal result of that item in that unit:
       laboratory    item = analyte (attribute | section); value = low / normal / high, recomputed from the value
                     and the stored reference range; analytes without a range keep abnormal / normal
       microbiology  item = specimen | test; value = abnormal (positive) / normal
       radiology     item = anatomy_canonical | attribute_canonical, one unit per report (note_id);
                     value = abnormal / equivocal / normal; the state wording is dropped at this level, so
                     state_canonical == __DROP__ pieces are kept (their finding_status is still valid)
       history, physical_exam   item = anatomy_canonical | attribute_canonical, one unit per admission
     not_evaluated = not observed (no row). Labs / micro / history / exam carry no time (admission-aggregated);
     radiology rows carry the report charttime and exam name.

Outputs (restricted MIMIC-derived, runs/ only) in runs/bayes_env/abd/:
  s.parquet  o.parquet  items.csv (per item: admissions observed, abnormal rate by disease)  summary.md

  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/abd_obs.py
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIECES = ROOT / "results/evidence_pieces_std"
RAW = ROOT / "data/raw_data"
OUT = ROOT / "runs/bayes_env/abd"
DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]
NARRATIVE = {"history", "physical_exam"}
SEVERITY = {"high": 2, "low": 2, "abnormal": 2, "equivocal": 1, "normal": 0}
MIN_ADM = 10  # an item is "usable" when observed in >= this many admissions

NUM = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")
REF = re.compile(r"ref\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)")


def lab_value(p):
    """low / normal / high from value vs reference range; fall back to finding_status."""
    m, r = NUM.match(str(p.get("value_unit") or "")), REF.search(str(p.get("qualifier") or ""))
    if m and r:
        v, lo, hi = float(m.group(1)), float(r.group(1)), float(r.group(2))
        return "low" if v < lo else "high" if v > hi else "normal"
    return p["finding_status"]


def radiology_meta():
    """note_id -> exam name, modality, region (from the disease CSVs) + charttime (MIMIC-IV-Note)."""
    rows = []
    for d in DISEASES:
        for h, r in pd.read_pickle(RAW / f"{d}_hadm_info_first_diag.pkl").items():
            rad = r["Radiology"]
            for x in (ast.literal_eval(rad) if isinstance(rad, str) else rad or []):
                rows.append((x["Note ID"], h, x["Exam Name"], x["Modality"], x["Region"]))
    meta = pd.DataFrame(rows, columns=["note_id", "hadm_id", "exam", "modality", "region"])
    t = pd.read_csv(RAW / "mimic_source/radiology.csv.gz", usecols=["note_id", "charttime"])
    a = pd.read_csv(RAW / "mimic_source/admissions.csv.gz", usecols=["hadm_id", "admittime"])
    return meta.merge(t, on="note_id", how="left").merge(a, on="hadm_id", how="left")


def load():
    s, rows = [], []
    for d in DISEASES:
        for line in open(PIECES / f"{d}.jsonl"):
            r = json.loads(line)
            s.append((r["hadm_id"], d))
            for p in r["pieces"]:
                src, fs = p["source_test"], p["finding_status"]
                if src == "laboratory":
                    v = lab_value(p)
                    if v == "not_evaluated":
                        continue
                    rows.append((r["hadm_id"], "laboratory", "adm", f"{p['attribute']} | {p['section']}", v))
                    continue
                if fs == "not_evaluated":
                    continue
                item = f"{p['anatomy_canonical']} | {p['attribute_canonical']}"
                if src == "microbiology":
                    rows.append((r["hadm_id"], "microbiology", "adm", item, fs))
                elif src in NARRATIVE:
                    rows.append((r["hadm_id"], src, "adm", item, fs))
                else:
                    rows.append((r["hadm_id"], "radiology", p["note_id"], item, fs))
    s = pd.DataFrame(s, columns=["hadm_id", "disease"])
    o = pd.DataFrame(rows, columns=["hadm_id", "source", "unit", "item", "value"])
    # most abnormal result per admission x unit x item
    o["sev"] = o.value.map(SEVERITY)
    o = (o.sort_values("sev").groupby(["hadm_id", "source", "unit", "item"], as_index=False).last()
         .drop(columns="sev"))
    return s, o


def item_table(s, o):
    """Per item: admissions observed and abnormal rate (any non-normal value) by disease."""
    x = o.merge(s, on="hadm_id")
    x["abn"] = x.value != "normal"
    per_adm = x.groupby(["source", "item", "hadm_id", "disease"]).abn.max().reset_index()
    t = per_adm.groupby(["source", "item"]).agg(n_adm=("hadm_id", "nunique")).reset_index()
    rate = per_adm.pivot_table(index=["source", "item"], columns="disease", values="abn", aggfunc="mean")
    cnt = per_adm.pivot_table(index=["source", "item"], columns="disease", values="abn", aggfunc="size")
    rate.columns = [f"abn_{c[:5]}" for c in rate.columns]
    cnt.columns = [f"n_{c[:5]}" for c in cnt.columns]
    t = t.merge(cnt.reset_index(), on=["source", "item"]).merge(rate.reset_index(), on=["source", "item"])
    return t.sort_values(["source", "n_adm"], ascending=[True, False])


def summary(s, o, items):
    n = len(s)
    lines = ["# Abdominal Bayes env: coarse observations\n",
             f"Admissions with atomic facts: {n} ({s.disease.value_counts().to_dict()}).\n",
             "| source | rows | items | items in >= %d adm | rows covered by those items | values |" % MIN_ADM,
             "|---|---:|---:|---:|---:|---|"]
    for src, g in o.groupby("source"):
        it = items[items.source == src]
        keep = set(it[it.n_adm >= MIN_ADM].item)
        lines.append(f"| {src} | {len(g)} | {g.item.nunique()} | {len(keep)} | "
                     f"{g.item.isin(keep).mean():.0%} | {g.value.value_counts().to_dict()} |")
    r = o[o.source == "radiology"]
    lines +=["", f"Radiology reports with facts: {r.unit.nunique()}; matched to exam metadata: "
              f"{r.modality.notna().groupby(r.unit).first().mean():.0%}.",
              "Reports by modality x region: "
              + str((r.drop_duplicates("unit").modality + "/" + r.drop_duplicates("unit").region)
                    .value_counts().head(8).to_dict()), ""]
    lines.append("## Most discriminating usable radiology items (abnormal rate by disease)\n")
    rad = items[(items.source == "radiology") & (items.n_adm >= MIN_ADM)].copy()
    abn = [c for c in rad.columns if c.startswith("abn_")]
    rad["spread"] = rad[abn].max(axis=1) - rad[abn].min(axis=1)
    top = rad.sort_values("spread", ascending=False).head(15)
    lines += ["| item | n_adm | " + " | ".join(abn) + " |", "|---|---:|" + "---:|" * len(abn)]
    lines += [f"| {r['item']} | {r.n_adm} | " + " | ".join(f"{r[c]:.2f}" for c in abn) + " |"
              for _, r in top.iterrows()]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s, o = load()
    meta = radiology_meta()
    rad = o.source == "radiology"
    o = o.merge(meta[["note_id", "exam", "modality", "region", "charttime", "admittime"]],
                left_on="unit", right_on="note_id", how="left").drop(columns="note_id")
    o.loc[~rad.values, ["exam", "modality", "region", "charttime", "admittime"]] = None
    items = item_table(s, o)
    s.to_parquet(OUT / "s.parquet", index=False)
    o.to_parquet(OUT / "o.parquet", index=False)
    items.to_csv(OUT / "items.csv", index=False)
    summary(s, o, items)
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
