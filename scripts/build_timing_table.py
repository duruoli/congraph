"""
Build the per-decision-step timing table: attach `timing_role` (pre/post-intervention)
to every annotated decision step so monitoring scans can be excluded from the deviation
analysis. See experiments/annotation/timing.py for the logic and rationale.

The join key is `note_id` ("<subject_id>-RR-<n>"), reconstructed from the derived
Radiology records (present today) and matched to MIMIC-IV-Note radiology.csv when it
arrives. Runs in three modes:

  --self-test          validate the intervention classifier on the derived data (no source)
  (no --source-dir)    degraded build: note_id + intervention types + text hint, role=unknown
  --source-dir DIR     full build: join charttime/admittime/procedure dates -> real role

Output: results/annotation_experiment/full/timing_roles.csv
  disease, hadm, step, rrn, note_id, modality,
  dev_belief, verification,                      (joined from belief_deviation_analysis.csv)
  interventions, first_intervention_type,       (from procedure titles / codes)
  charttime, admittime, first_intervention_date,
  timing_role, text_post_intervention_hint, exclude_from_deviation

Usage:
  /opt/anaconda3/bin/python3.12 scripts/build_timing_table.py --self-test
  /opt/anaconda3/bin/python3.12 scripts/build_timing_table.py
  /opt/anaconda3/bin/python3.12 scripts/build_timing_table.py --source-dir data/mimic_source
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.annotation.timing import (  # noqa: E402
    SourceTables, classify_procedure, interventions_from_titles,
    timing_role, MONITORING_ROLES, UNKNOWN,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = {d: ROOT / f"data/raw_data/{d}_hadm_info_first_diag.csv"
       for d in ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")}
ANN_DIR = ROOT / "results/annotation_experiment/full"
DEV_CSV = ANN_DIR / "belief_deviation_analysis.csv"

# Text fallback (no charttime yet): the validated tight markers from the raw-text audit.
_T1 = re.compile(r"(post[- ]?ercp|sphincterotom|biliary stent|cbd stent|common bile duct stent"
                 r"|cholecystostomy|percutaneous drain|drainage catheter|drain catheter|pigtail"
                 r"|interval (decrease|increase|improv|resolution|worsen|progress|develop|chang))", re.I)
_T2 = re.compile(r"((status post|s/p|recent|prior) (appendectomy|cholecystectomy))"
                 r"|phlegmonous changes in the post-surgical bed"
                 r"|post-?surgical (bed|changes?)[^.]{0,40}(append|cecum|gallbladder|cholecyst|pancrea|colon|sigmoid)", re.I)


def _raw_df(disease: str) -> pd.DataFrame:
    return pd.read_csv(RAW[disease])


def _note_ids_and_reports(disease: str, hadm: int) -> dict[int, tuple[str, str]]:
    """rrn -> (full note_id, report text) from the derived Radiology blob."""
    df = _raw_df(disease)
    row = df[df.hadm_id == hadm]
    if row.empty or not isinstance(row.iloc[0]["Radiology"], str):
        return {}
    out: dict[int, tuple[str, str]] = {}
    for rec in json.loads(row.iloc[0]["Radiology"]):
        nid = rec.get("Note ID", "")
        if "-RR-" in nid:
            out[int(nid.split("-RR-")[-1])] = (nid, rec.get("Report", ""))
    return out


def _text_hint(report: str) -> str:
    if _T1.search(report):
        return "T1"
    if _T2.search(report):
        return "T2"
    return ""


def self_test() -> None:
    """Validate the intervention classifier on the derived data — no source needed."""
    print("=== intervention classifier self-test (derived procedure titles) ===\n")
    rows = []
    for disease, path in RAW.items():
        df = _raw_df(disease)
        for _, r in df.iterrows():
            iv = interventions_from_titles(r)
            rows.append({"disease": disease, "hadm": int(r.hadm_id),
                         "types": "|".join(sorted(iv)), "n_types": len(iv)})
    t = pd.DataFrame(rows)
    print(f"stays total: {len(t)} | with >=1 HARD intervention: {(t.n_types>0).sum()} "
          f"({(t.n_types>0).mean()*100:.0f}%)\n")
    print("per-disease % of stays with a HARD intervention:")
    g = t.groupby("disease").agg(n=("hadm", "size"), iv=("n_types", lambda s: (s > 0).sum()))
    g["pct"] = (g.iv / g.n * 100).round(0)
    print(g.to_string(), "\n")
    print("intervention-type prevalence (stays):")
    from collections import Counter
    c = Counter(x for row in t.types for x in row.split("|") if x)
    for typ, n in c.most_common():
        print(f"  {n:4}  {typ}")
    # spot-check known cases from the raw-text audit. Two mechanisms:
    #   "proc" = this-admission HARD intervention -> caught by procedure classifier
    #   "text" = OLD surgery / stale anatomy (not in this stay's procedures) -> text hint only
    print("\nspot-check (mechanism-aware):")
    cases = [("appendicitis", 27993727, "proc", "appendectomy"),
             ("pancreatitis", 21849575, "proc", "cholecystectomy"),
             ("pancreatitis", 28684468, "proc", "ercp_therapeutic"),
             ("pancreatitis", 21061497, "text", "")]  # old cholecystectomy, none this stay
    for d, h, mech, exp in cases:
        r = _raw_df(d)[_raw_df(d).hadm_id == h].iloc[0]
        got = "|".join(sorted(interventions_from_titles(r))) or "(none)"
        rep = _note_ids_and_reports(d, h)
        any_hint = any(_text_hint(t[1]) for t in rep.values())
        if mech == "proc":
            ok = exp in got
            print(f"  {'OK ' if ok else 'XX '} {d} {h} [proc]: got [{got}]  expect [{exp}]")
        else:
            ok = got == "(none)" and any_hint
            print(f"  {'OK ' if ok else 'XX '} {d} {h} [text]: proc=[{got}] (want none), text_hint={any_hint}")


def build(source_dir: str | None) -> None:
    src = None
    if source_dir:
        src = SourceTables.load(source_dir)
        print(f"[source] loaded admissions/radiology/procedures from {source_dir}")
    else:
        print("[degraded] no --source-dir: timing_role=unknown; "
              "interventions from titles + text hint only")

    dev = pd.read_csv(DEV_CSV) if DEV_CSV.exists() else None
    dev_idx = {}
    if dev is not None:
        for _, r in dev.iterrows():
            dev_idx[(r.disease, int(r.hadm), int(r.step))] = (r.dev_belief, r.verification)

    rows = []
    for jf in sorted(ANN_DIR.glob("*.json")):
        d = json.loads(jf.read_text())
        if not (isinstance(d, dict) and "hadm_id" in d and "steps" in d):
            continue  # skip manifest.json / other aggregates
        disease, hadm = d["disease"], int(d["hadm_id"])
        nid_map = _note_ids_and_reports(disease, hadm)
        raw_row = _raw_df(disease)
        raw_row = raw_row[raw_row.hadm_id == hadm].iloc[0]
        ivs = interventions_from_titles(raw_row)

        admit = src.admittime(hadm) if src else None
        fdate, ftype = (src.first_intervention(hadm) if src else (None, None))
        # cross-check title-derived intervention if source disagrees on existence
        if ftype is None and ivs:
            ftype = sorted(ivs)[0]  # type known from titles even if its date is unknown
        ct_by_note = src.charttime_by_note(hadm) if src else {}

        for step in d["steps"]:
            i = step["step"]
            rrn = d["radiology_order"][i - 1]["rrn"] if i - 1 < len(d["radiology_order"]) else None
            note_id, report = nid_map.get(rrn, ("", ""))
            ct = ct_by_note.get(note_id)
            role = timing_role(ct, admit, fdate) if src else UNKNOWN
            hint = _text_hint(report)
            devb, vind = dev_idx.get((disease, hadm, i), ("", ""))
            # exclusion decision: real role if we have it, else fall back to text hint
            exclude = (role in MONITORING_ROLES) if src else bool(hint)
            rows.append({
                "disease": disease, "hadm": hadm, "step": i, "rrn": rrn, "note_id": note_id,
                "modality": step["ordered"].split(" (")[0],
                "dev_belief": devb, "verification": vind,
                "interventions": "|".join(sorted(ivs)),
                "first_intervention_type": ftype or "",
                "charttime": ct, "admittime": admit, "first_intervention_date": fdate,
                "timing_role": role,
                "text_post_intervention_hint": hint,
                "exclude_from_deviation": exclude,
            })

    out = pd.DataFrame(rows)
    dst = ANN_DIR / "timing_roles.csv"
    out.to_csv(dst, index=False)
    print(f"\nwrote {len(out)} decision steps -> {dst}")
    if src:
        print("\ntiming_role distribution:")
        print(out.timing_role.value_counts().to_string())
        print(f"\nexcluded as monitoring: {out.exclude_from_deviation.sum()} "
              f"({out.exclude_from_deviation.mean()*100:.0f}%)")
        print("\nexcluded steps by dev_belief x verification:")
        ex = out[out.exclude_from_deviation]
        if len(ex):
            print(pd.crosstab(ex.dev_belief, ex.verification, margins=True).to_string())
    else:
        print(f"\ntext-hint flagged (provisional exclude): {out.exclude_from_deviation.sum()} "
              f"({out.exclude_from_deviation.mean()*100:.0f}%)  "
              "-- replace with real timing_role once source CSVs land")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", help="dir with admissions.csv / radiology.csv / procedures_icd.csv (+.gz)")
    ap.add_argument("--self-test", action="store_true", help="validate classifier on derived data only")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    else:
        build(a.source_dir)


if __name__ == "__main__":
    main()
