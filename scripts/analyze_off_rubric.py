"""Offline deviation / off_rubric analysis over a directory of annotated cases.

Scans <dir>/<disease>_<hadm>.json (no API), and per decision step computes:
  dev_belief  : {follow, deviate_commission, off_rubric} vs assumed-disease sub-rubric
  dev_godview : event-aligned commission vs the patient's true-disease rubric path (tsc)
  verification : confirmed / disconfirmed / uninformative

Aggregates the population off_rubric rate overall and PER DISEASE (to test the
pilot hypothesis: off_rubric concentrates in cholecystitis + pancreatitis, the
hepatobiliary crossroads, and is rare in appendicitis + diverticulitis).

Usage: /opt/anaconda3/bin/python3.12 scripts/analyze_off_rubric.py [dir]
       default dir = results/annotation_experiment/batch
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiments.annotation.deviation import (  # noqa: E402
    IMAGING_KEYS, derived_belief, modality_of, godview_step_flags,
)

TSC = ROOT / "results" / "test_seq_comparison" / "test_sequence_comparison.csv"


def _imaging_seq(seq: str) -> list[str]:
    return [t.strip() for t in str(seq).split(",") if t.strip() in IMAGING_KEYS]


def main() -> None:
    ann_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "results" / "annotation_experiment" / "batch"
    tsc = pd.read_csv(TSC).set_index("patient_id")
    files = sorted(p for p in ann_dir.glob("*.json") if p.name != "manifest.json")
    if not files:
        print(f"no case json in {ann_dir}"); return

    rows = []
    for path in files:
        disease, hadm = path.stem.rsplit("_", 1)
        hadm = int(hadm)
        steps = json.loads(path.read_text())["steps"]
        ordered_seq = [modality_of(st["ordered"]) for st in steps]
        rubric_img = _imaging_seq(tsc.loc[hadm, "simulated_sequence"]) if hadm in tsc.index else []
        gv_flags, _ = godview_step_flags(rubric_img, ordered_seq)
        for i, st in enumerate(steps):
            m = st["metrics"]
            top = m["overall_top_branch"]
            oh = (st.get("representative_ex_ante") or {}).get("other_hypothesis", "")
            eff, dev = derived_belief(top, st["ordered"], oh)
            rows.append({
                "disease": disease, "hadm": hadm, "step": st["step"],
                "modality": ordered_seq[i], "top_branch": top, "eff_branch": eff,
                "mean_other": round(m["mean_differential"].get("other", 0.0), 3),
                "dev_belief": dev,
                "dev_godview": gv_flags[i],
                "verification": (st.get("verification") or {}).get("verification", ""),
            })

    out_csv = ann_dir / "off_rubric_analysis.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    n_cases = len(files)
    n_steps = len(rows)
    print(f"cases={n_cases}  decision steps={n_steps}\n")

    # per-disease off_rubric rate
    by_d = defaultdict(list)
    for r in rows:
        by_d[r["disease"]].append(r)
    print(f"{'disease':16} {'steps':>5} {'off_rubric':>10} {'rate':>6} {'follow':>7} {'dev_comm':>8}")
    for d in ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]:
        sub = by_d.get(d, [])
        if not sub:
            continue
        c = Counter(r["dev_belief"] for r in sub)
        off = c.get("off_rubric", 0)
        print(f"{d:16} {len(sub):5} {off:10} {off/len(sub):6.0%} "
              f"{c.get('follow',0):7} {c.get('deviate_commission',0):8}")
    overall = Counter(r["dev_belief"] for r in rows)
    off_all = overall.get("off_rubric", 0)
    print(f"\nOVERALL dev_belief: {dict(overall)}")
    print(f"population off_rubric rate (step-level): {off_all}/{n_steps} = {off_all/n_steps:.0%}")
    # case-level: fraction of cases whose FINAL belief is off-rubric
    final_off = 0
    for path in files:
        steps = json.loads(path.read_text())["steps"]
        if steps and steps[-1]["metrics"]["overall_top_branch"] == "other":
            final_off += 1
    print(f"case-level final-belief off_rubric: {final_off}/{n_cases} = {final_off/n_cases:.0%}")

    # verification split of off_rubric (quality: is off-rubric uniformly bad?)
    off_rows = [r for r in rows if r["dev_belief"] == "off_rubric"]
    if off_rows:
        print(f"\noff_rubric verification: {dict(Counter(r['verification'] for r in off_rows))}")

    # biliary post-hoc recovery: off_rubric steps whose other_hypothesis named a bile-duct
    # process, relabeled 'biliary' and judged vs R[biliary]={US,MRCP}. (eff_branch, not the
    # Mode-A top_branch, which stays 'other'.)
    bil = [r for r in rows if r["eff_branch"] == "biliary"]
    if bil:
        print(f"\nbiliary (post-hoc from off_rubric): {len(bil)} steps  "
              f"dev_belief={dict(Counter(r['dev_belief'] for r in bil))}  "
              f"verification={dict(Counter(r['verification'] for r in bil))}")
        for r in bil:
            print(f"    {r['disease']} {r['hadm']} s{r['step']} {r['modality']} "
                  f"-> {r['dev_belief']} ({r['verification']})")
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
