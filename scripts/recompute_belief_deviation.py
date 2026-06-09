"""Offline recompute of deviation labels over already-annotated cases (no API).

For each decision step of results/annotation_experiment/<disease>_<hadm>.json emits
TWO deviation columns (neither replaces the other — HANDOFF §1.5):

  dev_belief  : vs the sub-rubric of the disease the doctor ASSUMED at that step
                (argmax of reconstructed differential). {follow, deviate_commission,
                off_rubric}. This is what the certainty-trigger AGENT must learn.

  dev_godview : vs the patient's OWN rubric path for the TRUE (MIMIC) disease, taken
                from results/test_seq_comparison (simulated_sequence) and aligned
                event-by-event via multiset consumption (deviation.godview_step_flags),
                which fixes the §7 todo-1 "every repeat CT marked dev" over-counting.

Usage: /opt/anaconda3/bin/python3.12 scripts/recompute_belief_deviation.py
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiments.annotation.deviation import (  # noqa: E402
    IMAGING_KEYS, belief_deviation, modality_of, recommended_imaging,
    episode_omissions, godview_step_flags,
)
from scripts.run_annotation_experiment import CASES  # noqa: E402

ANN_DIR = ROOT / "results" / "annotation_experiment"
OUT_CSV = ANN_DIR / "summary_belief_deviation.csv"
TSC = ROOT / "results" / "test_seq_comparison" / "test_sequence_comparison.csv"


def _imaging_seq(seq: str) -> list[str]:
    return [t.strip() for t in str(seq).split(",") if t.strip() in IMAGING_KEYS]


def main() -> None:
    print("R[D] (programmatic from DISEASE_GRAPHS):")
    for d, r in recommended_imaging().items():
        print(f"  {d:16} {sorted(r)}")
    print()

    tsc = pd.read_csv(TSC).set_index("patient_id")
    rows = []
    for disease, hadm, label, _commission in CASES:
        path = ANN_DIR / f"{disease}_{hadm}.json"
        if not path.exists():
            print(f"!! missing {path.name}"); continue
        case = json.loads(path.read_text())
        steps = case["steps"]
        ordered_seq = [modality_of(st["ordered"]) for st in steps]

        # god-view: event-aligned vs this patient's rubric imaging path
        rubric_img = _imaging_seq(tsc.loc[hadm, "simulated_sequence"]) if hadm in tsc.index else []
        gv_flags, gv_omit = godview_step_flags(rubric_img, ordered_seq)

        # belief-view: vs assumed-disease sub-rubric
        final_top = steps[-1]["metrics"]["overall_top_branch"] if steps else None
        b_omit = episode_omissions(final_top, set(ordered_seq))

        print(f"== {disease} {hadm} ({label})  final_belief={final_top}")
        print(f"   rubric_imaging={rubric_img or '-'}  godview_omission={sorted(gv_omit) or '-'}  "
              f"belief_omission={sorted(b_omit) or '-'} ==")
        for i, st in enumerate(steps):
            m = st["metrics"]
            top = m["overall_top_branch"]
            mod = ordered_seq[i]
            dev_b = belief_deviation(top, st["ordered"])
            dev_g = gv_flags[i]
            vind = (st.get("vindication") or {}).get("vindication", "")
            diverge = (dev_b == "deviate_commission") != dev_g and dev_b != "off_rubric"
            flag = "  <-- DIVERGE" if diverge else ""
            rows.append({
                "disease": disease, "hadm": hadm, "case_label": label,
                "step": st["step"], "modality": mod, "top_branch": top,
                "mean_other": round(m["mean_differential"].get("other", 0.0), 3),
                "dev_belief": dev_b, "dev_godview": dev_g, "vindication": vind,
            })
            print(f"   step{st['step']} {mod:18} belief_top={top:14} "
                  f"dev_belief={dev_b:18} dev_godview={dev_g!s:5} vind={vind}{flag}")
        print()

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    bc = Counter(r["dev_belief"] for r in rows)
    print("=" * 70)
    print(f"steps total: {len(rows)}")
    print(f"dev_belief dist : {dict(bc)}")
    print(f"dev_godview True: {sum(r['dev_godview'] for r in rows)}")
    # cross-tab belief x godview
    print("\nbelief x godview crosstab:")
    for b in ["follow", "deviate_commission", "off_rubric"]:
        sub = [r for r in rows if r["dev_belief"] == b]
        if sub:
            gv = sum(r["dev_godview"] for r in sub)
            print(f"  {b:18} n={len(sub):2}  godview-commission={gv:2}  godview-follow={len(sub)-gv}")
    # vindication split for the two deviation-ish belief classes (point 1: vindication judges quality)
    for b in ["off_rubric", "deviate_commission"]:
        sub = [r for r in rows if r["dev_belief"] == b]
        if sub:
            print(f"\n{b} vindication: {dict(Counter(r['vindication'] for r in sub))}")
    print(f"\nwrote {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
