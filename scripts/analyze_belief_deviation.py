"""Step-by-step belief-conditioned deviation analysis (TRAVERSAL mechanism).

Replaces the set-membership labelling in analyze_off_rubric.py. Per decision step:

  dev_belief  : {follow, deviate, off_rubric} from belief_step_deviation — feed the
                step's REAL causally-masked pre-decision features into the belief
                (top_branch) sub-rubric graph, read the imaging the rubric wants HERE,
                compare to the doctor's action. (commission/omission/swap are NOT here.)
  dev_godview : event-aligned commission vs the patient's true-disease rubric path
                (tsc simulated_sequence) — SEQUENCE-level, unchanged.
  vindication : confirmed / disconfirmed / uninformative (ex-post, from annotation).

Pre-decision features come from data/rubric_features/{disease}_features.json,
which is ACCUMULATIVE (idx_k features already include test-k's result), so the state
before the doctor ordered the step-k imaging = idx_{k-1}.

RR-N caveat (features_extraction_rrn_order_bug memory): rubric_features steps are
in RAW MIMIC record order, the annotation decision steps are in RR-N order. When the two
imaging orders disagree there is no clean accumulative pre-state to feed, so those
patients are marked rrn_aligned=False and their steps are SKIPPED (dev_belief blank).

Usage: /opt/anaconda3/bin/python3.12 scripts/analyze_belief_deviation.py [ann_dir]
       default ann_dir = results/annotation_experiment/full
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
    IMAGING_KEYS, belief_step_deviation, modality_of, godview_step_flags, rubric_state,
)

TSC = ROOT / "results" / "test_seq_comparison" / "test_sequence_comparison.csv"
DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]


def _imaging_seq(seq: str) -> list[str]:
    return [t.strip() for t in str(seq).split(",") if t.strip() in IMAGING_KEYS]


def _load_features() -> dict[str, dict]:
    out = {}
    for d in DISEASES:
        out[d] = json.load(open(ROOT / "data" / "rubric_features" /
                                f"{d}_features.json"))["results"]
    return out


def _fe_for(fe_disease: dict, hadm: int) -> list | None:
    return fe_disease.get(str(hadm)) or fe_disease.get(hadm)


def main() -> None:
    ann_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "results" / "annotation_experiment" / "full"
    tsc = pd.read_csv(TSC).set_index("patient_id")
    FE = _load_features()
    files = sorted(p for p in ann_dir.glob("*.json") if p.name != "manifest.json")
    if not files:
        print(f"no case json in {ann_dir}"); return

    rows = []
    n_misaligned = 0
    for path in files:
        disease, hadm_s = path.stem.rsplit("_", 1)
        hadm = int(hadm_s)
        steps = json.loads(path.read_text())["steps"]
        ordered_seq = [modality_of(st["ordered"]) for st in steps]

        # god-view (sequence-level, unchanged): doctor imaging vs tsc rubric path
        rubric_img = _imaging_seq(tsc.loc[hadm, "simulated_sequence"]) if hadm in tsc.index else []
        gv_flags, _ = godview_step_flags(rubric_img, ordered_seq)

        # align annotation decision imaging (RR-N order) to rubric_features (raw order)
        fe_steps = _fe_for(FE[disease], hadm)
        fe_img_idx = [i for i, s in enumerate(fe_steps or []) if s["test_key"] in IMAGING_KEYS]
        fe_img_seq = [fe_steps[i]["test_key"] for i in fe_img_idx] if fe_steps else []
        aligned = fe_steps is not None and fe_img_seq == ordered_seq
        if not aligned:
            n_misaligned += 1

        for i, st in enumerate(steps):
            m = st["metrics"]
            top = m["overall_top_branch"]
            oh = (st.get("representative_ex_ante") or {}).get("other_hypothesis", "")
            vind = (st.get("vindication") or {}).get("vindication", "")
            row = {
                "disease": disease, "hadm": hadm, "step": st["step"],
                "modality": ordered_seq[i], "top_branch": top,
                "mean_other": round(m["mean_differential"].get("other", 0.0), 3),
                "rrn_aligned": aligned,
                "dev_godview": gv_flags[i],
                "vindication": vind,
            }
            if aligned:
                pre = fe_steps[fe_img_idx[i] - 1]["features"]
                eff, dev, rec = belief_step_deviation(top, pre, st["ordered"], oh)
                row.update({"eff_branch": eff, "rubric_rec": "|".join(rec) or "-",
                            "rubric_state": rubric_state(eff, pre), "dev_belief": dev})
            else:
                row.update({"eff_branch": "", "rubric_rec": "",
                            "rubric_state": "", "dev_belief": ""})
            rows.append(row)

    cols = ["disease", "hadm", "step", "modality", "top_branch", "eff_branch",
            "mean_other", "rubric_rec", "rubric_state", "dev_belief", "dev_godview",
            "vindication", "rrn_aligned"]
    out_csv = ann_dir / "belief_deviation_analysis.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)

    n_cases = len(files)
    n_steps = len(rows)
    judged = [r for r in rows if r["rrn_aligned"]]
    print(f"cases={n_cases}  steps={n_steps}  "
          f"rrn_misaligned cases (skipped)={n_misaligned}  judged steps={len(judged)}\n")

    print(f"{'disease':16} {'steps':>5} {'follow':>7} {'deviate':>8} {'off_rub':>8}")
    by_d = defaultdict(list)
    for r in judged:
        by_d[r["disease"]].append(r)
    for d in DISEASES:
        sub = by_d.get(d, [])
        if not sub:
            continue
        c = Counter(r["dev_belief"] for r in sub)
        print(f"{d:16} {len(sub):5} {c.get('follow',0):7} "
              f"{c.get('deviate',0):8} {c.get('off_rubric',0):8}")
    overall = Counter(r["dev_belief"] for r in judged)
    print(f"\nOVERALL dev_belief (judged): {dict(overall)}")

    # vindication x dev_belief crosstab — the analysis targets (Part B)
    print("\ndev_belief x vindication:")
    print(f"{'':18}{'confirmed':>10}{'disconfirmed':>13}{'uninform':>10}")
    for dev in ["follow", "deviate", "off_rubric"]:
        sub = [r for r in judged if r["dev_belief"] == dev]
        vc = Counter(r["vindication"] for r in sub)
        print(f"{dev:18}{vc.get('confirmed',0):10}{vc.get('disconfirmed',0):13}"
              f"{vc.get('uninformative',0):10}")

    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
