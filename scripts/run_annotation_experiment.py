"""6-8 case pilot for the doctor-reasoning annotation pipeline (HANDOFF §4).

For each case: build the causally-masked decision sequence (build_masked_view),
run 思路2 Mode-A ensemble + 思路1 verification (experiments.annotation.annotate),
then measure the four pilot questions:
  (1) belief-trajectory coherence  -> per-step mean differential dumped for inspection
  (2) ex-ante vs ex-post agreement -> verification label per step
  (3) triage-artifact share         -> 'other' mass + differential dispersion on deviation steps
  (4) disagreement tracks deviation -> ensemble diff_disagreement on deviating vs non-deviating steps

Usage: python3 scripts/run_annotation_experiment.py [--n-samples 5] [--model ...]
Outputs to results/annotation_experiment/.
"""
import sys, os, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_masked_view import RAW, load_lab_map, build_record  # noqa: E402
import pandas as pd  # noqa: E402
from experiments.annotation.annotate import annotate_case  # noqa: E402

# (disease, hadm, case_label, commission_modalities) — selected 4 diseases x dev/non-dev
CASES = [
    ("cholecystitis", 29573603, "deviation",     {"CT_Abdomen", "MRCP_Abdomen"}),
    ("cholecystitis", 29060924, "non_deviation", set()),
    ("appendicitis",  23202997, "deviation",     {"CT_Abdomen", "Ultrasound_Abdomen"}),
    ("appendicitis",  26877856, "non_deviation", set()),
    ("diverticulitis", 26371704, "deviation",    {"CT_Abdomen", "Ultrasound_Abdomen"}),
    ("diverticulitis", 27675389, "non_deviation", set()),
    ("pancreatitis",  21282967, "deviation",     {"MRI_Abdomen"}),
    ("pancreatitis",  21061497, "non_deviation", set()),
]

OUTDIR = "results/annotation_experiment"


def step_modality(ordered: str) -> str:
    # ordered = "CT Abdomen (CT ABD & PELVIS ...)" -> "CT_Abdomen"
    head = ordered.split("(")[0].strip()
    return "_".join(head.split()[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    ap.add_argument("--cases", type=int, default=len(CASES), help="run only first N cases")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    labmap = load_lab_map()
    raws: dict[str, pd.DataFrame] = {}
    rows = []

    for disease, hadm, label, commission in CASES[: args.cases]:
        if disease not in raws:
            raws[disease] = pd.read_csv(RAW[disease])
        row = raws[disease][raws[disease]["hadm_id"] == hadm]
        if row.empty:
            print(f"!! {disease} {hadm} not found"); continue
        record = build_record(disease, hadm, row.iloc[0], labmap)
        print(f"== {disease} {hadm} ({label}) : {record['n_decision_steps']} decision steps ==")
        result = annotate_case(record, model=args.model, n_samples=args.n_samples)

        with open(f"{OUTDIR}/{disease}_{hadm}.json", "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        for st in result["steps"]:
            m = st["metrics"]
            mod = step_modality(st["ordered"])
            is_dev_step = mod in commission
            vind = st.get("verification") or {}
            rows.append({
                "disease": disease, "hadm": hadm, "case_label": label,
                "step": st["step"], "ordered": st["ordered"], "modality": mod,
                "is_deviating_step": is_dev_step,
                "top_branch": m["overall_top_branch"],
                "mean_other": round(m["mean_differential"].get("other", 0.0), 3),
                "diff_disagreement": round(m["diff_disagreement"], 4),
                "top_branch_consistency": round(m["top_branch_consistency"], 3),
                "action_role": m["modal_action_role"],
                "verification": vind.get("verification", ""),
                "certainty_update": vind.get("certainty_update", ""),
            })
            print(f"   step{st['step']} {mod:16} dev={is_dev_step!s:5} "
                  f"top={m['overall_top_branch']:13} other={rows[-1]['mean_other']:.2f} "
                  f"disagree={rows[-1]['diff_disagreement']:.3f} "
                  f"vind={rows[-1]['verification']}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTDIR}/summary_steps.csv", index=False)

    if df.empty:
        print("no rows"); return

    print("\n" + "=" * 70)
    print("AGGREGATE (pilot questions)")
    print("=" * 70)
    # Q4: disagreement on deviating vs non-deviating decision steps
    for grp, name in [(df[df["is_deviating_step"]], "deviating steps"),
                      (df[~df["is_deviating_step"]], "non-deviating steps")]:
        if not grp.empty:
            print(f"[Q4] {name:22} n={len(grp):2}  "
                  f"mean disagreement={grp['diff_disagreement'].mean():.3f}  "
                  f"mean top-consistency={grp['top_branch_consistency'].mean():.3f}  "
                  f"mean 'other'={grp['mean_other'].mean():.3f}")
    # Q3: triage-artifact proxy = 'other' mass on deviating steps
    dev = df[df["is_deviating_step"]]
    if not dev.empty:
        print(f"[Q3] deviating steps with mean 'other' > 0.25 (triage net): "
              f"{(dev['mean_other'] > 0.25).sum()}/{len(dev)}")
    # Q2: verification distribution
    print(f"[Q2] verification dist: {df['verification'].value_counts().to_dict()}")
    print(f"\nwrote {OUTDIR}/summary_steps.csv  ({len(df)} decision steps)")


if __name__ == "__main__":
    main()
