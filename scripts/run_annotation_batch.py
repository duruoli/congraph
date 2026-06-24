"""Stratified-random batch annotation to estimate the population off_rubric rate.

Runs the FULL updated pipeline per case (思路2 Mode-A reconstruction at n=1 +
思路1 verification per decision step — same annotate_case as the pilot, just n=1),
on a seeded stratified-random sample of patients, EXCLUDING the 8 hand-picked pilot
cases (those were deliberately biased toward complex/deviation patients, so they
would over-state off_rubric). Outputs one JSON per case to results/annotation_experiment/batch/.

The off_rubric / dev_belief / dev_godview analysis is a SEPARATE offline step
(scripts/analyze_off_rubric.py) so it can be re-run without spending API.

Usage: /opt/anaconda3/bin/python3.12 scripts/run_annotation_batch.py [--seed 0]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from build_masked_view import RAW, load_lab_map, build_record  # noqa: E402
from experiments.annotation.annotate import annotate_case  # noqa: E402

TSC = ROOT / "results" / "test_seq_comparison" / "test_sequence_comparison.csv"
OUTDIR = ROOT / "results" / "annotation_experiment" / "batch"

PILOT = {29573603, 29060924, 23202997, 26877856, 26371704, 27675389, 21282967, 21061497}
# target cases per disease (sums to 30); appe/diver narrower so slightly fewer
TARGET = {"appendicitis": 7, "cholecystitis": 8, "diverticulitis": 7, "pancreatitis": 8}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    ap.add_argument("--outdir", default=str(OUTDIR),
                    help="output dir (default results/annotation_experiment/batch)")
    ap.add_argument("--cases-from", default=None,
                    help="path to an existing manifest.json; annotate EXACTLY those "
                         "(disease,hadm) cases instead of stratified sampling (for a "
                         "clean A/B re-run under a changed schema).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    labmap = load_lab_map()

    # --cases-from: pin the exact case list (no sampling) for an apples-to-apples re-run
    if args.cases_from:
        pinned = json.loads(Path(args.cases_from).read_text())
        manifest = []
        print(f"#### pinned re-run: {len(pinned)} cases -> {outdir} ####", flush=True)
        for m in pinned:
            disease, hadm = m["disease"], int(m["hadm"])
            raw = pd.read_csv(RAW[disease])
            row = raw[raw["hadm_id"] == hadm]
            if row.empty:
                print(f"  MISSING {disease} {hadm}"); continue
            record = build_record(disease, hadm, row.iloc[0], labmap)
            result = annotate_case(record, model=args.model, n_samples=1)
            (outdir / f"{disease}_{hadm}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2))
            manifest.append({"disease": disease, "hadm": hadm,
                             "n_decision_steps": record["n_decision_steps"]})
            print(f"  [{len(manifest)}/{len(pinned)}] {disease} {hadm}  "
                  f"{record['n_decision_steps']} steps", flush=True)
        (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        total_steps = sum(x["n_decision_steps"] for x in manifest)
        print(f"\nDONE: {len(manifest)} cases, {total_steps} steps -> {outdir}")
        return

    tsc = pd.read_csv(TSC)
    manifest = []

    for disease, target in TARGET.items():
        raw = pd.read_csv(RAW[disease])
        pool = (tsc[tsc["disease"] == disease]["patient_id"]
                .loc[lambda s: ~s.isin(PILOT)]
                .sample(frac=1.0, random_state=args.seed).tolist())
        kept = 0
        print(f"\n#### {disease}: target {target}, pool {len(pool)} ####", flush=True)
        for hadm in pool:
            if kept >= target:
                break
            row = raw[raw["hadm_id"] == hadm]
            if row.empty:
                continue
            record = build_record(disease, hadm, row.iloc[0], labmap)
            if record["n_decision_steps"] == 0:
                continue  # no abdominal imaging decision step -> nothing to label
            result = annotate_case(record, model=args.model, n_samples=1)
            (outdir / f"{disease}_{hadm}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2))
            kept += 1
            manifest.append({"disease": disease, "hadm": int(hadm),
                             "n_decision_steps": record["n_decision_steps"]})
            print(f"  [{kept}/{target}] {hadm}  {record['n_decision_steps']} steps", flush=True)

    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    total_steps = sum(m["n_decision_steps"] for m in manifest)
    print(f"\nDONE: {len(manifest)} cases, {total_steps} decision steps -> {outdir}")


if __name__ == "__main__":
    main()
