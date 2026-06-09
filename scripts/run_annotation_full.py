"""Annotate ALL 300 patients in joined_data with the 思路2 Mode-A + 思路1 vindication
pipeline (5-branch + open 'other', n=1 — biliary recovered POST-HOC offline).

Resume-safe: skips any (disease,hadm) that already has a JSON in --outdir, so it can be
re-launched after an interruption and only annotates what is missing. To avoid re-spending
on the 30 already-done unbiased batch cases, run with --seed-from results/annotation_experiment/batch
once: those JSONs are produced by the IDENTICAL annotate_case(n=1) call and are copied in.

The off_rubric / dev_belief / dev_godview analysis stays a SEPARATE offline step
(scripts/analyze_off_rubric.py <outdir>) so it never re-spends API.

Usage:
  /opt/anaconda3/bin/python3.12 scripts/run_annotation_full.py \
      --seed-from results/annotation_experiment/batch
"""
import argparse
import json
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from build_masked_view import RAW, load_lab_map, build_record  # noqa: E402
from experiments.annotation.annotate import annotate_case  # noqa: E402

JOINED = ROOT / "results" / "gap_analysis" / "joined_data.csv"
OUTDIR = ROOT / "results" / "annotation_experiment" / "full"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--seed-from", default=None,
                    help="dir of already-annotated JSONs (same pipeline/n=1) to copy in "
                         "before running, so those cases are skipped instead of re-spent.")
    ap.add_argument("--limit", type=int, default=None,
                    help="annotate at most this many NEW cases this run (for cost control).")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent cases (healthy calls are ~1s; congested providers "
                         "stall, so overlap them). Steps within a case stay serial.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.seed_from:
        src = Path(args.seed_from)
        copied = 0
        for f in src.glob("*.json"):
            if f.name == "manifest.json":
                continue
            dst = outdir / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                copied += 1
        print(f"seeded {copied} existing JSONs from {src} -> {outdir}", flush=True)

    labmap = load_lab_map()
    jd = pd.read_csv(JOINED)
    jd = jd.sort_values(["disease", "patient_id"]).reset_index(drop=True)
    total = len(jd)

    # Pre-load each disease's raw CSV once (was re-read per patient = 300x).
    raw_by_disease = {d: pd.read_csv(p).set_index("hadm_id") for d, p in RAW.items()}

    # Build the worklist locally (no API): skip existing JSONs and no-decision-step cases.
    manifest = []
    pending = []  # list of (disease, hadm, record)
    skipped_existing = skipped_nosteps = 0
    for _, row in jd.iterrows():
        disease, hadm = row["disease"], int(row["patient_id"])
        if (outdir / f"{disease}_{hadm}.json").exists():
            skipped_existing += 1
            manifest.append({"disease": disease, "hadm": hadm, "status": "existing"})
            continue
        raw = raw_by_disease[disease]
        if hadm not in raw.index:
            print(f"  MISSING raw {disease} {hadm}", flush=True)
            continue
        record = build_record(disease, hadm, raw.loc[hadm], labmap)
        if record["n_decision_steps"] == 0:
            skipped_nosteps += 1
            manifest.append({"disease": disease, "hadm": hadm, "status": "no_decision_step"})
            continue
        pending.append((disease, hadm, record))

    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"worklist: {len(pending)} to annotate, {skipped_existing} existing, "
          f"{skipped_nosteps} no-step  (workers={args.workers})", flush=True)

    lock = threading.Lock()
    counter = {"n": 0}

    def work(item):
        disease, hadm, record = item
        result = annotate_case(record, model=args.model, n_samples=1)
        (outdir / f"{disease}_{hadm}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2))
        with lock:
            counter["n"] += 1
            print(f"  [{counter['n']}/{len(pending)}] {disease} {hadm}  "
                  f"{record['n_decision_steps']} steps", flush=True)
        return {"disease": disease, "hadm": hadm,
                "n_decision_steps": record["n_decision_steps"], "status": "new"}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(work, it): it for it in pending}
        for fut in as_completed(futures):
            try:
                manifest.append(fut.result())
            except Exception as e:  # one bad case must not abort the rest (resume-safe)
                d, h, _ = futures[fut]
                print(f"  FAILED {d} {h}: {e!r}", flush=True)

    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    newly = sum(1 for m in manifest if m.get("status") == "new")
    print(f"\nDONE over {total} patients: {newly} newly annotated, "
          f"{skipped_existing} already present, {skipped_nosteps} had no decision step "
          f"-> {outdir}", flush=True)


if __name__ == "__main__":
    main()
