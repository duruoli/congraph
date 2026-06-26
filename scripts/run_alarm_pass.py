"""Run the ALARM pass (HANDOFF §2.5) over the existing Mode-A annotations.

For each (disease,hadm) that already has results/annotation_experiment/full/<d>_<h>.json,
rebuild the causally-masked view and run the two ex-ante alarm calls per decision step:
  step 1 alarm_detect  (blind to the order/reasoning)
  step 2 alarm_resolve (did the order resolve the detected flag, ex-ante)
-> results/annotation_experiment/alarm/<disease>_<hadm>.json

Resume-safe (skips existing alarm JSONs) + case-concurrent, like run_annotation_full.

Usage:
  /opt/anaconda3/bin/python3.12 scripts/run_alarm_pass.py            # all annotated cases
  /opt/anaconda3/bin/python3.12 scripts/run_alarm_pass.py --limit 1  # smoke test
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_masked_view import RAW, load_lab_map, build_record  # noqa: E402
from experiments.annotation.alarm import annotate_alarm_case  # noqa: E402
import pandas as pd  # noqa: E402

ANN_DIR = ROOT / "results" / "annotation_experiment" / "full"
OUTDIR = ROOT / "results" / "annotation_experiment" / "alarm"


def _has_error(out_file) -> bool:
    """True if a previously-written alarm JSON contains any parse/resolve error step,
    so a re-run redoes it instead of skipping (self-healing)."""
    try:
        d = json.loads(out_file.read_text())
    except Exception:
        return True
    for s in d.get("steps", []):
        if (s.get("alarm_detect") or {}).get("_parse_error"):
            return True
        if (s.get("alarm_resolve") or {}).get("addresses_alarm") == "error":
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    ap.add_argument("--ann-dir", default=str(ANN_DIR))
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most this many NEW cases (cost control / smoke test).")
    ap.add_argument("--cases", default=None,
                    help="comma-separated <disease>_<hadm> stems to restrict to "
                         "(e.g. 'pancreatitis_21282967,cholecystitis_29060924').")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    only = set(c.strip() for c in args.cases.split(",")) if args.cases else None

    ann_dir = Path(args.ann_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    labmap = load_lab_map()
    raw_by_disease = {d: pd.read_csv(p).set_index("hadm_id") for d, p in RAW.items()}

    pending = []  # (disease, hadm, record, annotation)
    skipped_existing = 0
    for f in sorted(ann_dir.glob("*.json")):
        if f.name == "manifest.json":
            continue
        if only is not None and f.stem not in only:
            continue
        out_f = outdir / f.name
        if out_f.exists() and not _has_error(out_f):
            skipped_existing += 1
            continue
        try:
            disease, hadm = f.stem.rsplit("_", 1)
            hadm = int(hadm)
        except ValueError:
            continue
        annotation = json.loads(f.read_text())
        if not annotation.get("steps"):
            continue
        raw = raw_by_disease.get(disease)
        if raw is None or hadm not in raw.index:
            print(f"  MISSING raw {disease} {hadm}", flush=True)
            continue
        record = build_record(disease, hadm, raw.loc[hadm], labmap)
        if record["n_decision_steps"] == 0:
            continue
        pending.append((disease, hadm, record, annotation))

    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"worklist: {len(pending)} to alarm-annotate, {skipped_existing} existing "
          f"(workers={args.workers})", flush=True)

    lock = threading.Lock()
    counter = {"n": 0}

    def work(item):
        disease, hadm, record, annotation = item
        result = annotate_alarm_case(record, annotation, model=args.model)
        (outdir / f"{disease}_{hadm}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2))
        with lock:
            counter["n"] += 1
            print(f"  [{counter['n']}/{len(pending)}] {disease} {hadm}  "
                  f"{record['n_decision_steps']} steps", flush=True)
        return {"disease": disease, "hadm": hadm, "status": "new"}

    newly = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(work, it): it for it in pending}
        for fut in as_completed(futures):
            try:
                fut.result(); newly += 1
            except Exception as e:
                d, h, _, _ = futures[fut]
                print(f"  FAILED {d} {h}: {e!r}", flush=True)

    # Manifest = EVERY case in the dir (test batches + new alike), so nothing is listed
    # separately. Records step count + how many steps fired a trigger / errored.
    manifest, n_err = [], 0
    for f in sorted(outdir.glob("*.json")):
        if f.name == "manifest.json":
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        steps = d.get("steps", [])
        triggered = sum(1 for s in steps
                        if any((s.get("alarm_detect") or {}).get(k, {}).get("present") in (True, "true", "yes")
                               for k in ("study_inadequacy", "discordance")))
        errs = sum(1 for s in steps if (s.get("alarm_detect") or {}).get("_parse_error"))
        n_err += errs
        manifest.append({"disease": d.get("disease"), "hadm": d.get("hadm_id"),
                         "n_steps": len(steps), "n_triggered": triggered, "n_parse_error": errs})
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nDONE: {newly} newly alarm-annotated this run, {len(manifest)} total cases in dir, "
          f"{n_err} steps with parse errors -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
