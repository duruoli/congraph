"""Curate an ALARM-BALANCED probe subset for BEHAVIORAL study of how the agents handle
each alarm situation — separate from the systematic eval.

WHY THIS EXISTS: the real test split (scripts/build_sft_examples.py) is random, stratified
by DISEASE and cut by PATIENT — it is population-representative (good for unbiased behavior
estimates) but NOT stratified by alarm. In the 56-step test set the double-alarm cell
(study_inadequacy AND discordance) has only 3 rows — too thin to study. This script pools
ALL splits (train+val+test) and hand-balances a subset with enough rows PER ALARM CELL,
so you can eyeball how base / SFT / ctx-qwen / ctx-sonnet each behave when a given alarm
fires.

HARD LIMIT — READ BEFORE USING THE OUTPUT: most cells can only be filled by borrowing from
TRAIN, so this subset HAS LEAK. It is for BEHAVIORAL / QUALITATIVE inspection ONLY. NEVER
compute fidelity / follow-rate metrics on it — those belong on the clean random test split.
Every row is tagged `source_split` and `leak` (True unless it came from test) so the leak is
never invisible. Selection prefers test > val > train exactly to keep leak as low as the
cell sizes allow.

The 4 alarm cells = the (study_inadequacy, discordance) presence combo:
  none  (F,F)   |  disc_only (F,T)  |  si_only (T,F)  |  both (T,T)

  python scripts/build_alarm_probe.py --per-cell 8
  python scripts/build_alarm_probe.py --per-cell 8 --seed 0
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT_DIR = ROOT / "data/training_set/sft"
CELL_NAMES = {(False, False): "none", (False, True): "disc_only",
              (True, False): "si_only", (True, True): "both"}
# fill order: clean test first, then val, then train (minimize leak)
SPLIT_PREF = ["test", "val", "train"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=8,
                    help="target rows per alarm cell (best-effort; capped by availability)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(SFT_DIR / "alarm_probe.jsonl"))
    return ap.parse_args()


def _cell(meta: dict) -> tuple[bool, bool]:
    al = meta.get("CERTAINTY", {}).get("alarm", {})
    return (bool(al.get("study_inadequacy", {}).get("present")),
            bool(al.get("discordance", {}).get("present")))


def main():
    args = parse_args()
    import random
    rng = random.Random(args.seed)

    # pool every split, remembering provenance
    pool: dict[tuple, list[dict]] = defaultdict(list)  # cell -> [example(with _src)]
    for split in SPLIT_PREF:
        path = SFT_DIR / f"{split}.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            ex = json.loads(line)
            ex["_src"] = split
            pool[_cell(ex["meta"])].append(ex)

    selected: list[dict] = []
    report = {}
    for cell, name in CELL_NAMES.items():
        cands = pool.get(cell, [])
        # order: preferred split first, then disease round-robin within split for balance
        by_split = defaultdict(lambda: defaultdict(list))
        for ex in cands:
            by_split[ex["_src"]][ex["meta"]["disease"]].append(ex)
        ordered: list[dict] = []
        for sp in SPLIT_PREF:
            dis_buckets = {d: exs[:] for d, exs in by_split[sp].items()}
            for exs in dis_buckets.values():
                rng.shuffle(exs)
            # round-robin across diseases so no single disease dominates the cell
            while any(dis_buckets.values()):
                for d in sorted(dis_buckets):
                    if dis_buckets[d]:
                        ordered.append(dis_buckets[d].pop())
        take = ordered[:args.per_cell]
        for ex in take:
            ex["meta"]["alarm_cell"] = name
            ex["meta"]["source_split"] = ex["_src"]
            ex["meta"]["leak"] = ex["_src"] != "test"
            selected.append(ex)
        report[name] = {
            "available_total": len(cands),
            "available_by_split": {sp: sum(1 for e in cands if e["_src"] == sp) for sp in SPLIT_PREF},
            "selected": len(take),
            "selected_leak": sum(1 for e in take if e["_src"] != "test"),
            "selected_by_disease": _count(take, lambda e: e["meta"]["disease"]),
        }

    for ex in selected:
        ex.pop("_src", None)
    rng.shuffle(selected)

    out = Path(args.out)
    with out.open("w") as f:
        for ex in selected:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    manifest = {
        "purpose": "BEHAVIORAL/qualitative alarm-handling inspection ONLY — HAS LEAK, "
                   "never use for fidelity metrics (those go on the random test split)",
        "per_cell_target": args.per_cell, "seed": args.seed,
        "total_selected": len(selected),
        "total_leak": sum(1 for e in selected if e["meta"]["leak"]),
        "by_cell": report,
    }
    (out.with_name("alarm_probe_manifest.json")).write_text(json.dumps(manifest, indent=2))

    print(f"[wrote] {out}  ({len(selected)} rows, {manifest['total_leak']} with leak)")
    for name, r in report.items():
        print(f"  {name:10s} selected={r['selected']:2d} "
              f"(leak {r['selected_leak']}) of {r['available_total']:3d} available "
              f"{r['available_by_split']}  disease={r['selected_by_disease']}")


def _count(items, key):
    c = defaultdict(int)
    for it in items:
        c[key(it)] += 1
    return dict(c)


if __name__ == "__main__":
    main()
