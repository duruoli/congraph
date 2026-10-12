"""Randomly sample service-transfer events by direction for terrain annotation.

Unit = one transfer event (prev_service non-null). Records which transfer of the
admission it is (anchor_index for condense/annotate) and how many radiology
reports precede it, so imaging-poor transfers stay visible rather than skipped.
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw_data/ibd_mimiciv_3_1"
DONE = {23605251, 25292961, 29942834}
PLAN = {"MED→SURG": 4, "SURG→MED": 4, "MED→OMED": 3, "MED→CMED": 3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/ibd_service_transfer_patterns/sample.csv")
    args = parser.parse_args()

    services = ds.dataset(SOURCE / "services", format="parquet").to_table(
        columns=["hadm_id", "transfertime", "prev_service", "curr_service"]).to_pylist()
    by_hadm = defaultdict(list)
    for r in services:
        if r["prev_service"]:
            by_hadm[r["hadm_id"]].append(r)
    events = defaultdict(list)
    for hadm, rows in by_hadm.items():
        rows.sort(key=lambda r: r["transfertime"])
        for i, r in enumerate(rows):
            events[f"{r['prev_service']}→{r['curr_service']}"].append((hadm, i, r["transfertime"], len(rows)))

    rng = random.Random(args.seed)
    picked = []
    for kind, n in PLAN.items():
        pool = sorted(e for e in events[kind] if e[0] not in DONE)
        picked += [(kind, *e) for e in rng.sample(pool, n)]

    rad = ds.dataset(SOURCE / "radiology", format="parquet").to_table(
        columns=["hadm_id", "charttime"], filter=ds.field("hadm_id").isin([p[1] for p in picked])).to_pylist()
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "hadm_id", "anchor_index", "transfertime", "n_transfers", "n_radiology_before"])
        for kind, hadm, i, t, n in picked:
            before = sum(1 for r in rad if r["hadm_id"] == hadm and r["charttime"] and r["charttime"] < t)
            w.writerow([kind, hadm, i, t, n, before])
            print(kind, hadm, i, t, n, before)


if __name__ == "__main__":
    main()
