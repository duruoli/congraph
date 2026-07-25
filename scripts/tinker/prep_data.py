"""Plan-T step 2: convert the existing pred_dev SFT data into Tinker conversation JSONL.

Your `data/training_set/{cls,cls_reason,sft}/*.jsonl` rows are already
`{"messages": [ {role, content}, ... ], "meta": {...}}`. Tinker's cookbook
`FromConversationFileBuilder` wants a JSONL of chat conversations; it does NOT need the
eval-only `meta`. This script just strips `meta` (keeping `messages`) and validates the roles,
writing `data/training_set/tinker/<arm>/{train,val}.jsonl`.

TEST is intentionally NOT converted here — the calibrated prob readout
(`scripts/tinker/eval_prob_tinker.py`) reads the ORIGINAL cls/cls_reason test.jsonl so it keeps
`meta.y`. Only train/val (the SFT inputs) are converted.

  python scripts/tinker/prep_data.py --arm a   # -> data/training_set/tinker/cls/*
  python scripts/tinker/prep_data.py --arm c   # -> data/training_set/tinker/cls_reason/*

NO Tinker dependency — pure stdlib, runs anywhere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# arm -> source data dir under data/training_set/
ARM_SRC = {
    "a": "cls",          # bare label target (one word)
    "c": "cls_reason",   # reasoning trace + deviation tail
    "sft": "sft",        # full certainty-agent trace (for b-NEW format reference, if needed)
}
VALID_ROLES = {"system", "user", "assistant"}


def convert(src_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        src = src_dir / f"{split}.jsonl"
        rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
        out_lines = []
        for i, r in enumerate(rows):
            msgs = r["messages"]
            roles = [m["role"] for m in msgs]
            assert set(roles) <= VALID_ROLES, f"{src}:{i} bad roles {roles}"
            assert roles[-1] == "assistant", f"{src}:{i} last turn must be assistant, got {roles[-1]}"
            # keep ONLY messages (drop meta / any extra keys); this is the Tinker conversation record
            out_lines.append(json.dumps({"messages": msgs}, ensure_ascii=False))
        (out_dir / f"{split}.jsonl").write_text("\n".join(out_lines) + "\n")
        print(f"  {split}: {len(out_lines)} rows -> {out_dir / f'{split}.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARM_SRC), required=True)
    args = ap.parse_args()
    src = ROOT / "data" / "training_set" / ARM_SRC[args.arm]
    out = ROOT / "data" / "training_set" / "tinker" / ARM_SRC[args.arm]
    print(f"[prep] arm={args.arm}  {src} -> {out}")
    convert(src, out)


if __name__ == "__main__":
    main()
