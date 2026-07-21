"""Approach C data: SFT target = reasoning trace + a deviation-prediction tail.

Same task as build_deviation_cls.py (predict P(doctor deviates | input)) but instead of a
bare label word, the assistant emits the FULL clinical reasoning trace (belief / gap /
expected / grounding, exactly as build_sft_examples.py) and then, as the LAST key,
"deviation": follow|deviate. Loss is on the whole assistant turn -> reasoning dominates the
token budget and the deviation label is a small tail ("even just a consequence"): the
probability is a readout CONDITIONED on the model's own generated reasoning, not a shortcut.

Everything except the target/system is imported from build_sft_examples.py (single source of
truth) so the split (seed-0, patient-stratified) and the INPUT rendering are byte-identical to
data/training_set/sft and data/training_set/cls -> the three approaches (a/b/c) are compared on
the same patients, same inputs, same binary label. See HANDOFF_pred_dev.md.

Binary label (LOCKED): follow->"follow"(0); {deviate,off_rubric}->"deviate"(1).
EX-ANTE ONLY: the target carries NO verification/appropriateness/outcome — else the prob would
leak the future.

Usage:
  /opt/anaconda3/envs/congraph/bin/python scripts/build_devreason.py [--preview N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "build_sft_examples", ROOT / "scripts" / "build_sft_examples.py")
_bse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bse)

POSITIVE = {"deviate", "off_rubric"}  # y=1

# reasoning system prompt + a deviation key appended to the schema (single source: _bse) -----
_DEV_KEY = (
    '  "other_hypothesis": when belief argmax is \'other\', name the leading non-rubric '
    'process; else "".\n'
    '  "deviation": Based on your reasoning above — your leading hypothesis (belief argmax) and '
    'the "modality" you chose — output "follow" if that modality is what the rubric recommends '
    'for that hypothesis here, else "deviate" (your study departs from the rubric, OR your '
    'leading hypothesis is outside the rubric). This is the LAST key.\n\nRoute yourself:')
DEVREASON_SYSTEM_TEMPLATE = _bse.SYSTEM_TEMPLATE.replace(
    '  "other_hypothesis": when belief argmax is \'other\', name the leading non-rubric '
    'process; else "".\n\nRoute yourself:', _DEV_KEY)
assert DEVREASON_SYSTEM_TEMPLATE != _bse.SYSTEM_TEMPLATE, "deviation-key injection failed"


def label_word(when_action: str) -> str:
    return "deviate" if when_action in POSITIVE else "follow"


def devreason_target(rec: dict) -> str:
    # reuse the exact reasoning trace, then append the deviation label as the last key
    obj = json.loads(_bse.assistant_target(rec))
    obj["deviation"] = label_word(rec["TARGET"]["when_action"])
    return json.dumps(obj, ensure_ascii=False)


def to_example(rec: dict, system: str) -> dict:
    when = rec["TARGET"]["when_action"]
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _bse.user_prompt(rec)},
            {"role": "assistant", "content": devreason_target(rec)},
        ],
        "meta": {
            "id": rec["id"], "disease": rec["disease"], "hadm": rec["hadm"], "step": rec["step"],
            "when_action": when,
            "label": label_word(when),
            "y": 1 if label_word(when) == "deviate" else 0,
            "effective_branch": rec["TARGET"]["effective_branch"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/training_set/train_steps.jsonl")
    ap.add_argument("--rubric", default="data/training_set/rubric_library.json")
    ap.add_argument("--out", default="data/training_set/cls_reason")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=0)
    args = ap.parse_args()

    inp = ROOT / args.inp if not Path(args.inp).is_absolute() else Path(args.inp)
    rubric_p = ROOT / args.rubric if not Path(args.rubric).is_absolute() else Path(args.rubric)
    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in inp.read_text().splitlines() if l.strip()]
    lib = json.loads(rubric_p.read_text())
    rubric_text = _bse.compact_rubric(lib)
    system = DEVREASON_SYSTEM_TEMPLATE.replace("__RUBRIC__", rubric_text)

    assign = _bse.split_cases(rows, args.val_frac, args.test_frac, args.seed)  # SAME split as sft/cls
    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in rows:
        buckets[assign[(r["disease"], r["hadm"])]].append(to_example(r, system))

    for name, exs in buckets.items():
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for e in exs:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def stats(exs):
        cases = {(e["meta"]["disease"], e["meta"]["hadm"]) for e in exs}
        pos = sum(e["meta"]["y"] for e in exs)
        return {"rows": len(exs), "cases": len(cases), "pos(deviate)": pos,
                "neg(follow)": len(exs) - pos,
                "base_rate": round(pos / len(exs), 3) if exs else 0.0,
                "when_action": dict(Counter(e["meta"]["when_action"] for e in exs))}
    overlap = ({(e["meta"]["disease"], e["meta"]["hadm"]) for e in buckets["train"]}
               & {(e["meta"]["disease"], e["meta"]["hadm"]) for e in buckets["test"]})
    manifest = {
        "task": "approach C: reasoning trace + deviation tail; read P(deviate) from the tail token",
        "label_map": "follow->follow(0); {deviate,off_rubric}->deviate(1)",
        "source": str(inp.relative_to(ROOT)),
        "system_prompt_chars": len(system),
        "split": {k: stats(v) for k, v in buckets.items()},
        "patient_leak_train_test": len(overlap),
        "config": vars(args),
    }
    (out_dir / "cls_reason_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"system prompt: {len(system)} chars")
    for k in ("train", "val", "test"):
        s = manifest["split"][k]
        print(f"{k:5s}: {s['rows']:3d} rows / {s['cases']:3d} cases  "
              f"pos={s['pos(deviate)']} neg={s['neg(follow)']} base_rate={s['base_rate']}  "
              f"when={s['when_action']}")
    print(f"patient leak (train∩test cases): {len(overlap)}")
    print(f"wrote {out_dir}/(train|val|test).jsonl + cls_reason_manifest.json")

    for e in buckets["train"][:args.preview]:
        print("\n" + "=" * 80)
        print("SYSTEM (schema tail):",
              e["messages"][0]["content"].split("=== RUBRIC ===")[0][-500:])
        print("\nASSISTANT:", e["messages"][2]["content"][:900], "...")
        print("  meta.y =", e["meta"]["y"], " when =", e["meta"]["when_action"])


if __name__ == "__main__":
    main()
