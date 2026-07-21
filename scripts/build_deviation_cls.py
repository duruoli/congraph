"""Compile the step-level training set into a BINARY deviation-classification dataset.

Target = P(deviate | input): given the causally-masked patient state + the rubric,
predict whether the physician's NEXT imaging move FOLLOWS the rubric-recommended study
or DEVIATES from it. This is a SUPERVISED classifier, NOT the reasoning-trace SFT and
NOT RL — the label is OBSERVED (meta.when_action, computed by belief_step_deviation).

Binary mapping (locked with user 2026-07-20):
    when_action == "follow"                    -> "follow"  (y=0)
    when_action in {"deviate", "off_rubric"}   -> "deviate" (y=1)
(off_rubric = the physician's working hypothesis left the rubric entirely = a form of
non-adherence, so it is folded into the positive class.)

Design carried over from build_sft_examples.py so the two datasets are comparable:
  - SAME patient-level, disease-stratified split (split_cases, seed 0) -> identical
    train/val/test patient partition as data/training_set/sft/ (no patient leak).
  - SAME rubric (compact_rubric) and SAME patient-state rendering (render_patient) ->
    identical INPUT; only the OUTPUT instruction + assistant target differ.
  - The rubric STAYS in the prompt: "deviation" is defined relative to the rubric, so the
    model must see it. Baseline + prior reports are legit inputs; this step's result, the
    ordered modality, and outcome/vindication are NOT in the input (no label leak).
  - Assistant target = ONE word ("follow" / "deviate"). At inference, read the softmax
    over these two tokens at the answer slot -> a calibrated P(deviate) (temperature-scale
    on val). The label word is chosen so the two options differ at the first token.

The whole render/split machinery is imported from build_sft_examples.py (single source of
truth) so the two datasets can never drift apart on split or input rendering.

Usage:
  /opt/anaconda3/envs/congraph/bin/python scripts/build_deviation_cls.py
      [--in data/training_set/train_steps.jsonl]
      [--rubric data/training_set/rubric_library.json]
      [--out data/training_set/cls]
      [--val-frac 0.15] [--test-frac 0.15] [--seed 0] [--preview N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- reuse build_sft_examples.py as the single source of truth for split + rendering ---
_spec = importlib.util.spec_from_file_location(
    "build_sft_examples", ROOT / "scripts" / "build_sft_examples.py")
_bse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bse)
compact_rubric = _bse.compact_rubric
render_patient = _bse.render_patient
split_cases = _bse.split_cases

POSITIVE = {"deviate", "off_rubric"}  # y=1 ; "follow" -> y=0

CLS_SYSTEM_TEMPLATE = """You are a clinical reasoning agent for adult acute abdominal pain. \
At each decision step you are given the patient's current state (baseline + the imaging \
reports available SO FAR, with this step's result hidden) and a rubric covering four index \
diseases plus an open 'other' slot.

Your job: PREDICT whether the physician's NEXT imaging move will FOLLOW the study the rubric \
recommends for the leading hypothesis, or DEVIATE from it.

Output EXACTLY one word, lowercase, no punctuation and no explanation:
  follow   - the next study matches what the rubric recommends at this point.
  deviate  - the next study differs from the rubric's recommendation, OR the working \
hypothesis is outside the rubric.

=== RUBRIC ===
__RUBRIC__"""


def cls_user_prompt(rec: dict) -> str:
    ps = rec["INPUT"]["patient_state"]
    active = rec["INPUT"].get("active_path")
    active_line = (f"Current working hypothesis (belief argmax last step): {active}"
                   if active else "No prior step — the differential is still open.")
    return (render_patient(ps) + "\n\n## Your task\n" + active_line +
            "\nPredict the physician's next move. Output one word: follow or deviate.")


def label_word(when_action: str) -> str:
    return "deviate" if when_action in POSITIVE else "follow"


def to_example(rec: dict, system: str) -> dict:
    when = rec["TARGET"]["when_action"]
    word = label_word(when)
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": cls_user_prompt(rec)},
            {"role": "assistant", "content": word},
        ],
        "meta": {  # NOT seen by the loss; used by eval/calibration
            "id": rec["id"], "disease": rec["disease"], "hadm": rec["hadm"], "step": rec["step"],
            "when_action": when,          # original 3-class label
            "label": word,                # binary word target
            "y": 1 if word == "deviate" else 0,
            "effective_branch": rec["TARGET"]["effective_branch"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/training_set/train_steps.jsonl")
    ap.add_argument("--rubric", default="data/training_set/rubric_library.json")
    ap.add_argument("--out", default="data/training_set/cls")
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
    rubric_text = compact_rubric(lib)
    system = CLS_SYSTEM_TEMPLATE.replace("__RUBRIC__", rubric_text)

    # identical split to the reasoning-trace SFT (same fn, same seed)
    assign = split_cases(rows, args.val_frac, args.test_frac, args.seed)
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
                "by_disease": dict(Counter(e["meta"]["disease"] for e in exs)),
                "when_action": dict(Counter(e["meta"]["when_action"] for e in exs))}
    overlap = ({(e["meta"]["disease"], e["meta"]["hadm"]) for e in buckets["train"]}
               & {(e["meta"]["disease"], e["meta"]["hadm"]) for e in buckets["test"]})
    manifest = {
        "task": "binary deviation classification: P(deviate|input)",
        "label_map": "follow->follow(0); {deviate,off_rubric}->deviate(1)",
        "source": str(inp.relative_to(ROOT)),
        "system_prompt_chars": len(system),
        "rubric_chars": len(rubric_text),
        "split": {k: stats(v) for k, v in buckets.items()},
        "patient_leak_train_test": len(overlap),
        "config": vars(args),
    }
    (out_dir / "cls_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"system prompt: {len(system)} chars (rubric {len(rubric_text)})")
    for k in ("train", "val", "test"):
        s = manifest["split"][k]
        print(f"{k:5s}: {s['rows']:3d} rows / {s['cases']:3d} cases  "
              f"pos(deviate)={s['pos(deviate)']} neg(follow)={s['neg(follow)']} "
              f"base_rate={s['base_rate']}  when={s['when_action']}")
    print(f"patient leak (train∩test cases): {len(overlap)}")
    print(f"wrote {out_dir}/(train|val|test).jsonl + cls_manifest.json")

    for e in buckets["train"][:args.preview]:
        print("\n" + "=" * 80)
        print("SYSTEM (tail):", e["messages"][0]["content"][-300:])
        print("\nUSER (tail):", e["messages"][1]["content"][-400:])
        print("\nASSISTANT:", repr(e["messages"][2]["content"]), "  meta.y=", e["meta"]["y"])


if __name__ == "__main__":
    main()
