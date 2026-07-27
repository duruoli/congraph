"""Compile the b-NEW (free-reasoning) dataset for Plan T.

b-NEW is the arm that answers the question "does RL DISCOVER useful reasoning structure?" — so it
must NOT be handed any structure. It is the instruct base with a FREE-reasoning prompt (no JSON
schema, no required `modality`/`rubric_recommended` fields like arm c), scored pre-RL and then
RLVR-trained. This script builds only the prompts + gold labels; there is no reasoning-trace target.

Comparability with arms a and c (locked):
  - SAME patient-level, disease-stratified split (split_cases, seed 0) -> identical train/val/test
    patient partition as cls/ and cls_reason/ (no patient leak).
  - SAME rubric (compact_rubric) and SAME patient-state rendering (render_patient) -> identical
    patient EVIDENCE. Only the OUTPUT REGIME differs (free reasoning vs one word vs structured
    schema); that difference IS the manipulation, so it is meant to differ.
  - SAME binary label: meta.y = 1 for {deviate, off_rubric}, 0 for follow (from meta.when_action).

The assistant turn is the gold answer line ONLY (`Answer: <word>`), carried so the row holds meta.y
for eval/reward. It is NOT a reasoning trace and MUST NOT be used as an SFT target: training on it
would teach an answer with no CoT and destroy the free-reasoning manipulation. b-NEW's path is
  (1) pre-RL eval:  eval_prob_tinker.py --generate-first --answer-cue "\n\nAnswer: "  (uses msgs[:2])
  (2) RLVR:         reward from meta.y via rl_reward.deviation_reward (uses msgs[:2] as the prompt)
so this assistant turn is never trained on. Format-only SFT is a last resort ONLY if pre-RL output
is unparseable, and even then needs a reasoning-bearing target, not this bare answer line.

Usage:
  /opt/anaconda3/envs/congraph/bin/python scripts/build_bnew.py
      [--in data/training_set/train_steps.jsonl] [--rubric data/training_set/rubric_library.json]
      [--out data/training_set/cls_free] [--val-frac 0.15] [--test-frac 0.15] [--seed 0] [--preview N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# reuse build_sft_examples.py as the single source of truth for split + rendering (as build_deviation_cls does)
_spec = importlib.util.spec_from_file_location(
    "build_sft_examples", ROOT / "scripts" / "build_sft_examples.py")
_bse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bse)
compact_rubric = _bse.compact_rubric
render_patient = _bse.render_patient
split_cases = _bse.split_cases

POSITIVE = {"deviate", "off_rubric"}  # y=1 ; "follow" -> y=0

# The answer cue MUST match eval_prob_tinker.py --answer-cue and rl_reward.parse_prediction.
ANSWER_CUE = "\n\nAnswer: "

BNEW_SYSTEM_TEMPLATE = """You are a clinical reasoning agent for adult acute abdominal pain. \
At each decision step you are given the patient's current state (baseline + the imaging reports \
available SO FAR, with this step's result hidden) and a rubric covering four index diseases plus \
an open 'other' slot.

Your job: PREDICT whether the physician's NEXT imaging move will FOLLOW the study the rubric \
recommends for the leading hypothesis, or DEVIATE from it. ("deviate" also covers the case where \
the working hypothesis lies outside the rubric.)

Reason about the case in your own words, and keep it CONCISE — a few sentences, not an essay. \
There is no required format or template; think freely.

When you are finished, end your response with a single final line in EXACTLY this form, lowercase, \
with nothing after it:
Answer: follow
or
Answer: deviate

=== RUBRIC ===
__RUBRIC__"""


def bnew_user_prompt(rec: dict) -> str:
    """SAME patient evidence as arm a/c (render_patient + active_line); only the task line differs."""
    ps = rec["INPUT"]["patient_state"]
    active = rec["INPUT"].get("active_path")
    active_line = (f"Current working hypothesis (belief argmax last step): {active}"
                   if active else "No prior step — the differential is still open.")
    return (render_patient(ps) + "\n\n## Your task\n" + active_line +
            "\nReason BRIEFLY about the physician's next move (a few sentences), then give your final "
            "answer on the last line as 'Answer: follow' or 'Answer: deviate'.")


def label_word(when_action: str) -> str:
    return "deviate" if when_action in POSITIVE else "follow"


def to_example(rec: dict, system: str) -> dict:
    when = rec["TARGET"]["when_action"]
    word = label_word(when)
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": bnew_user_prompt(rec)},
            # gold answer line ONLY — never an SFT target (see module docstring). Held for meta.y.
            {"role": "assistant", "content": f"Answer: {word}"},
        ],
        "meta": {
            "id": rec["id"], "disease": rec["disease"], "hadm": rec["hadm"], "step": rec["step"],
            "when_action": when,
            "label": word,
            "y": 1 if word == "deviate" else 0,
            "effective_branch": rec["TARGET"]["effective_branch"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/training_set/train_steps.jsonl")
    ap.add_argument("--rubric", default="data/training_set/rubric_library.json")
    ap.add_argument("--out", default="data/training_set/cls_free")
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
    system = BNEW_SYSTEM_TEMPLATE.replace("__RUBRIC__", rubric_text)

    assign = split_cases(rows, args.val_frac, args.test_frac, args.seed)   # identical split to cls/cls_reason
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
        "task": "b-NEW free-reasoning deviation prediction (instruct base + RLVR); prompts + gold only",
        "label_map": "follow->follow(0); {deviate,off_rubric}->deviate(1)",
        "answer_cue": ANSWER_CUE,
        "source": str(inp.relative_to(ROOT)),
        "system_prompt_chars": len(system),
        "rubric_chars": len(rubric_text),
        "assistant_turn": "gold answer line only ('Answer: <word>'); NOT an SFT target — see docstring",
        "split": {k: stats(v) for k, v in buckets.items()},
        "patient_leak_train_test": len(overlap),
        "config": vars(args),
    }
    (out_dir / "cls_free_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"system prompt: {len(system)} chars (rubric {len(rubric_text)}); answer_cue={ANSWER_CUE!r}")
    for k in ("train", "val", "test"):
        s = manifest["split"][k]
        print(f"{k:5s}: {s['rows']:3d} rows / {s['cases']:3d} cases  "
              f"pos(deviate)={s['pos(deviate)']} neg(follow)={s['neg(follow)']} "
              f"base_rate={s['base_rate']}  when={s['when_action']}")
    print(f"patient leak (train∩test cases): {len(overlap)}")
    print(f"wrote {out_dir}/(train|val|test).jsonl + cls_free_manifest.json")

    for e in buckets["train"][:args.preview]:
        print("\n" + "=" * 80)
        print("SYSTEM (tail):", e["messages"][0]["content"][-400:])
        print("\nUSER (tail):", e["messages"][1]["content"][-450:])
        print("\nASSISTANT:", repr(e["messages"][2]["content"]), "  meta.y=", e["meta"]["y"])


if __name__ == "__main__":
    main()
