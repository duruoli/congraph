"""Plan-T eval: read a calibrated P(deviate|input) from a Tinker-trained checkpoint.

Same readout + metrics as scripts/eval_deviation_cls.py (imported verbatim, so the panel is
identical across venues) — only the model backend changes from HF/peft to Tinker's SamplingClient.

For each row we teacher-force the two candidate answer words after the generation prompt:
  z = logprob("deviate") - logprob("follow")   ->  raw P(deviate) = sigmoid(z)
arm a: the answer slot is right after the prompt (target is one word).
arm c (--generate-first): SAMPLE the reasoning trace first, cut at the "deviation" key, then
  score follow/deviate CONDITIONED on the model's own generated reasoning.
Platt (a,b) is fit on VAL and applied to TEST. Reads the ORIGINAL cls/cls_reason JSONL (keeps meta.y).

  python scripts/tinker/eval_prob_tinker.py \
      --checkpoint tinker://<your-saved-sft-checkpoint>  --arm-name sft \
      --data data/training_set/cls  --out results/agent_inspection/tinker_deviation_a

⚠️ API-VERIFY on first run: the Tinker method names below (create_sampling_client, sample_async,
compute_logprobs_async, get_tokenizer, SamplingParams/ModelInput) are from the quickstart docs and
may need small tweaks against the installed SDK. Everything else (data, z-readout, Platt, metrics)
is your tested code.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for _p in (str(ROOT), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from experiments.llm_experiment.env_loader import load_tinker_key  # noqa: E402

# reuse the EXACT metric + Platt + reporting code from the HF eval (single source of truth)
from eval_deviation_cls import (  # noqa: E402
    CANDIDATES, apply_platt, fit_platt, report,
)


async def score_rows(rows, sampling_client, tokenizer, generate_first, max_new_tokens):
    """Return z = logprob(deviate) - logprob(follow) for each row, via Tinker."""
    import tinker
    from tinker import types

    cand_ids = {w: tokenizer.encode(w, add_special_tokens=False) for w in CANDIDATES}
    print(f"[tok] follow->{cand_ids['follow']}  deviate->{cand_ids['deviate']}")
    zs = []
    for row in rows:
        # generation prompt = system+user, with the assistant turn open
        prefix_ids = tokenizer.apply_chat_template(
            row["messages"][:2], add_generation_prompt=True, tokenize=True)

        if generate_first:
            prompt = types.ModelInput.from_ints(tokens=prefix_ids)
            params = types.SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
            res = await sampling_client.sample_async(
                prompt=prompt, num_samples=1, sampling_params=params)
            text = tokenizer.decode(res.sequences[0].tokens, skip_special_tokens=True)
            if '"deviation"' in text:
                reason = text.split('"deviation"')[0] + '"deviation": "'
            else:  # malformed: append the key onto whatever JSON body it produced
                reason = text.rstrip().rstrip('}').rstrip().rstrip(',') + ', "deviation": "'
            prefix_ids = prefix_ids + tokenizer.encode(reason, add_special_tokens=False)

        totals = {}
        for w, ids in cand_ids.items():
            full = types.ModelInput.from_ints(tokens=prefix_ids + ids)
            lp = await sampling_client.compute_logprobs_async(full)   # per-token logprob of `full`
            # candidate tokens sit at the tail; sum their logprobs
            totals[w] = float(sum(lp[-len(ids):]))
        zs.append(totals["deviate"] - totals["follow"])
    return zs


async def amain(args):
    import tinker

    data = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    val = [json.loads(l) for l in (data / "val.jsonl").read_text().splitlines() if l.strip()]
    test = [json.loads(l) for l in (data / "test.jsonl").read_text().splitlines() if l.strip()]
    yv = [r["meta"]["y"] for r in val]
    yt = [r["meta"]["y"] for r in test]

    load_tinker_key()   # reads .tinker_env into TINKER_API_KEY if not already set
    service = tinker.ServiceClient()
    sampling_client = service.create_sampling_client(
        base_model=args.base, model_path=args.checkpoint)   # trained LoRA checkpoint
    tokenizer = sampling_client.get_tokenizer()

    zv = await score_rows(val, sampling_client, tokenizer, args.generate_first, args.max_new_tokens)
    zt = await score_rows(test, sampling_client, tokenizer, args.generate_first, args.max_new_tokens)

    a, b = fit_platt(zv, yv)
    raw_t = [1.0 / (1.0 + math.exp(-z)) for z in zt]
    cal_t = apply_platt(zt, a, b)

    lines = ["### Tinker P(deviate|input)  arm=" + args.arm_name,
             f"base={args.base}  checkpoint={args.checkpoint}",
             f"val n={len(yv)} (pos {sum(yv)})  test n={len(yt)} (pos {sum(yt)})",
             "CAVEAT: N small => DIRECTIONAL; CIs wide. Platt (a,b) fit on VAL, applied to TEST.",
             f"ARM = {args.arm_name}  (platt a={a:.3f} b={b:.3f})"]
    report(f"{args.arm_name} test RAW", yt, raw_t, lines)
    report(f"{args.arm_name} test CALIBRATED", yt, cal_t, lines)

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text("\n".join(lines))
    out.with_suffix(".json").write_text(json.dumps(
        {"arm": args.arm_name, "base": args.base, "checkpoint": args.checkpoint,
         "platt": {"a": a, "b": b}, "test_z": zt, "test_raw": raw_t,
         "test_cal": cal_t, "test_y": yt}, indent=2))
    print("\n".join(lines))
    print(f"\nwrote {out}.txt / .json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--checkpoint", required=True, help="Tinker path/name of the trained LoRA")
    ap.add_argument("--arm-name", default="sft", help="label for the report (a / c / c_rl / b_new)")
    ap.add_argument("--data", default="data/training_set/cls",
                    help="cls for arm a; cls_reason for arm c (+ --generate-first)")
    ap.add_argument("--generate-first", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out", default="results/agent_inspection/tinker_deviation")
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
