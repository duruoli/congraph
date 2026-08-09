"""Generate bare-base (b-NEW pre-RL) reasoning traces for an arbitrary cls_free split.

Generation-ONLY (no teacher-forcing / scoring): used to dump the TRAIN traces for the bottom-up
reward emergence analysis (the eval script only dumps test). Same model / renderer / prompt /
sampling regime as eval_prob_tinker.py's pre-RL reference so the traces are directly comparable to
the 56 test traces already dumped.

Run with the TINKER env:  /opt/anaconda3/envs/tinker/bin/python scripts/tinker/gen_bnew_traces.py \
    --split train --out results/agent_inspection/tinker_deviation_bnew_pre_TRAIN_generations.jsonl
"""
from __future__ import annotations

import argparse, asyncio, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from experiments.llm_experiment.env_loader import load_tinker_key  # noqa: E402
from sft_common import BASE_MODEL, RENDERER_NAME  # noqa: E402


async def _gen_one(row, sem, sc, renderer, tokenizer, max_new_tokens, stop):
    from tinker import types
    msgs = row["messages"][:2]                       # system + user; generate the assistant turn
    async with sem:
        prompt = renderer.build_generation_prompt(msgs)
        params = types.SamplingParams(max_tokens=max_new_tokens, temperature=0.0, stop=stop)
        res = await sc.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
    return tokenizer.decode(res.sequences[0].tokens, skip_special_tokens=True)


async def amain(args):
    import tinker
    from tinker_cookbook.renderers import get_renderer

    data = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    rows = [json.loads(l) for l in (data / f"{args.split}.jsonl").read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    load_tinker_key()
    sc = tinker.ServiceClient().create_sampling_client(base_model=args.base, model_path=None)
    tokenizer = sc.get_tokenizer()
    renderer = get_renderer(args.renderer, tokenizer)
    stop = renderer.get_stop_sequences() or None

    print(f"generating {len(rows)} '{args.split}' traces (concurrency {args.concurrency}, "
          f"max_new_tokens {args.max_new_tokens}) from bare base {args.base}")
    sem = asyncio.Semaphore(args.concurrency)
    gens = await asyncio.gather(*[
        _gen_one(r, sem, sc, renderer, tokenizer, args.max_new_tokens, stop) for r in rows])

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # match the eval dump schema: {i, y, p, generation}. p is None (no scoring here).
    with out.open("w") as f:
        for i, (r, g) in enumerate(zip(rows, gens)):
            f.write(json.dumps({"i": i, "y": r["meta"]["y"], "p": None, "generation": g},
                               ensure_ascii=False) + "\n")
    print(f"wrote {out}  ({len(gens)} traces)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--renderer", default=RENDERER_NAME)
    ap.add_argument("--data", default="data/training_set/cls_free")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-new-tokens", type=int, default=1024)   # match the pre-RL reference regime
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/agent_inspection/tinker_deviation_bnew_pre_TRAIN_generations.jsonl")
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
