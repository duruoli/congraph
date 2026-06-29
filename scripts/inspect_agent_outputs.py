"""Manual eyeball of the direction-B LoRA agent: a THREE-WAY side-by-side per test step
of GOLD (the annotation) vs BASE (untuned Qwen2.5) vs SFT (base + your LoRA adapter), so
you can judge the reasoning trace qualitatively BEFORE the systematic eval
(scripts/eval_certainty_agent.py). NOT scoring — a human-readable comparison.

The base and SFT generations come from the SAME loaded weights: the adapter is toggled
off with PeftModel.disable_adapter() for the BASE pass, so memory ~= one 7B model.

Runs anywhere with enough memory — picks cuda > mps > cpu automatically. The weights live
on HF (Duruo/qwen2.5-7b-congraph-lora, PRIVATE) but inference runs on THIS machine; HF
does not serve it for you. On an Apple-Silicon Mac it works via MPS in bf16 (~14GB, slow,
~1-3 min/step; NO --load-4bit, bitsandbytes is CUDA-only). On Quest use the GPU. Heavy
deps import inside main so `--help` works on a CPU box.

  pip install "peft>=0.13" "transformers>=4.45" accelerate    # + bitsandbytes only on CUDA
  huggingface-cli login            # the adapter repo is PRIVATE

  python scripts/inspect_agent_outputs.py --n 6 --seed 0
  python scripts/inspect_agent_outputs.py --only sft          # skip the base pass (faster)
  python scripts/inspect_agent_outputs.py --id appendicitis_20279299_step1   # one specific step
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/training_set/sft/test.jsonl"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="Duruo/qwen2.5-7b-congraph-lora",
                    help="HF repo id or local dir of the LoRA adapter")
    ap.add_argument("--only", choices=["both", "base", "sft"], default="both",
                    help="which model passes to generate (default both = 3-way with GOLD)")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--n", type=int, default=6, help="number of steps to sample")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (ignored with --first/--id)")
    ap.add_argument("--first", action="store_true", help="take the first --n rows instead of random")
    ap.add_argument("--id", default=None, help="inspect one specific step id (overrides sampling)")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (reproducible)")
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA-style 4-bit load (saves VRAM)")
    ap.add_argument("--view-chars", type=int, default=900, help="chars of the patient view to print")
    ap.add_argument("--out", default="results/agent_inspection/inspect_outputs.txt",
                    help="write the human-readable side-by-side here (git-tracked so you can "
                         "push/pull it home to analyze); a sibling .jsonl holds the structured rows")
    return ap.parse_args()


def _extract_json(text: str) -> dict | None:
    """Pull the first balanced {...} block out of a model generation and parse it."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _belief_line(belief: dict | None) -> str:
    if not isinstance(belief, dict):
        return "(no parseable belief)"
    items = [(k, v) for k, v in belief.items() if isinstance(v, (int, float))]
    if not items:
        return "(no numeric belief)"
    items.sort(key=lambda kv: kv[1], reverse=True)
    argmax = items[0][0]
    dist = "  ".join(f"{k}={v:.2f}" for k, v in items)
    return f"argmax={argmax:<14} | {dist}"


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    if args.id:
        rows = [r for r in rows if r["meta"].get("id") == args.id]
        if not rows:
            raise SystemExit(f"id {args.id} not found in {args.data}")
    elif args.first:
        rows = rows[:args.n]
    else:
        import random
        random.Random(args.seed).shuffle(rows)
        rows = rows[:args.n]

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    if args.load_4bit and device != "cuda":
        raise SystemExit("--load-4bit needs CUDA (bitsandbytes); on Mac/CPU drop it.")
    print(f"[device] {device}  dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    # device_map="auto" only for CUDA (multi-GPU sharding); else load then .to(device)
    load_kwargs = dict(torch_dtype=dtype, quantization_config=quant)
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.base, **load_kwargs)
    if device != "cuda":
        model = model.to(device)

    have_adapter = args.only in ("both", "sft")
    if have_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    passes = {"both": ["base", "sft"], "base": ["base"], "sft": ["sft"]}[args.only]

    buf: list[str] = []          # mirrored to --out
    records: list[dict] = []     # structured rows -> sibling .jsonl

    def emit(s: str = ""):
        print(s)
        buf.append(s)

    emit(f"### {len(rows)} step(s)  |  passes={passes}  |  "
         f"{'greedy' if args.temperature == 0 else f'T={args.temperature}'}  |  "
         f"device={device}  |  adapter={args.adapter}  |  GOLD = the annotation target\n")

    def _generate(enc):
        input_len = enc["input_ids"].shape[1]
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
        if args.temperature and args.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=args.temperature)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        return tok.decode(out[0, input_len:], skip_special_tokens=True)

    for r in rows:
        meta = r["meta"]
        sys_msg, user_msg, gold_msg = r["messages"][0], r["messages"][1], r["messages"][-1]
        # return_dict=True -> a BatchEncoding (input_ids + attention_mask) we splat into
        # generate(); apply_chat_template's bare return_tensors form is not a plain tensor
        # in all transformers versions, which breaks generate's inputs.shape access.
        enc = tok.apply_chat_template(
            [sys_msg, user_msg], add_generation_prompt=True,
            return_tensors="pt", return_dict=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        outs = {}  # name -> raw generation
        for name in passes:
            if name == "base" and have_adapter:
                # toggle the LoRA OFF to get the untuned base from the same weights
                with model.disable_adapter():
                    outs[name] = _generate(enc)
            else:
                outs[name] = _generate(enc)

        gold = _extract_json(gold_msg["content"]) or {}
        preds = {name: _extract_json(g) for name, g in outs.items()}

        emit("=" * 100)
        emit(f"id={meta['id']}  disease={meta['disease']}  step={meta['step']}")
        emit(f"GOLD when_action={meta.get('when_action')}  effective_branch={meta.get('effective_branch')}  "
             f"verification={meta.get('REWARD', {}).get('verification')}  "
             f"appropriateness={meta.get('REWARD', {}).get('appropriateness')}")
        alarm = meta.get("CERTAINTY", {}).get("alarm", {})
        emit(f"GOLD alarm.any_trigger={alarm.get('any_trigger')}  "
             f"study_inadequacy={alarm.get('study_inadequacy', {}).get('present')}  "
             f"discordance={alarm.get('discordance', {}).get('present')}")
        emit("-" * 100)
        emit("PATIENT VIEW (trimmed):")
        emit(user_msg["content"][:args.view_chars].rstrip() + " ...")
        emit("-" * 100)
        emit(f"GOLD belief : {_belief_line(gold.get('belief'))}")
        for name in passes:
            p = preds[name]
            emit(f"{name.upper():>4} belief : {_belief_line(p.get('belief') if p else None)}")
        gm = gold.get("modality")
        line = f"GOLD modality={gm!r}"
        for name in passes:
            pm = (preds[name] or {}).get("modality")
            line += f"   {name.upper()}={pm!r}{'(MATCH)' if pm == gm else ''}"
        emit(line)
        emit("-" * 100)
        for name in passes:
            p = preds[name]
            if p is None:
                emit(f"[{name.upper()}] UNPARSEABLE — raw:\n" + outs[name][:1200] + "\n")
            else:
                emit(f"[{name.upper()}] JSON:\n" + json.dumps(p, indent=1, ensure_ascii=False)[:2000] + "\n")

        records.append({
            "id": meta["id"], "disease": meta["disease"], "step": meta["step"],
            "when_action": meta.get("when_action"),
            "effective_branch": meta.get("effective_branch"),
            "verification": meta.get("REWARD", {}).get("verification"),
            "appropriateness": meta.get("REWARD", {}).get("appropriateness"),
            "alarm_any": alarm.get("any_trigger"),
            "gold": {"belief": gold.get("belief"), "modality": gold.get("modality")},
            # full raw generation + parsed JSON for each pass, for offline analysis
            **{name: {"raw": outs[name], "parsed": preds[name]} for name in passes},
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(buf) + "\n")
    jsonl_path = out_path.with_suffix(".jsonl")
    with jsonl_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[wrote] {out_path}  (+ structured {jsonl_path.name}, {len(records)} rows)")


if __name__ == "__main__":
    main()
