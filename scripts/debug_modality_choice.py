"""Diagnose WHY the SFT agent (almost) always orders CT_Abdomen.

The inspector (scripts/inspect_agent_outputs.py) showed SFT picking CT 6/6 under greedy
decoding, even though the training labels are NOT CT-biased (CT 46% / US 45% overall;
pancreatitis is even US-leaning at US 46% / CT 33%). So "always CT" is not a label prior.
This script localizes the cause by reading the model's modality-token distribution directly,
teacher-forced, instead of trusting a single greedy argmax.

Per test step it computes, for BASE and SFT (same weights, adapter toggled), the probability
the model assigns to each candidate modality string at the `"modality": "<here>"` position,
under three FIXED beliefs:
  * GOLD   — the annotation's belief (the realistic conditioning),
  * chole@ — a synthetic belief pinned to cholecystitis 0.90 (clinically US-first),
  * appy@  — a synthetic belief pinned to appendicitis 0.90.

Reading the output:
  - If SFT P(CT) ~= 1.0 and P(US) ~= 0 EVEN under the chole@ belief  -> genuine MODE COLLAPSE:
    the modality head ignores the belief. A real training problem (fix: data balance / loss /
    decouple modality from the verbose-reasoning template / more epochs-vs-overfit).
  - If SFT keeps real US mass (e.g. 0.3-0.4) but just below CT under gold belief, and US RISES
    under chole@  -> NOT a collapse but a GREEDY-DECODING artifact + a learned
    "confident-disease -> CT-for-severity" lean. Fix is on the eval/decoding side (sample, or
    score the distribution, don't read argmax modality), not retraining.

Costs one short forward per (candidate modality x belief x model) — no 768-token generation,
so it's much faster than the inspector. Same device/þdtype/adapter handling.

  python scripts/debug_modality_choice.py --n 12 --seed 0
  python scripts/debug_modality_choice.py --id pancreatitis_29499458_step1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/training_set/sft/test.jsonl"

# the legal modality vocabulary, straight from the system prompt
MODALITIES = ["CT_Abdomen", "Ultrasound_Abdomen", "MRCP_Abdomen", "MRI_Abdomen", "CTU_Abdomen"]
# short aliases for compact printing
ALIAS = {"CT_Abdomen": "CT", "Ultrasound_Abdomen": "US", "MRCP_Abdomen": "MRCP",
         "MRI_Abdomen": "MRI", "CTU_Abdomen": "CTU"}
# fixed belief key order — must match the training serialization
BELIEF_KEYS = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis", "other"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="Duruo/qwen2.5-7b-congraph-lora")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--first", action="store_true")
    ap.add_argument("--id", default=None)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--out", default="results/agent_inspection/debug_modality.txt")
    return ap.parse_args()


def _pinned_belief(disease: str, p: float = 0.90) -> dict:
    """A synthetic belief pinned to `disease` at prob p, remainder spread over the rest."""
    rest = (1.0 - p) / (len(BELIEF_KEYS) - 1)
    return {k: (round(p, 2) if k == disease else round(rest, 2)) for k in BELIEF_KEYS}


def _ordered(belief: dict) -> dict:
    """Re-serialize a belief dict in the canonical training key order."""
    return {k: float(belief.get(k, 0.0)) for k in BELIEF_KEYS}


def _belief_prefix(belief: dict) -> str:
    """The assistant-turn prefix up to (and including) the open quote of the modality value,
    matching the compact json.dumps serialization used in the SFT targets."""
    return '{"belief": ' + json.dumps(_ordered(belief)) + ', "modality": "'


def main():
    args = parse_args()
    import torch
    import torch.nn.functional as F
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
        raise SystemExit("--load-4bit needs CUDA; on Mac/CPU drop it.")
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
    load_kwargs = dict(torch_dtype=dtype, quantization_config=quant)
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.base, **load_kwargs)
    if device != "cuda":
        model = model.to(device)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # pre-tokenize candidate continuations once (no special tokens; close the JSON string)
    cand_ids = {m: tok(m + '"', add_special_tokens=False).input_ids for m in MODALITIES}

    def modality_dist(prefix_ids: "torch.Tensor") -> dict:
        """P(each modality string | prefix) as a softmax over the 5 candidates' summed logprob."""
        totals = {}
        for m, ids in cand_ids.items():
            cont = torch.tensor(ids, device=prefix_ids.device)
            full = torch.cat([prefix_ids, cont]).unsqueeze(0)
            with torch.no_grad():
                logits = model(full).logits[0]
            logprobs = F.log_softmax(logits.float(), dim=-1)
            start = prefix_ids.shape[0] - 1  # position predicting the first continuation token
            s = sum(logprobs[start + j, tokid].item() for j, tokid in enumerate(ids))
            totals[m] = s
        keys = list(totals)
        t = torch.tensor([totals[k] for k in keys])
        probs = F.softmax(t, dim=0)
        return {k: probs[i].item() for i, k in enumerate(keys)}

    def both_dists(prompt_ids: "torch.Tensor", belief: dict) -> dict:
        """{'base':dist, 'sft':dist} for one fixed belief, from the SAME weights."""
        prefix = tok(_belief_prefix(belief), add_special_tokens=False, return_tensors="pt")
        prefix_ids = torch.cat([prompt_ids, prefix["input_ids"][0].to(prompt_ids.device)])
        with model.disable_adapter():
            base = modality_dist(prefix_ids)
        sft = modality_dist(prefix_ids)
        return {"base": base, "sft": sft}

    def fmt(dist: dict) -> str:
        items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        return "  ".join(f"{ALIAS[k]}={v:.2f}" for k, v in items)

    buf, records = [], []

    def emit(s=""):
        print(s)
        buf.append(s)

    emit(f"### modality-choice debug  |  {len(rows)} step(s)  |  device={device}  |  adapter={args.adapter}")
    emit("Each line: P(modality | fixed belief) over the 5 legal modalities, teacher-forced.")
    emit("GOLD=annotation belief; chole@.9 / appy@.9 = synthetic pinned beliefs (US-first clinically).\n")

    # running aggregates (gold belief only)
    agg = {"base": {"US": 0.0, "CT": 0.0, "argmax_US": 0}, "sft": {"US": 0.0, "CT": 0.0, "argmax_US": 0}}

    for r in rows:
        m = r["meta"]
        sys_msg, user_msg, gold_msg = r["messages"][0], r["messages"][1], r["messages"][-1]
        gold = json.loads(gold_msg["content"])
        gold_belief = gold.get("belief") or _pinned_belief(m["disease"])
        gold_mod = gold.get("modality")

        prompt = tok.apply_chat_template(
            [sys_msg, user_msg], add_generation_prompt=True,
            return_tensors="pt", return_dict=True)
        prompt_ids = prompt["input_ids"][0].to(model.device)

        beliefs = {
            f"GOLD(argmax={max(gold_belief, key=gold_belief.get)})": gold_belief,
            "chole@.9": _pinned_belief("cholecystitis"),
            "appy@.9": _pinned_belief("appendicitis"),
        }

        emit("=" * 100)
        emit(f"id={m['id']}  disease={m['disease']}  step={m['step']}  "
             f"when_action={m.get('when_action')}  GOLD modality={gold_mod} ({ALIAS.get(gold_mod, '?')})")
        rec = {"id": m["id"], "disease": m["disease"], "gold_modality": gold_mod, "probes": {}}
        for label, belief in beliefs.items():
            d = both_dists(prompt_ids, belief)
            emit(f"  belief={label:28s}  BASE: {fmt(d['base'])}")
            emit(f"  {'':36s}SFT : {fmt(d['sft'])}")
            rec["probes"][label] = d
            if label.startswith("GOLD"):
                for who in ("base", "sft"):
                    agg[who]["US"] += d[who]["Ultrasound_Abdomen"]
                    agg[who]["CT"] += d[who]["CT_Abdomen"]
                    if max(d[who], key=d[who].get) == "Ultrasound_Abdomen":
                        agg[who]["argmax_US"] += 1
        emit("")
        records.append(rec)

    n = len(rows)
    emit("=" * 100)
    emit("AGGREGATE under GOLD belief (mean over steps):")
    for who in ("base", "sft"):
        a = agg[who]
        emit(f"  {who.upper():4s}  mean P(US)={a['US']/n:.3f}  mean P(CT)={a['CT']/n:.3f}  "
             f"argmax==US on {a['argmax_US']}/{n} steps")
    emit("\nReading: SFT mean P(US) collapsing toward ~0 while BASE keeps real US mass => the")
    emit("fine-tune suppressed US. If US recovers under chole@.9 above, the bel->modality link")
    emit("survived and the greedy argmax just masks it; if not, the modality head truly collapsed.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(buf) + "\n")
    with out_path.with_suffix(".jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[wrote] {out_path}  (+ {out_path.with_suffix('.jsonl').name})")


if __name__ == "__main__":
    main()
