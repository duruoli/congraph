"""Direction-B SFT: LoRA fine-tune a SWAPPABLE base model on the certainty-trigger traces.

Behavior-cloning warm-start (HANDOFF §9(a)/(e)): x = (system+user) chat prompt,
y = the assistant JSON reasoning trace. Standard next-token cross-entropy, loss on
the ASSISTANT tokens only (the rubric/patient prompt is context, not a target).
REWARD is intentionally NOT used here — that is the later offline-RL stage.

The base is a `--base` flag. Qwen2.5-7B was the placeholder; the default is now
google/medgemma-27b-text-it (Gemma-3, medical-tuned, ~4x bigger). Two swap gotchas are
handled here (see experiments/agent/chat_compat.py): (1) medgemma is a GATED HF repo —
`huggingface-cli login` with a token that has accepted the license; (2) stock Gemma-3 has
NO `{% generation %}` markers, so assistant_only_loss can't mask the target — we swap in
configs/gemma3_assistant_loss_template.jinja and VALIDATE the mask before training.

Run ON A GPU MACHINE (not the dev box). Inputs are produced by
scripts/build_sft_examples.py (data/training_set/sft/{train,val}.jsonl, each line a
{"messages":[system,user,assistant], "meta":{...}} record).

  # transformers must be new enough to load Gemma-3:
  pip install "trl>=0.12" "peft>=0.13" "transformers>=4.50" accelerate bitsandbytes datasets

  # QLoRA on a single 48GB GPU (sweet spot for the 27B at 12k context):
  python scripts/train_lora_qwen.py \
      --base google/medgemma-27b-text-it \
      --data data/training_set/sft \
      --out runs/medgemma-27b-lora-certainty \
      --epochs 3 --batch 1 --grad-accum 16 --lr 2e-4 --load-4bit

Eval (reproduction + behavior-spec hit-rate per HANDOFF §5) is a SEPARATE script that
loads the adapter and scores data/training_set/sft/test.jsonl; deviate is derived from
the model's belief argmax vs the rubric (NOT a generated field).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.agent.chat_compat import (  # noqa: E402
    ensure_assistant_loss_template, validate_assistant_mask)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/medgemma-27b-text-it",
                    help="swappable base; medgemma-27b-text-it (Gemma-3) is the default, "
                         "Qwen/Qwen2.5-7B-Instruct was the placeholder")
    ap.add_argument("--data", default="data/training_set/sft")
    ap.add_argument("--out", default="runs/medgemma-27b-lora-certainty")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=12288)  # >= longest example (~10.2k tok); see length guard below
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA (bitsandbytes 4-bit)")
    # gradient checkpointing: at max-len 12288 the activations of a single 12k-token
    # forward dominate VRAM; checkpointing trades ~20% step time for a large activation
    # cut so the run fits on a 40GB A100. On by default; --no-grad-checkpointing to disable.
    ap.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction,
                    default=True, help="gradient checkpointing (default on)")
    ap.add_argument("--assistant-only-loss", action=argparse.BooleanOptionalAction,
                    default=True, help="loss on the assistant turn only (default on); "
                                       "needs a generation-marked chat template")
    ap.add_argument("--chat-template", default=None,
                    help="path to a generation-marked chat template; auto-selected for "
                         "Gemma-family bases when the stock one lacks the marker")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    # heavy deps imported inside main so --help works on the CPU dev box
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    data = Path(args.data)
    ds = load_dataset("json", data_files={
        "train": str(data / "train.jsonl"),
        "val": str(data / "val.jsonl"),
    })
    # the SFT loss only needs `messages`; drop the eval-only `meta` column
    ds = ds.remove_columns([c for c in ds["train"].column_names if c != "messages"])

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Chat-template compat: for a swapped base (e.g. Gemma-3 / medgemma) whose stock template
    # lacks the {% generation %} marker, swap in a generation-marked template so assistant_only_loss
    # can mask the target — then VALIDATE the mask is non-trivial before the expensive run.
    if args.assistant_only_loss:
        swapped = ensure_assistant_loss_template(tok, args.base, args.chat_template)
        if swapped:
            print(f"[chat-compat] swapped in a generation-marked template for '{args.base}'")
        validate_assistant_mask(tok, ds["train"][0]["messages"])

    # Truncation guard: the assistant JSON target is the LAST turn, so any example
    # longer than max_len gets its target silently truncated -> loss on a mangled
    # target. Count over-length examples up front and fail loudly if any exist.
    def _tok_len(ex):
        ids = tok.apply_chat_template(ex["messages"], tokenize=True)
        return len(ids)
    over = [(i, n) for i, n in ((i, _tok_len(ex)) for i, ex in enumerate(ds["train"]))
            if n > args.max_len]
    longest = max((_tok_len(ex) for ex in ds["train"]), default=0)
    print(f"[len-guard] longest train example = {longest} tok; max_len = {args.max_len}; "
          f"over-length = {len(over)}")
    if over:
        raise SystemExit(
            f"[len-guard] {len(over)} example(s) exceed --max-len={args.max_len} "
            f"(longest={longest}); their assistant target would be truncated. "
            f"Raise --max-len to >= {longest} (watch VRAM) or shrink the rubric.")

    quant = None
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=quant,
        torch_dtype=torch.bfloat16, device_map="auto")

    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        max_length=args.max_len,
        packing=False,
        gradient_checkpointing=args.grad_checkpointing,
        # non-reentrant checkpointing plays nicely with PEFT (avoids the use_cache /
        # input-grad warnings and the reentrant autograd edge cases).
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # train on the assistant turn only — the rubric+patient prompt is context.
        # (trl applies the tokenizer's chat template to the `messages` column; the template
        # was validated above to mark the assistant span with {% generation %}.)
        assistant_only_loss=args.assistant_only_loss,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model, args=cfg, processing_class=tok,
        train_dataset=ds["train"], eval_dataset=ds["val"], peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
