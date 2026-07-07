# medgemma-27b full-quality LoRA SFT on Quest (SLURM) — runbook

Direction-B SFT of the certainty-trigger agent, swapping the placeholder Qwen2.5-7B for
**`google/medgemma-27b-text-it`** (Gemma-3, medical-tuned). "Full quality" = **bf16 base +
LoRA, NO quantization** (do NOT pass `--load-4bit` — that is the QLoRA path we are avoiding).

Scripts: `scripts/train_lora_qwen.py` (base is a `--base` flag, medgemma is the default now).
Chat-template compat is handled by `experiments/agent/chat_compat.py` +
`configs/gemma3_assistant_loss_template.jinja` (see "Why the template swap" below).

---

## Instance config (decided 2026-07-07) — reasonable, with margin

| resource | request | why |
|---|---|---|
| GPU | `--gres=gpu:h100:1` (80 GB) | fits full bf16 LoRA of 27B (see budget) |
| system RAM | `--mem=128G` | checkpoint shards ~54 GB; ample headroom |
| wall time | `--time=08:00:00` | compute is ~2–4 h; 8 h is generous, 24 h wastes queue priority |
| CPUs | `--cpus-per-task=8` | dataloading only |

**VRAM budget on the 80 GB H100** (frozen base ⇒ no optimizer states on the 27B):
- base weights bf16: 27B × 2 = **~54 GB**
- LoRA params + Adam (fp32, LoRA-only): **<2 GB**
- activations @ 12k ctx, batch 1, **gradient checkpointing ON**: **~8–12 GB**
- **peak ≈ 64–68 GB → ~12–15 GB headroom.** ✅

**Wall-time estimate:** ~401 rows × 3 epochs ≈ 1200 example passes, ~5–10 s/example on H100
→ ~2–4 h compute + ~10 min load + per-epoch eval. 8 h is comfortable.

**If it OOMs** (unlikely): keep grad checkpointing on (default), then in order of preference —
(a) lower `--max-len` toward the true longest example (~10.2k tok, so `--max-len 10496` trims
activations without truncating any target — the len-guard will confirm), (b) 8-bit Adam
(`pip install bitsandbytes`, `--optim adamw_bnb_8bit` via a small edit), (c) last resort
`--load-4bit` (drops to QLoRA = not full-quality).

---

## sbatch script

**Ready to submit: `scripts/train_medgemma_sft.slurm`** (cloned from the working
`scripts/train_lora_qwen.slurm`; account `p31777`, partition `gengpu`, the no-`module load cuda`
+ `LD_LIBRARY_PATH` fix, and the offline-HF pattern all carried over — only GPU/mem/time/model
updated).

- Submit: `sbatch scripts/train_medgemma_sft.slurm`
- Watch: `squeue -u $USER` · `tail -f runs/medgemma_lora_<jobid>.out`
- **Before first submit, confirm `gengpu` actually offers H100s:** `sinfo -p gengpu -o "%P %G %N"`.
  The Qwen run used `gengpu` **A100s**; if Quest's H100s live in another partition/constraint,
  update `--partition` / `--gres` in the .slurm. (Fallback: an 80GB A100 also fits this run —
  swap `--gres=gpu:a100:1` — but slower than H100.)

The `.slurm` runs `scripts/train_lora_qwen.py --base google/medgemma-27b-text-it ... --max-len
12288` with **NO `--load-4bit`** (full quality); gradient checkpointing + assistant_only_loss +
the Gemma-3 template swap are on by default.

---

## Why the template swap (the one real gotcha)

`assistant_only_loss=True` needs the chat template to mark the assistant span with a
`{% generation %}` block. Qwen2.5's stock template has it; **stock Gemma-3 / medgemma does
NOT**, so without a fix the loss would land on the whole ~13.5k-char rubric prompt and drown
the signal. `train_lora_qwen.py` handles this automatically:
1. `ensure_assistant_loss_template()` detects the missing marker on a Gemma-family base and
   swaps in `configs/gemma3_assistant_loss_template.jinja` (stock Gemma-3 template + a
   generation block around the model turn; it folds the `system` message into the first user
   turn, as Gemma expects).
2. `validate_assistant_mask()` then runs the template on a real example and **aborts loudly if
   the assistant mask is trivial** (all 0s/1s) — so a broken template fails BEFORE the run,
   not silently. Expect a `[chat-compat] assistant mask OK — N/M tokens ...` line in the log.

To override with your own template: `--chat-template <path>`. To train on the full sequence
instead (not recommended here): `--no-assistant-only-loss`.

---

## Preconditions / checklist
- [ ] `huggingface-cli login` with a token that has **accepted the medgemma license** (gated → else 401).
- [ ] `transformers>=4.50` in the env (older can't load Gemma-3).
- [ ] `git pull` on Quest so `data/training_set/sft/{train,val,test}.jsonl`,
      `configs/gemma3_assistant_loss_template.jinja`, and `experiments/agent/chat_compat.py` are present.
- [ ] SFT data = the **pre-intervention (502-step) cut** (`belief_deviation_filtered.csv`,
      `~excluded_monitoring`) → train 278 / val 67 / test 56 after rrn-alignment. Not pre-admission, not unfiltered.

## After the run
- Adapter → `runs/medgemma-27b-lora-certainty/` (adapter_config.json + adapter_model.safetensors + tokenizer).
- Inspect: `python scripts/inspect_agent_outputs.py --base google/medgemma-27b-text-it --adapter runs/medgemma-27b-lora-certainty --n 6` (**run on the GPU node — 27B is ~54 GB bf16, will NOT fit a Mac MPS**).
- Eval panel: `python scripts/eval_certainty_agent.py --arms doctor base sft ctx-qwen ctx-sonnet --base-model google/medgemma-27b-text-it --adapter runs/medgemma-27b-lora-certainty`.
- Optional: upload adapter to a private HF repo (mirror the Qwen flow) so it's pullable home.
