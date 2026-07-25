# Plan T runbook — pred_dev on Tinker (Qwen3.6-35B-A3B)

All four arms train + eval on **Tinker** (managed), base **`Qwen/Qwen3.6-35B-A3B`**. Data is your
existing `data/training_set/{cls,cls_reason}` — unchanged. See `HANDOFF_pred_dev.md` for the design.

| arm | data | training | what it tests |
|---|---|---|---|
| **a** | `cls` (one-word target) | SFT | floor: bare label, no reasoning |
| **c** | `cls_reason` (reasoning + deviation tail) | SFT | structured reasoning IMPOSED |
| **c+RL** | `cls_reason` prompts + label reward | RLVR from the **c** checkpoint | RL on top of imposed structure |
| **b-NEW** | prompts + label reward | RLVR from the **instruct base** | structure DISCOVERED by RL |

> ⚠️ The Tinker method names in `eval_prob_tinker.py` come from the quickstart docs and STILL need
> fixing against the installed SDK (see step 5). The SFT (steps 3–4) is DONE and reuses the
> cookbook's tested `supervised.train`; only the reward + prob-readout are ours.

---

## Step 0 — prerequisites (once)
1. Sign up at the Tinker console, create an API key.
2. **DUA check**: Plan T uploads de-identified MIMIC-derived prompts to Tinker (a new vendor).
   Confirm this is acceptable under your PhysioNet DUA before proceeding.

## Step 1 — install + smoke test  ✅ DONE 2026-07-24
The SDK package is **`tinker`** (NOT `tinker-cookbook`, which is a PyPI stub), and it needs
**Python ≥3.11** — `congraph` is 3.10, so Tinker runs in a DEDICATED env:
```bash
conda create -n tinker python=3.11 -y
/opt/anaconda3/envs/tinker/bin/pip install tinker          # -> tinker 0.23.4 (+ transformers, tokenizers)
```
Key lives in repo-root **`.tinker_env`** (`TINKER_API_KEY=...`), loaded in Python via
`experiments.llm_experiment.env_loader.load_tinker_key()` (NEVER `source` it — smart-quote trap).
Smoke test (verified — `Qwen/Qwen3.6-35B-A3B` present, 28 models):
```bash
/opt/anaconda3/envs/tinker/bin/python -c "
import os,pathlib
for l in pathlib.Path('.tinker_env').read_text().splitlines():
    if l.strip().startswith('TINKER_API_KEY='): os.environ['TINKER_API_KEY']=l.split('=',1)[1].strip()
import tinker; print([m.model_name for m in tinker.ServiceClient().get_server_capabilities().supported_models])"
```
**Run ALL Tinker scripts with `/opt/anaconda3/envs/tinker/bin/python`, not congraph.**
Note: `Qwen3-30B-A3B-Instruct-2507` is ALSO on Tinker (the "retired" claim was stale) — but we
stay on `Qwen3.6-35B-A3B` (newer; Tinker manages the arch).

**`tinker_cookbook` (renderers, dataset builders, `supervised.train`, `rl.train`) is a SEPARATE
install and the PyPI `tinker-cookbook` is a 0.0.0 stub — it MUST come from git:**
```bash
git clone --depth 1 https://github.com/thinking-machines-lab/tinker-cookbook.git ~/tinker-cookbook
/opt/anaconda3/envs/tinker/bin/pip install -e ~/tinker-cookbook
```
✅ DONE 2026-07-24 (`tinker_cookbook 0.1.dev1+gd82f03c8e`; it downgrades transformers 5.14.1 → 5.5.4
and installs torch/datasets/chz — harmless, that env is Tinker-only).

## Step 2 — prep data (converts your JSONL → Tinker conversation files)
```bash
python scripts/tinker/prep_data.py --arm a   # -> data/training_set/tinker/cls/{train,val}.jsonl
python scripts/tinker/prep_data.py --arm c   # -> data/training_set/tinker/cls_reason/{train,val}.jsonl
```
(No Tinker dep; just strips `meta`, keeps `messages`.)

## Steps 3+4 — SFT arms a and c  ✅ DONE 2026-07-24 (commit `dd49202`)
Built as `scripts/tinker/sl_deviate_a.py` / `sl_deviate_c.py`, both thin `chz` entrypoints over
`scripts/tinker/sft_common.py` (shared config + `TwoFileConversationBuilder`), so a↔c differ ONLY
in the data directory.
```bash
/opt/anaconda3/envs/tinker/bin/python scripts/tinker/sl_deviate_a.py log_path=runs/tinker/pred_dev_a
/opt/anaconda3/envs/tinker/bin/python scripts/tinker/sl_deviate_c.py log_path=runs/tinker/pred_dev_c
# any field is CLI-overridable, e.g.  ... learning_rate=1e-4 num_epochs=2 batch_size=4
# smoke test (2 steps):  ... max_steps=2 save_every=2 eval_every=2 num_epochs=1
```
Results, val-NLL curves and all `tinker://` checkpoint paths: **`results/tinker/RESULTS_sft_a_c.md`**.

Three non-obvious choices baked into `sft_common.py` (do NOT "fix" them back):
- **Renderer = `qwen3_5_disable_thinking`, NOT `get_recommended_renderer_name()`'s `qwen3_5`.**
  Qwen3.6 is a hybrid thinking model. Both variants build the SAME supervised example (the empty
  `<think>\n\n</think>\n\n` block is inserted either way, since our targets carry no
  reasoning_content), but the GENERATION PROMPT differs: `qwen3_5` ends at `<think>\n`,
  `qwen3_5_disable_thinking` ends at `<think>\n\n</think>\n\n` — the latter is exactly the prefix
  training saw. Sampling/scoring with the thinking renderer would condition on a prefix that never
  occurred after SFT. **`eval_prob_tinker.py` must use the same renderer.**
- **`TwoFileConversationBuilder`, not the cookbook's `FromConversationFileBuilder`.** The latter
  takes ONE file and carves out its own random held-out set; that would re-split and leak patients
  across our LOCKED seed-0 patient-level 278/67/56 split. Ours reads train.jsonl and val.jsonl as
  given. The val dataset is auto-wrapped by `train.main` as an `NLLEvaluator` → `test/nll` in
  `metrics.jsonl` (all 67 rows, one forward call; batch size must be the full 67 because
  `SupervisedDataset.__len__` floor-divides and 67 is prime — a smaller batch silently drops rows).
- **`train_on_what=LAST_ASSISTANT_MESSAGE`** — identical to ALL_ASSISTANT_MESSAGES for our
  single-turn data, but avoids the extension-property warning (qwen3_5 has
  `has_extension_property=False`).

Verified before launch (Qwen3.6 tokenizer): arm a trains 3 tokens (`deviate<|im_end|>`), arm c
~574 (the JSON trace ending `..."deviation": "deviate"}<|im_end|>`); max prompt 9157 / 10372 tok
vs `max_length=16384`, so nothing truncates — which matters because the label is the LAST token.

**⚠️ Checkpoint selection: use best-val, not `final`.** Arm a's val NLL bottoms at end of epoch 1
(0.2639) and then rises to 0.3132 — it overfits the 278-row shortcut. Arm c peaks at epoch 2
(0.3147) and degrades only slightly. Using `final` for both would confound a↔c with "a trained
past its optimum". See RESULTS for the per-checkpoint paths.

## Step 5 — eval readout (arms a + c) → the panel  ⬅️ NEXT
```bash
# arm a: score follow/deviate right after the prompt  (best-val ckpt = epoch 1)
/opt/anaconda3/envs/tinker/bin/python scripts/tinker/eval_prob_tinker.py \
    --checkpoint tinker://c000136e-4a44-5279-9a0e-a79631aa0835:train:0/sampler_weights/000034 \
    --arm-name a --data data/training_set/cls \
    --out results/agent_inspection/tinker_deviation_a

# arm c: generate the reasoning first, then score the tail  (best-val ckpt = epoch 2)
/opt/anaconda3/envs/tinker/bin/python scripts/tinker/eval_prob_tinker.py \
    --checkpoint tinker://6621e009-cb6c-5fc6-b4de-2a9910f96232:train:0/sampler_weights/000068 \
    --arm-name c --generate-first --data data/training_set/cls_reason \
    --out results/agent_inspection/tinker_deviation_c
```
**`eval_prob_tinker.py` needs 2 fixes first** (checked against the installed SDK 0.23.4):
1. **Prompt building must go through the cookbook renderer, not `tokenizer.apply_chat_template`.**
   Replace the `apply_chat_template(...)` call with
   `get_renderer("qwen3_5_disable_thinking", tok).build_generation_prompt(row["messages"][:2])`
   (see `sft_common.RENDERER_NAME`). The HF template defaults to thinking mode and would end the
   prompt at `<think>\n` — a prefix the SFT'd model never saw. This is the single highest-risk
   line in the eval.
2. **Method names.** `create_sampling_client(model_path=..., base_model=...)` ✓ exists;
   `sample(prompt, num_samples, sampling_params)` and `compute_logprobs(prompt)` ✓ exist, but both
   return a `ConcurrentFuture` (`.result()`), and the `_async` variants are separate methods —
   the current `await sampling_client.sample_async(...)` / `compute_logprobs_async(...)` needs its
   awaiting checked. `compute_logprobs` returns `list[float | None]` over the WHOLE prompt, so the
   existing tail-slicing `lp[-len(ids):]` is right, but guard against `None` entries.
Each writes AUROC/AUPRC/Brier/logloss/acc@0.5/ECE (raw + calibrated, Platt on val→test) —
identical metric block to the Quest eval, so arms compare directly.

**Checkpoint gate:** confirm arms a + c produce sane, parseable output and calibrated ECE before
starting RL. RL only makes sense once c is a solid warm-start.

---

## Step 6 — RL: c+RL  (phase 2)
Adapt the cookbook RL recipe (`tinker_cookbook/recipes/math_rl/`), which subclasses `ProblemEnv`:
1. Copy `math_env.py` → `deviation_env.py`. Replace the math reward with ours:
   ```python
   from scripts.tinker.rl_reward import deviation_reward
   # in the env: get_question() -> the system+user prompt (messages[:2]) of a cls_reason row
   #             score a rollout -> deviation_reward(completion, gold_y=row["meta"]["y"])
   ```
   The `RLDatasetBuilder.__call__` returns your `cls_reason` train rows as the problem set
   (group_size ~8–16 rollouts per prompt so GRPO has within-group reward variance).
2. Copy `rl_basic.py` → `rl_c.py`; set `base_model="Qwen/Qwen3.6-35B-A3B"`, point the
   dataset at `deviation_env`, **warm-start from the arm-c checkpoint** (load the c LoRA as the
   RL init — the field for this is in the RL config; if unclear, ask and we'll check the SDK),
   `lr≈4e-5`, `max_tokens≈1024` (traces reach ~700 tok before the answer).
3. Run; record the checkpoint. Eval with `eval_prob_tinker.py --generate-first --arm-name c_rl`.

## Step 7 — RL: b-NEW  (phase 2)
Same env/reward as step 6, but:
- **Start from the instruct base**, NOT a c checkpoint. The prompt must instruct free reasoning:
  "Reason step by step, then output `deviate` or `follow`." (Keep it UNstructured — do NOT give it
  c's JSON schema; that's the whole manipulation.)
- If the pre-RL base can't reliably emit a parseable answer, do a **format-only** light SFT first
  (target = "<free reasoning> ... deviate/follow" with generic reasoning, NOT c's structured schema).
- Eval BOTH pre-RL and post-RL with `eval_prob_tinker.py --arm-name b_new` to show RL's structuring
  effect.

## Step 8 — assemble the panel
Collect the `tinker_deviation_{a,c,c_rl,b_new}.json` files into one table:

| arm | AUROC | AUPRC | Brier | ECE |
|---|---|---|---|---|
| a (label floor) | | | | |
| c (structured SFT) | | | | |
| c+RL | | | | |
| b-NEW pre-RL / post-RL | | | | |

**Lead with Brier/ECE, not AUROC** — RL is expected to sharpen accuracy while degrading
calibration; that tension is a finding. Contrasts to read: a↔c (structure helps?),
c↔c+RL (RL adds little once structured?), b-NEW pre↔post (RL induces structure?),
c+RL↔b-NEW+RL (imposed vs discovered converge?). N=56 → directional; report with CIs.

---

## What to hand me for phase 2
Once arms a+c run: paste the `ProblemEnv` / `RLDatasetBuilder` base-class signatures from your
installed `tinker_cookbook` (or any first-run error), and I'll write `deviation_env.py` + `rl_c.py`
concretely against the real API instead of the doc summary.
