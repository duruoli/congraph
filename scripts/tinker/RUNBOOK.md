# Plan T runbook — pred_dev on Tinker (Qwen3.6-35B-A3B)

All four arms train + eval on **Tinker** (managed), base **`Qwen/Qwen3.6-35B-A3B`**. Data is your
existing `data/training_set/{cls,cls_reason}` — unchanged. See `HANDOFF_pred_dev.md` for the design.

| arm | data | training | what it tests |
|---|---|---|---|
| **a** | `cls` (one-word target) | SFT | floor: bare label, no reasoning |
| **c** | `cls_reason` (reasoning + deviation tail) | SFT | structured reasoning IMPOSED |
| **c+RL** | `cls_reason` prompts + label reward | RLVR from the **c** checkpoint | RL on top of imposed structure |
| **b-NEW** | prompts + label reward | RLVR from the **instruct base** | structure DISCOVERED by RL |

> ⚠️ The Tinker method names in `eval_prob_tinker.py` come from the quickstart docs and may need
> small tweaks against the installed SDK — we iterate on first-run errors. The SFT/RL themselves
> reuse the cookbook's **tested** recipes; only the reward + prob-readout are ours.

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

## Step 2 — prep data (converts your JSONL → Tinker conversation files)
```bash
python scripts/tinker/prep_data.py --arm a   # -> data/training_set/tinker/cls/{train,val}.jsonl
python scripts/tinker/prep_data.py --arm c   # -> data/training_set/tinker/cls_reason/{train,val}.jsonl
```
(No Tinker dep; just strips `meta`, keeps `messages`.)

## Step 3 — SFT arm a (bare label)
Use the cookbook's tested SFT recipe with a conversation-file dataset:
1. Copy `tinker_cookbook/recipes/sl_basic.py` → `sl_deviate_a.py`.
2. Edit these fields:
   - `base_model = "Qwen/Qwen3.6-35B-A3B"`  (LoRA rank 32 is Tinker's default)
   - dataset builder → `FromConversationFileBuilder(file="data/training_set/tinker/cls/train.jsonl")`
     (sl_basic has this line commented — uncomment + point it here; add val similarly)
   - `renderer_name = model_info.get_recommended_renderer_name("Qwen/Qwen3.6-35B-A3B")`
   - train on assistant tokens only (default in the common config), `lr = 2e-4`, `num_epochs = 3`
   - `log_path = "/tmp/tinker/deviate_a"`
3. Run it: `python sl_deviate_a.py`
4. **Record the saved checkpoint path** it prints (a `tinker://...` id) → you pass it to eval as
   `--checkpoint`.

## Step 4 — SFT arm c (reasoning + deviation tail)
Same as step 3 but a second copy `sl_deviate_c.py` with
`FromConversationFileBuilder(file="data/training_set/tinker/cls_reason/train.jsonl")` and
`log_path=/tmp/tinker/deviate_c`. Record its checkpoint id too.

## Step 5 — eval readout (arms a + c) → the panel
```bash
# arm a: score follow/deviate right after the prompt
python scripts/tinker/eval_prob_tinker.py \
    --checkpoint <arm_a_ckpt> --arm-name a \
    --data data/training_set/cls \
    --out results/agent_inspection/tinker_deviation_a

# arm c: generate the reasoning first, then score the tail
python scripts/tinker/eval_prob_tinker.py \
    --checkpoint <arm_c_ckpt> --arm-name c --generate-first \
    --data data/training_set/cls_reason \
    --out results/agent_inspection/tinker_deviation_c
```
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
