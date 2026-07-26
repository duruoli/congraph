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
**`eval_prob_tinker.py` was REWRITTEN 2026-07-26 — the two fixes below are DONE**, verified against
the installed SDK 0.23.4 + the real Qwen3.6 tokenizer (offline; no API call made yet):
1. ✅ **Prompt now goes through the cookbook renderer**, `get_renderer(RENDERER_NAME, tok)
   .build_generation_prompt(messages[:2])`, importing `sft_common.RENDERER_NAME` so it can never
   drift from what SFT used. Confirmed the two prompts really do differ:
   disable_thinking ends `...assistant\n<think>\n\n</think>\n\n`, the HF chat template ends
   `...assistant\n<think>\n`. Bonus: under transformers 5.5.4 `apply_chat_template(tokenize=True)`
   returns an `Encoding`, not `list[int]`, so the old `prefix_ids + ids` would have crashed anyway.
   Arm c's conditioning now uses the renderer's `prefill=` argument (verified: prefix bytes
   identical, prefill appended after the generation suffix).
2. ✅ **Awaiting.** `sample_async` / `compute_logprobs_async` ARE true coroutines
   (`inspect.iscoroutinefunction` → True), so the existing `await` was already correct — only the
   sync `sample`/`compute_logprobs` return `ConcurrentFuture`. Tail-slice `lp[-len(ids):]` kept,
   with a hard error (not a silent skip) if any entry is `None`. Rows are now scored with bounded
   concurrency (`--concurrency`, default 8) instead of one at a time.

**Metrics rewritten for a calibrated-probability deliverable** (metric fns still imported from
`eval_deviation_cls.py`, so numbers stay cross-venue comparable):
- PRIMARY: **Brier + CI**, **BSS + CI** (`1 − Brier/Brier(const train base rate)`), **AUROC + CI**,
  **ECE(5 equal-frequency) + reliability table**.
- SECONDARY (appendix line): AUPRC, logloss, acc@0.5, ECE(10 equal-width) = the old Quest number.
- **SPREAD line** (quartiles, sd, frac in [0.4,0.6], frac at the extremes) — catches the two
  failure modes ECE cannot see: a useless-but-calibrated model, and RLVR collapse onto 0/1.
- **A const-base-rate BASELINE row** so every number has an anchor (that row is BSS = 0 by
  definition). Base rate comes from TRAIN, never test.
- `--checkpoint` is now OPTIONAL (omit → bare instruct base, i.e. the b-NEW pre-RL reference);
  `--dump-generations` writes the traces for the qualitative RL audit.

Offline self-tests that passed: const baseline reproduces `q̄(1−q̄)`=0.2449 and BSS=0.0000 exactly;
tie-safe equal-frequency binning (a naive version scored a CONSTANT predictor at ECE 0.11 instead
of 0.008 — the same artefact would hit an RL-collapsed model, which is all ties at 0/1); 200
random partition checks; a synthetic well-calibrated model gives ECE 0.013 / BSS +0.25; a synthetic
RL-collapsed model gives AUROC 0.62 but Brier 0.39 / BSS −0.60 / ECE 0.39 — i.e. the panel does
flag "RL sharpened discrimination while destroying calibration", which AUROC alone would call a win.

⚠️ Reading the output: `follow` is 1 token but `deviate` is 2 on this tokenizer, so raw z is
biased toward `follow` by a near-constant amount. Platt's intercept absorbs it → **read the
CALIBRATED row**; RAW's absolute level is not comparable across arms (ranking/AUROC is unaffected).

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
- **NO structured SFT, ever.** The only SFT allowed is a **format-only** light pass, and only if
  the pre-RL base cannot emit a parseable answer (target = "<free reasoning> … deviate/follow"
  with GENERIC reasoning). Teaching it c's schema would destroy the imposed-vs-discovered contrast.
- Eval BOTH pre-RL and post-RL with `eval_prob_tinker.py --arm-name b_new` to show RL's structuring
  effect.

### Two code gaps b-NEW hits that a and c did not (found 2026-07-26 while rewriting the eval)

**1. b-NEW needs its own PROMPT, and no data dir currently has it.**
`cls/` carries the "answer with one word" system instruction; `cls_reason/` carries c's JSON-schema
instruction. Neither is the free-reasoning prompt, and reusing either silently contaminates the
manipulation (`cls_reason`'s system message would hand b-NEW the structure RL is supposed to
discover). Fix before running b-NEW, pick one:
  - (a) a `scripts/build_bnew.py` that reuses `build_sft_examples`'s split/input verbatim and swaps
    ONLY the system output-instruction → `data/training_set/cls_free/{train,val,test}.jsonl`
    (preferred — same pattern as `build_deviation_cls.py` / `build_devreason.py`, keeps `meta.y`); or
  - (b) a `--system-override` flag on `eval_prob_tinker.py`. Cheaper, but then the RL env and the
    eval read the prompt from two different places — easy to drift. **(a) is the recommendation.**

**2. `--generate-first` cuts at the string `"deviation"`, which is c's JSON key.**
Free reasoning will not emit it, so every b-NEW row would fall into the malformed branch and get
the `, "deviation": "` graft appended — i.e. b-NEW would be scored through c's schema after all.
Needs a b-NEW answer convention, e.g. a `--answer-cue` flag defaulting to c's current behaviour and
set to something like `"\n\nAnswer: "` for b-NEW; the prefill then becomes
`<generated reasoning up to the cue> + cue` and the follow/deviate scoring proceeds unchanged.
Keep it CONSISTENT with `rl_reward.parse_prediction`, which already accepts a bare trailing
`deviate`/`follow` (its regex takes the LAST standalone occurrence — so the cue must not make the
answer word appear anywhere earlier, and free reasoning about "deviating" plausibly will; consider
scoring only after the cue, which the prefill approach already guarantees).

**Already supported, no work needed:** b-NEW **pre-RL** is just `eval_prob_tinker.py` with
`--checkpoint` OMITTED (bare instruct base) — that path exists. `--dump-generations` writes the
traces for the qualitative reward-hacking audit the handoff requires for the RL arms.

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
