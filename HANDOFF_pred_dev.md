# HANDOFF: pred_dev — a calibrated P(doctor deviates | input)

**Goal.** Predict, at each decision step, the probability that the physician's NEXT imaging
move DEVIATES from the rubric-recommended study. This is **behavioral** prediction — *will the
deviation happen* — NOT whether the deviation is warranted (that would be
verification/appropriateness, a different target). The final deliverable is **one calibrated
probability**, but for two of the three approaches that probability is a **readout on top of
clinical reasoning**, not a bare label map (user's design intuition, 2026-07-20: "reasoning is
underlying, the prob is a front-stage signal, even just a consequence").

**Why supervised, not RL.** The deviation label is OBSERVED (`meta.when_action` from
`belief_step_deviation`). Predicting a probability of an observed outcome = cross-entropy /
calibration, i.e. SFT. RL over an observed-label reward is a degenerate one-step bandit (higher
variance, worse calibration, no gain). RL's real use — counterfactual action exploration — needs
a policy+simulator the project already decided NOT to build. See memory
`deviation-classifier-supervised`, `rubric-update-question-driven`.

---

## Label (LOCKED with user 2026-07-20)

Binary. `follow` → 0 ; `{deviate, off_rubric}` → 1 (off_rubric = the working hypothesis left the
rubric = a form of non-adherence, folded into the positive class). Source = `meta.when_action`.
Base rates: train 0.579 / val 0.507 / test 0.571.

---

## Shared substrate — MUST be identical across a/b/c for a fair comparison

- **Split**: seed-0, patient-level, disease-stratified — the SAME partition as
  `data/training_set/sft/` and `data/training_set/cls/`. train 278 / val 67 / test 56 (36
  patients); patient-leak train∩test = 0.
- **Input**: the same causally-masked prompt (baseline + prior imaging reports + compact
  rubric). The rubric STAYS in the prompt — "deviation" is defined relative to it.
- **Calibration**: Platt scaling `sigmoid(a·z + b)`, `(a,b)` fit on **VAL**, applied to **TEST**.
  Applied to all three arms identically. (Do NOT reweight for class balance — it distorts the
  prior and breaks calibration; let Platt handle it.)
- **Metrics**: AUROC / AUPRC / Brier / log-loss / acc@0.5 / ECE + bootstrap CI, reported RAW and
  CALIBRATED. All implemented and sklearn-verified in `scripts/eval_deviation_cls.py`.
- **N=56 test ⇒ DIRECTIONAL**; CIs wide. Consider hadm-grouped k-fold for descriptive
  calibration only (never for fidelity, given patient-leak).

**Leakage rules (all arms).** Input excludes: the ordered modality (that DEFINES the label),
this step's imaging result, outcome/vindication, disease/ICD god-view. For arm c, the target
reasoning MUST be EX-ANTE — no `verification` / `appropriateness` / outcome in the target, else
the probability leaks the future.

---

## Base model + venue — PLAN T (LOCKED 2026-07-24): everything on Tinker, Qwen3.6-35B-A3B

**Decision:** run ALL pred_dev arms (a, c, c+RL, b-NEW) on **Tinker** (managed training service,
thinkingmachines) with base **`Qwen/Qwen3.6-35B-A3B`** (35B-total / 3B-active MoE, hybrid
Gated-DeltaNet + Gated-Attention, 262K ctx). Cost is not a constraint (user, 2026-07-24).

**Why Tinker + why 3.6 (the back-and-forth, resolved):**
- 3.6 is a NEW arch (multimodal `Qwen3_5MoeForConditionalGeneration`, `model_type qwen3_5_moe`,
  30/40 layers are linear-attention). On **self-hosted Quest** this BREAKS the trl/peft pipeline
  (multimodal class, thin attention-only LoRA over only 10/40 layers, bleeding-edge transformers).
  → 3.6-on-Quest was rejected.
- On **Tinker** the arch is THEIR managed problem → 3.6 "just works," and Tinker also removes the
  RL-rollout infra pain. So committing the whole chain to Tinker dissolves both problems at once.
- DeltaNet's only benefit (cheap 100K–1M context) is IRRELEVANT here (our examples ≤ ~10K tok); it
  is an efficiency arch, NOT a reasoning-quality boost — so picking it costs nothing for this task.
- Tinker's model list is CLOSED (no bring-your-own HF model) and it RETIRED
  `Qwen3-30B-A3B-Instruct-2507` (June 12 2026) → you CANNOT SFT on Quest-2507 then RL that same
  model on Tinker. The venue must be consistent per arm; Plan T makes it consistent everywhere.

**Consequences for this repo:**
- The Quest pred_dev pipeline (`train_lora_qwen.py` + the 4 slurms `train/eval_deviate_cls`,
  `train/eval_devreason`) is **reverted to its committed Qwen3-30B-A3B-2507 state and is NOT USED
  under Plan T** (kept for reference / any non-Tinker fallback). All pred_dev work now lives in
  **`scripts/tinker/`** (see `scripts/tinker/RUNBOOK.md`).
- Data (`data/training_set/cls`, `cls_reason`, `sft`) is UNCHANGED and reused verbatim — only the
  trainer/venue changes. `scripts/eval_deviation_cls.py`'s metric + Platt functions are imported by
  the Tinker eval so the panel stays identical across venues.
- ⚠️ MIMIC DUA: Plan T sends de-identified MIMIC-derived prompts to a NEW vendor (Tinker). You
  already send them to OpenRouter/Anthropic, but confirm Tinker is acceptable under your PhysioNet
  DUA before uploading. (Flagged, not blocking.)

## The approaches (an ablation on the SOURCE of reasoning structure × RL)

**REDESIGNED 2026-07-24 (with user).** The comparison is no longer the old a/b/c "how much
reasoning scaffolds the prob" ladder. Old **b (derive P from the reasoning-tuned agent)** is
RETIRED — the user does not want to train a reasoning-only-no-deviation model — and is preserved
in the **Appendix** below. The live set:

| arm | what it is | reasoning structure comes from |
|---|---|---|
| **a** | base → SFT direct label (one word, NO reasoning) | none (floor) |
| **c** | base → SFT structured reasoning trace + deviation tail | IMPOSED by SFT imitation |
| **c+RL** | RL (RLVR) on top of **c** | imposed, then RL-refined |
| **b (NEW)** | base + prompt-only "reason freely, then judge deviation", then RL | DISCOVERED by RL from free reasoning |

**The experiment = structure-source × RL.** Contrasts:

| contrast | isolates |
|---|---|
| a vs c | does structured reasoning-in-target help the calibrated prob at all |
| c vs c+RL | does RL add anything once structure is SFT'd in (**hypothesis: little**) |
| b(NEW) pre-RL vs post-RL | does RL *induce* useful structure from free reasoning (**hypothesis: a lot**) |
| c+RL vs b(NEW)+RL | imposed vs discovered structure — do they converge |

**RL = RLVR (R1-style), NOT the degenerate action-bandit** the project earlier ruled out. The
verifiable reward = did the predicted `follow`/`deviate` match the OBSERVED label; the optimization
target is the CoT tokens (a huge action space), sampled GRPO-style. This is different from
"RL over an observed-label reward as a one-step bandit" (that was RL *replacing* the label map).

**RL design notes (build into the reward/rollout):**
1. **Do SFT→RL, never R1-Zero-from-base.** `c+RL` warm-starts from the `c` adapter. `b(NEW)` starts
   from the *instruct* base + a prompt constraint (Qwen3.6-35B-A3B is instruction-tuned, so
   "reason then answer" works zero-shot) — if the pre-RL teacher-forced prob readout is unparseable,
   add a **format-only** light SFT (teach the output skeleton ONLY, NOT the structured content — that
   would contaminate the "unstructured" manipulation).
2. **RL fights the calibration deliverable.** RLVR optimizes 0/1 correctness → overconfidence →
   collapses P toward 0/1 (well-documented calibration degradation under RLHF/RLVR). The deliverable
   is a CALIBRATED prob → **report Brier/ECE, not just AUROC**; a plausible outcome is RL wins
   AUROC but loses calibration. Platt-on-val still applied, but it can't fully un-collapse a peaked
   distribution.
3. **Reward = the noisy, belief-conditioned label** → RL can amplify label noise / reward-hack
   (reasoning that's predictive but clinically spurious). AUDIT the RL traces qualitatively, not
   just the metric — matters doubly for an interpretability project.
4. **N=56 test → any RL-vs-SFT difference is directional**, not significant. Frame RL as a
   hypothesis probe.
5. **Naming caution**: "b" now means the free-reasoning→RL arm, NOT the old derive-from-reasoning
   arm (Appendix). Old result panels used "b" for the retired meaning — relabel when comparing.

**Where to run everything: Tinker (Plan T).** All four arms — SFT (a, c) AND RL (c+RL, b-NEW) —
run on Tinker with `Qwen/Qwen3.6-35B-A3B`, per the base-model section above. Runbook +
scripts in `scripts/tinker/`.

### a. base → SFT direct label (NO reasoning) — the floor
- **Idea**: base medgemma, target = ONE word (`follow`/`deviate`). Learns a direct input→label
  shortcut; the prob has no reasoning behind it.
- **Data**: `data/training_set/cls/{train,val,test}.jsonl` — **BUILT** (`scripts/build_deviation_cls.py`).
- **Train**: `python scripts/train_lora_qwen.py --base Qwen/Qwen3.6-35B-A3B
  --data data/training_set/cls --out runs/qwen3.6-35b-a3b-lora-deviate-cls
  --epochs 3 --batch 1 --grad-accum 16 --lr 2e-4` (default max-len 12288 fine). Or just
  `sbatch scripts/train_deviate_cls.slurm` (already wired to Qwen3.6-35B-A3B).
- **Read prob**: `python scripts/eval_deviation_cls.py --adapter runs/qwen3.6-35b-a3b-lora-deviate-cls`
  → `P(deviate)=sigmoid(logprob_deviate − logprob_follow)`, teacher-forced after the generation
  prompt. **BUILT.**
- **STATUS 2026-07-24: TRAINED ON TINKER.** `scripts/tinker/sl_deviate_a.py` (over shared
  `sft_common.py`), 3 epochs × 34 steps, lr 2e-4, LoRA r32, renderer `qwen3_5_disable_thinking`.
  Best-val checkpoint = **epoch 1**,
  `tinker://c000136e-4a44-5279-9a0e-a79631aa0835:train:0/sampler_weights/000034` (val NLL 0.2639).
  **a OVERFITS after epoch 1** (val NLL 0.264 → 0.281 → 0.313) — the predicted 278-row shortcut
  overfit, now observed. Do not use `final`. Full curves: `results/tinker/RESULTS_sft_a_c.md`.

### b. [RETIRED 2026-07-24 → APPENDIX] medgemma-reasoning-tuned → DERIVE the prob
> **This is the OLD arm b, kept for reference only.** The live arm b is now the
> free-reasoning→RL model (see the redesigned table above). Retired because the user does not
> want to train a reasoning-only-no-deviation model. The deriver code below still exists and works;
> it's just no longer part of the live comparison.

- **Idea**: use the EXISTING reasoning SFT adapter; deviation is a pure consequence of the
  model's belief+modality, never trained toward the label. Tests whether clinical reasoning
  alone already implies deviation.
- **Model**: `runs/medgemma-27b-lora-certainty` (the reasoning-trace medgemma SFT; on Quest /
  private HF).
- **Derive**: `P(deviate) = 1 − follow_prob`, where `follow_prob = Σ P(modality ∈ rubric_rec)`
  teacher-forced on the model's OWN generated belief (already computed by
  `scripts/eval_certainty_agent.py`; the `sft` arm's `follow_prob` is per-row in
  `results/agent_inspection/eval_panel_medgemma.jsonl`). off_rubric → follow_prob 0 →
  P(deviate)=1, consistent with the binary map.
- **Calibrate**: TEST follow_probs already exist; need a VAL pass —
  `eval_certainty_agent.py --arms sft --data data/training_set/sft/val.jsonl` — then Platt on
  val, apply to test.
- **STATUS**: **DERIVER BUILT (2026-07-21)** — `scripts/eval_dev_from_reasoning.py` (NO model
  inference: reads a val + test `eval_certainty_agent` panel dump, maps `follow_prob`→raw
  P(deviate)=1−follow_prob, z=logit, Platt on val, applies to test, prints the SAME metric block
  as `eval_deviation_cls.py`; reuses its metric+Platt functions; label-check vs `cls/*.meta.y`
  passes 56/56; supports `--arms sft base`; smoke-tested end-to-end on the test panel). The ONLY
  remaining piece is the **VAL pass on a GPU node** → `scripts/eval_medgemma_agent_val.slurm`
  (clone of `eval_medgemma_agent.slurm`, only `--data`=sft/val + `--out`=*_medgemma_val changed).
  test follow_probs already in `results/agent_inspection/eval_panel_medgemma.jsonl`.

### c. medgemma-base → SFT (reasoning + pred dev) — the target design, RECOMMENDED
- **FROM BASE, not from b.** Rationale: (1) clean ablation vs **a** — both start at base, both are
  label-supervised, the ONLY difference is whether the target contains reasoning, so a↔c isolates
  "does reasoning-in-target help." (2) c's target already contains the reasoning, so the reasoning
  capability is trained anyway — no need to warm-start from b. (3) warm-starting from b would make
  b and c share weights and confound the b↔c contrast.
- **Idea**: target = the full reasoning trace (belief dist / gap / expected / grounding, as in
  `build_sft_examples.py`) + an **explicit rubric-reference block** + a **deviation tail**. The
  reference block (added 2026-07-24, user's "deviate from what?" fix) is three keys between
  `other_hypothesis` and `deviation`: `rubric_recommended` (imaging keys the rubric wants for the
  leading hypothesis), `rubric_state` (recommends_imaging | terminal_* | wants_nonimaging |
  blocked | biliary | off_rubric), `rubric_rationale` (one sentence: which rubric decision-point +
  why). These are **deterministic verbalizations of the rubric's own routing on `eff_branch`**
  (biliary rescue included) — NOT LLM/annotation; ground-truth pulled by re-traversing
  `DISEASE_GRAPHS` on the reconstructed `pre_features` (validated: `rubric_recommended` 401/401 vs
  META, `rubric_state` 401/401 on eff_branch, and "follow iff modality ∈ rubric_recommended"
  reproduces the binary label 401/401). Then `"deviation": follow|deviate` as the LAST key = a
  GROUNDED comparison against a stated reference, not a free-floating guess (kills the
  hallucination path where the model asserts "deviate" with no "should-have-been"). Loss on the
  whole assistant turn → reasoning dominates; the deviation label is a small tail.
- **Read prob**: model GENERATES reasoning first, THEN teacher-force-score the `follow`/`deviate`
  token **conditioned on its own generated reasoning** → P(deviate); Platt-calibrate.
- **Train**: `train_lora_qwen.py --base Qwen/Qwen3.6-35B-A3B --data
  data/training_set/cls_reason --out runs/qwen3.6-35b-a3b-lora-devreason ...` (or
  `sbatch scripts/train_devreason_sft.slurm`, already wired).
- **STATUS**: **BUILT + rubric-reference REVISED 2026-07-24** — (1) `scripts/build_devreason.py`
  → `data/training_set/cls_reason/*` (rebuilt WITH the rubric_recommended/state/rationale keys;
  split identical to cls/sft 278/67/56, leak 0, 0 rows without reconstructable pre; deviation ==
  label AND == the follow-rule for all 278 train rows); (2) `eval_deviation_cls.py --generate-first`
  cut is at the first `"deviation"` occurrence, so the new keys (which never contain that string)
  correctly land in the conditioning context; (3) **`eval_devreason.slurm` `--max-new-tokens`
  raised 512→1024** — the trace now reaches median ~513 / max ~697 tokens BEFORE the deviation key,
  so 512 would truncate ~half the rows into the fallback.
- **STATUS 2026-07-24: TRAINED ON TINKER.** `scripts/tinker/sl_deviate_c.py`, identical config to
  arm a (only the data dir differs, so a↔c isolates target content). Best-val checkpoint =
  **epoch 2**, `tinker://6621e009-cb6c-5fc6-b4de-2a9910f96232:train:0/sampler_weights/000068`
  (val NLL 0.3147). **c does NOT overfit** the way a does — val NLL falls 0.852 → 0.315 and only
  edges back to 0.333 by step 85, i.e. the structured reasoning target acts as the regularizer the
  handoff predicted. NOTE a's and c's val NLLs are NOT comparable to each other (different numbers
  of target tokens); only the `eval_prob_tinker.py` panel compares arms.
- **Optional arm c′**: warm-start c from a reasoning adapter instead of base — tests whether a
  reasoning warm-start helps on N=278. Now largely SUBSUMED by `c+RL` (which does the RL refinement
  the warm-start was groping at); keep only if you want the pure SFT-warm-start ablation. Confounds
  the a↔c contrast → keep SEPARATE.

---

## Comparison plan (the payoff)

Run all four (a, c, c+RL, b-NEW) on the SAME test set, SAME binary label, SAME Platt-on-val
calibration, SAME metrics. Each contrast isolates one factor (structure-source × RL):

| contrast | isolates |
|---|---|
| **a vs c** | does putting structured reasoning IN the target help the calibrated deviation prob? |
| **c vs c+RL** | does RL add anything once structure is SFT'd in (**hypothesis: little**) |
| **b(NEW) pre vs post-RL** | does RL *induce* useful structure from free reasoning (**hypothesis: a lot**) |
| **c+RL vs b(NEW)+RL** | IMPOSED (SFT) vs DISCOVERED (RL) reasoning structure — do they converge |

Report a panel (one row per arm, plus b-NEW pre/post-RL): AUROC/AUPRC/Brier/ECE (raw + calibrated)
+ a reliability plot. **Lead with calibration (Brier/ECE), not just AUROC** — RL is expected to
sharpen accuracy while degrading calibration, and that tension is itself a finding.

---

## Caveats for the paper
- The label is **belief-conditioned** (which sub-rubric is traversed depends on the reconstructed
  belief argmax) → you are predicting "deviation as this project defines it," not a purely
  mechanical fact.
- N small → directional; lead with CIs.
- a can overfit the shortcut on 278 rows; c's structured reasoning is the regularizer.
- The RL arms optimize a NOISY belief-conditioned reward → can reward-hack into predictive-but-
  spurious reasoning; audit RL traces qualitatively. RL calibration degradation is expected.

## Submit sequence — PLAN T (Tinker). Full step-by-step: `scripts/tinker/RUNBOOK.md`

```bash
# 0. sign up + key; DUA check for the Tinker vendor
uv pip install tinker-cookbook && export TINKER_API_KEY=<key>
# 1. convert data (strips meta, keeps messages)
python scripts/tinker/prep_data.py --arm a   # -> data/training_set/tinker/cls/*
python scripts/tinker/prep_data.py --arm c   # -> data/training_set/tinker/cls_reason/*
# 2. SFT a + c: adapt cookbook sl_basic.py -> FromConversationFileBuilder on the tinker/* files,
#    base_model=Qwen/Qwen3.6-35B-A3B. Record each saved tinker:// checkpoint.
# 3. eval readout (same metric block as the Quest eval):
python scripts/tinker/eval_prob_tinker.py --checkpoint <a_ckpt> --arm-name a \
    --data data/training_set/cls          --out results/agent_inspection/tinker_deviation_a
python scripts/tinker/eval_prob_tinker.py --checkpoint <c_ckpt> --arm-name c --generate-first \
    --data data/training_set/cls_reason   --out results/agent_inspection/tinker_deviation_c
# 4. RL (phase 2, after a+c look good): adapt math_rl -> deviation_env using
#    scripts/tinker/rl_reward.deviation_reward; c+RL warm-starts from <c_ckpt>, b-NEW from base.
```

## File map
- **Plan-T (Tinker) — BUILT**: `scripts/tinker/prep_data.py` (JSONL→conversation converter, RAN:
  278/67 each arm), **`scripts/tinker/sft_common.py` + `sl_deviate_a.py` + `sl_deviate_c.py`
  (the SFT recipes — BUILT AND RUN 2026-07-24, both exit 0; see
  `results/tinker/RESULTS_sft_a_c.md`)**, `scripts/tinker/eval_prob_tinker.py` (calibrated P(deviate)
  readout via Tinker SamplingClient, imports the metric+Platt from `eval_deviation_cls.py`; **still
  needs the renderer fix — it uses `tokenizer.apply_chat_template`, which defaults to THINKING mode
  and ends the prompt at `<think>\n`, a prefix the SFT'd model never saw; must use the cookbook
  renderer `qwen3_5_disable_thinking` instead**), `scripts/tinker/rl_reward.py` (RLVR reward =
  1[pred==label] + format bonus, self-test passes), `scripts/tinker/RUNBOOK.md` (the step-by-step).
- **Tinker env**: `tinker_cookbook` must be installed from GIT (PyPI `tinker-cookbook` is a 0.0.0
  stub) — `pip install -e ~/tinker-cookbook` into the py3.11 `tinker` conda env. DONE 2026-07-24.
- **Data (reused verbatim)**: `scripts/build_deviation_cls.py`+`data/training_set/cls/*` (a);
  `scripts/build_devreason.py`+`data/training_set/cls_reason/*` (c); `eval_deviation_cls.py`
  (metric+Platt source of truth).
- **Quest pipeline (NOT used under Plan T)**: `train_lora_qwen.py` + slurms `train/eval_deviate_cls`,
  `train/eval_devreason` — reverted to committed `Qwen3-30B-A3B-Instruct-2507`; kept for reference.
- **TODO**: ~~(1) install Tinker + smoke test~~ DONE; ~~(2) SFT a+c~~ DONE 2026-07-24;
  (3) fix `eval_prob_tinker.py` (renderer + the `ConcurrentFuture` awaiting) and run the a/c
  readout off the best-val checkpoints; (4) phase-2 RL env (`deviation_env.py` + `rl_c.py`) —
  the cookbook is now installed at `~/tinker-cookbook`, so read `tinker_cookbook/rl/` and
  `recipes/math_rl/` signatures directly; (5) assemble the a / c / c+RL / b-NEW panel.
- **Appendix (retired old-b)**: `scripts/eval_dev_from_reasoning.py` + `eval_medgemma_agent_val.slurm`
  + `runs/medgemma-27b-lora-certainty` + `results/agent_inspection/eval_panel_medgemma.jsonl` —
  derive-P-from-reasoning-agent; still functional, out of the live comparison.
