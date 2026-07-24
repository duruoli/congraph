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

## The three approaches (an ablation ladder on "how much reasoning scaffolds the prob")

### a. medgemma-base → SFT direct label (NO reasoning) — the floor
- **Idea**: base medgemma, target = ONE word (`follow`/`deviate`). Learns a direct input→label
  shortcut; the prob has no reasoning behind it.
- **Data**: `data/training_set/cls/{train,val,test}.jsonl` — **BUILT** (`scripts/build_deviation_cls.py`).
- **Train**: `python scripts/train_lora_qwen.py --base google/medgemma-27b-text-it
  --data data/training_set/cls --out runs/medgemma-27b-lora-deviate-cls
  --epochs 3 --batch 1 --grad-accum 16 --lr 2e-4` (default max-len 12288 fine).
- **Read prob**: `python scripts/eval_deviation_cls.py --adapter runs/medgemma-27b-lora-deviate-cls`
  → `P(deviate)=sigmoid(logprob_deviate − logprob_follow)`, teacher-forced after the generation
  prompt. **BUILT.**
- **STATUS**: data + eval built; **training TODO on Quest.**

### b. medgemma-reasoning-tuned → DERIVE the prob (no new training)
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
- **Train**: `train_lora_qwen.py --base google/medgemma-27b-text-it --data
  data/training_set/cls_reason --out runs/medgemma-27b-lora-devreason ...`
- **STATUS**: **BUILT + rubric-reference REVISED 2026-07-24** — (1) `scripts/build_devreason.py`
  → `data/training_set/cls_reason/*` (rebuilt WITH the rubric_recommended/state/rationale keys;
  split identical to cls/sft 278/67/56, leak 0, 0 rows without reconstructable pre; deviation ==
  label AND == the follow-rule for all 278 train rows); (2) `eval_deviation_cls.py --generate-first`
  cut is at the first `"deviation"` occurrence, so the new keys (which never contain that string)
  correctly land in the conditioning context; (3) **`eval_devreason.slurm` `--max-new-tokens`
  raised 512→1024** — the trace now reaches median ~513 / max ~697 tokens BEFORE the deviation key,
  so 512 would truncate ~half the rows into the fallback. **TODO = run train+eval on Quest.**
- **Optional 4th arm c′**: warm-start c from `runs/medgemma-27b-lora-certainty` (b) instead of
  base — tests whether a reasoning warm-start helps on N=278. Keep it SEPARATE from the a/b/c
  ladder (it confounds the a↔c ablation).

---

## Comparison plan (the payoff)

Run all three (four with c′) on the SAME test set, SAME binary label, SAME Platt-on-val
calibration, SAME metrics. Each contrast isolates one factor:

| contrast | isolates |
|---|---|
| **a vs c** | does putting reasoning IN the target help the calibrated deviation prob? |
| **b vs c** | reasoning derived-through-rubric vs an explicitly-trained reasoning-grounded readout |
| **a vs b** | bare shortcut classifier vs pure clinical-reasoning derivation |
| (c vs c′) | does reasoning warm-start add anything on top of reasoning-in-target |

Report a small panel (one row per arm): AUROC/AUPRC/Brier/ECE (raw + calibrated) + a reliability
plot. Expected narrative to test: whether reasoning scaffolding buys calibration/AUROC over the
shortcut, and whether pure reasoning (b) already carries most of the signal.

---

## Caveats for the paper
- The label is **belief-conditioned** (which sub-rubric is traversed depends on the reconstructed
  belief argmax) → you are predicting "deviation as this project defines it," not a purely
  mechanical fact.
- N small → directional; lead with CIs.
- a can overfit the shortcut on 278 rows; b/c's reasoning is the regularizer the comparison is
  designed to expose.

## Submit sequence on Quest (git pull first)

```bash
# --- approach A (bare label) ---
sbatch scripts/train_deviate_cls.slurm          # -> runs/medgemma-27b-lora-deviate-cls
sbatch scripts/eval_deviate_cls.slurm           # -> results/.../deviation_cls_eval.{txt,json}

# --- approach C (reasoning + deviation tail) ---
sbatch scripts/train_devreason_sft.slurm        # -> runs/medgemma-27b-lora-devreason
sbatch scripts/eval_devreason.slurm             # -> results/.../deviation_cls_eval_devreason.{txt,json}

# --- approach B (derive from the existing reasoning SFT; NO training) ---
#   test follow_probs already in results/agent_inspection/eval_panel_medgemma.jsonl.
sbatch scripts/eval_medgemma_agent_val.slurm      # VAL pass -> eval_panel_medgemma_val.jsonl (GPU)
#   then, back home / CPU (no model):
python scripts/eval_dev_from_reasoning.py \
    --val-panel  results/agent_inspection/eval_panel_medgemma_val.jsonl \
    --test-panel results/agent_inspection/eval_panel_medgemma.jsonl --arms sft base
#   -> results/agent_inspection/deviation_dev_from_reasoning_eval.{txt,json}
```
All three write the same metric block (`eval_deviation_cls.py` reporting), so the panel is a
direct arm-by-arm read.

## File map
- **Built**: `scripts/build_deviation_cls.py` + `data/training_set/cls/*` (a data);
  `scripts/build_devreason.py` + `data/training_set/cls_reason/*` (c data);
  `scripts/eval_deviation_cls.py` (a read + `--generate-first` c read, sklearn-verified metrics);
  slurms `train_deviate_cls.slurm` / `eval_deviate_cls.slurm` (a),
  `train_devreason_sft.slurm` / `eval_devreason.slurm` (c).
- **Reuse**: `scripts/build_sft_examples.py`, `scripts/train_lora_qwen.py`,
  `scripts/eval_certainty_agent.py`, `configs/context_block.md`,
  `results/agent_inspection/eval_panel_medgemma.jsonl`, `runs/medgemma-27b-lora-certainty` (b).
- **Built (b)**: `scripts/eval_dev_from_reasoning.py` (1−follow_prob → Platt → shared metrics,
  CPU/no-model) + `scripts/eval_medgemma_agent_val.slurm` (the GPU val pass that feeds it).
- **TODO**: run the slurms on Quest (train a+c, val pass for b), then the CPU deriver for b;
  assemble the a/b/c panel.
