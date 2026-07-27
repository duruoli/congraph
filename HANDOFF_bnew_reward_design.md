# HANDOFF — design the b-NEW RL reward BOTTOM-UP from the base pre-RL trace analysis

**Purpose of this doc:** a self-contained brief for a *fresh conversation* whose single job is:
> Analyze what reasoning structure the bare base ALREADY produces (the b-NEW pre-RL traces), and
> design the b-NEW RLVR reward *from that analysis* — what emerged, what to strengthen, which parts.

Read this + memory `tinker-plan-t-setup` + `deviation-classifier-supervised` first. Invoke the
`tinker:research` skill. Run Tinker scripts with `/opt/anaconda3/envs/tinker/bin/python`; data/analysis
scripts with `/opt/anaconda3/envs/congraph/bin/python` and `PYTHONPATH="$PWD"`.

---

## WHY bottom-up (the mistake this avoids)

Top-down "reward the reasoning parts" failed a reality check: a crude 3-keyword presence reward
(modality / rubric / hypothesis) was **present in 56/56 (100%) of the concise pre-RL traces** →
zero variance → **no GRPO gradient** → the structure reward does nothing. GRPO learns only from
in-group reward *variance*. So a reward term is only useful when the behavior it scores appears
**sometimes but not always** in the base. Hence: measure what actually emerges, then reward where
there is variance + headroom.

## THE ANALYSIS TO RUN (the core of the fresh conversation)

For each candidate reasoning component, classify it on THREE axes and keep only components that pass
all three:

1. **Variance in the base** — presence/value across the pre-RL traces. Saturated (~100%) or absent
   (~0%) → NO usable gradient (absent also can't bootstrap from a bare prompt at all). Want ~30–70%.
2. **Gold-verifiable algorithmically** — is there a per-step gold to exact-match against, WITHOUT an
   LLM judge? (Categorical parts yes; free-text parts no — see below.)
3. **Correctness headroom** — does the base get it WRONG often? If it's already ~right, rewarding it
   teaches little. (Base overall deviation acc is only 30/56 = 0.54, below the 0.57 base rate, so
   the ANSWER itself has plenty of headroom.)

Reward the components that pass all three; for components that pass (1)+(3) but fail (2) (free text),
they can only be *guided by the prompt*, never rewarded (assess their quality POST-HOC with a
one-off LLM pass, which is out of the training loop and therefore allowed).

### Inputs
- **Base pre-RL traces (the thing to analyze):**
  `results/agent_inspection/tinker_deviation_bnew_pre_generations.jsonl`
  = 56 TEST traces, each `{i, y, p, generation}`, from the CONCISE bare prompt (the actual RL
  regime; median ~370 tok, 91% end with the `\n\nAnswer:` cue). To get MORE traces for a richer
  emergence picture, dump the TRAIN split too (eval currently dumps val+test; add train, or just
  re-run eval on a train-as-test copy). 278 train + 56 test would be a stronger sample.
- **Gold reasoning schema (per step):** `data/training_set/train_steps.jsonl`, row fields:
  - `TARGET.why_trace.other_hypothesis`  (= differential; FREE TEXT)
  - `TARGET.why_trace.information_gap`    (FREE TEXT, ~316 chars)
  - `TARGET.why_trace.expected_finding`   (FREE TEXT, ~258 chars)
  - `TARGET.why_trace.action_role`        (**CATEGORICAL**, 5: localize_source 221 / assess_severity
    81 / rule_in 57 / broaden_search 37 / rule_out 5)
  - `TARGET.why_trace.grounding`          (list of 5–14 quoted evidence items; semi-structured)
  - `TARGET.how_modality`                 (**CATEGORICAL**, 4: CT_Abdomen 179 / Ultrasound_Abdomen
    177 / MRCP_Abdomen 36 / MRI_Abdomen 9) — the physician's ACTUAL next study
  - `TARGET.when_action` / `META.dev_belief` (the follow/deviate label = the answer)
  - `META.rubric_recommended`, `META.rubric_state` (what the rubric wants — derivable, not a
    prediction)

### The categorical vs free-text split (already established — do not re-derive, verify)
- **Algorithmically rewardable (exact-match to gold, real variance):** `action_role` (5-way),
  `how_modality` (4-way), the answer (2-way). These ARE reasoning content ("what test / what for /
  follow-or-not"), not keyword presence.
- **NOT algorithmically quality-scoreable:** `differential`, `information_gap`, `expected_finding`,
  `grounding` — free text. Algorithmic reward can only check PRESENCE, which saturates the moment
  the prompt scaffolds them. Their QUALITY needs an LLM judge (ruled out for the reward loop) or
  embedding-similarity-to-gold (a cheaper middle path, still a model but not generative — an OPEN
  option to decide).

### Concrete analysis steps
1. Parse each base trace for the categorical predictions it makes (does it name a study? which of
   the 4? does it state/imply a purpose mappable to the 5 action_roles? does it commit or hedge?).
   Measure the DISTRIBUTION and, where gold exists, the ACCURACY (base vs gold how_modality /
   action_role). Low accuracy + spread = prime reward target.
2. Measure how often each FREE-TEXT gold component is even attempted in the base's free prose (this
   tells you what a general angle-guidance prompt would need to nudge, since these won't bootstrap
   from reward). Expect: the base does differential + grounding-ish reasoning already, but rarely
   labels an explicit "information gap" or "expected finding".
3. Correlate emergent components with correctness (does trace mentioning X predict a right answer?)
   to find which reasoning moves actually matter for the task, not just which are frequent.
4. Output: a table {component → variance, gold-available?, base-accuracy/headroom → reward decision
   (reward / prompt-guide-only / ignore)} and a proposed reward formula with weights.

## DECISIONS ALREADY LOCKED (carry forward, do not relitigate)
- **Algorithmic reward only; NO LLM judge in the training loop.** One-off LLM allowed for POST-HOC
  descriptive analysis of final traces (out of loop, doesn't affect training).
- **Correctness (answer vs gold dev) is the dominant term** — never let structure/format outweigh a
  right answer (keyword-stuffing guard).
- **Collapse is handled by SC, not by the reward.** The self-consistency readout is built
  (`eval_prob_tinker.py --self-consistency K`, K=16); b-NEW post-RL is EXPECTED to collapse to 0/1
  like c and SC recovers the graded P (see `results/tinker/RESULTS_eval_panel.md` § "SC readout
  CONFIRMS the mechanism"). Do NOT add a calibration term unless deliberately revisiting this.
- **Length:** concise prompt → median ~370 tok; budget ≈512, soft capped penalty (never hard
  truncate — it corrupts the answer).
- **Prompt design fork (A vs B):** bare-discovery (A) can only reward things that emerge in free
  text; a rich NAMED schema needs the format in the prompt (B) or it never appears. Current leading
  idea = a HYBRID: **general angle-guidance in prose (guides free reasoning quality, unscored) + a
  minimal PARSEABLE categorical tail** (`Predicted study:` / `Purpose:` / `Answer:`) that IS
  extracted and gold-matched. The free-text reasoning is guided, not extracted (user confirmed you
  cannot keyword-attribute info-gap vs expected-finding in free text).

## PROVISIONAL REWARD (the starting point to refine with the analysis)
```
R = 1[answer == gold_dev]                               # dominant
  + w_mod  · 1[pred_study   == gold how_modality]        # 4-way, exact match, real variance
  + w_role · 1[pred_purpose == gold action_role]         # 5-way, exact match, real variance
  + w_fmt  · [parseable tail present]  (GATE: missing → reward≈0, so structure isn't skipped)
  - w_len  · max(0, n_tokens - 512)                      # soft, capped
```
The analysis should confirm/adjust: which of {modality, action_role, ...} actually have variance +
headroom in the base, the weights, and whether to add embedding-similarity reward for any free-text
part.

## BUILD GAP to fix before RL (regardless of final reward)
- `scripts/build_bnew.py` writes `data/training_set/cls_free/` but its `meta` currently lacks
  `how_modality` and `action_role`. **Add them to `to_example`'s meta** (from `rec["TARGET"]
  ["how_modality"]` and `rec["TARGET"]["why_trace"]["action_role"]`) so the reward can read the gold.
- Reward code lives in `scripts/tinker/rl_reward.py`: `parse_prediction(completion, answer_cue)`,
  `deviation_reward` (c+RL), and a first-cut `bnew_reward` + `structure_components` (the SATURATED
  keyword version — REPLACE its structure term with the analysis-driven categorical-match terms).
  Self-test at `__main__`.

## AFTER the reward is designed (the remaining RL build — separate step)
- `scripts/tinker/deviation_env.py` (a cookbook `ProblemEnv`, adapt `~/tinker-cookbook/recipes/
  math_rl/math_env.py`) returning `bnew_reward` per rollout; `scripts/tinker/rl_bnew.py` entrypoint,
  GRPO group_size 8–16, warm-start from the bare base. Then eval post-RL with SC + `--answer-cue`,
  and fill the panel `results/tinker/RESULTS_eval_panel.md` (b-NEW pre↔post is the informative RL
  contrast; c↔c+RL has no calibration headroom because c already collapsed).

## KEY FILES
- traces: `results/agent_inspection/tinker_deviation_bnew_pre_generations.jsonl`
- gold: `data/training_set/train_steps.jsonl` (`TARGET.why_trace`, `TARGET.how_modality`, `META`)
- data build: `scripts/build_bnew.py` → `data/training_set/cls_free/`
- reward: `scripts/tinker/rl_reward.py`
- eval + SC: `scripts/tinker/eval_prob_tinker.py`
- panel + SC finding: `results/tinker/RESULTS_eval_panel.md`
- c ckpt (for reference): `tinker://6621e009-cb6c-5fc6-b4de-2a9910f96232:train:0/sampler_weights/000068`
