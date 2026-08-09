# b-NEW RL reward — bottom-up design from the pre-RL emergence analysis

**What this is.** The b-NEW RLVR reward, designed *from* an analysis of what the bare base already
produces. Method in `HANDOFF_bnew_reward_design.md`. The top-down "structure present" keyword reward
was **100% saturated** (zero GRPO gradient), so we measured emergence first and reward only components
that vary in the base AND have correctness headroom AND are gold-verifiable by exact match (no LLM
judge).

## Data analyzed
Bare-base b-NEW pre-RL greedy traces (concise prompt, renderer `qwen3_5_disable_thinking`,
`max_new_tokens=1024`):
- **278 TRAIN** (`scripts/tinker/gen_bnew_traces.py`) + **56 TEST** = **334 traces**.
- Gold joined from `data/training_set/cls_free/{train,test}.jsonl` `meta`
  (`y`, `effective_branch`, `rubric_recommended`, `how_modality`, `action_role` — added to the build).
- Analyzer: `scripts/tinker/analyze_emergence.py` (deterministic, no model).

## The structural fact behind the design
The gold verdict decomposes deterministically:
- **37% of steps** (`rubric_recommended = "-"`): the hypothesis/state is off-rubric → verdict = **deviate**,
  regardless of anything (skill: recognise off-rubric).
- **63% of steps**: verdict = `deviate ⟺ physician_study ∉ rubric_recommended_set`, keyed on
  `effective_branch`.

So the verdict sits on top of a short clinical-reasoning chain, and **every input to it except the
physician's study is a deterministic function of the given rubric+state** — but the base is bad at
*all* of them, so each is a real, non-saturated, gold-verifiable learning target.

## Emergence table (n=334; base heuristic accuracy → reward decision)

| reasoning node | gold field | variance in base | base accuracy / headroom | **reward decision** |
|---|---|---|---|---|
| **Leading diagnosis** (which disease / `other`=off-rubric) | `effective_branch` (6-way) | not given (60% open differential); named 100% | **0.574 → 0.426** | **REWARD (w=0.25)** |
| **Rubric recommends** (next imaging set, or `none`) | `rubric_recommended` (`|`-set / `-`) | 99% mention *something* | **0.457 → 0.543** (base writes its OWN guess in this slot) | **REWARD (w=0.25)** |
| **Predicted study** (physician's actual next imaging) | `how_modality` (4-way) | spread US/CT/MRCP/(MRI) | **0.536 → 0.464**; mod-right lifts verdict acc 0.489→0.595 | **REWARD (w=0.25)** |
| **Answer / verdict** (follow / deviate) | `meta.y` (2-way) | parseable 88% | **0.546 — BELOW base rate 0.578** → max headroom | **REWARD — DOMINANT (w=1.0)** |
| format (Answer line emitted) | — | 12% ramble to cap | — | **strict GATE → 0** |
| rubric-*reference* / hypothesis-*named* | — | **99% / 100% SATURATED** | — | **DROPPED (dead gradient)** |
| differential / info-gap / expected-finding | — (free text) | 25–61% but gameable | flat/weak causal link | prompt-guide only |
| `action_role` | `action_role` | not in the verdict mechanism | — | DROPPED (orthogonal) |

The decisive change from the failed keyword reward: score each part's **correctness** (~0.46–0.57 in
the base → real variance) instead of its **presence** (saturated → dead). All four rewarded terms are
categorical golds, exact-matchable without a judge.

## The reward (4-term process reward)
The prompt **explicitly** asks for four labelled lines after free-form reasoning prose (prose stays
free; only the summary tail is fixed — see `scripts/build_bnew.py`):
```
Leading diagnosis: <appendicitis|cholecystitis|diverticulitis|pancreatitis|biliary|other>
Rubric recommends: <CT_Abdomen|Ultrasound_Abdomen|MRCP_Abdomen|MRI_Abdomen (|-join several) | none>
Predicted study:   <CT_Abdomen|Ultrasound_Abdomen|MRCP_Abdomen|MRI_Abdomen>
Answer:            <follow|deviate>
```
```
R =  0                                              if Answer not parseable        # STRICT FORMAT GATE
  else
     1.0 · 1[answer == gold y]                      # verdict — DOMINANT (the deliverable)
   + 0.25· 1[dx     == gold effective_branch]       # situate: which disease / off-rubric
   + 0.25· 1[rec_set== gold rubric_recommended set] # apply rubric: next imaging set / none
   + 0.25· 1[study  == gold how_modality]           # predict the physician's actual next study
   − w_len· max(0, n_tokens − 512)                  # soft, capped
```
- Σ(sub-terms)=0.75 < w_ans=1.0 → a correct verdict can never be traded for partial credit on the steps.
- Each sub-field independently scores 0 if its line is absent/wrong; only the verdict is gated (the
  gate forces the 12% of ramblers to commit).
- Code: `scripts/tinker/rl_reward.py` (`bnew_reward(completion, meta)`, self-test passes).

## Validation (real base output under the new prompt)
16-row smoke (`gen_bnew_traces.py` on the 4-field prompt):
- **Field emission (bootstrap): dx 15/16, rec 16/16, study 14/16, Answer 15/16** → the base emits all
  four fields from the prompt instruction alone; **no format warm-start needed.**
- **Reward spread 0.00–1.75, mean 0.64** → strong in-group variance for GRPO.
- **Per-term base correctness: ans 0.31, dx 0.63, study 0.64, rec 0.13.** `rec` is lowest because the
  base writes its *own predicted* study into the "Rubric recommends" slot (it doesn't separate "what
  the guideline wants" from "what I think the doctor does") — precisely the conflation this term is
  meant to break. All four have headroom; none saturated.

## Framing note (a deliberate pivot)
This changes b-NEW from "does RL *discover* structure in free reasoning?" to **"given the required
reasoning steps, does RLVR on the *correctness* of each step build a better agent?"** Every reward
term is exact-match to a categorical gold with no LLM judge in the loop. b-NEW's OUTPUT now mirrors
arm c's structure, but reached by **RLVR-on-correctness** rather than c's **SFT imitation** of a fixed
schema — a clean paper contrast.

## Expected collapse → SC (unchanged)
With `rec` and `study` on the page, the verdict is a deterministic lookup of the model's own two
fields → it collapses to 0/1 like arm c. The graded probability is recovered by the self-consistency
readout (`eval_prob_tinker.py --self-consistency K`), **not** by a calibration term. Eval passes
`--answer-cue $'\nAnswer: '`.

## RESULTS — RLVR run + eval (2026-07-27)
**Training** (`rl_bnew.py`, GRPO group16×32prompts, lr 1e-5, max_tokens 768, warm-start bare base,
7 epochs = 63 steps; epoch-cycling added to `DeviationDataset` so `n_epochs` controls length). Clean
learning curve, NO collapse: over 63 steps reward 0.83→1.17, ans_match 0.47→0.68, dx 0.62→0.77,
rec 0.37→0.51 (base 0.13), std 0.44→0.66, **gated 0.125→0.000** (format fully learned). Final ckpt
`tinker://4f98cb19-3c99-52a3-a4d8-5cac7b75213c:train:0/sampler_weights/final`.

**Held-out TEST eval** (SC K=16, `--answer-cue $'\nAnswer: '`, Platt on val→test, n=56 directional):

| arm | Brier↓ | BSS↑ | AUROC↑ | AUPRC | acc@.5 | ECE(5) |
|---|---|---|---|---|---|---|
| const baseline | 0.2450 | 0 | 0.500 | 0.684 | 0.571 | — |
| b-NEW **pre-RL** (bare, 4-field) | 0.2471 | −0.009 | 0.555 | 0.650 | 0.554 | 0.155 |
| **b-NEW post-RL (SC)** | **0.2154** | **+0.121** | **0.712** | 0.767 | 0.679 | 0.166 |
| a (SFT label) | 0.2233 | +0.089 | 0.702 | — | — | 0.067 |
| c (SFT structured, SC) | 0.2116 | +0.136 | 0.723 | — | — | 0.144 |

READS: (1) **pre→post = chance → real signal** (AUROC CI clears 0.5, Brier beats baseline, BSS
positive) — the informative RL contrast the design predicted (c↔c+RL had no headroom). (2) **b-NEW
post-RL matches arms a/c**: RLVR-on-step-correctness from the bare base reaches structured-SFT quality
with NO imitation target — the paper payoff. (3) Calibration rougher than c (ECE 0.166 vs 0.042
greedy): RLVR sharpens discrimination while reliability lags; SC (unanimous 0.45, up from pre 0.12)
keeps it off the full 0/1 collapse. Files: `results/agent_inspection/tinker_deviation_bnew_{pre4f,rl}.{txt,json}`.

## Files
- analyzer `scripts/tinker/analyze_emergence.py` · generation `scripts/tinker/gen_bnew_traces.py`
- prompt+data `scripts/build_bnew.py` → `data/training_set/cls_free/` (meta carries the 4 golds)
- reward `scripts/tinker/rl_reward.py` (`bnew_reward`)
- NEXT (RL build): `deviation_env.py` (ProblemEnv returning `bnew_reward`) + `rl_bnew.py`
  (GRPO group 8–16, warm-start bare base), then SC eval pre↔post.
