# Plan T — calibrated P(deviate) panel (test n=56)

Produced by `scripts/tinker/eval_prob_tinker.py` (rewritten 2026-07-26, commits `e52498b` +
`a504791`). Raw outputs: `results/agent_inspection/tinker_deviation_<arm>.{txt,json}`.

All arms share: the locked seed-0 patient-level split (train 278 / val 67 / test 56, patient-leak
0), the same causally-masked input prompt, the same binary label (`follow`→0, `{deviate,
off_rubric}`→1), Platt `(a,b)` fit on VAL and applied to TEST, and the same metric code (imported
from `scripts/eval_deviation_cls.py`). Base `Qwen/Qwen3.6-35B-A3B`, renderer
`qwen3_5_disable_thinking`, LoRA r32.

## Panel (Platt-calibrated; lead with Brier/BSS, not AUROC)

| arm | Brier ↓ | BSS ↑ | AUROC ↑ | ECE(5) ↓ | cal. spread (sd) | RAW frac at 0/1 |
|---|---|---|---|---|---|---|
| const base-rate baseline | 0.2450 | +0.0000 | 0.500 | 0.008 | 0.000 | — |
| **a** (SFT direct label, ep1 ckpt) | 0.2233 | +0.0886 | 0.7025 | 0.0668 | 0.150 | 0.20 |
| **c** (SFT structured trace + tail, ep2 ckpt) | **0.2139** | **+0.1267** | **0.7214** | **0.0418** | 0.220 | **1.00** |
| c+RL | _not built_ | | | | | |
| b-NEW pre-RL | _not built_ | | | | | |
| b-NEW post-RL | _not built_ | | | | | |

95% CIs (bootstrap, 2000 draws):
- a — Brier 0.187–0.259, BSS −0.073–+0.243, AUROC 0.569–0.835. Platt a=0.582 b=+0.674.
- c — Brier 0.161–0.267, BSS −0.097–+0.340, AUROC 0.582–0.859. Platt **a=0.114** b=−0.342.

## a ↔ c: does structured reasoning in the target help?

**c wins on every primary metric — and every gap sits well inside the CIs.** Brier 0.214 vs 0.223,
BSS +0.127 vs +0.089, AUROC 0.721 vs 0.703, ECE 0.042 vs 0.067. The direction matches the
prediction (structured reasoning helps), the magnitudes do not support a claim at n=56. Report as
directional, with the CIs, and do not lead with "c is better".

Generation health for c: **56/56 traces emitted the `"deviation"` key**, so the cut landed cleanly
on every row and none fell into the malformed-graft fallback; all four schema fields present;
traces 1638–2712 chars, so `--max-new-tokens 1024` was adequate.

## ⚠️ FINDING: c's raw probability is already fully saturated, BEFORE any RL

The RAW appendix for c shows **100% of predictions at 0/1** (median 1.000, p25 0.009), vs 20% for
arm a. Platt rescues it only by shrinking hard — slope **a=0.114 for c vs 0.582 for a**, a 5×
stronger squash — mapping a near-binary signal into 0.25–0.82.

Why: c generates its own trace first, and by the time it reaches `"deviation": "` it has already
written `rubric_recommended` / `rubric_state` / `rubric_rationale`. The training data was built so
that "follow iff modality ∈ rubric_recommended" reproduces the label 401/401 — so the label is a
near-deterministic function of text the model has already committed to. It is **reading off its own
conclusion, not expressing a graded belief.** Greedy decoding (`temperature=0.0`) makes that trace
deterministic too, so there is no per-case gradation left anywhere in the readout.

**Consequences — these change the RL plan:**
1. The handoff's headline RL risk ("RLVR → overconfidence → P collapses to 0/1") **has already
   happened in c, without RL.** The `c ↔ c+RL` calibration contrast therefore has little headroom
   to show anything; do not expect it to. The informative RL contrast is now b-NEW pre↔post.
2. c's calibrated probability is a hard decision softened by a *global* constant, not a per-case
   uncertainty. Any claim that c is "better calibrated" must say this out loud — its ECE 0.042 is
   an artefact of Platt fitting one slope, not evidence the model knows when it is unsure.
3. **A different readout would give genuine gradation**: sample K traces at temperature > 0 and use
   the fraction voting `deviate` as P (self-consistency), instead of one greedy trace + a
   teacher-forced tail. That is a real design option for c / c+RL / b-NEW, and it is the honest way
   to get a graded probability out of a model whose reasoning commits before the answer. Costs K×
   generation. NOT yet implemented — flagged, not done.

## Reading arm a (the floor)

1. **It discriminates.** AUROC 0.7025 with CI 0.569–0.835 — the lower bound clears 0.5, so the
   model is genuinely reading something predictive out of the pre-decision case, not guessing.
2. **The probability is only weakly better than knowing nothing.** BSS +0.089 means it removed
   ~9% of the baseline uncertainty, and the CI crosses 0. DIRECTIONAL. For behavioural prediction
   off causally-masked inputs this is a plausible real-but-small effect, not a failure — but it is
   not yet a claim, and n=56 cannot make it one.
3. **Calibration is decent and the model is appropriately cautious.** ECE(5 eq-freq) 0.067;
   predictions span 0.255–0.738 with NOTHING beyond 0.9 or below 0.1. It is not collapsing to 0/1.
   That spread is the pre-RL reference point: if `c+RL`/`b-NEW+RL` come back with `frac <0.1 or
   >0.9` near 1.0, that is the predicted RLVR calibration collapse, and BSS/ECE will show it even
   if AUROC improves.
4. **The RAW appendix empirically confirmed the tokenisation caveat.** `deviate` is 2 tokens,
   `follow` is 1, so raw z is biased toward `follow`: raw predictions have median 0.295 and ALL
   FIVE reliability bins have a positive gap (actual rate above predicted). Platt's intercept
   `b=+0.674` corrects it (raw ECE 0.26 → calibrated 0.067). This is why the main panel is the
   calibrated row and raw lives in an appendix.

## Standing caveats (repeat in the paper)

- n=56 ⇒ every contrast is directional; lead with CIs.
- val is used twice: checkpoint selected on val NLL, then Platt fit on the same val. Mild at 2
  parameters, but state it. Test is the only untouched split.
- BSS ceiling is `Var(q)/base < 1`, not 1 — a low BSS is not automatically a bad model.
- The label is belief-conditioned (which sub-rubric is traversed depends on the reconstructed
  belief argmax), so this is "deviation as this project defines it".
