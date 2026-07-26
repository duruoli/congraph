# Plan T — calibrated P(deviate) panel (test n=56)

Produced by `scripts/tinker/eval_prob_tinker.py` (rewritten 2026-07-26, commits `e52498b` +
`a504791`). Raw outputs: `results/agent_inspection/tinker_deviation_<arm>.{txt,json}`.

All arms share: the locked seed-0 patient-level split (train 278 / val 67 / test 56, patient-leak
0), the same causally-masked input prompt, the same binary label (`follow`→0, `{deviate,
off_rubric}`→1), Platt `(a,b)` fit on VAL and applied to TEST, and the same metric code (imported
from `scripts/eval_deviation_cls.py`). Base `Qwen/Qwen3.6-35B-A3B`, renderer
`qwen3_5_disable_thinking`, LoRA r32.

## Panel (Platt-calibrated; lead with Brier/BSS, not AUROC)

| arm | Brier ↓ | BSS ↑ | AUROC ↑ | ECE(5) ↓ | pred spread (sd) |
|---|---|---|---|---|---|
| const base-rate baseline | 0.2450 | +0.0000 | 0.500 | 0.008 | 0.000 |
| **a** (SFT direct label, ep1 ckpt) | **0.2233** | **+0.0886** | **0.7025** | 0.0668 | 0.150 |
| c (SFT structured trace + tail, ep2 ckpt) | _pending_ | | | | |
| c+RL | _not built_ | | | | |
| b-NEW pre-RL | _not built_ | | | | |
| b-NEW post-RL | _not built_ | | | | |

95% CIs (bootstrap, 2000 draws): arm a Brier 0.187–0.259, BSS −0.073–+0.243, AUROC 0.569–0.835.

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
