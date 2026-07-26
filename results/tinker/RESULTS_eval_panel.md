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
| **c — self-consistency** (same ckpt, K=16 @ T=0.8, vote share) | 0.2116 | +0.1361 | 0.7233 | 0.1444 | 0.249 | **~0 (capped)** |
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

## ⚠️ FINDING: arm c turns a probability question into a deterministic lookup

**(Self-contained; this is the item to pick up in a fresh discussion.)**

### The observation
c's RAW predictions are **100% at 0/1** (median 1.000, p25 0.009) against 20% for arm a. Platt only
rescues them by shrinking 5× harder — slope **0.114 for c vs 0.582 for a** — mapping a near-binary
signal into the 0.25–0.82 range that the calibrated panel reports.

### The mechanism (verified, not inferred)
c's generated trace contains two fields *before* the answer:
```
"modality": "CT_Abdomen"        <- its guess at which study the doctor will order
"rubric_recommended": []        <- the studies it says the rubric wants
"rubric_state": "off_rubric"
"rubric_rationale": "Leading hypothesis is outside the four rubric diseases ('other');
                     the rubric offers no recommendation here."
"deviation": "deviate"          <- the token we score
```
So `deviation` is answering **"is the modality I just wrote inside the list I just wrote?"** — a
comparison of two strings the model has already committed to on paper, not a judgement under
uncertainty. Checked directly on the 56 dumped test traces: **the scored answer equals that
deterministic rule applied to the model's own two fields in 56/56 rows (100%)**.
(Consistent with how the data was built: "follow iff modality ∈ rubric_recommended" reproduces the
gold label 401/401 — see `build_devreason.py`.)

### Where the uncertainty went
It did not vanish; it moved **upstream** into the `modality` guess — which is the genuinely hard
question ("what will this doctor order?"). The model writes ONE definite modality string; it never
writes "probably CT, possibly ultrasound". The doubt is erased at that step, so nothing is left to
express by the time the answer token arrives. Arm a, having no intermediate text, must express all
of its uncertainty in that single word choice — which is exactly why a's predictions are graded
(0.255–0.738) and c's are not. Greedy decoding (`temperature=0.0`) removes the last remaining
source of variation, since only the single most likely trace is ever produced.

### Consequences — these change the RL plan
1. The handoff's headline RL risk ("RLVR → overconfidence → P collapses to 0/1") **has already
   happened in c, with no RL involved.** `c ↔ c+RL` therefore has almost no calibration headroom;
   do not expect that contrast to show anything. The informative RL contrast is **b-NEW pre↔post**.
2. c's ECE 0.042 is **not** evidence that c knows when it is unsure. It is one global Platt slope
   flattening a hard decision. Any "c is better calibrated" claim must state this.
3. **The honest fix is a self-consistency readout**: sample K traces at temperature > 0 and take
   P = the fraction voting `deviate`, instead of one greedy trace + a teacher-forced tail. Different
   sampled traces would guess different modalities, so the vote share is a real graded probability —
   and it is graded *at the step where the uncertainty actually lives*. Costs K× generation.
   **NOT implemented — flagged only.** Applies to c, c+RL and b-NEW alike.
4. Framing for the paper: imposing reasoning structure in the SFT target made the model *recite its
   own conclusion* rather than express a belief. That is a finding about structured-reasoning SFT,
   not a bug in this pipeline.

### SC readout CONFIRMS the mechanism (K=16, existing c checkpoint, no retrain)

Ran `eval_prob_tinker.py --self-consistency 16 --sc-temperature 0.8` on the same c checkpoint
(`results/agent_inspection/tinker_deviation_c_sc.{txt,json}`). Sample K traces per row at T>0, read
each trace's own hard deviate/follow decision, P = vote share (Laplace-smoothed, then the same Platt
path). This tests the claim above: if the doubt really lives in the upstream `modality` guess, then
independent traces should guess *different* modalities and the votes should split.

**They do.** The vote distribution over the 56 test rows:
- **only 6/56 rows are unanimous (16/16); 43/56 (77%) show a genuine split (2–14 of 16 votes).**
- RAW spread goes from the greedy collapse `frac<0.1 or >0.9 = 1.00` to **0.23**, min 0.088 / med
  0.618 / max 0.971 (0.971 is the Laplace cap for K=16 — SC *structurally* cannot report hard 0/1,
  which is honest: 16 samples can't justify >97% confidence).
- The split is predictive, not noise: the top vote bins are enriched for true deviate (16/16 → 6/6
  deviate, 15/16 → 5/5, 12/16 → 6:1) and the low bins for follow — hence AUROC is unchanged.

**Chain of proof.** Within any single trace, `deviation` is a deterministic function of
(`modality`, `rubric_recommended`) — verified 56/56 in the greedy finding above — and
`rubric_recommended` is fixed by the case. So a trace-to-trace *split in the deviation vote* can only
come from a *split in the guessed modality*. 77% split ⇒ the modality guess genuinely varies ⇒ the
uncertainty was live upstream all along and greedy decoding merely hid it. The collapse was a
**decoding artefact, not a knowledge limit.** (User's reorder hypothesis is thus corroborated
without retraining: the graded token was downstream of a hard commitment; sampling that commitment
restores the gradation.)

**What SC buys, and what it costs.**
- Predictive quality is **unchanged**: Brier 0.2116 vs greedy 0.2139, BSS +0.136 vs +0.127, AUROC
  0.723 vs 0.721 — all identical inside the CIs. So the honestly-graded probability is obtained at
  **no loss** of discrimination or Brier. Decision accuracy dips slightly (acc@0.5 0.625–0.66 vs
  greedy 0.679, within noise) — thresholding a graded prob loses a hair vs the model's own argmax.
- ECE(5) **rises 0.042 → 0.144**. This is NOT SC being worse-calibrated; greedy's 0.042 was the
  artefact called out above (one global Platt slope over a binary — nothing to be miscalibrated
  *about*). SC exposes real, non-monotonic structure: it is over-confident-deviate in the
  [0.66,0.74] band (actual 0.333, but n=6 — noisy) and under-confident low (bin [0.07,0.28] actual
  0.417). That is diagnostic and honest; the greedy panel simply couldn't show it. n=56 caveat.

**Takeaway for the plan.** SC is the honest readout layer for c / c+RL / b-NEW alike — it grades the
probability *at the step the uncertainty lives at*, needs no retrain, costs K× generation, and here
recovered a genuine graded signal with no predictive-quality loss. The schema-reorder re-SFT is now
**optional** (only needed if a single-sample structured arm that stays graded is wanted for the
paper); SC already gives the graded c readout the panel needs.

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
