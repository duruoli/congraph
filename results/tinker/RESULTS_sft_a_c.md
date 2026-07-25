# Plan T — SFT arms a and c on Tinker (2026-07-24)

Base `Qwen/Qwen3.6-35B-A3B`, LoRA rank 32, renderer `qwen3_5_disable_thinking`,
`train_on_what=LAST_ASSISTANT_MESSAGE`, max_length 16384, lr 2e-4 linear, batch_size 8,
3 epochs = 34 steps/epoch × 3 = 102 steps. Code commit `dd49202`.

Both runs completed successfully (exit 0). Val = the LOCKED 67-row val split, scored as
`test/nll` by the cookbook's `NLLEvaluator` (all 67 rows in one forward call).

## Val NLL trajectory

| eval step | arm a (`cls`, label-only target) | arm c (`cls_reason`, reasoning + label) |
|---|---|---|
| 0 (base) | 0.3802 | 0.8524 |
| 17 | 0.2973 | 0.3448 |
| 34 (end ep1) | **0.2639 ← best** | 0.3179 |
| 51 | 0.2773 | 0.3157 |
| 68 (end ep2) | 0.2806 | **0.3147 ← best** |
| 85 | 0.3132 | 0.3334 |

## Checkpoints (`sampler_path` — this is what `eval_prob_tinker.py --checkpoint` takes)

arm a — run `c000136e-4a44-5279-9a0e-a79631aa0835`

| name | epoch | sampler_path |
|---|---|---|
| 000034 | 1 | `tinker://c000136e-4a44-5279-9a0e-a79631aa0835:train:0/sampler_weights/000034` **(best val)** |
| 000068 | 2 | `tinker://c000136e-4a44-5279-9a0e-a79631aa0835:train:0/sampler_weights/000068` |
| final | 3 | `tinker://c000136e-4a44-5279-9a0e-a79631aa0835:train:0/sampler_weights/final` |

arm c — run `6621e009-cb6c-5fc6-b4de-2a9910f96232`

| name | epoch | sampler_path |
|---|---|---|
| 000034 | 1 | `tinker://6621e009-cb6c-5fc6-b4de-2a9910f96232:train:0/sampler_weights/000034` |
| 000068 | 2 | `tinker://6621e009-cb6c-5fc6-b4de-2a9910f96232:train:0/sampler_weights/000068` **(best val)** |
| final | 3 | `tinker://6621e009-cb6c-5fc6-b4de-2a9910f96232:train:0/sampler_weights/final` |

## Reading

1. **Arm a overfits after one epoch**; arm c does not (it improves through epoch 2 and degrades
   only slightly by step 85). This is exactly the handoff's predicted "a can overfit the shortcut
   on 278 rows; c's structured reasoning is the regularizer" — now observed, not just asserted.
   Consequence: **do NOT default to `final` for arm a** — use the epoch-1 checkpoint, or the
   a↔c contrast is confounded by a being trained past its optimum.
2. **The two arms' val NLLs are NOT comparable to each other.** Arm a's is a mean over ~3 target
   tokens (the label word + `<|im_end|>`, so partly diluted by a near-deterministic EOS); arm c's
   is a mean over ~500 reasoning tokens of which the label is a tiny tail. Only the calibrated
   panel from `eval_prob_tinker.py` (AUROC/AUPRC/Brier/ECE on the same 56-row test set) compares
   the arms. Val NLL here is a *within-arm* checkpoint-selection signal only.
3. Base-model val NLL differs hugely (0.38 vs 0.85) for the same reason — arm c's target is a
   long structured JSON the base model has never seen, so it starts far worse and has much more
   to learn. Its drop (0.85 → 0.31) is mostly learning the FORMAT, not the label.

## Caveat on checkpoint selection

Selecting on val NLL and then fitting the Platt (a, b) on the SAME val split reuses val twice.
With n=67 and a 2-parameter calibration this is mild, but it should be stated in the paper:
test-set numbers are the only untouched ones, and N=56 makes everything directional anyway.
