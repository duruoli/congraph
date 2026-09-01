# GPT-5.1 DIRECT second bridge-batch audit

- Prompt SHA-256: `c7f7ffae2271dc9305d0473ae4509b1deab21ee2257341f87694dcc949796ce9`
- Validator: `2.1.0`
- Retry protocol: `1.0.0-validator-feedback`
- Scope: 12 previously unannotated development patients, 22 decision steps, three patients per disease
- Recorded cost including the superseded invalid step and its repair: `$0.762669`
- Final structural validity: 22/22 steps
- Retained calls including superseded attempts: 29

## Validation and repair

The initial run left `appendicitis:21543797:s1` invalid after three attempts because its first-step transition was not `initial`. Repair mode resent only that failed step, retained the superseded version, and accepted the second repair attempt. A final validator 2.1 repair-only scan selected zero invalid trajectories.

Three other accepted steps needed one validator-feedback retry: `cholecystitis:27163978:s1` and `pancreatitis:24742601:s1` for first-step transition, and `diverticulitis:27065737:s1` for missing question evidence. The repaired appendicitis step's first new attempt also lacked question evidence.

## Targeted semantic audit

1. **Order-name specificity correction succeeded.** No accepted step quoted the literal current order in `current_question.evidence`. No question was found whose specific target depended only on the exam name.
2. **Closest specificity candidate was supported independently.** `diverticulitis:24910158:s1` asked about a hepatobiliary source, but the pre-order chart independently documented RUQ as well as RLQ tenderness and a reported history of undefined liver disease. The biliary assumption remained `weakly_supported`; the residual explicitly preserved uncertainty about non-hepatobiliary sources.
3. **Repeat logic improved.** `pancreatitis:23641635:s2`, a repeat CT, explicitly asked about progression or early complications not characterized on the recent CT, assigned `intent_support=weakly_supported`, and retained the missing new trigger/timing in `unsupported_residual`.
4. **Atomic separation was acceptable.** Established syndrome, intervention, and imaging facts were separated from speculative etiologies and complications. Review of all 20 `weakly_supported` assumptions found no clear merger of an established fact with a speculative causal claim.
5. **Uncertainty remained visible.** The batch contained 20 `weakly_supported` assumptions and no `unclear` assumptions. Current-order intent was `well_supported` in 21/22 steps and `weakly_supported` in the repeat-CT step. Seventeen of 22 steps used the maximum five assumptions, so sparsity remains a monitoring item rather than a blocking failure.
6. **Coverage stayed conservative.** Thirteen steps were `unanswered`, nine `partially_answered`, and none `sufficiently_answered` before the current order.

## Decision

The three targeted prompt corrections are sufficiently demonstrated in this bridge batch to permit a larger, still recoverable development batch under the same prompt hash and validator. Continue versioned reporting, review all repeats and low-support outputs, sample at least 20% of other steps, and preserve manual adjudications as overlays rather than overwriting model calls.
