# GPT-5.1 DIRECT bridge-batch audit

- Prompt SHA-256: `d9f0c01a505663454371289ea7f45995b2ea8f897a72053d8e90ad1bd738deb9`
- Validator: `2.0.0`
- Retry protocol: `1.0.0-validator-feedback`
- Scope: 12 previously unannotated development patients, 18 decision steps, three patients per disease
- Recorded cost: `$0.518180`
- Final structural validity: 18/18 steps
- Calls retained: 20; two steps required one validator-feedback retry

## Targeted audit result

The revised prompt improved uncertainty preservation and atomic separation, but the bridge batch does not yet justify expansion.

1. **Order-name over-specificity remains.** `appendicitis:29310170:s1` had only right-lower-quadrant pain, nausea/vomiting, leukocytosis, normal liver tests, and no pre-order imaging. The output nevertheless framed a specific gallstone/cholecystitis question, cited the literal liver/gallbladder ultrasound order as question evidence, and assigned `intent_support=well_supported`. The biliary assumption itself was only `weakly_supported`, and `unsupported_residual` acknowledged that the chart did not explain the biliary focus. This conflicts with the rule that the exam name cannot by itself establish the exact target and that evidence fidelity outranks specificity.
2. **Repeat/previous-study logic was present but not uniformly strong.** `pancreatitis:25976986:s1` incorporated the prior negative CT/RUQ ultrasound into assumptions, coverage, and the unsupported residual, but the primary question did not explicitly encode why a repeated/targeted ultrasound was needed after the earlier study. `diverticulitis:27299061:s1` appropriately retained uncertainty about whether the order was a repeat/confirmatory CT. `pancreatitis:23511843:s1` preserved ambiguity about the earlier CT but described it as an outside/repeated CT more specifically than the visible text established.
3. **Atomicity was generally improved.** Established syndrome/device facts were usually separated from speculative etiologies and complications. No clear established-plus-speculative proposition merger was found in the 18 accepted steps, although several propositions grouped multiple alternative diseases under one broad alternative-source claim.
4. **Uncertainty was retained, but `well_supported` remained dominant.** There were 28 `weakly_supported` and one `unclear` assumptions. Current-order intent was `well_supported` in 17/18 steps and `weakly_supported` in one. Sixteen of 18 steps used the maximum five assumptions, suggesting that output sparsity should continue to be monitored.

## Automated checks

- Aggregate coverage: 13 `unanswered`, 5 `partially_answered`, 0 `sufficiently_answered`.
- `appendicitis:29468247:s1` required one retry for `assumption_0_bad_evidence_count`.
- `pancreatitis:25976986:s1` required one retry for three assumption evidence-count errors and a non-initial first-step transition.
- A validator 2.0 repair-only rescan selected zero invalid trajectories, so no successful step was rerun.

## Decision

Retain all 12 bridge outputs as auditable, versioned development material. Do not merge them silently with the old prompt version and do not expand to the remaining development patients under this prompt hash. Strengthen the prompt and/or validator against using the literal current order as evidence for target specificity, then test the revised version on a fresh, non-overlapping bridge batch.

The confirmed order-specificity error was subsequently corrected in `manual_adjudication.json`. The overlay is authoritative for downstream analysis, while the original model output remains unchanged for audit.
