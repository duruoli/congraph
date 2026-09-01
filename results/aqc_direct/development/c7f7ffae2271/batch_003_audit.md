# GPT-5.1 DIRECT expansion batch 003 audit

- Frozen manifest: `data/aqc_direct/batch_c7f7ffae_003.json`
- Prompt SHA-256: `c7f7ffae2271dc9305d0473ae4509b1deab21ee2257341f87694dcc949796ce9`
- Run-time validator: `2.1.0`
- Scope: 32 previously unannotated development patients, 62 decision steps
- Disease counts: appendicitis 9, cholecystitis 9, diverticulitis 5, pancreatitis 9
- Recorded cost: `$2.082012`; one failed network call may not be represented in retained usage
- Final validator 2.1 validity: 62/62 steps
- Retained calls: 77; 15 accepted steps needed one validator-feedback retry

## Interruption and recovery

The first run saved nine patients and then lost its OpenRouter connection before the tenth patient was written. Dynamic `--stratified-new 32` would have selected a shifted set after those nine files existed, so the original selection was reconstructed and frozen in `batch_c7f7ffae_003.json`. The runner now supports `--patient-manifest`; recovery skipped the nine completed files and processed exactly the remaining 23 authorized patients.

## Automated and targeted review

- No accepted question used the literal current order as evidence.
- Coverage before the current order was 35 `unanswered`, 26 `partially_answered`, and one `sufficiently_answered`.
- Outputs contained 79 `weakly_supported` assumptions and one `unclear` assumption.
- Current-order intent was 51 `well_supported` and 11 `weakly_supported`.
- Forty-seven of 62 steps used five assumptions.

The sole `sufficiently_answered` step, `appendicitis:27993727:s2`, was an exact-repeat CT whose question asked about the current extent/complication status. The prior CT alone was incorrectly allowed to answer this new temporal question fully, while the repeat remained `well_supported` despite no documented new trigger. Several other exact repeats correctly mentioned interval change in the question or residual but still used `well_supported` after acknowledging that the timing trigger was missing. Confirmed corrections are recorded in `manual_adjudication_batch_003.json`; model outputs remain unchanged.

A subsequent full evidence-fidelity screen and manual review found 17 low-token-coverage evidence items in seven steps. Most were clinically reasonable absence statements placed incorrectly in `supporting_evidence` rather than `reason`; these were cleared in the overlay. `appendicitis:25665153:s1` was a substantive additional order-driven specificity error: migratory RLQ pain and normal hepatobiliary tests did not support a `well_supported` hepatobiliary question. That step was manually reconstructed around a broader source-localization question with only partial test capability and weak intent support. `pancreatitis:28938137:s1` also received narrower evidence-grounded severity and alternative-etiology propositions. The read-only screening report is `batch_003_content_audit.json`.

All eight steps meeting the strict low-risk definition (no exact repeat, no low/unclear assumption support, no evidence-fidelity alert, and well-supported intent) were reviewed; their assumption, question, coverage, and order-fit content was clinically coherent. Together with 100% review of exact repeats, weak/unclear intent, and evidence-fidelity flags, this supports use of the batch only after applying the adjudication overlay—not as raw model JSON alone.

## Post-adjudication verification

The overlay contains corrections for 13 of the 62 steps. Applying it in memory and rerunning the full batch produced:

- 32/32 manifest patients present;
- 62/62 steps valid under validator `2.2.0`;
- zero low-evidence-fidelity items at the prespecified 0.72 token-coverage screening threshold;
- no changes to the original model result files.

The machine-readable post-adjudication report is `batch_003_content_audit_adjudicated.json`. This verifies schema/rule compliance, evidence traceability, and the targeted clinical coherence review described above; it is not an independent clinical ground-truth adjudication of every proposition.

## Method update

The repeat rule was strengthened after this audit. This historical batch and its overlay remain governed by validator `2.2.0`. The subsequent standalone annotation schema retains `current_order_fit` only as a two-field review package: `question_grounding` records visible-record support for the inferred Q, and `test_question_capability` records whether the test can answer Q. The ambiguous `intent_support`, explanation, and gap fields are removed; no normative appropriateness is annotated. New annotations use wrapper `2.0.0-development`, validator `3.0.0`, and prompt hash `ca9e5be6060aa40099adb947b8be59aa817e12039eac0d4ec020e78beb4e306d`; this historical batch is not silently migrated.

## Decision

Retain batch 003 as versioned development material with the manual overlay applied in analysis. Do not expand under the c7 hash. Test validator 2.2 and the strengthened repeat prompt on a fresh non-overlapping bridge batch before processing the remaining development patients.
