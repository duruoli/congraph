# A/Q/C missing-middle prediction protocol

Status: development design; final test remains sealed.

## Claim under test

The confirmatory claim is not merely that A/Q/C correlates with an observed order. It is that an
order-blinded, pre-decision A/Q/C representation adds reproducible information about the next
observed imaging decision beyond simple clinical-history and prior-imaging baselines.

Prediction alone does not establish causal mediation, order appropriateness, or clinical benefit.

## Leakage boundary

The primary A/Q/C representation must be generated without the current order, current result, or
later events. Existing DIRECT annotations saw the current order and are restricted to a
leakage-prone upper-bound/sensitivity analysis. Removing the `ordered` field after annotation does
not make DIRECT annotations blinded.

Before annotation, audit baseline text for both current-result restatements and explicit prospective
mentions of the target modality/order. Every candidate must be adjudicated. Confirmed target-
revealing text is removed only by exact, hash-bound, non-destructive redaction.

## Primary estimands

1. Among all observed decision points, predict next modality family. Because MRI and CTU are rare,
   the primary class policy must be frozen after development feasibility analysis and before final
   test; collapsing rare classes must be clinically declared rather than chosen from test results.
2. Among noninitial observed decision points, predict repeat versus switch.

Continue versus stop is excluded because the last observed decision is right-censored and does not
establish a clinical stop decision.

## Model comparisons

- Null: training-fold class frequencies only.
- Baseline: step index and prior observed imaging metadata; disease may be used only in a declared
  sensitivity analysis because cohort assignment may encode downstream diagnosis.
- A/Q/C: structured order-blinded assumptions, question, requirements, and coverage.
- Baseline + A/Q/C: primary incremental-value comparison.
- Baseline + frozen pattern indicators: interpretable compression analysis.
- DIRECT A/Q/C: explicitly labeled leakage-prone upper bound, never primary evidence.

Text embeddings or free-text target strings are not part of the first confirmatory model. The first
analysis uses frozen categorical/count features so improvement can be attributed to defined A/Q/C
constructs rather than hidden lexical order clues.

## Splitting and metrics

All splits are grouped by patient. Development uses repeated grouped cross-validation with every
preprocessing and model-selection operation fit inside the training fold. Report macro-F1,
balanced accuracy, multiclass log loss for modality, and AUROC plus log loss for repeat/switch,
with patient-cluster bootstrap confidence intervals for paired model differences.

Also report disease-stratified results and leave-one-disease-out transport checks. These are
stability analyses, not independent replications.

## Gates before final test

1. Clear the development prediction-input leakage audit.
2. Pilot blinded A/Q/C extraction and verify structural validity and semantic stability.
3. Freeze the blinded prompt, schema, model, feature mapping, class policy, missing-value handling,
   cross-validation design, metrics, and pattern codebook.
4. Run and lock development analyses.
5. Open the 58-patient final test once for replication without revising definitions.
