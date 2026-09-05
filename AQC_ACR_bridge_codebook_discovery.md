# AQC–ACR Bridge Codebook Discovery

Status: active qualitative discovery protocol, started 2026-09-05. The A/Q/C development snapshot
was created before this work and must not be retroactively rewritten to resemble ACR. The
58-patient final-test partition remains sealed.

## 1. Scientific object

ACR describes conditional normative knowledge:

```text
ACR Context -> rated candidate Actions
```

A patient does not arrive with an ACR variant identifier. The clinician must interpret incomplete
and evolving evidence, decide which clinical frame and question matter, judge what prior evidence
has or has not answered, and determine whether an ACR Context applies. A/Q/C is an empirical
reconstruction of that clinician-side state:

```text
patient evidence -> A/Q/C -> patient-to-ACR bridge -> ACR Context/Action -> observed action
```

The object of discovery is the **bridge**, not merely agreement between an order and a guideline
rating. The goal is to identify recurrent bridge work that can be made explicit, structured, and
potentially delegated to AI.

## 2. What one annotation must distinguish

### 2.1 ACR correspondence or deviation

Identify all plausibly applicable ACR variants before judging the observed action. Code Context
correspondence as `exact`, `partial`, `multiple`, `uncertain`, or `out_of_scope`. Separately record
whether the observed action matches an ACR action family/protocol and its rating in each candidate
variant.

An action-family match without a Context match is not guideline concordance. Conversely, an action
that differs from the highest-rated option is not automatically an error: several actions may be
rated similarly, the Context may be only partial, or patient-specific constraints may intervene.

### 2.2 Patient-to-ACR instantiation

Record how patient-specific evidence supports, contradicts, or leaves unknown every material
predicate in the candidate ACR Context:

- presentation and suspected/known condition;
- severity or complication state;
- prior test and prior-result state;
- population, timing, and constraints;
- initial versus next imaging stage.

Instantiation is an interpretive operation, not keyword matching. Missing evidence must remain
`unknown`; it must not be converted into absence.

### 2.3 Bridge operation absent or under-specified in ACR

Record the work needed to move from the patient/AQC state to the candidate Context or action when
ACR does not state that operation explicitly. Candidate operations are initially open-coded.
Sensitizing examples include:

- interpreting whether a prior study addressed this particular Q;
- distinguishing technical adequacy from test–question capability;
- translating nonvisualization, indeterminate findings, or partial coverage into a next-step need;
- refining, replacing, reopening, or advancing a question after a result;
- moving between organ-system or guideline topics;
- resolving multiple partially applicable variants;
- applying patient-specific feasibility constraints to rated actions;
- deciding why a repeat, protocol change, or serial comparison is needed now.

These examples do not constitute a frozen taxonomy. New top-level operations may be added during
open coding, and categories should be merged only after case-level comparison.

### 2.4 Residual

Keep four residual types separate:

- `patient_specific_outside_acr`: relevant facts or needs outside the available variants;
- `practice_deviation_or_local_workflow`: observed practice not explained by the mapped ACR;
- `latent_or_unidentifiable`: preferences, availability, timing, or reasoning not recoverable from
  the record;
- `aqc_reconstruction_concern`: possible order-driven rationalization, unsupported specificity, or
  annotation inconsistency.

The residual is not automatically missing-middle knowledge. It becomes a candidate bridge pattern
only when it performs a necessary mapping function and recurs across cases.

### 2.5 Potential AI delegation

For each bridge operation, code:

- `structure_level`: `rule_like`, `ontology_mapping`, `evidence_synthesis`,
  `contextual_judgment`, or `not_identifiable`;
- `delegability`: `high`, `conditional`, `low`, or `unknown`;
- `required_inputs`: the evidence needed at inference time;
- `safety_boundary`: what requires clinician confirmation;
- `candidate_ai_role`: retrieve, summarize, map, monitor, flag, rank, or abstain.

AI suitability requires more than recurrence. The operation must have observable inputs, a
checkable output, acceptable ambiguity, and a safe abstention/escalation path.

## 3. Unit and evidence rules

The primary unit is one imaging decision step, interpreted within its preceding trajectory. The
annotation may use the effective A/Q/C, pre-order record, resulted prior imaging, observed order,
and the complete relevant ACR topic(s). It must not use later patient outcomes to improve the
mapping.

Every mapping records exact ACR topic/variant/action IDs. `variant_text` and `procedure` remain
authoritative. A/Q/C supplies hypotheses about clinician reasoning; it does not prove private
belief or actual ACR consultation.

## 4. Discovery workflow

1. Freeze the pre-ACR A/Q/C snapshot and its hashes.
2. Select a deliberately diverse development pilot containing easy matches, partial matches,
   cross-topic mappings, sequential imaging, repeat/switch decisions, and likely out-of-scope cases.
3. Manually crosswalk the first cases without a frozen bridge vocabulary.
4. Compare cases and induce a draft bridge-operation codebook.
5. Recode the pilot with the draft codebook; retain counterexamples and unresolved disagreements.
6. Use an LLM as a second coder only after the human draft exists. Give it A/Q/C and faithful ACR
   records, require cited IDs and evidence spans, and measure agreement with human coding.
7. Expand in fresh development batches until no recurrent top-level bridge operation appears.
8. Freeze the bridge codebook and mapping procedure before any final-test replication.

## 5. Initial pilot and stopping rule

The initial pilot contains 12 decision steps across four diseases, chosen for structural coverage,
not prevalence estimation. After the first four-case pass, revise only the bridge annotation
categories, not the underlying A/Q/C annotations or ACR extraction. Add fresh cases when a mapping
type lacks a counterexample or when a new top-level bridge operation appears.

Discovery saturation requires two consecutive fresh batches with no new recurrent top-level bridge
operation. Rare safety-relevant residuals remain documented even if they do not meet recurrence.

## 6. Claims this work can and cannot support

It can identify which parts of clinician bridge work are recurrent, explicit enough to structure,
and candidates for AI assistance. It can also show where ACR Contexts are too coarse, static, or
incomplete for observed longitudinal decisions.

It cannot establish that the reconstructed A/Q/C equals the clinician's private reasoning, that a
deviation is correct, or that an automatable operation is safe to delegate without prospective and
clinician validation. Prediction may later test whether frozen bridge features carry reproducible
decision information, but prediction is downstream validation rather than the discovery target.

## 7. Current artifacts

- `data/aqc_acr_bridge/pilot_v1/sample_manifest.json`: frozen 12-step development pilot.
- `data/aqc_acr_bridge/pilot_v1/manual_crosswalk_round1.jsonl`: first four manual crosswalks.
- `data/aqc_acr_bridge/pilot_v1/manual_crosswalk_round2.jsonl`: remaining eight crosswalks.
- `data/aqc_acr_bridge/pilot_v1/pilot_summary.md`: first-pass synthesis and boundaries.
- `data/aqc_acr_bridge/bridge_codebook_draft_v0_1.json`: provisional B1–B4 codebook.
