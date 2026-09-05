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
has or has not answered, and determine whether an ACR Context applies. A/Q/C is an independently
developed empirical reconstruction of that clinician-side state. It is a candidate explanation of
the bridge, not the annotation template by which the bridge will be defined:

```text
                              -> open patient Context -> ACR-value mapping -> candidate ACR Context
pre-order patient evidence -|
                              -> independently reconstructed A/Q/C

candidate ACR Context -> rated Actions -> observed action correspondence/residual
open patient Context + A/Q/C -> test which parts of the bridge A/Q/C captures
```

The object of discovery is the **bridge**, not merely agreement between an order and a guideline
rating. The goal is to identify recurrent bridge work that can be made explicit, structured, and
potentially delegated to AI.

## 2. ACR representation used by this study

### 2.1 Topic, Variant, Context value, and dimension are different levels

The current normative corpus contains four selected ACR topics, not the whole ACR catalog:

| ACR topic | Number of ACR Variants |
|---|---:|
| Right Lower Quadrant Pain | 3 |
| Right Upper Quadrant Pain | 5 |
| Left Lower Quadrant Pain | 3 |
| Acute Pancreatitis | 6 |
| Total | 17 |

Each `Variant` is an ACR-authored, table-level clinical scenario. Operationally, one Variant is one
coarse Context under which ACR rates multiple candidate Actions. The 17 Variants are not clusters
or simplified scenarios created by this project. Their authoritative `variant_text` is preserved
verbatim. Across the 17 Contexts, ACR supplies 141 Context--Action rating pairs; repeated procedures
under different Variants remain different pairs.

The project-created Context schema decomposes each exact `variant_text` into ten semantic
dimensions:

1. `presentation`;
2. `condition`;
3. `severity_or_complication`;
4. `prior_test`;
5. `prior_result`;
6. `population`;
7. `timing`;
8. `constraints_or_confounders`;
9. `imaging_stage`;
10. `encounter_status`.

These are ACR-derived analytic dimensions, not an official ACR ontology. Different Variants contain
different sparse subsets of the dimensions. An omitted dimension is unrestricted or unstated by
that Variant; it is not a negative patient predicate.

After decomposition and de-duplication, the 17 Variants contain 50 unique semantic Context values:

| Dimension | Unique values in the current corpus |
|---|---:|
| presentation | 17 |
| condition | 9 |
| severity or complication | 12 |
| prior test | 1 |
| prior result | 1 |
| population | 1 |
| timing | 4 |
| constraints or confounders | 2 |
| imaging stage | 2 |
| encounter status | 1 |
| Total | 50 |

The 50 items are de-duplicated values extracted from ACR wording, not semantic clusters induced
from patient records and not 50 separate Contexts. They are also not uniformly minimal logical
atoms. For example, `negative or equivocal` contains an OR relation, while `increased amylase and
lipase` contains a conjunction. Before deterministic matching, a thin compiled layer must give
such values stable predicate IDs, polarity, and explicit AND/OR or threshold logic. This does not
replace or reinterpret the authoritative Variant text.

Keep the two vocabularies distinct:

- `data/acr_normative/native_vocabulary.json` is the finite ACR Context vocabulary;
- `results/vocab/` normalizes anatomy, attribute, and state expressions in patient evidence.

### 2.2 Operational definitions for patient-Context extraction

The annotation prompt must define the ten dimensions rather than merely name them:

- `presentation`: current symptoms, signs, and laboratory manifestations, excluding diagnostic
  conclusions;
- `condition`: a suspected or known clinical condition organizing the current workup;
- `severity_or_complication`: severity, deterioration, or a complication beyond disease existence;
- `prior_test`: imaging completed before the current decision;
- `prior_result`: the reported or question-relative state of that earlier imaging;
- `population`: a patient group that can change guideline application;
- `timing`: position relative to symptom onset, disease course, intervention, or prior assessment;
- `constraints_or_confounders`: factors limiting interpretation, feasibility, or action choice;
- `imaging_stage`: procedural position such as initial, next, repeat, or post-intervention;
- `encounter_status`: visit-level status such as first-time presentation.

The extraction is sparse. Annotators record supported information and explicit negation; an empty
dimension means `not documented/unknown`, not absence. Every entry retains an exact evidence span
and distinguishes its epistemic source:

- `direct_observation`: symptom, sign, measurement, patient attribute, or completed-test metadata;
- `documented_clinician_judgment`: an assessment explicitly stated in the chart;
- `derived_from_trajectory_metadata`: a procedural fact such as initial versus next imaging;
- `reconstructed_judgment`: a synthesis proposed by the annotator or model but not explicitly
  documented;
- `latent_or_unidentifiable`: the record cannot recover the relevant judgment.

Because all chart text is mediated by documentation, these labels concern the epistemic operation,
not who typed the sentence. A judgment-dependent predicate must not automatically be described as
something the physician did unless the chart documents it. Order-induced inference must remain
separately flagged.

Two open channels are mandatory:

- `unmapped_value_within_dimension`: the patient has a value in one of the ten dimensions, but it
  has no equivalent among the 50 ACR values;
- `additional_dimension_outside_acr_schema`: a relevant feature or operation falls outside the ten
  dimensions altogether.

These channels prevent the ACR representation from censoring the missing middle it is meant to
help discover.

## 3. What one annotation must distinguish

### 3.1 Open patient Context before ACR matching

At each imaging decision step, first extract one shared `PatientContext_t` from the causally
available pre-order record. Provide the ten operational dimension definitions, but do not expose
the 50-value ACR vocabulary, A/Q/C annotation, or current order in this pass. Preserve native
patient wording and source spans rather than forcing a value into an ACR term.

The human task is not to fill 50 Boolean fields and not to inspect all 17 Variants. The annotator
records the small number of Context items actually supported at that step. All other ACR predicates
remain unknown unless explicitly contradicted.

### 3.2 Patient value to ACR-value mapping

In a separate pass, map each open patient-Context item to the finite ACR vocabulary using:

- `exact_or_equivalent`;
- `patient_value_broader`;
- `patient_value_narrower`;
- `related_judgment_required`;
- `contradicted`;
- `no_acr_equivalent`.

This pass may use the 50 ACR values, but it must retain the original open value and evidence span.
For example, `appendix not visualized` must not be silently converted to ACR's `negative or
equivocal ultrasound`; whether that mapping holds depends on the active question and is itself a
candidate bridge operation.

### 3.3 Candidate Variant generation and adjudication

A program compares the mapped patient values with compiled signatures of all 17 Variants and
reports supported, contradicted, and unknown required predicates. It presents only a small ranked
candidate set plus an out-of-scope option for adjudication. The program performs candidate
retrieval, not automatic assignment: generic and specific Variants can overlap, judgment-dependent
predicates remain ambiguous, and ACR does not supply a complete classifier or tie-breaking rule.

After adjudication, code Context correspondence as `exact`, `partial`, `multiple`, `uncertain`, or
`out_of_scope`. Only after this Context judgment should the observed order be revealed and matched
to an ACR action family/protocol and rating under each retained candidate Variant.

An action-family match without a Context match is not guideline concordance. Conversely, an action
that differs from the highest-rated option is not automatically an error: several actions may be
rated similarly, the Context may be only partial, or patient-specific constraints may intervene.

### 3.4 Patient-to-ACR instantiation

Record how patient-specific evidence supports, contradicts, or leaves unknown every material
predicate in the candidate ACR Context:

- presentation and suspected/known condition;
- severity or complication state;
- prior test and prior-result state;
- population, timing, and constraints;
- initial versus next imaging stage.

Instantiation is an interpretive operation, not keyword matching. Missing evidence must remain
`unknown`; it must not be converted into absence.

### 3.5 Bridge operation absent or under-specified in ACR

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

### 3.6 Residual

Keep four residual types separate:

- `patient_specific_outside_acr`: relevant facts or needs outside the available variants;
- `practice_deviation_or_local_workflow`: observed practice not explained by the mapped ACR;
- `latent_or_unidentifiable`: preferences, availability, timing, or reasoning not recoverable from
  the record;
- `aqc_reconstruction_concern`: possible order-driven rationalization, unsupported specificity, or
  annotation inconsistency.

The residual is not automatically missing-middle knowledge. It becomes a candidate bridge pattern
only when it performs a necessary mapping function and recurs across cases.

### 3.7 Potential AI delegation

For each bridge operation, code:

- `structure_level`: `rule_like`, `ontology_mapping`, `evidence_synthesis`,
  `contextual_judgment`, or `not_identifiable`;
- `delegability`: `high`, `conditional`, `low`, or `unknown`;
- `required_inputs`: the evidence needed at inference time;
- `safety_boundary`: what requires clinician confirmation;
- `candidate_ai_role`: retrieve, summarize, map, monitor, flag, rank, or abstain.

AI suitability requires more than recurrence. The operation must have observable inputs, a
checkable output, acceptable ambiguity, and a safe abstention/escalation path.

## 4. Unit, blinding, and evidence rules

The primary unit is one imaging decision step, interpreted within its preceding trajectory. The
authoritative input for open Context extraction is the causally masked pre-order raw record:
baseline history, examination, and laboratory data plus only prior imaging whose result is already
available. Existing `results/evidence_pieces` may support retrieval and normalization, but it is an
admission-level index with incomplete timing and must not be loaded directly as `O_t`.

Information is revealed in stages:

1. **Open extraction:** pre-order raw record only; hide A/Q/C, the current order, its result, later
   events, the 50-value vocabulary, and ACR action ratings.
2. **ACR mapping:** reveal the ACR Context vocabulary and Variant signatures; continue to hide A/Q/C
   and the current order.
3. **A/Q/C comparison:** reveal the pre-existing effective A/Q/C and test which direct and
   judgment-dependent mappings it captures. Do not revise A/Q/C to improve correspondence.
4. **Action comparison:** reveal the observed order and ACR Actions/ratings; code correspondence,
   deviation, and action-level residuals.

Later patient outcomes remain unavailable in every stage.

Every mapping records exact ACR topic/variant/action IDs. `variant_text` and `procedure` remain
authoritative. A/Q/C supplies hypotheses about clinician reasoning; it does not prove private
belief or actual ACR consultation.

## 5. Revised discovery workflow

1. Freeze the pre-ACR A/Q/C snapshot and its hashes.
2. Compile the existing 50-value vocabulary into stable predicate IDs and explicit logical
   signatures for the 17 Variants; do not re-extract or simplify ACR.
3. Define and test the open `PatientContext_t` extraction schema, including source-level labels and
   the two open residual channels.
4. Select approximately 12--20 development decision steps containing straightforward Contexts,
   partial matches, overlapping generic/specific Variants, sequential imaging, cross-topic states,
   and likely out-of-scope cases.
5. Annotate open patient Contexts from pre-order raw records while blinded to A/Q/C, current order,
   the ACR vocabulary, and action ratings.
6. Map open values to the ACR vocabulary, mechanically generate candidate Variants, and manually
   adjudicate only the short candidate list and judgment-dependent mappings.
7. Reveal A/Q/C and record which direct mappings, reconstructed judgments, transitions, and
   residuals it captures or misses.
8. Reveal the observed order and Actions/ratings; annotate action correspondence and deviation.
9. Compare cases and induce or revise the bridge-operation codebook. Retain counterexamples and
   unresolved disagreements.
10. Use an LLM as a second coder only after the human procedure is stable. Require evidence spans,
    source-level labels, and cited ACR IDs; measure agreement separately for open extraction,
    vocabulary mapping, Variant adjudication, and bridge operations.
11. Expand in fresh development batches. Only after the extraction and mapping procedure is
    reliable should it be applied to the remaining 235-patient/433-step development corpus.
12. Freeze the bridge codebook and mapping procedure before any final-test replication.

## 6. Status of the initial pilot and stopping rule

The existing initial pilot contains 12 decision steps across four diseases, chosen for structural
coverage rather than prevalence estimation. Its first-pass crosswalks began with A/Q/C and moved
too quickly to bridge-operation labels. They remain useful exploratory material but are not a
validated implementation of the revised raw-text-first procedure and must not be used to freeze
the codebook. Do not alter the underlying A/Q/C annotations or faithful ACR extraction.

Run a new method-calibration pilot of approximately 12--20 development steps with the staged
blinding above. Add fresh cases when a mapping type lacks a counterexample, when generic and
specific Variants cannot be distinguished, or when a new top-level bridge operation appears.

Discovery saturation requires two consecutive fresh batches with no new recurrent top-level bridge
operation. Rare safety-relevant residuals remain documented even if they do not meet recurrence.

## 7. Claims this work can and cannot support

It can identify which parts of clinician bridge work are recurrent, explicit enough to structure,
and candidates for AI assistance. It can also show where ACR Contexts are too coarse, static, or
incomplete for observed longitudinal decisions.

It cannot establish that the reconstructed A/Q/C equals the clinician's private reasoning, that a
deviation is correct, or that an automatable operation is safe to delegate without prospective and
clinician validation. Prediction may later test whether frozen bridge features carry reproducible
decision information, but prediction is downstream validation rather than the discovery target.

## 8. Current artifacts

- `data/aqc_acr_bridge/pilot_v1/sample_manifest.json`: frozen 12-step development pilot.
- `data/aqc_acr_bridge/pilot_v1/manual_crosswalk_round1.jsonl`: first four manual crosswalks.
- `data/aqc_acr_bridge/pilot_v1/manual_crosswalk_round2.jsonl`: remaining eight crosswalks.
- `data/aqc_acr_bridge/pilot_v1/pilot_summary.md`: first-pass synthesis and boundaries.
- `data/aqc_acr_bridge/bridge_codebook_draft_v0_1.json`: provisional B1–B4 codebook.
