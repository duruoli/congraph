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

The exact ACR phrases were first de-duplicated into 50 semantic Context values. Those values were
then re-audited bottom-up rather than accepting the legacy extraction fields as the ontology. The
result has two epistemic kinds and ten leaf dimensions:

```text
ACR Context condition
├── factual: directly checkable observation, attribute, time, test, or pathway metadata
│   ├── symptoms
│   ├── signs_and_labs
│   ├── patient_characteristics
│   ├── disease_timing
│   ├── prior_test
│   ├── encounter_stage
│   └── imaging_stage
└── inferential: a rule-derived or clinically interpreted state
    ├── diagnosis
    ├── severity_or_complication
    └── evidence_interpretation
```

Here `factual` means that the predicate can be checked from a report, measurement, patient
attribute, relative time, or trajectory metadata without an open-ended clinical interpretation. It
does not assert that the chart is error-free. `inferential` means that instantiating the predicate
requires a diagnostic, severity, complication, uncertainty, confounding, or result-interpretation
operation. A rule-derived label such as SIRS is inferential even when its calculation is
deterministic.

These are project-created, ACR-grounded analytic dimensions, not an official ACR ontology. The
legacy `context` fields in `data/acr_normative` remain a reproducible extraction index but are not
treated as the validated dimension definitions for bridge discovery. Exact `variant_text` remains
authoritative.

The reclassification accounts for all 50 values exactly once by primary semantic role:

| Epistemic kind | Dimension | Unique values |
|---|---|---:|
| factual | symptoms | 8 |
| factual | signs and labs | 12 |
| factual | patient characteristics | 1 |
| factual | disease timing | 4 |
| factual | prior test | 1 |
| factual | encounter stage | 1 |
| factual | imaging stage | 2 |
| inferential | diagnosis | 7 |
| inferential | severity or complication | 8 |
| inferential | evidence interpretation | 6 |
| **Total** | | **50** |

The row-level classification, source Variant IDs, boundary decisions, and compound logic are in
`data/aqc_acr_bridge/acr_context_value_dimension_audit_v1.csv`. The audit is the authority for the
project-created grouping; the exact ACR text remains the authority for each native value.

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

- `symptoms`: patient-reported manifestations and their explicit absence or persistence;
- `signs_and_labs`: observed or measured examination findings, vital signs, laboratory states, and
  their explicit absence or change over time;
- `patient_characteristics`: patient attributes that delimit applicability of the clinical
  scenario;
- `disease_timing`: position relative to symptom onset or disease course, distinct from the imaging
  workflow;
- `prior_test`: a test completed before the current decision, stored with enough metadata to link
  later interpretations to it;
- `encounter_stage`: the current episode's position in the visit or presentation sequence;
- `imaging_stage`: the decision's position in the imaging sequence, such as initial or next;
- `diagnosis`: a suspected, established, challenged, excluded, or unknown disease or etiologic
  frame;
- `severity_or_complication`: a rule-derived or clinically synthesized assessment of severity,
  deterioration, systemic response, or complication;
- `evidence_interpretation`: an assessment of what symptoms, laboratory evidence, or a prior test
  means, including atypicality, uncertainty, a reported imaging finding or limitation, confounding,
  and competing explanations.

The boundary between facts and inferences must be preserved. For example, hypotension and a falling
hematocrit are `signs_and_labs`; the conclusion `significant deterioration` is
`severity_or_complication`. A completed ultrasound is `prior_test`; describing it as negative,
equivocal, limited, or nonvisualizing is `evidence_interpretation` linked to that test. A compound
ACR phrase may therefore compile into predicates in more than one dimension while retaining the
native phrase and its AND/OR logic.

The extraction is sparse. Annotators record supported information and explicit negation; an empty
dimension means `not documented/unknown`, not absence. Every extracted item retains an exact
evidence span and distinguishes its epistemic source:

- `directly_documented_fact`: a symptom, sign, measurement, patient attribute, or completed-test
  fact stated in the visible record;
- `deterministic_derivation`: a threshold, elapsed-time, change, score, or trajectory fact obtained
  by an explicit reproducible rule;
- `documented_clinical_judgment`: a diagnostic, severity, complication, or evidence interpretation
  explicitly stated by a treating clinician or radiologist;
- `reconstructed_judgment`: a synthesis proposed by the annotator or model but not explicitly
  documented.

`latent_or_unidentifiable` is recorded separately when the operation needed to instantiate a
relevant Context condition cannot be recovered. It is not emitted as an ordinary Context item with
an invented value or evidence span.

Because all chart text is mediated by documentation, these labels concern the epistemic operation,
not merely who typed the sentence. A judgment-dependent predicate must not automatically be
described as something the physician did unless the chart documents it. Rule-derived and
reconstructed items must cite their input facts. Order-induced inference is forbidden in the open
pass because the current order is hidden.

Two open channels are mandatory across the staged workflow:

- `unmapped_value_within_dimension`: assigned during the ACR-mapping pass when an extracted patient
  value belongs to one of the ten dimensions but has no equivalent among the 50 ACR values;
- `additional_dimension_outside_acr_schema`: recorded during open extraction when a relevant
  feature or operation does not fit any of the ten dimensions.

The open extractor cannot label a value `unmapped_value_within_dimension`, because the ACR
vocabulary is hidden in that pass. It extracts the native value under its dimension; the separate
mapping pass determines whether an ACR equivalent exists.

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

The direction of breadth is always **patient value relative to ACR value**:

- `exact_or_equivalent`: the patient predicate and ACR predicate have the same operational meaning
  at the relevant assertion status, time, and scope;
- `patient_value_broader`: the patient predicate is less restrictive and does not entail the full
  ACR predicate, such as elevated lipase alone versus ACR's conjunctive `increased amylase and
  lipase`;
- `patient_value_narrower`: the patient predicate entails the ACR predicate but adds location,
  severity, certainty, numeric, etiologic, or protocol detail;
- `related_judgment_required`: the values are clinically connected, but neither equivalence nor
  entailment is available without an additional interpretation, threshold decision, active-question
  judgment, or resolution of AND/OR logic;
- `contradicted`: the patient item directly negates the ACR predicate at compatible time and scope;
  missing evidence, a later encounter stage, or a merely different predicate is not contradiction;
- `no_acr_equivalent`: none of the 50 values represents the item at comparable meaning; the item
  remains an `unmapped_value_within_dimension` and the ACR target is null.

One patient item may map to multiple ACR values. Each link is retained separately because the ACR
vocabulary contains near-synonymous native values with different source Variants (for example,
`elevated WBC count` and `leukocytosis`) and because one conjunctive patient item may instantiate
several atomic ACR values. Mapping operates on the item itself; Variant compatibility and complete
Context satisfaction are adjudicated only in the next stage.

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

- symptoms, signs/labs, and patient characteristics;
- diagnosis and severity/complication judgments;
- prior tests and evidence interpretations linked to those tests;
- disease timing, encounter stage, and imaging stage.

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
- `data/aqc_acr_bridge/acr_context_value_dimension_audit_v1.csv`: complete 50-row audit from
  ACR-native values to factual/inferential dimensions.
- `scripts/validate_acr_context_dimension_audit.py`: corpus-to-audit completeness and source check.
- `experiments/aqc_acr_bridge/prompts.py`: stage-1 blinded open patient-Context extraction prompt.
- `data/aqc_acr_bridge/pilot_v1/open_context_manual_v1/manual_context_items_v1.tsv`: 175-item
  hand-authored open-Context audit across 12 decision steps.
- `data/aqc_acr_bridge/pilot_v1/open_context_manual_v1/manual_acr_vocab_mapping_audit_v1.tsv`:
  item-to-ACR mapping audit, including explicit unmapped items and judgment-dependent links.
- `data/aqc_acr_bridge/pilot_v1/open_context_manual_v1/manual_acr_vocab_mappings_v1.jsonl`:
  machine-readable mapping output retaining the full patient item and ACR source Variant IDs.
- `scripts/map_manual_open_context_to_acr_vocab.py`: reproducible mapping compiler and validator.
