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
                              -> open patient Context -> ACR-predicate mapping -> candidate ACR Context
pre-order patient evidence -|
                              -> independently reconstructed A/Q/C

candidate ACR Context -> rated Actions -> observed action correspondence/residual
open patient Context + A/Q/C -> test which parts of the bridge A/Q/C captures
```

The object of discovery is the **bridge**, not merely agreement between an order and a guideline
rating. The goal is to identify recurrent bridge work that can be made explicit, structured, and
potentially delegated to AI.

## 2. ACR representation used by this study

### 2.1 Topic, Variant, predicate instance, and predicate type

The current normative corpus contains four selected ACR topics, not the whole ACR catalog:

| ACR topic | Number of ACR Variants |
|---|---:|
| Right Lower Quadrant Pain | 3 |
| Right Upper Quadrant Pain | 5 |
| Left Lower Quadrant Pain | 3 |
| Acute Pancreatitis | 6 |
| Total | 17 |

Each `Variant` is an ACR-authored, table-level clinical scenario under which ACR rates multiple
candidate Actions. The 17 Variants are not project-created clusters or simplified scenarios. Their
authoritative `variant_text` is preserved verbatim. Across the 17 Variants, ACR supplies 141
Context--Action rating pairs; repeated procedures under different Variants remain different pairs.

The current ACR-side representation is:

```text
4 ACR topics
└── 17 source Variants
    ├── normalized condition instances
    │   └── 12 predicate types
    ├── Boolean logic
    └── aggregate relations
```

A **predicate instance** is one judgeable condition that helps constitute a Variant, with its
arguments filled. A **predicate type** is the fixed ACR-side question form shared by comparable
instances, such as `symptom_state(symptom, status, site)` or
`test_interpretation(test, target, result)`. Predicate types standardize ACR wording without
discarding each instance's source phrase, target, scope, or polarity.

The 17 Variants currently compile into 91 condition instances: 59 factual and 32 inferential. This
is an instance count, not a de-duplicated vocabulary size; repeated predicates and members of
explicit alternatives remain separate when needed to preserve source logic. `factual` versus
`inferential` is an orthogonal epistemic kind, not a predicate type. A factual predicate can be
checked from an observation, attribute, time, test, or pathway state. An inferential predicate
requires a diagnostic, severity, uncertainty, confounding, or result-interpretation operation.

The new working files are:

- `data/aqc_acr_bridge/acr_variant_predicate_audit_v1.md`: human-readable review of all 17 Variants;
- `data/aqc_acr_bridge/acr_variant_predicate_audit_v1.json`: machine-readable condition instances,
  source spans, roles, derivations, aggregate structures, and logical groups;
- `data/aqc_acr_bridge/acr_predicate_types_v1.json`: draft predicate-type definitions.

The earlier 50-value/ten-dimension analysis and the patient extraction built from it are archived
under `data/aqc_acr_bridge/archive/ten_dimension_v1/`. They remain provenance records but are not
the active ACR comparison vocabulary or patient-extraction schema. Exact `variant_text` remains
authoritative; the compiled predicate layer remains revisable when a source ambiguity is found.

Keep the two vocabularies distinct:

- `data/acr_normative/native_vocabulary.json` is the legacy finite ACR extraction vocabulary;
- `results/vocab/` normalizes anatomy, attribute, and state expressions in patient evidence.

### 2.2 Open dimensions for patient-Context extraction

A **dimension** is a patient-side extraction question: an intuitive place to record a relevant
fact or judgment from the chart. It serves a different role from an ACR predicate type. The twelve
current predicate-type questions provide seed dimensions so that later comparison is tractable:

```text
patient attributes | symptoms | signs | laboratory findings | imaging findings | diagnoses
prior tests | test interpretations | aggregate assessments | disease timing
diagnosis/presentation stage | imaging stage
```

The dimension set is open, not limited to those twelve seeds. If a relevant item does not fit, the
extractor records `other_proposed_dimension` and gives the proposed dimension a concise name and
definition. Shared seed names make ACR comparison easier; they do not force patient evidence into
a closed ACR ontology. Recurrent proposed dimensions may later become named patient dimensions
without changing the fixed ACR predicate registry.

The fact/inference boundary is still preserved. For example, hypotension is a sign, whereas
`significant deterioration` is an aggregate assessment. A completed ultrasound is a prior-test
fact; calling its result equivocal is a test interpretation linked to that test. Each item carries
its own epistemic kind; its dimension does not replace that label.

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

- `other_proposed_dimension`: recorded during open extraction when a relevant item does not fit a
  seed dimension;
- `unmapped_value_within_dimension`: assigned during ACR mapping when a patient item has a usable
  dimension but no comparable ACR predicate.

The open extractor cannot label an item `unmapped_value_within_dimension`, because the ACR
predicates are hidden in that pass. It preserves the native item and evidence span; the separate
mapping pass determines whether an ACR equivalent exists.

These channels prevent the ACR representation from censoring the missing middle it is meant to
help discover.

## 3. What one annotation must distinguish

### 3.1 Open patient Context before ACR matching

At each imaging decision step, first extract one shared `PatientContext_t` from the causally
available pre-order record. Provide the seed dimension questions and the open
`other_proposed_dimension` channel, but do not expose normalized ACR predicates, Variant
signatures, A/Q/C annotation, or the current order. Preserve native patient wording and source
spans rather than forcing an item into an ACR predicate.

The human task is not to fill every ACR predicate and not to inspect all 17 Variants. The annotator
records the small number of Context items actually supported at that step. Unmentioned patient
dimensions remain empty; ACR-predicate status is assessed only in the mapping and Variant stages.

### 3.2 Patient item to ACR-predicate mapping

In a separate pass, map each open patient-Context item to ACR predicate instances using:

- `exact_or_equivalent`;
- `patient_value_broader`;
- `patient_value_narrower`;
- `related_judgment_required`;
- `contradicted`;
- `no_acr_equivalent`.

The direction of breadth is always **patient item relative to ACR predicate**:

- `exact_or_equivalent`: the patient predicate and ACR predicate have the same operational meaning
  at the relevant assertion status, time, and scope;
- `patient_value_broader`: the patient predicate is less restrictive and does not entail the full
  ACR predicate, such as elevated lipase alone versus ACR's conjunctive `increased amylase and
  lipase`;
- `patient_value_narrower`: the patient predicate entails the ACR predicate but adds location,
  severity, certainty, numeric, etiologic, or protocol detail;
- `related_judgment_required`: the items are clinically connected, but neither equivalence nor
  entailment is available without an additional interpretation, threshold decision, active-question
  judgment, or resolution of AND/OR logic;
- `contradicted`: the patient item directly negates the ACR predicate at compatible time and scope;
  missing evidence, a later encounter stage, or a merely different predicate is not contradiction;
- `no_acr_equivalent`: no ACR predicate represents the item at comparable meaning; the item
  remains an `unmapped_value_within_dimension` and the ACR target is null.

One patient item may map to multiple ACR predicate instances. Each link is retained separately
because equivalent predicates may occur in different source Variants and one compound patient
item may instantiate several ACR predicates. Mapping operates on the item itself; Variant
compatibility and complete Context satisfaction are adjudicated only in the next stage.

This pass may use the predicate registry, source phrases, and Variant signatures, but it must
retain the original patient item and evidence span. For example, `appendix not visualized` must not
be silently converted to an equivocal ultrasound for suspected appendicitis; whether that mapping
holds depends on the active question and is itself a candidate bridge operation.

### 3.3 Candidate Variant generation and adjudication

A program compares the mapped patient predicates with compiled signatures of all 17 Variants and
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

- patient attributes, symptoms, signs, laboratory findings, and imaging findings;
- diagnostic states, test history, test interpretations, and aggregate assessments;
- temporal position, diagnosis/presentation stage, and imaging stage.

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
   events, normalized ACR predicates, Variant signatures, and ACR action ratings. The seed dimension
   questions remain visible.
2. **ACR mapping:** reveal the predicate registry, source phrases, and Variant signatures; continue
   to hide A/Q/C and the current order.
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
2. Compile all 17 source Variants into normalized predicate instances, aggregate relations, and
   explicit logical signatures while preserving exact source text.
3. Define and test the open `PatientContext_t` extraction schema using the twelve seed dimensions,
   source-level labels, `other_proposed_dimension`, and `unmapped_value_within_dimension`.
4. Select approximately 12--20 development decision steps containing straightforward Contexts,
   partial matches, overlapping generic/specific Variants, sequential imaging, cross-topic states,
   and likely out-of-scope cases.
5. Annotate open patient Contexts from pre-order raw records while blinded to A/Q/C, current order,
   normalized ACR predicates, Variant signatures, and action ratings.
6. Map patient items to ACR predicates, mechanically generate candidate Variants, and manually
   adjudicate only the short candidate list and judgment-dependent mappings.
7. Reveal A/Q/C and record which direct mappings, reconstructed judgments, transitions, and
   residuals it captures or misses.
8. Reveal the observed order and Actions/ratings; annotate action correspondence and deviation.
9. Compare cases and induce or revise the bridge-operation codebook. Retain counterexamples and
   unresolved disagreements.
10. Use an LLM as a second coder only after the human procedure is stable. Require evidence spans,
    source-level labels, and cited ACR IDs; measure agreement separately for open extraction,
    predicate mapping, Variant adjudication, and bridge operations.
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
- `data/aqc_acr_bridge/acr_variant_predicate_audit_v1.md`: review table for all 17 ACR Variants.
- `data/aqc_acr_bridge/acr_variant_predicate_audit_v1.json`: condition instances, logic, aggregates,
  and source provenance.
- `data/aqc_acr_bridge/acr_predicate_types_v1.json`: current 12-type ACR predicate registry.
- `experiments/aqc_acr_bridge/prompts.py`: stage-1 blinded open patient-Context extraction prompt.
- `data/aqc_acr_bridge/archive/ten_dimension_v1/`: provenance-only archive of the superseded
  50-value/ten-dimension audit, its 12-case patient extraction and mappings, and their scripts.
