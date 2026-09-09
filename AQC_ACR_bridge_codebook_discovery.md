# AQC–ACR Bridge Codebook Discovery

Status: active qualitative discovery protocol, started 2026-09-05; last updated 2026-09-09. The
A/Q/C development snapshot was created before this work and must not be retroactively rewritten to
resemble ACR. The 58-patient final-test partition remains sealed.

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
    │   └── 11 predicate types
    ├── Boolean logic
    └── aggregate relations
```

A **predicate instance** is one judgeable condition that helps constitute a Variant, with its
arguments filled. A **predicate type** is the fixed ACR-side question form shared by comparable
instances, such as `symptom_state(symptom, site, state)` or
`test_interpretation(test, target, result)`. Predicate types standardize ACR wording without
discarding each instance's source phrase, target, scope, or polarity.

The 17 Variants currently compile into 78 condition instances: 46 factual and 32 inferential. This
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
The phrases `Initial imaging` and `Next imaging study` remain in authoritative `variant_text` but
are not compiled as predicates. Their condition-relative care-episode boundary is not operationally
stable in the patient records; explicit prior studies and results remain represented through
`test_history` and `test_interpretation`.

Keep the two vocabularies distinct:

- `data/acr_normative/native_vocabulary.json` is the legacy finite ACR extraction vocabulary;
- `results/vocab/` normalizes anatomy, attribute, and state expressions in patient evidence.

The final five global Variants (Acute Pancreatitis Variants 2–6) require especially careful use of
the compiled structure:

- Variant 13: `atypical_presentation` is the required aggregate. Equivocal amylase **and** lipase,
  possible confounding renal disease, and a possible non-pancreatitis diagnosis characterize that
  assessment; they are not ordinary Variant-level peers.
- Variant 14: `critical_illness` is the required aggregate. SIRS and a severe clinical score are
  indicators; APACHE-II, BISAP, and Marshall are three alternative sub-predicates of the clinical-
  severity-score assessment.
- Variant 15: established acute pancreatitis, persistent SIRS, severe clinical scores,
  leukocytosis, fever, and the stated 7–21-day temporal position are compiled conjunctively.
- Variant 16: known necrotizing pancreatitis and `significant_clinical_deterioration` are required;
  the listed abrupt laboratory/vital changes provide alternative evidence for the aggregate rather
  than independent required peers.
- Variant 17: established acute pancreatitis **and** a known pancreatic/peripancreatic collection
  **and** one or more of persistent abdominal pain, early satiety, nausea, vomiting, or signs of
  infection **and** more than four weeks after symptom onset.

### 2.2 Open dimensions for patient-Context extraction

A **dimension** is a patient-side extraction question: an intuitive place to record a relevant
fact or judgment from the chart. It serves a different role from an ACR predicate type. The eleven
current predicate-type questions provide seed dimensions so that later comparison is tractable:

```text
patient attributes | symptoms | signs | laboratory findings | imaging findings | diagnoses
prior tests | test interpretations | aggregate assessments | disease timing
diagnosis/presentation stage
```

The dimension set is open, not limited to those eleven seeds. If a relevant item does not fit, the
extractor records `other_proposed_dimension` and gives the proposed dimension a concise name and
definition. Shared seed names make ACR comparison easier; they do not force patient evidence into
a closed ACR ontology. Recurrent proposed dimensions may later become named patient dimensions
without changing the fixed ACR predicate registry.

The fact/inference distinction remains visible through the seed dimensions. For example,
hypotension is a sign, whereas `significant deterioration` is an aggregate assessment. A completed
ultrasound is a prior-test fact; calling its result equivocal is a test interpretation.

The extraction is sparse. Annotators record supported information and explicit negation; an empty
dimension means `not documented/unknown`, not absence. Predicate arguments use concise normalized
values, and every item retains an exact source evidence span.

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
`other_proposed_dimension` channel. Use concise normalized argument values while retaining exact
native wording in source evidence spans.

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
- temporal position and diagnosis/presentation stage.

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

## 4. Unit, evidence construction, and staged access

The primary unit is one imaging decision step, interpreted within its preceding trajectory. The
authoritative extraction input is the evidence that was available before that order. Hybrid
extraction partitions the **sources**, not the predicate types: deterministic code and the LLM may
both emit, for example, a sign or laboratory predicate when different source evidence supports it.

### 4.1 Deterministic source stream

The implemented algorithmic extractor currently produces:

- `lab_finding_state` from the structured laboratory JSON, reference ranges, and laboratory
  metadata;
- `sign_state` from explicitly labelled vital values in `Physical Examination`, retaining only the
  latest value for the same sign;
- `test_history` from metadata for all visible, already resulted prior imaging studies, retaining
  the latest item for the same normalized test identity.

It does not infer narrative examination findings, imaging findings, diagnoses, or integrated
assessments. It also does not infer generic longitudinal `change` fields or create a separate
`imaging_stage` predicate.

### 4.2 LLM text-complement stream

The LLM receives only the text needed for extraction:

- the same effective pre-order HPI used by the A/Q/C pipeline, including reviewed redactions of
  any sentence that reveals the current imaging result;
- `Physical Examination` after only the exact labelled-vital spans captured by the algorithm have
  been replaced by `[captured_vital]`;
- each visible prior resulted imaging report, preceded by a minimal modality/region/exam header.

The raw structured laboratory table and the algorithm's predicate output are not included in the
LLM request. Nevertheless, all eleven dimensions remain available in the LLM output: HPI,
examination prose, and imaging reports may mention a laboratory result, sign, or test that is not
available in the corresponding structured source.

The LLM prompt contains only the extraction task, the eleven precise dimension definitions, the
small predicate-specific JSON template, the requirement for exact evidence spans, the open
`other_proposed_dimension` channel, and the rule that undocumented information is unknown rather
than absent. Pipeline-specific background, A/Q/C, ACR Variants, current orders/results, and merge
instructions do not belong in the prompt.

Raw `Patient History` cannot automatically be treated as safe: some records restate the current
hidden report. The pilot runner must reuse the established A/Q/C input pipeline's effective HPI and
review decisions, verify hash-bound redactions, and fail closed on unresolved current-result
leakage. The current leakage-review artifact is not directly compatible with the older
`load_reviews()` contract, so it must be inspected or adapted rather than passed through blindly.
Existing `results/evidence_pieces` may aid retrieval and normalization, but it is an admission-level
index with incomplete timing and must not be loaded directly as the decision-step observation.

### 4.3 Staged access

The following are pipeline controls, not prose to add to the LLM prompt:

1. **Open extraction:** use only the effective pre-order inputs above. A/Q/C, the current order and
   result, later events, normalized ACR predicates, Variant signatures, and ratings remain
   unavailable.
2. **ACR mapping:** reveal the predicate registry, source phrases, aggregate structures, and Variant
   signatures; continue to hide A/Q/C and the current order.
3. **A/Q/C comparison:** reveal the pre-existing effective A/Q/C and test which direct and
   judgment-dependent mappings it captures. Do not revise A/Q/C to improve correspondence.
4. **Action comparison:** reveal the observed order and ACR Actions/ratings; code correspondence,
   deviation, and action-level residuals.

Later patient outcomes remain unavailable in every stage. Every mapping records exact ACR
topic/Variant/action IDs. `variant_text` and `procedure` remain authoritative. A/Q/C supplies
hypotheses about clinician reasoning; it does not prove private belief or actual ACR consultation.

## 5. Current implementation status and next task

### 5.1 Stable foundations

- [x] Freeze the pre-ACR development set and the structurally purposive 12-step pilot (11 unique
  patients across all four diseases); the final-test partition remains sealed.
- [x] Compile all 17 ACR Variants as 78 predicate instances with explicit Boolean logic, aggregate
  hierarchies, and source provenance.
- [x] Retain 11 ACR predicate types and remove `imaging_stage` from the active predicate layer.
- [x] Implement and test the deterministic extractor and concise LLM text-complement contract.
- [x] Archive the superseded ten-dimension work under `archive/ten_dimension_v1/`.

### 5.2 Completed manual calibration path

The API pilot was run through OpenRouter with `openai/gpt-5.1`, but only **3/12** outputs passed
strict validation; the other outputs mainly failed JSON or exact-evidence requirements. To make the
extraction inspectable and easy to debug, a separate manual baseline was therefore completed.

- **Manual extraction:** 12/12 valid decision-step files, containing 277 patient-context items and
  309 exact evidence spans.
- **Manual item-to-ACR mapping:** all 277 items reviewed; 119 mapped to at least one ACR predicate,
  153 retained as `unmapped_value_within_dimension`, and 5 retained as
  `other_proposed_dimension`. The mappings create 497 item-to-instance links and touch 63/78 ACR
  predicate instances; repeated links arise when the same predicate appears in several Variants.
- **Dimension audit:** provisionally retain one new candidate dimension,
  `intervention_or_support_state`, covering active organ support, temporary/implanted devices, or
  interventions that alter patient state or interpretation. Do not promote
  `imaging_feasibility_constraint` yet; treat it as a possible later bridge operation concerning
  whether a patient attribute or prior test constrains the next action.
- **Validation:** every non-null ACR ID was checked against the 78-instance registry; the 16 relevant
  tests pass.

These outputs are the active calibration inputs for the next task. They are not yet the final hybrid
pipeline: the HPI leakage preflight was intentionally postponed for this debugging pass, and the
deterministic and text-complement streams have not been merged.

### 5.3 Immediate next deliverable: Variant matching and adjudication

Evaluate each manual patient Context against the **17 compiled Variant signatures**. For every
material predicate in a Variant, record `supported`, `contradicted`, or `unknown`, then assign a
human-reviewed correspondence label: `exact`, `partial`, `multiple`, `uncertain`, or
`out_of_scope`.

Required rules:

- preserve each Variant's Boolean groups, required roles, and aggregate-member hierarchy;
- treat `exact_or_equivalent` and compatible `patient_value_narrower` links as direct support;
- manually adjudicate `patient_value_broader` and `related_judgment_required` links;
- use `contradicted` only for a direct negative at compatible scope; absent evidence is `unknown`;
- distinguish sequential Variants using explicit prior-test history and interpretation, not an
  inferred `imaging_stage`;
- rank a short candidate list if useful, but do not automatically force one Variant;
- keep A/Q/C, the observed order, ACR Actions/ratings, and later outcomes hidden.

Each case should preserve a predicate-level decision matrix and a short rationale so disagreements
can be inspected. Variant matching should also record any recurrent judgment needed to move from
patient evidence to the Variant—for example, whether a prior study answered the active question or
whether several partially applicable Variants can be resolved. These are candidates for the later
missing-middle codebook, not assumptions to silently encode.

### 5.4 Production-pipeline work still pending

After manual Variant calibration, return to the formal hybrid pipeline: complete causal HPI review,
merge deterministic and text-complement outputs using explicit per-type identity rules, preserve
evidence/provenance and conflicts, and rerun patient-to-ACR and Variant mapping on the merged
`PatientContext_t`. Only after Context/Variant adjudication should A/Q/C and observed actions be
revealed for bridge-operation analysis.

### 5.5 Expansion boundary

Do not run the full cohort until the 12-step extraction, merge identities, ACR mapping, and Variant
adjudication have been inspected and revised. The frozen active development manifest contains
**235 patients and 433 decision steps**, with zero final-test patients. Earlier discussion referred
to “236 patients”; reconcile that discrepancy before any batch run and do not substitute the much
larger raw source corpus. Freeze the bridge codebook and mapping procedure before any final-test
replication.

## 6. Pilot role and stopping rule

The frozen 12-step pilot is selected for structural coverage rather than prevalence estimation. Its
old manual crosswalks began with A/Q/C and moved too quickly to bridge-operation labels; they remain
provenance only and must not be used as Variant labels. The new manual extraction and ACR mappings
are the active calibration inputs. Do not alter the underlying A/Q/C annotations or faithful ACR
extraction.

Use these 12 steps for the first calibration run. Add fresh development cases only when the pilot
exposes an uncovered identity rule, a mapping type without a counterexample, an unresolved
generic/specific Variant distinction, or a new top-level bridge operation.

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

- `A-note.md`: concise conceptual notes distinguishing ACR Variants/predicates from open patient
  dimensions and recording unresolved bridge questions.
- `data/aqc_analysis/development_v1/manifest.json`: frozen active development manifest (235
  patients, 433 decision steps, zero final-test patients).
- `data/aqc_acr_bridge/pilot_v1/sample_manifest.json`: frozen 12-step development pilot.
- `data/aqc_acr_bridge/pilot_v1/manual_crosswalk_round1.jsonl`: first four manual crosswalks.
- `data/aqc_acr_bridge/pilot_v1/manual_crosswalk_round2.jsonl`: remaining eight crosswalks.
- `data/aqc_acr_bridge/pilot_v1/pilot_summary.md`: first-pass synthesis and boundaries.
- `data/aqc_acr_bridge/bridge_codebook_draft_v0_1.json`: provisional B1–B4 codebook.
- `data/aqc_acr_bridge/acr_variant_predicate_audit_v1.md`: review table for all 17 ACR Variants.
- `data/aqc_acr_bridge/acr_variant_predicate_audit_v1.json`: condition instances, logic, aggregates,
  and source provenance.
- `data/aqc_acr_bridge/acr_predicate_types_v1.json`: current 11-type ACR predicate registry.
- `scripts/run_aqc_acr_bridge_pilot.py` and
  `results/aqc_acr_bridge/pilot_v1/openai__gpt-5.1/`: API pilot runner and inspectable
  OpenRouter/`openai/gpt-5.1` inputs and outputs; 3/12 outputs pass strict validation and these are
  not the current extraction baseline.
- `results/aqc_acr_bridge/pilot_v1/manual_extraction_v1/`: current manual text-complement baseline;
  read `README.md`, `manifest.json`, and `review_notes.md` before using its 12 output files.
- `scripts/map_manual_patient_context_to_acr.py` and
  `results/aqc_acr_bridge/pilot_v1/manual_acr_mapping_v1/`: reproducible manual patient-item-to-ACR
  mapping; `summary.json` gives counts, `dimension_audit_v1.json` records dimension decisions, and
  `outputs/` contains the 12 case-level mappings.
- `experiments/aqc_acr_bridge/algorithmic_predicates.py`: deterministic laboratory, labelled-vital,
  prior-test-history extraction and labelled-vital masking.
- `experiments/aqc_acr_bridge/prompts.py`: concise LLM text-complement extraction prompt and output
  contract.
- `tests/test_aqc_acr_algorithmic_predicates.py` and
  `tests/test_aqc_acr_patient_prompt.py`: current hybrid-extraction tests.
- `tests/test_aqc_acr_bridge_pilot_runner.py` and
  `tests/test_manual_patient_context_to_acr_mapping.py`: API pilot-runner and manual ACR-mapping
  tests.
- `scripts/build_masked_view.py`, `scripts/audit_aqc_input_leakage.py`, and
  `data/aqc_prediction/development_v1/leakage_review.json`: relevant causal masking, reviewed HPI
  redaction, and leakage-review provenance; their interfaces are not assumed interchangeable.
- `data/aqc_acr_bridge/archive/ten_dimension_v1/`: provenance-only archive of the superseded
  50-value/ten-dimension audit, its 12-case patient extraction and mappings, and their scripts.

## 9. Handoff instructions for the next conversation

The next conversation should begin with these files:

1. this document;
2. `data/aqc_acr_bridge/acr_variant_predicate_audit_v1.json`;
3. `results/aqc_acr_bridge/pilot_v1/manual_extraction_v1/manifest.json` and `review_notes.md`;
4. `results/aqc_acr_bridge/pilot_v1/manual_acr_mapping_v1/summary.json`,
   `dimension_audit_v1.json`, and the 12 files in `outputs/`.

The immediate task is **manual Variant matching/adjudication for all 12 decision steps**. Do not
rerun extraction or item-to-ACR mapping first. For each step:

1. evaluate all 17 Variant signatures using their required predicates, Boolean logic, and aggregate
   hierarchy, then rank the disease-relevant candidates;
2. save a predicate matrix (`supported` / `contradicted` / `unknown`) with exact patient-item and
   ACR-instance references;
3. retain a short candidate list and rationale;
4. assign the reviewed label `exact`, `partial`, `multiple`, `uncertain`, or `out_of_scope`;
5. record unresolved judgment as a candidate bridge operation rather than silently resolving it.

Write the new work separately, preferably under
`results/aqc_acr_bridge/pilot_v1/manual_variant_adjudication_v1/`, with one output per step plus a
summary and README. Keep A/Q/C, current order/result, Actions/ratings, old manual crosswalk labels,
and later outcomes hidden during this task.

Completed at handoff: manual text extraction, dimension audit, and patient-item-to-ACR mapping.
Pending: Variant adjudication, formal deterministic/text merge, and later A/Q/C/action comparison.
