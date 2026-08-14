# ACR normative corpus

Status: **Track A complete** (2026-08-13). Schema: **v1.1.0**.

## Purpose

Faithfully represent four ACR topics as an independent normative input `N`:

```text
Context -> candidate Action -> final_rating
Context -> explicit relation among an Action set
```

It covers RLQ Pain, RUQ Pain, LLQ Pain, and Acute Pancreatitis. It does not impose A/Q/C, infer a
mandatory diagnostic path, or rewrite ACR to fit observed physician orders.

## Outcome

- 4 ACR topics: Right Lower Quadrant Pain, Right Upper Quadrant Pain, Left Lower Quadrant Pain,
  and Acute Pancreatitis.
- 17 ACR-defined clinical variants: the patient situations for which ACR rates imaging options.
- 141 rated context-action pairs: one ACR procedure row within one variant. These are not 141
  unique procedures and are not clusters.
- Complete rating, SOE, radiation, evidence-reference, rationale, and source provenance fields.
- 4 `equivalent_alternatives` and 4 `complementary` action-set relations explicitly stated by ACR.
- Extraction audit only: 90 source-delimited rationale text blocks were preserved, 12/12 stratified
  source-page samples passed, and source hashes and integrity checks pass. The count of rationale
  blocks is a completeness check, not a modeling variable or analytical outcome.

## Stored representation

```text
Topic
└── Variant = one Context
    ├── clinical_state
    │   ├── presentation
    │   ├── condition
    │   └── severity_or_complication
    ├── imaging_history
    ├── modifiers: population + timing + constraints
    ├── decision_stage: initial | next | unspecified
    ├── Actions[]
    │   └── exact procedure + components + final_rating + evidence + provenance
    └── action_relationships[]
        └── equivalent_alternatives | complementary over 2+ actions
```

`variant_text` and exact `procedure` wording remain authoritative. Structured fields are indexes
over the source wording.

`final_rating` (1-9; higher is more appropriate) is the only action-ranking metric. Ties remain
ties. Category, SOE, median, and vote distribution are retained for interpretation or audit, not
as alternative ranking values. Absence of an `action_relationships` entry means ACR did not state a
relationship; it does not mean independent or interchangeable.

## Plain-English field guide

### What does one record mean?

> For a patient situation defined by ACR (`variant/context`), an ACR expert panel judged how
> appropriate a candidate imaging test or procedure (`action`) would be and documented the
> supporting evidence and clinical reasoning.

Example: **right lower quadrant pain; initial imaging** → **CT abdomen and pelvis with IV
contrast** → the ACR panel assigned **Usually appropriate**, `final_rating = 9`, with **Strong**
evidence.

The ratings are expert judgments published by ACR. They are not orders observed in our patient
data, labels added by our annotators, or predictions made by a model.

### Context: what kind of patient situation is being considered?

- `variant_text`: ACR's complete description of the clinical situation. Read this first; it is the
  authoritative context.
- `clinical_state.presentation`: what is currently observed—symptoms, signs, or laboratory findings,
  such as right lower quadrant pain, fever, or elevated lipase.
- `clinical_state.condition`: the disease or clinical problem that ACR explicitly says is suspected
  or already known, such as suspected appendicitis or known necrotizing pancreatitis.
- `clinical_state.severity_or_complication`: any explicitly stated severity, deterioration, or
  complication, such as SIRS or suspected complications of diverticulitis.
- `imaging_history.prior_test`: imaging already performed before the candidate action.
- `imaging_history.prior_result`: what that earlier imaging showed, such as negative, equivocal, or
  nondiagnostic. This distinguishes a next-step decision from an initial-imaging decision.
- `imaging_history.source_phrases`: the exact words in `variant_text` supporting the extracted
  imaging history.
- `modifiers.population`: a patient group that changes the decision; currently pregnancy.
- `modifiers.timing`: where the patient is in the disease course, such as `<48–72 hours` or
  `>4 weeks` after onset.
- `modifiers.constraints_or_confounders`: a condition that limits test choice or complicates
  interpretation, such as AKI/CKD affecting pancreatic enzyme interpretation.
- `decision_stage.imaging_stage`: whether ACR is rating the first imaging test (`initial`), the next
  test after earlier imaging (`next`), or does not state the stage (`unspecified`).
- `decision_stage.encounter_status`: other visit-level wording, such as a first-time presentation.
- `decision_stage.source_phrase`: the exact ACR phrase supporting `initial` or `next`.

`variant_text` is copied from ACR. The other Context fields are our structured breakdown of that
text for matching and analysis. They do not add facts that are absent from the source.

### Action: which test or procedure did the panel judge?

- `procedure`: the complete procedure name in the ACR table, such as `CT abdomen and pelvis with IV
  contrast`. This is the actual option that received the rating.
- `action_family`: our broader normalized group, such as `ct_abdomen_pelvis`, used to find related
  procedures. It is a grouping aid, not a substitute for `procedure`.
- `action_components.modality`: the imaging technology, such as CT, US, MRI, or radiography.
- `action_components.body_region_or_target`: the anatomy examined or the target of an intervention.
- `action_components.protocol_terms`: details that distinguish similar tests, such as IV contrast,
  MRCP, Doppler, or a transvaginal approach.
- `action_components.procedure_role`: diagnostic imaging or an image-guided intervention.

`procedure` is copied from ACR. `action_family` and `action_components` are the only
project-created grouping layer, used for matching and comparison. This is deterministic
normalization, not statistical clustering.

### Rating and evidence: what did the ACR panel conclude?

- `appropriateness_category`: the ACR panel's plain-language recommendation for this procedure in
  this exact context: `Usually appropriate`, `May be appropriate`, or `Usually not appropriate`.
  `May be appropriate (Disagreement)` means the panel did not reach a stable shared view.
- `final_rating`: ACR's final 1–9 appropriateness score. Here, 7–9 means usually appropriate, 4–6 may
  be appropriate, and 1–3 usually not appropriate. This is the primary value for ranking actions
  within the same Context: higher is better, and equal scores remain tied.
- `strength_of_evidence` (SOE): ACR's judgment of how strongly the published evidence supports the
  recommendation, such as `Strong`, `Limited`, or `Expert Consensus`. It is not the appropriateness
  score, an individual doctor's confidence, or model confidence.
- `median_rating`: the median of the panelists' 1–9 votes, when ACR reports it. It is retained for
  audit; ranking uses `final_rating`.
- `final_tabulations`: how many panelists selected each score from 1 through 9. It shows agreement
  or disagreement within the panel; it is not another ranking target.
- `adult_rrl` / `pediatric_rrl`: ACR's typical relative radiation level for adults or children, not
  the radiation dose received by a specific patient.
- `evidence_references`: studies ACR attached to the recommendation, preserving the printed
  reference number, identifier, and study-quality label.

### Rationale, relations, and provenance: why, how are actions connected, and where is the source?

- `rationales`: ACR-authored "why this kind of test may or may not help" discussion blocks. Each
  block covers one procedure family within one clinical variant, including diagnostic performance,
  limitations, and supporting literature. These are extracted source sections, not explanations
  generated by this project. Rationale is optional supporting text, not a Context/Action/Rating
  dimension and not an action-ranking value.
- `rationale_ids`: links an Action to the relevant ACR narrative section. ACR often discusses a
  whole family such as CT or US together, so every sentence in a rationale may not apply uniquely
  to every contrast or protocol variant.
- `action_relationships`: a relation that ACR explicitly states between two or more Actions in the
  same Context. `equivalent_alternatives` means the actions can serve as alternatives;
  `complementary` means they provide different information and may be used together. The normalized
  label is ours, but `source_text` preserves the supporting ACR statement. No entry means ACR did
  not explicitly state a relationship.
- `provenance`: the official ACR URL, saved source file, page or section, and locator needed to
  verify an extracted value against the original source.

## Files by use

### Pipeline inputs

- **`acr_topics.json` - canonical source of truth.** Use when the pipeline needs complete context,
  grouped candidate actions, action-set relations, rationale, or provenance.
- **`acr_actions.jsonl` - action-level working table.** Usually the most convenient input for
  ranking, training, evaluation, and dataframe analysis. One row is one context-action-rating. It
  intentionally does not flatten action-set relations.
- `native_vocabulary.json` - optional vocabulary/configuration for normalized context terms and
  action families. Do not treat it as a separate knowledge source.

### Schema and maintenance

- `schema/acr_extraction.schema.json` - machine-readable contract for `acr_topics.json`.
- `scripts/extract_acr_normative.py` - deterministic rebuild script.
- `scripts/validate_acr_normative.py` - integrity and provenance validation.

### Archive and audit only

- `sources/` and `sources/manifest.json` - versioned official source snapshots and SHA-256 hashes.
- `audit/sample_audit.md` - manual source-page audit.
- `audit/ambiguities.md` - source inconsistencies and mapping edge cases.

These archive/audit files should not be loaded by the routine modeling pipeline.

## Downstream boundary

Treat v1.1 as read-only normative input `N`. Develop empirical A/Q/C independently from the
schema-free patient annotations. Reopen Track A only for a source-version update, provenance error,
or demonstrated schema defect.

```bash
python3 scripts/extract_acr_normative.py
python3 scripts/validate_acr_normative.py
```

Use `--refresh` only when intentionally creating a new ACR source snapshot.
