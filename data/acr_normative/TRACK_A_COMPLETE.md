# Track A completion update

Status: **complete as of 2026-08-13**. Operational schema version: **1.1.0**.

## What is complete

- Four official ACR topics are versioned locally: RLQ Pain, RUQ Pain, LLQ Pain, and Acute
  Pancreatitis.
- The corpus contains 17 variants, 141 rated actions, and 90 narrative rationale sections.
- Every action retains its exact procedure wording, rating data, SOE, radiation level, evidence
  references, rationale link, source URL/file, and page/HTML locator.
- Twelve stratified actions were visually checked against rendered source pages; all passed.
- Source hashes and full-corpus integrity checks pass.
- No A/Q/C field or inferred diagnostic pathway was introduced.

## Frozen operational representation

```text
Context
├── clinical_state
│   ├── presentation
│   ├── condition
│   └── severity_or_complication
├── imaging_history
│   ├── prior_test
│   ├── prior_result
│   └── source_phrases
├── modifiers
│   ├── population
│   ├── timing
│   └── constraints_or_confounders
└── decision_stage
    ├── imaging_stage: initial | next | unspecified
    ├── encounter_status
    └── source_phrase

Context -> candidate Action -> final_rating
```

Interpretation rules:

- `clinical_state` contains the presenting observations, the stated suspected/known condition, and
  severity or complication state.
- `imaging_history` is kept separate from presentation because it identifies prior-test-dependent
  next/switch decisions.
- `modifiers` groups population, timing, and constraints while retaining their subtypes.
- `decision_stage` is workflow position, not disease belief. `first time presentation` is encounter
  status rather than population.
- Exact `variant_text` remains authoritative; structured context is an index over its wording.

## Action ranking contract

- Primary metric: `final_rating`, integer 1-9.
- Direction: higher means more appropriate within the same ACR variant.
- Ties remain ties; do not manufacture a unique action path.
- `appropriateness_category` is explanatory and preserves explicit disagreement.
- SOE describes evidentiary support and must not be used to re-rank actions.
- Median and final tabulations are audit metadata, not alternate ranking targets.
- ACR may describe equally rated actions as alternatives or complementary procedures; consult the
  source-backed `action_relationships` and rationale before interpreting equal ratings as
  interchangeable. Four explicit alternative groups and four complementary groups are retained.

## Files for downstream work

- `acr_topics.json`: canonical nested corpus and rationale.
- `acr_actions.jsonl`: one context-action-rating row per action.
- `native_vocabulary.json`: reviewed context vocabulary and normalized action families.
- `schema/acr_extraction.schema.json`: v1.1 formal schema.
- `sources/manifest.json`: versioned source inventory and SHA-256 hashes.
- `audit/sample_audit.md`: manual audit record.
- `ambiguities.md`: source and mapping edge cases.
- `scripts/extract_acr_normative.py`: deterministic rebuild.
- `scripts/validate_acr_normative.py`: integrity validation.

## Boundary for the next stage

This corpus is the independent normative representation `N`. The next stage should induce and pilot
empirical A/Q/C from existing schema-free patient annotations. Do not derive A/Q/C labels from this
ACR schema and do not rewrite ACR contexts to make observed orders fit. Later comparisons may use:

```text
N              = this ACR context-action-final_rating corpus
AQC            = independently developed empirical state
N + AQC        = combined evaluation representation
unsupported    = preserved residual when neither source supports an order
```

Reopen Track A only for a documented source-version update, a provenance error, or a demonstrated
schema defect. Otherwise, downstream code should treat v1.1 as read-only input.
