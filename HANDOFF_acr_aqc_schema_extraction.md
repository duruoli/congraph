# Handoff: ACR extraction and A/Q/C knowledge decoding

## Read first

1. `rubric_update.md` — overall research direction and work plan.
2. `aqc_annotation_design.md` — current A/Q/C concepts and annotation logic.
3. `HANDOFF_annotation_pipeline.md` — how the existing schema-free, order-aware annotations were made.

## Decisions already made

- The old executable disease trees are historical artifacts, not a standard rubric or baseline.
- ACR and patient annotations are two independent knowledge sources.
- Extract ACR faithfully before imposing any ontology, especially A/Q/C.
- Existing annotations are schema-free discovery material and must not be overwritten.
- A/Q/C is induced from physician-order reconstructions, not derived from ACR.
- The main A/Q/C annotation is trajectory-level and order-aware: the actual order is shown, but its
  result and all later events are hidden.
- Keep these distinct: study adequacy, test–question capability, result status, and aggregate
  question coverage.
- A valid negative result is informative; it is not the same as indeterminate, nonvisualized, or not
  assessed.
- `advance/reroute/remedy/...` are summaries derived from assumption and question changes, not
  primitive labels to force first.
- Assumption types should be induced from existing schema-free annotations before the A/Q/C
  codebook is frozen.

## Next work, in order

### Track A — ACR normative extraction

Completed 2026-08-13. Outputs and the frozen operational contract are documented in
`data/acr_normative/README.md`.

1. [x] Acquire and version the four complete ACR topics: RLQ Pain, RUQ Pain, LLQ Pain, and Acute
   Pancreatitis.
2. [x] Extract every variant, action, rating, evidence strength, rationale, and exact provenance.
3. [x] Preserve original wording; do not use A/Q/C as the extraction template.
4. [x] Induce the native ACR context/action vocabulary from the extracted corpus.
5. [x] Manually audit a sample against the original text.

### Track B — Empirical A/Q/C development

1. Sample diverse existing schema-free annotations across all four diseases.
2. Open-code their verbatim assumptions, then cluster recurring types and levels.
3. Produce a provisional assumption codebook with `other/unclear`.
4. Draft a trajectory-level, order-aware A/Q/C prompt from `aqc_annotation_design.md`.
5. Pilot 10–20 trajectories in two ways:
   - reconstruct A/Q/C directly from masked trajectories plus actual orders;
   - recode the old open reasoning into A/Q/C.
6. Compare agreement and over-rationalization, revise the codebook/prompt, then freeze them before
   batch annotation.

### Later validation

Compare faithfully extracted ACR knowledge (`N`), pre-order inferred A/Q/C, and `N + A/Q/C` on
patient-level held-out next-image, repeat/switch/stop, and sequence prediction. Preserve an
unsupported residual rather than explaining every observed order post hoc.

## Recommended first task in the new window

Start with **Track B, steps 1–4**: sample diverse schema-free annotations, induce a provisional
assumption codebook, and draft the trajectory-level pilot prompt. Do not start batch A/Q/C
annotation until the assumption ontology and pilot prompt have been reviewed.

## Suggested opening prompt

> Read `HANDOFF_acr_aqc_schema_extraction.md`, `rubric_update.md`,
> `aqc_annotation_design.md`, and `data/acr_normative/README.md` completely. Treat the
> completed ACR v1.1 corpus as the independent normative representation `N`. Continue with Track B:
> sample diverse schema-free annotations, open-code assumptions, and prepare the paired A/Q/C pilot.
