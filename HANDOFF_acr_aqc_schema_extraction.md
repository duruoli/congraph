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

Preliminary steps 1–4 were prototyped 2026-08-14, but the initial source scan covered only 38
pilot+batch trajectories. The canonical `results/annotation_experiment/full` corpus contains 293
trajectories and 542 decision steps. Formal discovery must be redone from `full/`; the current
codebook and prompt are **not frozen**. See `data/aqc_development/README.md`.

1. [ ] Create patient-level discovery/held-out splits from `full/` and sample diverse trajectories
   across all four diseases and timing strata.
2. [ ] Open-code assumptions and questions in a reasoning-only blind view, then compare against the
   complete schema-light ex-ante fields.
3. [ ] Revise the preliminary assumption codebook and add a question codebook with `other/unclear`.
4. [ ] Revise the trajectory-level, order-aware A/Q/C prompt after the full-corpus discovery audit.
5. Pilot 10–20 trajectories in two ways:
   - reconstruct A/Q/C directly from masked trajectories plus actual orders;
   - recode the old open reasoning into A/Q/C.
6. Compare agreement and over-rationalization, revise the codebook/prompt, then freeze them before
   batch annotation.

### Later validation

Compare faithfully extracted ACR knowledge (`N`), pre-order inferred A/Q/C, and `N + A/Q/C` on
patient-level held-out next-image, repeat/switch/stop, and sequence prediction. Preserve an
unsupported residual rather than explaining every observed order post hoc.

## Recommended next task

Redo **Track B steps 1–4 from `full/`** using the documented discovery/held-out design, then execute
the paired pilot. Do not start batch A/Q/C annotation before the full-corpus audit and review.

## Suggested opening prompt

> Read `HANDOFF_acr_aqc_schema_extraction.md`, `rubric_update.md`,
> `aqc_annotation_design.md`, `data/acr_normative/README.md`, and
> `data/aqc_development/README.md` completely. Treat the completed ACR v1.1 corpus as the
> independent normative representation `N`. Continue with Track B: rebuild discovery/held-out
> sampling from `results/annotation_experiment/full`, run blind and schema-assisted open coding,
> then revise the paired A/Q/C pilot before execution.
