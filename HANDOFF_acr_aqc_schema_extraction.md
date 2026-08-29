# Handoff: ACR extraction and A/Q/C knowledge decoding

## Read first

1. `rubric_update.md` — overall research direction and work plan.
2. `aqc_annotation_design.md` — current A/Q/C concepts and annotation logic.
3. `HANDOFF_annotation_pipeline.md` — how the existing schema-free, order-aware annotations were made.

## Decisions already made

- Deviation is a downstream observable signal, not the central scientific problem. The central
  problem is the missing epistemic bridge between interpretable-but-ambiguous patient evidence and
  judgment-dependent guideline Contexts.
- The two main ambiguity sources are patient-side interpretation (including missingness, negative
  findings, and nonvisualization) and guideline Contexts that mix observable facts with clinical
  judgments. A/Q/C is the candidate minimal mediation state.
- The old executable disease trees are historical artifacts, not a standard rubric or baseline.
- ACR and patient annotations are two independent knowledge sources.
- Extract ACR faithfully before imposing any ontology, especially A/Q/C.
- Existing annotations are schema-free discovery material and must not be overwritten.
- A/Q/C is induced from physician-order reconstructions, not derived from ACR.
- The main A/Q/C annotation is trajectory-level and order-aware: the actual order is shown, but its
  result and all later events are hidden.
- Keep these distinct: study adequacy, test–question capability, result status, and aggregate
  question coverage.
- Q contains both the decision-relevant unknown and the dimensions that would count as answering it.
  C is the time-indexed coverage profile over those requirements, not the update mechanism.
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

1. [ ] Create a stable patient-level approximately 80/20 development/final-test split from `full/`,
   stratified within disease. Keep final-test patients unopened until the framework and models are
   frozen.
2. [ ] From development, select an initial approximately 24-trajectory codebook sample total
   (about six per disease) for structural diversity. This sample defines the annotation vocabulary;
   it does not estimate final pattern counts or prevalence.
3. [ ] Open-code assumptions and questions in a reasoning-only blind view, then compare against the
   complete schema-light ex-ante fields.
4. [ ] For each recurrent question, open-code the answer requirements that define what evidence
   would count as resolving it.
5. [ ] Revise the preliminary assumption codebook and add a question/answer-requirement codebook
   with `other/unclear`.
6. [ ] Add fresh non-overlapping development batches if new top-level types, recurrent answer
   requirements, or systematic residuals continue to appear. Twenty-four is a starting batch, not a
   fixed sample size; freeze only after fresh-case qualitative saturation.
7. [ ] Revise the trajectory-level, order-aware A/Q/C prompt after the full-corpus discovery audit;
   replace the current scalar-only coverage prototype with requirement-level coverage plus an
   optional summary.
8. Check the framework on about 16 unused development trajectories in two independent ways:
   - reconstruct A/Q/C directly from masked trajectories plus actual orders;
   - recode the old open reasoning into A/Q/C.
9. Compare agreement and over-rationalization, revise when necessary, and re-check on another fresh
   development batch before freezing. If A and Q agree closely, reuse the old reconstruction for
   bulk conversion where supported; C still requires causally available patient evidence.
10. After freezing, annotate the larger development corpus and discover/estimate patterns there.
    Use the untouched final-test partition only for replication and final prediction evaluation.

### Later validation

Compare faithfully extracted ACR knowledge (`N`), pre-order inferred A/Q/C, and `N + A/Q/C` on
patient-level held-out next-image, repeat/switch/stop, and sequence prediction. Preserve an
unsupported residual rather than explaining every observed order post hoc.

After representation validation, mine recurrent A/Q/C transitions and Context remappings that are
absent or under-specified in ACR. These are candidate practice-knowledge patterns, not normative
recommendations, until they replicate and receive clinician/external validation.

## Recommended next task

Redo **Track B codebook development from `full/`** using the documented development/final-test
design. The first concrete engineering task is to replace the hard-coded 16-trajectory prototype in
`scripts/build_aqc_discovery_sample.py` with reproducible patient-level split and development-sample
manifests over `results/annotation_experiment/full`. Build the first formal codebook from an initial
approximately 24 development trajectories, expand with fresh development batches until saturated,
and keep the final test unopened. Do not start batch A/Q/C annotation or next-test validation before
the full-corpus codebook audit.

## Suggested opening prompt

Use the complete, copy-ready prompt in `HANDOFF_aqc_codebook_discovery.md`.
