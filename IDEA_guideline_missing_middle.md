# Decoding the Missing Middle Between Patient Evidence and Clinical Guidelines

> **A guideline is a book indexed by clean clinical contexts, but the patient does not arrive with
> a page number.**

Status: canonical conceptual note, updated 2026-08-18. This document states the current project
direction. Historical deviation/certainty-agent documents remain useful as implementation records,
but no longer define the central scientific problem.

## 1. Central thesis

Guidelines provide conditional normative knowledge:

```text
given guideline Context X -> candidate Actions and appropriateness
```

Real care begins earlier. Patient evidence is incomplete, evolving, and does not identify its own
guideline Context. Clinicians must interpret the evidence, decide what clinical frame is active,
identify what remains unresolved, and judge whether the evidence is sufficient to move to another
Context or action.

```text
patient evidence -> epistemic bridge -> guideline Context -> action
```

This **epistemic bridge** is the missing middle. Deviation is only one observable downstream signal:
an action can differ from a guideline because the patient was mapped to another Context, matched
several Contexts, remained outside the represented Contexts, or required a transition that the
static guideline did not encode. Some deviations may still be unsupported. Therefore, the project
does not begin by assuming that every deviation is meaningful or correct.

ACR remains an independent normative reference. We do not know whether a physician consulted it,
and an observed physician action is not automatically appropriate.

## 2. Why the middle is missing

The gap has two interacting sources.

### 2.1 Patient-side ambiguity: evidence is not self-interpreting

Clinical records contain symptoms, measurements, and report findings, but their decision meaning is
conditional on the working assumption and question. Missingness, a negative finding,
nonvisualization, and an unmentioned target are not interchangeable.

For example, an ultrasound may report:

```text
Gallstones seen; common bile duct not visualized.
```

The same report supports different interpretations:

- under a biliary-pancreatitis assumption, gallstones are an etiologic clue;
- under a non-biliary frame, gallstones may be incidental;
- for the question "Are gallstones present?", the report provides an answer;
- for the question "Is the CBD obstructed?", the report leaves a required dimension open.

Thus, `adequate`, `inadequate`, `negative`, and `informative` are never free-standing labels. They
are relative to a target question and its answer requirements.

### 2.2 Guideline-side ambiguity: Contexts mix facts with judgment

A guideline Context may combine observable predicates with judgment-dependent conditions:

```text
observable facts:              RUQ pain, temperature, WBC count
judgment-dependent conditions: suspected biliary disease,
                               negative or equivocal ultrasound
```

The second layer cannot be recovered by direct lookup. It requires a clinician to decide what the
facts mean, whether a prior study addressed the relevant target, and which Context now applies.
A patient can therefore have an `exact`, `partial`, `multiple`, `uncertain`, or `out_of_scope`
relation to the available guideline Contexts.

### 2.3 The clinician supplies the epistemic bridge

Clinicians mediate the two ambiguities: they decide what the evidence means, which Context applies,
and what remains unanswered. A/Q/C is the project's minimal candidate representation of this role:

```text
A — Assumption
    What working proposition currently frames the workup?

Q — Question
    What must be resolved under A, and what would count as an answer?

C — Coverage
    Which answer requirements of Q have all currently available evidence addressed,
    and which remain open?
```

In one sentence:

> **A frames the problem; Q specifies what the evidence must resolve; C records what the evidence
> has actually resolved.**

The answer requirements are part of the operational representation of `Q`, not a fourth core state.
`C_t` is a time-indexed coverage profile over those requirements, not the updating mechanism itself.

## 3. The bridge is dynamic

New evidence does not map directly to an action. It first changes what has been answered and may
support or challenge the current assumption:

```text
new evidence -> update C -> reassess A and Q -> remap guideline Context -> action
```

Common transition patterns include:

- **remedy:** the same question remains open because the prior study was limited or did not assess
  the target;
- **adjudicate:** discordant evidence requires a test that can resolve the conflict;
- **advance:** one question closes and a downstream question about cause, severity, or complication
  opens;
- **reroute:** the current assumption is challenged or replaced, so another Context becomes relevant;
- **reopen:** time or clinical change makes a previously settled question uncertain again;
- **close:** no consequential imaging question remains.

Keep four judgments separate throughout this process:

1. **study adequacy:** was the examination technically usable for what it attempted?
2. **test–question capability:** could the test address the requirements of this Q in principle?
3. **result status:** positive, negative, indeterminate, or not assessed relative to Q;
4. **aggregate coverage:** what all evidence together has resolved about Q.

Do not replace these with a generic `accurate/inaccurate` label.

## 4. What longitudinal clinical data can and cannot reveal

Longitudinal data can support recovery of:

- recurrent, evidence-grounded assumptions and questions;
- question-specific answer requirements and coverage profiles;
- transitions between A/Q/C states after new results;
- repeated Context mismatches, out-of-scope states, compensatory tests, switches, and stopping logic;
- documented feasibility constraints that affect action realization.

It usually cannot uniquely recover:

- the physician's private belief or actual guideline consultation;
- undocumented patient or physician preferences;
- real-time availability, cost, workflow, or insurance constraints without credible proxies;
- whether a recurrent observed practice is clinically optimal or causally beneficial.

The empirical target is therefore a **recoverable decision-state representation**, not mind reading.
Repeated observed practice is candidate knowledge, not automatically normative knowledge.

## 5. Research program

### Aim 1 — Validate the representation

Test whether a strictly pre-order A/Q/C state captures decision-relevant structure:

```text
pre-order record -> infer A/Q/C -> predict next test or stop
```

The evaluation must prevent target leakage: the current order, its result, and later events cannot
be used to infer the A/Q/C state being evaluated. Compare patient evidence alone (`O`), guideline
knowledge alone (`N`), pre-order A/Q/C, and `N + A/Q/C` on held-out patients. Useful outcomes include
next-image ranking, repeat/switch/stop prediction, calibration, and transition prediction.

Incremental predictive value supports the claim that A/Q/C captures a useful part of the missing
middle. It does not prove unique recovery of the physician's mental state.

### Aim 2 — Surface knowledge left implicit by guidelines

Aggregate explicit trajectories of the form:

```text
(A_t, Q_t, C_t) -- test/result --> (A_{t+1}, Q_{t+1}, C_{t+1})
```

Identify recurrent coverage gaps, compensatory actions, assumption shifts, and Context transitions
that are absent or under-specified in the guideline representation. Retain a pattern only if it is
pre-order grounded, recurrent, interpretable, reproducible in held-out patients, and not a trivial
restatement of the observed order.

The output is a set of **candidate empirical transition rules or Context extensions**. Clinician
review, external replication, and eventually outcome or safety evidence are required before making
normative claims.

## 6. Intended contribution

The intended contribution is not another list of deviations. It is an explicit map of:

```text
what guidelines specify
+ what longitudinal data can recover as epistemic decision structure
+ what remains latent or unidentifiable
+ what remains unsupported
```

The scientific payoff of A/Q/C is that the bridge becomes inspectable, testable, and reusable:
prediction can validate the representation, while recurrent state transitions can expose candidate
knowledge that a static guideline does not encode.

## 7. Current empirical work plan

Status as of 2026-09-04: the formal development partition contains 235 patients and 433 imaging
decision steps. DIRECT A/Q/C annotation is complete for this partition. The separate final-test
partition contains 58 patients and 109 steps and remains unopened. The immediate task is therefore
not further development annotation, but construction and analysis of the effective development
annotations.

The work proceeds in the following stages. ACR must not be used to define the empirical patterns in
Stages 1--3; this prevents the normative representation from shaping what is recovered from the
patient trajectories.

### Stage 1 — Build one analysis-ready A/Q/C layer

Merge each original DIRECT output with its latest accepted non-destructive overlay and produce one
effective development dataset. The original model outputs remain unchanged. Because the latest
adjudicated corrections are accepted for analysis, the effective layer records which overlay was
applied but does not require a separate rationale for every correction.

The release must fix and verify:

- exactly 235 development patients and 433 decision steps;
- prompt, schema, validator, model, source, and overlay versions;
- stable patient, trajectory, step, question, requirement, and transition keys;
- explicit handling of invalid, low-evidence, weakly grounded, and unclear records;
- all accepted manual temporal and discordance corrections;
- no duplicate patients or steps and no final-test patient or content;
- preservation of schema-generation differences without silently renaming fields whose meanings
  changed, especially legacy `intent_support` and current `question_grounding`.

The core outputs are patient-, step-, requirement-, and transition-level tables plus a machine-
readable quality-control manifest. Invalid records fail the release rather than being silently
dropped. `unclear` and weakly supported values remain observable flags; their exclusion, when
needed, is pattern-specific and must be repeated as a sensitivity analysis.

### Stage 2 — Define the units of empirical pattern analysis

The primary object is a trajectory transition, not the marginal frequency of an isolated field.
Patterns should be derived from the underlying A/Q/C states and changes rather than accepted merely
because the annotation contains a named `derived_transition` label. Initial targets include:

- assumption retention, refinement, challenge, exclusion, or replacement;
- question continuation, refinement, replacement, reopening, or closure;
- answer requirements that remain unaddressed or only partially addressed;
- repeat imaging, modality switching, or stopping after a coverage gap;
- rerouting after test--question capability mismatch;
- explicit interval-comparison requirements that motivate serial imaging;
- sufficiently covered questions followed by additional imaging that remains unsupported.

Operational definitions must distinguish a true clinical stop from right censoring at discharge,
transfer, death, or the end of the observable record. Terms such as `escalation` must be defined by
observable changes in modality, protocol, or intervention rather than by an assumed hierarchy of
clinical appropriateness.

### Stage 3 — Discover candidate patterns in development

For every candidate pattern, report the number of eligible patients, trajectories, and steps; the
appropriate opportunity denominator; disease, modality, and prompt/schema-version strata;
supporting examples; counterexamples; unclear cases; and unsupported residuals. Check whether the
pattern persists after excluding weakly grounded or ambiguous records and whether it appears across
diseases rather than being created by one disease or annotation version.

These development results establish recurrence, interpretability, and structural coherence. They
must not be described as unbiased population prevalence, causal effects, normative recommendations,
or by themselves as proof that A/Q/C is a useful predictive bridge. Because DIRECT annotation saw
the actual order, any apparent `coverage gap -> action` relation remains vulnerable to order-driven
over-rationalization and requires a pre-order-only check.

### Stage 4 — Freeze the empirical pattern codebook and analysis plan

Before testing against ACR or opening final test, freeze for every retained pattern:

- its inclusion and exclusion rules;
- source and destination states;
- minimum required fields and evidence;
- opportunity denominator and patient-level deduplication rule;
- treatment of uncertainty, conflict, censoring, and missing fields;
- whether multiple patterns may coexist at one transition;
- cross-disease, annotation-version, and sensitivity analyses;
- the minimum recurrence and replication criteria.

`Remedy`, `adjudicate`, `advance`, `reroute`, `reopen`, and `close` remain derived summaries of
assumption, question, and coverage changes rather than primitive labels imposed on the data.

### Stage 5 — Validate A/Q/C as a useful missing middle

Internal pattern recurrence is necessary but not sufficient. The main representation test is
whether A/Q/C inferred without the target order improves held-out prediction:

```text
pre-order O_t -> infer A/Q/C_t -> predict next image, repeat, switch, or stop
```

Use patient-grouped development resampling to compare `O`, `O + N`, `O + inferred A/Q/C`, and
`O + N + inferred A/Q/C`. The current order, its result, and later events must remain hidden from
the A/Q/C inference used as a predictor. Order-aware DIRECT annotations may serve as reference or
training targets, but may not be inserted directly as features for predicting the same order.

Incremental discrimination and calibration support the claim that A/Q/C is a useful missing-middle
representation. They do not prove unique recovery of a physician's private mental state.

### Stage 6 — Map frozen empirical patterns to ACR and replicate

Only after the empirical pattern definitions are frozen should they be mapped systematically to ACR
Contexts. Classify relations as `exact`, `partial`, `multiple`, `uncertain`, or `out_of_scope`, then
identify recurrent transitions or Context extensions that ACR omits or under-specifies. This later
mapping must not retroactively redefine the empirical patterns.

After the pattern codebook, filters, inference method, and statistical plan are frozen, open the
58-patient final-test partition once for replication and final prediction evaluation. Final-test
results may confirm or fail to confirm a pattern; they may not be used to revise its definition.
