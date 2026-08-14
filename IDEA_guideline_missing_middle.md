# The Missing Middle Between Guidelines and Clinical Decisions

> **Guidelines rank tests for predefined contexts; clinicians must determine which context applies,
> manage what remains unknown, and decide how the diagnostic state should change.**

Status: conceptual working note. This document fixes the current project direction without closing
questions that still require empirical discovery.

## 1. The central problem

ACR provides conditional normative knowledge:

```text
given guideline Context X -> candidate Action -> appropriateness rating
```

It is not a complete patient-level diagnostic policy. In practice, the difficult reasoning often
occurs around this mapping:

```text
patient evidence
  -> construct the current clinical state
  -> decide whether one, several, or no guideline Contexts apply
  -> identify the consequential unanswered question
  -> judge whether existing evidence is sufficient
  -> choose a feasible and useful action
  -> update, continue, switch, advance, reroute, or stop
```

This is the **missing middle**. The issue is not simply that a guideline is wrong or incomplete.
The guideline deliberately starts from an already defined Context, while real care must first
construct, test, and revise that Context.

ACR is an independent normative reference in this project. We do not know whether a physician
actually consulted it, and an observed physician action is not automatically correct.

## 2. Where the missing middle appears

### Context construction

A Context mixes different epistemic kinds:

- directly recorded facts or measurements: age, pain location, temperature, laboratory values;
- derived states: leukocytosis, SIRS, severe scores, deterioration, temporal trends;
- working assumptions: suspected appendicitis, biliary disease, complication, alternative source;
- interpretations of prior evidence: negative/equivocal, adequate/nondiagnostic, target assessed or
  not assessed;
- decision-stage judgments: initial, next, persistent, recurrent, or post-intervention.

Only the first layer is close to direct lookup. The others require thresholds, aggregation,
temporal reasoning, or clinical inference. Thus, finding a phrase in the chart is not the same as
establishing membership in a guideline Context.

### Context reconciliation

A real patient may have an `exact`, `partial`, `multiple`, `uncertain`, or `out_of_scope` relation
to available guideline Contexts. Apparent deviation may therefore result from forcing a patient
into the wrong or overly narrow Context.

Repeated out-of-scope patterns may reveal **empirical context candidates** omitted from the
guideline representation: discordant evidence, a limited prior study, another anatomic source,
clinical deterioration, intervention planning, unusual comorbidity, or a diagnostic trajectory
not represented by existing variants. These are hypotheses for discovery, not predetermined
categories and not new normative recommendations.

### Epistemic management

A/Q/C represents the recoverable clinical core:

```text
A: What proposition is currently organizing the workup?
Q: What consequential unknown should the next action answer?
C: How sufficiently does all currently available evidence answer that question?
```

It can explain why a negative result closes one question but not another, why a study is repeated
or replaced, why the workup advances from diagnosis to cause or complication, and why the clinical
frame is rerouted.

### Action realization and transition

Appropriateness alone does not fully determine the next action. Choice can also depend on
question-specific capability, expected management consequence, urgency, contraindications,
patient burden, availability, cost, and preferences. After the result, a dynamic policy must update
the assumption and question and decide whether to continue or stop. Static variants rarely encode
this full transition logic.

## 3. Decompose uncertainty by identifiability

The project should not divide factors only into “structured” and “noise”:

| Level | Meaning | Treatment |
|---|---|---|
| Observed structured | Explicit pre-order facts that can be standardized | Use directly |
| Inferred structured | Recurrent states grounded in prior evidence but not stored as fields | Primary Track B target |
| Latent structured | Systematic factors not measured well in the current data | Use documented proxies or retain as latent |
| Idiosyncratic / error | Unstable, accidental, or unsupported variation | Preserve as residual |

Unmeasured does not mean unstructured. Cost, insurance, scanner availability, service workflow,
physician preference, and patient preference may be systematic, but cannot be assigned a specific
causal role unless the relevant data or credible proxies exist. Private physician beliefs, actual
guideline consultation, undocumented conversations, and real-time operational constraints are
usually not identifiable from an imaging order alone.

Accordingly, an observed action can be decomposed conceptually as:

```text
Y_t <- normative reference N
     + guideline-context relation M_t
     + epistemic state (A_t, Q_t, C_t)
     + documented feasibility F_t
     + latent implementation factors L_t
     + unsupported residual R_t
```

The goal is not to force every action into a rational explanation. It is to separate recoverable
clinical structure from unobserved implementation effects and genuinely unsupported residuals.

## 4. What real data can add

Track B should discover and test:

1. which guideline Context predicates are observed, derived, assumed, or interpretive;
2. whether each patient decision is an exact, partial, multiple, uncertain, or out-of-scope match;
3. recurrent A/Q/C states and transitions before the observed order;
4. documented feasibility constraints that change action choice;
5. which apparent deviations become context mismatch, guideline underdetermination,
   epistemically supported actions, operationally explained actions, or unsupported residuals.

A candidate hidden structure should be retained only if it is:

- available before the order and grounded in quotable evidence;
- recurrent across patients rather than a one-case story;
- clinically interpretable and not a restatement of the observed action;
- reproducible in held-out patients;
- useful beyond patient observations and ACR alone for explanation or prediction.

An empirically recurrent missing Context can reveal a representational gap in the guideline, but
observational frequency does not make it normatively appropriate. Safety, outcomes, and independent
evidence would be required for that stronger claim.

## 5. Project question and intended contribution

> **Which recurrent and observable structures bridge guideline-defined contexts and real
> patient-level diagnostic decisions, and which parts remain unidentifiable from clinical data?**

The intended contribution is an **uncertainty decomposition map**: what the guideline already
specifies, what patient data can restore as structured clinical reasoning, what remains a latent
implementation factor, and what remains unsupported.

Prediction is a validation instrument, not the whole scientific goal. If adding the recovered
structure improves held-out explanation or prediction especially for repeat, switch, advance,
reroute, and stop decisions, it supports the claim that the missing middle captures genuine dynamic
decision structure.

## 6. Open questions

- Should guideline-context relation `M_t` remain separate from A/Q/C, or become part of the final
  representation?
- Which feasibility variables `F_t` are sufficiently documented to model rather than leave latent?
- How should empirical out-of-scope Contexts be represented without prematurely expanding ACR?
- Should the primary evaluation emphasize action prediction, transition prediction, uncertainty
  decomposition, or a deliberately ordered combination of them?
- What additional provider, site, cost, availability, or patient-preference data would make latent
  implementation factors identifiable?

