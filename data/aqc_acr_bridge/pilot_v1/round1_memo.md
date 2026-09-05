# AQC–ACR bridge pilot: round-1 memo

Status: first-pass manual coding of 4/12 deliberately selected development steps. These cases are
structurally diverse boundary probes and do not estimate population frequencies.

## What appeared in all four cases

None of the four decisions had a clean one-variant lookup. Three had multiple or partial Context
relations; one was uncertain to out-of-scope. Yet the observed action often matched an ACR-rated
action family. This makes **Context correspondence** and **Action correspondence** separate
variables. An action match alone cannot establish guideline concordance.

The recurrent work was not adding another diagnosis to ACR. It was transforming patient evidence
into a guideline-usable state:

1. Interpret the previous result relative to the current Q rather than globally as positive,
   negative, or adequate.
2. Decide whether several partial ACR variants must be composed because no single Context contains
   the patient's clinical predicates and trajectory stage.
3. Remap across questions, stages, or guideline topics after new evidence.
4. Preserve `out_of_scope` when the action resembles an ACR option but the clinical Context does
   not match.

## Provisional operation families

The 11 open codes from round 1 can provisionally be grouped into four higher-level families. These
families remain revisable until the remaining pilot cases and fresh cases are coded.

### B1. Question-relative evidence interpretation

- `question_relative_prior_test_interpretation`
- `question_relative_equivocality`
- `negative_result_scope_limitation`

This family converts report findings into coverage of a particular question. It is where study
adequacy, target visualization, test capability, and question coverage must remain distinct.

### B2. Dynamic Context construction and remapping

- `cross_variant_stage_composition`
- `cross_topic_context_remapping`
- `severity_triggered_question_reopening`
- `question_advance_to_intervention_target`

This family represents longitudinal work largely absent from static ACR variants: constructing a
usable Context from partial variants, changing the active topic, or opening a downstream question.

### B3. Action realization under patient and test constraints

- `patient_constraint_application`
- `within_modality_target_refinement`
- `complementary_modality_translation`

This family connects an unresolved question to a feasible test or protocol. Round 1 suggests that
ACR may rate the resulting action while leaving the transition that made it relevant implicit.

### B4. Mapping restraint

- `out_of_scope_recognition`

This is a safety-relevant bridge operation. A useful system must sometimes decline to force a
patient into an available ACR Context, especially when only the observed action resembles an ACR
option.

## Preliminary AI decomposition

The potentially delegable unit is not the final clinical decision as a whole. It is a set of
smaller, checkable operations:

| Candidate AI task | Current assessment | Required safeguard |
|---|---|---|
| Extract patient evidence for each ACR predicate | Highly structureable | Show evidence spans and unknown predicates |
| Retrieve all exact/partial candidate variants | Highly structureable | Never collapse multiple partial matches into one silently |
| Determine whether prior imaging assessed each Q requirement | Structureable with verification | Preserve nonvisualized, indeterminate, and absent-report distinctions |
| Detect Context/stage/topic mismatch | Highly structureable | Treat mismatch as a flag, not proof of inappropriate care |
| Propose cross-topic or cross-variant remapping | Conditionally structureable | Require clinician confirmation and display alternatives |
| Choose the final action under emergency, feasibility, and intervention tradeoffs | Not yet delegable from these data | Clinician remains accountable; latent constraints require abstention |

The emerging candidate role for AI is therefore a **bridge workbench**: assemble predicates,
calculate question-relative coverage, retrieve and compare candidate Contexts, expose missing
predicates and stage mismatches, and flag where clinician judgment is still required.

## Next coding test

Code the remaining 8 pilot steps and test three provisional claims:

1. Question-relative prior-test interpretation recurs beyond the four initial cases.
2. Static ACR stage definitions systematically under-specify multistep trajectories.
3. Cross-topic remapping is recurrent rather than unique to biliary pancreatitis.

Counterexamples must be retained. If straightforward exact Context matches occur, they define the
boundary between ordinary guideline retrieval and genuine bridge work.
