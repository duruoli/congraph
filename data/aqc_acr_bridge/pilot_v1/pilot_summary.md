# AQC–ACR bridge discovery pilot summary

Status: first-pass manual crosswalk complete for 12/12 purposively selected development steps.
The bridge codebook remains provisional.

## Correspondence result

The first pass produced six `partial`, four `multiple_partial`, one `partial_to_uncertain`, and one
`uncertain_to_out_of_scope` overall mappings. There were no clean exact mappings. This must **not**
be interpreted as the frequency of ACR mismatch: the pilot intentionally selected multistep,
repeat/switch, constraint, and likely boundary cases.

The consistent qualitative result is that Context and Action correspondence are different:

- an observed action may exactly resemble a highly rated ACR procedure while the patient's stage
  or predicates fit no single variant;
- a procedure may be rated in an ACR variant but be used clinically for a different, longitudinal
  purpose;
- a Context may become usable only by combining an initial-imaging variant's clinical predicates
  with another variant's next-imaging logic;
- action similarity without Context support must permit `out_of_scope` rather than forced
  concordance.

## Bridge structure induced from the 12 cases

The 25 first-pass open codes cluster into four non-exclusive candidate families:

| Family | Meaning | Pilot steps containing family |
|---|---|---:|
| B1 Evidence → question coverage | Determine what prior evidence answered relative to Q | 6 |
| B2 Context construction/remapping | Instantiate, compose, switch, or reject ACR Contexts | 9 |
| B3 Longitudinal question transition | Advance, reroute, reopen, or monitor a Q | 7 |
| B4 Action realization | Apply capability, protocol, and patient constraints | 7 |

These counts show recurrence inside a deliberately selected sample; they are not prevalence
estimates. Multiple families commonly coexist in one decision.

## Residuals

Across the 12 annotations there were six patient-specific residuals outside the available ACR
variants, five latent/unidentifiable residuals, two A/Q/C reconstruction concerns, and one possible
practice/protocol deviation. This supports preserving residual as an explicit output rather than
forcing every observed action into either “guideline-concordant” or “missing-middle knowledge.”

## What currently looks structureable by AI

The most structureable components are:

1. extract evidence for every ACR predicate and preserve unknowns;
2. retrieve all exact and partial variants rather than one best-looking variant;
3. align prior report findings with A/Q/C answer requirements;
4. distinguish adequacy, capability, result status, and coverage;
5. expose stage, topic, protocol, and population mismatches;
6. maintain a longitudinal list of unresolved requirements;
7. normalize local examination names to ACR action families while retaining protocol differences.

The less safely delegable components are choosing among several partial Contexts, deciding whether
an emergency or intervention consequence justifies deviation, and acting when key feasibility or
preference information is latent. The candidate AI product is therefore a bridge workbench with
evidence-linked suggestions and abstention, not an autonomous guideline executor.

## Required next sample

The pilot is enriched for difficult cases and contains no exact match. The next fresh development
batch must deliberately include straightforward initial decisions expected to match one ACR
variant, plus counterexamples for B1–B4. This is necessary to distinguish genuine bridge work from
operations that appear universal only because of the pilot's selection design.
