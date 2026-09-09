# Manual ACR Variant adjudication: pilot v1

This directory evaluates each of the 12 manually extracted patient Contexts against all 17
compiled ACR Variant signatures. Each output contains a 78-row predicate matrix, preserved ACR
logic/aggregate structure, a short candidate list, a reviewed case-level correspondence label,
and candidate bridge operations.

## Predicate decisions

- `exact_or_equivalent` and compatible `patient_value_narrower` links directly support a predicate.
- A direct negative at compatible time/scope contradicts it.
- Every `patient_value_broader` and `related_judgment_required` link is retained and reviewed. It
  remains `unknown` unless a hand-authored condition judgment supplies entailment or contradiction.
- Missing evidence is `unknown`, never absence.
- Aggregate members and indicators do not automatically satisfy their parent assessment.
- Variant 6's associated-state group is supported when one or more members are supported, while
  its other required predicates remain conjunctive.

## Case-level labels used in this calibration

- `exact`: one retained Variant signature is fully supported without an equally plausible
  unresolved competing frame.
- `partial`: the closest Variant has some supported and some unknown required predicates.
- `multiple`: generic/specific, sequential, or cross-topic Variants remain simultaneously plausible.
- `uncertain`: the active disease/topic or complication source cannot be resolved from available
  evidence.
- `out_of_scope`: no selected Variant provides a compatible frame.

The candidate rank is a human review aid, not an automatic assignment. A generic one-predicate
Variant can be satisfied while a more specific Variant remains partial. The case-level label and
rationale explicitly preserve that overlap.

## Blinding and provenance

Only the active manual patient-item-to-ACR mappings and the compiled ACR predicate registry are
loaded. A/Q/C, the current order/result, ACR Actions/ratings, old crosswalk labels, and later
outcomes remain hidden. Upstream extraction and mapping files are hash-bound in every output.

One upstream concern is deliberately preserved rather than silently repaired: Crohn disease in
`appendicitis:20123918:s2` and `s3` appears both as a background `patient_attribute` and as an
`alternative_diagnosis`. The cited diagnostic-state evidence establishes only history, not a
current alternative-diagnosis role. Variant adjudication therefore records this as an unsupported
role assignment and does not use it as independent Variant support.
