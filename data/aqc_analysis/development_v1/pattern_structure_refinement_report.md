# Stage-3 A/Q/C pattern structural refinement

Status: development-only, ACR-blind, exploratory, and not frozen.

## Question reorientation (P01)

- `both_continuity_and_type_change`: 8 transitions / 7 patients.
- `continuity_only`: 11 transitions / 9 patients.
- `type_change_only`: 90 transitions / 69 patients.

## Persistent-open-question family (P02--P07)

- P02 contains 180 transitions; 90 have at least one P03/P04/P05/P07 flag and 90 have none.
- `AQC_P03`: 21 transitions / 20 patients.
- `AQC_P03+AQC_P04`: 3 transitions / 3 patients.
- `AQC_P03+AQC_P04+AQC_P05`: 19 transitions / 14 patients.
- `AQC_P03+AQC_P04+AQC_P05+AQC_P07`: 2 transitions / 1 patients.
- `AQC_P03+AQC_P04+AQC_P07`: 2 transitions / 2 patients.
- `AQC_P03+AQC_P05`: 15 transitions / 15 patients.
- `AQC_P03+AQC_P07`: 1 transitions / 1 patients.
- `AQC_P04`: 3 transitions / 3 patients.
- `AQC_P04+AQC_P05`: 1 transitions / 1 patients.
- `AQC_P05`: 14 transitions / 13 patients.
- `AQC_P05+AQC_P07`: 2 transitions / 1 patients.
- `AQC_P07`: 7 transitions / 6 patients.
- `none_of_P03_P04_P05_P07`: 90 transitions / 75 patients.

## Assumption composition (P08)

- `both_type_multiset_and_shared_type_status_change`: 92 transitions / 80 patients.
- `shared_type_status_change_only`: 25 transitions / 24 patients.
- `type_multiset_change_only`: 48 transitions / 40 patients.

## Order-support audit (P11)

- `current_question_grounding`: 29/189 candidates (0.1534); 19 patients.
- `legacy_intent_support`: 29/244 candidates (0.1189); 21 patients.

## Draft disposition before codebook freeze

- `AQC_P01`: **retain_and_split_signals** — Retain as a structural descriptor, but report continuity and question-type signals separately.
- `AQC_P02`: **retain_as_backbone_only** — It defines continued imaging with an open question; it is a conditional backbone, not a standalone predictive finding.
- `AQC_P03--AQC_P05`: **retain_as_overlapping_mechanism_flags** — Use within P02 and report mutually exclusive combinations; do not present their selected-denominator fractions as effect sizes.
- `AQC_P06`: **retain_under_P01** — It identifies reorientation after an informative result but cannot distinguish advance from reroute without review.
- `AQC_P07`: **retain_with_semantic_guardrail** — Complete targeted review retained 13 of 14 and marked one unclear; require an explicit temporal requirement plus the P02 backbone.
- `AQC_P08`: **refine_before_freeze** — Separate type-multiset from shared-type status changes; neither is proposition-level belief revision.
- `AQC_P09`: **downgrade_to_annotation_audit** — The sole material-discordance candidate was excluded as false discordance; no retained development occurrence remains.
- `AQC_P10`: **retain_as_rare_boundary_audit** — Complete review retained two audit cases and excluded one annotation inconsistency; this is not a stable empirical rate.
- `AQC_P11`: **retain_as_schema_specific_annotation_audit** — It is not a transition pattern and the legacy/current support constructs cannot be pooled.

These dispositions organize the next manual calibration step. They are not frozen definitions,
replication results, or evidence that A/Q/C predicts the next order.
