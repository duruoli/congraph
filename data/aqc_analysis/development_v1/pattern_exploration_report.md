# Development A/Q/C pattern exploration

Status: exploratory Stage-3 output; pattern definitions are not frozen.

This report is ACR-blind. Counts describe the 235-patient development partition and are
not prevalence estimates, causal effects, normative recommendations, or final-test results.
No rule uses the annotation's `derived_transition_reference` as an input.

## Candidate overview

| ID | Pattern | Opportunities | Candidate units | Patients | Strict units | Diseases |
|---|---|---:|---:|---:|---:|---:|
| AQC_P01 | question_reorientation | 198 | 109 | 81 | 27 | 4 |
| AQC_P02 | persistent_open_question | 198 | 180 | 132 | 50 | 4 |
| AQC_P03 | limited_study_persistent_gap | 70 | 63 | 49 | 20 | 4 |
| AQC_P04 | capability_limited_persistent_gap | 31 | 30 | 23 | 13 | 4 |
| AQC_P05 | unresolved_result_followup | 54 | 53 | 42 | 17 | 4 |
| AQC_P06 | informative_result_question_advance | 140 | 88 | 71 | 21 | 4 |
| AQC_P07 | temporal_requirement_serial_imaging | 198 | 14 | 7 | 3 | 2 |
| AQC_P08 | assumption_composition_change | 198 | 165 | 132 | 40 | 4 |
| AQC_P09 | material_discordance_followup | 198 | 1 | 1 | 1 | 1 |
| AQC_P10 | covered_question_with_further_imaging | 198 | 3 | 3 | 1 | 3 |
| AQC_P11 | order_support_concern | 433 | 58 | 40 | 12 | 4 |

## Definition diagnostics

- `AQC_P01`: no automatic definition warning.
- `AQC_P02`: near-universal within its opportunity set; discriminative value requires review.
- `AQC_P03`: near-universal within its opportunity set; discriminative value requires review.
- `AQC_P04`: near-universal within its opportunity set; discriminative value requires review.
- `AQC_P05`: near-universal within its opportunity set; discriminative value requires review.
- `AQC_P06`: no automatic definition warning.
- `AQC_P07`: no automatic definition warning.
- `AQC_P08`: no automatic definition warning.
- `AQC_P09`: rare boundary candidate; do not estimate a stable rate.
- `AQC_P10`: rare boundary candidate; do not estimate a stable rate.
- `AQC_P11`: no automatic definition warning.

## Example and counterexample queues

- `AQC_P01` candidates: `appendicitis:20276429:s1->appendicitis:20276429:s2`, `cholecystitis:20660601:s1->cholecystitis:20660601:s2`, `diverticulitis:20180280:s1->diverticulitis:20180280:s2`, `pancreatitis:20001800:s1->pancreatitis:20001800:s2`; opportunity counterexamples: `appendicitis:20123918:s1->appendicitis:20123918:s2`, `cholecystitis:20334898:s1->cholecystitis:20334898:s2`, `diverticulitis:21292285:s1->diverticulitis:21292285:s2`, `pancreatitis:20009550:s1->pancreatitis:20009550:s2`.
- `AQC_P02` candidates: `appendicitis:20123918:s1->appendicitis:20123918:s2`, `cholecystitis:20334898:s1->cholecystitis:20334898:s2`, `diverticulitis:20180280:s1->diverticulitis:20180280:s2`, `pancreatitis:20001800:s1->pancreatitis:20001800:s2`; opportunity counterexamples: `appendicitis:22881737:s1->appendicitis:22881737:s2`, `cholecystitis:21166109:s1->cholecystitis:21166109:s2`, `pancreatitis:20594178:s1->pancreatitis:20594178:s2`, `appendicitis:29167179:s1->appendicitis:29167179:s2`.
- `AQC_P03` candidates: `appendicitis:20123918:s1->appendicitis:20123918:s2`, `cholecystitis:20660601:s2->cholecystitis:20660601:s3`, `diverticulitis:21292285:s1->diverticulitis:21292285:s2`, `pancreatitis:20001800:s1->pancreatitis:20001800:s2`; opportunity counterexamples: `cholecystitis:22566240:s2->cholecystitis:22566240:s3`, `pancreatitis:20594178:s2->pancreatitis:20594178:s3`, `cholecystitis:26014747:s2->cholecystitis:26014747:s3`, `cholecystitis:27269012:s1->cholecystitis:27269012:s2`.
- `AQC_P04` candidates: `appendicitis:20123918:s1->appendicitis:20123918:s2`, `cholecystitis:20660601:s2->cholecystitis:20660601:s3`, `diverticulitis:26371704:s3->diverticulitis:26371704:s4`, `pancreatitis:20009550:s1->pancreatitis:20009550:s2`; opportunity counterexamples: `pancreatitis:21247680:s1->pancreatitis:21247680:s2`.
- `AQC_P05` candidates: `appendicitis:20646985:s1->appendicitis:20646985:s2`, `cholecystitis:20334898:s1->cholecystitis:20334898:s2`, `diverticulitis:21292285:s1->diverticulitis:21292285:s2`, `pancreatitis:20418179:s1->pancreatitis:20418179:s2`; opportunity counterexamples: `cholecystitis:27269012:s1->cholecystitis:27269012:s2`.
- `AQC_P06` candidates: `appendicitis:20276429:s1->appendicitis:20276429:s2`, `cholecystitis:20660601:s1->cholecystitis:20660601:s2`, `diverticulitis:20180280:s1->diverticulitis:20180280:s2`, `pancreatitis:20001800:s1->pancreatitis:20001800:s2`; opportunity counterexamples: `appendicitis:20123918:s1->appendicitis:20123918:s2`, `cholecystitis:20639685:s1->cholecystitis:20639685:s2`, `diverticulitis:23962694:s1->diverticulitis:23962694:s2`, `pancreatitis:20009550:s1->pancreatitis:20009550:s2`.
- `AQC_P07` candidates: `appendicitis:20123918:s1->appendicitis:20123918:s2`, `pancreatitis:21282967:s3->pancreatitis:21282967:s4`, `appendicitis:27628911:s1->appendicitis:27628911:s2`, `appendicitis:27993727:s1->appendicitis:27993727:s2`; opportunity counterexamples: `appendicitis:20123918:s2->appendicitis:20123918:s3`, `cholecystitis:20334898:s1->cholecystitis:20334898:s2`, `diverticulitis:20180280:s1->diverticulitis:20180280:s2`, `pancreatitis:20001800:s1->pancreatitis:20001800:s2`.
- `AQC_P08` candidates: `appendicitis:20123918:s1->appendicitis:20123918:s2`, `cholecystitis:20334898:s1->cholecystitis:20334898:s2`, `diverticulitis:20180280:s1->diverticulitis:20180280:s2`, `pancreatitis:20001800:s1->pancreatitis:20001800:s2`; opportunity counterexamples: `appendicitis:23562407:s1->appendicitis:23562407:s2`, `cholecystitis:20660601:s2->cholecystitis:20660601:s3`, `diverticulitis:22429578:s1->diverticulitis:22429578:s2`, `pancreatitis:20720063:s2->pancreatitis:20720063:s3`.
- `AQC_P09` candidates: `diverticulitis:23962694:s2->diverticulitis:23962694:s3`; opportunity counterexamples: `appendicitis:20123918:s1->appendicitis:20123918:s2`, `cholecystitis:20334898:s1->cholecystitis:20334898:s2`, `diverticulitis:20180280:s1->diverticulitis:20180280:s2`, `pancreatitis:20001800:s1->pancreatitis:20001800:s2`.
- `AQC_P10` candidates: `appendicitis:22881737:s2`, `cholecystitis:21166109:s2`, `pancreatitis:21849575:s4`; opportunity counterexamples: `appendicitis:20123918:s2`, `cholecystitis:20334898:s2`, `diverticulitis:20180280:s2`, `pancreatitis:20001800:s2`.
- `AQC_P11` candidates: `appendicitis:22710495:s1`, `cholecystitis:20660601:s4`, `diverticulitis:21793374:s2`, `pancreatitis:20001800:s2`; opportunity counterexamples: `appendicitis:20123918:s1`, `cholecystitis:20334898:s1`, `diverticulitis:20180280:s1`, `pancreatitis:20001800:s1`.

## Highest-overlap candidate pairs

- `AQC_P02` + `AQC_P08`: 149 target/anchor steps.
- `AQC_P01` + `AQC_P08`: 100 target/anchor steps.
- `AQC_P01` + `AQC_P02`: 94 target/anchor steps.
- `AQC_P01` + `AQC_P06`: 88 target/anchor steps.
- `AQC_P06` + `AQC_P08`: 83 target/anchor steps.
- `AQC_P02` + `AQC_P06`: 74 target/anchor steps.
- `AQC_P02` + `AQC_P03`: 63 target/anchor steps.
- `AQC_P02` + `AQC_P05`: 53 target/anchor steps.
- `AQC_P03` + `AQC_P08`: 48 target/anchor steps.
- `AQC_P05` + `AQC_P08`: 38 target/anchor steps.
- `AQC_P03` + `AQC_P05`: 36 target/anchor steps.
- `AQC_P02` + `AQC_P11`: 36 target/anchor steps.
- `AQC_P08` + `AQC_P11`: 32 target/anchor steps.
- `AQC_P02` + `AQC_P04`: 30 target/anchor steps.
- `AQC_P01` + `AQC_P03`: 28 target/anchor steps.

## Boundaries retained for Stage 4

- `close_or_stop` remains unidentifiable without an explicit observation-window/censoring rule.
- `escalation` remains undefined until modality, protocol, and intervention changes receive a purely observational hierarchy.
- `AQC_P09` uses only `materially_discordant`; `indeterminate` cases remain a separate manual audit set.
- `AQC_P11` must be reported separately by schema generation; its two support fields are not pooled as equivalent measurements.
- High candidate fractions in P02--P05 may reflect how opportunity sets were defined and require counterexample review before freezing.
