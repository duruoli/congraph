# ACR Variant Predicate Audit v1

Status: draft for manual review. This audit covers all 17 selected ACR Variants. Exact
`variant_text` is authoritative; normalized predicates are project-created comparison units. The
matching ACR narrative is used only where it explicitly clarifies a missing target or scope, and
those additions are marked.

## How to read this audit

- `surface phrase` is the exact text evidence.
- `condition instance` is the comparison-ready meaning recovered from that text and its Variant.
- `predicate type` is the reusable question form shared by ACR and patient-record extraction.
- `factual` versus `inferential` is a separate epistemic property, not a predicate type.
- `required`, `alternative`, and `illustrative` describe the phrase's role in the Variant.

Machine-readable details, including exact spans, derivations, roles, and logical groups, are in
`acr_variant_predicate_audit_v1.json`. Predicate definitions are in
`acr_predicate_types_v1.json`.

## Predicate types induced from the 17 Variants

| Predicate type | Intuitive question | Usual kind |
|---|---|---|
| `patient_attribute` | What patient characteristic changes applicability? | factual |
| `symptom_state` | What symptom is present, absent, or persistent, and where? | factual |
| `objective_finding_state` | What sign, vital sign, or laboratory state is present or changing? | factual |
| `diagnostic_state` | What disease or complication is suspected, established, or known? | inferential |
| `etiology_state` | Is the cause of the presentation known or suspected? | inferential |
| `test_history` | What test was already completed? | factual |
| `test_interpretation` | What did a test mean with respect to which target? | inferential |
| `presentation_interpretation` | How typical is the presentation for a diagnostic target? | inferential |
| `severity_or_course_assessment` | What synthesized severity or trajectory state applies? | inferential |
| `evidence_relation` | How does one item qualify or alter another interpretation? | inferential |
| `temporal_position` | Where is the patient relative to a clinical time anchor? | factual |
| `decision_stage` | Where is the encounter or imaging decision in its sequence? | factual |

## 1. Right Lower Quadrant Pain, Variant 1

> Right lower quadrant pain. Initial imaging.

```text
symptom_state(abdominal_pain, site=right_lower_quadrant, present)       [factual, required]
decision_stage(imaging, initial)                                      [factual, required]
```

Logic: both conditions jointly define the Variant.

## 2. Right Lower Quadrant Pain, Variant 2

> Right lower quadrant pain, fever, leukocytosis. Suspected appendicitis. Initial imaging.

```text
symptom_state(abdominal_pain, site=right_lower_quadrant, present)       [factual, required]
objective_finding_state(fever, present)                                [factual, required]
objective_finding_state(white_blood_cell_count, high)                  [factual, required]
diagnostic_state(appendicitis, suspected)                              [inferential, required]
decision_stage(imaging, initial)                                      [factual, required]
```

Logic: the title presents a joint scenario; it contains no OR marker.

## 3. Right Lower Quadrant Pain, Variant 3

> Pregnant woman. Right lower quadrant pain, fever, leukocytosis. Suspected appendicitis. Initial imaging.

```text
patient_attribute(pregnancy, present)                                 [factual, required]
symptom_state(abdominal_pain, site=right_lower_quadrant, present)       [factual, required]
objective_finding_state(fever, present)                                [factual, required]
objective_finding_state(white_blood_cell_count, high)                  [factual, required]
diagnostic_state(appendicitis, suspected)                              [inferential, required]
decision_stage(imaging, initial)                                      [factual, required]
```

Logic: pregnancy is a separate applicability condition; it is not part of the appendicitis judgment.

## 4. Right Upper Quadrant Pain, Variant 1

> Right upper quadrant pain. Unknown etiology. Initial Imaging.

```text
symptom_state(abdominal_pain, site=right_upper_quadrant, present)       [factual, required]
etiology_state(RUQ_pain, unknown)                                     [inferential, required]
decision_stage(imaging, initial)                                      [factual, required]
```

Scope: `Unknown etiology` is incomplete alone; its target is the current RUQ-pain presentation.

## 5. Right Upper Quadrant Pain, Variant 2

> Right upper quadrant pain. Suspected biliary disease. Initial imaging.

```text
symptom_state(abdominal_pain, site=right_upper_quadrant, present)       [factual, required]
etiology_state(RUQ_pain, biliary_disease, suspected)                   [inferential, required]
decision_stage(imaging, initial)                                      [factual, required]
```

Scope: biliary disease is the suspected cause of the RUQ-pain presentation.

## 6. Right Upper Quadrant Pain, Variant 3

> Right upper quadrant pain. No fever and no high white blood cell (WBC) count. Suspected biliary disease. Negative or equivocal ultrasound. Next imaging study.

```text
symptom_state(abdominal_pain, site=right_upper_quadrant, present)       [factual, required]
objective_finding_state(fever, absent)                                 [factual, required]
objective_finding_state(white_blood_cell_count, not_high)              [factual, required]
etiology_state(RUQ_pain, biliary_disease, suspected)                   [inferential, required]
test_history(ultrasound, completed_before_current_decision)            [factual, required]
test_interpretation(ultrasound, target=acute_cholecystitis,
                    result=negative_or_equivocal)                      [inferential, required]
decision_stage(imaging, next_after_ultrasound)                         [factual, required]
```

Logic: no fever AND no high WBC. Ultrasound result is negative OR equivocal. The title gives only
the broad frame `Suspected biliary disease`; the Variant 3 narrative clarifies the result target as
acute cholecystitis and additionally says there is no alternative diagnosis.

## 7. Right Upper Quadrant Pain, Variant 4

> Right upper quadrant pain. Fever, elevated WBC count. Suspected biliary disease. Negative or equivocal ultrasound. Next imaging study.

```text
symptom_state(abdominal_pain, site=right_upper_quadrant, present)       [factual, required]
objective_finding_state(fever, present)                                [factual, required]
objective_finding_state(white_blood_cell_count, high)                  [factual, required]
etiology_state(RUQ_pain, biliary_disease, suspected)                   [inferential, required]
test_history(ultrasound, completed_before_current_decision)            [factual, required]
test_interpretation(ultrasound, target=acute_cholecystitis,
                    result=negative_or_equivocal)                      [inferential, required]
decision_stage(imaging, next_after_ultrasound)                         [factual, required]
```

Logic: fever and elevated WBC are presented jointly. Ultrasound result is negative OR equivocal.
The Variant 4 narrative clarifies the target as acute cholecystitis and additionally says there is
no alternative diagnosis.

## 8. Right Upper Quadrant Pain, Variant 5

> Right upper quadrant pain. Suspected acalculous cholecystitis. Negative or equivocal ultrasound. Next imaging study.

```text
symptom_state(abdominal_pain, site=right_upper_quadrant, present)       [factual, required]
diagnostic_state(acalculous_cholecystitis, suspected)                  [inferential, required]
test_history(ultrasound, completed_before_current_decision)            [factual, required]
test_interpretation(ultrasound, target=acalculous_cholecystitis,
                    result=negative_or_equivocal)                      [inferential, required]
decision_stage(imaging, next_after_ultrasound)                         [factual, required]
```

Scope: the same surface phrase `Negative or equivocal ultrasound` has a more specific target here
than in Variants 3 and 4.

## 9. Left Lower Quadrant Pain, Variant 1

> Left lower quadrant pain. Initial imaging.

```text
symptom_state(abdominal_pain, site=left_lower_quadrant, present)        [factual, required]
decision_stage(imaging, initial)                                      [factual, required]
```

## 10. Left Lower Quadrant Pain, Variant 2

> Left lower quadrant pain. Suspected diverticulitis. Initial imaging.

```text
symptom_state(abdominal_pain, site=left_lower_quadrant, present)        [factual, required]
diagnostic_state(diverticulitis, suspected)                            [inferential, required]
decision_stage(imaging, initial)                                      [factual, required]
```

## 11. Left Lower Quadrant Pain, Variant 3

> Left lower quadrant pain. Suspected complication(s) of diverticulitis. Initial imaging.

```text
symptom_state(abdominal_pain, site=left_lower_quadrant, present)        [factual, required]
diagnostic_state(complication_of_diverticulitis, suspected)            [inferential, required]
decision_stage(imaging, initial)                                      [factual, required]
```

Scope: ACR does not specify which complication. A record-level abscess or perforation would be a
narrower patient condition, not an exact repetition of the ACR phrase.

## 12. Acute Pancreatitis, Variant 1

> Suspected acute pancreatitis. First time presentation. Epigastric pain and increased amylase and lipase. Less than 48 to 72 hours after symptom onset. Initial imaging.

```text
diagnostic_state(acute_pancreatitis, suspected)                        [inferential, required]
decision_stage(acute_pancreatitis_presentation, first_time)            [factual, required]
symptom_state(abdominal_pain, site=epigastrium, present)                [factual, required]
objective_finding_state(amylase, increased)                            [factual, required]
objective_finding_state(lipase, increased)                             [factual, required]
temporal_position(current_decision, <48_to_72_hours,
                  anchor=acute_pancreatitis_symptom_onset)             [factual, required]
decision_stage(imaging, initial)                                      [factual, required]
```

Logic: epigastric pain AND increased amylase AND increased lipase. The threshold remains
`48 to 72 hours`; this audit does not invent a single cutoff.

## 13. Acute Pancreatitis, Variant 2

> Suspected acute pancreatitis. Initial presentation with atypical signs and symptoms; including equivocal amylase and lipase values (possibly confounded by acute kidney injury or chronic kidney disease) and when diagnoses other than pancreatitis may be possible (bowel perforation, bowel ischemia, etc.). Initial imaging.

```text
diagnostic_state(acute_pancreatitis, suspected)                        [inferential, required]
decision_stage(acute_pancreatitis_presentation, initial)               [factual, required]
presentation_interpretation(signs_and_symptoms, acute_pancreatitis,
                            atypical)                                  [inferential, required]
test_interpretation(amylase, acute_pancreatitis, equivocal)            [inferential, illustrative]
test_interpretation(lipase, acute_pancreatitis, equivocal)             [inferential, illustrative]
evidence_relation(renal_disease possibly_confounds enzyme_result)      [inferential, possible qualifier]
evidence_relation(alternative_to_pancreatitis may_be_possible)         [inferential, alternative scenario]
decision_stage(imaging, initial)                                      [factual, required]
```

Logic uncertainty: the wording does not cleanly say that equivocal enzymes, renal confounding,
and alternative diagnoses must all coexist. They must not be compiled as strict AND without
adjudication.

## 14. Acute Pancreatitis, Variant 3

> Acute pancreatitis. Critically ill, systemic inflammatory response syndrome (SIRS), severe clinical scores (eg, Acute Physiology, Age, and Chronic Health Evaluation [APACHE]-II, Bedside Index for Severity in AP [BISAP], or Marshall). Greater than 48 to 72 hours after onset of symptoms.

```text
diagnostic_state(acute_pancreatitis, established)                      [inferential, required]
severity_or_course_assessment(critical_illness, present)               [inferential, descriptor]
severity_or_course_assessment(SIRS, present)                           [inferential, descriptor]
severity_or_course_assessment(severe_score,
                              system=APACHE_II_or_BISAP_or_Marshall)   [inferential, descriptor]
temporal_position(current_decision, >48_to_72_hours,
                  anchor=acute_pancreatitis_symptom_onset)             [factual, required]
```

Logic uncertainty: commas do not establish whether all three severity descriptors are mandatory
or overlapping ways of identifying severe illness.

## 15. Acute Pancreatitis, Variant 4

> Acute pancreatitis. Continued SIRS, severe clinical scores, leukocytosis, and fever. Greater than 7 to 21 days after onset of symptoms.

```text
diagnostic_state(acute_pancreatitis, established)                      [inferential, required]
severity_or_course_assessment(SIRS, persistent)                        [inferential, course descriptor]
severity_or_course_assessment(severe_clinical_score, present)          [inferential, severity descriptor]
objective_finding_state(white_blood_cell_count, high)                  [factual, objective descriptor]
objective_finding_state(fever, present)                                [factual, objective descriptor]
temporal_position(current_decision, >7_to_21_days,
                  anchor=acute_pancreatitis_symptom_onset)             [factual, required]
```

Logic uncertainty: the final `and` favors conjunction, but the title does not explain whether
every descriptor is mandatory. The 7-to-21-day range is preserved.

## 16. Acute Pancreatitis, Variant 5

> Known necrotizing pancreatitis. Significant deterioration in clinical status, including abrupt decrease in hemoglobin or hematocrit, hypotension, tachycardia, tachypnea, abrupt change in fever curve, or increase in white blood cells.

```text
diagnostic_state(necrotizing_pancreatitis, known)                      [inferential, required]
severity_or_course_assessment(clinical_status, significant_deterioration)
                                                                        [inferential, required]

one or more evidence items supporting deterioration:
  objective_finding_state(hemoglobin, abrupt_decrease)                 [factual, alternative]
  objective_finding_state(hematocrit, abrupt_decrease)                 [factual, alternative]
  objective_finding_state(blood_pressure, hypotension)                 [factual, alternative]
  objective_finding_state(heart_rate, tachycardia)                     [factual, alternative]
  objective_finding_state(respiratory_rate, tachypnea)                 [factual, alternative]
  objective_finding_state(fever_curve, abrupt_change)                  [factual, alternative]
  objective_finding_state(white_blood_cell_count, increase)            [factual, alternative]
```

Logic: known necrotizing pancreatitis AND significant deterioration. The listed findings are
alternative evidence for the deterioration judgment, not seven jointly required conditions.

## 17. Acute Pancreatitis, Variant 6

> Acute pancreatitis. Known pancreatic or peripancreatic fluid collections with continued abdominal pain, early satiety, nausea, vomiting, or signs of infection. Greater than 4 weeks after symptom onset.

```text
diagnostic_state(acute_pancreatitis, established)                      [inferential, required]
diagnostic_state(pancreatic_or_peripancreatic_fluid_collection, known) [inferential, required]

one or more associated states:
  symptom_state(abdominal_pain, persistent)                            [factual, alternative]
  symptom_state(early_satiety, present)                               [factual, alternative]
  symptom_state(nausea, present)                                      [factual, alternative]
  symptom_state(vomiting, present)                                    [factual, alternative]
  diagnostic_state(collection_infection, suspected_by_signs)
                                                                        [inferential, alternative]

temporal_position(current_decision, >4_weeks,
                  anchor=acute_pancreatitis_symptom_onset)             [factual, required]
```

Logic: acute pancreatitis AND known collection AND one-or-more associated states AND timing.
`Signs of infection` is a synthesized condition unless its component findings are separately
available in the patient record.

## Main audit findings

1. The old 50-value list mixes complete conditions, arguments, and incomplete fragments.
2. ACR result words such as `negative` and `equivocal` require both a test and a diagnostic target.
3. The same surface phrase can compile differently under different Variants.
4. ACR contains nested logic: explicit AND, explicit OR, alternative evidence, examples, and
   punctuation whose logical force remains uncertain.
5. ACR-to-record comparison should operate on condition instances while retaining surface phrases
   and unresolved logic for audit.
