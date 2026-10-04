# IBD 20-patient action-step pilot

Run `python3 scripts/build_ibd_action_steps.py` to regenerate the action-step
files from the fixed 20-patient event ledger. The selector uses standard Python
only. It does not read report text, lab values, discharge notes, or retrospective
ICD diagnoses when choosing steps.

## What counts as an action step

An action step is a group of recorded actions with the same patient, admission,
event time, and time precision. The candidate actions are:

- admission (care transition);
- lab or microbiology specimen collection time;
- radiology exam/chart time;
- prescription start time (intended treatment, **not administration**);
- billed procedure date (a 24-hour interval, **not an exact time**).

An action is not automatically a clinician decision, and this selector does not
pair an exam with its later report. A treatment can occur between an exam and its
report.

The subsequent pre-action decision-candidate extraction is described in
`IBD_DECISION_STEP_PILOT_20.md`. It uses this action stream as observable
outcomes while keeping decision-side evidence separate.

## Information steps for eight-axis state labels

Run `python3 scripts/build_ibd_information_steps.py` to generate
`runs/ibd_information_steps_pilot_20/information_steps.jsonl`. This is the
primary candidate step stream for labeling what is known about a patient at
time `t`. Each step is anchored at `available_at`, when a selected result or
report becomes available, rather than at test collection/exam time. The
20-patient first pass has 505 information steps from 1,675 availability
checkpoints: admission, all radiology reports, all microbiology result updates,
and selected lab results (CRP, calprotectin, ESR, albumin, lactate, CMV viral
load). The lab
rule is provisional; all omitted results remain in the checkpoint ledger for
review. Each selected step also has `new_since_previous_step_event_ids`, which
includes unselected results that arrived between two selected steps. A report's
arrival does not prove a clinician read it.

The action and information streams are not forced into alternating pairs.
Action steps describe recorded actions; information steps describe changes to
the available evidence. Prescription and procedure times are too uncertain to
merge directly into a strict as-of eight-axis snapshot without review.

## Selection rule

`candidate_actions.jsonl` contains every grouped action, including unselected
groups. `action_steps.jsonl` contains groups with at least one of:

- admission;
- any lab collection (same-time lab rows are one group);
- GI/IBD-related microbiology, imaging, or procedure according to the regular
  expressions in `scripts/build_ibd_action_steps.py`;
- steroid/aminosalicylate, biologic/immunomodulator, antimicrobial, or
  nutritional prescription according to the regular expressions in that script.

These medication groups describe the drug name, not its indication. For example,
a steroid prescription is not automatically an IBD treatment.

Other prescriptions and unrelated tests/procedures remain visible in the
candidate file. The selection rules are deliberately explicit and provisional.
The pilot should review both false positives and missed clinically important
actions before using the same policy for another 100 patients.

## Output and temporal rules

The default private output directory is `runs/ibd_action_steps_pilot_20/`:

- `candidate_actions.jsonl`: all grouped action candidates and selection flags;
- `action_steps.jsonl`: selected steps with consecutive `step_index` per patient;
- `selection_report.json`: counts and known limitations.

Each action keeps its source event ID and source key. `event_at` is the recorded
event time; `available_at` is the result/report availability proxy where known.
`strict_asof_action_evidence=false` means the action record itself was not known
to be available by `event_at`. In particular, a lab result must not be read at
the specimen-collection step merely because the lab row links to that step.

Date-only procedures carry `event_end_exclusive`; their midnight display
position does not establish order among other actions that day. Unknown times
remain unknown. Eight-axis labels are not produced by this script.

## First run

The fixed 20-patient pilot produced 2,156 candidate action groups and 987
selected steps. Of the selected steps, 602 include lab collection, 55 have
date-only procedure time, and 57 have an action record known to be available at
its event time (the admission anchors). These figures are counts of action
anchors, not counts of decisions or eight-axis annotations.
