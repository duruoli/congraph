# IBD 20-patient pre-action decision-candidate pilot

Run `python3 scripts/build_ibd_decision_steps.py` to regenerate the private
`runs/ibd_decision_steps_pilot_20/` files from the fixed event ledger. The
script derives action groups itself, so it can run on any ledger produced by
`scripts/build_ibd_event_timelines.py`. For example:

```bash
python3 scripts/build_ibd_decision_steps.py \
  --ledger-dir runs/ibd_event_timeline_pilot_100_additional \
  --output-dir runs/ibd_decision_steps_pilot_100_additional
```

This is an automatic **candidate** extraction for the proposed decision-step
unit, not a final set of confirmed decisions.

## Unit and files

One candidate step precedes a recorded GI imaging exam, GI microbiology
specimen collection, selected prescription start, or relevant billed procedure
day. Same-time target actions form one step; date-only procedures on the same
day form a bundle whose internal order is unknown. The target action is stored
separately:

- `decision_candidates.jsonl`: action proxy boundary, definitely earlier
  available-evidence event IDs, overlapping/ambiguous event IDs, and empty
  fields for the decision question and management domain.
- `observed_outcomes.jsonl`: subsequent recorded target action and source IDs.
- `selection_report.json`: counts and limitations.

The eight axes should describe the **decision context after clinical review**.
They must not be populated from the action in `observed_outcomes.jsonl` or from
its later result. A clinical reviewer must decide whether a candidate really
represents a recoverable decision, what question was at issue, and whether its
management domain can be supported by pre-action evidence. The separate
observed action can then be used as an outcome, not an input label.

## Timing and eligibility

The extract lacks test and medication **order times**. Radiology chart/exam
time, microbiology specimen time, prescription start, and billed procedure day
are action proxies. The algorithm needs only relative order: an information
event is `before` when its availability interval ends before the action proxy
interval begins. Exact timestamps are compared with `<`. Date-only availability
is placed before a target only after that whole date has ended. Equal-time or
same-day availability is recorded separately as `overlap`, never inserted into
the candidate input. The target action itself, later results, discharge notes,
retrospective ICD diagnoses, and records with unknown availability are excluded.

This is an **upper bound on information available before the recorded action**,
not a verified information set before the underlying decision. A planned
treatment may already appear in an earlier report before its prescription
start; that establishes an earlier plan, but not necessarily a finalized order
or the absence of a later proceed/hold decision. An available result also does
not prove clinician review. Thus
`pre_order_evidence_certified` remains false for every automatically extracted
candidate. A clinical reviewer must distinguish initial selection from later
confirmation and reject implementation-only anchors or move the decision
anchor to an earlier documented planning point where possible.

This **local extract** omits `poe` and `poe_detail`. Original MIMIC-IV has
provider order entries with `ordertime` and, for a subset of orders,
`poe_detail.field_name = 'Indication'`. Future extraction of those tables may
replace some action proxies with observed order anchors and improve relative
ordering. It will not reveal every private clinical deliberation or provide an
indication for every order.

Generic lab panels and admission transitions do not seed steps in this pass.
Repeated starts of the same drug and route within an admission remain visible
but receive lower review priority; dose changes cannot be identified from the
current ledger. Prescriptions indicate intended starts, not administration or
indication. Previous IBD-coded admissions may contribute earlier evidence;
non-IBD-coded and outpatient records are absent.

## First run

The 20-patient pilot yields **332 candidates across all 20 patients**: 35
imaging, 35 microbiology, 207 prescription-start, and 55 date-only procedure
groups. Eleven have no definitely earlier evidence; all 55 date-only procedure
groups have overlapping same-day information. One hundred prescription groups
are lower-priority repeats of a recorded drug and route. On the disjoint
100-patient pilot the same algorithm produces 1,524 candidates across 98
patients. These are counts of action-conditioned candidates, **not** verified
clinical decisions. The stream cannot identify true no-action or stopping
decisions. Clinical review and a separate opportunity-sampling policy are
needed before treating this as a decision-step dataset.
