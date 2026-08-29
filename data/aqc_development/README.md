# Empirical A/Q/C development

Status: **formal first-layer development draft; lexically stable, not frozen or clinically
saturated**.

## Boundary

This directory is a separate analysis layer derived from the existing rubric-free, order-aware
annotations in `results/annotation_experiment/full/`. The original annotations are unchanged.
ACR v1.1 is an independent, read-only normative input and was not used to define, select, or code
A/Q/C types.

Discovery and sampling exclude verification, deviation, ACR ratings, the current order result,
later events, and final-diagnosis correctness. Disease is used only to preserve sampling coverage;
it is not exposed as an A/Q/C answer and does not define four disease-specific ontologies.

## Corpus split

The eligible corpus contains 293 patient trajectories and 542 decision steps. The stable split uses
SHA-256 within disease with the salt recorded in `split_manifest.json`:

| Partition | Patients | Steps | Appendicitis | Cholecystitis | Diverticulitis | Pancreatitis |
|---|---:|---:|---:|---:|---:|---:|
| Development | 235 | 433 | 58 | 72 | 27 | 78 |
| Final test | 58 | 109 | 13 | 18 | 7 | 20 |

All steps from a patient inherit one partition. Patient `appendicitis/20123918` was opened once for
source-schema inspection, so it is explicitly assigned to development without replacement; this
keeps the remaining 58-patient final test untouched rather than promoting a development record that
had already been profiled. Final-test entries in the split manifest contain
identity/path/count metadata only and were not profiled or opened for sample selection or coding.

## Development sample

The initial formal discovery batch contains 24 trajectories (six per disease) and 83 steps. Two
fresh non-overlapping development batches add 12 trajectories each (three per disease), producing a
48-patient, 137-step analytical layer. Selection is deterministic maximum variation with salted
hash tie-breaking over pre-order structure: length, repeat/switch, modality sequence, old action
role, timing, prior-study limitation, nonvisualization/not-assessed language, and differential
dispersion. These counts describe a purposive sample, not prevalence.

Across all 48 trajectories the audit contains 3 single-step and 45 multi-step trajectories, 18
repeat trajectories, 39 modality switches, 25 prior-study-limited trajectories, 17 target-
nonvisualized trajectories, 3 target-not-assessed trajectories, and 15 post-intervention
trajectories. See `diversity_audit.json` for modality sequences, roles, and timing strata.

## Two reading views

`discovery_open_coding.jsonl` stores one row per decision step:

1. `view_1_reasoning_only`: verbatim `reasoning`, with the old differential/information-gap/
   expected-finding/action-role scaffold hidden;
2. `view_2_schema_light`: the complete allowed ex-ante reconstruction fields, copied verbatim;
3. `view_comparison`: candidates appearing only after the schema-light view and a scaffold-
   induction flag.

Every row retains patient, step, source path, and source digest. No ACR, verification, deviation,
current result, or later event is serialized into this layer.

The deterministic first pass found at least one schema-only candidate in 129 of 137 steps. This is
a strong warning that the old field structure shapes candidate recovery. It is therefore incorrect
to call the codebook qualitatively saturated merely because the same lexical rule set returned no
new top-level family in two fresh batches.

## Formal first-version contracts

### Assumption

`assumption_codebook_v1.json` uses atomic propositions with independent type, level, status,
evidence, and support. The top-level types are:

- `syndrome_or_source_frame`
- `disease_or_finding_identity`
- `etiology_or_mechanism`
- `severity_extent_or_course`
- `complication`
- `alternative_source`
- `intervention_or_device_state`
- `other`
- `unclear`

The rules explicitly preserve hierarchical uncertainty. For example, established pancreatitis and
suspected biliary etiology are two propositions, not one averaged confidence label.

### Question and answer requirements

`question_codebook_v1.json` requires one primary decision-relevant unknown and optional supported
secondary questions. Question types are source localization, existence/identity,
etiology/mechanism, severity/extent, complication, alternative source, intervention/device state,
other, and unclear. Every question states what positive and negative answers would change.

`answer_requirements[]` contains information dimensions, never preferred modalities. The current
dimensions are target assessment, presence/absence, anatomic localization, finding identity,
etiologic mechanism, severity/extent, temporal course/response, complication characterization,
alternative-source discrimination, device position/integrity, device/intervention function,
other, and unclear.

### Coverage

`coverage_contract_v1.json` records each declared requirement separately as:

```text
unaddressed | partially_addressed | sufficiently_addressed
```

Each entry also carries exact supporting evidence and direction:

```text
supports | refutes | mixed | no_direction
```

An optional aggregate (`unanswered | partially_answered | sufficiently_answered`) is shorthand
only. Study adequacy, test-question capability, result status, and aggregate coverage remain
separate. A valid negative may sufficiently cover a requirement with direction `refutes`;
nondiagnostic, nonvisualized, and not-assessed findings are not valid negatives.

## Saturation and freeze status

The initial 24-patient batch introduced all candidate top-level families returned by the current
rules. Neither 12-patient check batch introduced another lexical family or answer-requirement
dimension. This establishes **lexical stability under the current rules**, not qualitative
saturation.

The codebook remains `not_frozen` because:

- 129/137 steps show potential old-scaffold induction;
- schema-only candidates have not been independently adjudicated;
- the direct masked-chart versus recode framework check has not been run on fresh development
  patients;
- requirement-level coverage has a contract but not an independently reviewed annotation batch;
- inter-rater or clinician review has not been performed.

Do not open final test, start bulk annotation, estimate final patterns, or claim prevalence at this
stage. The next gate is an independent dual-route framework check on unused development patients,
followed by manual/clinical adjudication and another fresh development re-check if the schema
changes.

## Files

- `split_manifest.json`: reproducible patient-level development/final-test split.
- `development_sample_manifest.json`: initial and two fresh development batches.
- `diversity_audit.json`: structural-coverage audit.
- `discovery_open_coding.jsonl`: traceable two-view candidate coding.
- `assumption_codebook_v1.json`: atomic assumption rules.
- `question_codebook_v1.json`: question and answer-requirement rules.
- `coverage_contract_v1.json`: requirement-level coverage contract.
- `audit_records.json`: residual, boundary, scaffold, and over-rationalization audit.
- `saturation_audit.json`: per-round candidate changes and honest freeze conclusion.
- `open_coding_memos.md`: historical prototype memos; use the formal files above for current work.
- `provisional_assumption_codebook.json`: historical 16-trajectory prototype, explicitly superseded.
- `../../experiments/aqc/prompts.py`: updated output contract.
- `../../scripts/build_aqc_discovery_sample.py`: deterministic split/sample/open-coding rebuild.
- `../../scripts/validate_aqc_development.py`: integrity, enum, partition, and leakage checks.

## Local commands

```bash
python scripts/build_aqc_discovery_sample.py
python scripts/validate_aqc_development.py
```

The repository-local Windows environment uses Python 3.12:

```powershell
.\.venv\Scripts\python.exe scripts\build_aqc_discovery_sample.py
.\.venv\Scripts\python.exe scripts\validate_aqc_development.py
```

If that virtual environment is unavailable, the split/sample/open-coding rebuild can also be run
with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_aqc_discovery_sample.ps1
```

The formal codebooks and audit files are curated analytical artifacts; rebuilding the sample does
not overwrite them.
