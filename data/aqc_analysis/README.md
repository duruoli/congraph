# A/Q/C analysis layers

`development_v1/` is the canonical effective development layer for empirical A/Q/C pattern
analysis. Build it with:

```bash
python scripts/build_aqc_analysis_ready.py
```

The builder reads only patients assigned to development by
`data/aqc_development/split_manifest.json`, selects the GPT-5.1 DIRECT annotation for each patient,
and applies an explicit allowlist of final adjudication overlays. Original model outputs are never
modified. Dry-run and unadjudicated proposed overlays are excluded.

The release contains:

- `patients.jsonl`: one row per patient/trajectory;
- `steps.jsonl`: one row per imaging decision, including the complete effective annotation;
- `requirements.jsonl`: one row per question answer requirement and its matched coverage entry;
- `transitions.jsonl`: one row per adjacent within-patient decision transition;
- `effective_annotations.jsonl`: effective nested patient trajectories;
- `manifest.json`: counts, versions, input hashes, quality flags, and analysis policies.

Per-correction rationales are intentionally not copied into the analysis layer. Input file paths and
hashes retain reproducibility. Legacy `intent_support` and current `question_grounding` remain
separate fields because their semantics are not interchangeable.

An invalid or incomplete step fails the build. Unclear, uncertain, or weakly supported values are
retained and flagged rather than globally removed. A pattern-specific complete-case analysis must
be accompanied by a sensitivity analysis that includes these records.

## Stage 2 pattern units

`pattern_codebook_draft_v1.json` defines the first operational, ACR-blind pattern units,
opportunity denominators, exclusions, coexistence policy, and interpretation limits. It is a
development draft, not a frozen result.

Run the feasibility audit with:

```bash
python scripts/audit_aqc_pattern_units.py
```

The generated `development_v1/pattern_unit_feasibility.json` contains mechanical counts used to
check whether definitions are executable and sufficiently populated. These counts are not yet
scientific findings. No rule uses the annotation's `derived_transition` field as an input.

## Stage 3 development exploration

Run the ACR-blind development exploration with:

```bash
python scripts/analyze_aqc_patterns.py
python scripts/refine_aqc_pattern_structure.py
python scripts/select_aqc_pattern_calibration_sample.py
```

It produces:

- `pattern_opportunities.jsonl`: every pattern-specific denominator row with candidate status;
- `pattern_occurrences.jsonl`: candidate rows with compact source/target A/Q/C snapshots;
- `pattern_review_queue.jsonl`: rare, unclear, temporal, weak-assumption, or incompletely audited
  candidates requiring targeted review, deduplicated by anchor step;
- `pattern_manual_review_sample.jsonl`: all high-risk candidate classes plus deterministic diverse
  candidate and opportunity-counterexample examples for every draft pattern;
- `pattern_exploration_summary.json`: patient-deduplicated and disease/schema/action-stratified
  summaries, strict sensitivity counts, version diagnostics, and pattern overlaps;
- `pattern_exploration_report.md`: concise human-readable development report.
- `pattern_structure_refinement.json` and `pattern_structure_refinement_report.md`: decompositions
  of the broad P01/P08 signals, mutually exclusive P02-family mechanism combinations,
  schema-separated P11 diagnostics, and draft pre-freeze dispositions.
- `pattern_calibration_sample.jsonl` and `pattern_calibration_sample_manifest.json`: a
  deterministic, disease-diverse definition-calibration set for P01, the P02 family, P06, and
  decomposed P08. It is selected for semantic coverage and counterexample discovery, not frequency
  estimation.

These remain exploratory development artifacts. They do not use ACR, do not open final test, and do
not establish population prevalence, causality, appropriateness, or predictive usefulness.
