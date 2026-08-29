# Empirical A/Q/C development

Status: **preliminary Track B prototype; formal full-corpus discovery not complete** (2026-08-14).

## Boundary

This directory is derived only from the existing A/Q/C-free, schema-light, order-aware patient
annotations.
The completed ACR v1.1 corpus remains the independent normative representation `N` and is neither
loaded nor used as an A/Q/C template.

The original annotations under `results/annotation_experiment/` are unchanged. Generated files here
copy assumption-bearing fields verbatim and add a separate provisional coding layer.

## Scope correction

The canonical Mode-A corpus is `results/annotation_experiment/full/*.json`: 293 patient trajectories
and 542 decision steps (appendicitis 71, cholecystitis 90, diverticulitis 34, pancreatitis 98).

The current prototype initially searched only the 8 root pilot files plus 30 `batch/` files: 38
trajectories and 81 steps. All 38 patients occur in `full/`, but five hand-picked pilot files use a
different ensemble representative than their `full/` n=1 version: 23202997, 29573603, 26371704,
27675389, and 21282967. Therefore the current 16-trajectory open coding and its counts are valid only
as a preliminary pilot-derived analysis. They are not the formal full-corpus discovery sample.

## Preliminary work completed

- Audited a 38-trajectory pilot+batch subset and subsequently identified the 293-trajectory canonical
  `full/` corpus that must replace it for formal sampling.
- Fixed a maximum-variation development sample of 16 trajectories (4 per disease) and 48 steps.
- Open-coded the verbatim differential, other hypothesis, information gap, expected finding, and
  reasoning fields.
- Produced a preliminary assumption-type codebook with explicit `other` and `unclear` residuals.
- Drafted two schema-identical pilot prompts: direct masked-trajectory A/Q/C and recoding of the old
  open reasoning.

Open-code assignment counts are descriptive of a purposive, multi-label sample and are not
prevalence estimates.

The current prototype extracted only **text-level assumption-type clusters**. It did not yet split
every step into atomic assumption propositions or assign proposition-specific status. It also did
not empirically induce a question codebook, identify the answer requirements that define what would
resolve each question, annotate requirement-level coverage, or produce patient-level A/Q/C
trajectories. The `QUESTION_TYPES` and scalar `pre_order_coverage` currently present in
`experiments/aqc/prompts.py` are seed hypotheses copied from the design logic, not findings from open
coding, and must be revised after the full-corpus discovery audit.

The current conceptual contract is:

```text
A frames the problem
Q specifies the decision-relevant unknown and its answer requirements
C records which requirements all causally available evidence has addressed
```

`C` is a time-indexed coverage profile, not the update mechanism. Study adequacy, test–question
capability, result status, and aggregate coverage remain separate judgments.

## Files

- `discovery_sample_manifest.json` — fixed trajectory selection, source paths, and diversity counts.
- `discovery_open_coding.jsonl` — one decision step per line; source wording is copied verbatim and
  `open_type_codes` are explicitly provisional.
- `provisional_assumption_codebook.json` — definitions, inclusion/exclusion rules, proposition-level
  statuses, and `other/unclear`.
- `open_coding_memos.md` — clustering decisions, boundary cases, and freeze criteria.
- `../../experiments/aqc/prompts.py` — paired direct/recode pilot prompts and shared JSON contract.
- `../../scripts/build_aqc_discovery_sample.py` — deterministic rebuild.
- `../../scripts/validate_aqc_development.py` — source-integrity, enum, and causal-leakage checks.

## What the source annotation already contains

Mode-A annotation was structured from its first committed version. Each step contains a constrained
`differential`, free-text `other_hypothesis`, `information_gap`, `expected_finding`, categorical
`action_role`, appropriateness, grounding, and prose `reasoning`. It was rubric-free and A/Q/C-free,
but not form-free. Earlier next-test LLM experiments also returned `{next_test, reasoning}` rather
than unconstrained physician-order narratives. No intentionally fully unstructured annotation pass
was found in the repository or its git history.

Consequently, the current work can explore the contents, hierarchy, and status of assumptions and
can normalize question language, but it cannot use the mere presence of a differential or
information gap as proof that A/Q/C emerged without prompting. See `../../aqc_annotation_design.md`
§1.1 and §7.1.

## Key design decisions

An assumption is an atomic proposition, not one label for a whole order. A trajectory can therefore
retain `pancreatitis is established` while separately storing `biliary etiology is suspected` and
`necrosis is excluded`. Each proposition receives its own type, level, and status.

Trajectory-level annotation must be sequential. Putting every per-step masked view into one model
call would expose an early order's result inside a later view. The pilot instead carries the prior
A/Q/C state forward one decision point at a time, while each call sees only information available
before its current order.

The direct and recode arms share one output contract. Neither arm sees the current result,
verification label, later events, disease-conditioned rubric, or ACR. The direct arm does not see
the old reasoning; the recode arm sees only the old ex-ante reconstruction fields.

## Formal full-corpus codebook-development and final-test plan

Do not simply draw another convenient set of complex cases. Use the following patient-level,
reproducible design before checking or batch-applying the codebook:

1. **Define the eligible source.** Read annotations only from `full/*.json`; attach real timing roles
   from `full/timing_roles.csv`. Use diagnostic steps as the primary domain. Keep post-intervention
   steps as a named secondary stratum so device/intervention assumptions can be studied without
   silently changing the main diagnostic target.
2. **Split before qualitative selection.** Make a stable patient-level development/final-test split,
   approximately 80/20, within each disease. Stratifying within disease preserves all four
   prespecified domains and their approximate corpus proportions in both partitions; it does not
   create disease-specific A/Q/C ontologies, and the disease label is not shown as an annotation
   answer. Do not use ACR ratings, rubric deviation, final diagnosis correctness, verification, or
   post-order outcomes to create the split. Keep the final-test partition unopened until the
   codebook, annotation procedure, pattern definitions, and models are frozen.
3. **Initial codebook sample.** From the development partition select about 24 trajectories total
   (approximately six per disease, not 24 per disease) by maximum variation over trajectory length,
   repeat/switch pattern, modality sequence, old action role, prior-study limitation, differential
   concentration/`other`, and timing stratum. Combine a reproducible stratified/random component
   with explicit supplementation of rare but prespecified structures. This sample develops the
   annotation vocabulary; it is not used to estimate population prevalence or final pattern counts.
4. **Two coding views.** First open-code prose `reasoning` with field names such as `differential`,
   `information_gap`, and `action_role` hidden. Then code the complete schema-light ex-ante record.
   Compare which assumption and question types survive the blind view, which answer requirements
   recur, and which appear tied to the old scaffold.
5. **Iterate to qualitative saturation inside development.** Twenty-four is only the first batch.
   If fresh cases introduce new top-level assumption/question types, recurrent answer-requirement
   dimensions, or systematic `other/unclear` clusters, add a new non-overlapping development batch
   (for example 8–16 trajectories), revise, and check again on another fresh batch. Record every
   addition and why it was needed. A defensible freeze requires fresh batches to stop producing
   material schema changes across diseases and major timing/sequence strata.
6. **Independent framework check.** Select approximately 16 non-overlapping trajectories from the
   still-unused development pool (about four per disease) with the same variation audit. For each
   case, compare two independent ways of constructing A/Q/C: (a) annotate from the causally masked
   chart plus the actual order without seeing the old reconstruction; and (b) convert the old
   ex-ante reconstruction into A/Q/C without seeing the current result or later events. Neither
   method sees ACR, deviation labels, or verification. If the two methods agree closely on A and Q,
   the old reconstruction may be reused as the main bulk source for those fields, with independent
   re-annotation concentrated on disagreements and weakly supported cases. Requirement-level C
   still requires the causally available patient evidence and cannot be recovered reliably from the
   old reasoning text alone.
7. **Freeze criteria.** Compare atomic-proposition compatibility, type/status agreement, question
   target/type, answer requirements, positive/negative consequence, requirement-level coverage,
   `other/unclear`, unsupported residuals, and over-rationalization. Revise when necessary and check
   the revision on another fresh development batch. Agreement tests dependence on the old annotation
   scaffold; agreement alone does not prove recovery of physician intent or clinical correctness.
8. **Discover patterns only after freezing.** Apply the frozen A/Q/C contract to the larger
   development partition, preserving the original annotations, and use this larger set—not the
   initial 24 cases—to discover and estimate recurrent transitions and other empirical patterns.
9. **Final validation.** Open the approximately 20% final-test partition only after all development
   choices are fixed. Test whether the codebook covers new cases, whether development patterns
   replicate, and whether pre-order A/Q/C improves held-out next-image/repeat/switch/stop prediction.
   If final-test cases are used to revise the codebook, that test is spent and a new untouched test
   set is required.

### What “representative” means here

The initial qualitative sample is structurally diverse, not statistically representative of pattern
prevalence. Maximum-variation selection reduces the chance of missing rare decision structures, but
cannot guarantee completeness. Completeness is assessed iteratively through fresh-case saturation.
Frequencies, disease comparisons, and claims that a pattern is recurrent must come from the larger
development corpus and then replicate in the untouched final test.

The observation vocabulary in `results/vocab` is a nested EvidencePiece schema inside patient
Context, not the assumption ontology. It should ground propositions and coverage and later support
an explicit patient-context-to-ACR-context bridge; it must not define assumption types.

## Next work

Implement the full-corpus development/final-test split and initial development-sample manifests
above, redo the open coding, and revise the A/Q/C output contract to include
`answer_requirements[]` under Q and requirement-level coverage entries under C. Expand the
codebook-development sample if fresh cases show that the first 24 have not reached saturation. Do
not open the final test, freeze the codebook, or batch-annotate from the current 16-trajectory
prototype.

```bash
/opt/anaconda3/bin/python3.12 scripts/build_aqc_discovery_sample.py
/opt/anaconda3/bin/python3.12 scripts/validate_aqc_development.py
```
