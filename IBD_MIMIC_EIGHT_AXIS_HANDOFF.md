# IBD MIMIC-IV: Eight-Axis Annotation Handoff

## Goal

Represent IBD patient steps with the controlled vocabulary in `unified_axes_v1.yml` so they can be compared with literature passages. Shared labels mean **topic overlap**, not that a paper's claim applies to a patient.

The example Nigel JATS file has 50 article-level `axis:value` assignments, each linked to supporting paragraphs through `evidence=...`. YAML defines the possible values; JATS selects values and evidence for this article. Do not copy every article label onto every paragraph.

Key sources: [axis vocabulary](unified_axes_v1.yml), [annotated Nigel JATS](nigel_vertex-track-a-universe-2026-07-09_AGA_CGH_13Issue_IBDReview_Changing_Global_Epidemiology_of_Inflammatory_Bowel_Dis_corpus.labeled.current.jats.xml), and [the project's missing-middle concept note](IDEA_guideline_missing_middle.md).

## The eight axes: one question each

| Axis | Intuition | Example YAML values |
| --- | --- | --- |
| `disease` | What disease or clinical problem is in focus? | `ibd_crohns`, `ibd_ulcerative_colitis` |
| `management_domain` | What kind of decision is at hand? | `diagnosis_workup`, `therapy_selection`, `monitoring` |
| `intervention_class` | What kind of action or treatment is involved? | `corticosteroid`, `biologic`, `surgical` |
| `diagnostic_modality` | How is information obtained or disease characterized? This includes monitoring, not only initial diagnosis. | `colonoscopy`, `cross_sectional_imaging_ct`, `biomarker_assay` |
| `clinical_phase` | Where is the case in its disease or treatment trajectory? | `pre_diagnosis`, `maintenance`, `complication_acute` |
| `population` | Which patient subgroup matters? | `adults`, `older_adults`, `immunocompromised` |
| `setting` | In what care environment? | `hospitalized_inpatient`, `emergency_acute`, `outpatient` |
| `anatomy` | Which organ or region? This is a secondary locator; disease alone may not determine extent. | `small_bowel`, `colon`, `rectum` |

`management_domain` is the **decision type**; `clinical_phase` is the **trajectory position**. `diagnostic_modality` obtains information; `intervention_class` changes care. The axes are not a factual-versus-judgment split: a test may be recorded, while its purpose requires inference. A/Q/C's *Q* is more specific than `management_domain`: “diagnosis workup” versus “is there an abscess?”

## Work completed

- `MIMIC_DISEASE_EXTRACTION_GUIDE.md` describes the local extract at `data/raw_data/ibd_mimiciv_3_1/`. Its ICD cohort label is retrospective; it does not date clinical recognition of IBD.
- `scripts/build_ibd_event_timelines.py` builds a patient-level event ledger across IBD-coded admissions, retaining source references, event time, availability time, and precision. It groups available results into **availability checkpoints**, which are candidate step anchors rather than observed clinician decisions.
- Two disjoint pilots are in Git-ignored `runs/`: `ibd_event_timeline_pilot_20/` (20 patients, 57 admissions, 13,097 events, 1,675 checkpoints) and `ibd_event_timeline_pilot_100_additional/` (100 more patients, 265 admissions, 63,093 events, 8,418 checkpoints). Each has `manifest.json`, `events.jsonl`, `availability_checkpoints.jsonl`, `timeline_preview.md`, and `validation_report.json`.
- Both pilots pass row coverage, ownership, ordering, and checkpoint-link checks; 60 event times per batch were independently checked against source Parquet. **No eight-axis or A/Q/C labels exist yet.**

## Temporal and epistemic rules for annotation

- Label what was **supported at time `t`**. Do not backfill early steps with discharge summaries or retrospective ICD diagnoses. Distinguish documented history, working hypothesis, prescription, and performed action. Unknown is neither `n_a` nor `all`.
- Lab `charttime` usually precedes result availability (`storetime`); radiology `storetime` approximates report completion. Microbiology `storetime` can be its last update. Prescription `starttime` does not prove administration or exact order time.
- Procedures have a date, represented as a 24-hour interval with unknown within-day order. One prescription in the 100-patient pilot is untimed. The extract omits non-IBD-coded admissions and outpatient history; its first IBD-coded admission need not be the patient's first diagnosis.

## Next task

The subsequent pre-action candidate extraction is documented in
`IBD_DECISION_STEP_PILOT_20.md`. It places recorded target actions in a separate
outcome file and does not yet establish actual order times or decision questions.

1. Review pre-action candidates against source records to decide which are recoverable clinical decisions, determine each decision question, and record timing uncertainty. Add a separate sampling policy for possible no-action/stop opportunities.
2. Define an eight-axis annotation schema using YAML-approved values plus evidence IDs, availability time, assertion status, and uncertainty. Distinguish patient state and decision focus from the separately stored observed action. Allow supported multi-values; leave unsupported axes unknown.
3. Label a diverse subset of the 20-patient decision-candidate pilot, audit disease/phase and action status against prior evidence, then refine rules before labeling the additional 100. Test literature retrieval afterward; keep A/Q/C as a separate reasoning layer.
