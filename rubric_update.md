# Rubric Update — extracting normative and empirical imaging-decision knowledge

## 1. Project scope

This project explains **which image is ordered next, in what order, and why imaging is repeated,
switched, or stopped** for:

- appendicitis;
- cholecystitis;
- diverticulitis;
- acute pancreatitis.

The model may use all information available before an order—symptoms, examination, labs, diagnoses,
prior images, reports, and time course—but its predicted actions are limited to imaging:

```text
US | CT | MRI/MRCP | HIDA | X-ray | repeat | switch | no further imaging
```

We are not trying to model every lab, treatment, or the physician's private mental process.

---

## 2. Normative source: extract ACR before imposing a schema

The old disease-specific trees were artificial compilations of heterogeneous sources. Their source
selection and executable translation were neither complete nor sufficiently faithful. They are
historical artifacts, not a standard rubric, guideline baseline, or model to preserve in the new
study.

The primary normative source is the **ACR Appropriateness Criteria**. Before deciding how to
formalize it, we will extract every relevant clinical variant and imaging option from the original
text. A provisional surface form is:

```text
clinical variant X
    → imaging option Y
    → usually appropriate | may be appropriate | usually not appropriate
```

This surface form is an observation to test, not a schema to assume. For every rule, extraction will
preserve at least:

```text
source topic + variant ID + original variant text
context: presentation/symptoms | suspected or known condition | population
         timing | severity/complication | prior imaging and result | constraints
action: modality | body region/protocol | initial/next/repeat/interventional role
rating: appropriateness category and numeric rating
evidence strength: stored separately from appropriateness
rationale and exact source provenance
```

Only after this faithful extraction will recurring context and action types be induced from the ACR
corpus. `Assumption / Question / Coverage` must not be used as a compulsory extraction template.

Relevant topics:

- ACR Right Lower Quadrant Pain;
- ACR Right Upper Quadrant Pain;
- ACR Left Lower Quadrant Pain;
- ACR Acute Pancreatitis.

ACR primarily supports the choice of imaging or image-guided intervention for a stated clinical
scenario. It usually defines a **set of rated actions**, not one mandatory diagnostic path and not a
complete disease-management guideline.

Disease-specific sources remain secondary references:

- WSES 2025: appendicitis;
- WSES 2020 and TG18: cholecystitis;
- WSES 2020: diverticulitis;
- ACG 2024 and Revised Atlanta: acute pancreatitis.

They define clinical states such as risk, diagnosis, severity, and complication. They are retained
with separate provenance and are not silently merged into ACR or converted into a single composite
tree. TG18 and Revised Atlanta are not used as complete imaging policies.

---

## 3. Parallel empirical source: order-aware reconstruction

The empirical knowledge source is different from ACR. The existing annotation pipeline was designed
as open, order-aware abductive reconstruction:

```text
pre-order patient record + physician's actual ordered test
    → a plausible reconstruction of what the physician may have been trying to establish
```

The actual order was deliberately shown. The annotation task was not next-test prediction: the order
served as evidence for recovering possible `when / how / why` reasoning while reducing the tendency
of an LLM to substitute its own preferred test. The order result and later information were hidden
from the ex-ante reconstruction and used separately for local verification.

These annotations were rubric-free and were **not originally generated under an A/Q/C ontology**.
They contain fields suggested during the earlier annotation work—such as differential, information
gap, expected finding, action role, and grounding—but A/Q/C was induced only later by comparing the
reconstructed traces. The traces therefore remain a discovery corpus rather than pre-existing A/Q/C
labels.

The reconstruction is not a direct observation of the physician's private mental state. It is a
plausible account constrained by the chart and the observed action, and it may contain knowledge
contributed by the annotating model. Claims should therefore concern recurrent explanatory patterns,
not unique recovery of true intentions.

## 4. Candidate explicit representation: `A/Q/C`

`A/Q/C` is a candidate schema for decoding recurrent knowledge in the empirical reconstructions. It
is **parallel to**, not derived from, the schema induced from ACR. The two representations will first
be developed independently and then compared for overlap, omissions, and conflicts.

The operational definitions, transition logic, and proposed two-pass annotation procedure are kept
in [`aqc_annotation_design.md`](aqc_annotation_design.md). In brief, A/Q/C is the core state;
question-relative study adequacy and assumption–evidence discordance are relations that drive its
cross-step updates rather than additional free-standing state variables.

### Assumption `A_t`

What clinical frame is active before the order?

```text
suspected disease | confidence/risk | confirmed disease
suspected etiology | severity | complication | alternative diagnosis
```

Assumption is one part of the context. Age, pregnancy, resource availability, and contraindications
remain separate observed conditions.

### Question `Q_t`

What decision-relevant unknown must be resolved under the current assumption, and what dimensions
would count as an answer?

```text
existence | etiology | severity | complication | alternative diagnosis
```

During discovery, the actual order may help annotate a reference question `Q*`. Store its target,
type, answer requirements, and decision consequence. During held-out evaluation, the target order
cannot be used: `Q_t` and its requirements must be inferred from pre-order information.

### Coverage `C_t(Q_t)`

Which answer requirements of the current question have all available evidence addressed, and which
remain open?

```text
per requirement: unaddressed | partially addressed | sufficiently addressed
optional summary: unanswered | partially answered | sufficiently answered
```

Coverage depends on the question, available observations, prior image quality, and whether the
relevant anatomy or finding was actually assessed. `C_t` is a time-indexed state/profile, not the
tracking mechanism itself. `negative` is not the same as `inadequate`.

---

## 5. Imaging loop represented by A/Q/C

```text
pre-order information I_t
        ↓
assumption A_t + open question Q_t + coverage C_t
        ↓
candidate actions under observed context and normative guidance
        ↓
ordered image T_t
        ↓
new image/report O_{t+1}
        ├── updates requirement-level coverage C
        └── supports or challenges the current assumption A
                         ↓
                  reassess A and Q
                         ↓
                  remap guideline Context
```

This produces three common transitions:

- **close/advance:** the question is answered; stop imaging or move to another question;
- **retry/switch:** the question remains open because the prior image was inadequate;
- **reroute:** the result changes the assumption and opens a different question.

The prediction target is:

```text
P(next image or stop | I_t, A_t, Q_t, C_t, ACR context)
```

---

## 6. What a mismatch means

A mismatch between an observed order and an extracted ACR rule set is not automatically a guideline
deviation.

Under the new baseline, residuals are separated into:

1. **extraction/formalization loss:** the ACR text allows the action but the structured
   representation omitted or distorted the relevant condition;
2. **guideline-underdetermined:** several actions are acceptable or the situation is not covered;
3. **missing explanatory state:** assumption, question, or coverage explains the order;
4. **unsupported order:** the action remains difficult to justify from ex-ante information.

The goal is not to make every observed order correct. The model must preserve an unsupported floor.

---

## 7. Research aims and main hypothesis

> A temporally blinded representation of assumption, question, and coverage will explain and predict
> held-out imaging sequences better than ACR clinical variants alone, without relying on
> patient-specific exceptions or post-order information.

Compare independently constructed representations:

| Representation | Source and state |
|---|---|
| `N` | faithfully extracted ACR context–action–rating rules |
| `E` | open order-aware physician-reasoning reconstructions |
| `AQC` | explicit A/Q/C coding induced from `E` |
| `N + AQC` | normative rules plus the candidate empirical explanatory state |

Evaluate on held-out patients:

- next-image prediction;
- ordered imaging sequence;
- repeat, switch, and stop prediction;
- coverage of observed orders by the guideline-allowed set;
- model complexity and residual audit.

Prediction supports the usefulness of the latent state; it does not prove that it uniquely recovers
the physician's true mental process.

The project has two sequential aims:

1. **Validate the representation.** Infer A/Q/C using only pre-order information and test whether it
   adds held-out predictive value for the next image, repeat/switch/stop, and state transition beyond
   patient evidence and ACR Contexts alone.
2. **Discover missing knowledge.** Mine recurrent A/Q/C transitions, coverage gaps, compensatory
   actions, and Context remappings that are absent or under-specified in ACR. Treat them as candidate
   empirical knowledge until they replicate and pass clinician or outcome-based validation.

---

## 8. Work plan

### Task 1 — Extract the normative ACR corpus

- [x] Extract every relevant ACR variant and action without using A/Q/C as the template.
- [x] Preserve original text, population, presentation, prior image/result, timing, severity,
      constraints, appropriateness rating, evidence strength, rationale, and provenance.
- [x] Induce a context/action vocabulary from the extracted corpus and document ambiguous cases.
- [x] Validate a sample against the source text before using the structured corpus as a baseline.

Track A was completed on 2026-08-13. The operational ACR schema is `data/acr_normative` v1.1:

```text
Context = clinical_state + imaging_history + modifiers + decision_stage
Action  = exact procedure + normalized action components
Rank    = final_rating (1-9; higher is more appropriate; ties retained)
```

Appropriateness category, SOE, median, vote distribution, rationale, and provenance remain
available but are not alternative ranking metrics. See `data/acr_normative/README.md`.

### Task 2 — Build imaging trajectories

- [ ] Normalize image modality, body region, protocol, and timestamp.
- [ ] Construct patient-level sequences with all pre-order information.
- [ ] Represent repeat, modality switch, and imaging stop.

### Task 3 — Recode and test `A/Q/C`

- [x] Preserve the existing open annotations unchanged as the discovery corpus.
- [ ] Create the smallest reusable codebook for assumption, question, and coverage. A preliminary
      38-trajectory-subset prototype exists in `data/aqc_development`, but formal discovery must be
      redone from the 293-trajectory `results/annotation_experiment/full` corpus.
- [ ] Induce question-specific answer requirements from the discovery corpus; do not treat the
      current seed `QUESTION_TYPES` or a scalar coverage label as frozen findings.
- [ ] On 10–20 trajectories, run paired structured passes: (a) recode the existing open
      reconstruction into A/Q/C, and (b) independently annotate A/Q/C from the pre-order record plus
      actual order without showing the old reconstruction. Compare them to detect framing dependence
      before revising and freezing the codebook.
- [ ] After freezing the codebook, recode the remaining corpus rather than replacing the original
      annotations.
- [ ] Separate retrospective, order-aware reference labels `(A*, Q*, C*)` from pre-order-only
      inferred labels `(A_t, Q_t, C_t)` used at evaluation time.

### Task 4 — Compare the two knowledge structures

- [ ] Compare the independently induced ACR and A/Q/C vocabularies: overlap, missing dimensions,
      contradictions, and disease/topic coverage.
- [ ] Test `N`, pre-order `AQC`, and `N + AQC` with strict patient-level temporal blinding.
- [ ] Plot explanatory/predictive gain against added state complexity.
- [ ] Audit remaining residuals rather than automatically absorbing them.

### Task 5 — Discover and validate candidate transition knowledge

- [ ] Mine recurrent `(A_t, Q_t, C_t) -> result -> (A_{t+1}, Q_{t+1}, C_{t+1})` motifs.
- [ ] Identify which motifs are absent, partial, or ambiguous in the extracted ACR Contexts.
- [ ] Replicate candidate motifs in held-out patients and audit for institutional workflow artifacts.
- [ ] Review clinically important candidates with physicians; require external or outcome evidence
      before interpreting recurrent practice as normative knowledge.

The immediate next task is **Task 3: create patient-level discovery/held-out splits from the full
Mode-A corpus**. In the discovery partition, open-code assumptions, questions, and the dimensions
that would count as answering each question. Then revise the prompt so Q contains explicit answer
requirements and C records requirement-level coverage. Run the paired 10–20-trajectory A/Q/C pilot,
audit over-rationalization, and freeze the codebook before batch annotation. Treat the completed ACR
corpus as an independent normative input `N`; do not modify it to fit empirical A/Q/C annotations.
