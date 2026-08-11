# Rubric Update — explaining clinical imaging orders

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

## 2. Guideline baseline

The old disease-specific trees were artificial compilations of heterogeneous sources. They should
not be treated as the guideline itself.

The new primary baseline is the **ACR Appropriateness Criteria**, which uses the same structure
across the four imaging problems:

```text
clinical variant X
    → imaging option Y
    → usually appropriate | may be appropriate | usually not appropriate
```

Relevant topics:

- ACR Right Lower Quadrant Pain;
- ACR Right Upper Quadrant Pain;
- ACR Left Lower Quadrant Pain;
- ACR Acute Pancreatitis.

ACR defines a **set of acceptable imaging actions**, not one mandatory path.

Disease-specific sources remain secondary references:

- WSES 2025: appendicitis;
- WSES 2020 and TG18: cholecystitis;
- WSES 2020: diverticulitis;
- ACG 2024 and Revised Atlanta: acute pancreatitis.

They define clinical states such as risk, diagnosis, severity, and complication. TG18 and Revised
Atlanta are not used as complete imaging policies.

---

## 3. Proposed explanatory state

ACR variants describe part of the clinical context, but do not fully explain a patient's longitudinal
imaging sequence. We test whether three latent variables provide the missing structure.

### Assumption `A_t`

What clinical frame is active before the order?

```text
suspected disease | confidence/risk | confirmed disease
suspected etiology | severity | complication | alternative diagnosis
```

Assumption is one part of the context. Age, pregnancy, resource availability, and contraindications
remain separate observed conditions.

### Question `Q_t`

What decision-relevant unknown is the image intended to answer?

```text
existence | etiology | severity | complication | alternative diagnosis
```

During discovery, the actual order may help annotate a reference question `Q*`. During held-out
evaluation, the target order cannot be used: `Q_t` must be inferred from pre-order information.

### Coverage `C_t(Q_t)`

How well has the current question already been answered?

```text
unanswered | partially answered | answered
```

Coverage depends on the question, available observations, prior image quality, and whether the
relevant anatomy or finding was actually assessed. `negative` is not the same as `inadequate`.

---

## 4. Imaging loop

```text
pre-order information I_t
        ↓
assumption A_t + open question Q_t + coverage C_t
        ↓
ACR-allowed imaging set under the observed context
        ↓
ordered image T_t
        ↓
new image/report O_{t+1}
        ├── updates coverage: was Q answered?
        └── updates assumption: did the clinical frame change?
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

## 5. What an apparent deviation means

A mismatch with the old tree is only a **rubric residual**. It is not automatically a guideline
deviation.

Under the new baseline, residuals are separated into:

1. **formalization loss:** the textual/ACR guideline allows the action but the executable rubric
   omitted the relevant condition;
2. **guideline-underdetermined:** several actions are acceptable or the situation is not covered;
3. **missing explanatory state:** assumption, question, or coverage explains the order;
4. **unsupported order:** the action remains difficult to justify from ex-ante information.

The goal is not to make every observed order correct. The model must preserve an unsupported floor.

---

## 6. Main hypothesis

> A temporally blinded representation of assumption, question, and coverage will explain and predict
> held-out imaging sequences better than ACR clinical variants alone, without relying on
> patient-specific exceptions or post-order information.

Compare:

| Model | State |
|---|---|
| `R0` | old deterministic disease tree |
| `R1` | ACR clinical variant and set-valued imaging policy |
| `R2` | `R1 + assumption` |
| `R3` | `R2 + question` |
| `R4` | `R3 + coverage` |

Evaluate on held-out patients:

- next-image prediction;
- ordered imaging sequence;
- repeat, switch, and stop prediction;
- coverage of observed orders by the guideline-allowed set;
- model complexity and residual audit.

Prediction supports the usefulness of the latent state; it does not prove that it uniquely recovers
the physician's true mental process.

---

## 7. Work plan

### Task 1 — Build the normative baseline

- [ ] Extract the four ACR topics into `clinical variant → appropriateness-rated image set` rules.
- [ ] Preserve population, prior image/result, timing, contraindication, and source provenance.
- [ ] Map each old-rubric edge to `source-supported | transformed | added | unsupported`.

### Task 2 — Build imaging trajectories

- [ ] Normalize image modality, body region, protocol, and timestamp.
- [ ] Construct patient-level sequences with all pre-order information.
- [ ] Represent repeat, modality switch, and imaging stop.

### Task 3 — Define and test `A/Q/C`

- [ ] Create the smallest reusable label set for assumption, question, and coverage.
- [ ] Annotate 10–20 trajectories first; revise labels until cases can be represented consistently.
- [ ] Separate retrospective reference `Q*` from pre-order predicted `Q_t`.

### Task 4 — Run the comparison

- [ ] Compare `R0–R4` with strict patient-level temporal blinding.
- [ ] Plot prediction gain against added state complexity.
- [ ] Audit remaining residuals rather than automatically absorbing them.

The immediate next task is **Task 1: extract the four ACR rule sets and compare them with the old
rubric**. This establishes a defensible baseline before building the latent reasoning model.
