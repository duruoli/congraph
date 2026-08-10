# Rubric Update — a minimal state model of clinical testing

## 1. Thesis

Clinical testing is a partially observed sequential decision process. The patient has an underlying
clinical state; tests generate imperfect observations; observations update both what the current
question has learned and what the physician believes; the remaining decision-relevant uncertainty
determines the next test.

A standard rubric conditions mainly on a disease label and a narrow pathway position. A physician
acts on a richer state. A **deviation** is therefore initially a residual produced by an insufficient
state representation—not automatically a physician error, and not automatically a reasonable action.

The project asks:

> What is the smallest clinically coherent state that makes the physician's next test, test order,
> and reasonable deviations derivable from information available before the order?

The proposed factorization of that state is **belief, open question, and question coverage**. This is
a testable model of clinical reasoning, not a claim that the physician's private mental process can
be uniquely recovered from behavior.

---

## 2. Core state

At step `t`:

- `I_t`: observations available before the next order—history, examination, laboratory values,
  prior tests, reports, and report quality.
- `A_t`: the physician's revisable belief state about the patient's latent clinical state.
- `Q_t`: the current decision-relevant unknown. Resolving it could change a clinical action.
- `C_t(Q_t)`: how much the available evidence has answered the current question.
- `T_t`: the ordered test.
- `O_{t+1}`: the observation produced by that test, including whether the result is interpretable
  and adequate for the question.

These objects must not be collapsed:

1. **Patient state** is how the patient actually is, whether observed or not.
2. **Observation state** is what evidence is currently available.
3. **Belief state** is what the physician infers from that evidence.

Coverage is not solely a patient fact or physician belief. It is a question-relative relation:

```text
C_t(Q) = coverage(observations available at t, question Q, test/report quality)
```

The same observation may adequately answer one question and be inadequate or irrelevant for
another.

---

## 3. One clinical loop

```text
observations I_t
      │
      ├──> belief update A_t         (what the physician believes)
      │
      └──> coverage update C_t(Q)   (what the current question has learned)
                       │
                 open question Q_t
                       │
             next-test policy π(A_t, Q_t, C_t, I_t)
                       │
                 ordered test T_t
                       │
              new observation O_{t+1}
                       └──> repeat
```

There is only one test-routing rule:

```text
T_t = π(A_t, Q_t, C_t, I_t)
```

A new observation has two distinct update effects:

```text
C_{t+1} = U_C(C_t, O_{t+1}, Q_t)   # Does the observation answer the question?
A_{t+1} = U_A(A_t, O_{t+1})        # Does it change the clinical belief?
```

The familiar special cases are consequences of this loop:

- **Advance:** the current question is adequately answered; close it and select the next
  decision-relevant question.
- **Inadequate / retry:** coverage remains insufficient; the question stays open and another test
  may be routed to the same information need.
- **Inconsistent / reroute:** the observation materially changes the belief state; update or replace
  the question and route a test under the revised belief.

These are transitions, not external alarm rules.

---

## 4. Representation constraints

### Belief state

The assumption library is a shared, finite vocabulary of clinically meaningful claims. A patient's
belief state instantiates claims from that library with evidence, confidence, and status:

```text
belief_claim := {
  scope: frame | disease | etiology | severity | complication | alternative,
  target: <clinical target>,
  confidence: <calibrated value or level>,
  status: active | supported | contradicted | ruled_out | unknown,
  evidence_refs: [<pre-order observation ids>]
}
```

Beliefs must be produced by a cross-patient update rule rather than freely invented per case:

```text
A_{t+1} = U_A(A_t, O_{t+1})
```

### Open question

A question is not any missing fact. It is an unresolved variable whose answer could change the next
clinical action. `existence`, `etiology`, `severity`, and `complication` are recurring purposes, not a
mandatory sequence.

### Guard against post-hoc explanation

Any state added to explain a deviation must be:

1. available before the order;
2. reproducible across patients;
3. low-dimensional and parsimonious;
4. clinically and normatively defensible.

The model must never use the target order or later documentation to reconstruct the pre-order belief
or question.

---

## 5. What a deviation can mean

A deviation from the standard rubric has four possible explanations:

1. **Missing state:** the rubric has the relevant question but lacks a guard such as prior-test
   adequacy or a belief condition.
2. **Missing question:** the physician is pursuing a decision-relevant unknown absent from the
   rubric.
3. **Missing belief representation:** the question or route depends on a clinical frame or
   alternative the rubric cannot express.
4. **Unsupported action:** no ex-ante, parsimonious, clinically defensible completion makes the test
   reasonable.

The aim is to separate rubric incompleteness (1–3) from unsupported testing (4), not to explain every
observed action as correct. A useful updated rubric should retain a non-zero error floor and an
over-testing brake.

---

## 6. Falsifiable claim and evaluation

> Compared with a disease-and-pathway rubric, a temporally blinded model of belief, open question,
> and question coverage will more accurately predict the next test, test order, and reasonable
> deviations in held-out patients, beyond improvement attributable merely to added model capacity.

Evaluation must be patient-level and strictly ex-ante. At each step the model sees only information
available before the order. Report:

- next-test prediction;
- ordered-test sequence prediction;
- prediction and classification of deviations;
- question-closing versus retry/reroute transitions;
- calibration and error analysis;
- performance relative to equally flexible baselines;
- complexity of the added state.

Prediction supports the proposed latent model but does not uniquely identify the physician's true
mental process. Stronger evidence comes from counterfactual transition tests: changing an adequate
result to inadequate should preserve the question and induce retry; changing a consistent result to
inconsistent should update belief and may induce rerouting.

The central empirical object is the **derivability curve**:

| Model state | Added information | Expected residual deviation |
|---|---|---:|
| `S0` | disease + standard pathway | baseline |
| `S1` | + open question | lower |
| `S2` | + question coverage / adequacy | lower |
| `S3` | + belief state | lower |
| floor | unsupported or irreducibly ambiguous actions | non-zero |

Each state increment must be declared before evaluation. This tests whether the proposed structure
explains behavior rather than merely memorizing exceptions.

---

## 7. Minimal next experiment

Use one disease cohort and reconstruct step-level trajectories:

```text
(I_t, A_t, Q_t, C_t, T_t, O_{t+1})
```

Then:

1. enforce temporal blinding of `A_t`, `Q_t`, and `C_t`;
2. compare `S0–S3` on held-out patients;
3. measure next-test, ordering, and deviation prediction;
4. audit which residuals are missing state, missing question, missing belief, or unsupported action;
5. test whether the remaining errors concentrate in test timing and question priority.

The project succeeds if a small shared state vocabulary produces reproducible gains and a clinically
meaningful residual floor. It fails if gains require patient-specific assumptions, post-order
information, or an ever-expanding exception list.
