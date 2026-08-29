# A/Q/C Annotation Design

## 1. Goal

Build order-aware A/Q/C reference representations for knowledge discovery, then use them to develop
and test a separate pre-order inference model. During reference annotation, use the real imaging
order as a clue to reconstruct a plausible physician reasoning chain:

```text
previous question → previous test/result → updated assumption
                  → current question → current order
```

The **annotation task** is order-aware knowledge discovery, not next-test prediction. The current
test result and later events remain hidden. The reconstruction is a plausible explanation grounded
in the chart, not a claim that we uniquely recover the physician's private thoughts. In later
validation, the target order must also be hidden and A/Q/C must be inferred from pre-order evidence
alone before predicting the next image or stop.

### 1.1 Source layers and nested schemas

Keep four representations separate:

| Symbol | Representation | Source | Epistemic role |
|---|---|---|---|
| `O_t` | causally available patient observations | raw chart and prior reports | observed patient evidence |
| `E_t` | order-aware reasoning reconstruction | `results/annotation_experiment/full/*.json` | plausible, LLM-reconstructed explanation |
| `A/Q/C_t` | decoded decision state | induced/tested from `E_t` and grounded in `O_t` | candidate empirical representation |
| `N` | ACR Context–Action–Rating corpus | `data/acr_normative` | independent normative representation |

`results/vocab` standardizes observation content inside `O_t`. Its source chain is:

```text
data/raw_data
    -> scripts/fragment_evidence.py
    -> results/evidence_pieces/*.jsonl
    -> scripts/normalize_vocab.py
    -> results/vocab/{anatomy,attribute,state}_{vocab,map}.json
```

The atomic observation should be treated as more than `attribute + state`:

```text
EvidencePiece = anatomy + attribute + state/value + finding_status
                + source/provenance + time + qualifier
```

It is a **small schema repeated inside entries of a larger patient-context schema**:

```text
PatientContext_t
├── clinical_observations: EvidencePiece[]
├── imaging_history: Study[]
│   ├── modality + region + protocol + order/result time
│   ├── adequacy/limitations
│   └── findings: EvidencePiece[]
├── modifiers_and_constraints
└── decision_stage
```

Here `decision_stage` means the procedural position of the imaging decision—for example initial,
next, repeat, or post-intervention—not the biological stage of the disease. It is an operational
Context field, not a fourth epistemic state and not one of the two primary ambiguity sources.

Thus the vocabulary is not a complete Context ontology and is not itself ACR Context. It provides
the standardized observation language for parts of `clinical_observations`, prior imaging results,
and some modifiers. ACR `condition`, severity/complication state, and decision stage may require
higher-level aggregation or inference and must not be equated directly with an evidence piece.

The same observation layer supports A/Q/C without defining its latent ontology:

```text
O_t grounds A_t and Q_t
O_t × Q_t supports coverage C_t(Q_t)
prior Study metadata + findings support adequacy/capability/result-status judgments
O_t can later be matched to N.Context through an explicit bridge
```

Raw text remains authoritative; canonical evidence pieces are indexes for comparison and coverage.
Do not load an admission-wide `results/evidence_pieces` row directly into a decision point: future
reports would leak, and some labs/microbiology are currently admission-global. Construct `O_t` from
the causally masked record and retain exact source quotes.

## 2. Core state: A/Q/C

### Assumption `A_t`

The proposition currently organizing the workup, for example:

```text
acute pancreatitis is present
the pancreatitis has a biliary cause
a pancreatic collection is infected
the pain may come from another organ
```

Store two separate properties:

- **type/level:** disease existence, etiology, severity, complication, alternative source, or other;
- **status:** suspected, likely, established, challenged, or excluded.

Assumptions are hierarchical. Pancreatitis may be established while its biliary etiology remains
only suspected.

### Question `Q_t`

The main decision-relevant unknown that the current order appears intended to answer, together with
the dimensions that would count as answering it:

```text
Does appendicitis exist?
Is there a CBD stone?
Has necrosis or an abscess developed?
Is another disease causing the symptoms?
```

Record its target, type, what a positive or negative answer would change, and its **answer
requirements**. An answer requirement is a question-specific dimension that available evidence
must address; it is not a preferred modality. For example, a question about CBD obstruction may
require adequate assessment of the duct plus evidence about dilation, stone, or another obstructing
process. The question should not merely restate the modality. If an order serves several purposes,
identify one primary question and optional secondary questions.

For the minimal three-state framework, answer requirements are stored inside `Q_t` rather than as a
fourth state variable:

```text
Q_t = decision-relevant unknown + answer requirements + decision consequence
```

### Coverage `C_t(Q_t)`

The current profile of how well **all evidence available before the current order** addresses the
answer requirements of that question. Each requirement should be marked separately:

```text
unaddressed | partially addressed | sufficiently addressed
```

For each requirement, record the supporting evidence and whether its direction supports, refutes,
mixes, or gives no direction toward the question. An optional summary can compress the profile to:

```text
unanswered | partially answered | sufficiently answered
```

Coverage belongs to `(all current evidence, question requirements)`, not to one test. Several weak
observations may jointly answer a requirement; a technically excellent scan may contribute nothing
to a requirement outside its scope. `C_t` is the state being tracked. The rule that incorporates new
evidence and produces `C_{t+1}` is a separate update mechanism.

The minimal relation is:

```text
A frames Q -> Q specifies what must be covered -> C records what has been covered
```

## 3. Keep four judgments separate

### Study adequacy

Was the completed examination/report technically usable for what it attempted to examine?

```text
diagnostic | limited but diagnostic | nondiagnostic | unknown
```

Examples of limitations include motion, bowel gas, body habitus, nonvisualization of the intended
target, incomplete acquisition/protocol, or incomplete documentation.

### Test–question capability

Could that modality, body region, and protocol address the answer requirements of this particular
question in principle?

```text
capable | partially capable | not capable | uncertain
```

A diagnostic pelvic ultrasound may answer an ovarian question but provide no answer about the
appendix. The study itself is not inadequate; its scope does not cover the second question.

### Result status

What answer did the report give for the question?

```text
positive | negative | indeterminate | not assessed
```

`Negative` is a normal and potentially valuable answer: the relevant abnormality was adequately
sought and not found. It is different from `indeterminate` (examined but unresolved) and
`not assessed` (outside the examination/report). Nonvisualization is not automatically negative.

### Aggregate coverage

Coverage combines study quality, requirement-specific test capability, result status, clinical
findings, labs, other images, and prior plausibility:

```text
diagnostic study + valid negative + sufficient total evidence → question may close toward refutation
nondiagnostic study or target not visualized              → question usually remains open
diagnostic study aimed at another question                → little or no coverage of this question
```

## 4. Discordance

Discordance asks whether important evidence is materially inconsistent with the current assumption.
It is a relation between evidence and `A`, not another state variable.

```text
concordant | materially discordant | indeterminate | not applicable
```

Flag discordance only when:

1. two evidence streams can be quoted;
2. the conflict is clinically important, not ordinary noise;
3. the current assumption cannot comfortably explain both;
4. the conflict could change the next question or action.

Possible pairs include clinical–laboratory, clinical–imaging, laboratory–imaging, two images, or an
unexpected temporal change.

## 5. Cross-step transitions

First annotate changes in assumptions and questions; then derive an intuitive transition label.

### Assumption change

```text
establish | retain | refine | challenge | exclude | replace
```

Identify which proposition changed and at what level. A disease assumption can remain stable while
an etiologic assumption changes.

### Question continuity

```text
initial | same | refined | new | reopened
```

### Derived patterns

| Pattern | Meaning |
|---|---|
| Remedy | The previous study was insufficient, so the same question remains open and the physician repeats or switches imaging. |
| Adjudicate | Evidence conflicts with the assumption, so the next test explores or resolves the conflict. |
| Advance | The previous question is answered and the relevant higher-level assumption remains; a downstream question about cause, severity, or complication follows. |
| Reroute | The focal assumption is challenged, excluded, or replaced, so the workup moves under another clinical frame. |
| Reopen | Time or clinical change makes a previously answered question uncertain again. |
| Close | The question is answered and no consequential imaging question remains. |

`Advance` usually preserves the disease-level assumption. `Reroute` can occur at any level. For
example, a negative MRCP may reroute the assumed etiology without overturning established
pancreatitis.

## 6. Temporal logic

At current decision point `t`:

```text
(A_{t-1}, Q_{t-1}, C_{t-1}) → prior test Y_{t-1} → prior result O_t
                                                        ↓
                     assess adequacy, requirement-specific capability,
                     result status, and evidence–assumption discordance
                                                        ↓
                         update C_t → reassess A_t and Q_t
                                                        ↓
                              explain current observed order Y_t
```

The result of current order `Y_t` is not used until the next decision point.

ACR is not shown during this empirical annotation. Guideline Context mapping is a downstream,
independent comparison:

```text
(O_t, A_t, Q_t, C_t) × N.Contexts -> exact | partial | multiple | uncertain | out_of_scope
```

## 7. Assumption ontology

Do not lock assumption types from intuition alone:

1. open-code assumption propositions in a diverse sample of the existing A/Q/C-free,
   schema-light annotations;
2. cluster recurring types and levels across all four diseases;
3. keep the original proposition plus a normalized type and an `other/unclear` option;
4. freeze the codebook before large-scale annotation;
5. audit residual cases for missing types.

The seed types in this document are hypotheses to test against the raw annotations, not a closed
ontology.

### 7.1 What Track B is intended to recover

The existing Mode-A corpus is not form-free text. It was generated as strict JSON with a constrained
five-branch `differential`, free-text `other_hypothesis`, `information_gap`, `expected_finding`, a
categorical `action_role`, appropriateness, grounding, and a prose `reasoning` field. It is better
described as **A/Q/C-free, schema-light reconstruction**. In particular, `differential` is already an
assumption proxy and `information_gap` is already a question proxy. Their existence cannot by
itself prove that A/Q/C emerged without framing.

Track B should therefore recover and test four specific products:

1. **Assumption propositions:** atomic propositions, their hierarchy/level, recurring empirical
   types, proposition-specific status, and `other/unclear` residuals.
2. **Question representation:** recurring target/type, one primary versus optional secondary
   questions, answer requirements, and what positive or negative answers would change. This is a
   normalization and audit of free-text gaps, not merely renaming `information_gap` to `Q`.
3. **Coverage rules:** a requirement-level profile of how all causally available observations
   answer a question, separating study adequacy, test–question capability, result status, and the
   optional aggregate summary. Coverage cannot be extracted from the old reasoning text alone; it
   must be reconstructed from `O_t` plus the answer requirements in `Q_t`.
4. **Trajectory updates:** assumption change, question continuity, discordance, and only then a
   derived transition summary, while preserving unsupported orders.

The intended downstream use is:

```text
order-aware A/Q/C* reference labels for knowledge discovery
        -> learn/infer pre-order A/Q/C without the current order at evaluation time
        -> compare N vs A/Q/C vs N+A/Q/C on held-out next-image/repeat/switch/stop
```

Useful compression or agreement does not prove recovery of private physician thought. Evidence for
a shared logic requires codebook saturation on held-out trajectories, independent coding views,
an unsupported residual, and incremental held-out predictive value beyond `O_t` and `N`.

### 7.2 Codebook sample versus pattern-discovery corpus

Do not confuse the small qualitative codebook sample with the corpus used to discover and estimate
final patterns. From the 293-patient `full/` corpus, first make a stable patient-level approximately
80/20 development/final-test split within disease. Keep the final test untouched while developing
the representation. Within development, an initial approximately 24 trajectories total (about six
per disease) are read deeply to define atomic assumption types/statuses, question types/targets,
answer requirements, and coverage rules. They are selected for structural variation and are not a
statistically representative prevalence sample.

Twenty-four is an initial batch rather than a fixed sample size. Add fresh, non-overlapping
development batches when they reveal new top-level types, recurrent answer-requirement dimensions,
or systematic `other/unclear` clusters. Check every revision on another fresh development batch and
freeze only after qualitative saturation across diseases and major timing/sequence strata. After
freezing, apply the codebook to the larger development partition; that larger coded set is where
recurrent transition patterns and frequencies are discovered. The untouched final test is then used
to test codebook coverage, pattern replication, and pre-order predictive value. Any patient used to
revise the codebook belongs to development and cannot remain part of the final test.

## 8. Annotation procedure

The main unit is an entire trajectory, not an isolated order. For each current order, the LLM sees
the causally available record, previous tests and results, and the actual current order. It jointly
reconstructs:

1. the previous question and what the previous result did to it;
2. the updated assumption;
3. whether the question continued, advanced, reopened, or rerouted;
4. the current question, its answer requirements, and its pre-order coverage profile;
5. why the current test could answer it.

Annotate a small pilot in two ways:

- directly reconstruct A/Q/C from the masked trajectory plus actual orders;
- recode the existing schema-light reasoning into A/Q/C.

In plain terms, these are two independent annotation routes for the same historical cases: one
starts from the causally available patient record, while the other converts the old reasoning
annotation. Their agreement measures dependence on the old annotation scaffold. If A and Q agree
closely, the old annotations may be reused as the main bulk source for those fields, with independent
re-annotation focused on disagreements and weak support. This does not eliminate the need to build a
new A/Q/C data layer, and requirement-level C still requires causally available patient evidence.
Agreement alone is not proof of clinical correctness or recovery of private physician intent.
Preserve the old schema-light annotations rather than overwriting them.

## 9. Prompt requirements

- Use the actual order to infer intent, but require support in the visible chart.
- Never show the current result or later events.
- Maintain one coherent A/Q/C chain across the trajectory.
- Separate disease certainty from uncertainty about etiology, severity, and complication.
- Separate study adequacy, test–question capability, result status, and aggregate coverage.
- Define the answer requirements inside Q before assigning C; do not let C invent its own target.
- Record C as a requirement-level state grounded in all causally available evidence, with an
  optional aggregate summary.
- Treat a valid negative as informative; do not confuse it with nonvisualization.
- Require explicit evidence for material discordance.
- Allow multiple plausible explanations and `unclear/weakly supported`.
- Do not force every real order to fit A/Q/C or rationalize unsupported imaging.
