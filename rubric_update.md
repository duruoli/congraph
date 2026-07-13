# Rubric Update — a question-driven redesign

> Working design doc. Motivation: the "wrap the rubric with alarm-override rules" patch (A2/A3/A1
> as external booleans) is too shallow — it treats the rubric's blind spots as exceptions bolted on
> top. This doc explores the deeper move the analysis points to: **rebuild the rubric so that
> inadequacy, ask-the-right-question, and (tentatively) discordance are native consequences of the
> rubric's own state, not add-ons.** Grounded in the `study_inadequacy` corpus (185 flagged steps,
> `results/annotation_experiment/alarm/*.json` joined to `full/*.json`).

---

## 0. The two build directions (2026-07-13 — the organizing frame)

Everything below decomposes into **two construction directions**. They are not parallel: **Direction 2
is the substrate Direction 1 runs on.** (Discordance / A3 is deliberately SET ASIDE for now — folding it
in makes the thread too diffuse; revisit after these two are standing.)

**Direction 1 — build the Question–Test tree (the procedural spine, §3/§6).**
Nodes = the LLM-annotated `target question`s (Phase-1 done: seed was `existence / etiology / severity /
complication / broaden`; **`broaden` was subsequently REMOVED — it is a belief-track re-route, not a
question type, see §6h — so the node set is the 4 remaining types**, `results/question_targets/`). The **hard part is not the nodes,
it is deriving ONE shared, consistent Q–T tree per disease** — the edges and their ordering. Two
constraints the data imposes:
- **It is a question-type LATTICE / partial order, NOT a linear pointer.** 452/542 steps are
  multi-label — a single step holds 2–3 open questions at once (panc `complication+etiology`=91,
  `complication+severity`=61). The §3 "advance one pointer existence→etiology→severity→complication"
  model is too linear; the shared object is a lattice, and each patient traverses a *subset path*
  through it. That is how "one shared tree" (lattice fixed) coexists with "every patient differs"
  (path differs).
- **Cross-disease consistency lives in the question-TYPE vocabulary, not the content.** The 5 types
  are shared across all 4 diseases; the disease-specific part (what `etiology` *means*: stone vs
  alcohol vs perforation-source) is pushed down into `required(S)` per (question, disease). So the
  tree is shared in shape, specialized in its guards.

**Direction 2 — build the evidence pieces = the shared signal ontology `S` (§6c).**
To let the tree *flow*, its edges need conditions (guards). The language of those conditions is `S`:
fragment every report into structured **evidence pieces** = `(anatomy/名词 + state/状态)` records,
cluster them to discover a **名词库 (anatomy vocabulary)** and **状态库 (state vocabulary)**, and these
become the raw material for the two gates — the Q→T **guard** state and the T→Q **adequacy** gate.
Coverage spans **physical exam + radiology + laboratory** — every patient-state signal in the raw data.
Note the split by perception method: **labs/vitals are already tabular** `(name, value)` → parse, don't
build a text-splitter; the fragmentation task proper is **radiology reports + physical-exam narrative**.

**Why 2 is under 1:** `observed(S)` is the ONE accumulating state vector (§6 "state is one vector"). The
SAME vector both *guards the Q→T edges* (Direction 1's conditions) and *drives the T→Q adequacy gate*.
So without `S` there is no language to write Direction 1's edges in — build (or at least schema-fix) `S`
first.

**The `S` schema (what "schema" means here) — v1 LOCKED 2026-07-13.** `S` ≈ the noun-state combination,
but "schema" is the **fixed slot TEMPLATE** each evidence piece is decomposed into, so pieces become
machine-queryable conditions (a flat "pancreatic tail not visualized" string can't be queried; a slotted
record can). Validated against the real demand-side phrasing (`sought_dimensions`, all 542 rows) before
locking. Template for one element of `observed(S)` / `coverable(S)`:
```
evidence_piece := {
  # CORE (every piece)
  anatomy:        <organ/region>       # 名词库 — clustered, CORPUS-WIDE  (appendix, CBD, pancreatic tail, GB wall, adnexa, aorta, ureter, mesentery, terminal ileum, RLQ…)
  attribute:      <property>           # wall / diameter / stones / fat-stranding / fluid / enhancement / compressibility / sonographic-Murphy
  state:          <raw descriptor>     # 状态库 — clustered  (thickened / dilated / absent / necrosis / pneumatosis / non-compressible / non-visualized / normal …)
  finding_status: abnormal|normal|not_evaluated|equivocal   # ← the NORMALIZED enum the GATES read; carries inadequacy ①②
  source_test:    <modality/exam>      # key for coverable(S)-per-test (capability template)
  # OPTIONAL
  value+unit:     <numeric>            # thresholded guards (appendix >6mm)
  qualifier:      <hedge / protocol>   # "not fully visualized" / "limited by bowel gas" / "non-contrast" → inadequacy ①③ detail
  section:        findings|impression  # provenance / dedup
}
```
名词库 and 状态库 are SLOTS discovered by clustering; the schema is the fixed template, the vocabulary
fills it. The gates query BY SLOT: `adequacy(R,q) = required(q) ⊆ covered(R)` reads
**`(anatomy, attribute)` pairs** with `finding_status ∈ {normal,abnormal}` (i.e. actually evaluated); a
guard reads e.g. `pancreatic_tail.state == non_visualized`.

**Three decisions the `sought_dimensions` validation settled (2026-07-13):**
- **The atomic covered/required unit is the `(anatomy, attribute)` PAIR, so `attribute` is CORE not
  optional** — the demand side is dominantly sub-organ: the *same* organ recurs with *different*
  attributes as *distinct* required dims (gallbladder × {wall / stones / pericholecystic-fluid /
  sonographic-Murphy}; CBD × {diameter / obstructing-stone / dilation}). `anatomy` alone would let a
  study that never assessed the GB *wall* pass "GB was covered."
- **`state` (free, clustered) vs `finding_status` (normalized enum) SPLIT is confirmed** — the state
  vocabulary is genuinely rich (thickened/dilated/stranding/necrosis/pneumatosis/non-compressible),
  clustering has real work; the gate keys on the small enum, not the free field.
- **`anatomy` vocabulary is CORPUS-WIDE, not per-disease** — demands routinely reach neighbors /
  alternatives (adnexa, aorta, ureter+hydronephrosis, mesentery, portal vein). Index disease only
  *weights* the vocabulary.

**Two elements deliberately NOT slotted in v1 (the fragmentation prompt STRIPS them):**
- **implication tails** ("...*indicating* cholecystitis", "...*to grade* severity") = the
  finding→hypothesis link = **M-matrix / belief-track = the discordance track set aside.** Strip, don't
  slot.
- **temporal/comparative dims** ("*interval change* in size", "*persistence/increase* of free air") —
  a few dims are vs-prior not absolute; fold into `state` as a value ("interval-increased") for v1,
  flag a proper comparative axis as later-maybe.

**⚠️ Clustering ≠ the gates (the load-bearing caveat, ties to §6c).** Clustering discovers only the
VOCABULARY (名词库/状态库 = the nodes). It gives NEITHER (a) `question → required(S)` (which dims answer a
question — needs standard medical knowledge + verification mining), NOR (b) the thresholds, NOR (c) the
tree edges/ordering. Recipe = **clustering (vocab) + standard knowledge (structure/edges/required sets)
+ verification (thresholds, keep it non-circular)**. "Evidence pieces → gates" is really "evidence
pieces are the *raw material* for gates."

**⚠️ "verification" is NOT "outcome = confirmed" (correction, 2026-07-13, replaces the old
"vindication").** The success criterion below and the §6c recipe both said "absorb only vindicated
deviations." Renamed to **verification**, and widened: a deviation is *verified / worth absorbing* when
it was **ex-ante justified — alarm firing OR belief diffuse — regardless of whether the outcome
confirmed or disconfirmed.** The agent-training analysis is explicit: *a mistake judgment NEVER rests on
a single outcome; outcome only updates next-step belief* (`agent_training_plan` §2.5), and most
DISCONFIRMED deviations are legitimate (rule-out-by-design, serendipity redirect — `disconfirm_preadmission_findings`).
So a broad net that comes back negative but was the right ex-ante move IS verified and absorbable. What
verification still EXCLUDES is the over-imaging brake = **alarm silent ∧ belief concentrated ∧
same-modality repeat** (the ex-ante distinguisher, not the outcome). Absorbing all deviation would import
the doctor's mistakes; absorbing only outcome-confirmed would throw away the reasonable rule-outs.

---

## 1. Empirical typology of "inadequate" (185 flagged steps)

A study is flagged inadequate for **4 distinct mechanisms**. Each has a *different* detection
signature when reading a report — this matters because "how do we systematically judge inadequacy"
has a different answer per type.

| # | Mechanism | Share | Report signature (the detection sign) | Grounded example |
|---|-----------|-------|----------------------------------------|------------------|
| **①** | **VISUALIZATION** — target organ obscured / not shown (technical) | **~113/185 (61%)** | Report **self-declares**: "not visualized / not fully visualized / tail obscured / limited by bowel gas / suboptimal / body habitus / not definitively" | panc 20474777 (lipase 4555 but pancreatic tail not fully visualized); chole 21948836 (distal CBD not visualized); appe 20123918 s2 (tail obscured by bowel gas) |
| **②** | **COVERAGE** — target organ never in the report's field / findings | ~12 | Report **does not mention** the organ of the current question; it answered a *different* question | appe 20123918 s1 (pelvic US reported normal tubes/ovaries — appendix never mentioned) |
| **③** | **PROTOCOL** — modality ran without the parameter the question needs | ~9 | "without IV contrast / non-contrast / unenhanced" + the question requires enhancement | diver 22429578 (thyrotoxicosis → no IV contrast → solid-organ & vascular eval limited); panc 20009550 (high creatinine → no contrast → appendix not definitively seen) |
| **④** | **CAPABILITY** — modality *intrinsically* cannot answer the current question (grade / etiology) | lexical scan caught only 3, **but `action_role=assess_severity` = 66 steps** → this class is large and hides in semantics ("leaving extent/severity unanswered"), not in "cannot stage" | US confirms pancreatitis exists but cannot grade necrosis; US insensitive to microlithiasis |

**Detection method per type (the "sign" for systematic report reading):**
- ① = **lexical scan** of a closed hedge vocabulary (highest yield, easiest, ~60%). The report says it itself.
- ② = **coverage check**: is the organ that the current question is about present in the findings section? Absent → flag. Needs a `question → organ(s)-that-must-appear` map.
- ③ = **protocol-line check**: contrast/parameter missing ∧ question needs it.
- ④ = **modality × question capability table** (a knowledge rule, NOT in the report text): which modality can answer which question (existence / etiology / severity / complication).

**Caveat / next read:** ① is essentially solved (lexical) but is also the *shallowest* type — the
report already admitted it. The classes that actually need defining are **② and ④**, especially ④.
The regex under-counts ④ badly (3) while `assess_severity`=66 says it is the real mass; those notes
phrase it semantically. **Next step: read the 66 `assess_severity` notes in full to nail the ④
definition** (and decide whether "existence confirmed → grade severity" vs "existence confirmed →
find etiology" should be two separate question types).

---

## 2. The load-bearing insight: **adequacy is question-relative**

Type ④ exposes the core structure: **a study is never adequate/inadequate in the absolute — only
relative to the question currently open.** The same abdominal US is *adequate* for "is this
pancreatitis?" and *inadequate* for "how severe is the necrosis?" The study did not change; **the
question moved, and adequacy flipped.**

Consequence: the two alarms we wanted to add are **not independent modules** —
`inadequacy` and `ask-the-right-question / question-moved` are **two readings of one predicate**:

```
answered(current_question, study_history) ?
   NO  → inadequacy alarm  → do NOT advance; re-image (better modality / coverage / protocol)
   YES → question resolved → advance the question pointer; route the next modality BY THE NEW
                             QUESTION, not by the disease label   (= "ask the right question", A1)
```

So A1 (question-moved) and A2 (inadequacy) collapse into **one machine**: an adequacy check between
the completed study and the open question, whose two outcomes are "advance" vs "re-image".

---

## 3. Proposed architecture: a question-driven rubric

**State is no longer "which disease node am I at" but "which QUESTION is currently open."**

- **Question stack / pointer.** Questions are ordered per disease: `existence` (is it X?) →
  `etiology` (why?) → `severity` (how bad?) → `complication`. The pointer advances as questions
  are answered; it can also be pushed back up (see discordance).
- Each question carries: candidate answers; the **modalities capable of answering it** (capability,
  type ④); the **organs/fields that must be covered** (coverage, type ②); **protocol requirements**
  (type ③).
- **Adequacy = a computable predicate** `adequate(study, question)` = coverage ∩ capability ∩
  protocol satisfied. This is exactly types ②③④; type ① is a study-quality flag that fails
  adequacy even when the modality was capable in principle.
- **A2 inadequacy** = `adequate(last_study, current_question) == False` → re-image, don't advance.
- **A1 ask-right-question** = when the current question *is* answered, advance the pointer and
  **route by the new question** (this is the whole "route by clinical question, not disease label"
  finding — it becomes the pointer-advance rule, not a special case).

In this frame the rubric no longer "stops reasoning too early" because *stopping* is redefined:
you stop only when the question stack is empty, not when the first disease-first study completes.

---

## 4. Where discordance (A3) fits — GROUNDED (215 discordance-flagged steps)

Dug into all 215 `discordance.present` steps (`alarm/*.json` ⋈ `full/*.json`; the reasoning lives in
the `evidence` field — `note` is empty in 199/215, and `evidence` is already written as a two-pole
contrast sentence, 99/215 with an explicit connector: versus / yet / contrasted-with / out of
proportion / cannot explain).

**Definition (authoritative, from the detector prompt `prompts.py:129`) — discordance is
MODALITY-AGNOSTIC:** *a MARKED conflict between two streams of evidence that the leading suspected
diagnosis CANNOT explain, striking enough to change management.* The prompt lists BOTH "grossly
abnormal labs/vitals with a benign abdomen" AND "**imaging findings out of proportion to the clinical
picture**" as first-class cases. ⚠️ **Correction of an earlier overclaim in this doc / memory:
discordance is NOT "an objective NON-IMAGING stream."** Imaging is in the conflict in **91/215 (42%)**
of flags — as the *violation* pole 64× (severe finding vs benign exam/leading Dx, e.g. CT-perforation
with a soft abdomen; MRCP cecal colitis under a post-ERCP biliary lead) and as the *anchor* pole 20×
(normal/negative study vs abnormal labs). Only **124/215 (58%)** are purely labs/vitals/exam with no
imaging. The "non-imaging" phrasing was a category error: it named A3's *distinctive contribution*
(the streams the question-tree does not natively track — labs/vitals/UA are the purest of these) and
mistook that for the *definition*.

**The extractable structure is a 3-slot, stream-agnostic tuple:**
```
discordance := {
  expectation:  the picture the leading hypothesis PREDICTS / is consistent with   # usu. benign exam; can be expected imaging
  violation:    {stream, value, direction}     # stream ∈ {exam, vital, lab, UA, hCG, PRIOR-imaging}  ← modality-agnostic
  relation:     out_of_proportion(severity) | points_elsewhere(localization) | internally_contradictory
}
```
The load-bearing classification axis is **`relation`, NOT imaging-vs-non-imaging.** The `violation`
is an unexplained **residual**; note the detector is masked/blind, so the imaging it can use as a pole
is only ALREADY-RESULTED `visible_prior_imaging`, never the study being ordered. Residual-stream
tallies (multi-count of 215): WBC/bands 134, LFT/biliary 97, lactate 89, acid-base 59, lipase/amylase
58, vitals/hemodynamic 49, D-dimer/vascular 29, cardiac 19, renal 17, coag 14, Ca/UA/hCG 6/5/2.

**The flow response is a CONTINUUM in the residual size (≈ `other_mass`), not two discrete modes:**
- **small residual — disease still peaked (58/215):** keep the diagnosis, **forward-attach** a
  reconciliation sub-question (second source / severity / etiology). e.g. chole 27738411 (UA→urinary
  second source), panc 22944548 (lipase 14240 + benign exam → severity).
- **large residual — other-heavy (108/215):** localization collapses → **re-open / broaden** (the
  "倒车"). e.g. diver 21292285 (unstable+peritonitic, CT no source), diver 23962694 (other=0.85),
  appe 23562407 (hCG+→gynecologic). (49 mid.)

`other_mass` is the ready-made operationalization of the residual; disease-max vs other_mass is a
sound proxy for which flow fires. **Backtrack-to-a-deprioritized-sibling is only the sub-case where
the residual LOCALIZES to a specific alternative** (hCG→gyn, UA→urinary); when the residual is
diffuse (no clean sibling) the op is just "widen." So A3 = **re-open proportional to the residual;
retreat to a specific sibling when the residual localizes, else cast a broad net.**

**Layered picture of the three alarms — by WHAT each reads:**
- **A2 inadequacy = STRUCTURE** (node-local): reads `(study, current_question)` → adequacy predicate
  (coverage/capability/protocol) → re-image vs pass.
- **A1 ask-right-question = LOCAL forward transition**: reads `(question, answered?)` → advance the
  pointer, route next modality by the new question. *(Note: A1 is flow, not structure — the earlier
  "A1+A2 both structural" grouping is wrong; the clean cut is A2=structure vs A1/A3=transitions.)*
- **A3 discordance = RESIDUAL-DRIVEN re-open transition**: reads `(leading hypothesis, all objective
  streams — exam/labs/vitals/UA AND already-resulted prior imaging)` → the stream the hypothesis
  cannot explain = unexplained residual → re-open with magnitude ∝ residual, mode set by `relation`
  (localized → retreat to the named sibling; diffuse/severity → widen).

**Honest structural cost of A3 (confirmed, harder, second-phase):** it needs **all objective streams
IN the state PLUS a model of what the leading hypothesis PREDICTS for each stream** (to compute the
residual) — heavier than A1/A2. The extra streams over A1/A2 are the non-imaging ones (labs/vitals/UA)
— which is A3's distinctive load, though the violation can equally be an off-target *prior* imaging
finding. A3 co-occurs with inadequacy only **26%** of the time → a genuinely independent channel.

---

## 5. Two-track architecture: a parallel belief-track alongside the question tree

The question tree (§3) is the **procedural/normative** axis; it cannot by itself hold A3, because
A3 lives in **belief space**. Add a second, parallel structure — the two run coupled:

- **Question tree (procedural):** nodes = questions; holds **A1** (advance) and **A2**
  (adequacy/re-image). = the rubric.
- **Belief track (epistemic):** a set of **live NAMED hypotheses with mass**, evolving over time —
  the 4 diseases + 'other' **expanded into named out-of-rubric hypotheses** (gynecologic, urinary,
  mesenteric ischemia, biliary duct, RPOC…). **This already exists in the data** = the reconstructed
  `differential` + `other_hypothesis`.

**Coupling (the two tracks cross-drive each other):**
- belief argmax → **selects the active branch** of the question tree (routing; = existing
  `top_branch`/`active_path`).
- question-tree outcomes (verification ±) → **update belief mass**.
- **A3 discordance** = computed IN the belief track (residual = mass the named disease-hypotheses
  cannot absorb, ≈ `other_mass`) → **acts on the question tree**: re-open ∝ residual (localized
  residual → jump to the sibling question-branch the residual names; diffuse → widen the active set).
  The belief track is exactly the state A3 needs per §4's cost.

**What this unlocks (the load-bearing payoff — ties back to the project's core goal):**
- **The belief track is the rubric-incompleteness / OUTLIER detector.** A live belief-track
  hypothesis with **no corresponding question-node** in the rubric tree = the rubric is missing a
  branch for THIS patient = the patch signal. **Outlier = belief track escapes the question tree.**
- It **retroactively explains the biliary post-hoc branch**: a belief-track hypothesis (biliary
  duct) with no rubric node → a branch was hand-patched. The two-track view makes that ad-hoc move
  systematic — strong evidence the architecture is right.
- "Completing the rubric" = promoting belief-track hypotheses that **recur across patients** into
  their own question-node = **exactly step 5 synthesis of the rubric-update pipeline.**

**Refinement (important):** the belief track must carry **named hypotheses with mass, NOT a scalar
`other_mass`** — otherwise the localized-residual "retreat to a specific sibling" case (hCG→gyn) has
nothing to point at. `other_hypothesis` already supplies the names; structure them into the track.

**Design cost:** the coupling **protocol** (routing / update / re-open rules) is the hard part.

## 6. The executable graph: an alternating question↔test state machine (2026-07-11/12)

The two-track idea (§5) becomes runnable as a **procedural spine** (this section) *driven and
interrupted by* the belief track. The spine is an alternating graph of two node types:

```
     belief track ─── argmax 选中当前活跃 Qi ───┐        ┌── 残差>阈值 → push 新 Q(可能树外)
        ▲ 更新 mass                             │        │
        │                                       ▼        ▲
   ┌────┴───────────────────────────────── [Question Qi] │
   │      ──state guard g1──▶ [Test A]           │   labs/vitals/前片(≠当前 test 的 result)
   │ Qi ────state guard g2──▶ [Test B] ──下单出报告──▶ observed(S) 累积进 state
   │      ──state guard g3──▶ [Test C]                    │
   │                                                      ├─ adequate(report,Qi)=False ─▶ 回 Qi(**re-image / 模态升级边**,必换 test = A2)
   │                                                      └─ adequate=True ─┬─ confirmed ─▶ [Q severity]
   └────────────────────────────────────────────────────────────────────  ├─ excluded ──▶ [Q sibling/done]
                                                                           └─ partial ───▶ …
```

- **Q→T edges = state guards.** Each out-edge of a question carries a condition on the *accumulated*
  state → routes to a different test. This is `f(Qi, enriched_state) → test` (§ below). The guards are
  exactly the absorbed-deviation state variables (pretest_prob / body_habitus / prior_study_adequacy /
  sub_question).
- **T→Q is a TWO-STAGE gate, not one edge.** (1) **adequacy gate first**: decompose the report into
  `observed(S)`, test `required(Qi) ⊆ diagnostically-covered?` — **inadequate → re-image edge back to Qi**
  (retry with a **different** modality = **A2**; inadequacy does NOT branch, it retries). (2) **only if
  adequate, fork on the ANSWER value** (= verification: result vs `expected_finding`): confirmed→
  severity, excluded→sibling/done, partial→… This second stage is the "test result routes to a
  different question" = **A1**.

  **⚠️ Naming fix (2026-07-12): "self-loop" is misleading — it loops on the QUESTION node, NOT the
  test.** A2 is a **re-image / modality-escalation edge**: return to Qi, but Qi's out-edge now carries
  `prior_study_adequacy=inadequate` as a guard → it routes to a *different, better-suited* test (US→CT,
  noncon→con CT, US→MRCP). It is **never "repeat the same test twice."** Empirically this IS the dominant
  multi-step pattern: **79% of 249 consecutive step-pairs change modality** (US→CT 75, CT→US 59, US→MRCP
  31); only 21% same-modality (and those are mostly protocol escalation noncon→con). Because state is
  richer each pass (`prior_study_adequacy`), it is not a true loop — the test choice changes every
  round, terminating when a test reaches adequacy (or capability is exhausted → question unanswerable).
  **The adequacy gate is exactly what separates legitimate re-image (A2, `adequate==False`) from the
  defensive re-scan we want to REMOVE (over-imaging brake, `adequate==True` yet re-scanned): "same test
  twice" is the brake, and A2 refuses to fire on it.**
- **discordance is NOT an edge on this chain.** It is a **parallel interrupt from the belief track**
  that reads ALL objective streams (labs/vitals/UA + already-resulted prior imaging — often NOT the
  current test's result, e.g. lactate 3.8 + benign exam before any imaging returns) and can `push a
  new Q` at ANY node (localized residual → sibling Q; diffuse → widen). Drawing it as a "test-result→Q"
  edge mis-models the ~42% of flags triggered by labs/vitals alone.
- **State is ONE accumulating vector, not per-step fresh.** `report → observed(S)` updates it; the
  SAME state both guards the Q→T edges and drives the T→Q adequacy/answer test. That is why Qi's
  out-edge can read "is the prior study adequate / how high is pretest" — they are already in state.

### 6b. "Updating the rubric" = enriching the state until the doctor's test becomes derivable
The binding is NOT static `question→test`; it is a deterministic function `f(question, ENRICHED
state) → test`. **"Deviation" is only relative to the raw rubric's impoverished state (disease label
only).** Absorbing a deviate+confirmed finding = **discovering the missing state variable** that made
the doctor's choice non-derivable:

| absorbed finding (deviate+confirmed) | state var added to `f` | after absorption `f` derives |
|---|---|---|
| adult/obese skip US → CT | `pretest_prob / body_habitus` | high-pretest adult → CT (no "deviation") |
| overturn stop when prior study equivocal/limited | `prior_study_adequacy` (= §3/§6c adequacy predicate output) | inadequate prior → keep imaging |
| CT→US/MRCP when the question is biliary | `sub_question(duct vs parenchyma)` | question=duct → US/MRCP |

`f` **collapses to a single test** when state is rich enough (the derivable core); it stays a *set*
only where (a) genuinely underdetermined, or (b) the rubric *correctly disagrees* with the doctor.

**Success criterion (must守住): absorb only VERIFIED deviations, not all deviation.** ⚠️ **VERIFIED ≠
outcome-confirmed** (see §0): a deviation is verified when it was **ex-ante justified (alarm firing OR
belief diffuse)**, whether or not the result came back positive — most disconfirmed deviations are
legitimate rule-outs and ARE absorbable. What stays excluded is the over-imaging brake (**alarm silent ∧
belief concentrated ∧ same-modality repeat**), which is doctor over-imaging / defensive re-scan / error
(the disconfirm/brake findings B1–B3, DD-1). So the "adult/obese skip US→CT" table row absorbs on the
ex-ante state var (pretest/habitus), NOT on that one CT confirming. Target = "no *verified* deviation
left underivable, AND the rubric still diverges where the doctor was wrong." Residual divergence from the
doctor is not failure — it is the rubric refusing a suboptimal order. Metric = step-6 **asymmetric**
validation: recover verified-deviation→follow WITHOUT starting to endorse bad orders. Aiming for literal "zero deviation vs doctor" would import the doctor's mistakes
and make the rubric descriptive instead of normative.

### 6c. One shared signal ontology `S` is the mandatory substrate (ties §1/§4/§6 together)
Construction recipe for the whole system = **clustering (discover nodes/vocabulary) + standard medical
knowledge (structure, templates, edges) + verification validation (set thresholds, keep it
non-circular)**. Clustering alone CANNOT give thresholds, the `question→required(S)` sets, or the tree
edges/ordering — those need standard knowledge + supervised mining against observed verification.

To make adequacy, discordance and questions *compose* (not grow three incompatible vocabularies),
define ONE canonical signal/dimension ontology `S` (per test-type dimensions + labs + vitals + exam),
and have everything reference it:

| mapping | meaning | used by |
|---|---|---|
| `report → observed(S)` | extraction / fragmentation (perception layer; labs already structured — don't build a text-splitter for tabular data) | shared |
| `test-type → coverable(S)` | capability template (finite # of test types) | adequacy ④ |
| `question → required(S)` | which dimensions answer this question (adequacy is question-RELATIVE — targeted set-cover on required dims, NOT an aggregate "missing > k" threshold) | **adequacy** |
| `hypothesis → expected_profile(S)` | the `hypothesis × stream` matrix `M` (signed/ranged) | **discordance** |

Then `adequacy(R,q) = required(q) ⊆ diagnostically-covered(R)`; `discordance(observed, h) = ∃ s:
observed abnormal ∧ expected_profile(h) can't predict it`. The report-dimension template also unifies
all 4 inadequacy types (§1) as operations on ONE template: ②=required dim absent, ①=dim present but
hedged, ③=dim present but protocol-limited, ④=required dim not in this test's `coverable(S)`.

### 6d. Discordance without enumerating pairs — read `M` two ways
Never enumerate all stream-pairs (O(n²)). One pole of every conflict is ALWAYS the leading
hypothesis's prediction ⇒ O(n) residuals against ONE reference. Build `M[h][s]` (does hypothesis `h`
predict/explain abnormality in stream `s`, and in what direction). **Row-wise (fix h):** observed
abnormal `s` with `M[h][s]=∅` → discordant residual (the "is it discordant?" test). **Column-wise (fix
s):** which OTHER `h'` has `M[h'][s]=✓` → the residual localizes to that named sibling (routing). The
`relation` type falls out: stream hits exactly one sibling = localization→retreat; systemic streams
(lactate/bands/AG/hemodynamics) explained by NO single disease = diffuse→widen; joint-pattern
violation = internally_contradictory (needs a coherence rule, not a marginal lookup). **Build `M`**
by (a) a small hand seed (4 diseases × ~15 streams) and (b) mining existing annotation:
`grounding`/`expected_finding` = the ✓ cells, `discordance.evidence` = the ∅ cells. Perception (LLM/
rules extract structured `observed(S)`) is split from judgment (algorithmic `M` lookup + threshold) —
more auditable than today's end-to-end-LLM alarm.

### 6e. Phase-1 question_target extraction — DONE (2026-07-12, `scripts/extract_question_targets.py`)
Re-grounded the OPEN question of all 542 steps from free-text `information_gap`(+`expected_finding`)
via Claude Sonnet/OpenRouter → `results/question_targets/`. Multi-label seed {existence, etiology,
severity, complication, broaden} + open `other:` escape; also extracts modality-free `sought_dimensions`
(raw material for `required(S)`). **540/542 labeled, 452 multi-label** (characterize questions come in
bundles). Results:
- **`action_role` incompleteness CONFIRMED, hard:** old `localize_source` splits into **etiology=189,
  complication=115**, existence=126, broaden=112, severity=37; old `assess_severity` hides
  complication=128. **etiology & complication (the two targets `action_role` LACKS) are the largest
  classes corpus-wide** (panc etiology=164/complication=120, chole etiology=109). This is the empirical
  justification for the question-target ontology over `action_role`.
- **NO `other` spans** → the 5-label seed set is closed at this granularity (no 6th target).
- **`sought_dimensions` is clean** (anatomy/pathology only, no modality words) → usable required(S) seed.
- **broaden↔discordance are STATISTICALLY INDEPENDENT** (lift=0.93; P(discord|broaden)=0.37 vs
  P(discord|~broaden)=0.42). Empirically refutes "conflict→broaden" as an identity and **confirms the
  two-track claim that broaden (question-tree move) and discordance (belief-track interrupt) are
  different, independent layers.**
- **⚠️ broaden is too loose (228/542 = 42%, flat across n_prior)** — RESOLVED by removal, see §6h. (It
  over-fired because it was catching "differential breadth" + non-visualization; the clean definition
  turned out to be a belief-track event, not a question type, so it leaves the label set entirely.)
- severity & complication co-fire heavily (panc complication+severity=61) but have distinct marginals →
  keep separate for now; the merge decision is a downstream analysis, not baked into extraction.

### 6h. `broaden` REMOVED from the question-target label set (2026-07-13 DECISION)
The 5th seed label `broaden` is **dropped**. The clean definition of broaden — *prior evidence made the
leading diagnosis untenable, so switch to a new possibility* — is a **belief-track / routing event, NOT
a question type.** The other four (existence/etiology/severity/complication) are all questions asked
*about one already-held hypothesis* (the tree's nodes); broaden is a *switch between hypotheses/trees*,
a different ontological kind. Forcing it into the same multi-label slot is a category error and is
exactly why it over-fired (42%, flat across n_prior). Consequences:
- **Question-tree label set = {existence, etiology, severity, complication}** — homogeneous "verify /
  characterize a held hypothesis" nodes. `existence` = verify a suspected disease is present (its
  "which disease among several" flavor is just existence under *diffuse* belief — NOT a separate type;
  `existence` well-attested corpus-wide, old `action_role=rule_in`≈57 steps are pure confirm/rule-out).
- **Choosing / switching the leading hypothesis** (t0 prior AND broaden re-route) lives in the
  **belief track** — the SAME place, deliberately SET ASIDE with discordance/A3. broaden is set aside
  with it. It is not "downgraded"; it was never a question type.
- **Re-route is TARGETED, not a meta re-walk** (corrects an earlier framing): on discordance you do NOT
  re-enter a blank "which disease?" node — the violating abnormal stream `s` with `M[leading][s]=∅`
  self-localizes to the sibling `h'` where `M[h'][s]=✓` (§6d column-read), so you jump straight into
  `h'`'s tree with the suspicion already in hand; the question there is again plain `existence(h')`.
- This also **dissolves the broaden↔discordance "independence" puzzle** above: the independence was an
  artifact of the polluted label; under the clean definition broaden *is* the discordance re-route.
- **Extraction impact:** `results/question_targets/extractions.jsonl` keeps its `broaden` spans on disk
  (no re-run needed / no re-run planned); downstream tree construction simply IGNORES the `broaden`
  label and reads only the 4 types. `sought_dimensions` is unaffected.

### 6f. DEFERRED BLOCKER — lab + microbiology per-test timestamps not yet sourced (2026-07-13)
The `Laboratory Tests` and `Microbiology` columns in `data/raw_data/{disease}_hadm_info_first_diag.csv`
are **admission-aggregated `{itemid: value}` dicts with NO per-test `charttime`.** To place a lab/micro
result on the timeline (what the doctor knew *before* a given imaging decision — the ordering the
question tree, §3/§6, is built on), each result needs its real chart time. That must come from MIMIC-IV
raw source, and is **deliberately DEFERRED** (not blocking the current Direction-1/Direction-2 build off
narrative + already-timed radiology).

- **Where the timestamps live — the `hosp` module, NOT `icu`** (the icu link is the wrong module for
  labs/micro): `https://physionet.org/content/mimiciv/3.1/hosp/`.
- **Files to download when we do this:**
  - `hosp/labevents.csv.gz` — carries `charttime`, `storetime`, `valuenum`, `ref_range_lower/upper`,
    `flag`. ⚠️ ~2 GB compressed / ~100M+ rows → filter by our `hadm_id` set on ingest, don't load whole.
  - `hosp/microbiologyevents.csv.gz` — `charttime`/`chartdate`, `storetime`, specimen fields (small).
  - `hosp/d_labitems.csv.gz` — itemid→label dictionary (partly covered already by
    `data/raw_data/lab_test_mapping.csv`).
- **Already present** in `data/raw_data/mimic_source/` (git-ignored): `admissions.csv.gz`,
  `radiology.csv.gz` (report charttimes — imaging already timed), `procedures_icd.csv.gz`,
  `d_icd_procedures.csv.gz`. So only **labevents + microbiologyevents (+ d_labitems)** are missing.
  Same PhysioNet-credentialed / AWS-S3-access-point route as the note pull ([[real-charttime-timing]]).
- **Consequence for Direction 2 ingestion (§6c path A):** until sourced, lab/micro evidence pieces can be
  built as *untimed* `observed(S)` (finding_status still free from the CSV's ref ranges), but they can NOT
  yet be split into pre-/post-decision — so any lab/micro-dependent guard or adequacy check stays
  admission-global, not step-relative, for now.

### 6g. How raw fields map onto the locked `S` schema — THREE ingestion paths, not one (2026-07-13)
`observed(S)` = the union of three ingestion paths, split by *perception method* (extends §0 "labs are
already tabular — don't build a text-splitter"). Reads `data/raw_data/{disease}_hadm_info_first_diag.csv`
columns: `Patient History`, `Physical Examination`, `Laboratory Tests`, `Reference Range Lower/Upper`,
`Microbiology`, `Microbiology Spec`, `Radiology`.

- **Path A — tabular parse, NO LLM (Laboratory + Microbiology + vitals block).**
  `Laboratory Tests` is `{itemid: value}` and the SAME CSV already carries `Reference Range Lower/Upper`
  per itemid ⇒ slot-fill is mechanical: `attribute`=analyte (itemid→label via `lab_test_mapping.csv`),
  `anatomy`=organ/system the analyte indexes (or `serum/blood`), `state`=raw value string,
  `value+unit`=parsed numeric, **`finding_status`=valuenum vs ref_range → normal/abnormal (FREE — the
  normalized enum costs ~nothing for labs)**, `source_test`=laboratory. Microbiology parallels: anatomy=
  specimen site (`Microbiology Spec` itemid), state=growth result, finding_status=growth/no-growth. The
  vitals line (`Temp/HR/BP/RR/SpO2`) is quasi-tabular: regex-split + physiologic thresholds.
- **Path B — narrative fragmentation, LLM (Physical-Exam organ-system narrative + Radiology reports).**
  `Abd: soft, non tender, non distended` → anatomy=abdomen, attribute=tenderness/distension,
  state=soft/non-tender, finding_status=normal. This — together with radiology — is the *real*
  text-splitter task (the fragmentation prompt of Direction 2).
- **Path C — Patient History = the demand/trigger side, slot lightly.** History is symptom + PMH
  narrative (RLQ pain / anorexia / denies fever); these are symptom pieces (anatomy=RLQ, attribute=pain,
  state=present|absent) but they mostly *raise* questions rather than *cover* them — don't force
  imaging-style anatomy×attribute; keep present/absent.

## 7. Open questions to resolve next (in order)

1. **Read the 66 `assess_severity` notes** → pin down type ④ (capability) and decide whether
   `severity` and `etiology` are two distinct question types or one "post-existence" bucket.
2. **① vs ④ — same concept or two?** Should the technical "not-visualized" (①) and the modality
   "can't-answer-this-question" (④) be one `inadequate` node or two native states? (User's open
   question; drives whether we unify or split them in the rubric.)
3. **Build the `question → {organs, modalities, protocol}` table** for the 4 diseases — this is the
   concrete artifact the question-driven rubric needs; it operationalizes ②③④.
4. **Verify A3 reduces to stack operations** across the discordance corpus before designing its state.

---

## Provenance
- Corpus: `study_inadequacy.present==True` steps from `results/annotation_experiment/alarm/*.json`
  (285→185 flagged), joined by `(disease, hadm, step)` to `results/annotation_experiment/full/*.json`
  for `representative_ex_ante` (information_gap, action_role).
- Alarm pass origin: `experiments/annotation/alarm.py` + `scripts/run_alarm_pass.py` (A2
  study_inadequacy detector is blind to the doctor's order → a function of patient state, which is
  what keeps any downstream rubric non-circular).
- Loosen/brake findings this builds on: `llm_annotation_analysis.md` (authoritative 2026-06-30
  pre-admission cut); generalized principles in `configs/context_block.md`.
