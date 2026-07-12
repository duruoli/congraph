# Rubric Update — a question-driven redesign

> Working design doc. Motivation: the "wrap the rubric with alarm-override rules" patch (A2/A3/A1
> as external booleans) is too shallow — it treats the rubric's blind spots as exceptions bolted on
> top. This doc explores the deeper move the analysis points to: **rebuild the rubric so that
> inadequacy, ask-the-right-question, and (tentatively) discordance are native consequences of the
> rubric's own state, not add-ons.** Grounded in the `study_inadequacy` corpus (185 flagged steps,
> `results/annotation_experiment/alarm/*.json` joined to `full/*.json`).

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
- question-tree outcomes (vindication ±) → **update belief mass**.
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

## 6. Open questions to resolve next (in order)

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
