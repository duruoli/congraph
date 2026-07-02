<!-- Context injected into the ctx-* eval arms (ctx-qwen, ctx-sonnet). NOT used by base/SFT.
Distilled from llm_annotation_analysis.md (authoritative, 2026-06-30 pre-admission real-charttime
cut: deviate+confirmed loosening insights; disconfirm-group brake findings). Deliberately kept
GENERALIZED — principles only, no specific modality prescriptions, quotes, or case examples, so the
context arms reason from the pattern rather than copying instances. A2 = the study_inadequacy alarm
channel, A3 = the discordance alarm channel (the eval's alarm-landing anchor). Edit this file to
change what the context arms know; the eval script reads it verbatim. -->

# Clinical override knowledge (apply ON TOP of the rubric)

The rubric routes imaging by DISEASE and treats a completed first-line study as the end of the
question. Its blind spots: it does not notice that the diagnosis is already settled and the useful
question has moved on; it trusts a study it never quality-checked; and it reads one data axis at a
time. Below are the situations where deviating from the rubric is repeatedly right (Section A,
"the rubric stopped reasoning too early") and the symmetric situations where a negative result or a
second study is not actually informative (Section B, the brake). Default to the rubric; deviate
only when a pattern below clearly fits, and name which one.

## A. When to LOOSEN — the rubric stopped reasoning too early

1. **The diagnosis is already settled, so the question has moved on.** Once the disease question
   is answered, do not re-order a study to re-confirm it. Route by the question that is still open
   — severity, etiology, or complications — which may call for a different modality than the
   disease-first one the rubric names. Route by the current clinical question, not the disease
   label.

2. **A non-diagnostic or low-quality study is not a negative.** (The study_inadequacy alarm.) The
   rubric treats a completed first-line test as terminal because it cannot judge study QUALITY.
   When the prior study was inconclusive or technically limited, the question is still open —
   proceed to the next appropriate study rather than accepting the inadequate result as an answer.
   Absence of evidence from a test that could not have seen the finding is not evidence of absence.

3. **Discordance between two data streams is itself the alarm.** (The discordance alarm.) When two
   information streams disagree, trust the mismatch over the rubric's single-axis logic and image
   to resolve the conflict. Two sub-cases: when one stream is overwhelming, keep the diagnosis and
   re-attribute the mismatch to severity, etiology, or a known test limitation; when both streams
   are ambiguous and conflicting, let localization collapse and broaden the differential rather
   than staying falsely confident.

## B. When to BRAKE — a negative is not a failure, one adequate look is enough

1. **Rule-out-by-design: the negative is the product, not a surprise.** Distinguish a test meant to
   CONFIRM the leading diagnosis from a test meant to EXCLUDE an actionable alternative. For the
   latter, a clean negative is the informative, management-changing result. Pre-register a high
   probability of a negative, read that negative as success, and lower the etiology belief — not
   decision-confidence. Once the question has been cleanly answered once, stop re-escalating it.

2. **One adequate study is enough — suppress a confirmatory second, weaker study.** This is the
   brake, and it is narrow: it is not "stop scanning" in general, since legitimate severity-staging
   continues. Once one adequate study has answered the actionable question, do not re-ask the same
   question with a weaker modality.

3. **Do not suppress the broad search that overturns the working diagnosis.** A wide study that
   disconfirms the leading hypothesis but reveals the real pathology is a good action. Judge it by
   its aim, not by luck: it counts when the deviation was locally appropriate and aimed at the
   region or question where the truth was found, not when the truth was an incidental catch. Judge
   the action, not whether the prediction held.

## How to use this

- Default to the rubric's recommended modality and action.
- Loosen only when an A-pattern clearly fits: question-moved, inadequate-study, or discordance —
  and name which one.
- Before adding a test after a negative or equivocal result, check the B-patterns: a
  rule-out-by-design negative (calibrate, lower the etiology belief, do not re-escalate), an
  already-answered question you would only re-ask with a weaker modality (suppress it), or a
  genuinely open question from an inadequate study (proceed — that is A2).
- Ground every deviation in specific findings from the patient state.
