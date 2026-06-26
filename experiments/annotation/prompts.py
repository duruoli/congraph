"""Prompts for 思路2 Mode-A doctor-reasoning reconstruction and 思路1 verification.

Design (HANDOFF §1, annotation_agent_design memory):
  - Mode A = explain the GIVEN doctor action (the ordered test is provided), NEVER
    let the model pick the next test. Anchoring on the real action suppresses the
    LLM's textbook/rubric attachment.
  - RUBRIC-FREE: no rubric, no disease label, no outcome in the input. The model
    plays an experienced acute-care physician using general clinical knowledge.
  - Differential = 5 triage branches + an open "other" slot (Figure 0).
  - Extractive grounding: every claim must cite a concrete field from the visible context.
  - Verification (思路1) is a SEPARATE ex-post call that DOES see the masked result,
    used as a reward/quality label only — never fed back into Mode A.
"""
from __future__ import annotations

import json
from typing import Any

# 5 triage branches + open 'other' (Figure 0). The bile-DUCT axis (biliary) is NOT a
# forced differential option here: making it compete in the softmax over-attributed
# established pancreatitis (the LLM rationalised any US/MRCP order as "biliary intent",
# stealing mass from a 0.75-confident pancreatitis without hallucinating — a forced-choice
# artifact). Instead biliary is recovered POST-HOC as a sub-label of off_rubric steps from
# other_hypothesis text — see experiments/annotation/deviation.py derived_biliary().
BRANCHES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis", "other"]

MODE_A_SYSTEM = """You are an experienced acute-care physician (emergency medicine + general surgery) reconstructing, step by step, the clinical reasoning of ANOTHER physician who evaluated a patient with acute abdominal pain.

You are shown, for one decision point: the baseline workup (history, exam, labs), the results of any imaging that was already done BEFORE this point, and the ONE imaging test the treating physician chose to order next. You do NOT see the result of that test or anything that happened afterward.

Your job is NOT to decide what test you would order. The order is a fact. Your job is to explain, in the treating physician's shoes, the belief state and information gap that make THIS order rational at THIS moment — using only general clinical knowledge.

Hard rules:
- Use ONLY the visible context. Never use hindsight, final diagnosis, or any guideline/rubric checklist.
- Every clinical claim must be grounded in a concrete field you can quote from the visible context. If you cannot quote support, do not assert it.
- Be honest about uncertainty. If the order looks weakly motivated or like a wide net, say so — do not invent a crisp rationale.
- Output STRICT JSON only, no prose outside the JSON object."""


def _schema_block() -> str:
    return """Return a JSON object with exactly these keys:
{
  "differential": {"appendicitis": <0-1>, "cholecystitis": <0-1>, "diverticulitis": <0-1>, "pancreatitis": <0-1>, "other": <0-1>},
      // your estimate of the treating physician's belief BEFORE this test; the five numbers should sum to ~1.0
  "other_hypothesis": "<if 'other' is non-trivial, the specific non-listed cause(s) suspected (e.g. gynecologic, urologic, bowel obstruction, biliary-duct obstruction/choledocholithiasis); else empty>",
  "information_gap": "<the one thing the physician is uncertain about that THIS order is meant to resolve>",
  "expected_finding": "<the concrete finding on THIS test the physician is likely hoping/expecting to see or rule out (an ex-ante prediction)>",
  "action_role": "<one of: rule_in | rule_out | broaden_search | assess_severity | localize_source>",
  "appropriateness": "<yes | partial | no> — does this order address a real, stated gap given the visible context?",
  "appropriateness_reason": "<one sentence>",
  "grounding": ["<field>: <quoted value>", "..."],   // concrete evidence you used
  "reasoning": "<1-3 sentences, in the treating physician's shoes>"
}"""


def build_mode_a_user(step: dict[str, Any], baseline: dict[str, Any]) -> str:
    """One Mode-A decision-point prompt from a build_masked_view decision_point dict."""
    prior = step.get("visible_prior_imaging", [])
    if prior:
        prior_txt = "\n\n".join(
            f"[Prior imaging {i+1}: {p['modality']} {p['region']} ({p['exam']})]\n{p['report']}"
            for i, p in enumerate(prior)
        )
    else:
        prior_txt = "(none — only the baseline workup below was available)"

    return f"""## Baseline workup (always visible)

[History]
{baseline['patient_history']}

[Physical examination]
{baseline['physical_examination']}

[Laboratory tests]
{baseline['laboratory_tests']}

## Imaging already resulted BEFORE this decision point
{prior_txt}

## The order to explain
At this point the treating physician ordered: **{step['ordered']}**
(You do NOT see its result.)

{_schema_block()}"""


VINDICATION_SYSTEM = """You are auditing, after the fact, whether one imaging test confirmed the treating physician's prior expectation. You are given: the physician's ex-ante expected finding for the test, and the ACTUAL report text of that test. Judge LOCALLY — only whether the result matched what the physician was looking for at that step, NOT whether the final diagnosis was right. Output STRICT JSON only."""


def build_verification_user(expected_finding: str, information_gap: str, actual_report: str) -> str:
    return f"""## Physician's ex-ante expectation for this test
information_gap: {information_gap}
expected_finding: {expected_finding}

## Actual report of the test
{actual_report}

Return a JSON object with exactly these keys:
{{
  "verification": "<confirmed | disconfirmed | uninformative>",
      // confirmed = result matched the expected_finding / resolved the gap as hoped
      // disconfirmed = result contradicted the expected_finding
      // uninformative = result did not address the gap either way
  "actual_finding": "<one-sentence summary of what the report showed>",
  "relation_to_expectation": "<one sentence: how the actual finding relates to expected_finding>",
  "certainty_update": "<up | down | flat>"   // direction the physician's certainty should move
}}"""


# ---------------------------------------------------------------------------
# ALARM pass (HANDOFF §2.5) — two SEPARATE calls, both EX-ANTE (never see THIS
# step's result). The split is for INDEPENDENCE: step 1 is given ONLY the masked
# chart (no order, no reasoning) so detection can't be contaminated by the doctor's
# own words; step 2 then judges resolution given the order + reasoning.
# Two red-flag types only: study_inadequacy (A2), discordance (A3) — these are the
# genuinely OBJECTIVE, blind-detectable flags (exist in the chart regardless of the
# doctor). A1 (advanced_question: dx already established, severity/etiology question
# remains) is NOT here: it is a property of the reconstructed BELIEF, not an external
# flag, so it has no independent blind detection and its resolution = Mode-A's own
# `appropriateness`. A1 is therefore DERIVED ALGORITHMICALLY in the compiler
# (belief concentrated on a rubric disease + severity/etiology gap), not LLM-detected.
# The "no-trigger + deviate/repeat = over-imaging" flag is likewise deterministic.
# ---------------------------------------------------------------------------

ALARM_DETECT_SYSTEM = """You are a physician reading a patient's chart partway through a workup for acute abdominal pain. Identify whether the information visible so far contains either of two red-flag situations that should make a careful physician hesitate before defaulting to the routine next step:

- study_inadequacy: a prior imaging study was technically limited / non-diagnostic / did not visualize the organ in question, so the clinical question it was meant to answer is still open (an inadequate study is not a negative result).
- discordance: a MARKED conflict between two streams of evidence that the leading suspected diagnosis CANNOT explain — e.g. grossly abnormal labs/vitals (very high WBC, elevated lactate, hemodynamic instability) with a benign abdomen, or imaging findings out of proportion to the clinical picture. Do NOT flag findings that are EXPECTED for / consistent with the suspected diagnosis (mild-to-moderate leukocytosis with focal tenderness in appendicitis; high lipase with epigastric pain in pancreatitis), nor normal labs, nor an exam that merely evolved between two timepoints. The conflict must be striking enough to change management.

Rules:
- Every positive finding must be GROUNDED in a concrete quote from the visible context; if you cannot quote support, present=false.
- A routine, coherent workup — no contradiction and no inadequate prior study — has neither. Report them absent; do not manufacture one.
- Output STRICT JSON only."""


def build_alarm_detect_user(baseline: dict[str, Any], visible_prior: list[dict[str, Any]]) -> str:
    if visible_prior:
        prior_txt = "\n\n".join(
            f"[Prior imaging {i+1}: {p['modality']} {p['region']} ({p['exam']})]\n{p['report']}"
            for i, p in enumerate(visible_prior)
        )
    else:
        prior_txt = "(none — only the baseline workup below is available)"
    return f"""## Baseline workup
[History]
{baseline['patient_history']}

[Physical examination]
{baseline['physical_examination']}

[Laboratory tests]
{baseline['laboratory_tests']}

## Imaging already resulted SO FAR
{prior_txt}

Return a JSON object with exactly these keys:
{{
  "study_inadequacy": {{"present": <true|false>, "score": <0-1>, "evidence": "<quoted field, or empty>"}},
  "discordance": {{"present": <true|false>, "score": <0-1>, "evidence": "<quoted field, or empty>"}},
  "summary": "<one sentence on the salient red flag, or 'none'>"
}}"""


ALARM_RESOLVE_SYSTEM = """You are checking whether a physician's next imaging order — together with the reconstructed reasoning behind it — adequately handles a situation that was flagged in the chart. Weigh TWO things together:

1. Capability — can the ordered TEST actually resolve the situation:
   - study_inadequacy: it can image what the inadequate prior study failed to — the right modality / region / technique for the still-open question.
   - discordance: it can adjudicate the conflict — a result that would tip the balance one way or the other, not leave both readings equally possible.

2. Intent — does the REASONING (information_gap / expected_finding / action_role / reasoning) show the physician was actually pursuing that situation. The reasoning is holistic and need NOT name the flag explicitly; it is enough that its stated aim encompasses the situation. If the test is mechanically capable but the reasoning is aimed elsewhere and only covers the flag incidentally, that is `partial`, not `yes`.

If several situations were flagged, judge against the MOST important one; a test that handles one but ignores a more pressing one is at best `partial` with `unaddressed_alarm=true`.

Judge EX-ANTE: an appropriate test that later returns negative still handles the flag well — never judge by the result. Output STRICT JSON only."""


def build_alarm_resolve_user(alarm: dict[str, Any], ordered: str, ex_ante: dict[str, Any]) -> str:
    return f"""## Red flags detected in the chart (from an independent reader)
study_inadequacy: {json.dumps(alarm.get('study_inadequacy', {}), ensure_ascii=False)}
discordance: {json.dumps(alarm.get('discordance', {}), ensure_ascii=False)}
summary: {alarm.get('summary', '')}

## What the treating physician did next
ordered: {ordered}
information_gap: {ex_ante.get('information_gap', '')}
expected_finding: {ex_ante.get('expected_finding', '')}
action_role: {ex_ante.get('action_role', '')}
reasoning: {ex_ante.get('reasoning', '')}

Return a JSON object with exactly these keys:
{{
  "addresses_alarm": "<yes | partial | no>",
      // yes = the order + reasoning covers/resolves the flagged situation(s)
  "reason": "<one sentence, ex-ante, not result-based>",
  "unaddressed_alarm": <true|false>   // a flagged situation is left unhandled by the order
}}"""
