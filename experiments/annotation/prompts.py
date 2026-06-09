"""Prompts for 思路2 Mode-A doctor-reasoning reconstruction and 思路1 vindication.

Design (HANDOFF §1, annotation_agent_design memory):
  - Mode A = explain the GIVEN doctor action (the ordered test is provided), NEVER
    let the model pick the next test. Anchoring on the real action suppresses the
    LLM's textbook/rubric attachment.
  - RUBRIC-FREE: no rubric, no disease label, no outcome in the input. The model
    plays an experienced acute-care physician using general clinical knowledge.
  - Differential = 5 triage branches + an open "other" slot (Figure 0).
  - Extractive grounding: every claim must cite a concrete field from the visible context.
  - Vindication (思路1) is a SEPARATE ex-post call that DOES see the masked result,
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


def build_vindication_user(expected_finding: str, information_gap: str, actual_report: str) -> str:
    return f"""## Physician's ex-ante expectation for this test
information_gap: {information_gap}
expected_finding: {expected_finding}

## Actual report of the test
{actual_report}

Return a JSON object with exactly these keys:
{{
  "vindication": "<confirmed | disconfirmed | uninformative>",
      // confirmed = result matched the expected_finding / resolved the gap as hoped
      // disconfirmed = result contradicted the expected_finding
      // uninformative = result did not address the gap either way
  "actual_finding": "<one-sentence summary of what the report showed>",
  "relation_to_expectation": "<one sentence: how the actual finding relates to expected_finding>",
  "certainty_update": "<up | down | flat>"   // direction the physician's certainty should move
}}"""
