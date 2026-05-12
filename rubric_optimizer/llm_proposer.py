"""llm_proposer.py — Calls GPT-4o (or other model) to propose a rubric change.

propose_change(summary, client, model) → ChangeRecord | None

The summary passed in is the full output from summarizer.build_summary(),
which already contains the failure analysis, source code snippets, and the
INSTRUCTION block telling the LLM exactly what format to use.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import yaml

_PKG  = Path(__file__).parent
_ROOT = _PKG.parent
for _p in [str(_ROOT), str(_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trial_log import ChangeRecord

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert in clinical decision support system optimization.

You will receive a rubric optimizer trial summary that includes:
  - Current diagnostic accuracy metrics per disease
  - Failure analysis for the worst-performing disease
  - The CURRENT Python source code of the rubric and scoring functions
    (this is what the file looks like RIGHT NOW — it may differ from previous iterations)
  - An INSTRUCTION block at the end telling you the EXACT output format

Your job: propose ONE specific, targeted code change to improve the
diagnostic accuracy of the failing disease.

CRITICAL RULES FOR old_code:
  • old_code MUST be a substring of the code shown in the "CURRENT RUBRIC CODE" section.
  • Read the current code carefully — previous iterations may have already changed it.
  • If you see that a change you considered was already applied, propose something DIFFERENT.
  • Do NOT propose replacing code that is no longer in the file.

CRITICAL OUTPUT RULES:
1. Output ONLY the change block — start with --- on its own line, end with --- on its own line.
2. Do NOT include any explanation, commentary, or text outside the --- delimiters.
3. The old_code field MUST be copied EXACTLY from the current source code shown above
   (character-for-character, including whitespace, quotes, and comments).
4. Do NOT copy the "OR ..." placeholder text into field values — pick one specific value.
5. Keep old_code and new_code as YAML block scalars using the | character,
   indented by exactly 2 spaces relative to the key.
6. Do NOT change both files in one proposal — pick one file only.
7. When the rubric edge for the worst disease looks unchanged, consider editing
   scoring parameters in diagnosis_distribution.py instead (W_RUBRIC, W_EMPIRICAL,
   W_TRIAGE, CONFIRMED_BONUS, DEPTH_CAP, TRIAGE_PRIOR_RATIO).
"""

# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_ALLOWED_FILES = {"rubric_graph.py", "diagnosis_distribution.py"}


def _strip_or_placeholder(text: str) -> str:
    """Remove trailing ' OR ...' from a field value (LLM may copy template verbatim)."""
    return re.sub(r'\s+OR\s+.*', '', text).strip()


def _parse_yaml_block(text: str) -> Optional[dict]:
    """Extract and parse the YAML block between --- markers."""
    # Match content between first --- and closing ---
    match = re.search(r'---\s*\n(.*?)(?:\n---|\Z)', text, re.DOTALL)
    if not match:
        return None
    content = match.group(1)
    try:
        parsed = yaml.safe_load(content)
        if isinstance(parsed, dict):
            return parsed
        print(f"  [Parser] YAML parsed to non-dict type: {type(parsed)}")
    except yaml.YAMLError as e:
        print(f"  [Parser] YAML error: {e}")
    return None


def _parse_response(raw: str) -> Optional[ChangeRecord]:
    """Parse LLM response text into a ChangeRecord. Returns None on failure."""
    parsed = _parse_yaml_block(raw)
    if parsed is None:
        return None

    required = {"target_file", "change_type", "description", "old_code", "new_code", "rationale"}
    missing = required - set(parsed.keys())
    if missing:
        print(f"  [Parser] Missing keys: {missing}")
        return None

    target_file = _strip_or_placeholder(str(parsed["target_file"]))
    change_type = _strip_or_placeholder(str(parsed["change_type"]))

    if target_file not in _ALLOWED_FILES:
        print(f"  [Parser] Invalid target_file: {target_file!r} (must be one of {_ALLOWED_FILES})")
        return None

    old_code = str(parsed["old_code"])
    new_code = str(parsed["new_code"])
    if not old_code.strip():
        print("  [Parser] old_code is empty")
        return None

    return ChangeRecord(
        target_file = target_file,
        change_type = change_type,
        description = str(parsed["description"]).strip(),
        old_code    = old_code,
        new_code    = new_code,
        rationale   = str(parsed["rationale"]).strip(),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def propose_change(
    summary:     str,
    client,
    model:       str = "gpt-4o",
    max_retries: int = 3,
) -> Optional[ChangeRecord]:
    """
    Send the trial summary to the LLM and parse its proposed change.

    Parameters
    ----------
    summary     : full text from summarizer.build_summary() — includes metrics,
                  failure analysis, rubric source code, and the INSTRUCTION block.
    client      : openai.OpenAI instance
    model       : model name (default: "gpt-4o")
    max_retries : retry up to this many times if the response cannot be parsed

    Returns
    -------
    ChangeRecord if successful, None if all retries are exhausted.
    """
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": summary},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model       = model,
                messages    = messages,
                temperature = 0.3,
                max_tokens  = 2048,
            )
            raw_text = response.choices[0].message.content or ""

            print(f"\n{'─'*60}")
            print(f"  LLM RESPONSE (attempt {attempt}/{max_retries})")
            print(f"{'─'*60}")
            print(raw_text)
            print(f"{'─'*60}")

            change = _parse_response(raw_text)
            if change is not None:
                return change

            # Parsing failed — ask the LLM to fix its output
            print(f"  [Parser] Attempt {attempt}: parse failed. Asking LLM to retry…")
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({
                "role": "user",
                "content": (
                    "Your response could not be parsed into the required format.\n"
                    "Please output ONLY the change block, strictly following this template:\n"
                    "---\n"
                    "target_file:  rubric_graph.py\n"
                    "change_type:  edge_condition\n"
                    "description:  <one sentence>\n"
                    "old_code: |\n"
                    "  <exact lines from the source code, indented 2 spaces>\n"
                    "new_code: |\n"
                    "  <replacement lines, indented 2 spaces>\n"
                    "rationale: |\n"
                    "  <2-4 sentences>\n"
                    "---\n"
                    "Start your response with --- on the very first line."
                ),
            })

        except Exception as e:
            print(f"  [LLM API] Attempt {attempt}/{max_retries} failed: {e}")

    print("  [LLM] All retry attempts exhausted — no valid change proposed.")
    return None
