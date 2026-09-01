"""Compatibility helpers for the completed historical DIRECT/RECODE comparison."""
from __future__ import annotations

import json
from typing import Any

from experiments.aqc import prompts


RECODE_SYSTEM = prompts.ANNOTATION_SYSTEM + """

Recode the supplied old schema-free, ex-ante reconstruction into A/Q/C. Treat its wording as fallible source material: preserve what it supports, mark over-rationalized or internally unsupported claims as weak, and do not add facts from outside it."""


def build_recode_user(
    old_ex_ante: dict[str, Any],
    ordered: str,
    prior_aqc: dict[str, Any] | None,
) -> str:
    """Build one legacy RECODE request without current result or later events."""
    allowed = {
        key: old_ex_ante.get(key)
        for key in (
            "differential",
            "other_hypothesis",
            "information_gap",
            "expected_finding",
            "action_role",
            "appropriateness",
            "appropriateness_reason",
            "grounding",
            "reasoning",
        )
    }
    return f"""## Old schema-free ex-ante reconstruction
{json.dumps(allowed, ensure_ascii=False, indent=2)}

## Prior A/Q/C state carried from the preceding decision point
{json.dumps(prior_aqc, ensure_ascii=False) if prior_aqc else '(none; this is the first decision order)'}

## Actual current order represented by that reconstruction
{ordered}

No current result, verification label, or later event is available. Return exactly this JSON contract:
{json.dumps(prompts.output_contract(), ensure_ascii=False, indent=2)}"""
