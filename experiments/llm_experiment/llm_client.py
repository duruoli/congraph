"""Thin wrapper around OpenRouter chat completions for next-test recommendation.

Returns (next_test, reasoning, raw_response). On parse failure returns
("UNPARSEABLE", raw_text, raw_response).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openai import OpenAI

from experiments.llm_experiment.env_loader import load_openrouter_key
from experiments.llm_experiment.prompts import SYSTEM_PROMPT

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_CLIENT: OpenAI | None = None


def _client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        api_key = load_openrouter_key()
        # max_retries handles 429/5xx with exponential backoff inside SDK.
        _CLIENT = OpenAI(
            api_key=api_key,
            base_url=_OPENROUTER_BASE_URL,
            max_retries=5,
            timeout=60.0,
        )
    return _CLIENT


@dataclass
class LLMCallResult:
    next_test: str
    reasoning: str
    raw: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    follows_rubric: bool | None = None   # set only for llm_rubric condition
    deviation_reason: str = ""


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _try_parse_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def call_llm(
    *,
    user_prompt: str,
    model: str = "anthropic/claude-sonnet-4-6",
    temperature: float = 0.0,
    max_tokens: int = 400,
) -> LLMCallResult:
    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    msg = resp.choices[0].message.content or ""
    parsed = _try_parse_json(msg)
    pt = getattr(resp.usage, "prompt_tokens", 0) or 0
    ct = getattr(resp.usage, "completion_tokens", 0) or 0
    if not parsed or "next_test" not in parsed:
        return LLMCallResult(
            next_test="UNPARSEABLE", reasoning=msg[:500], raw=msg, model=model,
            prompt_tokens=pt, completion_tokens=ct,
        )
    follows_rubric_raw = parsed.get("follows_rubric")
    follows_rubric = bool(follows_rubric_raw) if follows_rubric_raw is not None else None
    deviation_reason = str(parsed.get("deviation_reason", ""))[:500]
    return LLMCallResult(
        next_test=str(parsed["next_test"]),
        reasoning=str(parsed.get("reasoning", ""))[:1000],
        raw=msg,
        model=model,
        prompt_tokens=pt,
        completion_tokens=ct,
        follows_rubric=follows_rubric,
        deviation_reason=deviation_reason,
    )
