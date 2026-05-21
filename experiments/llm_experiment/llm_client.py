"""Thin wrapper around OpenAI chat completions for next-test recommendation.

Returns (next_test, reasoning, raw_response). On parse failure returns
("UNPARSEABLE", raw_text, raw_response).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openai import OpenAI

from experiments.llm_experiment.env_loader import load_openai_key
from experiments.llm_experiment.prompts import SYSTEM_PROMPT


_CLIENT: OpenAI | None = None


def _client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        load_openai_key()
        # max_retries handles 429/5xx with exponential backoff inside SDK.
        _CLIENT = OpenAI(max_retries=5, timeout=60.0)
    return _CLIENT


@dataclass
class LLMCallResult:
    next_test: str
    reasoning: str
    raw: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


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
    model: str = "gpt-4o",
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
    if not parsed or "next_test" not in parsed:
        return LLMCallResult(
            next_test="UNPARSEABLE",
            reasoning=msg[:500],
            raw=msg,
            model=model,
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        )
    return LLMCallResult(
        next_test=str(parsed["next_test"]),
        reasoning=str(parsed.get("reasoning", ""))[:1000],
        raw=msg,
        model=model,
        prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
    )
