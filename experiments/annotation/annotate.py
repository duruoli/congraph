"""Run 思路2 Mode-A reconstruction (with ensemble) + 思路1 verification over the
causally-masked decision sequence of one patient.

Reuses scripts/build_masked_view.build_record for the masked view and
experiments.llm_experiment.env_loader for the OpenRouter key.
"""
from __future__ import annotations

import json
import re
from statistics import mean, pstdev
from typing import Any

from openai import OpenAI

from experiments.annotation.prompts import (
    BRANCHES, MODE_A_SYSTEM, VINDICATION_SYSTEM,
    build_mode_a_user, build_verification_user,
)
from experiments.llm_experiment.env_loader import load_openrouter_key

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_CLIENT: OpenAI | None = None
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        # Short timeout so a request routed to a congested OpenRouter provider fails
        # fast and retries onto a fresh (usually healthy ~1s) provider, instead of
        # hanging the full 90s. A few quick retries beats one long stall.
        _CLIENT = OpenAI(api_key=load_openrouter_key(), base_url=_OPENROUTER_BASE_URL,
                         max_retries=3, timeout=45.0)
    return _CLIENT


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _parse(text: str) -> dict | None:
    raw = _strip_fences(text or "")
    m = _JSON_OBJ_RE.search(raw)
    for cand in (raw, m.group(0) if m else None):
        if not cand:
            continue
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def call_json(system: str, user: str, *, model: str, temperature: float,
              max_tokens: int) -> dict[str, Any]:
    resp = _client().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature, max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    msg = resp.choices[0].message.content or ""
    parsed = _parse(msg)
    return {"parsed": parsed, "raw": msg}


def _norm_diff(d: dict | None) -> dict[str, float]:
    """Coerce a model differential into a normalized vector over BRANCHES."""
    out = {b: 0.0 for b in BRANCHES}
    if isinstance(d, dict):
        for b in BRANCHES:
            try:
                out[b] = max(0.0, float(d.get(b, 0.0)))
            except (TypeError, ValueError):
                out[b] = 0.0
    s = sum(out.values())
    if s > 0:
        out = {b: v / s for b, v in out.items()}
    return out


def _disagreement(diffs: list[dict[str, float]], roles: list[str],
                  approps: list[str]) -> dict[str, Any]:
    """Ensemble-disagreement metrics = honest uncertainty proxy (HANDOFF §1.6)."""
    n = len(diffs)
    mean_diff = {b: mean(d[b] for d in diffs) for b in BRANCHES} if n else {b: 0.0 for b in BRANCHES}
    # mean per-branch std across samples (higher = more disagreement)
    branch_std = {b: (pstdev([d[b] for d in diffs]) if n > 1 else 0.0) for b in BRANCHES}
    diff_disagreement = mean(branch_std.values()) if branch_std else 0.0
    # top-branch consistency
    def top(d): return max(d, key=d.get)
    overall_top = top(mean_diff) if mean_diff else None
    top_consistency = (sum(1 for d in diffs if top(d) == overall_top) / n) if n else 0.0
    # modal action_role consistency
    def modal(xs):
        return max(set(xs), key=xs.count) if xs else None
    role_modal = modal(roles)
    role_consistency = (sum(1 for r in roles if r == role_modal) / n) if n else 0.0
    return {
        "mean_differential": mean_diff,
        "branch_std": branch_std,
        "diff_disagreement": diff_disagreement,
        "overall_top_branch": overall_top,
        "top_branch_consistency": top_consistency,
        "modal_action_role": role_modal,
        "action_role_consistency": role_consistency,
        "appropriateness_dist": {v: approps.count(v) for v in set(approps)} if approps else {},
    }


def annotate_case(record: dict[str, Any], *, model: str, n_samples: int = 1,
                  ensemble_temp: float = 0.2) -> dict[str, Any]:
    baseline = record["baseline"]
    steps_out = []
    for dp in record["decision_points"]:
        user = build_mode_a_user(dp, baseline)
        samples = []
        for _ in range(n_samples):
            r = call_json(MODE_A_SYSTEM, user, model=model,
                          temperature=ensemble_temp, max_tokens=1500)
            samples.append(r["parsed"] or {"_unparsed": r["raw"]})
        diffs = [_norm_diff(s.get("differential")) for s in samples]
        roles = [str(s.get("action_role", "")) for s in samples]
        approps = [str(s.get("appropriateness", "")) for s in samples]
        metrics = _disagreement(diffs, roles, approps)

        # verification uses the masked result of THIS step + the modal-ish ex-ante guess
        rep = dp.get("masked_result_of_this_test", "")
        # pick the sample whose differential is closest to the ensemble mean as the representative ex-ante
        def dist(d): return sum(abs(d[b] - metrics["mean_differential"][b]) for b in BRANCHES)
        rep_idx = min(range(len(diffs)), key=lambda i: dist(diffs[i])) if diffs else 0
        rep_sample = samples[rep_idx] if samples else {}
        vind = call_json(
            VINDICATION_SYSTEM,
            build_verification_user(
                str(rep_sample.get("expected_finding", "")),
                str(rep_sample.get("information_gap", "")),
                rep,
            ),
            model=model, temperature=0.0, max_tokens=300,
        )["parsed"]

        steps_out.append({
            "step": dp["step"],
            "ordered": dp["ordered"],
            "n_visible_prior": len(dp.get("visible_prior_imaging", [])),
            "ensemble": samples,
            "representative_ex_ante": rep_sample,
            "metrics": metrics,
            "verification": vind,
        })
    return {
        "hadm_id": record["hadm_id"],
        "disease": record["disease"],
        "n_decision_steps": record["n_decision_steps"],
        "radiology_order": record["radiology_order"],
        "steps": steps_out,
    }
