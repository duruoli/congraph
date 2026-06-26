"""ALARM pass (HANDOFF §2.5) — two-call, ex-ante annotation of each decision step,
run OFFLINE over the existing Mode-A annotations + the causally-masked view.

  step 1  alarm_detect  : BLIND to the doctor's order/reasoning -> objective red-flag
                          detection (study_inadequacy A2, discordance A3) from the
                          masked pre-decision view only.
  step 2  alarm_resolve : given step-1 alarm + the doctor's ordered test + the
                          reconstructed ex-ante reasoning -> did the action RESOLVE
                          the flag (ex-ante, NEVER result-based)?

Reuses experiments.annotation.annotate.call_json (same OpenRouter client / JSON parse)
and scripts/build_masked_view.build_record for the masked view. The "no-alarm +
deviate/repeat = over-imaging" judgment is NOT here — it is deterministic and lives in
the training-set compiler (belief concentration + deviation algo + modality repeat).
"""
from __future__ import annotations

from typing import Any

from experiments.annotation.annotate import call_json
from experiments.annotation.prompts import (
    ALARM_DETECT_SYSTEM, ALARM_RESOLVE_SYSTEM,
    build_alarm_detect_user, build_alarm_resolve_user,
)


_TRIGGER_KEYS = ("study_inadequacy", "discordance")


def _ex_ante_of_step(ann_step: dict[str, Any]) -> dict[str, Any]:
    """The representative ex-ante Mode-A reconstruction stored per step."""
    ex = ann_step.get("representative_ex_ante")
    return ex if isinstance(ex, dict) else {}


def _is_present(flag: Any) -> bool:
    v = flag.get("present") if isinstance(flag, dict) else None
    return v is True or (isinstance(v, str) and v.strip().lower() in ("true", "yes", "1"))


def _any_trigger(detect: dict[str, Any]) -> bool:
    return any(_is_present(detect.get(k)) for k in _TRIGGER_KEYS)


def _detect_with_retry(baseline: dict[str, Any], dp: dict[str, Any], *, model: str,
                       attempts: int = 2) -> tuple[dict | None, str]:
    """step-1 detect with a high token budget (long grounded evidence + prior reports
    truncate at 500) and a retry — a TRUNCATED reply is unparseable JSON, which must NOT
    be silently read as 'no trigger'. Returns (parsed|None, last_raw); a dict is accepted
    only if it carries the expected trigger keys."""
    user = build_alarm_detect_user(baseline, dp.get("visible_prior_imaging", []))
    raw = ""
    for _ in range(attempts):
        r = call_json(ALARM_DETECT_SYSTEM, user, model=model, temperature=0.0, max_tokens=1000)
        raw = r["raw"]
        p = r["parsed"]
        if isinstance(p, dict) and any(k in p for k in _TRIGGER_KEYS):
            return p, raw
    return None, raw


def annotate_alarm_case(record: dict[str, Any], annotation: dict[str, Any], *,
                        model: str) -> dict[str, Any]:
    """record = build_masked_view.build_record(...); annotation = the stored full/*.json
    for the same (disease,hadm). Decision points and annotation['steps'] are in the same
    RR-N order, so we zip them."""
    baseline = record["baseline"]
    ann_steps = {s["step"]: s for s in annotation.get("steps", [])}
    steps_out = []
    for dp in record["decision_points"]:
        step = dp["step"]
        ann = ann_steps.get(step, {})
        ex_ante = _ex_ante_of_step(ann)

        # step 1: detect (blind to ordered test + reasoning)
        detect, raw = _detect_with_retry(baseline, dp, model=model)
        if detect is None:
            # parse/truncation failure — record it VISIBLY, never as a fake no-trigger
            # (a silent {} would corrupt the dataset). Re-run picks these up.
            steps_out.append({
                "step": step, "ordered": dp["ordered"],
                "n_visible_prior": len(dp.get("visible_prior_imaging", [])),
                "alarm_detect": {"_parse_error": True, "_raw": (raw or "")[:800]},
                "alarm_resolve": {"addresses_alarm": "error",
                                  "reason": "step-1 detect failed to parse",
                                  "unaddressed_alarm": None},
            })
            continue

        # gate: no trigger -> skip the step-2 API call (filter algorithmically).
        # over-imaging for these no-trigger steps is judged DETERMINISTICALLY in the
        # compiler (belief concentration + deviation algo + modality repeat).
        if _any_trigger(detect):
            # step 2: resolution quality (sees triggers + doctor action + reasoning)
            resolve = call_json(
                ALARM_RESOLVE_SYSTEM,
                build_alarm_resolve_user(detect, dp["ordered"], ex_ante),
                model=model, temperature=0.0, max_tokens=400,
            )["parsed"] or {}
        else:
            resolve = {"addresses_alarm": "not_applicable",
                       "reason": "no trigger detected — step-2 skipped",
                       "unaddressed_alarm": False, "api_skipped": True}

        steps_out.append({
            "step": step,
            "ordered": dp["ordered"],
            "n_visible_prior": len(dp.get("visible_prior_imaging", [])),
            "alarm_detect": detect,
            "alarm_resolve": resolve,
        })
    return {
        "hadm_id": record["hadm_id"],
        "disease": record["disease"],
        "n_decision_steps": record["n_decision_steps"],
        "steps": steps_out,
    }
