"""Plan-T RL reward (venue-independent) for the c+RL and b-NEW arms.

RLVR reward = did the model's predicted deviation match the OBSERVED label. The optimization
target is the CoT the model emits BEFORE the answer; this function only scores the final answer
token against the gold label. Pure stdlib — the Tinker RL env (a ProblemEnv subclass adapted from
tinker_cookbook/recipes/math_rl/math_env.py) imports `deviation_reward` and returns it per rollout.

Design (see HANDOFF_pred_dev.md RL notes):
- reward is BINARY correctness (1.0 if predicted == gold, else 0.0), + a small format bonus so the
  model reliably emits a parseable answer (mirrors math_env's format_coef). Keep format_coef small
  so it doesn't dominate the correctness signal.
- gold label: meta.y (1 = {deviate, off_rubric}, 0 = follow) from the SAME cls/cls_reason rows.
- ⚠️ CALIBRATION: this 0/1 reward pushes the policy toward overconfidence — expected, flagged in the
  handoff. The calibrated prob is recovered post-hoc by Platt in eval_prob_tinker.py, not here.
"""
from __future__ import annotations

import re

_POS = "deviate"
_NEG = "follow"


def parse_prediction(completion: str, answer_cue: str | None = None) -> str | None:
    """Pull the predicted answer word from a generated completion.
    Returns 'deviate'|'follow'|None.

    Precedence:
      1. answer_cue (b-NEW free reasoning, e.g. "\\n\\nAnswer: "): the word right after the LAST
         cue occurrence. This is required because free reasoning ABOUT deviating sprinkles the
         words 'deviate'/'follow' through the prose, so the bare-word fallback (step 3) would grab
         a mid-reasoning mention instead of the committed answer.
      2. JSON `"deviation": "deviate"` tail (arm c / structured).
      3. fallback: the LAST standalone occurrence of either word."""
    if answer_cue:
        idx = completion.rfind(answer_cue)
        if idx != -1:
            m = re.match(r'\s*"?(follow|deviate)"?', completion[idx + len(answer_cue):], re.I)
            if m:
                return m.group(1).lower()
    m = re.search(r'"deviation"\s*:\s*"(follow|deviate)"', completion)
    if m:
        return m.group(1)
    # fall back to the LAST standalone occurrence of either word
    hits = re.findall(r'\b(follow|deviate)\b', completion)
    return hits[-1] if hits else None


def deviation_reward(completion: str, gold_y: int, format_coef: float = 0.1,
                     answer_cue: str | None = None) -> float:
    """gold_y: 1 = deviate/off_rubric (positive), 0 = follow. Returns reward in [0, 1+format_coef]."""
    pred = parse_prediction(completion, answer_cue)
    if pred is None:
        return 0.0                      # unparseable -> no format bonus, no correctness
    gold_word = _POS if gold_y == 1 else _NEG
    correct = 1.0 if pred == gold_word else 0.0
    return correct + format_coef        # format bonus for producing a parseable answer at all


# ------------------------- b-NEW (Package A) composite reward -------------------------
# Algorithmic, NO LLM judge. Reward = deviation-correctness (dominant) + PRESENCE of the target
# reasoning parts in the FREE text (closed-vocab keyword hit, NOT token imitation) + a small format
# bonus for a parseable answer - a soft length penalty. The prompt does NOT name the parts, so their
# presence VARIES across rollouts -> the structure term has in-group variance for GRPO to climb.
# Closed vocabularies, calibrated to the surface forms the base actually writes (see the pre-RL
# trace audit). Word-boundary, case-insensitive.
_MODALITY_PAT = re.compile(
    r'(?:\bct\b|ct[_ ]?abd|computed tomog|ultrasound|\bus\b|sonograph|\bmri\b|mrcp|hida|'
    r'x-?ray|radiograph|\bkub\b|ercp)', re.I)
_RUBRIC_PAT = re.compile(r'(?:rubric|recommend|guideline|first[- ]?line|standard of care)', re.I)
_HYP_PAT = re.compile(
    r'(?:appendicitis|cholecystitis|diverticulitis|pancreatitis|\bhypothes|differential|suspect)', re.I)

_STRUCTURE_PATS = {"modality": _MODALITY_PAT, "rubric": _RUBRIC_PAT, "hypothesis": _HYP_PAT}


def structure_components(completion: str, answer_cue: str | None = None) -> dict[str, bool]:
    """Which target reasoning parts appear in the REASONING text (before the answer cue, so the
    mandatory 'Answer:' line can't be what satisfies a component). All closed-vocab, algorithmic."""
    body = completion
    if answer_cue and answer_cue in completion:
        body = completion.rsplit(answer_cue, 1)[0]
    return {name: bool(pat.search(body)) for name, pat in _STRUCTURE_PATS.items()}


def bnew_reward(completion: str, gold_y: int, *, answer_cue: str = "\n\nAnswer: ",
                w_struct: float = 0.3, w_format: float = 0.1, w_len: float = 0.0002,
                len_budget: int = 512, len_pen_cap: float = 0.2, n_tokens: int | None = None,
                return_breakdown: bool = False):
    """Composite b-NEW reward. correctness dominates (gap 1.0) > structure (<=w_struct) so the model
    can never trade a right answer for keyword-stuffing. n_tokens: true rollout length from the
    sampler; if None, approximated as len(chars)/4 (fine for offline audit, pass the real count in RL).
    return_breakdown=True -> (total, dict) for auditing; else -> float total."""
    pred = parse_prediction(completion, answer_cue)
    gold_word = _POS if gold_y == 1 else _NEG
    correct = 1.0 if (pred is not None and pred == gold_word) else 0.0
    comps = structure_components(completion, answer_cue)
    struct = sum(comps.values()) / len(comps)          # fraction of parts present, in [0,1]
    fmt = 1.0 if pred is not None else 0.0
    if n_tokens is None:
        n_tokens = len(completion) // 4
    overflow = max(0, n_tokens - len_budget)
    len_pen = min(len_pen_cap, w_len * overflow)        # soft, capped so it never dominates
    total = correct + w_struct * struct + w_format * fmt - len_pen
    if return_breakdown:
        return total, {"pred": pred, "correct": correct, "components": comps, "struct": struct,
                       "format": fmt, "n_tokens": n_tokens, "overflow": overflow,
                       "len_pen": round(len_pen, 4), "total": round(total, 4)}
    return total


if __name__ == "__main__":  # tiny self-test
    CUE = "\n\nAnswer: "
    assert parse_prediction('... "deviation": "deviate"}') == "deviate"
    assert parse_prediction("I think we should follow") == "follow"
    assert parse_prediction("no answer here") is None
    assert deviation_reward('"deviation": "deviate"', 1) == 1.1
    assert deviation_reward('"deviation": "follow"', 1) == 0.1
    assert deviation_reward("garbage", 0) == 0.0
    # b-NEW cue path: word right after the cue, case-insensitive.
    assert parse_prediction("reasoning...\n\nAnswer: deviate", CUE) == "deviate"
    assert parse_prediction("reasoning...\n\nAnswer: DEVIATE", CUE) == "deviate"
    # THE TRAP the cue exists for: committed answer is 'follow', but a later stray line ends in
    # 'deviate', so the bare-word fallback (no cue) grabs the WRONG word; the cue path fixes it.
    trap = "They might deviate.\n\nAnswer: follow\n(ordering CT instead of US would be a deviate.)"
    assert parse_prediction(trap) == "deviate"          # no cue -> last bare word -> WRONG
    assert parse_prediction(trap, CUE) == "follow"      # cue-anchored -> RIGHT
    assert deviation_reward(trap, 0, answer_cue=CUE) == 1.1    # gold=follow, cue-correct
    assert deviation_reward(trap, 1, answer_cue=CUE) == 0.1    # gold=deviate, wrong

    # bnew_reward: structure counted from reasoning body, correctness dominates.
    full = ("The leading hypothesis is appendicitis; the rubric recommends a CT scan for high-risk "
            "cases.\n\nAnswer: deviate")
    t, bd = bnew_reward(full, 1, return_breakdown=True)        # correct + all 3 parts + format
    assert bd["correct"] == 1.0 and bd["struct"] == 1.0 and bd["components"] == {
        "modality": True, "rubric": True, "hypothesis": True}, bd
    assert abs(bd["total"] - (1.0 + 0.3 + 0.1)) < 1e-6, bd
    bare = "It will differ.\n\nAnswer: deviate"                # correct but NO parts mentioned
    t2, bd2 = bnew_reward(bare, 1, return_breakdown=True)
    assert bd2["correct"] == 1.0 and bd2["struct"] == 0.0, bd2
    assert t > t2                                              # structure earns strictly more
    wrong_rich = full.replace("deviate", "follow")            # all parts but WRONG answer
    assert bnew_reward(wrong_rich, 1) < t2                     # correctness gap (1.0) > structure (0.3)
    print("rl_reward self-test OK")
