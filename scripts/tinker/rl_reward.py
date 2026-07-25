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


def parse_prediction(completion: str) -> str | None:
    """Pull the predicted answer word from a generated completion.
    Accepts either a JSON-ish `"deviation": "deviate"` tail (arm c / structured) or a bare
    trailing `deviate`/`follow` (arm b-NEW / free reasoning). Returns 'deviate'|'follow'|None."""
    m = re.search(r'"deviation"\s*:\s*"(follow|deviate)"', completion)
    if m:
        return m.group(1)
    # fall back to the LAST standalone occurrence of either word
    hits = re.findall(r'\b(follow|deviate)\b', completion)
    return hits[-1] if hits else None


def deviation_reward(completion: str, gold_y: int, format_coef: float = 0.1) -> float:
    """gold_y: 1 = deviate/off_rubric (positive), 0 = follow. Returns reward in [0, 1+format_coef]."""
    pred = parse_prediction(completion)
    if pred is None:
        return 0.0                      # unparseable -> no format bonus, no correctness
    gold_word = _POS if gold_y == 1 else _NEG
    correct = 1.0 if pred == gold_word else 0.0
    return correct + format_coef        # format bonus for producing a parseable answer at all


if __name__ == "__main__":  # tiny self-test
    assert parse_prediction('... "deviation": "deviate"}') == "deviate"
    assert parse_prediction("I think we should follow") == "follow"
    assert parse_prediction("no answer here") is None
    assert deviation_reward('"deviation": "deviate"', 1) == 1.1
    assert deviation_reward('"deviation": "follow"', 1) == 0.1
    assert deviation_reward("garbage", 0) == 0.0
    print("rl_reward self-test OK")
