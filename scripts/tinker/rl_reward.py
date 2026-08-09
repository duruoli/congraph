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

# b-NEW answer cue: the four output fields are CONSECUTIVE single-newline lines, so the final
# verdict line is "\nAnswer: ". MUST match build_bnew.ANSWER_CUE and eval_prob_tinker --answer-cue.
ANSWER_CUE_DEFAULT = "\nAnswer: "


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


# ------------------------- b-NEW composite reward (4-term process reward) -------------------------
# Designed FROM the 334-trace pre-RL emergence analysis (results/tinker/RESULTS_bnew_reward_emergence.md).
# The prompt EXPLICITLY asks for four labelled lines (see build_bnew.py); the reward scores the
# CORRECTNESS of each — NOT its mere presence. Presence saturates in the base (rubric-mention 99%,
# hypothesis-named 100% -> zero GRPO gradient); correctness of each part is only ~0.46-0.57 in the base
# -> real variance + headroom. The four parts are the nodes of the actual clinical-reasoning chain,
# each with a clean categorical gold, each exact-matchable without an LLM judge:
#   1. Leading diagnosis -> meta.effective_branch  (6-way; 'other' = off all rubrics; base ~0.574)
#   2. Rubric recommends -> meta.rubric_recommended (a '|'-joined SET, or '-'/none; base ~0.457)
#   3. Predicted study   -> meta.how_modality       (4-way physician's actual next study; base ~0.536)
#   4. Answer (verdict)  -> meta.y                   (2-way; base 0.546, BELOW base rate = max headroom)
# The dropped free-text why-trace parts (differential/info-gap/expected-finding) have no exact gold,
# are keyword-gameable, and correlate weakly with a correct verdict.
DX_CUE = "Leading diagnosis:"
REC_CUE = "Rubric recommends:"
STUDY_CUE = "Predicted study:"       # these three MUST match build_bnew.py's cues
# surface form -> the 4 gold imaging classes. Order matters: MRCP before MRI (so 'MRCP' isn't caught
# by a looser rule), canonical names before bare abbreviations.
_STUDY_CANON = [
    ("MRCP_Abdomen",       re.compile(r'mrcp', re.I)),
    ("CT_Abdomen",         re.compile(r'\bct\b|ct[_ ]?abd|computed tomog', re.I)),
    ("Ultrasound_Abdomen", re.compile(r'ultrasound|sonograph|\bus\b', re.I)),
    ("MRI_Abdomen",        re.compile(r'\bmri\b', re.I)),
]
_DX_VOCAB = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis", "biliary", "other"]


def _field_line(completion: str, cue: str) -> str | None:
    """The text on the LAST occurrence of a labelled line (e.g. after 'Predicted study:'), or None if
    the cue is absent. Returns "" when the cue is present but nothing follows it (e.g. the rollout was
    truncated right after the label) — must NOT index [0] into an empty splitlines() (that IndexError
    killed a full RL run)."""
    idx = completion.rfind(cue)
    if idx == -1:
        return None
    lines = completion[idx + len(cue):].splitlines()
    return lines[0] if lines else ""


def parse_study(completion: str, study_cue: str = STUDY_CUE) -> str | None:
    """The physician-study the model COMMITS to on the `Predicted study:` line (one of 4), or None.
    Reads only that line, so a modality merely discussed in the prose can't satisfy it."""
    line = _field_line(completion, study_cue)
    if line is None:
        return None
    for canon, pat in _STUDY_CANON:
        if pat.search(line):
            return canon
    return None


def parse_dx(completion: str, dx_cue: str = DX_CUE) -> str | None:
    """The leading diagnosis on the `Leading diagnosis:` line, canonicalised to the 6-way vocab."""
    line = _field_line(completion, dx_cue)
    if line is None:
        return None
    low = line.lower()
    # cholecystitis before biliary/cholelithiasis so the more specific disease wins when both appear
    for name in _DX_VOCAB:
        if name in low:
            return name
    return None


def parse_rec_set(completion: str, rec_cue: str = REC_CUE) -> frozenset | None:
    """The rubric-recommended imaging on the `Rubric recommends:` line, as a SET of the 4 classes.
    'none'/'-'/empty -> empty set (the off-rubric / terminal case). None only if the line is absent."""
    line = _field_line(completion, rec_cue)
    if line is None:
        return None
    if re.search(r'\bnone\b|^\s*-\s*$', line, re.I) or not line.strip():
        return frozenset()
    return frozenset(canon for canon, pat in _STUDY_CANON if pat.search(line))


def gold_rec_set(gold_rubric_recommended: str | None) -> frozenset:
    """meta.rubric_recommended ('-' or a '|'-joined set like 'MRCP_Abdomen|Ultrasound_Abdomen')
    -> the canonical SET. '-'/None -> empty (off-rubric / nothing recommended)."""
    if not gold_rubric_recommended or gold_rubric_recommended == "-":
        return frozenset()
    return frozenset(p.strip() for p in gold_rubric_recommended.split("|") if p.strip())


def bnew_reward(completion: str, meta: dict, *, answer_cue: str = ANSWER_CUE_DEFAULT,
                w_ans: float = 1.0, w_dx: float = 0.25, w_rec: float = 0.25, w_std: float = 0.25,
                w_len: float = 0.0002, len_budget: int = 512, len_pen_cap: float = 0.2,
                n_tokens: int | None = None, return_breakdown: bool = False):
    """4-term process reward. `meta` carries the golds (from build_bnew: y, effective_branch,
    rubric_recommended, how_modality). Structure:

        R = FORMAT_GATE * ( w_ans·[answer==gold]        # verdict, DOMINANT (the deliverable)
                          + w_dx ·[dx==effective_branch] # situate: which disease / off-rubric
                          + w_rec·[rec_set==gold_set]    # apply rubric: next imaging set / none
                          + w_std·[study==how_modality]  # predict the physician's next study
                          - len_penalty )

    Weights: Σ(sub-terms)=0.75 < w_ans=1.0, so a correct verdict can never be traded for partial
    credit on the steps. FORMAT GATE (no parseable `Answer:` -> 0) applies the format pressure the
    12% of base ramblers need. Each sub-field independently scores 0 if its line is absent/wrong."""
    # STRICT format gate (see the old note): require the actual Answer cue, no bare-word fallback.
    pred = parse_prediction(completion, answer_cue) if answer_cue in completion else None
    if pred is None:
        if return_breakdown:
            return 0.0, {"pred": None, "gated": True, "total": 0.0}
        return 0.0

    gold_word = _POS if int(meta["y"]) == 1 else _NEG
    ans_match = 1.0 if pred == gold_word else 0.0
    dx = parse_dx(completion)
    dx_match = 1.0 if (dx is not None and dx == meta.get("effective_branch")) else 0.0
    rec = parse_rec_set(completion)
    rec_match = 1.0 if (rec is not None and rec == gold_rec_set(meta.get("rubric_recommended"))) else 0.0
    study = parse_study(completion)
    std_match = 1.0 if (study is not None and study == meta.get("how_modality")) else 0.0

    if n_tokens is None:
        n_tokens = len(completion) // 4
    overflow = max(0, n_tokens - len_budget)
    len_pen = min(len_pen_cap, w_len * overflow)
    total = (w_ans * ans_match + w_dx * dx_match + w_rec * rec_match + w_std * std_match) - len_pen
    if return_breakdown:
        return total, {"pred": pred, "gated": False, "ans_match": ans_match,
                       "dx": dx, "dx_match": dx_match, "rec": sorted(rec) if rec is not None else None,
                       "rec_match": rec_match, "study": study, "std_match": std_match,
                       "n_tokens": n_tokens, "len_pen": round(len_pen, 4), "total": round(total, 4)}
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

    # field parsers: read only the committed line; canonicalise; MRCP before MRI; sets for rec.
    assert parse_study("Predicted study: CT Abdomen/Pelvis\nAnswer: follow") == "CT_Abdomen"
    assert parse_study("Predicted study: abdominal ultrasound") == "Ultrasound_Abdomen"
    assert parse_dx("Leading diagnosis: acute cholecystitis") == "cholecystitis"
    assert parse_dx("Leading diagnosis: other (gynecologic)") == "other"
    assert parse_rec_set("Rubric recommends: MRCP_Abdomen|Ultrasound_Abdomen") == frozenset(
        {"MRCP_Abdomen", "Ultrasound_Abdomen"})
    assert parse_rec_set("Rubric recommends: none") == frozenset()
    assert parse_rec_set("no line") is None
    # REGRESSION: cue present but truncated with nothing after it -> "" not IndexError (killed a run).
    assert parse_study("reasoning...\nPredicted study:") is None
    assert parse_dx("Leading diagnosis:") is None
    assert parse_rec_set("Rubric recommends:") == frozenset()
    assert bnew_reward("blah\nPredicted study:", {"y": 1, "effective_branch": "x",
                       "rubric_recommended": "-", "how_modality": "CT_Abdomen"}) == 0.0   # no Answer -> gated, no crash
    assert gold_rec_set("MRCP_Abdomen|Ultrasound_Abdomen") == frozenset({"MRCP_Abdomen", "Ultrasound_Abdomen"})
    assert gold_rec_set("-") == frozenset()
    # a modality only DISCUSSED in prose (not on the committed line) does NOT count.
    assert parse_study("I considered a CT but chose nothing.\n\nAnswer: deviate") is None

    # bnew_reward (4-term): FORMAT GATE, verdict dominant, three categorical process terms.
    META = {"y": 1, "effective_branch": "appendicitis", "rubric_recommended": "Ultrasound_Abdomen",
            "how_modality": "CT_Abdomen"}
    full = ("exam is equivocal; leukocytosis argues appendicitis.\n"
            "Leading diagnosis: appendicitis\nRubric recommends: Ultrasound_Abdomen\n"
            "Predicted study: CT_Abdomen\nAnswer: deviate")             # all four CORRECT (consecutive lines)
    t, bd = bnew_reward(full, META, return_breakdown=True)
    assert bd["ans_match"] == 1 and bd["dx_match"] == 1 and bd["rec_match"] == 1 and bd["std_match"] == 1, bd
    assert abs(bd["total"] - (1.0 + 0.25 + 0.25 + 0.25)) < 1e-6, bd
    # wrong dx only -> loses just w_dx.
    wrong_dx = full.replace("Leading diagnosis: appendicitis", "Leading diagnosis: pancreatitis")
    assert abs(bnew_reward(wrong_dx, META) - (1.0 + 0.25 + 0.25)) < 1e-6
    # WRONG verdict but all three steps right -> 0.75 < any right-verdict reward: verdict dominates.
    wrong_ans = full.replace("Answer: deviate", "Answer: follow")
    assert abs(bnew_reward(wrong_ans, META) - (0.25 + 0.25 + 0.25)) < 1e-6
    bare_right = "leukocytosis argues appendicitis.\nAnswer: deviate"     # right verdict, no field lines
    assert abs(bnew_reward(bare_right, META) - 1.0) < 1e-6                # answer only
    assert bnew_reward(wrong_ans, META) < bnew_reward(bare_right, META)   # right verdict (1.0) beats rich-but-wrong (0.75)
    # rec as a SET: biliary case, both modalities required.
    META_BIL = {"y": 0, "effective_branch": "biliary",
                "rubric_recommended": "MRCP_Abdomen|Ultrasound_Abdomen", "how_modality": "Ultrasound_Abdomen"}
    bil = ("bile-duct process.\nLeading diagnosis: biliary\n"
           "Rubric recommends: Ultrasound_Abdomen|MRCP_Abdomen\nPredicted study: Ultrasound_Abdomen\nAnswer: follow")
    _, bdb = bnew_reward(bil, META_BIL, return_breakdown=True)
    assert bdb["rec_match"] == 1 and bdb["ans_match"] == 1 and bdb["dx_match"] == 1, bdb
    # FORMAT GATE: no parseable Answer -> 0 even with all other fields correct.
    rambler = "Leading diagnosis: appendicitis\nPredicted study: CT_Abdomen\nand I ramble on"
    assert bnew_reward(rambler, META) == 0.0
    print("rl_reward self-test OK")
