"""Belief-conditioned deviation labelling.

Two distinct notions of "deviation" (kept as separate columns, neither replaces
the other — see HANDOFF §1.5):

  dev_godview : action vs the rubric of the *true* (MIMIC discharge) disease.
                god-view, hindsight; measures over-attachment / leakage-as-signal.
                (computed elsewhere from tsc actual-vs-simulated; here we only
                accept a precomputed commission set for comparison.)

  dev_belief  : action vs the sub-rubric of the disease the doctor *assumed at
                that step* = argmax of the reconstructed differential (top_branch).
                This is what the certainty-trigger AGENT must learn: at inference
                it only knows its current belief, so "deviate from rubric" can only
                mean "deviate from the rubric of what I currently think it is".

R[D] (recommended imaging per disease sub-rubric) is extracted *programmatically*
from pipeline.rubric_graph.DISEASE_GRAPHS (union of node.required_tests, filtered
to imaging modalities) so it stays in sync with the rubric and is never hand-copied.
"""
from __future__ import annotations

import re
from functools import lru_cache

from pipeline.rubric_graph import DISEASE_GRAPHS

# imaging test keys the rubric can recommend (Radiograph_Chest is excluded from
# commission upstream and is never a DECISION step, so it is not listed here)
IMAGING_KEYS = {
    "Ultrasound_Abdomen", "CT_Abdomen", "MRI_Abdomen",
    "MRCP_Abdomen", "CTU_Abdomen", "HIDA",
}

# Bile-DUCT axis (annotation_agent_design memory, RAW-GROUNDED audit): a coherent entity
# MIMIC buckets under cholecystitis/pancreatitis but the rubric graph has no node for.
# NOT a Mode-A differential branch — making it compete in the softmax over-attributed
# established pancreatitis (forced-choice mass-stealing, not hallucination). Instead it is
# recovered POST-HOC from off_rubric steps' other_hypothesis (see derived_belief). R[biliary]
# = duct-evaluating modalities (US screens CBD/gallstones; MRCP maps the tree), so a biliary
# belief that orders CT is deviate_commission (went broad), not off_rubric.
EXTRA_BRANCH_IMAGING = {
    "biliary": frozenset({"Ultrasound_Abdomen", "MRCP_Abdomen"}),
}

# Bile-duct / obstructive-biliary terms in an off_rubric step's other_hypothesis text.
# Validated on the 30-case batch: tags the 6 ductal off_rubric steps, zero false hits on
# the appendicitis/diverticulitis broad-net steps.
BILIARY_OTHER_HYP = re.compile(
    r"choledocho|cholangit|common bile duct|\bCBD\b|biliary (?:obstruct|dilat|tree|duct|colic)"
    r"|bile leak|bilom|ductal dilat|post-ERCP|ampull|sphincterotom",
    re.I,
)


@lru_cache(maxsize=1)
def recommended_imaging() -> dict[str, frozenset[str]]:
    """R[D] = imaging modalities recommended by disease D's sub-rubric.

    Source of truth = DISEASE_GRAPHS node.required_tests (same object the rubric
    simulator traverses), so this never drifts from the rubric definition.
    """
    out: dict[str, frozenset[str]] = {}
    for disease, graph in DISEASE_GRAPHS.items():
        tests: set[str] = set()
        for node in graph.nodes.values():
            tests |= set(getattr(node, "required_tests", []) or [])
        out[disease] = frozenset(tests & IMAGING_KEYS)
    out.update(EXTRA_BRANCH_IMAGING)  # biliary 6th branch (no DISEASE_GRAPHS node)
    return out


def modality_of(ordered: str) -> str:
    """'CT Abdomen (CT ABD & PELVIS ...)' -> 'CT_Abdomen' (matches runner)."""
    head = ordered.split("(")[0].strip()
    return "_".join(head.split()[:2])


def belief_deviation(top_branch: str, ordered: str) -> str:
    """Step-level belief-conditioned status.

      off_rubric         : belief sits on 'other' (or an unknown branch) -> no
                           sub-rubric to deviate from; do NOT judge follow/deviate
                           (and do NOT call it 'wrong' a priori — vindication judges
                           quality; off-rubric steps are the outlier-patch signal).
      follow             : ordered imaging IS in R[top_branch].
      deviate_commission : believes top_branch yet orders imaging outside R[top_branch]
                           (e.g. believes cholecystitis but orders CT).
    """
    R = recommended_imaging()
    if top_branch not in R:          # 'other' or anything without a sub-rubric
        return "off_rubric"
    return "follow" if modality_of(ordered) in R[top_branch] else "deviate_commission"


def derived_belief(top_branch: str, ordered: str,
                   other_hypothesis: str | None) -> tuple[str, str]:
    """Post-hoc biliary recovery, then belief-conditioned status.

    Only off_rubric steps (top_branch is 'other', i.e. NOT one of the four reconstructed
    disease graphs) are eligible: if their other_hypothesis names a bile-duct/obstructive
    process, relabel the effective belief 'biliary' and judge it against R[biliary]={US,MRCP}.
    A reconstructed disease belief is NEVER reclassified (this is what prevents the over-
    attribution of established pancreatitis seen when biliary competed in the softmax).

    Returns (effective_branch, dev_status) where dev_status ∈ {follow, deviate_commission,
    off_rubric}. 'biliary' + US/MRCP -> follow; 'biliary' + CT -> deviate_commission.
    """
    eff = top_branch
    if top_branch not in DISEASE_GRAPHS and other_hypothesis \
            and BILIARY_OTHER_HYP.search(other_hypothesis):
        eff = "biliary"
    return eff, belief_deviation(eff, ordered)


def episode_omissions(top_branch_final: str, ordered_modalities: set[str]) -> set[str]:
    """Episode-level omission = rubric of the FINAL assumed disease recommended
    imaging that was never ordered in the whole episode. (off_rubric -> empty.)"""
    R = recommended_imaging()
    if top_branch_final not in R:
        return set()
    return set(R[top_branch_final]) - set(ordered_modalities)


def godview_step_flags(rubric_imaging_seq: list[str],
                       ordered_seq: list[str]) -> tuple[list[bool], set[str]]:
    """Event-aligned god-view commission per decision step (fixes HANDOFF §7 todo-1).

    god-view = action vs the patient's OWN rubric path (tsc simulated_sequence),
    conditioned on the true (MIMIC) disease. Instead of set-membership (which marks
    every repeat CT as a deviation), consume the rubric imaging as a MULTISET:
      - first occurrence of a rubric-recommended modality  -> follow (consume it)
      - a modality not in the rubric, OR an extra repeat    -> commission

    Returns (flags, omissions): flags[i] True = step i is commission;
    omissions = rubric imaging never matched by any step.
    """
    from collections import Counter
    remaining = Counter(rubric_imaging_seq)
    flags: list[bool] = []
    for m in ordered_seq:
        if remaining.get(m, 0) > 0:
            remaining[m] -= 1
            flags.append(False)            # follow (covered by rubric path)
        else:
            flags.append(True)             # commission (off-rubric or extra repeat)
    omissions = {k for k, v in remaining.items() if v > 0}
    return flags, omissions
