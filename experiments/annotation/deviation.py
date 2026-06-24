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
                Computed STEP-BY-STEP via rubric traversal (belief_step_deviation):
                feed the step's real causally-masked pre-decision features into the
                belief graph -> the imaging the rubric wants HERE -> compare to the
                doctor's action -> {follow, deviate, off_rubric}. Single-step has only
                same/not-same; commission / omission / order-swap are SEQUENCE-level
                and belong to god-view, NOT here.

R[D] (recommended imaging per disease sub-rubric) is extracted *programmatically*
from pipeline.rubric_graph.DISEASE_GRAPHS (union of node.required_tests, filtered
to imaging modalities). It is now used only for the synthetic 'biliary' branch (no
graph node) and episode-level omission; the step-level judgment uses live traversal.
"""
from __future__ import annotations

import re
from functools import lru_cache

from pipeline.rubric_graph import DISEASE_GRAPHS
from pipeline.traversal_engine import traverse_graph

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
#
# 'biliary colic' is deliberately NOT a token: colic is GALLBLADDER/cystic-duct pain
# (gallbladder-WALL axis), conceptually not the bile-DUCT axis this branch represents. At
# full-300 scale it tagged 6 steps (4 of them buried behind a gyn/SBO lead) — pure noise for
# the ductal entity. (full-300 audit, annotation_agent_design memory.)
BILIARY_OTHER_HYP = re.compile(
    r"choledocho|cholangit|common bile duct|\bCBD\b|biliary (?:obstruct|dilat|tree|duct)"
    r"|bile leak|bilom|ductal dilat|post-ERCP|ampull|sphincterotom",
    re.I,
)


def _leading_clause(text: str) -> str:
    """First top-level hypothesis of an other_hypothesis differential.

    Splits on the first DEPTH-0 ',' or ';' so a parenthetical list stays with its clause and
    intra-hypothesis '/' alternation ("choledocholithiasis / biliary obstruction") is NOT a
    break. Used to gate the biliary relabel to a LEADING ductal belief: at full-300 scale ~22%
    of regex hits had biliary buried 3rd-4th behind an SBO / post-op / mesenteric-ischemia / gyn
    lead (e.g. chole 20972818 "SBO primary concern…", panc 29581468 "mesenteric ischemia…biloma",
    chole 22521761 "post-bypass complication…biliary") — those are broad-net steps that should
    stay off_rubric, not biliary deviate_commission. (annotation_agent_design memory.)
    """
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch in ",;" and depth == 0:
            return text[:i]
    return text


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


def rubric_recommended_imaging(belief: str, pre_features: dict) -> list[str]:
    """Imaging the belief's sub-rubric wants NEXT, given the pre-decision feature state.

    This is the STEP-LOCAL rubric recommendation: traverse the belief disease graph
    against the causally-masked features available *before* the doctor's action
    (rubric_features idx_{k-1}; idx_k already contains test-k's result), and read
    the pending imaging at the rubric frontier. Empty = the rubric is at a terminal,
    is blocked (its gate conditions don't fire on the recorded features), or only wants
    a non-imaging test here — in every such case the rubric is NOT asking for this image.

    'biliary' has no DISEASE_GRAPHS node (post-hoc synthetic branch) -> fall back to its
    static duct-modality set R[biliary]={US,MRCP}.
    """
    if belief == "biliary":
        return sorted(EXTRA_BRANCH_IMAGING["biliary"])
    if belief not in DISEASE_GRAPHS:
        return []
    r = traverse_graph(DISEASE_GRAPHS[belief], pre_features)
    return sorted(set(r.pending_tests) & IMAGING_KEYS)


def rubric_state(belief: str, pre_features: dict) -> str:
    """WHY the rubric does/doesn't want imaging here — splits the 'deviate' bucket so
    over-testing/staging (rubric already at a diagnosis) is told apart from rubric
    INCOMPLETENESS (gate conditions don't fire on the recorded features).

      terminal_confirmed / terminal_excluded / terminal_low_risk
                       : sub-rubric reached a diagnostic leaf -> any further imaging is
                         over-testing / staging / complication-search, NOT a missed path.
      recommends_imaging : rubric wants a specific image next (follow if it matches).
      wants_nonimaging   : rubric wants a non-imaging test (e.g. Lab) before any image.
      blocked            : reachable, required tests done, but all gate edges are False
                         -> the rubric has NO path for this patient = incompleteness gap.
      biliary / off_rubric : synthetic branch / belief='other'.
    """
    if belief == "biliary":
        return "biliary"
    if belief not in DISEASE_GRAPHS:
        return "off_rubric"
    r = traverse_graph(DISEASE_GRAPHS[belief], pre_features)
    if r.terminal_type:
        return r.terminal_type
    if set(r.pending_tests) & IMAGING_KEYS:
        return "recommends_imaging"
    if r.pending_tests:
        return "wants_nonimaging"
    return "blocked"


def belief_step_deviation(top_branch: str, pre_features: dict, ordered: str,
                          other_hypothesis: str | None = None
                          ) -> tuple[str, str, list[str]]:
    """Step-by-step belief-conditioned status via rubric TRAVERSAL (replaces the old
    set-membership `belief_deviation`).

    At a single step the only well-posed question is: does the rubric of the disease the
    doctor *currently believes* recommend, in THIS feature state, the image the doctor
    actually ordered? -> two outcomes (commission/omission/order-swap are SEQUENCE-level
    notions, judged by god-view, not here):

      off_rubric : belief sits on 'other' with no biliary rescue -> no sub-rubric to judge
                   against (do NOT call it 'wrong'; verification judges quality — these are
                   the outlier-patch signal).
      follow     : the ordered imaging IS in the rubric's step-local recommendation.
      deviate    : the ordered imaging is NOT what the rubric wants here — because the
                   rubric wanted a different image, a non-imaging test, or nothing at all
                   (terminal / blocked gate). Per design, all of these are `deviate`.

    Post-hoc biliary rescue is preserved: an off_rubric step whose other_hypothesis names a
    bile-duct process (leading clause) is relabeled effective belief 'biliary' and judged
    vs R[biliary]={US,MRCP}. A reconstructed disease belief is NEVER reclassified.

    Returns (effective_branch, dev_status, rubric_recommended_imaging).
    """
    eff = top_branch
    if top_branch not in DISEASE_GRAPHS and other_hypothesis \
            and BILIARY_OTHER_HYP.search(_leading_clause(other_hypothesis)):
        eff = "biliary"

    if eff not in DISEASE_GRAPHS and eff != "biliary":   # genuine 'other'
        return eff, "off_rubric", []

    rec = rubric_recommended_imaging(eff, pre_features)
    status = "follow" if modality_of(ordered) in rec else "deviate"
    return eff, status, rec


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
