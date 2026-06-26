"""Compile the step-level training set for the certainty-trigger agent (HANDOFF §8).

This is a PACKAGER, not a new-information step. It reshapes already-computed
annotation material into one row per retained decision step, each row a fixed
three-part record  INPUT -> TARGET -> REWARD  (+ derived CERTAINTY, + META labels).

Inputs (all already on disk):
  - results/annotation_experiment/full/belief_deviation_filtered.csv
        labels per step (dev_belief, rubric_state, rubric_rec, eff_branch,
        verification) + the PROVISIONAL timing filter (excluded_monitoring).
  - results/annotation_experiment/full/<disease>_<hadm>.json
        the ex-ante reconstructed reasoning trace + ex-post verification per step.
  - data/raw_data/<disease>_hadm_info_first_diag.csv  (via build_masked_view)
        baseline + RR-N ordered radiology reports -> causally-masked patient state.
  - pipeline.rubric_graph.DISEASE_GRAPHS
        the belief sub-rubric served to the agent (agent stage GIVES the rubric).

The five transforms (HANDOFF §8), faithfully:
  1. CAUSAL MASK   INPUT.patient_state = baseline + ONLY this step's visible prior
                   imaging reports (build_masked_view); this test's result + all
                   later reports are dropped. Verification never enters INPUT.
  2. RUBRIC LIBRARY   serve the FULL triage every step (all 4 disease sub-rubrics +
                   open 'other'), written once to rubric_library.json and referenced
                   per row. The agent self-routes (it must OUTPUT its belief argmax),
                   so a single pre-selected sub-rubric is NOT given — that would leak
                   the belief target. INPUT.active_path = the prior-step belief argmax
                   (the current working hypothesis; PAST info, not leakage; None at
                   step 1). deviate is then judged wrt the rubric of the branch the
                   agent's OWN argmax belief lands on (biliary recovered post-hoc).
  3. TWO-CHANNEL CERTAINTY  belief = the reconstructed differential (max_p / entropy
                   / other_mass); alarm = the OVERRIDE switch built from A2/A3 (the
                   offline two-call LLM alarm pass: study_inadequacy + discordance, read
                   from results/annotation_experiment/alarm/*.json) + A1 derived here
                   algorithmically (belief concentrated on a rubric disease & a severity/
                   etiology/complication gap; resolution = appropriateness). Replaces the
                   earlier provisional keyword alarm (HANDOFF §2.5.2 / §8 transform 3).
  4. HYBRID TARGET  default TARGET = the doctor's actual action (imitation). Steps
                   flagged should-suppress = over-imaging by the §2.5 criterion-2
                   (outcome-free): alarm SILENT (no A2/A3/A1) AND belief CONCENTRATED
                   (low entropy / low other_mass) AND same-modality repeat. These are
                   down-weighted (default) or relabeled when_action->stop (--suppress-mode
                   stop). The OLD DD-1 proxy (deviate+disconfirmed+terminal) is dropped —
                   it rested on a single outcome and contradicted C1/C2. FD-2 stale-
                   anatomy needs charttime -> still deferred.
  5. LEAKAGE STRIP  Discharge Dx / ICD / Procedures never present (build_masked_view
                   never emits them); disease-godview + verification kept out of INPUT
                   (they live in META/REWARD only).

Row scope: rrn_aligned==True (have the dev_belief WHEN label) AND not
excluded_monitoring (the provisional timing filter already drops post-intervention
contaminated steps). rrn-misaligned steps lack a clean pre-state -> skipped (count
reported). Deviation is a LABEL not a filter -> follow / deviate / off_rubric all kept.

PROVISIONAL-DATA PREMISE: the timing filter is the coarse text-hint version (real
MIMIC charttime pending). The interface is stable — when real timestamps land,
regenerate belief_deviation_filtered.csv (same columns) and re-run this script
unchanged. See agent_training_plan memory.

Usage:
  /opt/anaconda3/bin/python scripts/compile_training_set.py
      [--ann-dir results/annotation_experiment/full]
      [--out data/training_set]
      [--suppress-mode downweight|stop|off]   (default downweight)
      [--suppress-weight 0.3]
      [--include-misaligned]                   (also emit rrn_aligned==False, no WHEN label)
      [--preview N]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from pipeline.rubric_graph import DISEASE_GRAPHS  # noqa: E402
from scripts.build_masked_view import (  # noqa: E402
    RAW, build_record, load_lab_map,
)

# --- transform 3: ALARM channel = real A2/A3 (LLM pass) + derived A1 --------
# Replaces the earlier provisional keyword version (HANDOFF §8 transform 3). The two
# objective external red flags A2 (study_inadequacy) / A3 (discordance) come from the
# offline two-call LLM alarm pass (results/annotation_experiment/alarm/*.json, run by
# scripts/run_alarm_pass.py); A1 (advanced_question) is NOT an external flag and is
# DERIVED here algorithmically (HANDOFF §2.5.2).
_TRIGGER_KEYS = ("study_inadequacy", "discordance")
_A1_TAU = 0.5
# etiology / complication gap language (severity is already an action_role -> below).
_A1_GAP = re.compile(r"etiolog|underlying cause|complication|perforat|abscess"
                     r"|necros|ischemi|gangren|biliary obstruct", re.I)


def _flag_present(flag) -> bool:
    v = flag.get("present") if isinstance(flag, dict) else None
    return v is True or (isinstance(v, str) and v.strip().lower() in ("true", "yes", "1"))


def derive_a1(top_branch, max_p, action_role, info_gap, appropriateness) -> dict:
    """A1 (advanced_question) is a property of the BELIEF channel, not an external red
    flag, so a blind LLM re-judgment fights the reconstructed belief (HANDOFF §2.5.2 —
    measured 5/12 misfires) -> derived algorithmically: belief CONCENTRATED on a RUBRIC
    disease (max_p>=tau, argmax != 'other') AND the open question has moved past
    diagnosis to severity / etiology / complication. Resolution quality = this step's
    `appropriateness` (the gap-addressing judgment Mode-A already made)."""
    role = action_role
    if isinstance(role, list):
        role = " ".join(map(str, role))
    role = str(role).strip().lower()
    concentrated = (top_branch in DISEASE_GRAPHS
                    and max_p is not None and max_p >= _A1_TAU)
    advanced = (role == "assess_severity") or bool(_A1_GAP.search(str(info_gap or "")))
    present = bool(concentrated and advanced)
    return {"present": present,
            "concentrated": bool(concentrated),
            "advanced_question": bool(advanced),
            "resolution": appropriateness if present else "",
            "basis": f"belief max_p>={_A1_TAU} on a rubric disease "
                     "& severity/etiology/complication gap"}


def alarm_channel(alarm_step, top_branch, max_p, ex_ante, prior_disconfirms) -> dict:
    """Two-channel ALARM signal: A2/A3 from the offline LLM alarm pass + derived A1.
    `any_trigger` = the OVERRIDE switch (HANDOFF §2.5.1) used by transform 4."""
    detect = (alarm_step or {}).get("alarm_detect") or {}
    resolve = (alarm_step or {}).get("alarm_resolve") or {}
    a2 = detect.get("study_inadequacy") or {}
    a3 = detect.get("discordance") or {}
    a2_present = _flag_present(a2)
    a3_present = _flag_present(a3)
    a1 = derive_a1(top_branch, max_p, ex_ante.get("action_role", ""),
                   ex_ante.get("information_gap", ""), ex_ante.get("appropriateness", ""))
    return {
        "any_trigger": bool(a2_present or a3_present or a1["present"]),
        "study_inadequacy": {"present": a2_present, "score": a2.get("score"),
                             "evidence": a2.get("evidence", "")},            # A2 (LLM)
        "discordance": {"present": a3_present, "score": a3.get("score"),
                        "evidence": a3.get("evidence", "")},                 # A3 (LLM)
        "advanced_question": a1,                                             # A1 (derived)
        "resolve": {"addresses_alarm": resolve.get("addresses_alarm", ""),
                    "unaddressed_alarm": resolve.get("unaddressed_alarm"),
                    "reason": resolve.get("reason", "")},
        "prior_disconfirmations": prior_disconfirms,        # results visible here, no leak
        "alarm_parse_error": bool(detect.get("_parse_error")),
        "alarm_present": alarm_step is not None,
        "method": "llm_a2a3+algorithmic_a1",
    }


# --- transform 4: should-suppress (§2.5 criterion 2, REDEFINED) -------------
# OLD (wrong) proxy: deviate + disconfirmed + terminal/blocked rubric_state — it rests
# on a single OUTCOME (disconfirmed) and contradicts C1/C2 (the SAME structural spot is
# a valuable patch when confirmed; outcome isn't available ex-ante). CORRECT, outcome-
# free criterion (HANDOFF §2.5 / §2.5.1):
#     alarm SILENT  AND  belief CONCENTRATED (low entropy / low other_mass)
#                   AND  SAME-modality repeat
# = over-imaging past a de-facto stop. If ANY alarm fires (A2/A3 external red flag OR A1
# advanced question) OR belief is diffuse, the override is a VALUABLE deviation (rubric
# limitation), not over-imaging. FD-2 stale anatomy still needs charttime -> deferred.
_SUPPRESS_TAU = 0.5
_SUPPRESS_OTHER_MAX = 0.3


def _base_modality(m) -> str:
    """Map a modality string to its base (CT_Abdomen->ct, prior 'CT'->ct, 'Ultrasound
    Abdomen'->ultrasound) so the CSV modality and the visible-prior modality compare."""
    return str(m or "").replace(" ", "_").split("_")[0].strip().lower()


def same_modality_repeat(how, visible_prior_imaging) -> bool:
    base = _base_modality(how)
    if not base:
        return False
    priors = {_base_modality(vp.get("modality")) for vp in (visible_prior_imaging or [])}
    return base in priors


def is_should_suppress(alarm_any: bool, belief: dict, same_repeat: bool) -> bool:
    mp, om = belief.get("max_p"), belief.get("other_mass")
    concentrated = (mp is not None and mp >= _SUPPRESS_TAU
                    and om is not None and om <= _SUPPRESS_OTHER_MAX)
    return bool((not alarm_any) and concentrated and same_repeat)


# --- transform 3: belief-channel summary -----------------------------------
def belief_summary(differential: dict) -> dict:
    dist = {k: float(v) for k, v in (differential or {}).items()}
    if not dist:
        return {"dist": {}, "max_p": None, "entropy": None, "other_mass": None}
    ent = -sum(p * math.log(p) for p in dist.values() if p > 0)
    return {"dist": {k: round(v, 3) for k, v in dist.items()},
            "max_p": round(max(dist.values()), 3),
            "entropy": round(ent, 3),
            "other_mass": round(dist.get("other", 0.0), 3)}


# --- transform 2: rubric LIBRARY (all 4 + open 'other') ---------------------
# The agent is given ALL FOUR disease sub-rubrics + the open 'other' slot every
# step (the full triage), NOT a single pre-selected one. Reasons (see design
# discussion in agent_training_plan memory):
#   - serving only the argmax-belief sub-rubric LEAKS the belief target (the
#     differential argmax is something the agent must PRODUCE this step);
#   - the agent must self-route, exactly as at inference, then deviate is judged
#     wrt the rubric of the branch the agent's OWN belief lands on.
# The library is constant across rows -> written once to rubric_library.json and
# referenced per row (keeps the jsonl lean; the prompt-builder joins them).
# `active_path` (the prior-step belief argmax) is the agent's current working
# hypothesis — PAST information, so not leakage; it tells the agent which path it
# was on without committing this step's belief. biliary stays POST-HOC (not a
# served branch): it is recovered from an 'other' belief's other_hypothesis at
# judge time, mirroring the annotation pipeline.
def _trim(s: str, n: int = 600) -> str:
    s = str(s or "").strip()
    return s if len(s) <= n else s[:n] + " …"


def serialize_graph(disease: str) -> dict:
    g = DISEASE_GRAPHS[disease]
    nodes = [{"id": n.id, "label": getattr(n, "label", n.id),
              "type": getattr(n, "node_type", ""),
              "required_tests": list(getattr(n, "required_tests", []) or []),
              "description": _trim(getattr(n, "description", ""))}
             for n in g.nodes.values()]
    edges = [{"from": e.source, "to": e.target, "when": getattr(e, "label", "")}
             for e in g.edges]
    return {"disease": disease, "kind": "disease_subrubric", "root": g.root,
            "nodes": nodes, "edges": edges}


def build_rubric_library() -> dict:
    """The full triage served to the agent every step: 4 disease graphs + open 'other'.
    No step-specific frontier here (the agent traverses against the patient features
    itself); the frontier the reconstructed belief implied stays in META for eval."""
    lib = {d: serialize_graph(d) for d in DISEASE_GRAPHS}
    lib["other"] = {
        "disease": "other", "kind": "open_differential",
        "note": "No sub-rubric. Open acute-abdomen differential (gynecologic, urologic, "
                "bowel obstruction, mesenteric ischemia, bile-duct/biliary obstruction, …). "
                "Choosing 'other' = leaving the rubric; judged by outcome, not a priori wrong.",
        "biliary_axis": "If the 'other' belief is a bile-DUCT process (choledocholithiasis / "
                        "cholangitis / CBD obstruction / post-ERCP), the duct-evaluating "
                        "modalities are Ultrasound_Abdomen and MRCP_Abdomen.",
    }
    return lib


# --- annotation JSON access -------------------------------------------------
def load_case(ann_dir: Path, disease: str, hadm: int) -> dict | None:
    p = ann_dir / f"{disease}_{hadm}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_alarm_case(alarm_dir: Path, disease: str, hadm: int) -> dict[int, dict]:
    """Per-step A2/A3 alarm records (results/annotation_experiment/alarm/*.json),
    keyed by step. Missing file -> {} (A1 still derives; flagged in the channel)."""
    p = alarm_dir / f"{disease}_{hadm}.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    return {int(s["step"]): s for s in data.get("steps", [])}


def build_row(r, case, masked, alarm_steps, suppress_mode, suppress_weight,
              prior_disconfirms, active_path) -> dict | None:
    """Assemble one INPUT/TARGET/REWARD record for filtered-CSV row `r`."""
    step = int(r.step)
    jstep = next((s for s in case["steps"] if int(s["step"]) == step), None)
    if jstep is None:
        return None
    ex = jstep.get("representative_ex_ante") or {}
    vind = jstep.get("verification") or {}

    dp = next((d for d in masked["decision_points"] if int(d["step"]) == step), None)
    if dp is None:
        return None
    # causal mask: visible prior imaging WITHOUT this test's result / later reports
    patient_state = {
        "baseline": masked["baseline"],
        "visible_prior_imaging": [
            {k: v for k, v in vp.items()}            # includes report text of PRIOR tests only
            for vp in dp["visible_prior_imaging"]
        ],
    }

    when = r.dev_belief                              # follow / deviate / off_rubric
    how = r.modality

    # two-channel certainty: belief (differential shape) + alarm (A2/A3 LLM + A1 derived)
    belief = belief_summary(ex.get("differential"))
    alarm = alarm_channel(alarm_steps.get(step), r.top_branch, belief.get("max_p"),
                          ex, prior_disconfirms)
    repeat = same_modality_repeat(how, dp["visible_prior_imaging"])

    # transform 4: over-imaging = alarm silent + belief concentrated + same-modality repeat
    suppress = is_should_suppress(alarm["any_trigger"], belief, repeat)
    weight = 1.0
    if suppress and suppress_mode == "stop":
        when, how = "stop", None
    elif suppress and suppress_mode == "downweight":
        weight = suppress_weight
    # suppress_mode == "off" -> leave target untouched (pure imitation)

    return {
        "id": f"{r.disease}_{r.hadm}_step{step}",
        "disease": r.disease, "hadm": int(r.hadm), "step": step,
        "eff_branch": r.eff_branch,

        "INPUT": {
            "patient_state": patient_state,
            # full triage served every step; agent self-routes (see build_rubric_library)
            "rubric_library_ref": "rubric_library.json",
            "candidate_branches": list(DISEASE_GRAPHS.keys()) + ["other"],
            # prior-step belief argmax = current working hypothesis (PAST info, no leak);
            # None at the first step (no prior belief yet -> agent routes from scratch).
            "active_path": active_path,
        },

        "TARGET": {
            # belief_branch = the differential argmax the agent must output (in 4+other);
            # effective_branch = the rubric the deviate label is judged against, after the
            # post-hoc biliary rescue (may be 'biliary' when belief_branch=='other').
            "belief_branch": r.top_branch,
            "effective_branch": r.eff_branch,
            "when_action": when,                    # follow / deviate / off_rubric / stop
            "how_modality": how,
            "why_trace": {
                "belief": belief.get("dist"),
                "other_hypothesis": ex.get("other_hypothesis", ""),
                "information_gap": ex.get("information_gap", ""),
                "expected_finding": ex.get("expected_finding", ""),
                "action_role": ex.get("action_role", ""),
                "grounding": ex.get("grounding", []),
            },
        },

        "CERTAINTY": {                              # derived; NOT fed as INPUT verbatim
            "belief": belief,
            "alarm": alarm,
        },

        "REWARD": {                                 # ex-post; NEVER in INPUT
            "verification": vind.get("verification", r.verification),
            "certainty_update": vind.get("certainty_update", ""),
            "appropriateness": ex.get("appropriateness", ""),
            "appropriateness_reason": ex.get("appropriateness_reason", ""),
            "sample_weight": weight,
            "should_suppress": bool(suppress),
            "suppress_basis": ("overimaging_alarm_silent_belief_concentrated_modality_repeat"
                               if suppress else ""),
        },

        "META": {                                   # eval-stratification labels (leakage-safe out of INPUT)
            "dev_belief": r.dev_belief,
            "dev_godview": bool(r.dev_godview),
            "rubric_state": r.rubric_state,
            "rubric_recommended": r.rubric_rec,
            "top_branch": r.top_branch,
            "mean_other": float(r.mean_other) if pd.notna(r.mean_other) else None,
            "timing_role": r.timing_role if pd.notna(r.timing_role) else "unknown",
            "rrn_aligned": bool(r.rrn_aligned),
            "provisional_timing_filter": True,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", default="results/annotation_experiment/full")
    ap.add_argument("--alarm-dir", default="results/annotation_experiment/alarm")
    ap.add_argument("--out", default="data/training_set")
    ap.add_argument("--suppress-mode", choices=["downweight", "stop", "off"],
                    default="downweight")
    ap.add_argument("--suppress-weight", type=float, default=0.3)
    ap.add_argument("--include-misaligned", action="store_true")
    ap.add_argument("--preview", type=int, default=0)
    args = ap.parse_args()

    ann_dir = ROOT / args.ann_dir if not Path(args.ann_dir).is_absolute() else Path(args.ann_dir)
    alarm_dir = ROOT / args.alarm_dir if not Path(args.alarm_dir).is_absolute() else Path(args.alarm_dir)
    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ann_dir / "belief_deviation_filtered.csv")

    keep = ~df.excluded_monitoring.fillna(False)
    if not args.include_misaligned:
        keep &= df.rrn_aligned.fillna(False)
    sub = df[keep].copy()

    labmap = load_lab_map()
    masked_cache: dict[tuple, dict] = {}
    raw_cache: dict[str, pd.DataFrame] = {}

    def masked_for(disease: str, hadm: int):
        ck = (disease, hadm)
        if ck not in masked_cache:
            if disease not in raw_cache:
                raw_cache[disease] = pd.read_csv(RAW[disease])
            rw = raw_cache[disease]
            rrow = rw[rw["hadm_id"] == hadm]
            masked_cache[ck] = None if rrow.empty else \
                build_record(disease, hadm, rrow.iloc[0], labmap)
        return masked_cache[ck]

    # per (disease,hadm): prior-step disconfirmation count + prior-step belief argmax
    # (both run over ALL steps, incl. excluded/misaligned, since the agent at step k
    # would have seen step k-1 regardless of our row filtering).
    prior_dc: dict[tuple, dict[int, int]] = {}
    prior_belief: dict[tuple, dict[int, str]] = {}
    for (disease, hadm), grp in df.sort_values(["disease", "hadm", "step"]).groupby(["disease", "hadm"]):
        run = 0
        dc_m, pb_m, last_belief = {}, {}, None
        for _, rr in grp.iterrows():
            dc_m[int(rr.step)] = run
            pb_m[int(rr.step)] = last_belief         # belief argmax of the PREVIOUS step
            if str(rr.verification) == "disconfirmed":
                run += 1
            last_belief = rr.top_branch if pd.notna(rr.top_branch) else last_belief
        prior_dc[(disease, hadm)] = dc_m
        prior_belief[(disease, hadm)] = pb_m

    alarm_cache: dict[tuple, dict] = {}
    missing_alarm: set[tuple] = set()

    def alarm_for(disease: str, hadm: int) -> dict[int, dict]:
        ck = (disease, hadm)
        if ck not in alarm_cache:
            steps = load_alarm_case(alarm_dir, disease, hadm)
            alarm_cache[ck] = steps
            if not steps:
                missing_alarm.add(ck)
        return alarm_cache[ck]

    rows, skipped = [], Counter()
    for r in sub.itertuples(index=False):
        case = load_case(ann_dir, r.disease, int(r.hadm))
        if case is None:
            skipped["no_annotation_json"] += 1
            continue
        masked = masked_for(r.disease, int(r.hadm))
        if masked is None:
            skipped["no_raw_record"] += 1
            continue
        ck = (r.disease, int(r.hadm))
        pdc = prior_dc.get(ck, {}).get(int(r.step), 0)
        active_path = prior_belief.get(ck, {}).get(int(r.step), None)
        alarm_steps = alarm_for(r.disease, int(r.hadm))
        row = build_row(r, case, masked, alarm_steps, args.suppress_mode,
                        args.suppress_weight, pdc, active_path)
        if row is None:
            skipped["step_not_found"] += 1
            continue
        rows.append(row)

    # constant rubric library (all 4 + open 'other'), referenced by every row
    (out_dir / "rubric_library.json").write_text(
        json.dumps(build_rubric_library(), indent=2, ensure_ascii=False))

    out_jsonl = out_dir / "train_steps.jsonl"
    with out_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # manifest / provenance
    when_dist = Counter(x["TARGET"]["when_action"] for x in rows)
    vind_dist = Counter(x["REWARD"]["verification"] for x in rows)
    suppress_n = sum(x["REWARD"]["should_suppress"] for x in rows)
    by_disease = Counter(x["disease"] for x in rows)
    first_step = sum(x["INPUT"]["active_path"] is None for x in rows)
    # alarm-channel provenance (real A2/A3 + derived A1)
    alarm_stats = {
        "any_trigger": sum(x["CERTAINTY"]["alarm"]["any_trigger"] for x in rows),
        "study_inadequacy_A2": sum(x["CERTAINTY"]["alarm"]["study_inadequacy"]["present"] for x in rows),
        "discordance_A3": sum(x["CERTAINTY"]["alarm"]["discordance"]["present"] for x in rows),
        "advanced_question_A1": sum(x["CERTAINTY"]["alarm"]["advanced_question"]["present"] for x in rows),
        "alarm_parse_error": sum(x["CERTAINTY"]["alarm"]["alarm_parse_error"] for x in rows),
        "rows_without_alarm_json": sum(not x["CERTAINTY"]["alarm"]["alarm_present"] for x in rows),
        "missing_alarm_cases": [f"{d}_{h}" for d, h in sorted(missing_alarm)],
    }
    resolve_dist = Counter(
        x["CERTAINTY"]["alarm"]["resolve"]["addresses_alarm"]
        for x in rows if x["CERTAINTY"]["alarm"]["any_trigger"])
    manifest = {
        "n_rows": len(rows),
        "source_csv": str((ann_dir / "belief_deviation_filtered.csv").relative_to(ROOT)),
        "ann_dir": str(ann_dir.relative_to(ROOT)),
        "alarm_dir": str(alarm_dir.relative_to(ROOT)),
        "alarm_channel": "A2/A3 from offline LLM alarm pass + A1 derived algorithmically "
                         "(belief max_p>=%.2f on a rubric disease & severity/etiology/"
                         "complication gap)" % _A1_TAU,
        "rubric_library": "rubric_library.json (all 4 disease sub-rubrics + open 'other'; "
                          "referenced by every row via INPUT.rubric_library_ref)",
        "config": vars(args),
        "filters": {
            "total_steps": len(df),
            "excluded_monitoring": int(df.excluded_monitoring.fillna(False).sum()),
            "rrn_misaligned": int((~df.rrn_aligned.fillna(False)).sum()),
            "include_misaligned": args.include_misaligned,
        },
        "skipped": dict(skipped),
        "when_action_dist": dict(when_dist),
        "verification_dist": dict(vind_dist),
        "by_disease": dict(by_disease),
        "alarm_stats": alarm_stats,
        "alarm_resolve_dist_over_triggered": dict(resolve_dist),
        "should_suppress_n": int(suppress_n),
        "should_suppress_basis": "alarm silent (no A2/A3/A1) & belief concentrated "
                                 f"(max_p>={_SUPPRESS_TAU}, other_mass<={_SUPPRESS_OTHER_MAX}) "
                                 "& same-modality repeat (§2.5 criterion 2)",
        "first_step_rows_no_active_path": first_step,
        "rubric_serving": "FULL triage every step (no leakage of belief argmax); agent "
                          "self-routes, deviate judged wrt the agent's argmax belief rubric. "
                          "INPUT.active_path = prior-step belief argmax (past info).",
        "provisional_timing_filter": True,
        "note": "Framework build on PROVISIONAL text-hint timing filter; re-run unchanged "
                "when real MIMIC charttime lands (same CSV interface).",
    }
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"wrote {out_jsonl}  ({len(rows)} rows)")
    print(f"wrote {out_dir / 'train_manifest.json'}")
    print(f"\nby disease : {dict(by_disease)}")
    print(f"when_action: {dict(when_dist)}")
    print(f"verification: {dict(vind_dist)}")
    print(f"alarm      : any={alarm_stats['any_trigger']}/{len(rows)}  "
          f"A2={alarm_stats['study_inadequacy_A2']} A3={alarm_stats['discordance_A3']} "
          f"A1={alarm_stats['advanced_question_A1']}  "
          f"resolve={dict(resolve_dist)}")
    if alarm_stats["alarm_parse_error"] or alarm_stats["rows_without_alarm_json"]:
        print(f"alarm warn : parse_error={alarm_stats['alarm_parse_error']} "
              f"rows_without_alarm_json={alarm_stats['rows_without_alarm_json']} "
              f"missing_cases={alarm_stats['missing_alarm_cases']}")
    print(f"should_suppress ({args.suppress_mode}): {suppress_n}")
    if skipped:
        print(f"skipped    : {dict(skipped)}")

    for x in rows[:args.preview]:
        print("\n" + "=" * 80)
        print(json.dumps(x, indent=2, ensure_ascii=False)[:2500])


if __name__ == "__main__":
    main()
