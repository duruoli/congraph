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
                   / other_mass); alarm = PROVISIONAL keyword signal (discordance /
                   study-adequacy / dual-stream conflict in the ex-ante text) + the
                   count of prior-step disconfirmations (visible at this step).
                   ⚠ alarm is the provisional rule version pending the clarify
                   decision (keyword vs light-LLM tag) — see HANDOFF §8 transform 3.
  4. HYBRID TARGET  default TARGET = the doctor's actual action (imitation). Steps
                   flagged should-suppress (DD-1 proxy: deviate + disconfirmed at a
                   terminal/blocked rubric state = over-imaging past a valid STOP)
                   are either down-weighted (default) or relabeled when_action->stop
                   (--suppress-mode stop). FD-2 stale-anatomy needs charttime -> TODO.
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

# --- transform 4: should-suppress (DD-1 proxy) -----------------------------
# DD-1 = deviate + disconfirmed while the rubric is at a valid terminal / blocked
# state = defensive over-imaging past a STOP (HANDOFF §2-B4 / §8 transform 4).
# FD-2 (stale anatomy) is intentionally NOT included: it needs real charttime to
# separate this-admission post-op scans from old surgery -> deferred until source
# data lands (annotation_agent_design post-intervention block).
_TERMINAL_STATES = {"terminal_confirmed", "terminal_excluded",
                    "terminal_low_risk", "blocked"}


def is_should_suppress(row) -> bool:
    return (row.dev_belief == "deviate"
            and row.verification == "disconfirmed"
            and row.rubric_state in _TERMINAL_STATES)


# --- transform 3: provisional alarm-channel keyword signals -----------------
_ALARM_PATTERNS = {
    "discordance": re.compile(
        r"discordan|discrepan|out of proportion|inconsistent|does not (?:match|fit)"
        r"|atypical|unexpected|cannot (?:be )?reconcile|mismatch|conflict", re.I),
    "study_adequacy": re.compile(
        r"inadequate|limited study|technically limited|non-?diagnostic|suboptimal"
        r"|not (?:well )?visualized|could not be (?:visualized|assessed)"
        r"|poorly visualized|incomplete (?:study|exam)|equivocal|nonspecific", re.I),
    "dual_stream": re.compile(
        r"persist\w* despite|despite (?:a )?(?:normal|negative|benign)"
        r"|clinical\w*[^.]{0,40}(?:but|versus|vs\.?)[^.]{0,40}(?:imag|exam|lab)"
        r"|worsening despite", re.I),
}


def alarm_signal(ex_ante: dict, prior_disconfirms: int) -> dict:
    """PROVISIONAL alarm channel: keyword flags over the ex-ante text + the running
    count of prior-step disconfirmations (results visible at this step, not leakage).
    Marked provisional — swap for the light-LLM tag once that clarify is resolved."""
    text = " ".join(str(ex_ante.get(k, "")) for k in
                    ("information_gap", "expected_finding", "reasoning"))
    flags = {name: bool(p.search(text)) for name, p in _ALARM_PATTERNS.items()}
    score = sum(flags.values()) + prior_disconfirms
    return {"score": score, "flags": flags,
            "prior_disconfirmations": prior_disconfirms, "method": "provisional_keyword"}


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


def build_row(r, case, masked, suppress_mode, suppress_weight,
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
    suppress = is_should_suppress(r)
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
                "belief": belief_summary(ex.get("differential")).get("dist"),
                "other_hypothesis": ex.get("other_hypothesis", ""),
                "information_gap": ex.get("information_gap", ""),
                "expected_finding": ex.get("expected_finding", ""),
                "action_role": ex.get("action_role", ""),
                "grounding": ex.get("grounding", []),
            },
        },

        "CERTAINTY": {                              # derived; NOT fed as INPUT verbatim
            "belief": belief_summary(ex.get("differential")),
            "alarm": alarm_signal(ex, prior_disconfirms),
        },

        "REWARD": {                                 # ex-post; NEVER in INPUT
            "verification": vind.get("verification", r.verification),
            "certainty_update": vind.get("certainty_update", ""),
            "appropriateness": ex.get("appropriateness", ""),
            "appropriateness_reason": ex.get("appropriateness_reason", ""),
            "sample_weight": weight,
            "should_suppress": bool(suppress),
            "suppress_basis": "DD1_overimaging_past_stop" if suppress else "",
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
    ap.add_argument("--out", default="data/training_set")
    ap.add_argument("--suppress-mode", choices=["downweight", "stop", "off"],
                    default="downweight")
    ap.add_argument("--suppress-weight", type=float, default=0.3)
    ap.add_argument("--include-misaligned", action="store_true")
    ap.add_argument("--preview", type=int, default=0)
    args = ap.parse_args()

    ann_dir = ROOT / args.ann_dir if not Path(args.ann_dir).is_absolute() else Path(args.ann_dir)
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
        row = build_row(r, case, masked, args.suppress_mode, args.suppress_weight,
                        pdc, active_path)
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
    manifest = {
        "n_rows": len(rows),
        "source_csv": str((ann_dir / "belief_deviation_filtered.csv").relative_to(ROOT)),
        "ann_dir": str(ann_dir.relative_to(ROOT)),
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
        "should_suppress_n": int(suppress_n),
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
    print(f"should_suppress ({args.suppress_mode}): {suppress_n}")
    if skipped:
        print(f"skipped    : {dict(skipped)}")

    for x in rows[:args.preview]:
        print("\n" + "=" * 80)
        print(json.dumps(x, indent=2, ensure_ascii=False)[:2500])


if __name__ == "__main__":
    main()
