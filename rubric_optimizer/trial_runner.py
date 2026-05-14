"""trial_runner.py — Evaluate all patients through the rubric pipeline.

Produces a list of PatientRecord (one per patient × chosen step) with
enough information for downstream failure analysis and attribution.

Usage
-----
  python rubric_optimizer/trial_runner.py                    # first-step, all diseases
  python rubric_optimizer/trial_runner.py --step last        # final-step
  python rubric_optimizer/trial_runner.py --disease chole*   # one disease
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── sys.path: project root (for pipeline modules) + this package ───────────
_PKG  = Path(__file__).parent
_ROOT = _PKG.parent
for _p in [str(_ROOT), str(_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.clinical_session import ClinicalSession
from pipeline.diagnosis_distribution import _GRAPH_COND_EDGE_COUNTS

DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]

RESULTS_DIR: Path = _ROOT / "results"
DISEASE_FILES: dict[str, Path] = {
    d: RESULTS_DIR / f"{d}_features.json" for d in DISEASES
}


# ---------------------------------------------------------------------------
# TraversalSummary — compact, serialisable per-disease traversal snapshot
# ---------------------------------------------------------------------------

@dataclass
class TraversalSummary:
    """What the rubric engine did for one disease at one patient step."""
    disease:              str
    conditional_triggers: int       # data-driven edge firings
    total_cond_edges:     int       # total conditional edges in this graph
    trigger_fraction:     float     # conditional_triggers / total_cond_edges
    terminal_type:        str | None  # "terminal_confirmed" | "terminal_excluded" | "terminal_low_risk" | None
    terminal_node:        str | None
    triage_activated:     bool
    frontier:             list[str]   # node IDs where traversal stopped


# ---------------------------------------------------------------------------
# PatientRecord — one patient × one step evaluated
# ---------------------------------------------------------------------------

@dataclass
class PatientRecord:
    """Full snapshot of one pipeline evaluation."""
    pid:            str
    true_disease:   str
    step_index:     int
    step_label:     str
    features:       dict

    # Diagnosis distribution
    predicted:      str               # primary diagnosis (argmax)
    is_correct:     bool
    probabilities:  dict[str, float]  # P(d) for all 4 diseases
    raw_scores:     dict[str, float]  # pre-softmax weighted scores

    # Per-disease rubric traversal
    traversal:      dict[str, TraversalSummary]  # disease → summary


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_disease(disease: str) -> dict[str, list[dict]]:
    """Return {patient_id: [step, ...]} from the results JSON."""
    path = DISEASE_FILES[disease]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["results"]


# ---------------------------------------------------------------------------
# Per-step evaluation
# ---------------------------------------------------------------------------

def _evaluate_step(pid: str, true_disease: str, step: dict) -> PatientRecord:
    """Run one step of one patient through the full pipeline."""
    features = step["features"]
    session  = ClinicalSession(features)
    state    = session.assess()

    traversal_summaries: dict[str, TraversalSummary] = {}
    for d, tr in state.traversal.diseases.items():
        total_cond = _GRAPH_COND_EDGE_COUNTS.get(d, 1)
        frac       = tr.conditional_triggers / total_cond if total_cond else 0.0
        traversal_summaries[d] = TraversalSummary(
            disease              = d,
            conditional_triggers = tr.conditional_triggers,
            total_cond_edges     = total_cond,
            trigger_fraction     = frac,
            terminal_type        = tr.terminal_type,
            terminal_node        = tr.terminal_node,
            triage_activated     = tr.triage_activated,
            frontier             = list(tr.frontier),
        )

    return PatientRecord(
        pid          = pid,
        true_disease = true_disease,
        step_index   = step["step_index"],
        step_label   = step["step_label"],
        features     = dict(features),
        predicted    = state.primary_diagnosis,
        is_correct   = (state.primary_diagnosis == true_disease),
        probabilities = dict(state.distribution.probabilities),
        raw_scores    = dict(state.distribution.raw_scores),
        traversal     = traversal_summaries,
    )


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trial(
    diseases:      list[str] | None = None,
    step_selector: str = "first",   # "first" | "last" | "all"
) -> list[PatientRecord]:
    """
    Run the rubric pipeline on all patients and return PatientRecord list.

    Parameters
    ----------
    diseases      : limit to these diseases (default: all 4)
    step_selector : which steps to evaluate per patient
                    "first" — only the first step (HPI + PE + Labs)
                    "last"  — only the last step (maximum evidence)
                    "all"   — every step (produces multiple records per patient)
    """
    diseases = diseases or DISEASES
    records:  list[PatientRecord] = []

    for disease in diseases:
        patients = load_disease(disease)
        for pid, steps in patients.items():
            if   step_selector == "first": selected = [steps[0]]
            elif step_selector == "last":  selected = [steps[-1]]
            else:                          selected = steps

            for step in selected:
                records.append(_evaluate_step(pid, disease, step))

    return records


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(records: list[PatientRecord]) -> dict:
    """Compute per-disease and overall accuracy."""
    per_disease: dict[str, dict] = {}
    for d in DISEASES:
        sub     = [r for r in records if r.true_disease == d]
        correct = [r for r in sub if r.is_correct]
        if not sub:
            continue
        per_disease[d] = {
            "n":        len(sub),
            "correct":  len(correct),
            "accuracy": len(correct) / len(sub),
        }

    n_total   = len(records)
    n_correct = sum(1 for r in records if r.is_correct)
    return {
        "overall": {
            "n":        n_total,
            "correct":  n_correct,
            "accuracy": n_correct / n_total if n_total else 0.0,
        },
        "per_disease": per_disease,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Run one rubric optimizer trial.")
    p.add_argument("--disease", choices=DISEASES, default=None,
                   help="Limit to one disease (default: all).")
    p.add_argument("--step", choices=["first", "last", "all"], default="first",
                   help="Which patient step to evaluate (default: first).")
    args = p.parse_args()

    diseases = [args.disease] if args.disease else None
    records  = run_trial(diseases=diseases, step_selector=args.step)
    metrics  = compute_metrics(records)

    W = 60
    print(f"\n{'═'*W}")
    print(f"  TRIAL RUN  (step={args.step}, n={len(records)} records)")
    print(f"{'═'*W}")
    ov = metrics["overall"]
    print(f"  Overall : {ov['correct']}/{ov['n']}  ({ov['accuracy']:.1%})")
    print(f"  {'─'*56}")
    for d, m in metrics["per_disease"].items():
        print(f"  {d:<16}  {m['correct']:>4}/{m['n']:<4}  ({m['accuracy']:.1%})")
    print(f"{'═'*W}\n")


class TrialRunner:
    """Programmatic entry point for batch rubric trials (see :func:`run_trial`)."""

    @staticmethod
    def run_trial(
        diseases: list[str] | None = None,
        step_selector: str = "first",
    ) -> list[PatientRecord]:
        return run_trial(diseases=diseases, step_selector=step_selector)

    @staticmethod
    def compute_metrics(records: list[PatientRecord]) -> dict:
        return compute_metrics(records)


if __name__ == "__main__":
    main()
