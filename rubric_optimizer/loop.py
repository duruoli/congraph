"""loop.py — Main entry point for the rubric optimizer.

This script is the orchestrator.  In the current interactive mode it:
  1. Runs a trial evaluation (all patients, first step by default)
  2. Performs failure analysis on the worst-performing disease
  3. Builds and prints the LLM summary
  4. Logs the trial to results/rubric_optimizer_log.jsonl

After reading the summary, you (or an LLM API call) propose a code change
following the format in the INSTRUCTION block.  Then:
  - Apply the change to rubric_graph.py or diagnosis_distribution.py
  - Run this script again
  - If accuracy improves, commit and continue; otherwise revert with git checkout

Usage
-----
  # Baseline trial (first run — logs trial #0)
  python rubric_optimizer/loop.py

  # Subsequent trials (after applying a proposed change)
  python rubric_optimizer/loop.py

  # Override which step to evaluate
  python rubric_optimizer/loop.py --step last

  # Focus on a specific disease's failures
  python rubric_optimizer/loop.py --focus cholecystitis

  # Custom log file location
  python rubric_optimizer/loop.py --log /tmp/opt_log.jsonl
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

_PKG  = Path(__file__).parent
_ROOT = _PKG.parent
for _p in [str(_ROOT), str(_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trial_runner   import run_trial, compute_metrics, DISEASES
from failure_analyzer import analyze_failures, FailureAnalysis
from trial_log      import (
    TrialRecord, MetricSnapshot, ChangeRecord,
    append_trial, load_trials, make_timestamp, next_trial_id,
)
from summarizer     import build_summary


# ---------------------------------------------------------------------------
# Helper: pick worst disease (lowest accuracy)
# ---------------------------------------------------------------------------

def _worst_disease(metrics: dict, focus: str | None = None) -> str:
    if focus and focus in metrics["per_disease"]:
        return focus
    return min(
        metrics["per_disease"],
        key=lambda d: metrics["per_disease"][d]["accuracy"],
    )


# ---------------------------------------------------------------------------
# Core: run one trial and emit its summary
# ---------------------------------------------------------------------------

def run_one_trial(
    step_selector: str = "first",
    focus:         str | None = None,
    log_path:      str | None = None,
    outcome:       str = "baseline",
    outcome_reason: str = "No change applied — baseline trial.",
    change_applied: ChangeRecord | None = None,
) -> TrialRecord:
    """
    Run a full evaluation, do failure analysis, print summary, log to JSONL.

    Parameters
    ----------
    step_selector  : "first" | "last" | "all"
    focus          : force failure analysis on this disease (override worst)
    log_path       : path to JSONL log file (None = default)
    outcome        : "baseline" | "accepted" | "rejected"
    outcome_reason : explanation string
    change_applied : ChangeRecord if a change was applied before this trial
    """
    history  = load_trials(log_path)
    trial_id = len(history)

    # ── 1. Evaluate ────────────────────────────────────────────────────────
    print(f"\n[Trial #{trial_id}] Evaluating ({step_selector} step, {len(DISEASES)} diseases)…")
    records = run_trial(step_selector=step_selector)
    metrics = compute_metrics(records)

    # ── 2. Identify worst disease & failure analysis ───────────────────────
    worst = _worst_disease(metrics, focus)
    fa    = analyze_failures(records, worst)

    # ── 3. Build delta vs previous trial ──────────────────────────────────
    prev_acc = history[-1].metrics.overall_accuracy if history else None
    delta    = (
        metrics["overall"]["accuracy"] - prev_acc
        if prev_acc is not None else None
    )

    # ── 4. Construct TrialRecord ───────────────────────────────────────────
    trial = TrialRecord(
        trial_id       = trial_id,
        timestamp      = make_timestamp(),
        step_selector  = step_selector,
        metrics        = MetricSnapshot(
            overall_accuracy = metrics["overall"]["accuracy"],
            per_disease      = {
                d: m["accuracy"] for d, m in metrics["per_disease"].items()
            },
        ),
        delta_accuracy  = delta,
        worst_disease   = worst,
        change_applied  = change_applied,
        outcome         = outcome,
        outcome_reason  = outcome_reason,
        attribution     = fa.attribution_hypothesis,
    )

    # ── 5. Log ────────────────────────────────────────────────────────────
    append_trial(trial, log_path)
    print(f"  Logged → {log_path or 'results/rubric_optimizer_log.jsonl'}")

    # ── 6. Print summary ──────────────────────────────────────────────────
    summary = build_summary(trial, history, fa)
    print(summary)

    return trial


# ---------------------------------------------------------------------------
# Acceptance check helper (for use after applying a proposed change)
# ---------------------------------------------------------------------------

def evaluate_change(
    prev_trial:    TrialRecord,
    change:        ChangeRecord,
    step_selector: str = "first",
    log_path:      str | None = None,
    w_acc: float = 1.0,
) -> TrialRecord:
    """
    Re-evaluate after applying a proposed change and decide accept/reject.

    A change is ACCEPTED when overall_accuracy improves (or is unchanged)
    relative to the previous trial.  If it degrades, it is REJECTED and you
    should revert with:  git checkout rubric_graph.py  (or diagnosis_distribution.py)
    """
    new_trial = run_one_trial(
        step_selector  = step_selector,
        log_path       = log_path,
        outcome        = "pending",       # will be overwritten below
        outcome_reason = "(computing…)",
        change_applied = change,
    )

    prev_acc = prev_trial.metrics.overall_accuracy
    new_acc  = new_trial.metrics.overall_accuracy

    if new_acc >= prev_acc:
        outcome        = "accepted"
        outcome_reason = (
            f"Accuracy improved from {prev_acc:.1%} → {new_acc:.1%} "
            f"(+{new_acc - prev_acc:.1%})."
        )
    else:
        outcome        = "rejected"
        outcome_reason = (
            f"Accuracy degraded from {prev_acc:.1%} → {new_acc:.1%} "
            f"({new_acc - prev_acc:.1%}). Revert with: "
            f"git checkout {change.target_file}"
        )

    # Update the record in-place and re-log (overwrite is not supported by
    # JSONL append; we just log a corrected entry with the same trial_id)
    new_trial.outcome        = outcome
    new_trial.outcome_reason = outcome_reason
    append_trial(new_trial, log_path)

    print(f"\n[Trial #{new_trial.trial_id}]  {outcome.upper()}  — {outcome_reason}")
    return new_trial


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one rubric optimizer trial and print the LLM summary."
    )
    p.add_argument(
        "--step",
        choices=["first", "last", "all"],
        default="first",
        help="Which patient step to evaluate (default: first).",
    )
    p.add_argument(
        "--focus",
        choices=DISEASES,
        default=None,
        metavar="DISEASE",
        help="Force failure analysis on this disease instead of the worst.",
    )
    p.add_argument(
        "--log",
        default=None,
        metavar="PATH",
        help="Path to JSONL trial log file (default: results/rubric_optimizer_log.jsonl).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_one_trial(
        step_selector = args.step,
        focus         = args.focus,
        log_path      = args.log,
    )


if __name__ == "__main__":
    main()
