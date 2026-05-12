"""auto_loop.py — Fully automated rubric optimizer using LLM API.

Runs up to N iterations of:
  1. Build trial summary (metrics + failure analysis + rubric source)
  2. Call GPT-4o → get a proposed code change
  3. Apply the change to rubric_graph.py or diagnosis_distribution.py
  4. Reload affected Python modules (no subprocess restart needed)
  5. Re-evaluate accuracy → accept or reject
  6. If rejected: git checkout to revert the file, reload modules

Early-stop conditions (either triggers a stop):
  - N consecutive non-improvements (default: 3) — counts both rejected changes
    AND accepted-but-neutral (0% gain) changes
  - Overall accuracy reaches target  (default: 0.80)

Requirements
------------
  .env file in the project root containing:
      OPENAI_API_KEY=sk-...

  Run with the congraph conda env:
      conda run -n congraph python rubric_optimizer/auto_loop.py

Usage
-----
  python rubric_optimizer/auto_loop.py
  python rubric_optimizer/auto_loop.py --n 20 --target-acc 0.75
  python rubric_optimizer/auto_loop.py --step last --model gpt-4o
  python rubric_optimizer/auto_loop.py --focus diverticulitis --n 10
"""

from __future__ import annotations

import importlib
import linecache
import os
import sys
import argparse
from pathlib import Path

# ── sys.path: project root + this package ──────────────────────────────────
_PKG  = Path(__file__).parent
_ROOT = _PKG.parent
for _p in [str(_ROOT), str(_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Load .env BEFORE importing openai ──────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from openai import OpenAI

# ── Import all modules that may be reloaded.
#    IMPORTANT: use `import module` (not `from module import func`) so that
#    module-level references stay live after importlib.reload().
import rubric_graph          # noqa: E402
import traversal_engine      # noqa: E402
import diagnosis_distribution  # noqa: E402
import clinical_session      # noqa: E402
import trial_runner          # noqa: E402
import failure_analyzer      # noqa: E402
import summarizer            # noqa: E402
import trial_log             # noqa: E402

# Local rubric_optimizer helpers
import llm_proposer    # noqa: E402
import change_applier  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LOG_PATH      = _ROOT / "results" / "rubric_optimizer" / "auto_loop_log.jsonl"
REGRESSION_THRESHOLD  = 0.02   # max per-disease drop below historical best
DEFAULT_MAX_NO_IMPROVE = 3     # consecutive rejection limit


# ---------------------------------------------------------------------------
# Module reload helpers
# ---------------------------------------------------------------------------

# Correct reload order respects the import dependency graph:
#   rubric_graph  ← traversal_engine  ← clinical_session  ← trial_runner
#                ←  failure_analyzer
#                ←  summarizer
#   diagnosis_distribution  ← clinical_session  ← trial_runner

_RELOAD_RUBRIC_ORDER = [
    "rubric_graph",
    "traversal_engine",
    "clinical_session",
    "trial_runner",
    "failure_analyzer",
    "summarizer",
]

_RELOAD_DISTR_ORDER = [
    "diagnosis_distribution",
    "clinical_session",
    "trial_runner",
    "failure_analyzer",
    "summarizer",
]


def _reload_affected(changed_file: str) -> None:
    """
    Reload modules whose behavior changes when changed_file is modified.

    After reload, any attribute access via `module.attr` sees the new code.
    (Attributes cached with `from module import attr` are NOT updated —
    that's why we use `import module` throughout this file.)
    """
    order = (
        _RELOAD_RUBRIC_ORDER if changed_file == "rubric_graph.py"
        else _RELOAD_DISTR_ORDER
    )
    # Invalidate linecache so inspect.getsource() reads the updated file
    linecache.clearcache()
    for mod_name in order:
        mod = sys.modules.get(mod_name)
        if mod is not None:
            importlib.reload(mod)
            print(f"    ↺  reloaded {mod_name}")


# ---------------------------------------------------------------------------
# Trial helpers (thin wrappers that always go through module references)
# ---------------------------------------------------------------------------

def _run_trial(step_selector: str) -> tuple:
    """
    Run a fresh diagnostic trial.

    Returns
    -------
    (records, metrics, worst_disease, failure_analysis)
      records  : list[PatientRecord]
      metrics  : dict with "overall" and "per_disease" keys
      worst    : name of the disease with lowest accuracy
      fa       : FailureAnalysis for worst_disease
    """
    records = trial_runner.run_trial(step_selector=step_selector)
    metrics = trial_runner.compute_metrics(records)

    worst = min(
        metrics["per_disease"],
        key=lambda d: metrics["per_disease"][d]["accuracy"],
    )
    fa = failure_analyzer.analyze_failures(records, worst)
    return records, metrics, worst, fa


def _build_and_log_trial(
    step_selector:  str,
    log_path:       str,
    metrics:        dict,
    worst:          str,
    fa,
    outcome:        str,
    outcome_reason: str,
    change_applied,
    focus:          str | None = None,
) -> tuple:
    """
    Construct a TrialRecord, append it to the log, print the summary.

    If focus is set and present in per_disease metrics, the failure analysis
    uses focus instead of worst (only affects summary display; worst in the
    record reflects the actual worst).

    Returns
    -------
    (trial_record, summary_string)
    """
    history  = trial_log.load_trials(log_path)
    trial_id = len(history)

    prev_acc = history[-1].metrics.overall_accuracy if history else None
    delta    = (
        metrics["overall"]["accuracy"] - prev_acc
        if prev_acc is not None else None
    )

    # If focus was requested, override the failure analysis for summary display
    display_fa = fa
    display_worst = worst
    if focus and focus in metrics["per_disease"] and focus != worst:
        display_worst = focus
        # We'd need records for this but we don't have them here; keep fa as-is.
        # The summary will still use the computed worst for the FA section.

    trial = trial_log.TrialRecord(
        trial_id       = trial_id,
        timestamp      = trial_log.make_timestamp(),
        step_selector  = step_selector,
        metrics        = trial_log.MetricSnapshot(
            overall_accuracy = metrics["overall"]["accuracy"],
            per_disease      = {
                d: m["accuracy"]
                for d, m in metrics["per_disease"].items()
            },
        ),
        delta_accuracy  = delta,
        worst_disease   = worst,
        change_applied  = change_applied,
        outcome         = outcome,
        outcome_reason  = outcome_reason,
        attribution     = display_fa.attribution_hypothesis,
    )

    trial_log.append_trial(trial, log_path)

    summary = summarizer.build_summary(trial, history, display_fa)
    print(summary)

    return trial, summary


def _best_per_disease(history: list) -> dict[str, float]:
    """Return highest accuracy ever recorded per disease across all trials."""
    best: dict[str, float] = {}
    for t in history:
        for d, acc in t.metrics.per_disease.items():
            if d not in best or acc > best[d]:
                best[d] = acc
    return best


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_auto_loop(
    n_iterations:       int   = 15,
    target_accuracy:    float = 0.80,
    max_no_improvement: int   = DEFAULT_MAX_NO_IMPROVE,
    step_selector:      str   = "first",
    model:              str   = "gpt-4o",
    log_path:           str | None = None,
    focus:              str | None = None,
) -> None:
    """
    Run the automated rubric optimization loop.

    Parameters
    ----------
    n_iterations       : maximum number of optimization iterations
    target_accuracy    : stop early if overall accuracy reaches this
    max_no_improvement : stop early after this many consecutive rejections
    step_selector      : "first" | "last" | "all" — which patient step to evaluate
    model              : OpenAI model name for proposing changes
    log_path           : JSONL log file path (None → default auto_loop_log.jsonl)
    focus              : disease name to focus failure analysis on (None → worst)
    """
    log_path = log_path or str(DEFAULT_LOG_PATH)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    # ── API key ─────────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "\n[ERROR] OPENAI_API_KEY not set.\n"
            "Create a .env file in the project root:\n"
            "  echo 'OPENAI_API_KEY=sk-...' > .env\n"
            "or export it in your shell: export OPENAI_API_KEY=sk-..."
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    W = 72
    print(f"\n{'═'*W}")
    print(f"  AUTOMATED RUBRIC OPTIMIZER")
    print(f"  Model            : {model}")
    print(f"  Max iterations   : {n_iterations}")
    print(f"  Target accuracy  : {target_accuracy:.0%}")
    print(f"  Max consecutive  : {max_no_improvement} rejections → early stop")
    print(f"  Step selector    : {step_selector}")
    print(f"  Log file         : {log_path}")
    print(f"{'═'*W}\n")

    # ── Baseline trial ───────────────────────────────────────────────────────
    print("[Step 0] Running baseline trial…")
    records, metrics, worst, fa = _run_trial(step_selector)
    if focus:
        worst_for_fa = focus if focus in metrics["per_disease"] else worst
        fa = failure_analyzer.analyze_failures(
            [r for r in records], worst_for_fa
        )

    prev_trial, current_summary = _build_and_log_trial(
        step_selector  = step_selector,
        log_path       = log_path,
        metrics        = metrics,
        worst          = worst,
        fa             = fa,
        outcome        = "baseline",
        outcome_reason = "Automated loop — baseline evaluation.",
        change_applied = None,
        focus          = focus,
    )

    consecutive_rejections = 0
    accepted_count         = 0
    rejected_count         = 0
    last_iteration         = 0

    # ── Optimization loop ────────────────────────────────────────────────────
    for iteration in range(1, n_iterations + 1):
        last_iteration = iteration
        acc_str = f"{prev_trial.metrics.overall_accuracy:.1%}"

        print(f"\n{'═'*W}")
        print(f"  ITERATION {iteration}/{n_iterations}   "
              f"current_acc={acc_str}   "
              f"consecutive_rej={consecutive_rejections}")
        print(f"{'═'*W}")

        # ── 1. Call LLM ──────────────────────────────────────────────────────
        print(f"\n[Iter {iteration}] Calling {model} for a change proposal…")
        change = llm_proposer.propose_change(current_summary, client, model=model)

        if change is None:
            print(f"[Iter {iteration}] No valid proposal obtained — skipping iteration.")
            consecutive_rejections += 1
        else:
            print(f"\n[Iter {iteration}] Proposed change:")
            print(f"  File       : {change.target_file}")
            print(f"  Type       : {change.change_type}")
            print(f"  Description: {change.description}")

            # ── 2. Apply change ───────────────────────────────────────────────
            ok, err = change_applier.apply_change(change)
            if not ok:
                print(f"\n[Iter {iteration}] Could not apply change: {err}")
                consecutive_rejections += 1
                # Summary stays the same (code unchanged)
                continue

            print(f"\n[Iter {iteration}] Applied. Reloading affected modules…")
            _reload_affected(change.target_file)

            # ── 3. Evaluate ───────────────────────────────────────────────────
            print(f"\n[Iter {iteration}] Evaluating with changed code…")
            records_new, metrics_new, worst_new, fa_new = _run_trial(step_selector)

            new_acc  = metrics_new["overall"]["accuracy"]
            prev_acc = prev_trial.metrics.overall_accuracy

            # Per-disease regression guard (same as loop.evaluate_change)
            history_before = trial_log.load_trials(log_path)
            best_pd        = _best_per_disease(history_before)

            regressions: list[str] = []
            for disease, d_m in metrics_new["per_disease"].items():
                d_acc = d_m["accuracy"]
                floor = best_pd.get(disease, 0.0) - REGRESSION_THRESHOLD
                if d_acc < floor:
                    regressions.append(
                        f"{disease}: {d_acc:.1%} < floor {floor:.1%}"
                    )

            # ── 4. Accept / reject ────────────────────────────────────────────
            if new_acc >= prev_acc and not regressions:
                improved = new_acc > prev_acc   # strict improvement
                outcome        = "accepted"
                outcome_reason = (
                    f"{prev_acc:.1%} → {new_acc:.1%} "
                    f"({'+'if improved else '='}{new_acc - prev_acc:.1%}); "
                    f"all diseases within {REGRESSION_THRESHOLD:.0%} of best."
                )

                new_trial, new_summary = _build_and_log_trial(
                    step_selector  = step_selector,
                    log_path       = log_path,
                    metrics        = metrics_new,
                    worst          = worst_new,
                    fa             = fa_new,
                    outcome        = outcome,
                    outcome_reason = outcome_reason,
                    change_applied = change,
                    focus          = focus,
                )

                if improved:
                    print(f"\n  ✓ ACCEPTED (+improvement)  "
                          f"{prev_acc:.1%} → {new_acc:.1%} "
                          f"(+{new_acc - prev_acc:.1%})")
                    consecutive_rejections = 0   # only reset on real gain
                else:
                    print(f"\n  ~ ACCEPTED (no gain, neutral)  "
                          f"{prev_acc:.1%} → {new_acc:.1%}  "
                          f"[consecutive_no_improve stays at {consecutive_rejections}]")
                    consecutive_rejections += 1   # neutral = still no progress

                prev_trial      = new_trial
                current_summary = new_summary
                accepted_count += 1

            else:
                if regressions:
                    outcome_reason = (
                        f"Per-disease regression detected: {'; '.join(regressions)}. "
                        f"Overall {prev_acc:.1%} → {new_acc:.1%}."
                    )
                else:
                    outcome_reason = (
                        f"Overall accuracy degraded "
                        f"{prev_acc:.1%} → {new_acc:.1%} "
                        f"({new_acc - prev_acc:.1%})."
                    )

                _build_and_log_trial(
                    step_selector  = step_selector,
                    log_path       = log_path,
                    metrics        = metrics_new,
                    worst          = worst_new,
                    fa             = fa_new,
                    outcome        = "rejected",
                    outcome_reason = outcome_reason,
                    change_applied = change,
                    focus          = focus,
                )

                print(f"\n  ✗ REJECTED  {prev_acc:.1%} → {new_acc:.1%} "
                      f"({new_acc - prev_acc:.1%})")
                print(f"    Reason: {outcome_reason}")

                # ── 4b. Revert ────────────────────────────────────────────────
                print(f"\n  Reverting {change.target_file} via git checkout…")
                rev_ok, rev_err = change_applier.revert_change(change)

                if not rev_ok:
                    # Fallback: swap new_code ↔ old_code in place
                    print(f"  [WARN] git checkout failed ({rev_err}). "
                          f"Attempting in-place revert…")
                    fallback = trial_log.ChangeRecord(
                        target_file = change.target_file,
                        change_type = change.change_type,
                        description = "auto-revert (git fallback)",
                        old_code    = change.new_code,
                        new_code    = change.old_code,
                        rationale   = "auto-revert",
                    )
                    rb_ok, rb_err = change_applier.apply_change(fallback)
                    if not rb_ok:
                        print(
                            f"\n[FATAL] Could not revert {change.target_file}!\n"
                            f"  Manual fix: git checkout {change.target_file}\n"
                            f"  Error: {rb_err}"
                        )
                        sys.exit(1)
                    print(f"  In-place revert succeeded.")
                else:
                    print(f"  Reverted via git checkout.")

                print(f"  Reloading modules after revert…")
                _reload_affected(change.target_file)

                # Rebuild summary for reverted state (same code as prev_trial)
                # Run trial again to get a fresh FailureAnalysis
                records_rev, metrics_rev, worst_rev, fa_rev = _run_trial(step_selector)
                history_rev = trial_log.load_trials(log_path)
                # Build summary using prev_trial (unchanged) + updated history
                current_summary = summarizer.build_summary(prev_trial, history_rev, fa_rev)

                consecutive_rejections += 1
                rejected_count += 1

        # ── Early-stop checks ────────────────────────────────────────────────
        if consecutive_rejections >= max_no_improvement:
            print(
                f"\n[Auto-stop] {consecutive_rejections} consecutive non-improvements "
                f"(rejected OR accepted-but-neutral) ≥ limit ({max_no_improvement}) "
                f"— stopping early."
            )
            break

        if prev_trial.metrics.overall_accuracy >= target_accuracy:
            print(
                f"\n[Auto-stop] Target accuracy {target_accuracy:.0%} reached "
                f"({prev_trial.metrics.overall_accuracy:.1%}) — stopping."
            )
            break

    # ── Final report ─────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'═'*W}")
    print(f"  Iterations completed : {last_iteration}")
    print(f"  Changes accepted     : {accepted_count}")
    print(f"  Changes rejected     : {rejected_count}")
    print(f"  Final accuracy       : {prev_trial.metrics.overall_accuracy:.1%}")
    print(f"  Per-disease breakdown:")
    for d in trial_runner.DISEASES:
        acc = prev_trial.metrics.per_disease.get(d, float("nan"))
        print(f"    {d:<18} : {acc:.1%}")
    print(f"  Log file             : {log_path}")
    print(f"{'═'*W}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Automated rubric optimizer — uses LLM to iteratively improve rubrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n", type=int, default=15, metavar="N",
        help="Maximum number of optimization iterations.",
    )
    p.add_argument(
        "--target-acc", type=float, default=0.80, metavar="ACC",
        help="Stop early when overall accuracy reaches this value (0–1).",
    )
    p.add_argument(
        "--max-no-improve", type=int, default=DEFAULT_MAX_NO_IMPROVE, metavar="K",
        help="Stop early after K consecutive rejected proposals.",
    )
    p.add_argument(
        "--step", choices=["first", "last", "all"], default="first",
        help="Which patient step to evaluate.",
    )
    p.add_argument(
        "--model", default="gpt-4o",
        help="OpenAI model for change proposals.",
    )
    p.add_argument(
        "--log", default=None, metavar="PATH",
        help="Path to JSONL log file (default: results/rubric_optimizer/auto_loop_log.jsonl).",
    )
    p.add_argument(
        "--focus", choices=trial_runner.DISEASES, default=None, metavar="DISEASE",
        help="Force failure analysis on this disease instead of the worst one.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_auto_loop(
        n_iterations       = args.n,
        target_accuracy    = args.target_acc,
        max_no_improvement = args.max_no_improve,
        step_selector      = args.step,
        model              = args.model,
        log_path           = args.log,
        focus              = args.focus,
    )


if __name__ == "__main__":
    main()
