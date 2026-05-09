"""summarizer.py — Build a structured, LLM-readable trial summary.

build_summary() is the key function.  Its output is:
  • Human-readable for manual review
  • LLM-parseable for automated proposers
  • Self-contained: includes metrics, history, attribution, AND the relevant
    rubric source code, so the LLM doesn't need to ask for context.

The summary ends with an explicit instruction block telling the LLM exactly
what format its response should take.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

_PKG  = Path(__file__).parent
_ROOT = _PKG.parent
for _p in [str(_ROOT), str(_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rubric_graph as _rg
from trial_runner import DISEASES
from trial_log import TrialRecord
from failure_analyzer import FailureAnalysis


# ---------------------------------------------------------------------------
# Source code extraction helpers
# ---------------------------------------------------------------------------

_BUILD_FN: dict[str, str] = {
    "appendicitis":   "_build_appendicitis_graph",
    "cholecystitis":  "_build_cholecystitis_graph",
    "diverticulitis": "_build_diverticulitis_graph",
    "pancreatitis":   "_build_pancreatitis_graph",
}

_HELPER_FNS: dict[str, list[str]] = {
    "cholecystitis":  ["_tg18_group_a", "_tg18_group_b", "_tg18_suspected",
                       "_tg18_us_positive", "_tg18_organ_dysfunction", "_tg18_grade_ii_local"],
    "appendicitis":   ["alvarado_score"],
    "pancreatitis":   ["bisap_score", "_revised_atlanta_criteria_count"],
    "diverticulitis": [],
}


def _get_source(fn_name: str) -> str:
    fn = getattr(_rg, fn_name, None)
    if fn is None:
        return f"# (function {fn_name!r} not found in rubric_graph.py)\n"
    try:
        return inspect.getsource(fn)
    except Exception:
        return f"# (source extraction failed for {fn_name!r})\n"


def _rubric_source(disease: str) -> str:
    """Return the helper functions + _build_X_graph() source for a disease."""
    parts: list[str] = []
    for fn in _HELPER_FNS.get(disease, []):
        parts.append(_get_source(fn))
    build_fn = _BUILD_FN.get(disease)
    if build_fn:
        parts.append(_get_source(build_fn))
    return "\n\n".join(parts)


def _scoring_params_snippet() -> str:
    """Return the scoring hyperparameter block from diagnosis_distribution.py."""
    dd_path = _ROOT / "diagnosis_distribution.py"
    try:
        text  = dd_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        # Find the ACTUAL code line: `CONFIRMED_BONUS:     float = ...`
        start = next(
            (i for i, l in enumerate(lines)
             if l.startswith("CONFIRMED_BONUS")),
            None,
        )
        if start is None:
            return "# (CONFIRMED_BONUS line not found)"
        end = next(
            (i for i, l in enumerate(lines) if i > start and l.startswith("W_TRIAGE")),
            start + 12,
        )
        return "\n".join(lines[start : end + 2])
    except Exception:
        return "# (could not read diagnosis_distribution.py)"


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_summary(
    current_trial:    TrialRecord,
    history:          list[TrialRecord],
    failure_analysis: FailureAnalysis,
    max_history:      int = 4,
) -> str:
    """
    Build the full LLM summary for one trial.

    Parameters
    ----------
    current_trial    : the just-completed TrialRecord
    history          : all previous TrialRecord objects (oldest first)
    failure_analysis : FailureAnalysis for current_trial.worst_disease
    max_history      : how many recent historical trials to include in the prompt
    """
    lines: list[str] = []
    W = 72

    def sep(title: str = "") -> None:
        if title:
            lines.append(f"\n{'─'*W}")
            lines.append(f"  {title}")
            lines.append(f"{'─'*W}")
        else:
            lines.append(f"{'─'*W}")

    # ════════════════════════════════════════════════════════════════════════
    # Header
    # ════════════════════════════════════════════════════════════════════════
    lines.append(f"{'═'*W}")
    lines.append(f"  RUBRIC OPTIMIZER — TRIAL #{current_trial.trial_id}")
    lines.append(f"  {current_trial.timestamp}   step_selector={current_trial.step_selector}")
    lines.append(f"{'═'*W}")

    # ── Current metrics ──────────────────────────────────────────────────────
    sep("CURRENT METRICS")
    m = current_trial.metrics
    delta_str = (
        f"  (Δ {current_trial.delta_accuracy:+.1%} vs previous trial)"
        if current_trial.delta_accuracy is not None
        else "  (baseline — no previous trial)"
    )
    lines.append(f"  Overall accuracy : {m.overall_accuracy:.1%}{delta_str}")
    for d in DISEASES:
        acc    = m.per_disease.get(d, float("nan"))
        marker = "  ← WORST" if d == current_trial.worst_disease else ""
        lines.append(f"    {d:<16} : {acc:.1%}{marker}")

    # ── Trial history ────────────────────────────────────────────────────────
    sep("RECENT TRIAL HISTORY  (oldest → newest)")
    recent = history[-max_history:] if history else []
    if not recent:
        lines.append("  (no previous trials)")
    for t in recent:
        lines.append(
            f"\n  Trial #{t.trial_id}  [{t.outcome.upper():<8}]  "
            f"acc={t.metrics.overall_accuracy:.1%}"
            + (f"  Δ{t.delta_accuracy:+.1%}" if t.delta_accuracy is not None else "")
        )
        if t.change_applied:
            ch = t.change_applied
            lines.append(f"    Change : {ch.change_type} in {ch.target_file}")
            lines.append(f"    Desc   : {ch.description}")
            lines.append(f"    Reason : {t.outcome_reason}")
        else:
            lines.append(f"    Change : (none — baseline)")
        lines.append(f"    Attribution hypothesis was: {t.attribution[:120]}")

    # ════════════════════════════════════════════════════════════════════════
    # Failure analysis
    # ════════════════════════════════════════════════════════════════════════
    fa = failure_analysis
    sep(
        f"FAILURE ANALYSIS — {fa.disease.upper()}   "
        f"{fa.n_failures} failures / {fa.n_failures + fa.n_correct} patients "
        f"({fa.n_failures / (fa.n_failures + fa.n_correct):.0%} error rate)"
    )

    lines.append(f"  Most confused as : {fa.most_confused_as}  "
                 f"{fa.confused_as_counts}")
    lines.append(
        f"  Rubric activation (cond_triggers / total_cond_edges):\n"
        f"    failures : {fa.mean_trig_frac_failures:.1%}\n"
        f"    correct  : {fa.mean_trig_frac_correct:.1%}"
    )

    # Edge gaps
    if fa.edge_gaps:
        lines.append(f"\n  ── Edge condition trigger rates (|gap| ranked) ──")
        lines.append(
            f"  {'Source → Target':<38}  {'Edge label':<36}  "
            f"{'fail%':>5}  {'corr%':>5}  {'gap':>6}"
        )
        lines.append(f"  {'─'*90}")
        for eg in fa.edge_gaps[:8]:
            src_tgt = f"{eg.source}→{eg.target}"
            lines.append(
                f"  {src_tgt:<38}  {eg.edge_label:<36.36}  "
                f"{eg.rate_in_failures:>5.0%}  {eg.rate_in_correct:>5.0%}  "
                f"{eg.gap:>+6.0%}"
            )

    # Feature gaps
    if fa.feature_gaps:
        lines.append(f"\n  ── Feature presence rates in failures vs correct (|gap| ranked) ──")
        lines.append(
            f"  {'Feature':<38}  {'fail%':>5}  {'corr%':>5}  {'gap':>6}"
        )
        lines.append(f"  {'─'*60}")
        for fg in fa.feature_gaps[:8]:
            lines.append(
                f"  {fg.feature:<38}  "
                f"{fg.rate_in_failures:>5.0%}  {fg.rate_in_correct:>5.0%}  "
                f"{fg.gap:>+6.0%}"
            )

    sep("ATTRIBUTION HYPOTHESIS")
    lines.append(f"  {fa.attribution_hypothesis}")

    # ════════════════════════════════════════════════════════════════════════
    # Relevant source code
    # ════════════════════════════════════════════════════════════════════════
    sep(f"CURRENT RUBRIC CODE — {fa.disease} (rubric_graph.py)")
    lines.append(_rubric_source(fa.disease))

    sep("CURRENT SCORING PARAMETERS (diagnosis_distribution.py)")
    lines.append(_scoring_params_snippet())

    # ════════════════════════════════════════════════════════════════════════
    # Instruction block
    # ════════════════════════════════════════════════════════════════════════
    sep("INSTRUCTION")
    lines.append(
        "  Based on the failure analysis and code above, propose ONE specific\n"
        "  code change to improve the accuracy of the failing disease.\n"
        "\n"
        "  RULES:\n"
        "  1. Change only ONE thing: one edge condition OR one scoring parameter.\n"
        "  2. If editing rubric_graph.py: only modify the _build_X_graph() shown above\n"
        "     (or one of its helper functions shown above).\n"
        "  3. If editing diagnosis_distribution.py: change only one numeric constant.\n"
        "  4. Do NOT change both files in one proposal.\n"
        "  5. old_code must be an EXACT substring of the current file (for safe replacement).\n"
        "\n"
        "  RESPOND IN THIS FORMAT:\n"
        "  ---\n"
        "  target_file:  rubric_graph.py  OR  diagnosis_distribution.py\n"
        "  change_type:  edge_condition   OR  scoring_param   OR  triage_condition\n"
        "  description:  <one sentence>\n"
        "  old_code: |\n"
        "    <exact lines to replace>\n"
        "  new_code: |\n"
        "    <replacement lines>\n"
        "  rationale: |\n"
        "    <2-4 sentences explaining why this change should help>\n"
        "  ---"
    )

    lines.append(f"\n{'═'*W}")
    return "\n".join(lines)
