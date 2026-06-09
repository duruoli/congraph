"""Prompt builders for the LLM next-test recommendation experiment.

Three conditions:
  - llm_features_only : only patient features
  - llm_full          : patient features + rubric recommendation + sub-rubric
                        graph + top-5 similar-patient sequences
  - llm_rubric        : patient features + rubric recommendation + sub-rubric
                        graph; LLM explicitly decides to follow or deviate,
                        outputting follows_rubric + deviation_reason

Order-sensitivity check toggles the ordering of (sub-rubric + rubric rec)
vs (KNN block) within llm_full.
"""
from __future__ import annotations

from pipeline.feature_schema import VALID_TESTS, default_features

from experiments.llm_experiment.knn_neighbors import TrainPatient
from experiments.llm_experiment.rubric_serializer import serialize_subrubric


SYSTEM_PROMPT = (
    "You are an emergency physician triaging acute abdominal pain. "
    "At each decision point, recommend the single next diagnostic test "
    "to perform, choosing from a fixed vocabulary. "
    "If you believe enough information is already available to commit to "
    "a diagnosis and no further testing is warranted, return STOP. "
    "Output strict JSON only — no prose, no markdown fences."
)

OUTPUT_INSTRUCTIONS = (
    "Respond with a single JSON object exactly of the form:\n"
    '{"next_test": "<test_key>", "reasoning": "1-2 sentence rationale"}\n'
    f'where <test_key> is one of: {list(VALID_TESTS)} OR the string "STOP".\n'
    "Do NOT recommend a test that has already been completed."
)

RUBRIC_OUTPUT_INSTRUCTIONS = (
    "Based on this patient's specific presentation and your medical knowledge, "
    "decide whether the rubric recommendation is appropriate for this patient, "
    "or whether their situation warrants a different approach.\n"
    "Respond with a single JSON object exactly of the form:\n"
    '{"next_test": "<test_key>", "follows_rubric": <bool>, '
    '"deviation_reason": "<if follows_rubric is false: why this patient warrants deviation>", '
    '"reasoning": "1-2 sentence rationale"}\n'
    f'where <test_key> is one of: {list(VALID_TESTS)} OR the string "STOP".\n'
    "Do NOT recommend a test that has already been completed.\n"
    '"follows_rubric" must be true if next_test matches the rubric recommendation, '
    "false otherwise. "
    '"deviation_reason" is required when follows_rubric is false, empty string otherwise.'
)


# ---------------------------------------------------------------------------
# Feature presentation
# ---------------------------------------------------------------------------

_DEFAULTS = default_features()


def _present_features(features: dict) -> str:
    """Show only True booleans and non-default scalars/strings + tests_done."""
    lines: list[str] = []
    for k, v in features.items():
        if k == "tests_done":
            continue
        if k == "lab_itemids":
            continue
        if k not in _DEFAULTS:
            # Custom/extra key (e.g. computed flags). Show if truthy.
            if v not in (None, "", [], False, 0, 0.0):
                lines.append(f"  - {k}: {v}")
            continue
        default = _DEFAULTS[k]
        if isinstance(default, bool):
            if v:
                lines.append(f"  - {k}: True")
        elif isinstance(default, str):
            if v and v != default:
                lines.append(f"  - {k}: {v}")
        elif isinstance(default, (int, float)):
            if v != default:
                lines.append(f"  - {k}: {v}")
    done = features.get("tests_done", [])
    completed = "Tests already completed: " + (", ".join(done) if done else "(none)")
    body = "\n".join(lines) if lines else "  (no positive findings recorded)"
    return f"## Patient findings (only positive / non-default shown)\n{body}\n\n{completed}"


# ---------------------------------------------------------------------------
# KNN block
# ---------------------------------------------------------------------------

def _present_knn(neighbors: list[tuple[TrainPatient, float]]) -> str:
    if not neighbors:
        return "## Similar prior patients\n(none available)"
    lines = ["## Top-5 similar prior patients (by step-0 feature similarity)",
             "Each line: rank | disease | similarity distance (smaller = closer) | full test sequence"]
    for i, (tp, dist) in enumerate(neighbors, 1):
        seq = " -> ".join(tp.test_sequence) if tp.test_sequence else "(no tests recorded)"
        lines.append(f"  {i}. disease={tp.disease}  d={dist:.3f}  seq: {seq}")
    return "\n".join(lines)


def _present_knn_deviation(
    neighbors: list[tuple[TrainPatient, float]],
    rubric_pending: list[str],
    tests_done: list[str],
) -> str:
    """KNN block annotated with per-neighbor deviation from rubric.

    For each similar prior patient, shows which rubric-recommended tests they
    performed vs skipped, and which extra tests they ordered beyond the rubric.
    This surfaces the local-shortcut and local-addition signals that are hidden
    in raw sequences.
    """
    if not neighbors:
        return "## Similar prior patients\n(none available)"

    # Tests rubric still wants (excluding Lab_Panel and already-done tests)
    done_set = set(tests_done) | {"Lab_Panel"}
    rubric_set = set(rubric_pending) - done_set

    header_lines = [
        "## Top-5 similar prior patients (deviation-annotated)",
        "Shows how each similar patient's test choices compared to what the rubric "
        "currently recommends for this patient.",
    ]
    if rubric_set:
        header_lines.append(
            "Rubric currently recommends: " + ", ".join(sorted(rubric_set))
        )
    else:
        header_lines.append("Rubric has no further recommendations at this point.")

    lines = header_lines + [""]
    for i, (tp, dist) in enumerate(neighbors, 1):
        neighbor_set = set(tp.test_sequence)

        if rubric_set:
            covered = sorted(rubric_set & neighbor_set)
            skipped = sorted(rubric_set - neighbor_set)
            extra   = sorted((neighbor_set - rubric_set) - done_set)
            parts = []
            parts.append("covered: " + (", ".join(covered) if covered else "(none)"))
            parts.append("skipped: " + (", ".join(skipped) if skipped else "(none)"))
            parts.append("added:   " + (", ".join(extra)   if extra   else "(none)"))
            detail = " | ".join(parts)
        else:
            # Rubric is silent — fall back to showing raw sequence
            seq = " -> ".join(tp.test_sequence) if tp.test_sequence else "(no tests)"
            detail = f"seq: {seq}"

        lines.append(f"  {i}. disease={tp.disease}  d={dist:.3f}")
        lines.append(f"     {detail}")

    return "\n".join(lines)


def _deviation_summary(
    neighbors: list[tuple[TrainPatient, float]],
    rubric_pending: list[str],
    tests_done: list[str],
    min_fraction: float = 0.6,
) -> str:
    """One-sentence summary of the strongest deviation signal across neighbors.

    Placed just before OUTPUT_INSTRUCTIONS so the LLM sees it at the decision
    point.  Only emits a sentence when >= min_fraction of neighbors show a
    consistent skip or add signal.  Returns empty string if no strong signal.
    """
    if not neighbors:
        return ""

    done_set = set(tests_done) | {"Lab_Panel"}
    rubric_set = set(rubric_pending) - done_set
    n = len(neighbors)

    if not rubric_set:
        return ""

    # Count per-test skip and add across all neighbors
    skip_counts: dict[str, int] = {}
    add_counts:  dict[str, int] = {}
    tests_done_label = ", ".join(sorted(done_set - {"Lab_Panel"})) or "Lab_Panel"

    for tp, _ in neighbors:
        neighbor_set = set(tp.test_sequence)
        for t in rubric_set - neighbor_set:
            skip_counts[t] = skip_counts.get(t, 0) + 1
        for t in (neighbor_set - rubric_set) - done_set:
            add_counts[t] = add_counts.get(t, 0) + 1

    sentences = []

    # Skip signal
    for test, cnt in sorted(skip_counts.items(), key=lambda x: -x[1]):
        if cnt / n >= min_fraction:
            done_str = (f" after {tests_done_label}" if tests_done_label else "")
            sentences.append(
                f"Clinical precedent: {cnt} of {n} similar patients chose to skip "
                f"{test}{done_str}."
            )

    # Add signal (only if no strong skip sentence already)
    if not sentences:
        for test, cnt in sorted(add_counts.items(), key=lambda x: -x[1]):
            if cnt / n >= min_fraction:
                sentences.append(
                    f"Clinical precedent: {cnt} of {n} similar patients added "
                    f"{test} beyond the rubric recommendation."
                )

    return "\n".join(sentences)


# ---------------------------------------------------------------------------
# Rubric block
# ---------------------------------------------------------------------------

def _present_rubric_recommendation(
    rubric_next_test: str | None, disease: str
) -> str:
    rec = rubric_next_test if rubric_next_test else "(rubric did not recommend a next test at this point)"
    return f"## Rubric simulator recommendation for next test\n{rec}\n"


def _present_subrubric_graph(disease: str) -> str:
    return "## Full sub-rubric graph (clinical guideline structure)\n" + serialize_subrubric(disease)


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def build_user_prompt(
    *,
    features: dict,
    condition: str,           # "llm_features_only" | "llm_full" | "llm_full_deviation" | "llm_rubric"
    disease: str,             # oracle disease
    rubric_next_test: str | None = None,
    rubric_pending_tests: list[str] | None = None,   # all rubric-pending tests (deviation only)
    neighbors: list[tuple[TrainPatient, float]] | None = None,
    info_order: str = "rubric_first",
) -> str:
    parts: list[str] = []
    parts.append(_present_features(features))

    if condition == "llm_features_only":
        parts.append(OUTPUT_INSTRUCTIONS)
        return "\n\n".join(parts)

    if condition == "llm_rubric":
        parts.append(_present_rubric_recommendation(rubric_next_test, disease))
        parts.append(_present_subrubric_graph(disease))
        parts.append(RUBRIC_OUTPUT_INSTRUCTIONS)
        return "\n\n".join(parts)

    if condition not in ("llm_full", "llm_full_deviation"):
        raise ValueError(f"Unknown condition: {condition}")

    rubric_blocks = [
        _present_rubric_recommendation(rubric_next_test, disease),
        _present_subrubric_graph(disease),
    ]

    if condition == "llm_full_deviation":
        tests_done = list(features.get("tests_done", []))
        knn_block = _present_knn_deviation(
            neighbors or [],
            rubric_pending=rubric_pending_tests or [],
            tests_done=tests_done,
        )
        summary = _deviation_summary(
            neighbors or [],
            rubric_pending=rubric_pending_tests or [],
            tests_done=tests_done,
        )
    else:
        knn_block = _present_knn(neighbors or [])
        summary = ""

    if info_order == "rubric_first":
        parts.extend(rubric_blocks)
        parts.append(knn_block)
    elif info_order == "knn_first":
        parts.append(knn_block)
        parts.extend(rubric_blocks)
    else:
        raise ValueError(f"Unknown info_order: {info_order}")

    if summary:
        parts.append(summary)
    parts.append(OUTPUT_INSTRUCTIONS)
    return "\n\n".join(parts)
