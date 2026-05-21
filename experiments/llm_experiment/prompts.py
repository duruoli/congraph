"""Prompt builders for the LLM next-test recommendation experiment.

Two conditions:
  - llm_features_only : only patient features
  - llm_full          : patient features + rubric recommendation + sub-rubric
                        graph + top-5 similar-patient sequences

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
    condition: str,                              # "llm_features_only" | "llm_full"
    disease: str,                                 # oracle disease
    rubric_next_test: str | None = None,
    neighbors: list[tuple[TrainPatient, float]] | None = None,
    info_order: str = "rubric_first",             # "rubric_first" | "knn_first"
) -> str:
    parts: list[str] = []
    parts.append(_present_features(features))

    if condition == "llm_features_only":
        parts.append(OUTPUT_INSTRUCTIONS)
        return "\n\n".join(parts)

    if condition != "llm_full":
        raise ValueError(f"Unknown condition: {condition}")

    rubric_blocks = [
        _present_rubric_recommendation(rubric_next_test, disease),
        _present_subrubric_graph(disease),
    ]
    knn_block = _present_knn(neighbors or [])

    if info_order == "rubric_first":
        parts.extend(rubric_blocks)
        parts.append(knn_block)
    elif info_order == "knn_first":
        parts.append(knn_block)
        parts.extend(rubric_blocks)
    else:
        raise ValueError(f"Unknown info_order: {info_order}")

    parts.append(OUTPUT_INSTRUCTIONS)
    return "\n\n".join(parts)
