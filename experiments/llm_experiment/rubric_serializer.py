"""Serialize a RubricGraph (disease sub-rubric) into an LLM-readable block.

We use ``edge.label`` as the natural-language condition description — the
lambda body is uninterpretable, but the labels in rubric_graph.py already
carry the clinical text (e.g. "Score 4-6 - Intermediate Risk -> Ultrasound").
"""
from __future__ import annotations

from pipeline.rubric_graph import DISEASE_GRAPHS, RubricGraph


def serialize_subrubric(disease: str) -> str:
    if disease not in DISEASE_GRAPHS:
        raise KeyError(disease)
    g: RubricGraph = DISEASE_GRAPHS[disease]
    lines: list[str] = []
    lines.append(f"# Sub-rubric: {disease}")
    lines.append(f"Root node: {g.root}")
    lines.append("")
    lines.append("## Nodes")
    for nid, node in g.nodes.items():
        rt = f"  required_tests={node.required_tests}" if node.required_tests else ""
        desc = node.description.replace("\n", " | ") if node.description else ""
        lines.append(
            f"- {nid} [{node.node_type}] {node.label}{rt}"
            + (f"\n    desc: {desc}" if desc else "")
        )
    lines.append("")
    lines.append("## Edges (condition is clinical, evaluated against current features)")
    for e in g.edges:
        lines.append(f"- {e.source} --> {e.target}  | condition: {e.label}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(serialize_subrubric("appendicitis"))
