"""traversal_engine.py

Traversal engine: locates a patient in every clinical rubric graph by
evaluating edge conditions against the current feature dict.

Algorithm: breadth-first traversal from each graph's root.
A node becomes "reachable" when:
  - it is the root node, OR
  - at least one predecessor is reachable, that predecessor's required_tests
    are all satisfied, AND the connecting edge condition evaluates to True.

Node states after traversal:
  "reached"       — passed through; at least one outgoing edge was taken
  "pending"       — reachable, but required_tests not yet completed
  "blocked"       — reachable, required_tests done, but all outgoing edges
                    evaluate to False (this patient's data is a dead-end here)
  "frontier_leaf" — reachable, required_tests done, no outgoing edges
                    (leaf of the graph — may be a routing node or a clinical
                    terminal depending on node.node_type)
  "unvisited"     — never reached by the traversal

The `triage` graph is run first to identify which disease sub-rubrics are
"activated" by the main routing edges.  All four disease sub-rubrics are
then traversed in parallel regardless of triage activation, but each
TraversalResult records the triage_activated flag.

Design decision — traversal is DECOUPLED from diagnosis distribution:
  TraversalResult carries depth / triage_activated / terminal_type which
  serve as clean signals to a separate DiagnosisDistribution layer.
  This keeps traversal a pure, deterministic, testable function.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from rubric_graph import (
    ALL_GRAPHS,
    DISEASE_GRAPHS,
    TRIAGE_GRAPH,
    RubricGraph,
)


# ---------------------------------------------------------------------------
# Routing node id → disease sub-rubric name mapping
# ---------------------------------------------------------------------------

_ROUTE_TO_DISEASE: dict[str, str] = {
    "ROUTE_APPENDICITIS":  "appendicitis",
    "ROUTE_CHOLECYSTITIS": "cholecystitis",
    "ROUTE_DIVERTICULITIS": "diverticulitis",
    "ROUTE_PANCREATITIS":  "pancreatitis",
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NodeStatus:
    """Traversal status for a single rubric node."""
    node_id: str
    # "reached" | "pending" | "blocked" | "frontier_leaf" | "unvisited"
    status: str
    # Tests still missing when status == "pending"
    missing_tests: list[str] = field(default_factory=list)


@dataclass
class TraversalResult:
    """Result of traversing one rubric graph against a patient feature dict."""

    disease: str
    node_statuses: dict[str, NodeStatus]

    # Nodes where traversal stopped (status != "reached")
    frontier: list[str]

    # Clinical diagnostic terminal reached (None = still in progress)
    # node_type must be "terminal_confirmed" or "terminal_excluded"
    terminal_node: Optional[str]
    terminal_type: Optional[str]   # "terminal_confirmed" | "terminal_excluded" | None

    # Tests needed at pending frontier nodes (in order of encounter)
    pending_tests: list[str]

    # Proxy for evidence strength: # of nodes we actually processed
    # (status in {"reached", "pending", "blocked", "frontier_leaf"})
    depth: int

    # Did the main triage graph route to this disease?
    triage_activated: bool = False

    # ---------- convenience properties ----------

    @property
    def confirmed(self) -> bool:
        return self.terminal_type == "terminal_confirmed"

    @property
    def excluded(self) -> bool:
        return self.terminal_type == "terminal_excluded"

    @property
    def in_progress(self) -> bool:
        return self.terminal_node is None


@dataclass
class FullTraversalResult:
    """Consolidated traversal across the triage graph + all disease sub-rubrics."""

    triage: TraversalResult
    diseases: dict[str, TraversalResult]   # disease name → result

    # Diseases ordered by evidence strength (primary hypothesis first)
    ranked_diseases: list[str]

    # Top-ranked disease (None only if no disease graphs ran)
    primary_hypothesis: Optional[str]

    # Next tests to order: primary-hypothesis pending tests first, then others
    recommended_next_tests: list[str]


# ---------------------------------------------------------------------------
# Core single-graph traversal
# ---------------------------------------------------------------------------

def _edge_ok(edge, features: dict) -> bool:
    """Safely evaluate one edge condition.  Returns False on any exception."""
    try:
        return bool(edge.condition(features))
    except Exception:
        return False


def traverse_graph(graph: RubricGraph, features: dict) -> TraversalResult:
    """
    BFS traversal of *graph* against *features*.

    Returns a TraversalResult describing which nodes were reached, where
    traversal stopped, and what tests are needed to advance further.
    """
    # ── build outgoing-edge index ──────────────────────────────────────────
    outgoing: dict[str, list] = defaultdict(list)
    for edge in graph.edges:
        outgoing[edge.source].append(edge)

    tests_done: set[str] = set(features.get("tests_done", []))

    node_statuses: dict[str, NodeStatus] = {}
    frontier: list[str] = []
    terminal_node: Optional[str] = None
    terminal_type: Optional[str] = None

    # ── BFS ───────────────────────────────────────────────────────────────
    reachable: set[str] = {graph.root}
    queue: deque[str] = deque([graph.root])

    while queue:
        nid = queue.popleft()
        node = graph.nodes[nid]

        # 1. Check required tests
        missing = [t for t in node.required_tests if t not in tests_done]
        if missing:
            node_statuses[nid] = NodeStatus(nid, "pending", missing)
            frontier.append(nid)
            continue

        out_edges = outgoing[nid]

        # 2. Leaf node (no outgoing edges)
        if not out_edges:
            node_statuses[nid] = NodeStatus(nid, "frontier_leaf")
            frontier.append(nid)
            ntype = node.node_type
            if ntype in ("terminal_confirmed", "terminal_excluded"):
                terminal_node = nid
                terminal_type = ntype
            continue

        # 3. Try outgoing edges
        any_taken = False
        for edge in out_edges:
            if _edge_ok(edge, features):
                tgt = edge.target
                if tgt not in reachable:
                    reachable.add(tgt)
                    queue.append(tgt)
                any_taken = True

        if any_taken:
            node_statuses[nid] = NodeStatus(nid, "reached")
        else:
            node_statuses[nid] = NodeStatus(nid, "blocked")
            frontier.append(nid)

    # ── mark unvisited nodes ───────────────────────────────────────────────
    for nid in graph.nodes:
        if nid not in node_statuses:
            node_statuses[nid] = NodeStatus(nid, "unvisited")

    # ── collect pending tests (in encounter order, deduped) ────────────────
    pending_tests: list[str] = []
    for nid in frontier:
        ns = node_statuses[nid]
        if ns.status == "pending":
            for t in ns.missing_tests:
                if t not in pending_tests:
                    pending_tests.append(t)

    # ── depth = # of nodes we actually touched (any status except "unvisited") ──
    depth = sum(
        1 for ns in node_statuses.values() if ns.status != "unvisited"
    )

    return TraversalResult(
        disease=graph.disease,
        node_statuses=node_statuses,
        frontier=frontier,
        terminal_node=terminal_node,
        terminal_type=terminal_type,
        pending_tests=pending_tests,
        depth=depth,
    )


# ---------------------------------------------------------------------------
# Full multi-graph traversal
# ---------------------------------------------------------------------------

def run_full_traversal(features: dict) -> FullTraversalResult:
    """
    Traverse the triage graph + all four disease sub-rubrics.

    Triage determines which diseases are "activated" (primary hypotheses);
    all sub-rubrics are traversed regardless so partial evidence is captured
    even for non-activated diseases.

    Ranking:
      confirmed terminal       +10 points
      triage-activated disease  +5 points
      traversal depth            +1 per node (proxy for evidence strength)
    """
    # 1. Triage
    triage_result = traverse_graph(TRIAGE_GRAPH, features)

    # 2. Triage activation flags (routing nodes appear as "frontier_leaf"
    #    in the triage graph when their condition fires)
    triage_activated: dict[str, bool] = {
        disease: (
            triage_result.node_statuses.get(route_nid) is not None
            and triage_result.node_statuses[route_nid].status == "frontier_leaf"
        )
        for route_nid, disease in _ROUTE_TO_DISEASE.items()
    }

    # 3. Disease sub-rubric traversals (parallel, independent)
    disease_results: dict[str, TraversalResult] = {}
    for name, graph in DISEASE_GRAPHS.items():
        result = traverse_graph(graph, features)
        result.triage_activated = triage_activated.get(name, False)
        disease_results[name] = result

    # 4. Rank by evidence strength
    def _score(name: str) -> int:
        r = disease_results[name]
        return (
            (10 if r.confirmed else 0)
            + (5 if r.triage_activated else 0)
            + r.depth
        )

    ranked_diseases = sorted(disease_results, key=_score, reverse=True)
    primary_hypothesis = ranked_diseases[0] if ranked_diseases else None

    # 5. Recommended next tests (primary hypothesis first, then others, deduped)
    recommended: list[str] = []
    if primary_hypothesis:
        for t in disease_results[primary_hypothesis].pending_tests:
            if t not in recommended:
                recommended.append(t)
    for name in ranked_diseases:
        for t in disease_results[name].pending_tests:
            if t not in recommended:
                recommended.append(t)

    return FullTraversalResult(
        triage=triage_result,
        diseases=disease_results,
        ranked_diseases=ranked_diseases,
        primary_hypothesis=primary_hypothesis,
        recommended_next_tests=recommended,
    )


# ---------------------------------------------------------------------------
# Human-readable report (for debugging / demo)
# ---------------------------------------------------------------------------

_STATUS_ICON = {
    "reached":       "✓",
    "pending":       "⏳",
    "blocked":       "✗",
    "frontier_leaf": "◉",
    "unvisited":     "·",
}


def print_traversal_report(result: FullTraversalResult) -> None:
    """Print a compact traversal report to stdout."""
    # ── triage summary ─────────────────────────────────────────────────────
    tr = result.triage
    activated = [d for d, a in zip(_ROUTE_TO_DISEASE.values(),
                                   [result.diseases[d].triage_activated
                                    for d in _ROUTE_TO_DISEASE.values()])
                 if a]
    print(f"\n{'═'*60}")
    print(f"  TRIAGE   depth={tr.depth}  activated={activated or '(none)'}")
    print(f"{'═'*60}")

    # ── disease sub-rubrics ────────────────────────────────────────────────
    for name in result.ranked_diseases:
        r = result.diseases[name]
        tag = ""
        if r.confirmed:
            tag = f"  ✓ CONFIRMED → {r.terminal_node}"
        elif r.excluded:
            tag = f"  ✗ EXCLUDED  → {r.terminal_node}"
        elif r.pending_tests:
            tag = f"  ⏳ pending: {r.pending_tests}"
        else:
            tag = "  ✗ blocked"

        primary_mark = " ◀ PRIMARY" if name == result.primary_hypothesis else ""
        activated_mark = " [TRIAGE✓]" if r.triage_activated else ""
        print(f"\n  {name.upper():16s} depth={r.depth}{activated_mark}{primary_mark}")
        print(f"  {'─'*56}")

        for nid, ns in r.node_statuses.items():
            if ns.status == "unvisited":
                continue
            icon = _STATUS_ICON.get(ns.status, "?")
            missing_str = (
                f"  ← needs {ns.missing_tests}" if ns.missing_tests else ""
            )
            print(f"    {icon} {nid}{missing_str}")
        print(f"  └─{tag}")

    # ── recommendation ─────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Recommended next tests: {result.recommended_next_tests or '(none — terminal reached)'}")
    print(f"  Primary hypothesis    : {result.primary_hypothesis}")
    print(f"{'═'*60}\n")
