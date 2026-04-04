"""viz_app.py — Interactive patient visualizer for the ConGraph pipeline.

Visualizes for a selected patient + evidence step:
  ① Diagnosis probability distribution (Plotly horizontal bar chart)
  ② Next-test recommendations
  ③ Patient's position in each rubric graph (Triage + 4 disease graphs),
     color-coded by traversal node status

Run with:
    streamlit run viz_app.py

Requirements (in addition to existing project deps):
    pip install streamlit plotly
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# ── make sure congraph modules are importable ─────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_session import ClinicalSession  # noqa: E402
from real_pipeline import load_disease, DISEASE_FILES  # noqa: E402
from rubric_graph import ALL_GRAPHS, RubricGraph  # noqa: E402
from traversal_engine import TraversalResult  # noqa: E402


# ── Visual constants ──────────────────────────────────────────────────────────

STATUS_COLOR: dict[str, str] = {
    "reached":       "#27ae60",   # green      — traversal passed through
    "pending":       "#f39c12",   # amber      — waiting for a required test
    "blocked":       "#e74c3c",   # red        — dead-end (all edges false)
    "frontier_leaf": "#2980b9",   # blue       — leaf / terminal reached
    "unvisited":     "#dfe6e9",   # light grey — never reached
}

STATUS_LABEL: dict[str, str] = {
    "reached":       "✓ reached",
    "pending":       "⏳ pending test",
    "blocked":       "✗ blocked",
    "frontier_leaf": "◉ frontier / terminal",
    "unvisited":     "· unvisited",
}

NODE_SHAPE: dict[str, str] = {
    "start":              "ellipse",
    "assessment":         "box",
    "decision":           "diamond",
    "terminal_confirmed": "box",
    "terminal_excluded":  "box",
    "routing":            "parallelogram",
}

BORDER_COLOR: dict[str, str] = {
    "terminal_confirmed": "#27ae60",
    "terminal_excluded":  "#c0392b",
}

DISEASE_COLOR: dict[str, str] = {
    "appendicitis":   "#e74c3c",
    "cholecystitis":  "#e67e22",
    "diverticulitis": "#27ae60",
    "pancreatitis":   "#2980b9",
}


# ── DOT-string builder ────────────────────────────────────────────────────────

def _dot_escape(text: str) -> str:
    """Escape a string for safe embedding in a DOT quoted label."""
    return (
        text
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def make_dot(graph: RubricGraph, result: TraversalResult) -> str:
    """
    Build a Graphviz DOT string for *graph*, coloring nodes by their
    traversal status in *result*.
    """
    lines = [
        "digraph {",
        "  rankdir=TB;",
        '  graph [fontname="Helvetica" bgcolor="transparent" splines=ortho];',
        '  node [fontname="Helvetica" style="filled,rounded" fontsize=11 margin="0.2,0.1"];',
        '  edge [fontname="Helvetica" fontsize=9 color="#7f8c8d"];',
    ]

    for nid, node in graph.nodes.items():
        ns      = result.node_statuses.get(nid)
        status  = ns.status if ns else "unvisited"
        color   = STATUS_COLOR[status]
        shape   = NODE_SHAPE.get(node.node_type, "box")
        fcolor  = "#ffffff" if status in ("reached", "blocked", "frontier_leaf") else "#2c3e50"
        border  = BORDER_COLOR.get(node.node_type, "#2c3e50")
        pwidth  = "2.5" if status in ("pending", "blocked", "frontier_leaf") else "1.2"

        # Build node label: node_id (line 1) + missing tests if pending (line 2)
        label = _dot_escape(nid)
        if status == "pending" and ns and ns.missing_tests:
            label += "\\n⏳ " + _dot_escape(", ".join(ns.missing_tests))

        # tooltip = full clinical label + status
        tooltip = _dot_escape(f"{node.label} [{status}]")
        if node.description:
            tooltip += "\\n" + _dot_escape(node.description[:180])

        lines.append(
            f'  {nid} ['
            f'label="{label}" '
            f'shape={shape} '
            f'fillcolor="{color}" '
            f'fontcolor="{fcolor}" '
            f'color="{border}" '
            f'penwidth={pwidth} '
            f'tooltip="{tooltip}"'
            f'];'
        )

    for edge in graph.edges:
        elabel = _dot_escape(edge.label[:45])
        lines.append(
            f'  {edge.source} -> {edge.target} [label="{elabel}"];'
        )

    lines.append("}")
    return "\n".join(lines)


# ── Distribution chart ────────────────────────────────────────────────────────

def make_dist_chart(distribution, traversal) -> go.Figure:
    """Horizontal Plotly bar chart of P(diagnosis | features)."""
    # Reverse so highest-probability disease appears at the top
    diseases = list(reversed(distribution.ranked))
    probs    = [distribution.prob(d) * 100 for d in diseases]

    bar_colors = []
    for d in diseases:
        r = traversal.diseases[d]
        if r.confirmed:
            bar_colors.append("#27ae60")
        elif r.excluded:
            bar_colors.append("#e74c3c")
        elif r.triage_activated:
            bar_colors.append(DISEASE_COLOR.get(d, "#3498db"))
        else:
            bar_colors.append("#95a5a6")

    hover_texts = []
    for d in diseases:
        r   = traversal.diseases[d]
        p   = distribution.prob(d)
        tags = []
        if r.triage_activated:
            tags.append("triage ✓")
        if r.confirmed:
            tags.append(f"CONFIRMED → {r.terminal_node}")
        elif r.excluded:
            tags.append(f"excluded → {r.terminal_node}")
        elif r.in_progress:
            tags.append("in progress")
        hover_texts.append(
            f"<b>{d}</b><br>"
            f"P = {p:.1%}<br>"
            f"depth = {r.depth}<br>"
            f"raw score = {distribution.raw_scores[d]:.2f}<br>"
            + "<br>".join(tags)
        )

    fig = go.Figure(go.Bar(
        x=probs,
        y=diseases,
        orientation="h",
        marker_color=bar_colors,
        marker_line_color="#2c3e50",
        marker_line_width=0.8,
        text=[f"{p:.1f}%" for p in probs],
        textposition="auto",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=8, b=8),
        xaxis=dict(
            title="Probability (%)",
            range=[0, 108],
            gridcolor="#ecf0f1",
        ),
        yaxis=dict(title="", tickfont=dict(size=13)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    return fig


# ── Node-status legend HTML ───────────────────────────────────────────────────

def _legend_html() -> str:
    items = [
        (STATUS_COLOR[s], STATUS_LABEL[s])
        for s in ("reached", "pending", "blocked", "frontier_leaf", "unvisited")
    ]
    badges = "".join(
        f'<span style="background:{c};color:#fff;'
        f'padding:2px 8px;border-radius:4px;margin-right:6px;'
        f'font-size:12px;">{lbl}</span>'
        for c, lbl in items
    )
    return f"<div style='margin-bottom:6px'>{badges}</div>"


# ── Streamlit app ─────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="ConGraph — Patient Visualizer",
        page_icon="🏥",
        layout="wide",
    )

    st.title("🏥 ConGraph — Clinical Rubric Graph Visualizer")
    st.caption(
        "Select a patient and drag the **Evidence step** slider to see how "
        "the diagnosis distribution and rubric-graph position evolve as "
        "evidence accumulates."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Patient Selection")

        disease = st.selectbox(
            "Disease dataset",
            list(DISEASE_FILES.keys()),
            format_func=str.capitalize,
        )

        with st.spinner("Loading patients…"):
            patients = load_disease(disease)

        pid = st.selectbox("Patient ID", list(patients.keys()))
        steps = patients[pid]

        step_idx = st.slider(
            "Evidence step",
            min_value=0,
            max_value=len(steps) - 1,
            value=len(steps) - 1,
            help="0 = HPI + PE only; max = all evidence collected",
        )

        step_meta = steps[step_idx]

        st.divider()
        st.subheader("Step info")
        st.write(f"**Label:** {step_meta['step_label']}")
        if step_meta.get("test_key"):
            st.write(f"**New test:** `{step_meta['test_key']}`")
        tests_done = step_meta["features"].get("tests_done", [])
        if tests_done:
            st.write("**Tests done:**")
            for t in tests_done:
                st.write(f"  • `{t}`")
        else:
            st.write("**Tests done:** *(none — HPI + PE only)*")

        st.divider()
        st.markdown(f"🏷 Ground truth: **{disease}**")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    with st.spinner("Running pipeline…"):
        session = ClinicalSession(step_meta["features"])
        state   = session.assess()

    traversal_map: dict[str, TraversalResult] = {
        "triage": state.traversal.triage,
        **state.traversal.diseases,
    }

    # ── Row 1: Distribution + Recommendations ─────────────────────────────────
    col_dist, col_rec = st.columns([3, 2], gap="large")

    with col_dist:
        st.subheader("Diagnosis Distribution")
        primary = state.distribution.primary
        r_pri   = state.traversal.diseases[primary]
        if r_pri.confirmed:
            badge = f"✅ CONFIRMED → `{r_pri.terminal_node}`"
        elif r_pri.excluded:
            badge = f"❌ excluded → `{r_pri.terminal_node}`"
        elif r_pri.triage_activated:
            badge = "🔵 primary hypothesis (triage-activated)"
        else:
            badge = "🔘 primary hypothesis"
        st.caption(f"Primary: **{primary}** — {badge}")
        st.plotly_chart(
            make_dist_chart(state.distribution, state.traversal),
            use_container_width=True,
        )

    with col_rec:
        st.subheader("Next Tests Recommended")
        if state.recommendations:
            for i, rec in enumerate(state.recommendations, 1):
                advances_str = ", ".join(rec.advances)
                st.markdown(
                    f"**{i}. `{rec.test}`** &nbsp;"
                    f"*(relevance: {rec.relevance:.2f})*  \n"
                    f"Advances: {advances_str}"
                )
        else:
            st.success(
                "No further tests needed — all active diagnoses resolved."
            )

    st.divider()

    # ── Row 2: Rubric graphs ───────────────────────────────────────────────────
    st.subheader("Rubric Graph Positions")
    st.markdown(_legend_html(), unsafe_allow_html=True)

    graph_keys = [
        "triage",
        "appendicitis",
        "cholecystitis",
        "diverticulitis",
        "pancreatitis",
    ]

    # Build tab labels with depth / outcome badges
    tab_labels = []
    for key in graph_keys:
        tr = traversal_map[key]
        if key == "triage":
            activated = [
                d for d in state.traversal.diseases
                if state.traversal.diseases[d].triage_activated
            ]
            badge = f"→ {', '.join(activated)}" if activated else "→ (none)"
            tab_labels.append(f"Triage  {badge}")
        else:
            if tr.confirmed:
                badge = f"✅ confirmed  d={tr.depth}"
            elif tr.excluded:
                badge = f"❌ excluded  d={tr.depth}"
            else:
                badge = f"d={tr.depth}"
            tab_labels.append(f"{key.capitalize()}  [{badge}]")

    tabs = st.tabs(tab_labels)

    for tab, key in zip(tabs, graph_keys):
        with tab:
            tr = traversal_map[key]

            # Status count summary
            counts: dict[str, int] = {}
            for ns in tr.node_statuses.values():
                counts[ns.status] = counts.get(ns.status, 0) + 1
            summary_parts = [
                f"{STATUS_LABEL[s]}: **{counts[s]}**"
                for s in ("reached", "pending", "blocked", "frontier_leaf", "unvisited")
                if counts.get(s, 0) > 0
            ]
            st.caption("  |  ".join(summary_parts))

            # Graphviz chart
            dot_str = make_dot(ALL_GRAPHS[key], tr)
            st.graphviz_chart(dot_str, use_container_width=True)

            # Collapsible frontier details
            if tr.frontier:
                with st.expander(f"Frontier nodes ({len(tr.frontier)})"):
                    for nid in tr.frontier:
                        ns   = tr.node_statuses[nid]
                        node = ALL_GRAPHS[key].nodes[nid]
                        icon = {
                            "pending":       "⏳",
                            "blocked":       "✗",
                            "frontier_leaf": "◉",
                        }.get(ns.status, "?")
                        st.markdown(
                            f"**{icon} `{nid}`** — *{node.label}* &nbsp; "
                            f"`[{ns.status}]`"
                        )
                        if ns.missing_tests:
                            st.markdown(
                                f"&nbsp;&nbsp;&nbsp;&nbsp;"
                                f"Missing tests: `{'`, `'.join(ns.missing_tests)}`"
                            )
                        if node.description:
                            st.caption(
                                "\u00a0\u00a0\u00a0\u00a0"
                                + node.description[:240].replace("\n", "  \n\u00a0\u00a0\u00a0\u00a0")
                            )

    # ── Footer: raw features ───────────────────────────────────────────────────
    with st.expander("Raw patient features (current step)"):
        features_display = {
            k: v for k, v in step_meta["features"].items()
            if v not in (False, 0, 0.0, [], "Other", "")
        }
        st.json(features_display)


if __name__ == "__main__":
    main()
