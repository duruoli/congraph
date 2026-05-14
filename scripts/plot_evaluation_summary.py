"""plot_evaluation_summary.py — Visualize evaluate.py output (fixed-τ summary).

Reads results/evaluation_summary.csv (one row per strategy) and produces:

  1. A 2×3 grouped-panel figure with one subplot per metric:
       Accuracy | Mean Tests | Mean Cost (USD) | Mean Burden | % Stopped
     Each subplot is a horizontal bar chart comparing all strategies.

  2. A single composite radar / spider chart (optional, --radar flag).

Usage
-----
  python scripts/plot_evaluation_summary.py
  python scripts/plot_evaluation_summary.py --csv results/evaluation_summary.csv
  python scripts/plot_evaluation_summary.py --save-dir results/figs --dpi 200
  python scripts/plot_evaluation_summary.py --radar
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Global aesthetics ─────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["DejaVu Sans"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "figure.facecolor":   "#F9F9FB",
    "axes.facecolor":     "#F9F9FB",
})

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parents[1]
DEFAULT_CSV = _ROOT / "results" / "evaluation_summary.csv"
DEFAULT_FIG = _ROOT / "results" / "figs"

# ── Strategy display properties: 1-step KNN ─────────────────────────────────
STRATEGY_ORDER = ["random", "bfs", "ig", "actual", "actual_real"]

STRATEGY_META: dict[str, dict] = {
    "actual_real": dict(label="Actual",              color="#2166AC"),
    "actual":      dict(label="KNN-estimated",        color="#4393C3"),
    "ig":          dict(label="IG re-ranked",         color="#1A9850"),
    "bfs":         dict(label="BFS (rubric-guided)",  color="#66BD63"),
    "random":      dict(label="Random",               color="#BDBDBD"),
}

# ── Strategy display properties: multi-step feature-sim ──────────────────────
SIM_STRATEGY_ORDER = ["random_sim", "bfs_sim", "ig_sim", "actual_sim", "actual_real"]

SIM_STRATEGY_META: dict[str, dict] = {
    "actual_real": dict(label="Actual (ceiling)",    color="#2166AC"),
    "actual_sim":  dict(label="KNN-simulated",        color="#4393C3"),
    "ig_sim":      dict(label="IG re-ranked (sim)",   color="#1A9850"),
    "bfs_sim":     dict(label="BFS (sim)",            color="#66BD63"),
    "random_sim":  dict(label="Random (sim)",         color="#BDBDBD"),
}

# ── Metric definitions ────────────────────────────────────────────────────────
@dataclass
class MetricDef:
    col:       str           # CSV column name
    label:     str           # axis label
    pct:       bool          # True → format as percentage
    unit:      str           # unit string appended to title
    good_dir:  str           # "high" or "low" (for annotation)

METRICS: list[MetricDef] = [
    MetricDef("accuracy",     "Diagnostic Accuracy",  pct=True,  unit="",      good_dir="high"),
    MetricDef("mean_n_tests", "Mean Tests Ordered",   pct=False, unit="tests", good_dir="low"),
    MetricDef("mean_cost",    "Mean Diagnostic Cost", pct=False, unit="USD",   good_dir="low"),
    MetricDef("mean_burden",  "Mean Patient Burden",  pct=False, unit="score", good_dir="low"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: str | Path) -> list[dict]:
    """Return list of row dicts with numeric fields cast appropriately."""
    rows: list[dict] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "strategy":     row["strategy"].strip(),
                "n_patients":   int(row["n_patients"]),
                "accuracy":     float(row["accuracy"]),
                "mean_n_tests": float(row["mean_n_tests"]),
                "mean_cost":    float(row["mean_cost"]),
                "mean_burden":  float(row["mean_burden"]),
                "pct_stopped":  float(row["pct_stopped"]),
            })
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val(rows: list[dict], strategy: str, col: str) -> float:
    for r in rows:
        if r["strategy"] == strategy:
            return r[col]
    return 0.0


def _best_strategy(rows: list[dict], col: str, good_dir: str) -> str:
    """Return the strategy name that is best for this metric."""
    fn = max if good_dir == "high" else min
    return fn(rows, key=lambda r: r[col])["strategy"]


# ---------------------------------------------------------------------------
# Auto-detect strategy family
# ---------------------------------------------------------------------------

def _detect_meta(
    rows: list[dict],
) -> tuple[list[str], dict[str, dict]]:
    """Return (order, meta) matching the strategy names in rows."""
    strat_names = {r["strategy"] for r in rows}
    if strat_names & {"actual_sim", "bfs_sim", "ig_sim", "random_sim"}:
        return SIM_STRATEGY_ORDER, SIM_STRATEGY_META
    return STRATEGY_ORDER, STRATEGY_META


# ---------------------------------------------------------------------------
# Main figure: 2×2 panel
# ---------------------------------------------------------------------------

def make_panel_figure(
    rows: list[dict],
    tau: float,
    save_path: str | Path | None,
    dpi: int = 180,
    strategy_order: list[str] | None = None,
    strategy_meta:  dict[str, dict] | None = None,
) -> plt.Figure:
    """
    2-row × 2-col figure (Accuracy | Tests / Cost | Burden) with a shared
    legend placed below the grid.
    """
    if strategy_order is None or strategy_meta is None:
        strategy_order, strategy_meta = _detect_meta(rows)
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9),
    )
    fig.patch.set_facecolor("#F0F2F6")
    fig.subplots_adjust(
        top=0.87, bottom=0.16,
        left=0.14, right=0.97,
        hspace=0.42, wspace=0.42,
    )

    n_patients = rows[0]["n_patients"] if rows else 0
    fig.suptitle(
        f"Strategy Comparison at Fixed Stopping Threshold  τ = {tau:.4f}\n"
        f"ConGraph Diagnostic Pipeline  ·  {n_patients} test patients",
        fontsize=14, fontweight="bold", color="#1A1A2E",
        y=0.97,
    )

    # 4 metric panels in reading order
    metric_axes = [
        (axes[0, 0], METRICS[0]),   # accuracy
        (axes[0, 1], METRICS[1]),   # tests
        (axes[1, 0], METRICS[2]),   # cost
        (axes[1, 1], METRICS[3]),   # burden
    ]

    # Ordered strategies (skip unknown ones)
    strats = [s for s in strategy_order if any(r["strategy"] == s for r in rows)]
    y_pos  = np.arange(len(strats))
    bar_h  = 0.6

    bar_containers: list = []   # for figure-level legend

    for ax, metric in metric_axes:
        vals   = [_val(rows, s, metric.col) for s in strats]
        colors = [strategy_meta[s]["color"] for s in strats]
        best_s = _best_strategy(rows, metric.col, metric.good_dir)

        bars = ax.barh(
            y_pos, vals,
            height=bar_h,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        if not bar_containers:   # collect once for the legend
            bar_containers = list(bars)

        # Value annotations
        x_max = max(vals) if vals else 1.0
        for yi, (v, s) in enumerate(zip(vals, strats)):
            is_best = s == best_s
            disp = f"{v:.0%}" if metric.pct else (
                f"${v:,.0f}" if metric.unit == "USD" else f"{v:.2f}"
            )
            ax.text(
                v + x_max * 0.015, yi,
                disp + (" ★" if is_best else ""),
                va="center", ha="left",
                fontsize=9,
                fontweight="bold" if is_best else "normal",
                color="#1A9850" if is_best else "#444444",
            )

        # Axis styling
        unit_str = f" ({metric.unit})" if metric.unit else ""
        ax.set_title(metric.label + unit_str, fontsize=11, fontweight="bold",
                     color="#1A1A2E", pad=8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [strategy_meta[s]["label"] for s in strats],
            fontsize=9,
        )
        ax.set_xlim(0, x_max * 1.22)
        ax.tick_params(axis="x", labelsize=9)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
        ax.set_facecolor("#F9F9FB")

        if metric.pct:
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        # Subtle best-bar highlight
        best_idx = strats.index(best_s)
        ax.axhspan(
            best_idx - bar_h / 2, best_idx + bar_h / 2,
            color="#FFF9C4", alpha=0.5, zorder=1,
        )

    # ── Figure-level legend (below grid) ───────────────────────────────────────
    handles = [
        matplotlib.patches.Patch(
            facecolor=strategy_meta[s]["color"],
            edgecolor="white",
            label=strategy_meta[s]["label"],
        )
        for s in strategy_order if any(r["strategy"] == s for r in rows)
    ]
    handles += [
        matplotlib.patches.Patch(facecolor="none", edgecolor="none",
                                  label="★ = best for that metric"),
        matplotlib.patches.Patch(facecolor="#FFF9C4", edgecolor="#AAAAAA",
                                  label="highlighted = best row"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        fontsize=8.5,
        title="Strategies",
        title_fontsize=9,
        columnspacing=1.2,
        handlelength=1.5,
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------

def make_radar_figure(
    rows: list[dict],
    tau: float,
    save_path: str | Path | None,
    dpi: int = 180,
    strategy_order: list[str] | None = None,
    strategy_meta:  dict[str, dict] | None = None,
) -> plt.Figure:
    """
    Spider / radar chart normalising 4 metrics to [0, 1] where 1 = best.
    Metrics where lower is better are inverted before normalisation.
    """
    if strategy_order is None or strategy_meta is None:
        strategy_order, strategy_meta = _detect_meta(rows)

    strats = [s for s in strategy_order if any(r["strategy"] == s for r in rows)]

    # 4 radar axes, matching the 2×2 bar chart
    radar_metrics = [
        ("accuracy",     "Accuracy",          "high"),
        ("mean_n_tests", "Efficiency\n(fewer tests)", "low"),
        ("mean_cost",    "Cost\nSavings",      "low"),
        ("mean_burden",  "Patient\nComfort",   "low"),
    ]
    N = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Normalise each metric to [0, 1] across strategies
    def normalise(col: str, good_dir: str) -> dict[str, float]:
        vals = {s: _val(rows, s, col) for s in strats}
        lo, hi = min(vals.values()), max(vals.values())
        span = hi - lo or 1.0
        if good_dir == "high":
            return {s: (v - lo) / span for s, v in vals.items()}
        else:
            return {s: (hi - v) / span for s, v in vals.items()}

    norm_vals: list[dict[str, float]] = [
        normalise(col, gd) for col, _, gd in radar_metrics
    ]

    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw=dict(polar=True),
    )
    fig.patch.set_facecolor("#F0F2F6")
    ax.set_facecolor("#F9F9FB")

    for strat in strats:
        meta = strategy_meta[strat]
        vals_norm = [nv[strat] for nv in norm_vals]
        vals_norm += vals_norm[:1]
        ax.plot(angles, vals_norm, color=meta["color"],
                linewidth=2.2, linestyle="-", label=meta["label"])
        ax.fill(angles, vals_norm, color=meta["color"], alpha=0.13)

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [label for _, label, _ in radar_metrics],
        fontsize=9,
    )
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="#888888")
    ax.set_ylim(0, 1)
    ax.grid(color="#CCCCCC", linestyle="--", linewidth=0.8)

    n_patients = rows[0]["n_patients"] if rows else 0
    fig.suptitle(
        f"Strategy Radar — normalised metrics  |  τ = {tau:.4f}\n"
        f"{n_patients} test patients  ·  1 = best possible for each axis",
        fontsize=11, fontweight="bold", color="#1A1A2E",
    )
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.38, 1.15),
        fontsize=8,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        title="Strategies",
        title_fontsize=9,
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {save_path}")
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _infer_tau(csv_path: str | Path) -> float:
    """Try to read tau from header comment or fall back to 0.7461."""
    # The CSV itself doesn't store tau; it's hard-coded in evaluate.py.
    # We just return the canonical value used in evaluation.
    return 0.7461


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv",      default=str(DEFAULT_CSV),
                    help="Path to evaluation_summary CSV.")
    ap.add_argument("--save-dir", default=str(DEFAULT_FIG),
                    help="Directory to save PNGs.")
    ap.add_argument("--dpi",      type=int,   default=180)
    ap.add_argument("--tau",      type=float, default=None,
                    help="τ value (for title). Defaults to 0.7461.")
    ap.add_argument("--radar",    action="store_true",
                    help="Also produce a radar/spider chart.")
    ap.add_argument("--show",     action="store_true",
                    help="Show figures interactively.")
    args = ap.parse_args()

    rows = load_csv(args.csv)
    tau  = args.tau if args.tau is not None else 0.7461
    n_p  = rows[0]["n_patients"] if rows else "?"

    # Auto-detect 1-step vs multi-step and output name stem
    sorder, smeta = _detect_meta(rows)
    csv_stem = Path(args.csv).stem   # e.g. evaluation_summary or evaluation_summary_multistep

    print(f"Loaded {args.csv}")
    print(f"  Strategies : {[r['strategy'] for r in rows]}")
    print(f"  n_patients : {n_p}")
    print(f"  τ          : {tau}")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 2×2 panel bar chart ──────────────────────────────────────────────
    panel_path = (save_dir / f"{csv_stem}_panel.png") if save_dir else None
    fig_panel  = make_panel_figure(rows, tau, panel_path, args.dpi,
                                   strategy_order=sorder, strategy_meta=smeta)
    if args.show or not save_dir:
        plt.show()
    else:
        plt.close(fig_panel)

    # ── 2. Radar chart (optional) ─────────────────────────────────────────────
    if args.radar:
        radar_path = (save_dir / f"{csv_stem}_radar.png") if save_dir else None
        fig_radar  = make_radar_figure(rows, tau, radar_path, args.dpi,
                                       strategy_order=sorder, strategy_meta=smeta)
        if args.show or not save_dir:
            plt.show()
        else:
            plt.close(fig_radar)


if __name__ == "__main__":
    main()
