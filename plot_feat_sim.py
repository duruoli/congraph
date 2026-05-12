"""plot_feat_sim.py — Feature-Simulator Strategy Metrics Visualization

Reads the CSV produced by evaluating KNN-based feature simulation strategies
(columns: alpha, tau, strategy, n_patients, accuracy, mean_stop_tests,
mean_cost, mean_burden, pct_early_stop) and generates figures with four
metric panels:

  • Accuracy        vs τ
  • Mean tests      vs τ   (speed)
  • Mean cost       vs τ
  • Mean burden     vs τ

Two figure layouts are produced per alpha value found in the CSV:
  1. All strategies overlaid (one PNG per alpha)
  2. Same but showing only "deployable" strategies (excludes actual_real /
     actual_sim which are oracle / cheating baselines) — suffix "_deployable"

A comparison figure with one sub-panel per alpha is also produced for each
metric.

Usage
-----
  python plot_feat_sim.py
  python plot_feat_sim.py --csv results/test_feature_sim.csv
  python plot_feat_sim.py --save-dir results/figs --dpi 150
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from itertools import groupby

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

DEFAULT_CSV = Path(__file__).parent / "results" / "test_feature_sim.csv"

# ── Strategy visual styles ────────────────────────────────────────────────────
_STRATEGY_STYLE: dict[str, dict] = {
    "actual_real": dict(
        color="#2c7bb6", marker="o",  linestyle="-",
        label="Actual (real features)", zorder=5,
    ),
    "actual_sim": dict(
        color="#74add1", marker="s",  linestyle="--",
        label="Actual (simulated features)", zorder=4,
    ),
    "bfs_sim": dict(
        color="#d7191c", marker="^",  linestyle="-",
        label="BFS (simulated)", zorder=3,
    ),
    "ig_sim": dict(
        color="#1a9641", marker="D",  linestyle="-",
        label="IG re-ranked (simulated)", zorder=6,
    ),
    "random_sim": dict(
        color="#999999", marker="x",  linestyle=":",
        label="Random (simulated)", zorder=2,
    ),
}

# ── Metric definitions ────────────────────────────────────────────────────────
_METRICS = [
    ("accuracy",        "Diagnostic accuracy",  True),   # (col, ylabel, pct_fmt)
    ("mean_stop_tests", "Mean tests ordered",   False),
    ("mean_cost",       "Mean cost",            False),
    ("mean_burden",     "Mean burden",          False),
]

_DEPLOYABLE = {"bfs_sim", "ig_sim", "random_sim"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: str | Path) -> dict[float, dict[str, list[dict]]]:
    """
    Return {alpha: {strategy: [row_dict, ...]}} sorted by tau ascending.

    row_dict keys: tau, alpha, strategy, accuracy, mean_stop_tests,
                   mean_cost, mean_burden, pct_early_stop, n_patients
    """
    rows_all: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_all.append({
                "alpha":          float(row["alpha"]),
                "tau":            float(row["tau"]),
                "strategy":       row["strategy"].strip(),
                "accuracy":       float(row["accuracy"]),
                "mean_stop_tests": float(row["mean_stop_tests"]),
                "mean_cost":      float(row["mean_cost"]),
                "mean_burden":    float(row["mean_burden"]),
                "pct_early_stop": float(row["pct_early_stop"]),
                "n_patients":     int(row["n_patients"]),
            })

    # Group by alpha → strategy
    result: dict[float, dict[str, list[dict]]] = {}
    rows_all.sort(key=lambda r: (r["alpha"], r["strategy"], r["tau"]))
    for alpha, alpha_group in groupby(rows_all, key=lambda r: r["alpha"]):
        result[alpha] = {}
        for strat, strat_group in groupby(alpha_group, key=lambda r: r["strategy"]):
            result[alpha][strat] = sorted(strat_group, key=lambda r: r["tau"])
    return result


# ---------------------------------------------------------------------------
# Single-alpha 4-panel figure
# ---------------------------------------------------------------------------

def _plot_four_metrics(
    strat_data: dict[str, list[dict]],
    alpha: float,
    n_patients: int,
    deployable_only: bool,
    ax_arr: list[plt.Axes],
) -> None:
    """
    Fill four Axes with one metric each for the given alpha/strategy data.

    ax_arr: [ax_accuracy, ax_tests, ax_cost, ax_burden]
    """
    strategies = [s for s in strat_data if (not deployable_only or s in _DEPLOYABLE)]

    for ax, (col, ylabel, pct_fmt) in zip(ax_arr, _METRICS):
        for strat in strategies:
            rows = strat_data[strat]
            style = _STRATEGY_STYLE.get(strat, dict(
                color="#555555", marker="o", linestyle="-", label=strat, zorder=2,
            ))
            taus = [r["tau"] for r in rows]
            vals = [r[col]   for r in rows]
            ax.plot(
                taus, vals,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.8, markersize=5,
                label=style["label"],
                zorder=style["zorder"],
            )

        ax.set_xlabel("Stopping threshold τ (bits)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel + f"  |  α={alpha:.2f}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # high τ (fast) on left; low τ (thorough) on right
        ax.legend(fontsize=7, loc="best")
        if pct_fmt:
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(xmax=1, decimals=0)
            )


def make_single_alpha_figure(
    strat_data: dict[str, list[dict]],
    alpha: float,
    n_patients: int,
    deployable_only: bool,
) -> plt.Figure:
    """Return a 2×2 figure with four metric panels for one alpha."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    suffix = "  (deployable strategies only)" if deployable_only else ""
    fig.suptitle(
        f"Feature-Simulator Strategy Comparison — {n_patients} test patients"
        f"  |  α={alpha:.2f}{suffix}\n"
        "ConGraph diagnostic pipeline  ·  KNN feature simulation",
        fontsize=12, fontweight="bold",
    )
    _plot_four_metrics(strat_data, alpha, n_patients, deployable_only, ax_list)
    return fig


# ---------------------------------------------------------------------------
# Per-metric comparison figure across alpha values
# ---------------------------------------------------------------------------

def make_metric_comparison_figure(
    data: dict[float, dict[str, list[dict]]],
    metric_col: str,
    metric_label: str,
    pct_fmt: bool,
    n_patients: int,
    deployable_only: bool,
) -> plt.Figure:
    """
    One sub-panel per alpha value showing the given metric for all strategies.
    """
    alphas = sorted(data.keys())
    n_cols = min(3, len(alphas))
    n_rows = math.ceil(len(alphas) / n_cols)

    fig, axes_arr = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.5 * n_rows),
        constrained_layout=True,
    )
    # Normalise to flat list
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes_arr]
    elif n_rows == 1 or n_cols == 1:
        axes_flat = list(axes_arr.flat) if hasattr(axes_arr, "flat") else [axes_arr]
    else:
        axes_flat = list(axes_arr.flat)

    suffix = "  (deployable only)" if deployable_only else ""
    fig.suptitle(
        f"{metric_label} — comparison across α  |  {n_patients} test patients{suffix}\n"
        "ConGraph  ·  KNN feature simulation",
        fontsize=12, fontweight="bold",
    )

    for idx, alpha in enumerate(alphas):
        ax = axes_flat[idx]
        strat_data = data[alpha]
        strategies = [s for s in strat_data if (not deployable_only or s in _DEPLOYABLE)]

        for strat in strategies:
            rows = strat_data[strat]
            style = _STRATEGY_STYLE.get(strat, dict(
                color="#555555", marker="o", linestyle="-", label=strat, zorder=2,
            ))
            ax.plot(
                [r["tau"]       for r in rows],
                [r[metric_col]  for r in rows],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.8, markersize=5,
                label=style["label"],
                zorder=style["zorder"],
            )

        ax.set_xlabel("τ (bits)", fontsize=9)
        ax.set_ylabel(metric_label, fontsize=9)
        ax.set_title(f"α={alpha:.2f}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        ax.legend(fontsize=6, loc="best")
        if pct_fmt:
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(xmax=1, decimals=0)
            )

    # Hide unused axes
    for idx in range(len(alphas), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv",      default=str(DEFAULT_CSV),
                    help="Path to the feature-sim results CSV.")
    ap.add_argument("--save-dir", default=None,
                    help="Directory to save PNGs. If omitted, plt.show() is called.")
    ap.add_argument("--dpi",      type=int, default=150)
    args = ap.parse_args()

    data = load_csv(args.csv)
    alphas = sorted(data.keys())

    # Infer n_patients from first available row
    first_strat_data = data[alphas[0]]
    first_rows = next(iter(first_strat_data.values()))
    n_patients = first_rows[0]["n_patients"]

    print(f"Loaded {args.csv}")
    print(f"  α values : {alphas}")
    print(f"  strategies: {sorted(next(iter(data.values())).keys())}")
    print(f"  n_patients: {n_patients}")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Per-alpha 4-panel figures ──────────────────────────────────────────
    metric_short = {
        "accuracy":        "accuracy",
        "mean_stop_tests": "tests",
        "mean_cost":       "cost",
        "mean_burden":     "burden",
    }

    for alpha in alphas:
        for deployable_only in (False, True):
            fig = make_single_alpha_figure(
                data[alpha], alpha, n_patients, deployable_only
            )
            if save_dir:
                suffix = "_deployable" if deployable_only else ""
                fname = f"feat_sim_4metrics_alpha_{alpha:.2f}{suffix}.png"
                out = save_dir / fname
                fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
                print(f"Saved → {out}")
                plt.close(fig)
            else:
                plt.show()

    # ── 2. Per-metric comparison across alpha values ──────────────────────────
    for col, ylabel, pct_fmt in _METRICS:
        for deployable_only in (False, True):
            fig = make_metric_comparison_figure(
                data, col, ylabel, pct_fmt, n_patients, deployable_only
            )
            if save_dir:
                suffix = "_deployable" if deployable_only else ""
                mname = metric_short[col]
                fname = f"feat_sim_{mname}_alpha_comparison{suffix}.png"
                out = save_dir / fname
                fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
                print(f"Saved → {out}")
                plt.close(fig)
            else:
                plt.show()

    # ── 3. Per-metric per-alpha individual figures ────────────────────────────
    for alpha in alphas:
        for col, ylabel, pct_fmt in _METRICS:
            for deployable_only in (False, True):
                fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
                strat_data = data[alpha]
                strategies = [s for s in strat_data
                              if (not deployable_only or s in _DEPLOYABLE)]

                for strat in strategies:
                    rows = strat_data[strat]
                    style = _STRATEGY_STYLE.get(strat, dict(
                        color="#555555", marker="o", linestyle="-", label=strat, zorder=2,
                    ))
                    ax.plot(
                        [r["tau"]  for r in rows],
                        [r[col]    for r in rows],
                        color=style["color"],
                        marker=style["marker"],
                        linestyle=style["linestyle"],
                        linewidth=1.8, markersize=5,
                        label=style["label"],
                        zorder=style["zorder"],
                    )

                dep_str = "  (deployable only)" if deployable_only else ""
                fig.suptitle(
                    f"{ylabel}  |  α={alpha:.2f}{dep_str}\n"
                    "ConGraph  ·  KNN feature simulation",
                    fontsize=11, fontweight="bold",
                )
                ax.set_xlabel("Stopping threshold τ (bits)", fontsize=10)
                ax.set_ylabel(ylabel, fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.invert_xaxis()
                ax.legend(fontsize=8, loc="best")
                if pct_fmt:
                    ax.yaxis.set_major_formatter(
                        matplotlib.ticker.PercentFormatter(xmax=1, decimals=0)
                    )

                if save_dir:
                    suffix = "_deployable" if deployable_only else ""
                    mname = metric_short[col]
                    fname = f"feat_sim_{mname}_alpha_{alpha:.2f}{suffix}.png"
                    out = save_dir / fname
                    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
                    print(f"Saved → {out}")
                    plt.close(fig)
                else:
                    plt.show()


if __name__ == "__main__":
    main()
