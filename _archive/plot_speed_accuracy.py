"""plot_speed_accuracy.py — Speed-Accuracy visualization

Reads the CSV produced by speed_accuracy_eval.py and generates a 2×2 figure:

  Left column  — fix α, sweep τ
    (1,1) Accuracy    vs τ  for actual / random / bfs / IG(α_fixed)
    (2,1) Mean tests  vs τ  same strategies

  Right column — fix τ, sweep α   (only when an α sweep exists in the CSV)
    (1,2) Accuracy    vs α  for IG, one line per selected τ;
          horizontal reference bands for actual / bfs at same τ values
    (2,2) Mean tests  vs α  same

The "Speed-Accuracy Frontier" (accuracy vs mean_steps) is intentionally
omitted — mean_steps is a *dependent* variable, not a valid x-axis.

Usage
-----
  python plot_speed_accuracy.py
  python plot_speed_accuracy.py --csv path/to/file.csv
  python plot_speed_accuracy.py --fixed-alpha 0.5 --n-tau-slices 5
  python plot_speed_accuracy.py --save-dir results/figs
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.cm as cm
import numpy as np

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "results" / "speed_accuracy.csv"

# ── Baseline strategy styles ──────────────────────────────────────────────────
_BASELINE_STYLE: dict[str, dict] = {
    "actual": dict(color="#2c7bb6", marker="o",  linestyle="-",  label="Actual order", zorder=4),
    "random": dict(color="#999999", marker="s",  linestyle="--", label="Random order", zorder=2),
    "bfs":    dict(color="#d7191c", marker="^",  linestyle="-",  label="BFS order",    zorder=3),
}
_IG_BASE_COLOR = "#1a9641"   # used for single-α IG or fixed-α highlight


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: str | Path) -> dict[str, list[dict]]:
    """
    Return {strategy_key: [row_dict, ...]} sorted by tau ascending.

    Strategy keys: "actual", "random", "bfs", "ig_0.00", "ig_0.50", …
    Legacy CSVs (strategy = "ig") are handled transparently.
    """
    data: dict[str, list[dict]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row["strategy"]
            if s not in data:
                data[s] = []
            alpha_str = row.get("alpha", "")
            data[s].append({
                "tau":            float(row["tau"]),
                "accuracy":       float(row["accuracy"]),
                "mean_steps":     float(row["mean_steps"]),
                "pct_early_stop": float(row["pct_early_stop"]),
                "n_patients":     int(row["n_patients"]),
                "alpha":          float(alpha_str) if alpha_str else float("nan"),
            })
    for s in data:
        data[s].sort(key=lambda r: r["tau"])
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ig_alpha_from_key(key: str) -> float | None:
    """Parse alpha from "ig_0.50"; returns None for legacy "ig" key."""
    parts = key.split("_", 1)
    if len(parts) == 2:
        try:
            return float(parts[1])
        except ValueError:
            pass
    return None


def _ig_keys_sorted(data: dict[str, list[dict]]) -> list[str]:
    """Return IG strategy keys sorted by α value."""
    ig = [k for k in data if k not in _BASELINE_STYLE]
    return sorted(ig, key=lambda k: (
        _ig_alpha_from_key(k) if _ig_alpha_from_key(k) is not None else 0.5
    ))


def _closest_ig_key(ig_keys: list[str], target_alpha: float) -> str:
    """Return the IG key whose α is closest to target_alpha."""
    def _dist(k: str) -> float:
        a = _ig_alpha_from_key(k)
        return abs((a if a is not None else 0.5) - target_alpha)
    return min(ig_keys, key=_dist)


def _row_at_tau(rows: list[dict], tau: float) -> dict | None:
    return next((r for r in rows if abs(r["tau"] - tau) < 1e-9), None)


def _alpha_val(key: str, row: dict) -> float:
    """Get α value from row or key (handles legacy format)."""
    a = row["alpha"]
    if math.isnan(a):
        parsed = _ig_alpha_from_key(key)
        a = parsed if parsed is not None else 0.5
    return a


# ---------------------------------------------------------------------------
# Left column — fixed α, sweep τ
# ---------------------------------------------------------------------------

def _plot_fixed_alpha(
    data: dict[str, list[dict]],
    ig_key: str,
    ax_acc: plt.Axes,
    ax_spd: plt.Axes,
) -> None:
    """
    Plot accuracy and mean_tests vs τ for baselines + one IG curve.
    Annotates the fixed α value on the IG line in both panels.
    """
    alpha_val = _ig_alpha_from_key(ig_key)
    alpha_label = f"α={alpha_val:.2f}" if alpha_val is not None else None

    ig_label = f"IG re-ranked  ({alpha_label})" if alpha_label else "IG re-ranked"
    ig_style = dict(
        color=_IG_BASE_COLOR, marker="D", linestyle="-",
        label=ig_label, zorder=5,
    )

    fixed_str = f"fixed {alpha_label}" if alpha_label else "fixed IG"
    for ax, ykey, ylabel, title in [
        (ax_acc, "accuracy",   "Diagnostic accuracy",  f"Accuracy vs τ  |  {fixed_str}"),
        (ax_spd, "mean_steps", "Mean tests ordered",   f"Mean Tests vs τ  |  {fixed_str}"),
    ]:
        # Baselines
        for bname, bstyle in _BASELINE_STYLE.items():
            if bname not in data:
                continue
            rows = data[bname]
            ax.plot(
                [r["tau"] for r in rows],
                [r[ykey]  for r in rows],
                color=bstyle["color"], marker=bstyle["marker"],
                linestyle=bstyle["linestyle"], linewidth=1.8, markersize=5,
                label=bstyle["label"], zorder=bstyle["zorder"],
            )

        # IG fixed-α line
        ig_rows = data.get(ig_key, [])
        taus = [r["tau"] for r in ig_rows]
        vals = [r[ykey]  for r in ig_rows]
        ax.plot(
            taus, vals,
            color=ig_style["color"], marker=ig_style["marker"],
            linestyle=ig_style["linestyle"], linewidth=2.2, markersize=6,
            label=ig_style["label"], zorder=ig_style["zorder"],
        )

        # Annotate α on the midpoint of the IG line (only when alpha is known)
        if taus and alpha_label:
            mid = len(taus) // 2
            ax.annotate(
                alpha_label,
                xy=(taus[mid], vals[mid]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=9, color=_IG_BASE_COLOR, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=_IG_BASE_COLOR,
                          alpha=0.75, lw=0.8),
            )

        ax.set_xlabel("Stopping threshold τ (bits)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # high τ (fast/uncertain) on left, low τ (slow/confident) on right
        ax.legend(fontsize=8, loc="best")

    ax_acc.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))


# ---------------------------------------------------------------------------
# Right column — fixed τ, sweep α
# ---------------------------------------------------------------------------

def _selected_taus(ig_keys: list[str], data: dict[str, list[dict]], n: int) -> list[float]:
    """Pick n evenly-spaced τ values from the IG sweep."""
    all_taus = sorted({r["tau"] for r in data[ig_keys[0]]})
    step = max(1, (len(all_taus) - 1) // (n - 1)) if n > 1 else len(all_taus)
    picks = all_taus[::step][:n]
    # always include the last τ if not already covered
    if all_taus[-1] not in picks:
        picks = picks[: n - 1] + [all_taus[-1]]
    return picks


def _plot_fixed_tau(
    data: dict[str, list[dict]],
    ig_keys: list[str],
    ax_acc: plt.Axes,
    ax_spd: plt.Axes,
    n_tau_slices: int = 5,
) -> None:
    """
    Plot accuracy and mean_tests vs α for a set of fixed τ values.
    Draws horizontal reference lines for actual / bfs at each τ.
    """
    sel_taus = _selected_taus(ig_keys, data, n_tau_slices)
    tau_colors = cm.plasma(np.linspace(0.15, 0.85, len(sel_taus)))

    for ax, ykey, ylabel, title in [
        (ax_acc, "accuracy",   "Diagnostic accuracy", "Accuracy vs α  |  fixed τ slices"),
        (ax_spd, "mean_steps", "Mean tests ordered",  "Mean Tests vs α  |  fixed τ slices"),
    ]:
        legend_baseline_drawn: set[str] = set()

        for i, tau in enumerate(sel_taus):
            color = matplotlib.colors.to_hex(tau_colors[i])

            # IG line across α values
            alphas_plot, vals_plot = [], []
            for key in ig_keys:
                row = _row_at_tau(data[key], tau)
                if row is None:
                    continue
                alphas_plot.append(_alpha_val(key, row))
                vals_plot.append(row[ykey])

            if alphas_plot:
                ax.plot(
                    alphas_plot, vals_plot,
                    marker="o", linewidth=1.8, markersize=5,
                    color=color, label=f"IG  τ={tau:.2f}",
                )

            # Horizontal reference lines for baselines at this τ
            for bname, bstyle in [
                ("bfs",    dict(linestyle="--")),
                ("actual", dict(linestyle=":")),
            ]:
                if bname not in data:
                    continue
                brow = _row_at_tau(data[bname], tau)
                if brow is None:
                    continue
                lbl = f"{_BASELINE_STYLE[bname]['label']}  τ={tau:.2f}" \
                      if bname not in legend_baseline_drawn else "_nolegend_"
                ax.axhline(
                    brow[ykey],
                    color=_BASELINE_STYLE[bname]["color"],
                    linestyle=bstyle["linestyle"],
                    linewidth=1.0, alpha=0.55,
                    label=lbl,
                )
                legend_baseline_drawn.add(bname)

        ax.set_xlabel("IG weight α  (0 = BFS, 1 = pure IG)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best", ncol=1)

    ax_acc.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",          default=str(DEFAULT_CSV))
    ap.add_argument("--save-dir",     default=None,
                    help="Directory to save PNG. If omitted, plt.show() is called.")
    ap.add_argument("--dpi",          type=int,   default=150)
    ap.add_argument("--fixed-alpha",  type=float, default=0.5,
                    help="α value to fix for the left-column (τ sweep) plots.")
    ap.add_argument("--n-tau-slices", type=int,   default=5,
                    help="Number of τ iso-lines in the right-column (α sweep) plots.")
    args = ap.parse_args()

    data = load_csv(args.csv)
    n_patients = data[next(iter(data))][0]["n_patients"]
    ig_keys = _ig_keys_sorted(data)

    print(
        f"Loaded {args.csv}  |  strategies: {list(data)}  "
        f"|  n_patients={n_patients}  |  IG keys: {ig_keys}"
    )

    has_alpha_sweep = len(ig_keys) > 1

    # Choose the IG key closest to the requested fixed alpha
    ig_key_fixed = _closest_ig_key(ig_keys, args.fixed_alpha) if ig_keys else None
    fixed_alpha_actual = _ig_alpha_from_key(ig_key_fixed) if ig_key_fixed else None

    # ── Layout ────────────────────────────────────────────────────────────────
    if has_alpha_sweep:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        (ax_acc_tau, ax_acc_alpha), (ax_spd_tau, ax_spd_alpha) = axes
    else:
        fig, axes = plt.subplots(2, 1, figsize=(7, 9), constrained_layout=True)
        ax_acc_tau, ax_spd_tau = axes
        ax_acc_alpha = ax_spd_alpha = None

    alpha_str = (
        f"α={fixed_alpha_actual:.2f}" if fixed_alpha_actual is not None else "IG re-ranked"
    )
    sweep_str = (
        f"  ×  α sweep ({len(ig_keys)} values)" if has_alpha_sweep else ""
    )
    fig.suptitle(
        f"Speed-Accuracy Analysis — {n_patients} test patients\n"
        f"ConGraph diagnostic pipeline  |  τ sweep{sweep_str}",
        fontsize=13, fontweight="bold",
    )

    # Left column: fixed α, sweep τ
    if ig_key_fixed:
        _plot_fixed_alpha(data, ig_key_fixed, ax_acc_tau, ax_spd_tau)

    # Right column: fixed τ, sweep α
    if has_alpha_sweep and ax_acc_alpha is not None:
        _plot_fixed_tau(data, ig_keys, ax_acc_alpha, ax_spd_alpha,
                        n_tau_slices=args.n_tau_slices)

    # ── Save or show ──────────────────────────────────────────────────────────
    if args.save_dir:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        out = save_path / "speed_accuracy_curves.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
