"""Part 4 — Efficiency tradeoff (Δlength, Δcost) stratified by CertaintyScore.

Inputs
------
  results/test_seq_comparison/test_sequence_comparison.csv
        Rubric (oracle-routing) vs actual test sequences for n=300.
  results/gap_analysis/part3_certainty_score.csv
        Per-patient step-0 CertaintyScore (oracle disease).
  results/gap_analysis/joined_data.csv
        Provides has_commission / has_omission / has_order_swap flags.

Cost mapping
------------
  CMS 2025 PFS midpoints, transcribed from evaluation/test_burden_cost.py
  docstring.  We DO NOT rely on the cost_mapping.csv loader path because
  data/raw_data/cost_mapping.csv is not present in the working tree.

Convention
----------
  Δlength = actual_length − rubric_length  (positive ⇒ doctor did more)
  Δcost   = actual_cost   − rubric_cost   (positive ⇒ doctor spent more)

  Two Radiograph_Chest variants are reported:
    include_chest : everything counted as-is.
    exclude_chest : Radiograph_Chest stripped from actual sequences
                    (matches Part 1's commission-exclusion convention,
                    since the rubric never prescribes a chest x-ray).

Outputs
-------
  results/gap_analysis/part4_efficiency_tradeoff.csv
  results/gap_analysis/part4_per_patient.csv
  results/gap_analysis/part4_figures/{delta_length,delta_cost}_by_certainty.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "gap_analysis"
FIG_DIR = OUT_DIR / "part4_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]

# CMS 2025 PFS midpoints (USD).  See evaluation/test_burden_cost.py docstring.
TEST_COST_USD: dict[str, float] = {
    "Lab_Panel":          49.0,
    "Radiograph_Chest":   67.0,
    "Ultrasound_Abdomen": 170.0,
    "HIDA_Scan":          550.0,
    "CT_Abdomen":         785.0,
    "MRCP_Abdomen":       1225.0,
    "MRI_Abdomen":        1375.0,
}

EXCLUDE_FROM_ACTUAL = {"Radiograph_Chest"}  # Part 1 convention


def _parse_seq(s: str) -> list[str]:
    if pd.isna(s):
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def _cost(seq: list[str]) -> float:
    return float(sum(TEST_COST_USD.get(t, 0.0) for t in seq))


def main() -> None:
    comp = pd.read_csv(RESULTS / "test_seq_comparison" / "test_sequence_comparison.csv")
    comp["patient_id"] = comp["patient_id"].astype(str)
    comp["sim_list"]    = comp["simulated_sequence"].apply(_parse_seq)
    comp["actual_list"] = comp["actual_sequence"].apply(_parse_seq)

    cs = pd.read_csv(OUT_DIR / "part3_certainty_score.csv")
    cs["patient_id"] = cs["patient_id"].astype(str)

    joined = pd.read_csv(OUT_DIR / "joined_data.csv")
    joined["patient_id"] = joined["patient_id"].astype(str)

    df = comp.merge(
        cs[["patient_id", "certainty_score_oracle", "perceived_top_diagnosis", "n_triggered"]],
        on="patient_id", how="left",
    ).merge(
        joined[["patient_id", "has_commission", "has_omission", "has_order_swap"]],
        on="patient_id", how="left",
    )

    if df["certainty_score_oracle"].isna().any():
        miss = df[df["certainty_score_oracle"].isna()]
        raise RuntimeError(f"{len(miss)} patients missing CertaintyScore")

    # Per-patient Δlength / Δcost in both variants
    def _row_metrics(row) -> dict:
        sim, actual = row["sim_list"], row["actual_list"]
        actual_strip = [t for t in actual if t not in EXCLUDE_FROM_ACTUAL]
        return {
            "rubric_length":          len(sim),
            "actual_length":          len(actual),
            "actual_length_no_chest": len(actual_strip),
            "rubric_cost":            _cost(sim),
            "actual_cost":            _cost(actual),
            "actual_cost_no_chest":   _cost(actual_strip),
            "delta_length":           len(actual) - len(sim),
            "delta_length_no_chest":  len(actual_strip) - len(sim),
            "delta_cost":             _cost(actual) - _cost(sim),
            "delta_cost_no_chest":    _cost(actual_strip) - _cost(sim),
        }

    metrics = df.apply(_row_metrics, axis=1).apply(pd.Series)
    pat = pd.concat([df, metrics], axis=1)

    # Disease-specific certainty terciles
    pat["certainty_tercile"] = (
        pat.groupby("disease")["certainty_score_oracle"]
           .transform(lambda s: pd.qcut(s, q=3, labels=["low", "mid", "high"], duplicates="drop"))
    )
    pat.to_csv(OUT_DIR / "part4_per_patient.csv", index=False)

    # Stratified summary
    metric_cols = [
        "rubric_length", "actual_length", "actual_length_no_chest",
        "rubric_cost", "actual_cost", "actual_cost_no_chest",
        "delta_length", "delta_length_no_chest",
        "delta_cost", "delta_cost_no_chest",
    ]
    summary_rows = []
    for disease, g in pat.groupby("disease"):
        for tercile in ["low", "mid", "high"]:
            sub = g[g["certainty_tercile"] == tercile]
            if not len(sub):
                continue
            row = {"disease": disease, "certainty_tercile": tercile, "n": len(sub),
                   "commission_rate": sub["has_commission"].mean(),
                   "omission_rate":   sub["has_omission"].mean(),
                   "order_swap_rate": sub["has_order_swap"].mean(),
                   "mean_certainty":  sub["certainty_score_oracle"].mean()}
            for c in metric_cols:
                row[f"{c}_mean"] = sub[c].mean()
                row[f"{c}_sd"]   = sub[c].std(ddof=1) if len(sub) > 1 else 0.0
            summary_rows.append(row)
        # disease-level aggregate
        sub = g
        row = {"disease": disease, "certainty_tercile": "all", "n": len(sub),
               "commission_rate": sub["has_commission"].mean(),
               "omission_rate":   sub["has_omission"].mean(),
               "order_swap_rate": sub["has_order_swap"].mean(),
               "mean_certainty":  sub["certainty_score_oracle"].mean()}
        for c in metric_cols:
            row[f"{c}_mean"] = sub[c].mean()
            row[f"{c}_sd"]   = sub[c].std(ddof=1)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "part4_efficiency_tradeoff.csv", index=False)
    print(f"Wrote {OUT_DIR/'part4_efficiency_tradeoff.csv'}  ({len(summary)} rows)")

    # ----- Figures -----
    _plot_strata(pat, "delta_length", "ΔLength (actual − rubric)", FIG_DIR / "delta_length_by_certainty.png")
    _plot_strata(pat, "delta_length_no_chest", "ΔLength (excl. chest x-ray)",
                 FIG_DIR / "delta_length_no_chest_by_certainty.png")
    _plot_strata(pat, "delta_cost", "ΔCost (USD; actual − rubric)", FIG_DIR / "delta_cost_by_certainty.png")
    _plot_strata(pat, "delta_cost_no_chest", "ΔCost (USD; excl. chest x-ray)",
                 FIG_DIR / "delta_cost_no_chest_by_certainty.png")
    _plot_scatter(pat, FIG_DIR / "certainty_vs_delta_length.png")
    print(f"Wrote figures to {FIG_DIR}")

    # Console summary
    cols_show = [
        "disease", "certainty_tercile", "n", "mean_certainty",
        "delta_length_mean", "delta_length_sd",
        "delta_cost_no_chest_mean", "delta_cost_no_chest_sd",
        "commission_rate", "omission_rate",
    ]
    print()
    print(summary[cols_show].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def _plot_strata(pat: pd.DataFrame, ycol: str, ylabel: str, fig_path: Path) -> None:
    fig, axes = plt.subplots(1, len(DISEASES), figsize=(4 * len(DISEASES), 4), sharey=True)
    for ax, disease in zip(axes, DISEASES):
        sub = pat[pat["disease"] == disease]
        data = [sub[sub["certainty_tercile"] == t][ycol].dropna().values
                for t in ["low", "mid", "high"]]
        bp = ax.boxplot(data, tick_labels=["low", "mid", "high"], showmeans=True)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_title(f"{disease}\n(n={len(sub)})")
        ax.set_xlabel("certainty tercile")
        # annotate means
        for i, dat in enumerate(data, start=1):
            if len(dat):
                ax.annotate(f"μ={dat.mean():.2f}", xy=(i, dat.mean()),
                            xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=8, color="darkred")
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"{ylabel} by step-0 CertaintyScore tercile (oracle routing)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(pat: pd.DataFrame, fig_path: Path) -> None:
    fig, axes = plt.subplots(1, len(DISEASES), figsize=(4 * len(DISEASES), 4), sharey=True)
    for ax, disease in zip(axes, DISEASES):
        sub = pat[pat["disease"] == disease]
        ax.scatter(sub["certainty_score_oracle"], sub["delta_length"],
                   alpha=0.5, s=18)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_title(f"{disease}  (n={len(sub)})")
        ax.set_xlabel("CertaintyScore (oracle d)")
    axes[0].set_ylabel("ΔLength")
    fig.suptitle("Step-0 CertaintyScore vs. ΔLength (oracle routing)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
