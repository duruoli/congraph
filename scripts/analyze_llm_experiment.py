#!/usr/bin/env python3
"""Compute gap_comparison_table.csv and certainty_stratified.csv.

Convention (rubric-relative, same as scripts/gap_analysis.py):
  commission = agent ordered extra tests not recommended by rubric
  omission   = agent skipped rubric-recommended tests

Conditions compared: llm_features_only, llm_full, doctor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.llm_experiment.gap_analysis import build_full_tables


def _sample_reasoning_md(steps_csv: Path, out: Path, n_samples: int = 30) -> None:
    df = pd.read_csv(steps_csv)
    # Stratified sample: try to cover (condition, disease, cs_tercile)
    if len(df) <= n_samples:
        picks = df
    else:
        picks = (
            df.groupby(["condition", "disease", "cs_tercile"], group_keys=False, observed=True)
              .apply(lambda g: g.sample(min(2, len(g)), random_state=42))
              .reset_index(drop=True)
        )
        if len(picks) > n_samples:
            picks = picks.sample(n_samples, random_state=42).reset_index(drop=True)
        elif len(picks) < n_samples:
            extra = df.drop(picks.index, errors="ignore").sample(
                min(n_samples - len(picks), len(df) - len(picks)), random_state=42
            )
            picks = pd.concat([picks, extra]).reset_index(drop=True)

    lines = ["# LLM reasoning samples\n\n"]
    lines.append(f"Sampled {len(picks)} step-level reasoning snippets from {len(df)} total.\n\n")
    for _, r in picks.iterrows():
        lines.append(
            f"## patient {r['patient_id']} | {r['disease']} | cs={r['cs_tercile']} | "
            f"cond={r['condition']} | order={r['info_order']} | step {r['step_index']}\n"
        )
        lines.append(f"- tests_done_before: `{r['tests_done_before']}`\n")
        lines.append(f"- rubric_next: `{r['rubric_next_test']}`\n")
        lines.append(f"- knn_top1_disease: `{r.get('knn_top1_disease', '')}`\n")
        lines.append(f"- llm_next_test: **{r['llm_next_test']}**\n")
        lines.append(f"- termination_reason: {r['termination_reason']}\n")
        if r.get("condition") == "llm_rubric":
            follows = r.get("follows_rubric", "")
            dev = r.get("deviation_reason", "")
            lines.append(f"- follows_rubric: {follows}\n")
            if dev:
                lines.append(f"- deviation_reason: {dev}\n")
        lines.append(f"\n> {r['llm_reasoning']}\n\n")
    out.write_text("".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default=str(ROOT / "results" / "llm_experiment"))
    p.add_argument("--seq-file", default="llm_recommendations.csv")
    p.add_argument("--steps-file", default="llm_recommendations_steps.csv")
    args = p.parse_args()
    in_dir = Path(args.in_dir)
    seq_csv = in_dir / args.seq_file
    steps_csv = in_dir / args.steps_file

    per_patient, gap_by_cd, gap_by_cs = build_full_tables(seq_csv)

    per_patient.to_csv(in_dir / "per_patient_gap.csv", index=False)
    gap_by_cd.to_csv(in_dir / "gap_comparison_table.csv", index=False)
    gap_by_cs.to_csv(in_dir / "certainty_stratified.csv", index=False)
    print(f"wrote: per_patient_gap.csv  ({len(per_patient)} rows)")
    print(f"wrote: gap_comparison_table.csv  ({len(gap_by_cd)} rows)")
    print(f"wrote: certainty_stratified.csv  ({len(gap_by_cs)} rows)")

    _sample_reasoning_md(steps_csv, in_dir / "llm_reasoning_samples.md")
    print("wrote: llm_reasoning_samples.md")

    print("\n=== gap_comparison_table ===")
    print(gap_by_cd.to_string(index=False))
    print("\n=== certainty_stratified ===")
    print(gap_by_cs.to_string(index=False))


if __name__ == "__main__":
    main()
