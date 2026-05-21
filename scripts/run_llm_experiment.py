#!/usr/bin/env python3
"""Drive the LLM next-test recommendation experiment.

Two modes:
  --mode order      : 3 patients × 2 info orders × 1 condition (llm_full).
                      Writes results/llm_experiment/order_sensitivity_check.csv
                      and order_sensitivity_steps.csv.
  --mode main       : full 30-patient × 2 conditions run (llm_full uses
                      info_order='rubric_first' by default; if --average-orders
                      is passed, both orders are run and stored separately).
                      Writes llm_recommendations.csv and llm_step_records.csv.

The experiment uses the same train/test split as run_rubric_simulator_oracle.py
(seed=42, n_test=300, min_tests=2). All sampled patients live in the test split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knn.feature_simulator import FeatureSimulator

from experiments.llm_experiment.env_loader import load_openai_key
from experiments.llm_experiment.knn_neighbors import PatientKNN
from experiments.llm_experiment.runner import (
    CohortRunConfig,
    build_train_test_split,
    run_cohort,
    trajectories_to_seq_df,
    trajectories_to_step_df,
)
from experiments.llm_experiment.sampling import (
    _join_cs_tsc,
    sample_30,
    sample_order_sensitivity,
)


def all_test_sample() -> "pd.DataFrame":
    """All 300 test patients with disease + CS tercile metadata."""
    import pandas as pd
    df = _join_cs_tsc()
    keep = [
        "patient_id", "disease", "certainty_score_oracle", "cs_tercile",
        "perceived_top_diagnosis", "actual_sequence", "simulated_sequence",
    ]
    return df[keep].reset_index(drop=True)


def _save(traj_list, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_df = trajectories_to_seq_df(traj_list)
    step_df = trajectories_to_step_df(traj_list)
    seq_path = out_dir / f"{prefix}.csv"
    step_path = out_dir / f"{prefix}_steps.csv"
    seq_df.to_csv(seq_path, index=False)
    step_df.to_csv(step_path, index=False)
    print(f"  wrote {seq_path}  ({len(seq_df)} trajectories)")
    print(f"  wrote {step_path}  ({len(step_df)} step records)")
    return seq_df, step_df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["order", "main"], required=True)
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--max-steps", type=int, default=7)
    p.add_argument("--out-dir", default=str(ROOT / "results" / "llm_experiment"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--average-orders", action="store_true",
                   help="In main mode, run both info orders for llm_full")
    p.add_argument("--all-test", action="store_true",
                   help="In main mode, use all 300 test patients instead of the 30-patient sample")
    p.add_argument("--workers", type=int, default=1,
                   help="ThreadPoolExecutor workers; 1 = sequential")
    args = p.parse_args()

    load_openai_key()

    train, test = build_train_test_split(seed=args.seed)
    print(f"train patients: {sum(len(v) for v in train.values())}  "
          f"test patients: {sum(len(v) for v in test.values())}")
    simulator = FeatureSimulator(k=15).fit(train)
    knn = PatientKNN(train)
    print(f"FeatureSimulator transitions: {simulator.n_transitions}  "
          f"KNN train patients: {len(knn.entries)}")

    if args.mode == "order":
        sample_df = sample_order_sensitivity(seed=args.seed)
        cfg = CohortRunConfig(
            sample_df=sample_df,
            conditions=["llm_full"],
            info_orders=["rubric_first", "knn_first"],
            max_steps=args.max_steps,
            model=args.model,
            parallel_workers=args.workers,
        )
        trajs = run_cohort(cfg=cfg, train_dict=train, test_dict=test,
                           simulator=simulator, knn=knn)
        _save(trajs, Path(args.out_dir), "order_sensitivity_check")
        return

    # main mode
    sample_df = all_test_sample() if args.all_test else sample_30(seed=args.seed)
    print(f"main mode sample size: {len(sample_df)} patients "
          f"({'all test' if args.all_test else '30-patient stratified'})")
    info_orders = ["rubric_first", "knn_first"] if args.average_orders else ["rubric_first"]
    cfg = CohortRunConfig(
        sample_df=sample_df,
        conditions=["llm_features_only", "llm_full"],
        info_orders=info_orders,
        max_steps=args.max_steps,
        model=args.model,
        parallel_workers=args.workers,
    )
    trajs = run_cohort(cfg=cfg, train_dict=train, test_dict=test,
                       simulator=simulator, knn=knn)
    _save(trajs, Path(args.out_dir), "llm_recommendations")


if __name__ == "__main__":
    main()
