#!/usr/bin/env python3
"""Run pipeline.rubric_simulator.simulate_cohort_rubric with oracle_routing=True.

Loads cohort JSONs from results/*_features.json (same layout as knn_feature_eval),
fits FeatureSimulator on the training split, then:

    results = simulate_cohort_rubric(patients, simulator, oracle_routing=True)

where ``patients`` is the test cohort only (nested dict). Writes JSON under results/.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.knn_feature_eval import DISEASE_FILES, load_all
from knn.feature_simulator import FeatureSimulator
from pipeline.rubric_simulator import simulate_cohort_rubric, summarize_rubric_simulations


def _nested_test_dict(
    test_rows: list[tuple[str, str, list[dict]]],
) -> dict[str, dict[str, list[dict]]]:
    patients: dict[str, dict[str, list[dict]]] = {d: {} for d in DISEASE_FILES}
    for disease, pid, steps in test_rows:
        patients[disease][pid] = steps
    return patients


def _nested_train_dict(
    train_rows: list[tuple[str, str, list[dict]]],
) -> dict[str, dict[str, list[dict]]]:
    d: dict[str, dict[str, list[dict]]] = {x: {} for x in DISEASE_FILES}
    for disease, pid, steps in train_rows:
        d[disease][pid] = steps
    return d


def main() -> None:
    p = argparse.ArgumentParser(description="Rubric cohort sim with oracle_routing=True.")
    p.add_argument("--n-test", type=int, default=300, help="Test patients after shuffle.")
    p.add_argument("--n-train", type=int, default=None, help="Train for KNN; default rest.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--min-tests", type=int, default=2)
    p.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "rubric_sim_oracle.json"),
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    all_patients = load_all()
    flat: list[tuple[str, str, list[dict]]] = [
        (disease, pid, steps)
        for disease, patients in all_patients.items()
        for pid, steps in patients.items()
    ]
    min_steps = args.min_tests + 1
    flat_filtered = [(d, pid, s) for d, pid, s in flat if len(s) >= min_steps]

    random.seed(args.seed)
    random.shuffle(flat_filtered)

    n_test = min(args.n_test, len(flat_filtered))
    n_train = args.n_train
    if n_train is None:
        n_train = len(flat_filtered) - n_test
    n_train = min(n_train, len(flat_filtered) - n_test)

    test_rows = flat_filtered[:n_test]
    train_rows = flat_filtered[n_test : n_test + n_train]

    train_dict = _nested_train_dict(train_rows)
    patients = _nested_test_dict(test_rows)

    simulator = FeatureSimulator(k=args.k)
    simulator.fit(train_dict)

    results = simulate_cohort_rubric(
        patients,
        simulator,
        oracle_routing=True,
        verbose=args.verbose,
    )

    summary = summarize_rubric_simulations(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "summary": summary,
        "meta": {
            "n_test": n_test,
            "n_train": n_train,
            "seed": args.seed,
            "k": args.k,
            "min_tests_filter": args.min_tests,
            "oracle_routing": True,
        },
        "results": {
            disease: {pid: asdict(res) for pid, res in per.items()}
            for disease, per in results.items()
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
