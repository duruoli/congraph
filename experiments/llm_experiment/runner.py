"""LLM next-test recommendation: open-loop runner.

For each (patient, condition) we run an open-loop trajectory:
  1. Compute current_features (step-k counterfactual via FeatureSimulator)
  2. Compute rubric's next-test (via traverse_graph + first pending_test)
  3. Compute top-5 similar training patients (via PatientKNN)
  4. Build the prompt, call LLM, parse next_test
  5. If next_test == "STOP" or invalid or already-done or sim fails → terminate
  6. Otherwise FeatureSimulator advances features along the LLM's pick, append to seq
  7. Repeat until max_steps or STOP

We share one FeatureSimulator + PatientKNN across all patients/conditions,
fit on the same train pool used by run_rubric_simulator_oracle.py (seed=42,
n_test=300, min_tests=2).

Outputs: a list of per-step records and a per-patient sequence summary.
"""
from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from evaluation.knn_feature_eval import DISEASE_FILES, load_all
from knn.feature_simulator import FeatureSimulator
from pipeline.feature_schema import VALID_TESTS
from pipeline.rubric_graph import DISEASE_GRAPHS
from pipeline.traversal_engine import traverse_graph

from experiments.llm_experiment.knn_neighbors import PatientKNN
from experiments.llm_experiment.llm_client import call_llm
from experiments.llm_experiment.prompts import build_user_prompt


# ---------------------------------------------------------------------------
# Split that matches scripts/run_rubric_simulator_oracle.py
# ---------------------------------------------------------------------------

def build_train_test_split(
    *,
    n_test: int = 300,
    n_train: Optional[int] = None,
    seed: int = 42,
    min_tests: int = 2,
) -> tuple[dict, dict]:
    all_patients = load_all()
    flat = [
        (disease, pid, steps)
        for disease, patients in all_patients.items()
        for pid, steps in patients.items()
    ]
    min_steps = min_tests + 1
    flat = [t for t in flat if len(t[2]) >= min_steps]
    random.seed(seed)
    random.shuffle(flat)
    n_test = min(n_test, len(flat))
    if n_train is None:
        n_train = len(flat) - n_test
    n_train = min(n_train, len(flat) - n_test)
    test_rows = flat[:n_test]
    train_rows = flat[n_test : n_test + n_train]

    def _nest(rows):
        d: dict[str, dict] = {disease: {} for disease in DISEASE_FILES}
        for disease, pid, steps in rows:
            d[disease][pid] = steps
        return d

    return _nest(train_rows), _nest(test_rows)


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    patient_id: str
    disease: str
    cs_tercile: str
    condition: str
    info_order: str           # "rubric_first" | "knn_first" | "n/a"
    step_index: int            # 0-based step at which this LLM call was made
    tests_done_before: list[str]
    rubric_next_test: Optional[str]
    knn_top1_disease: Optional[str]
    llm_next_test: str
    llm_reasoning: str
    terminated: bool           # True if loop terminated at this step
    termination_reason: str    # "stop" | "invalid" | "already_done" | "sim_failed" | "ok"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_s: float = 0.0
    follows_rubric: Optional[bool] = None   # set only for llm_rubric condition
    deviation_reason: str = ""


@dataclass
class TrajectoryRecord:
    patient_id: str
    disease: str
    cs_tercile: str
    condition: str
    info_order: str
    llm_sequence: list[str]
    n_steps: int
    termination_reason: str
    steps: list[StepRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-patient open-loop trajectory
# ---------------------------------------------------------------------------

def _rubric_next_test(features: dict, disease: str) -> Optional[str]:
    """Return the first pending test the rubric sub-rubric is waiting on."""
    res = traverse_graph(DISEASE_GRAPHS[disease], features)
    if res.terminal_node is not None:
        return None
    return res.pending_tests[0] if res.pending_tests else None


def _rubric_all_pending(features: dict, disease: str) -> list[str]:
    """Return all tests the rubric still recommends from the current state."""
    res = traverse_graph(DISEASE_GRAPHS[disease], features)
    if res.terminal_node is not None:
        return []
    return list(res.pending_tests)


def run_one_trajectory(
    *,
    patient_id: str,
    disease: str,
    cs_tercile: str,
    initial_features: dict,
    condition: str,                          # llm_features_only | llm_full
    simulator: FeatureSimulator,
    knn: PatientKNN,
    info_order: str = "rubric_first",
    max_steps: int = 7,
    model: str = "anthropic/claude-sonnet-4-6",
    verbose: bool = False,
) -> TrajectoryRecord:
    current = initial_features
    llm_sequence: list[str] = []
    steps: list[StepRecord] = []
    termination_reason = "max_steps"

    for step_index in range(max_steps):
        tests_done_before = list(current.get("tests_done", []))
        rubric_next = _rubric_next_test(current, disease)
        use_knn = condition in ("llm_full", "llm_full_deviation")
        neighbors = knn.top_n(current, n=5) if use_knn else []
        knn_top1_disease = neighbors[0][0].disease if neighbors else None
        rubric_pending = (
            _rubric_all_pending(current, disease)
            if condition == "llm_full_deviation" else None
        )

        prompt = build_user_prompt(
            features=current,
            condition=condition,
            disease=disease,
            rubric_next_test=rubric_next,
            rubric_pending_tests=rubric_pending,
            neighbors=neighbors,
            info_order=info_order,
        )

        t0 = time.time()
        res = call_llm(user_prompt=prompt, model=model)
        elapsed = time.time() - t0

        nt = res.next_test.strip()

        rec = StepRecord(
            patient_id=patient_id, disease=disease, cs_tercile=cs_tercile,
            condition=condition, info_order=info_order,
            step_index=step_index,
            tests_done_before=tests_done_before,
            rubric_next_test=rubric_next,
            knn_top1_disease=knn_top1_disease,
            llm_next_test=nt,
            llm_reasoning=res.reasoning,
            terminated=False, termination_reason="ok",
            prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
            elapsed_s=elapsed,
            follows_rubric=res.follows_rubric,
            deviation_reason=res.deviation_reason,
        )

        if nt == "STOP":
            rec.terminated = True
            rec.termination_reason = "stop"
            steps.append(rec)
            termination_reason = "stop"
            break
        if nt not in VALID_TESTS:
            rec.terminated = True
            rec.termination_reason = "invalid"
            steps.append(rec)
            termination_reason = "invalid"
            break
        if nt in tests_done_before:
            rec.terminated = True
            rec.termination_reason = "already_done"
            steps.append(rec)
            termination_reason = "already_done"
            break

        new_features = simulator.simulate_features(current, nt)
        if new_features is None:
            rec.terminated = True
            rec.termination_reason = "sim_failed"
            steps.append(rec)
            llm_sequence.append(nt)
            termination_reason = "sim_failed"
            break

        steps.append(rec)
        llm_sequence.append(nt)
        current = new_features
        if verbose:
            print(f"    step {step_index}: LLM -> {nt}  (rubric -> {rubric_next})", flush=True)

    return TrajectoryRecord(
        patient_id=patient_id, disease=disease, cs_tercile=cs_tercile,
        condition=condition, info_order=info_order,
        llm_sequence=llm_sequence, n_steps=len(llm_sequence),
        termination_reason=termination_reason, steps=steps,
    )


# ---------------------------------------------------------------------------
# Cohort runner
# ---------------------------------------------------------------------------

@dataclass
class CohortRunConfig:
    sample_df: pd.DataFrame                  # cols: patient_id, disease, cs_tercile, ...
    conditions: list[str]
    info_orders: list[str]                    # applied per condition that uses full info
    max_steps: int = 7
    model: str = "anthropic/claude-sonnet-4-6"
    verbose: bool = True
    parallel_workers: int = 1                 # >1 enables ThreadPoolExecutor


def _build_jobs(
    cfg: CohortRunConfig, test_dict: dict
) -> list[dict]:
    jobs: list[dict] = []
    for _, row in cfg.sample_df.iterrows():
        pid = str(row["patient_id"])
        disease = row["disease"]
        cs_tercile = row["cs_tercile"]
        steps = test_dict.get(disease, {}).get(pid) or test_dict.get(disease, {}).get(int(pid))
        if steps is None:
            if cfg.verbose:
                print(f"  [skip] {disease}/{pid} not in test split", flush=True)
            continue
        initial = steps[0]["features"]
        for condition in cfg.conditions:
            orders = cfg.info_orders if condition in ("llm_full", "llm_full_deviation") else ["n/a"]
            for info_order in orders:
                jobs.append({
                    "patient_id": pid, "disease": disease, "cs_tercile": cs_tercile,
                    "initial_features": initial, "condition": condition,
                    "info_order": info_order,
                })
    return jobs


def run_cohort(
    *,
    cfg: CohortRunConfig,
    train_dict: dict,
    test_dict: dict,
    simulator: FeatureSimulator,
    knn: PatientKNN,
) -> list[TrajectoryRecord]:
    jobs = _build_jobs(cfg, test_dict)
    if cfg.verbose:
        print(f"  scheduled {len(jobs)} trajectories  "
              f"(parallel_workers={cfg.parallel_workers})", flush=True)

    def _run_one(job) -> TrajectoryRecord:
        return run_one_trajectory(
            patient_id=job["patient_id"], disease=job["disease"],
            cs_tercile=job["cs_tercile"], initial_features=job["initial_features"],
            condition=job["condition"], simulator=simulator, knn=knn,
            info_order=job["info_order"], max_steps=cfg.max_steps,
            model=cfg.model, verbose=False,
        )

    out: list[TrajectoryRecord] = []
    done = 0
    if cfg.parallel_workers <= 1:
        for job in jobs:
            if cfg.verbose:
                print(f"  {job['disease']}/{job['patient_id']} cond={job['condition']} order={job['info_order']}",
                      flush=True)
            out.append(_run_one(job))
            done += 1
            if cfg.verbose and done % 20 == 0:
                print(f"  ... {done}/{len(jobs)} done", flush=True)
    else:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=cfg.parallel_workers) as ex:
            futures = {ex.submit(_run_one, job): job for job in jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    rec = fut.result()
                except Exception as exc:
                    print(f"  [FAIL] {job['disease']}/{job['patient_id']} "
                          f"cond={job['condition']}: {exc}", flush=True)
                    continue
                out.append(rec)
                done += 1
                if cfg.verbose and done % 20 == 0:
                    elapsed = time.time() - t0
                    print(f"  ... {done}/{len(jobs)} done  ({elapsed:.0f}s elapsed)",
                          flush=True)
        if cfg.verbose:
            print(f"  total time: {time.time() - t0:.0f}s", flush=True)

    # Deterministic ordering for reproducible CSVs
    out.sort(key=lambda r: (r.disease, str(r.patient_id), r.condition, r.info_order))
    if cfg.verbose:
        print(f"  total trajectories: {len(out)}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def trajectories_to_step_df(trajs: Iterable[TrajectoryRecord]) -> pd.DataFrame:
    rows = []
    for t in trajs:
        for s in t.steps:
            rows.append({
                "patient_id": s.patient_id,
                "disease": s.disease,
                "cs_tercile": s.cs_tercile,
                "condition": s.condition,
                "info_order": s.info_order,
                "step_index": s.step_index,
                "tests_done_before": ", ".join(s.tests_done_before),
                "rubric_next_test": s.rubric_next_test or "",
                "knn_top1_disease": s.knn_top1_disease or "",
                "llm_next_test": s.llm_next_test,
                "llm_reasoning": s.llm_reasoning,
                "terminated": s.terminated,
                "termination_reason": s.termination_reason,
                "prompt_tokens": s.prompt_tokens,
                "completion_tokens": s.completion_tokens,
                "elapsed_s": round(s.elapsed_s, 3),
                "follows_rubric": s.follows_rubric,
                "deviation_reason": s.deviation_reason,
            })
    return pd.DataFrame(rows)


def trajectories_to_seq_df(trajs: Iterable[TrajectoryRecord]) -> pd.DataFrame:
    rows = []
    for t in trajs:
        rows.append({
            "patient_id": t.patient_id,
            "disease": t.disease,
            "cs_tercile": t.cs_tercile,
            "condition": t.condition,
            "info_order": t.info_order,
            "llm_sequence": ", ".join(t.llm_sequence),
            "n_steps": t.n_steps,
            "termination_reason": t.termination_reason,
        })
    return pd.DataFrame(rows)
