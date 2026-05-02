"""real_pipeline.py — Pipeline on real patient data from results/

Loads the four disease feature JSONs under results/ and runs every patient
through the full pipeline (Traversal → Distribution → Recommender) step by step.

Modes
-----
  python real_pipeline.py                          # accuracy sweep, all diseases
  python real_pipeline.py --disease appendicitis   # sweep one disease
  python real_pipeline.py --stats                  # dataset statistics only
  python real_pipeline.py --demo appendicitis                   # first patient
  python real_pipeline.py --demo appendicitis 20890008          # specific patient
  python real_pipeline.py --demo cholecystitis --all-steps      # every step verbosely
  python real_pipeline.py --test                   # mixed-dataset test pipeline
  python real_pipeline.py --test --n-test 50 --seed 7 --k 20   # tune parameters
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from collections import defaultdict

DISEASES_LIST = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]


def _entropy(probs: dict[str, float]) -> float:
    """Shannon entropy (bits) of a probability distribution."""
    return -sum(p * math.log2(p) for p in probs.values() if p > 0)

# ── locate results/ relative to this file ────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
DISEASE_FILES: dict[str, Path] = {
    "appendicitis":  RESULTS_DIR / "appendicitis_features.json",
    "cholecystitis": RESULTS_DIR / "cholecystitis_features.json",
    "diverticulitis": RESULTS_DIR / "diverticulitis_features.json",
    "pancreatitis":  RESULTS_DIR / "pancreatitis_features.json",
}

# ── pipeline imports ──────────────────────────────────────────────────────────
from clinical_session import ClinicalSession
from empirical_scorer import EmpiricalScorer
import diagnosis_distribution as _dd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_disease(disease: str) -> dict[str, list[dict]]:
    """Return {patient_id: [step, ...]} for one disease."""
    path = DISEASE_FILES[disease]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["results"]


def load_all() -> dict[str, dict[str, list[dict]]]:
    """Return {disease: {patient_id: [step, ...]}}."""
    return {d: load_disease(d) for d in DISEASE_FILES}


# ---------------------------------------------------------------------------
# Single-patient runner
# ---------------------------------------------------------------------------

def run_patient(steps: list[dict]) -> list[tuple[dict, object]]:
    """
    Run one patient through the full pipeline, step by step.

    Each step already carries a *cumulative* feature dict (all evidence so far),
    so we create a fresh ClinicalSession per step using that snapshot directly.

    Returns list of (step_meta, AssessmentState) pairs.
    """
    results = []
    for step in steps:
        features = step["features"]
        session = ClinicalSession(features)
        state = session.assess()
        results.append((step, state))
    return results


def _rec_hits(step_results: list[tuple[dict, object]], steps: list[dict]
              ) -> tuple[int, int, int]:
    """
    Compare recommendations from step i against the actual test at step i+1.

    Returns (top1_correct, hit3_correct, total) where total counts only
    intermediate steps that have a non-null next test_key.
    """
    top1 = hit3 = total = 0
    for i in range(len(step_results) - 1):
        actual_next = steps[i + 1].get("test_key")
        if not actual_next:
            continue
        _, state = step_results[i]
        recs = [r.test for r in state.recommendations]
        total += 1
        if recs and recs[0] == actual_next:
            top1 += 1
        if actual_next in recs[:3]:
            hit3 += 1
    return top1, hit3, total


# ---------------------------------------------------------------------------
# Accuracy sweep
# ---------------------------------------------------------------------------

def run_sweep(diseases: list[str]) -> None:
    """Run all patients for the given diseases, print per-disease accuracy."""

    grand_total = grand_correct = 0
    disease_stats: dict[str, dict] = {}

    for disease in diseases:
        patients = load_disease(disease)

        # Per-patient: track first-step and final-step correctness
        first_correct = first_total = 0
        final_correct = final_total = 0

        # Accuracy by number of steps in the patient record (complexity bucket)
        bucket_correct: dict[str, int] = defaultdict(int)
        bucket_total:   dict[str, int] = defaultdict(int)

        rec_top1 = rec_hit3 = rec_total = 0

        # first-step predicted primary for each patient (for bias matrix)
        first_predicted: list[str] = []
        first_entropy_sum = 0.0

        for pid, steps in patients.items():
            step_results = run_patient(steps)
            n_steps = len(steps)
            bucket = f"{n_steps} step{'s' if n_steps != 1 else ''}"

            # First step
            _, first_state = step_results[0]
            first_total += 1
            if first_state.primary_diagnosis == disease:
                first_correct += 1
            first_predicted.append(first_state.primary_diagnosis)
            first_entropy_sum += _entropy(first_state.distribution.probabilities)

            # Final step
            _, final_state = step_results[-1]
            final_total += 1
            if final_state.primary_diagnosis == disease:
                final_correct += 1
                bucket_correct[bucket] += 1
            bucket_total[bucket] += 1

            # Recommendation accuracy (intermediate steps only)
            t1, h3, tot = _rec_hits(step_results, steps)
            rec_top1  += t1
            rec_hit3  += h3
            rec_total += tot

        # first-step predicted distribution (how often each dx was predicted)
        first_pred_counts: dict[str, int] = defaultdict(int)
        for p in first_predicted:
            first_pred_counts[p] += 1

        disease_stats[disease] = {
            "first_correct": first_correct, "first_total": first_total,
            "final_correct": final_correct, "final_total": final_total,
            "bucket_correct": dict(bucket_correct),
            "bucket_total":   dict(bucket_total),
            "rec_top1": rec_top1, "rec_hit3": rec_hit3, "rec_total": rec_total,
            "first_pred_counts": dict(first_pred_counts),
            "first_entropy_mean": first_entropy_sum / first_total if first_total else 0.0,
        }
        grand_total   += final_total
        grand_correct += final_correct

    # ── Print ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print(f"  REAL PATIENT PIPELINE — ACCURACY SWEEP")
    print(f"{'═'*68}")

    grand_rec_top1 = grand_rec_hit3 = grand_rec_total = 0

    for disease, st in disease_stats.items():
        f_acc   = st["first_correct"] / st["first_total"] if st["first_total"] else 0
        fin_acc = st["final_correct"] / st["final_total"] if st["final_total"] else 0
        rec_t1  = st["rec_top1"] / st["rec_total"] if st["rec_total"] else 0
        rec_h3  = st["rec_hit3"] / st["rec_total"] if st["rec_total"] else 0
        print(f"\n  {disease.upper()}  ({st['final_total']} patients)")
        print(f"  {'─'*60}")
        print(f"  First-step accuracy (HPI+PE+labs) : "
              f"{st['first_correct']}/{st['first_total']}  ({f_acc:.1%})"
              f"  entropy={st['first_entropy_mean']:.3f} bits")
        print(f"  Final-step accuracy (all evidence) : "
              f"{st['final_correct']}/{st['final_total']}  ({fin_acc:.1%})")
        print(f"  Next-test top-1 accuracy           : "
              f"{st['rec_top1']}/{st['rec_total']}  ({rec_t1:.1%})")
        print(f"  Next-test hit@3 accuracy           : "
              f"{st['rec_hit3']}/{st['rec_total']}  ({rec_h3:.1%})")

        # First-step predicted distribution (bias check)
        n = st["first_total"]
        pred_parts = "  ".join(
            f"{d[:4]}={st['first_pred_counts'].get(d, 0):3d}({st['first_pred_counts'].get(d, 0)/n:.0%})"
            for d in DISEASES_LIST
        )
        print(f"  First-step predicted (bias check)  : {pred_parts}")

        # Show breakdown by patient complexity (sorted by step count)
        def _step_sort(k: str) -> int:
            return int(k.split()[0])
        for bucket in sorted(st["bucket_total"], key=_step_sort):
            bc = st["bucket_correct"].get(bucket, 0)
            bn = st["bucket_total"][bucket]
            bacc = bc / bn if bn else 0
            print(f"    [{bucket:12s}]  {bc}/{bn}  ({bacc:.1%})")

        grand_rec_top1  += st["rec_top1"]
        grand_rec_hit3  += st["rec_hit3"]
        grand_rec_total += st["rec_total"]

    overall_acc    = grand_correct / grand_total if grand_total else 0
    overall_rec_t1 = grand_rec_top1 / grand_rec_total if grand_rec_total else 0
    overall_rec_h3 = grand_rec_hit3 / grand_rec_total if grand_rec_total else 0
    print(f"\n{'─'*68}")
    print(f"  OVERALL diagnosis  {grand_correct}/{grand_total}  ({overall_acc:.1%})")
    print(f"  OVERALL next-test top-1  "
          f"{grand_rec_top1}/{grand_rec_total}  ({overall_rec_t1:.1%})")
    print(f"  OVERALL next-test hit@3  "
          f"{grand_rec_hit3}/{grand_rec_total}  ({overall_rec_h3:.1%})")
    print(f"{'═'*68}\n")


# ---------------------------------------------------------------------------
# Statistics mode
# ---------------------------------------------------------------------------

def run_stats() -> None:
    print(f"\n{'═'*64}")
    print(f"  DATASET STATISTICS")
    print(f"{'═'*64}")

    grand = 0
    for disease, path in DISEASE_FILES.items():
        patients = load_disease(disease)
        step_counts: dict[int, int] = defaultdict(int)
        for steps in patients.values():
            step_counts[len(steps)] += 1
        total = len(patients)
        grand += total
        print(f"\n  {disease.upper()}  ({total} patients)")
        print(f"  {'─'*52}")
        for n, cnt in sorted(step_counts.items()):
            print(f"    {n} step(s): {cnt} patients")

    print(f"\n  {'─'*52}")
    print(f"  TOTAL  {grand} patients")
    print(f"{'═'*64}\n")


# ---------------------------------------------------------------------------
# Flip analysis — wrong initial dx → correct final dx
# ---------------------------------------------------------------------------

def run_flips(diseases: list[str]) -> None:
    """
    Find all patients whose first-step primary diagnosis was WRONG but whose
    final-step primary diagnosis is CORRECT (a 'flip').

    For each flip case, print the full step-by-step diagnosis trajectory so we
    can see how the distribution evolved as more evidence arrived.
    """
    DISEASES_SHORT = {d: d[:4].upper() for d in DISEASES_LIST}

    print(f"\n{'═'*68}")
    print(f"  FLIP ANALYSIS  — initial wrong Dx → final correct Dx")
    print(f"{'═'*68}")

    total_patients = 0
    total_flips = 0

    for disease in diseases:
        patients = load_disease(disease)
        flip_cases: list[tuple[str, list, list]] = []

        for pid, steps in patients.items():
            total_patients += 1
            if len(steps) < 2:
                continue  # single-step patients cannot flip
            step_results = run_patient(steps)

            _, first_state = step_results[0]
            _, final_state = step_results[-1]

            first_wrong   = first_state.primary_diagnosis != disease
            final_correct = final_state.primary_diagnosis == disease

            if first_wrong and final_correct:
                flip_cases.append((pid, steps, step_results))

        total_flips += len(flip_cases)

        print(f"\n  {disease.upper()}  — {len(flip_cases)} flip(s) "
              f"out of {len(patients)} patients")
        print(f"  {'─'*64}")

        for pid, steps, step_results in flip_cases:
            print(f"\n  Patient {pid}  ({len(steps)} steps)")
            header = f"  {'Step':>4}  {'Label':<32}  {'Primary Dx':<16}  Distribution"
            print(header)
            print(f"  {'─'*len(header.rstrip())}")

            for i, (step_meta, state) in enumerate(step_results):
                label = step_meta["step_label"][:30]
                primary = state.primary_diagnosis
                is_first = (i == 0)
                is_last  = (i == len(step_results) - 1)

                # Build a compact probability bar for the four diseases
                probs = state.distribution.probabilities
                bar_parts = []
                for d in DISEASES_LIST:
                    p = probs.get(d, 0.0)
                    marker = "▶" if d == disease else " "
                    bar_parts.append(f"{marker}{DISEASES_SHORT[d]}={p:4.1%}")
                dist_str = "  ".join(bar_parts)

                flip_tag = ""
                if is_first and primary != disease:
                    flip_tag = "  ← WRONG"
                elif is_last and primary == disease:
                    flip_tag = "  ← CORRECT ✓"
                elif primary == disease:
                    flip_tag = "  ← correct"

                print(f"  {step_meta['step_index']:>4}  {label:<32}  "
                      f"{primary:<16}  {dist_str}{flip_tag}")

    print(f"\n{'─'*68}")
    print(f"  Total flip cases : {total_flips} / {total_patients} patients")
    print(f"  (initial wrong → final correct)")
    print(f"{'═'*68}\n")


# ---------------------------------------------------------------------------
# Demo / narrative mode
# ---------------------------------------------------------------------------

def run_demo(disease: str, patient_id: str | None = None,
             all_steps: bool = False) -> None:
    patients = load_disease(disease)
    if not patients:
        print(f"No patients found for {disease!r}.")
        return

    if patient_id is None:
        patient_id = next(iter(patients))
    elif patient_id not in patients:
        print(f"Patient {patient_id!r} not found in {disease}. "
              f"Available IDs (first 10): {list(patients)[:10]}")
        return

    steps = patients[patient_id]
    step_results = run_patient(steps)

    print(f"\n{'█'*64}")
    print(f"  DEMO: {disease.upper()}  |  Patient {patient_id}")
    print(f"  {len(steps)} evidence step(s)")
    print(f"{'█'*64}")

    for (step_meta, state), idx in zip(step_results, range(len(step_results))):
        is_last = (idx == len(step_results) - 1)
        if not all_steps and not is_last and len(step_results) > 1:
            # In condensed mode, only show intermediate summaries, full last step
            tests = step_meta["features"].get("tests_done", [])
            print(f"\n  ── step {step_meta['step_index']}  {step_meta['step_label']}")
            print(f"     tests: {tests}")
            print(f"     {state.summary()}")
            continue

        print(f"\n  {'─'*62}")
        print(f"  Step {step_meta['step_index']} — {step_meta['step_label']}")
        if step_meta["test_key"]:
            print(f"  New test : {step_meta['test_key']}  (note: {step_meta.get('note_id','')})")
        print(f"  {'─'*62}")
        state.print_report()

    # Final verdict
    _, final_state = step_results[-1]
    verdict = "✓ CORRECT" if final_state.primary_diagnosis == disease else "✗ WRONG"
    print(f"  Ground truth : {disease}")
    print(f"  Primary Dx   : {final_state.primary_diagnosis}")
    print(f"  Verdict      : {verdict}\n")


# ---------------------------------------------------------------------------
# Mixed-dataset test pipeline
# ---------------------------------------------------------------------------

def run_test_pipeline(
    n_test: int = 100,
    seed: int = 42,
    k_neighbors: int = 15,
    verbose: bool = True,
) -> None:
    """
    Mixed-dataset test pipeline.

    Algorithm
    ---------
    1. Load all four disease datasets and merge into one pool of
       (patient_id, disease, steps) triples.
    2. Shuffle with ``seed`` and split: first ``n_test`` → test set,
       remainder → training corpus.
    3. Build an EmpiricalScorer (KNN) from the training corpus using each
       patient's *last* step (most complete feature snapshot).
    4. For every test patient, pick one step **at random** as the pipeline
       input (simulating arriving at an arbitrary point in the workup).
    5. Run the full pipeline (Traversal → Distribution w/ empirical score
       → Recommender) and print the distribution + recommendations.
    6. Report per-disease and overall accuracy.
    """
    random.seed(seed)

    # ── 1. Build combined pool ─────────────────────────────────────────────
    all_patients: list[tuple[str, str, list[dict]]] = []
    for disease in DISEASE_FILES:
        patients = load_disease(disease)
        for pid, steps in patients.items():
            all_patients.append((pid, disease, steps))

    random.shuffle(all_patients)
    n_test = min(n_test, len(all_patients))

    test_set   = all_patients[:n_test]
    train_set  = all_patients[n_test:]

    # ── 2. Train empirical scorer on train corpus ──────────────────────────
    # Use each patient's FIRST step (HPI + Physical Exam + Basic Labs) as the
    # reference vector.  This mirrors the real clinical entry point: when a new
    # patient arrives only the initial assessment is universally available.
    train_pairs: list[tuple[dict, str]] = [
        (steps[0]["features"], disease)
        for _, disease, steps in train_set
    ]
    scorer = EmpiricalScorer(k=k_neighbors)
    scorer.fit(train_pairs)
    _dd.set_empirical_scorer(scorer)

    lbl_dist = scorer.label_distribution()
    print(f"\n{'═'*68}")
    print(f"  MIXED-DATASET TEST PIPELINE")
    print(f"{'═'*68}")
    print(f"  Total patients  : {len(all_patients)}")
    print(f"  Training corpus : {scorer.n_train}  "
          + "  ".join(f"{d}={lbl_dist[d]}" for d in lbl_dist))
    print(f"  Test set        : {n_test}  (seed={seed})")
    print(f"  KNN k           : {k_neighbors}")
    print(f"{'─'*68}")

    # ── 3. Run test ────────────────────────────────────────────────────────
    per_disease_correct: dict[str, int] = defaultdict(int)
    per_disease_total:   dict[str, int] = defaultdict(int)
    grand_correct = 0
    grand_rec_top1 = grand_rec_hit3 = grand_rec_total = 0

    for idx, (pid, disease, steps) in enumerate(test_set, 1):
        # Pick a random step from this patient's sequence
        chosen_step = random.choice(steps)
        features    = chosen_step["features"]
        step_label  = chosen_step["step_label"]

        session = ClinicalSession(features)
        state   = session.assess()

        is_correct = state.primary_diagnosis == disease
        if is_correct:
            grand_correct += 1
            per_disease_correct[disease] += 1
        per_disease_total[disease] += 1

        # Recommendation accuracy across all steps of this patient
        step_results = run_patient(steps)
        t1, h3, tot = _rec_hits(step_results, steps)
        grand_rec_top1  += t1
        grand_rec_hit3  += h3
        grand_rec_total += tot

        if verbose:
            verdict = "✓" if is_correct else "✗"
            print(f"\n  [{idx:>3d}/{n_test}]  {verdict}  "
                  f"pid={pid}  true={disease}  step='{step_label}'")
            print(f"  {'─'*62}")

            # Distribution
            print(f"  DISTRIBUTION")
            for d in state.distribution.ranked:
                p   = state.distribution.prob(d)
                r   = state.traversal.diseases[d]
                bar = "█" * int(p * 28)
                tag = ""
                if r.confirmed:
                    tag = f"  ✓ confirmed [{r.terminal_node}]"
                elif r.excluded:
                    tag = f"  ✗ excluded  [{r.terminal_node}]"
                elif r.triage_activated:
                    tag = "  ◀ triage"
                marker = "▶ " if d == disease else "  "
                print(f"  {marker}{d:16s} {p:5.1%}  {bar:<28s}{tag}")

            # Recommendations
            print(f"  NEXT TESTS RECOMMENDED")
            from recommender import format_recommendations
            rec_str = format_recommendations(state.recommendations)
            for line in rec_str.splitlines():
                print(f"  {line}")

    # ── 4. Summary ────────────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print(f"  SUMMARY  (n_test={n_test}  seed={seed}  k={k_neighbors})")
    print(f"{'─'*68}")
    for disease in DISEASE_FILES:
        tot = per_disease_total[disease]
        if tot == 0:
            continue
        cor = per_disease_correct[disease]
        print(f"  {disease:16s}  {cor}/{tot}  ({cor/tot:.1%})")
    overall = grand_correct / n_test if n_test else 0
    overall_rec_t1 = grand_rec_top1 / grand_rec_total if grand_rec_total else 0
    overall_rec_h3 = grand_rec_hit3 / grand_rec_total if grand_rec_total else 0
    print(f"{'─'*68}")
    print(f"  OVERALL diagnosis       {grand_correct}/{n_test}  ({overall:.1%})")
    print(f"  OVERALL next-test top-1 "
          f"{grand_rec_top1}/{grand_rec_total}  ({overall_rec_t1:.1%})")
    print(f"  OVERALL next-test hit@3 "
          f"{grand_rec_hit3}/{grand_rec_total}  ({overall_rec_h3:.1%})")
    print(f"{'═'*68}\n")

    # Clean up: remove empirical scorer so other modes are unaffected
    _dd.clear_empirical_scorer()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real patient pipeline.")
    p.add_argument("--disease",    default=None,
                   choices=list(DISEASE_FILES), help="Limit sweep to one disease.")
    p.add_argument("--stats",      action="store_true", help="Dataset statistics only.")
    p.add_argument("--flips",      action="store_true",
                   help="Show patients whose initial Dx was wrong but final Dx is correct.")
    p.add_argument("--demo",       nargs="?", const="__unset__", metavar="DISEASE",
                   help="Demo mode. Optionally specify disease.")
    p.add_argument("--patient",    default=None, metavar="PATIENT_ID",
                   help="Patient ID for demo mode.")
    p.add_argument("--all-steps",  action="store_true",
                   help="In demo mode, print full report for every step.")
    # ── test pipeline ──────────────────────────────────────────────────────
    p.add_argument("--test",       action="store_true",
                   help="Mixed-dataset test pipeline with KNN empirical scorer.")
    p.add_argument("--n-test",     type=int, default=100, metavar="N",
                   help="Number of test patients to sample (default: 100).")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed for test/train split and step selection (default: 42).")
    p.add_argument("--k",          type=int, default=15, metavar="K",
                   help="Number of nearest neighbors for empirical scorer (default: 15).")
    p.add_argument("--quiet",      action="store_true",
                   help="In --test mode, suppress per-patient output (summary only).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.stats:
        run_stats()
        return

    if args.flips:
        diseases = [args.disease] if args.disease else list(DISEASE_FILES)
        run_flips(diseases)
        return

    if args.demo is not None:
        disease = args.demo if args.demo != "__unset__" else "appendicitis"
        if disease not in DISEASE_FILES:
            print(f"Unknown disease {disease!r}. Choose from: {list(DISEASE_FILES)}")
            sys.exit(1)
        run_demo(disease, patient_id=args.patient, all_steps=args.all_steps)
        return

    if args.test:
        run_test_pipeline(
            n_test=args.n_test,
            seed=args.seed,
            k_neighbors=args.k,
            verbose=not args.quiet,
        )
        return

    # Sweep mode
    diseases = [args.disease] if args.disease else list(DISEASE_FILES)
    run_sweep(diseases)


if __name__ == "__main__":
    main()
