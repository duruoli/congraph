# LLM-experiment handoff to the main-triage rerun

Companion to `results/gap_analysis/handoff_summary.md`.  This file lists
what carries over and what must be recomputed when the rubric pass is
re-run under main-triage routing (instead of the current
`oracle_routing=True` baseline).

## What carries over as-is

- **30-patient sample** (`sample_30` in
  `experiments/llm_experiment/sampling.py`) is derived from
  `part3_certainty_score.csv` + `test_sequence_comparison.csv` and the
  `seed=42, n_test=300, min_tests=2` split.  The split anchor is the
  same as the oracle pass; if the rerun keeps it, the sample is
  reproducible.
- **The LLM prompts** themselves (patient features + sub-rubric graph +
  KNN neighbours) are independent of triage routing.  No rerun of the
  LLM call is needed *for inputs* — only the comparator changes.
- **`order_sensitivity_check*.csv`** stand on their own (they only
  measure prompt-order stability, not actual vs predicted accuracy).

## What must be recomputed

- **`per_patient_gap.csv`, `gap_comparison_table.csv`,
  `certainty_stratified.csv`.**  These compare LLM sequences against
  `actual_sequence` and against the `rubric_only` baseline pulled from
  `test_sequence_comparison.csv::simulated_sequence`.  Under
  main-triage routing the rubric_only column changes (different routes,
  different termination) so the `rubric_only` rows in
  `gap_comparison_table.csv` must be recomputed.  LLM rows do not change
  unless the prompt's `rubric_next_test` differs — which it will if the
  sub-rubric used for `llm_full` is no longer the oracle one (see
  next point).
- **`llm_full` if the rerun decides sub-rubric by triage-perceived
  diagnosis.**  Current pass uses oracle disease for the sub-rubric
  block; switching to `perceived_top_diagnosis` (the open question
  flagged in the gap-analysis handoff) requires re-running the LLM
  calls because the prompt content changes.  Scale: 300 patients ×
  1 condition (llm_full) × ~2-3 calls/patient ≈ 600-900 calls,
  ~$2.5 at GPT-4o pricing.
- **`certainty_stratified.csv` interpretation.**  If the rerun adopts
  `certainty_score_top` as the primary stratifier (also flagged in the
  gap-analysis handoff), regenerate the tercile labels before running
  `analyze_llm_experiment.py`.  Tercile assignments will shift for
  patients where `oracle_eq_top=0`.

## Methodology decisions held constant

| Decision | Value |
| --- | --- |
| Train/test split | `seed=42`, `n_test=300`, `min_tests=2` (matches oracle pass) |
| Sample size | 300 patients (entire test split; pilot 30-patient stratified sample preserved under `*_pilot.*`) |
| Parallelism | `ThreadPoolExecutor(max_workers=10)`, 600 trajectories in 206 s wall-clock |
| LLM | GPT-4o, `temperature=0`, JSON mode, `max_tokens=400` |
| Max steps | 7 per patient (`STOP` is allowed at any step) |
| FeatureSimulator | `k=15`, fitted on train split only (same params as `run_rubric_simulator_oracle.py`) |
| KNN neighbour count | 5, distance = L2 on encoded step-k features against train pool |
| Sub-rubric serialization | full nodes + edges with `edge.label` as condition text |
| Lab_Panel handling | Prepended to LLM sequences before gap comparison (matches `scripts/extract_test_sequences.py`) |
| Radiograph_Chest commission convention | Excluded from commission set (matches `scripts/gap_analysis.py::EXCLUDE_FROM_COMMISSION`) |
| info_order in main run | `rubric_first` |

## Open items

- Re-decide whether `llm_full` should keep oracle sub-rubric or switch
  to triage-perceived sub-rubric in the rerun.
- Re-decide CS stratifier (`certainty_score_oracle` vs
  `certainty_score_top`) — both columns are already in
  `part3_certainty_score.csv`.
- Main run already at n=300; further extension would require a new
  test-split pool (e.g. drop the `min_tests=2` filter) — not in scope
  for the supplementary.
