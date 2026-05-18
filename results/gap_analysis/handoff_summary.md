# Handoff to the main-triage-routing rerun

Current pass uses **oracle routing** (`oracle_routing=True` in
`scripts/run_rubric_simulator_oracle.py`), where each patient is fed
into their ground-truth disease sub-rubric.  A near-parallel rerun is
planned that replaces oracle routing with the actual main-triage graph
(`pipeline/rubric_graph.py::TRIAGE_GRAPH`).  This file flags exactly
what carries over and what has to be rebuilt.

---

## What is **independent of routing** and can be reused as-is

- **`part3_condition_weights.csv` — the w(c, d) table.**  It is built
  from training-set step-0 features and the *patient's recorded disease
  label*.  Routing never enters the estimator.  Same table can be
  loaded by the next pass without recomputation.

- **The condition-feature list** (`CONDITION_FEATURES_BINARY` +
  `pain_location` dummies in `scripts/part3_certainty.py`) is derived
  from `rubric_graph.py` source, not from any simulation output.  If
  the rubric graphs change, regenerate; otherwise reuse.

- **`part3_certainty_score.csv` — step-0 CertaintyScore per test
  patient (with d = oracle disease).**  This score is computed from
  the patient's step-0 features, which are read directly from the
  features JSONs and are *not* produced by the simulator.  As long
  as the 300-patient test split is preserved (`seed=42`,
  `n_test=300`, `min_tests=2`), the same CSV is valid.

- **Per-patient gap flags** (`has_commission`, `has_omission`,
  `has_order_swap`) in `joined_data.csv` depend on the rubric vs
  actual sequence comparison.  The rubric sequence is generated under
  oracle routing in this pass, so these labels **will change** under
  main-triage routing.  Do not blindly reuse them.

## What **will change under main-triage routing** and must be rebuilt

- **Every Part 4 number.**  `part4_per_patient.csv`,
  `part4_efficiency_tradeoff.csv`, and all four figures are computed
  against `test_sequence_comparison.csv`, whose `simulated_sequence`
  column is the oracle-routing rubric trajectory.  Under main triage
  the rubric trajectory can:
    1. enter a different sub-rubric than the patient's true disease,
    2. terminate earlier (e.g. EXTENDED_DIFFERENTIAL) or later (extra
       triage labs / a longer dispatch path), and
    3. fail to enter any sub-rubric for atypical presentations.
  Each of these alters `rubric_length`, `rubric_cost`, and therefore
  every Δ.  **Regenerate Part 4 end to end.**

- **`part3_regression_comparison.txt`/`.csv`.**  The gap labels
  themselves change under main-triage routing (see above), so the
  baseline AUCs from `gap_analysis.py` change too.  CertaintyScore
  values do not change, but the regression refit will look different.
  **Rerun the regression after Part 4 gap labels are refreshed.**

## Methodology decisions that must be held constant between the two passes

For the rerun to be a clean apples-to-apples comparison, keep the
following identical to this pass:

| Decision | Value used in this pass |
| --- | --- |
| Train/test split | `seed=42`, `n_test=300`, `min_tests=2` (filter applied to all four feature JSONs before shuffle) |
| CertaintyScore anchor disease `d` | **oracle disease** (primary) + perceived-top diagnosis (sensitivity) |
| Feature set in `w(c,d)` | `scripts/part3_certainty.py::CONDITION_FEATURES_BINARY` + 4 `pain_location_*` dummies |
| Smoothing for w(c,d) | Haldane-Anscombe α = 0.5; reliability cutoff `n_pos_total < 5` flags `unreliable=True` |
| Cost mapping | CMS 2025 PFS midpoints hard-coded in `scripts/part4_efficiency.py::TEST_COST_USD` (Lab_Panel 49, Radiograph_Chest 67, Ultrasound_Abdomen 170, HIDA_Scan 550, CT_Abdomen 785, MRCP_Abdomen 1225, MRI_Abdomen 1375) |
| Radiograph_Chest handling | Excluded from `actual` in the `_no_chest` variant, matching Part 1's `EXCLUDE_FROM_COMMISSION` convention.  Both variants are reported. |
| Certainty stratification | Disease-specific terciles of `certainty_score_oracle` via `pd.qcut(q=3)` |
| Regression CV | L1 logistic, `Cs=10`, stratified `min(5, n_pos, n_neg)`-fold, `random_state=42` |

## Open questions that the rerun should resolve (not in scope here)

- **Routing-level certainty.**  In the oracle pass, "certainty" is the
  patient's evidence for their *true* disease.  Under main triage,
  the more honest measure may be `argmax_d Σ_c w(c, d)` — i.e. the
  triage-perceived diagnosis.  The `certainty_score_top` column in
  `part3_certainty_score.csv` is precomputed for this purpose;
  consider using it as the **primary** stratifier in the rerun.
- **Sub-rubric mismatches.**  When triage routes a patient into a
  different disease sub-rubric than their ground truth, the rubric
  trajectory may compare against the *wrong* template.  Decide before
  the rerun whether such cases are kept (and counted as commissions /
  omissions against the wrongly-routed rubric) or filtered.
- **Diverticulitis anomaly.**  Part 4 shows ΔCost *rising* with
  certainty for diverticulitis (opposite to the other three diseases).
  n=35 is small and there are 0 omissions, so the contrast cannot
  trade off.  Worth checking whether main-triage routing produces any
  omissions for diverticulitis, which would let us see if the pattern
  flips.
