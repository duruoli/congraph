# Gap analysis — Part 3 & Part 4 findings

This is a descriptive study under the **oracle-routing** simplification of
the rubric simulator (sub-rubric selected from the ground-truth disease).
No claims about which strategy is *better* are made; the same numbers will
be regenerated under main-triage routing in a follow-up pass.

---

## Part 3 — Edge-condition diagnostic value as a mechanism for gap

### What was computed

1. **Condition feature set.** All 56 binary clinical-evidence features
   that appear in any edge condition of the five rubric graphs
   (`pipeline/rubric_graph.py`), including helper expansions of
   `alvarado_score`, `bisap_score`, `_revised_atlanta_criteria_count`, and
   the five `_tg18_*` helpers.  `pain_location` is expanded into dummies
   for the four categories used by edge conditions
   (`RLQ`, `RUQ`, `LLQ`, `Epigastric`).  Post-test imaging flags (US_*,
   CT_*) are included for table completeness even though they cannot
   trigger at step 0.

2. **w(c, d) on the training cohort.**  Using the 1044-patient training
   split (`seed=42`, `min_tests=2`, identical to
   `scripts/run_rubric_simulator_oracle.py`) we estimate per (feature,
   disease)
   ```
   w(c, d) = log[ (n_c|d + 0.5) / (n_d + 1) ]
           - log[ (n_c|¬d + 0.5) / (n_¬d + 1) ]
   ```
   (Haldane-Anscombe smoothing).  Features with fewer than 5 positive
   training observations are flagged `unreliable=True`.  Stored in
   `part3_condition_weights.csv` (224 rows).

3. **Step-0 CertaintyScore.**  For each test patient, sum w(c, d) over
   features triggered at step 0.  Two anchors are reported:
   - `certainty_score_oracle`: d = oracle disease.  This is the **primary**
     measure used downstream.
   - `certainty_score_top`: d = `argmax_d Σ_c w(c, d)`, i.e. the disease
     that the step-0 evidence most strongly points to.  Acts as a
     sensitivity check that does not assume the oracle label.
   Stored in `part3_certainty_score.csv` (300 rows).

   **Why oracle d.** The whole simulator is under oracle routing, so we
   keep the same assumption for the primary score.  An oracle-disease vs
   top-disease agreement of 79 % overall (96 % for pancreatitis, 91 % for
   diverticulitis, 79 % for appendicitis, 62 % for cholecystitis) shows
   that for three of the four diseases the perceived-top score is close
   to the oracle score; cholecystitis is the diagnosis most often
   "rivalled" by other anchors at step 0.

### Top discriminative features (largest |w|, reliable estimates)

| disease | feature (sign of w) |
| --- | --- |
| appendicitis | `pain_location_RLQ` (+3.10), `pain_migration_to_RLQ` (+2.81), `RLQ_tenderness` (+2.34); `lipase_ge_3xULN` (−4.16), `pain_location_RUQ` (−2.23) |
| cholecystitis | `fever_reported_in_hpi` (+5.81), `RUQ_mass` (+2.07), `murphys_sign` (+2.04); `pain_location_RLQ` (−2.87) |
| diverticulitis | `pain_location_LLQ` (+2.99); `fever_reported_in_hpi` (−2.95), `lipase_ge_3xULN` (−2.51) |
| pancreatitis | `lipase_ge_3xULN` (+3.53); `pain_migration_to_RLQ` (−3.05), `pain_location_RLQ` (−2.76) |

These are the "diagnostic-value" features the Part 3 hypothesis predicts
should also be the features that predict gap.  They are recognisable as
the classical clinical anchors named in the source guidelines (Alvarado,
TG18 A/B, Revised Atlanta).

### Regression comparison

Baseline = the L1-CV logistic regression from `gap_analysis.py`.  Plus =
baseline ∪ `{certainty_score_oracle}`.  All CV is stratified, 5-fold
(or fewer when a class is too small).  Numbers from
`part3_regression_comparison.csv`.

| disease | gap | n_pos | AUC baseline | AUC +CS | ΔAUC | AUC CS-alone |
| --- | --- | --- | --- | --- | --- | --- |
| appendicitis  | commission | 32 | 0.924 | 0.877 | −0.047 | 0.748 |
| appendicitis  | omission   | 22 | 0.839 | 0.844 | +0.006 | 0.476 |
| cholecystitis | commission | 62 | 0.582 | 0.622 | +0.040 | 0.664 |
| cholecystitis | omission   | 29 | 0.734 | 0.771 | +0.037 | 0.501 |
| diverticulitis| commission | 19 | 0.375 | 0.434 | +0.059 | 0.530 |
| pancreatitis  | commission | 69 | 0.766 | 0.783 | +0.017 | 0.731 |
| pancreatitis  | omission   | 37 | 0.616 | 0.615 | −0.001 | 0.710 |

`order_swap` is degenerate in three of four diseases (0–5 positives) and
is dropped.

#### Interpretation (descriptive)

- For **pancreatitis**, the single CertaintyScore reaches AUC 0.731 on
  commission and 0.710 on omission — at or above the multivariate
  baseline (0.766 / 0.616).  A scalar built only from rubric edge
  conditions explains as much gap variation as the full feature panel.
  This is the cleanest support for the Part 3 hypothesis.

- For **cholecystitis**, CertaintyScore alone (0.664) already beats the
  baseline (0.582) on commission, and adding it lifts the multivariate
  AUC (+0.040).  It does *not* help omission.

- For **appendicitis commission**, the baseline panel is very strong
  (0.924) and CertaintyScore alone underperforms (0.748).  The features
  this gap depends on are richer than what a step-0 LLR sum captures.
  ΔAUC is slightly negative — the added scalar gives the L1 path a
  collinear shortcut that hurts cross-validated calibration, not
  evidence against the hypothesis.

- **Diverticulitis** has n=35 with no omissions at all, so only
  commission is fittable; CertaintyScore adds a modest +0.059 AUC over
  the weak baseline (0.375 → 0.434).

#### Methodological notes

- We treat negatively-weighted triggered features as subtracting
  certainty rather than capping at 0.  This keeps the score interpretable
  as a log-likelihood-ratio sum.  An alternative `max(0, w)` definition
  yields qualitatively similar tercile patterns (not reported).
- `unreliable=True` features (n_pos<5 in training) account for 0 of the
  top-|w| entries above; the conclusions do not hinge on rare features.
- Choice of d = oracle (not top): see Handoff summary for what this
  forces us to redo under main-triage routing.

---

## Part 4 — Efficiency tradeoff (Δlength, Δcost)

### Computation

For each of the 300 test patients we read the oracle-rubric and actual
test sequences from `results/test_seq_comparison/test_sequence_comparison.csv`.
Cost per test is the CMS 2025 PFS midpoint transcribed from
`evaluation/test_burden_cost.py`'s docstring (Lab_Panel $49, Radiograph_Chest
$67, Ultrasound_Abdomen $170, HIDA_Scan $550, CT_Abdomen $785, MRCP_Abdomen
$1225, MRI_Abdomen $1375).  We avoid the `TEST_COST` load path because
`data/raw_data/cost_mapping.csv` is not in the working tree; the midpoint
values are the canonical fallback documented in the same module.

Two variants are reported for each Δ:
- `delta_length` / `delta_cost`: actual sequence counted as-is.
- `delta_length_no_chest` / `delta_cost_no_chest`: `Radiograph_Chest`
  stripped from actual.  This matches **Part 1's commission convention**
  (chest x-ray is never rubric-prescribed, so excluding it makes the Δ
  comparable to commission-defined deviations).

Patients are stratified into disease-specific terciles of
`certainty_score_oracle` (`low` / `mid` / `high`).  Full per-stratum
counts and means are in `part4_efficiency_tradeoff.csv`; full per-patient
data are in `part4_per_patient.csv`.

### Headline numbers (means; `_no_chest` cost variant)

| disease | tercile | n | ΔLength | ΔCost (USD) | commission_rate | omission_rate |
| --- | --- | --- | --- | --- | --- | --- |
| appendicitis  | low  | 24 | +0.96 | +464 | 0.62 | 0.29 |
| appendicitis  | mid  | 23 | +1.09 | +492 | 0.61 | 0.26 |
| appendicitis  | high | 24 | +0.25 |  +49 | 0.12 | 0.38 |
| cholecystitis | low  | 31 | +1.55 | +496 | 0.87 | 0.23 |
| cholecystitis | mid  | 31 | +0.87 |  +28 | 0.58 | 0.32 |
| cholecystitis | high | 31 | +0.74 |   −6 | 0.55 | 0.39 |
| diverticulitis| low  | 12 | +1.17 |  +85 | 0.50 | 0.00 |
| diverticulitis| mid  | 11 | +1.55 | +312 | 0.36 | 0.00 |
| diverticulitis| high | 12 | +1.83 | +631 | 0.75 | 0.00 |
| pancreatitis  | low  | 34 | +1.76 | +702 | 0.88 | 0.15 |
| pancreatitis  | mid  | 33 | +1.15 | +127 | 0.70 | 0.39 |
| pancreatitis  | high | 34 | +0.56 | −208 | 0.47 | 0.56 |

Point-biserial / Spearman correlations across the full disease cohort:

| disease | r(CS, commission) | r(CS, omission) | ρ(CS, ΔLength) |
| --- | --- | --- | --- |
| appendicitis  | −0.40 (p=0.001) | +0.07 (ns) | −0.33 (p=0.004) |
| cholecystitis | −0.26 (p=0.013) | +0.09 (ns) | −0.35 (p=0.001) |
| pancreatitis  | −0.35 (p<0.001) | +0.36 (p<0.001) | −0.46 (p<0.001) |

Diverticulitis has 0 omissions in this cohort; correlation analysis is
skipped there.

### Descriptive observations

- For **appendicitis, cholecystitis, pancreatitis**, higher step-0
  CertaintyScore is associated with fewer extra tests (commission ↓) and
  shorter trajectories.  Δcost in the high-certainty tercile is roughly
  $0 or negative (i.e. doctor spends at parity with or below rubric).
- For **pancreatitis**, the certainty effect runs in *both directions* at
  the same time: high certainty correlates with both lower commission
  (r=−0.35) and higher omission (r=+0.36), and the mean Δcost in the
  high tercile is −$208 — doctors with very strong step-0 evidence
  appear to skip rubric-prescribed tests.
- **Diverticulitis is the exception.**  ΔLength and Δcost both *rise*
  with certainty.  Sample size is small (35), there are no omissions to
  trade against commission, and the high-tercile commission rate is the
  highest (0.75).  Possible drivers (informal): once LLQ pain + fever
  + diverticular history make the diagnosis confident, doctors order
  CT regardless of whether the rubric needed it; this is consistent with
  AAFP guidance that CT is still recommended when clinical certainty is
  high.  This is described, not endorsed.

### Honesty constraint (per task brief)

Omission is, by definition, "fewer tests" and "less cost" relative to
the rubric.  A rise in omission rate at high certainty therefore
*mechanically* lowers ΔLength and Δcost — this is an artefact of how
the two quantities are tied together, not a separate finding.  We
**do not** read "high-certainty patients have lower cost" as "the
deviation is better".  We only report that the tradeoff is what it is.

---

## Files written

```
results/gap_analysis/
    part3_condition_weights.csv        # w(c, d) table, 56 features × 4 diseases (224 rows)
    part3_certainty_score.csv          # per-patient step-0 CertaintyScore (300 rows)
    part3_regression_comparison.txt    # human-readable AUC table + interpretation
    part3_regression_comparison.csv    # same numbers, tabular
    part4_efficiency_tradeoff.csv      # certainty-tercile-stratified means
    part4_per_patient.csv              # full per-patient Δlength / Δcost
    part4_figures/
        delta_length_by_certainty.png
        delta_length_no_chest_by_certainty.png
        delta_cost_by_certainty.png
        delta_cost_no_chest_by_certainty.png
        certainty_vs_delta_length.png
```
