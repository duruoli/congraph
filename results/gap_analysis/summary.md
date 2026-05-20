# Gap analysis — per disease

**Convention** (doctor-side):

- `commission` = `actual \ sim` − `{Radiograph_Chest}` (doctor ordered extra tests beyond rubric)

- `omission` = `sim \ actual` (doctor skipped rubric-recommended tests)

- `order_swap` = relative order of shared tests differs


## Gap rates


| disease | n | commission_rate | commission_n | omission_rate | omission_n | order_swap_rate | order_swap_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| appendicitis | 71 | 0.451 | 32 | 0.310 | 22 | 0.070 | 5 |
| cholecystitis | 93 | 0.667 | 62 | 0.312 | 29 | 0.011 | 1 |
| diverticulitis | 35 | 0.543 | 19 | 0.000 | 0 | 0.000 | 0 |
| pancreatitis | 101 | 0.683 | 69 | 0.366 | 37 | 0.030 | 3 |


## Model AUCs


| disease | gap | n | n_pos | base_rate | l1_cv_auc | l1_n_nonzero | rf_cv_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| appendicitis | commission | 71 | 32 | 0.451 | 0.924 | 18.000 | 0.904 |
| appendicitis | omission | 71 | 22 | 0.310 | 0.839 | 22.000 | 0.854 |
| appendicitis | order_swap | 71 | 5 | 0.070 | 0.500 | 17.000 | 0.670 |
| cholecystitis | commission | 93 | 62 | 0.667 | 0.582 | 27.000 | 0.612 |
| cholecystitis | omission | 93 | 29 | 0.312 | 0.734 | 19.000 | 0.814 |
| cholecystitis | order_swap | 93 | 1 | 0.011 |  |  |  |
| diverticulitis | commission | 35 | 19 | 0.543 | 0.375 | 1.000 | 0.549 |
| diverticulitis | omission | 35 | 0 | 0.000 |  |  |  |
| diverticulitis | order_swap | 35 | 0 | 0.000 |  |  |  |
| pancreatitis | commission | 101 | 69 | 0.683 | 0.766 | 24.000 | 0.781 |
| pancreatitis | omission | 101 | 37 | 0.366 | 0.616 | 30.000 | 0.658 |
| pancreatitis | order_swap | 101 | 3 | 0.030 | 0.500 | 10.000 | 0.762 |


## Top features (BH q ≤ 0.20) by Fisher exact, per (disease, gap)


### appendicitis — commission

| feature | rate_when_gap | rate_when_no_gap | odds_ratio | p_value | bh_q |
| --- | --- | --- | --- | --- | --- |
| WBC_gt_10k | 0.406 | 0.949 | 0.037 | 0.000 | 0.000 |
| RLQ_tenderness | 0.406 | 0.795 | 0.177 | 0.001 | 0.021 |
| anorexia | 0.062 | 0.308 | 0.150 | 0.015 | 0.163 |



### appendicitis — omission

| feature | rate_when_gap | rate_when_no_gap | odds_ratio | p_value | bh_q |
| --- | --- | --- | --- | --- | --- |
| WBC_gt_10k | 1.000 | 0.571 | inf | 0.000 | 0.004 |



### cholecystitis — omission

| feature | rate_when_gap | rate_when_no_gap | odds_ratio | p_value | bh_q |
| --- | --- | --- | --- | --- | --- |
| LFTs_elevated | 0.931 | 0.422 | 18.500 | 0.000 | 0.000 |
| RUQ_tenderness | 0.759 | 0.484 | 3.346 | 0.014 | 0.186 |
| anorexia | 0.000 | 0.172 | 0.000 | 0.015 | 0.186 |



### pancreatitis — commission

| feature | rate_when_gap | rate_when_no_gap | odds_ratio | p_value | bh_q |
| --- | --- | --- | --- | --- | --- |
| RUQ_tenderness | 0.261 | 0.031 | 10.941 | 0.005 | 0.120 |
| pain_location_Other | 0.174 | 0.000 | inf | 0.009 | 0.120 |
| lipase_ge_3xULN | 0.710 | 0.938 | 0.163 | 0.010 | 0.120 |
| pain_location_Epigastric | 0.275 | 0.531 | 0.335 | 0.015 | 0.141 |
| fever_temp_ge_37_3 | 0.000 | 0.094 | 0.000 | 0.030 | 0.192 |
| LFTs_elevated | 0.667 | 0.875 | 0.286 | 0.031 | 0.192 |



### pancreatitis — omission

| feature | rate_when_gap | rate_when_no_gap | odds_ratio | p_value | bh_q |
| --- | --- | --- | --- | --- | --- |
| lipase_ge_3xULN | 0.973 | 0.672 | 17.581 | 0.000 | 0.011 |


---

## Multivariate model results and cross-validation

L1 logistic regression (CV) coefficients are in `models/{disease}_{gap}_l1_coef.csv`; Random Forest importances are in `models/{disease}_{gap}_rf_importance.csv`. This section summarises direction and convergence with the univariate findings above.

**Reading guide.** Negative L1 coefficient = feature *reduces* the gap type; positive = feature *promotes* it. A feature that appears with a consistent sign across all three methods (Fisher OR direction, L1 sign, RF importance) is considered robustly supported.

---

### Appendicitis — commission  (L1 AUC 0.924 / RF 0.904 — strongest model in the study)

**Univariate and multivariate convergence** (all three methods agree):

| feature | Fisher OR | L1 coef | RF rank | interpretation |
| --- | --- | --- | --- | --- |
| WBC_gt_10k | 0.04 (↓) | −9.9 | 1st | present → fewer commissions |
| RLQ_tenderness | 0.18 (↓) | −8.8 | 2nd | present → fewer commissions |
| anorexia | 0.15 (↓) | −9.5 | 3rd | present → fewer commissions |

**Multivariate-only findings** (not univariate-significant at BH q ≤ 0.20, but nonzero in L1 and top-10 RF):

| feature | L1 coef | direction |
| --- | --- | --- |
| rebound_tenderness | −5.9 | fewer commissions |
| pain_migration_to_RLQ | −5.4 | fewer commissions |
| fever_temp_ge_37_3 | −5.3 | fewer commissions |
| RUQ_tenderness | +4.7 | more commissions |
| symptom_duration_over_72h | +3.9 | more commissions |

**Pattern.** The six negative-coefficient features are all classical appendicitis signs (high diagnostic specificity); when present doctors stay close to the rubric. The two positive features (`RUQ_tenderness`, prolonged symptoms) represent atypical or complicated presentations where clinicians add extra tests.

---

### Appendicitis — omission  (L1 AUC 0.839 / RF 0.854)

**Univariate and multivariate convergence:**

| feature | Fisher OR | L1 coef | RF rank | interpretation |
| --- | --- | --- | --- | --- |
| WBC_gt_10k | inf (↑) | +13.0 | 1st | present → more omissions |

**Multivariate-only findings:**

| feature | L1 coef | direction |
| --- | --- | --- |
| fever_temp_ge_37_3 | +8.2 | more omissions |
| RLQ_tenderness | +7.3 | more omissions |

**Pattern.** The same high-certainty features that *reduce* commission also *increase* omission — WBC, fever, RLQ tenderness flip sign between the two gap types. This is the clearest evidence of a certainty-driven tradeoff: strong initial evidence leads doctors to skip rubric-prescribed tests.

---

### Cholecystitis — commission  (L1 AUC 0.582 / RF 0.612 — weak)

No features survived BH q ≤ 0.20 in univariate analysis. L1 retains only a small set with modest coefficients:

| feature | L1 coef | RF rank |
| --- | --- | --- |
| RLQ_tenderness | +2.4 | — |
| WBC_gt_18k | +2.1 | 7th |
| RUQ_tenderness | −1.9 | 1st |

**Pattern.** Canonical cholecystitis sign (`RUQ_tenderness`) negatively associated with commission; atypical location (`RLQ_tenderness`) positively associated. The near-chance AUC indicates cholecystitis commission is largely unpredictable from step-0 features alone.

---

### Cholecystitis — omission  (L1 AUC 0.734 / RF 0.814)

**Univariate and multivariate convergence:**

| feature | Fisher OR | L1 coef | RF rank | interpretation |
| --- | --- | --- | --- | --- |
| LFTs_elevated | 18.5 (↑) | +3.0 | 1st | present → more omissions |
| anorexia | 0.00 (↓) | −3.2 | 3rd | present → fewer omissions |

**Multivariate-only findings:**

| feature | L1 coef | direction |
| --- | --- | --- |
| pain_location_Other | −3.6 | fewer omissions |

**Pattern.** Elevated liver enzymes — the biochemical confirmation of cholestasis — strongly predicts that doctors skip rubric-prescribed workup. This parallels the WBC pattern in appendicitis: a definitive lab marker reduces the perceived need to complete the full rubric.

---

### Diverticulitis — commission  (L1 AUC 0.375 / RF 0.549 — near chance)

No univariate-significant features; L1 retains only `pain_location_LLQ` (coef 0.26, single nonzero feature). RF ranks `pain_location_LLQ` first (importance 0.14) but the model is near chance.

**Pattern.** Commission for diverticulitis is structurally driven: the rubric graph blocks at `CLINICAL_ASSESSMENT` for patients with LLQ pain but without fever, leaving Lab_Panel as the entire simulated sequence; doctors then add CT, which registers as commission. This structural artefact cannot be captured by step-0 features, explaining the near-zero AUC.

---

### Pancreatitis — commission  (L1 AUC 0.766 / RF 0.781)

**Univariate and multivariate convergence:**

| feature | Fisher OR | L1 coef | RF rank | interpretation |
| --- | --- | --- | --- | --- |
| RUQ_tenderness | 10.9 (↑) | +2.1 | 1st | present → more commissions |
| pain_location_Other | inf (↑) | +4.0 | 4th | present → more commissions |
| lipase_ge_3xULN | 0.16 (↓) | not retained | 3rd | present → fewer commissions |
| fever_temp_ge_37_3 | 0.00 (↓) | −2.1 | — | present → fewer commissions |
| LFTs_elevated | 0.29 (↓) | not retained | 2nd | present → fewer commissions |
| pain_location_Epigastric | 0.34 (↓) | not retained | 5th | present → fewer commissions |

**Multivariate-only findings:**

| feature | L1 coef | direction |
| --- | --- | --- |
| prior_diverticular_disease | +3.0 | more commissions |
| anorexia | +2.3 | more commissions |

**Pattern.** Atypical presentations (`RUQ_tenderness`, `pain_location_Other`) prompt extra testing. High-certainty pancreatitis markers (`lipase_ge_3xULN`, `LFTs_elevated`, epigastric location, absent fever) all reduce commission, consistent with Part 3 CertaintyScore findings. `lipase_ge_3xULN` and `LFTs_elevated` do not survive L1 regularisation but remain prominent in RF, suggesting collinearity with retained features. `prior_diverticular_disease` appearing as a commission predictor is likely a confound: patients with known diverticular history may be more extensively worked up regardless of the presenting disease.

---

### Pancreatitis — omission  (L1 AUC 0.616 / RF 0.658)

**Univariate and multivariate convergence:**

| feature | Fisher OR | L1 coef | RF rank | interpretation |
| --- | --- | --- | --- | --- |
| lipase_ge_3xULN | 17.6 (↑) | +2.8 | 1st | present → more omissions |

**Multivariate-only findings:**

| feature | L1 coef | direction |
| --- | --- | --- |
| fever_temp_ge_37_3 | +6.4 | more omissions |
| rebound_tenderness | +6.4 | more omissions |

**Pattern.** Mirrors appendicitis: the pancreatitis-specific certainty marker (`lipase_ge_3xULN`) that reduces commission here promotes omission. Fever and rebound tenderness — indicators of severe or systemic disease — also increase omission, consistent with clinicians abbreviating the workup when diagnosis and severity are already apparent.

---

## Cross-disease summary

**Asymmetric certainty effect.** For appendicitis, cholecystitis, and pancreatitis, each disease has a key high-certainty lab marker (WBC_gt_10k / LFTs_elevated / lipase_ge_3xULN) that is negative for commission and positive for omission. The same step-0 signal that suppresses over-testing simultaneously predicts under-testing relative to the rubric.

**Atypical presentation promotes commission.** Features signalling diagnostic uncertainty or atypical location (`RUQ_tenderness` in appendicitis, `pain_location_Other` and `RUQ_tenderness` in pancreatitis, `symptom_duration_over_72h`) consistently carry positive L1 coefficients for commission across diseases.

**Diverticulitis is the exception.** Near-chance AUC (L1 0.375) reflects a structural gap in the rubric rather than feature-driven behaviour; the commission pattern there cannot be explained by step-0 clinical features.

**Univariate and multivariate agree on direction for all jointly significant features.** No feature showed a sign reversal between Fisher OR and L1 coefficient, confirming that the univariate screening is not badly confounded in this dataset.
