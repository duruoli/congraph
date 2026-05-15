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

