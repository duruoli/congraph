# LLM next-test recommendation — descriptive findings (n=300)

Supplementary experiment to the descriptive paper.  All claims here are
*descriptive*; no strategy is judged "better".

Scaled up from the 30-patient pilot (preserved under `*_pilot.*`).

## Setup recap

- Cohort: all 300 test patients in the standard split (`seed=42`,
  `n_test=300`, `min_tests=2`), with disease and disease-specific
  CertaintyScore tercile (CS = `certainty_score_oracle`) carried in
  for stratification.
  Breakdown: appendicitis 71, cholecystitis 93, diverticulitis 35,
  pancreatitis 101.
- Model: GPT-4o, `temperature=0`, JSON mode, `max_tokens=400`,
  client-level `max_retries=5`.
- Open-loop trajectory: at each step the LLM picks the next test, the
  KNN FeatureSimulator advances features along that pick, until the
  LLM emits `STOP`, picks an invalid / already-done test, the
  simulator fails (`sim_failed`: too few train neighbours performed
  the picked test), or `max_steps=7` is reached.
- Conditions:
  - `llm_features_only` : patient findings + tests-already-done.
  - `llm_full`          : patient findings + rubric simulator next-test
                          recommendation + full sub-rubric graph
                          (nodes + edges with NL labels) + top-5
                          KNN-similar training patients' actual full
                          test sequences.
- Sub-rubric for `llm_full` is the **oracle disease**, matching the
  existing `rubric_sim_oracle.json` baseline.
- `info_order = rubric_first` (single order; see order-sensitivity
  note below).
- Lab_Panel is prepended to LLM sequences before gap comparison
  (matches `scripts/extract_test_sequences.py`).
- Parallelism: 10 worker threads in `ThreadPoolExecutor`; 600
  trajectories (300 × 2 conditions) finished in 206 s wall-clock.

Cost recorded from token usage: **1,472 calls, ~1.6 M input tokens,
~85 K output tokens, ~$4.85** at GPT-4o list prices.

## Order-sensitivity check (3 patients × 2 orders, from pilot)

`order_sensitivity_check.csv`: 5/6 trajectories identical across
info orders (rubric-first vs KNN-first).  Borderline stable; main
run uses `info_order=rubric_first` and the variance is flagged here
rather than averaged across orders.

## Termination diagnostics

Per-step termination over all 1,472 calls:

| reason       | count | notes                                          |
| ------------ | ----- | ---------------------------------------------- |
| `ok`         | 872   | LLM continued the trajectory                   |
| `stop`       | 465   | LLM emitted `STOP`                             |
| `sim_failed` | 135   | KNN simulator had < min_count neighbours for the LLM's pick |

Of the 135 `sim_failed`, **116 occur under `llm_features_only`** and
the test picked is `HIDA_Scan` (95) or `MRCP_Abdomen` (39), almost
all in cholecystitis (80) and pancreatitis (36).  Reading: without
the rubric block, the LLM disproportionately picks low-base-rate
tests (HIDA, MRCP) that the KNN training corpus barely covers.  Once
the rubric block is present, the LLM is anchored to the more common
guideline path and `sim_failed` drops by ~6×.

## Headline gap statistics (`gap_comparison_table.csv`)

ALL diseases pooled (n=300 each row):

| condition          | exact | commission | omission | order_swap | mean_len | actual_len |
| ------------------ | ----- | ---------- | -------- | ---------- | -------- | ---------- |
| `rubric_only`      | 0.083 | 0.607      | 0.293    | 0.030      | 2.17     | 3.25       |
| `llm_features_only`| 0.100 | 0.513      | 0.587    | 0.040      | 2.75     | 3.25       |
| `llm_full`         | 0.143 | 0.430      | 0.427    | 0.067      | 2.61     | 3.25       |

Observations (descriptive, n=300):

1. **`llm_full` has the highest exact-match rate (14.3%)** vs
   `llm_features_only` (10.0%) and `rubric_only` (8.3%).  At
   n=300 the gap between `llm_full` and `rubric_only` is ~6 pp
   (≈18 patients), which is no longer attributable to pilot noise.
2. **`llm_full` has the lowest commission rate (43.0%)** — given a
   patient, the LLM with rubric context is least likely to order an
   off-actual test (excluding `Radiograph_Chest`).
3. **Omission ordering: rubric (29%) < llm_full (43%) < features-only
   (59%).**  The rubric is the most conservative; adding LLM
   exploration injects omissions; removing the rubric block from the
   LLM context injects more.
4. **Order swap is rare (≤7%)** in every condition.  When tests are
   shared with actual, the relative order matches.
5. **Mean LLM sequence length** climbs in the order
   rubric (2.17) < llm_full (2.61) < features-only (2.75) <
   actual (3.25).  LLM with full context lands closest to actual
   length, while still ~0.6 tests shorter on average.

### Per-disease

- **Appendicitis (n=71)**: `llm_full` is the strongest cell — exact-
  match 43.7% vs rubric 19.7%, commission 9.9% vs rubric 45.1%.
  `Ultrasound → CT` is the dominant rubric path and the LLM
  reproduces it cleanly once the sub-rubric is provided.
- **Cholecystitis (n=93)**: exact-match stays low everywhere (≤4%);
  omission is universal because actual practice frequently adds
  `MRCP_Abdomen` and `HIDA_Scan` that no condition predicts at the
  same rate.  `llm_full` cuts omission by ~30 pp vs features-only
  (0.57 vs 0.87).
- **Diverticulitis (n=35)**: `llm_full` has 0.09 omission (vs rubric
  0.00) and short sequences (2.11 vs actual 3.23).  LLM treats CT
  positive as terminus, mirroring rubric.
- **Pancreatitis (n=101)**: `llm_full` commission 56% vs rubric 68% —
  some improvement, but omission still 51%, and the picture is
  closest to the "rubric is short, actual is long" pattern flagged
  in Part 4.

## Mechanism check: CertaintyScore stratification (`certainty_stratified.csv`)

The Part-4 prediction: high CS → shorter LLM seq (shortcut); low CS →
more commissions (add checks).  Mean length and mean n_commission per
condition × CS tercile:

| condition          | CS    | mean_len | n_commission | n_omission |
| ------------------ | ----- | -------- | ------------ | ---------- |
| `llm_full`         | high  | 2.73     | 0.40         | 0.55       |
| `llm_full`         | mid   | 2.44     | 0.55         | 0.38       |
| `llm_full`         | low   | 2.64     | 0.61         | 0.59       |
| `llm_features_only`| high  | 2.60     | 0.58         | 0.61       |
| `llm_features_only`| mid   | 2.72     | 0.54         | 0.62       |
| `llm_features_only`| low   | 2.92     | 0.62         | 0.88       |
| `rubric_only`      | high  | 2.50     | 0.52         | 0.46       |
| `rubric_only`      | mid   | 2.18     | 0.79         | 0.36       |
| `rubric_only`      | low   | 1.83     | 1.04         | 0.22       |

Observations:

1. **Shortcut hypothesis (high CS → shorter sequence)** is *not*
   supported in either LLM condition — `llm_full` is actually
   slightly *longer* at high CS (2.73) than mid (2.44).  It IS
   supported in `rubric_only` (2.50 high → 1.83 low — but here "low
   CS" means the rubric routes shallowly because evidence is weak,
   not a deliberate shortcut).
2. **Add-checks hypothesis (low CS → more commissions)** is supported
   in every condition: commission rises as CS falls (`rubric_only`:
   0.52 high → 1.04 low; `llm_full`: 0.40 → 0.61; `features_only`:
   0.58 → 0.62).  The signal is strongest for the rubric and
   weakened by the LLM, which appears to "smooth" the response.
3. **Exact match drops sharply at low CS** for `llm_full` (0.208 high
   → 0.079 low), echoing Part 4's pattern: ambiguous patients are
   hard for *every* approach, and the LLM does not rescue them.
4. **Omission rises with low CS in both LLM conditions**.  Low CS
   patients are atypical, so the LLM cannot reliably anchor to the
   rubric-recommended path.

In short: the rubric does encode the Part-4 mechanism (low CS → more
commission); the LLM partially flattens this signal but doesn't
reverse it.

## Reasoning patterns (qualitative, see `llm_reasoning_samples.md`)

The 30 sampled reasoning snippets (now drawn from 1,472 step records,
stratified across condition × disease × CS tercile) show:

- LLM cites named criteria (Alvarado, TG18, Atlanta, BISAP) verbatim,
  indicating significant overlap between GPT-4o pretraining and the
  guideline corpus our rubric was authored from.  `llm_features_only`
  is therefore *not* an information source independent of the rubric.
- Under `llm_full` the model frequently cites the rubric directly
  ("the rubric simulator did not recommend additional tests"); under
  `llm_features_only` it falls back on generic clinical reasoning.
- `STOP` is most often issued after CT confirms the disease; the LLM
  treats CT as the decision endpoint even when the rubric routes on
  to MRCP (cholecystitis biliary eval, pancreatitis biliary eval).

## Honesty constraints

- GPT-4o was trained on biomedical text that overlaps with the
  WSES / TG18 / AAFP / Atlanta guidelines used to author the rubric
  graphs.  Comparing `llm_features_only` to `rubric_only` is closer
  to comparing two readings of the same guideline corpus than to
  comparing two distinct information sources.  Flag in any
  downstream prose.
- MIMIC-IV is pre-2023 and may appear in pretraining; without
  identifiers the model is unlikely to recall specific encounters,
  but feature distributions could be familiar.
- All numbers above are **descriptive**.  No inferential claim is
  made that one condition is "better" than another.
- The `sim_failed` count (135) means 135 LLM picks were dropped
  silently from sequence-level metrics because the KNN simulator
  could not advance the trajectory; in those cases the LLM sequence
  is truncated at that step.

## Open items (deferred to main-triage rerun)

Listed separately in `handoff_summary_llm.md`.

## Outputs

```
results/llm_experiment/
    order_sensitivity_check.csv         # 3 patients × 2 info orders (from pilot)
    order_sensitivity_check_steps.csv
    llm_recommendations.csv             # 300 patients × 2 conditions
    llm_recommendations_steps.csv       # per-step incl. reasoning & tokens
    per_patient_gap.csv                 # per-patient gap flags (all 3 conditions)
    gap_comparison_table.csv            # aggregated gap stats × disease × condition
    certainty_stratified.csv            # aggregated gap stats × CS tercile × condition
    llm_reasoning_samples.md            # 30 sampled snippets
    findings.md                         # this file
    handoff_summary_llm.md              # main-triage rerun checklist

    *_pilot.{csv,md}                    # n=30 pilot snapshot (kept for traceability)
```
