"""Join belief_deviation_analysis.csv with timing_roles.csv and emit a filtered
dataset that drops post-intervention (monitoring) decision steps.

Uses the real `timing_role` once source data is present; falls back to the
provisional `text_post_intervention_hint` (exclude_from_deviation) in degraded mode.

Output: results/annotation_experiment/full/belief_deviation_filtered.csv
        (+ prints DD/FD before-vs-after so the contamination effect is visible)
"""
from pathlib import Path
import pandas as pd

D = Path("results/annotation_experiment/full")
dev = pd.read_csv(D / "belief_deviation_analysis.csv")
tim = pd.read_csv(D / "timing_roles.csv")

key = ["disease", "hadm", "step"]
m = dev.merge(tim[key + ["timing_role", "text_post_intervention_hint",
                         "exclude_from_deviation", "first_intervention_type"]],
              on=key, how="left")
m["excluded_monitoring"] = m["exclude_from_deviation"].fillna(False)
m.to_csv(D / "belief_deviation_filtered.csv", index=False)

kept = m[~m.excluded_monitoring]
mode = "REAL timing_role" if (tim.timing_role != "unknown").any() else "PROVISIONAL text-hint"
print(f"[{mode}] total steps {len(m)} | excluded {int(m.excluded_monitoring.sum())} "
      f"| kept {len(kept)}")
print(f"wrote {D/'belief_deviation_filtered.csv'}\n")

def cell(df, dev_b, vind):
    return len(df[(df.dev_belief == dev_b) & (df.vindication == vind)])

print("DD/FD cells  (deviate/follow x disconfirmed) — before -> after filter:")
for lab, dv in [("FD follow+disconfirmed", "follow"),
                ("DD deviate+disconfirmed", "deviate")]:
    b, a = cell(m, dv, "disconfirmed"), cell(kept, dv, "disconfirmed")
    print(f"  {lab:26} {b:3} -> {a:3}   (-{b-a})")
print("\nexcluded steps by dev_belief x vindication:")
ex = m[m.excluded_monitoring]
if len(ex):
    print(pd.crosstab(ex.dev_belief, ex.vindication, margins=True).to_string())
