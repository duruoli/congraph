"""Join belief_deviation_analysis.csv with timing_roles.csv and emit filtered
datasets that drop monitoring decision steps, at two cut lines.

Uses the real `timing_role` once source data is present; falls back to the
provisional `text_post_intervention_hint` (exclude_from_deviation) in degraded mode.

Two cuts (both emitted, contamination printed before->after):
  A. INTERVENTION cut  -> belief_deviation_filtered.csv
     keeps everything except post_intervention (the unambiguous-monitoring tail).
  B. PRE-ADMISSION cut -> belief_deviation_preadmission.csv
     keeps ONLY pre_admission (charttime < admittime): the purest ED/outpatient
     diagnostic test-selection phase, before the disease is established. This
     also drops the post_admission_diagnostic grey zone (admitted-but-pre-
     intervention staging/re-scan), which the intervention cut lets through.
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
    return len(df[(df.dev_belief == dev_b) & (df.verification == vind)])

print("DD/FD cells  (deviate/follow x disconfirmed) — before -> after filter:")
for lab, dv in [("FD follow+disconfirmed", "follow"),
                ("DD deviate+disconfirmed", "deviate")]:
    b, a = cell(m, dv, "disconfirmed"), cell(kept, dv, "disconfirmed")
    print(f"  {lab:26} {b:3} -> {a:3}   (-{b-a})")
print("\nexcluded steps by dev_belief x verification:")
ex = m[m.excluded_monitoring]
if len(ex):
    print(pd.crosstab(ex.dev_belief, ex.verification, margins=True).to_string())

# --- Cut B: pre-admission only (purest diagnostic phase) -------------------- #
pre = m[m.timing_role == "pre_admission"].copy()
pre.to_csv(D / "belief_deviation_preadmission.csv", index=False)
print(f"\n[{mode}] PRE-ADMISSION cut: kept {len(pre)} / {len(m)} steps "
      f"(dropped {len(m) - len(pre)}: post_admission_diagnostic + same_day + post_intervention)")
print(f"wrote {D/'belief_deviation_preadmission.csv'}")
print("\nDD/FD cells — intervention-cut (kept) -> pre-admission cut:")
for lab, dv in [("FD follow+disconfirmed", "follow"),
                ("DD deviate+disconfirmed", "deviate")]:
    a, p = cell(kept, dv, "disconfirmed"), cell(pre, dv, "disconfirmed")
    print(f"  {lab:26} {a:3} -> {p:3}   (-{a-p})")
