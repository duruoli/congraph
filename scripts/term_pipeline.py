"""Run the 'thing x recurrence x transfer' exploration for one sample definition, end to end.

  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/term_pipeline.py --cohort ibd_primary_age
  ... --steps profile,auc_word          (only some steps)

A new comparison group = one new entry in transfer_bow_model.COHORTS, then run this.
Steps (see scripts/TERM_PIPELINE.md for inputs/outputs and the manual review points):
  profile      who is in the groups (background facts, imaging done)       cohort_profile.py
  extract      radiology report -> concept mentions, only reports not yet in the shared cache
                                                                            term_concepts.py
  table        mentions -> per-sample concept counts + label-blind screen   term_concept_table.py
  auc_word     one-feature AUC, words + structured items                    term_auc.py --unit word
  auc_concept  one-feature AUC, concepts                                    term_auc.py --unit concept
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = {
    "profile": ["scripts/cohort_profile.py"],
    "extract": ["scripts/term_concepts.py"],
    "table": ["scripts/term_concept_table.py"],
    "auc_word": ["scripts/term_auc.py", "--unit", "word"],
    "auc_concept": ["scripts/term_auc.py", "--unit", "concept"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", default="all", help="name in transfer_bow_model.COHORTS")
    parser.add_argument("--steps", default=",".join(STEPS), help="comma-separated subset, run in this order")
    args = parser.parse_args()
    from scripts.transfer_bow_model import COHORTS
    if args.cohort not in COHORTS:
        sys.exit(f"unknown cohort {args.cohort!r}; known: {', '.join(COHORTS)}")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    for step in args.steps.split(","):
        print(f"\n=== {step} ({args.cohort}) ===", flush=True)
        subprocess.run([sys.executable, *STEPS[step], "--cohort", args.cohort], cwd=ROOT, env=env, check=True)
    out = "runs/term_auc" + ("" if args.cohort == "all" else f"/{args.cohort}")
    print(f"\nresults: {out}/profile.md, words.md, concepts.md")


if __name__ == "__main__":
    main()
