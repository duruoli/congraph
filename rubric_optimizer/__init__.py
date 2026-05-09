"""rubric_optimizer — LLM-driven self-iteration framework for the ConGraph rubric engine.

Loop
----
  1. trial_runner  : evaluate all patients → list[PatientRecord]
  2. failure_analyzer : attribute failures per disease → FailureAnalysis
  3. trial_log     : persist TrialRecord to JSONL
  4. summarizer    : package everything into an LLM-readable summary
  5. (LLM proposes ONE code change)
  6. apply change → re-run → accept / reject → goto 1
"""
