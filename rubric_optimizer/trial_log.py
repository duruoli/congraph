"""trial_log.py — Persistent JSONL log of rubric optimizer trials.

Each line is one TrialRecord serialised as JSON.  The log is the LLM's
"memory": recent records are included in every trial summary so the LLM
can see what was tried, whether it worked, and why.

File location (default): results/rubric_optimizer_log.jsonl
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

_PKG  = Path(__file__).parent
_ROOT = _PKG.parent
for _p in [str(_ROOT), str(_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_LOG_PATH: Path = _ROOT / "results" / "rubric_optimizer_log.jsonl"


# ---------------------------------------------------------------------------
# ChangeRecord — what code was modified this trial
# ---------------------------------------------------------------------------

@dataclass
class ChangeRecord:
    """Describes the one code change applied in this trial."""
    target_file:  str   # "rubric_graph.py" | "diagnosis_distribution.py"
    change_type:  str   # "edge_condition" | "scoring_param" | "triage_condition"
    description:  str   # short human-readable summary (1 sentence)
    old_code:     str   # exact snippet replaced (must be findable in the file)
    new_code:     str   # replacement snippet
    rationale:    str   # LLM's reasoning (why this change was proposed)


# ---------------------------------------------------------------------------
# MetricSnapshot — per-disease and overall accuracy at one trial
# ---------------------------------------------------------------------------

@dataclass
class MetricSnapshot:
    overall_accuracy: float
    per_disease:      dict[str, float]   # disease → accuracy (float 0–1)


# ---------------------------------------------------------------------------
# TrialRecord — one complete iteration
# ---------------------------------------------------------------------------

@dataclass
class TrialRecord:
    """
    Full record of one rubric optimizer iteration.

    Fields
    ------
    trial_id        : monotonically increasing integer
    timestamp       : ISO 8601 UTC string
    step_selector   : "first" | "last" | "all"
    metrics         : MetricSnapshot (accuracy only; speed/cost tracked separately)
    delta_accuracy  : overall_accuracy − previous trial's overall_accuracy
                      None for trial 0 (baseline)
    worst_disease   : disease with lowest per_disease accuracy
    change_applied  : ChangeRecord | None  (None = baseline or no change)
    outcome         : "baseline" | "accepted" | "rejected"
    outcome_reason  : short explanation (why accepted or rejected)
    attribution     : attribution_hypothesis from FailureAnalysis for worst_disease
    """
    trial_id:        int
    timestamp:       str
    step_selector:   str

    metrics:         MetricSnapshot
    delta_accuracy:  float | None

    worst_disease:   str
    change_applied:  ChangeRecord | None
    outcome:         str   # "baseline" | "accepted" | "rejected"
    outcome_reason:  str
    attribution:     str   # failure hypothesis for worst_disease


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _to_dict(record: TrialRecord) -> dict:
    d = asdict(record)
    # asdict handles nested dataclasses automatically
    return d


def _from_dict(d: dict) -> TrialRecord:
    m  = d.pop("metrics")
    ch = d.pop("change_applied")
    record = TrialRecord(
        **d,
        metrics        = MetricSnapshot(**m),
        change_applied = ChangeRecord(**ch) if ch is not None else None,
    )
    return record


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------

def append_trial(record: TrialRecord, path: Path | str | None = None) -> None:
    """Append one TrialRecord to the JSONL log (creates file if absent)."""
    log_path = Path(path) if path else DEFAULT_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_to_dict(record), ensure_ascii=False) + "\n")


def load_trials(path: Path | str | None = None) -> list[TrialRecord]:
    """Load all TrialRecord objects from the JSONL log (oldest first)."""
    log_path = Path(path) if path else DEFAULT_LOG_PATH
    if not log_path.exists():
        return []
    records: list[TrialRecord] = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(_from_dict(json.loads(line)))
    return records


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def make_timestamp() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def next_trial_id(path: Path | str | None = None) -> int:
    """Return the next trial_id (= number of existing records)."""
    return len(load_trials(path))
