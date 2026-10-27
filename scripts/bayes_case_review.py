#!/usr/bin/env python3
"""Bayes env v0, step 1.6 (e): read the charts behind first-time non-routine orders.

After (d) the gap between doctors' orders and EV is not the definitional artefact; two candidates remain:
the doctor's question is not one of the 10 problems (s), or the doctor knew things the model does not (o).
Sample: first-time orders of non-routine actions (as in bayes_ev_first), 2 per action for the 12 most ordered,
seeded. Per case: what was ordered (and with what), its EV rank that day, the model's top-EV action, the
10 coded problems, then from the discharge note the chief complaint, HPI, hospital course, and radiology
impressions reported before the order.

Outputs runs/bayes_env/v0/ev/case_review/cases.md (local only: MIMIC text)

  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/bayes_case_review.py
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from scripts.bayes_ev import OUT
from scripts.bayes_ev_first import ROUTINE
from scripts.bayes_v0 import DATA, RAW_DIR

PER_ACTION, N_ACTIONS, SEED = 2, 12, 0
DIR = OUT / "case_review"
CUT = {"hpi": 1500, "course": 3500, "rad": 400}


def section(text, name, nxt):
    """text between a header and the first of the next headers."""
    m = re.search(rf"\n{name}:?\s*\n", text, flags=re.I)
    if not m:
        return ""
    rest = text[m.end():]
    ends = [e.start() for h in nxt if (e := re.search(rf"\n{h}:?\s*\n", rest, flags=re.I))]
    return rest[:min(ends)].strip() if ends else rest.strip()


def clip(t, n):
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t if len(t) <= n else t[:n] + " […]"


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(OUT / "ev.parquet")
    problems = [c for c in ev.columns if c not in {"hadm_id", "day", "action", "EV", "ordered", "H", "before",
                                                    "EV_std", "H_std"}]
    ev["pct"] = ev.groupby(["hadm_id", "day"]).EV.rank(method="average").sub(1) / (ev.action.nunique() - 1)
    share = ev.groupby("action").ordered.mean()
    first = ev[~ev.action.isin(share.index[share >= ROUTINE]) & ~ev.before & ev.ordered]
    acts = first.action.value_counts().index[:N_ACTIONS]
    rng = np.random.default_rng(SEED)
    pick = pd.concat([first[first.action.eq(a)].iloc[rng.choice((first.action == a).sum(), PER_ACTION, replace=False)]
                      for a in acts])
    top = ev.loc[ev.groupby(["hadm_id", "day"]).EV.idxmax()].set_index(["hadm_id", "day"])

    events = pd.read_parquet(DATA / "actions/events.parquet")
    s = pd.read_parquet(DATA / "s.parquet").set_index("hadm_id")
    notes = pd.read_parquet(RAW_DIR / "discharge_notes").sort_values("note_seq").groupby("hadm_id").text.last()
    rad = pd.read_parquet(RAW_DIR / "radiology", columns=["hadm_id", "charttime", "exam_name", "text"])
    coh = pd.read_parquet(RAW_DIR / "cohort").set_index("hadm_id")

    L = ["# Case review: first-time non-routine orders (local only, MIMIC text)", "",
         f"{len(pick)} cases = {PER_ACTION} per action × the {len(acts)} most frequent first-time non-routine actions "
         f"(seed {SEED}). EV pct = the ordered action's EV percentile on that day's 28-action menu.", ""]
    for i, r in enumerate(pick.itertuples(), 1):
        h, d, a = r.hadm_id, r.day, r.action
        e = events[events.hadm_id.eq(h) & events.action.eq(a) & events.day.eq(d)]
        t0 = pd.Timestamp(e.t.min())
        same = sorted(set(events[events.hadm_id.eq(h) & events.t.eq(e.t.min())].action) - {a})
        tp = top.loc[(h, d)]
        coded = [k for k in problems if s.loc[h, k] == 1] if h in s.index else []
        c = coh.loc[h]
        text = notes.get(h, "")
        rr = rad[rad.hadm_id.eq(h) & (pd.to_datetime(rad.charttime) < t0)].sort_values("charttime")
        L += [f"## {i}. `{a}` — hadm {h}, day {d}", "",
              f"- patient: {c.gender}, {c.age}, {c.ibd_type}; admitted {c.admittime}; ordered {t0}",
              f"- ordered with (same draw): {', '.join(same) or '—'}",
              f"- EV of this action {r.EV * 1000:.2f} ×1e-3 nats, pct {r.pct:.2f}; model's top that day: "
              f"`{tp.action}` ({tp.EV * 1000:.2f})",
              f"- coded problems (s=1): {', '.join(coded) or 'none of the 10'}", "",
              f"**Chief complaint:** {clip(section(text, 'Chief Complaint', ['Major Surgical', 'History of Present']), 300)}", "",
              f"**HPI:** {clip(section(text, 'History of Present Illness', ['Past Medical History', 'PMH']), CUT['hpi'])}", "",
              f"**Radiology before the order** ({len(rr)}):"]
        for q in rr.itertuples():
            imp = section(q.text, "IMPRESSION", ["NOTIFICATION", "RECOMMENDATION"]) or q.text[-CUT["rad"]:]
            L.append(f"- {q.charttime} {q.exam_name}: {clip(' '.join(imp.split()), CUT['rad'])}")
        course = section(text, "Brief Hospital Course", ["Medications on Admission", "Discharge Medications",
                                                          "Discharge Disposition"])
        L += ["", f"**Hospital course:** {clip(course, CUT['course']) or '(not found)'}", "",
              f"**Discharge diagnosis:** {clip(section(text, 'Discharge Diagnosis', ['Discharge Condition']), 400)}",
              "", "---", ""]
    (DIR / "cases.md").write_text("\n".join(L) + "\n")
    print(f"{len(pick)} cases -> {DIR / 'cases.md'}")


if __name__ == "__main__":
    main()
