#!/usr/bin/env python3
"""Atomic facts from radiology reports: two methods side by side on a small sample.

An atomic fact = the smallest statement about the patient that can be true or false on its own,
in slots (same schema family as fragment_evidence.py, extended for this track):

  anatomy         organ / region / structure the statement is about        "sigmoid colon"
  attribute       the property assessed                                    "wall"
  state           descriptor as written                                    "thickened"
  finding_status  abnormal / normal / not_evaluated / equivocal
  assertion       present / absent / possible / historical                 (ConText-like)
  kind            observation (what is seen) / interpretation (the reader's conclusion)
  value_unit, qualifier, quote (verbatim supporting span)

Only FINDINGS + IMPRESSION are split; the indication is the ordering doctor's question, not a fact.

Methods
  concept  pair scispaCy/UMLS concept mentions inside one sentence (runs/term_concepts cache):
           each finding concept -> nearest anatomy concept; free, rule-based
  llm      gpt-5-mini via OpenRouter (experiments.annotation.annotate.call_json)

  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/atomic_facts.py --method concept
  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/atomic_facts.py --method llm
  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/atomic_facts.py --compare

Sample = every report of the 3 pilot admissions + N random primary-IBD reports (seed 0).
Restricted MIMIC-derived output: runs/atomic_facts/ only.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from scripts.term_concepts import sections

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw_data/ibd_mimiciv_3_1"
OUT = ROOT / "runs/atomic_facts"
PILOT = [23605251, 25292961, 29942834]
MODEL = "openai/gpt-5-mini"  # switched from Sonnet at the user's request (2026-10-16)

FIELDS = ["anatomy", "attribute", "state", "finding_status", "assertion", "kind", "value_unit", "qualifier", "quote"]
FINDING_STATUS = {"abnormal", "normal", "not_evaluated", "equivocal"}
ASSERTION = {"present", "absent", "possible", "historical"}
KIND = {"observation", "interpretation"}

SYSTEM = """You split the FINDINGS and IMPRESSION of a radiology report into ATOMIC FACTS about the patient.

An atomic fact is the smallest statement that can be true or false on its own. For EACH one emit:
{
  "anatomy":   organ / region / structure the fact is ABOUT, lower-case noun phrase
               (e.g. sigmoid colon, terminal ileum, liver, common bile duct, peritoneal cavity, left basilic vein, lung bases).
  "attribute": the property assessed (e.g. wall, lumen, diameter, enhancement, fat stranding, free air, fluid, stones,
               thrombus, density, size, haustra). Use "overall" only for a whole-organ verdict ("liver is normal").
  "state":     the descriptor as written (e.g. thickened, dilated, present, absent, hypodense, nonocclusive, unremarkable).
  "finding_status": exactly one of {abnormal, normal, not_evaluated, equivocal}
               normal = stated normal / unremarkable / absent pathology; abnormal = a pathologic finding;
               not_evaluated = named but not assessed / not visualized / limited; equivocal = indeterminate.
  "assertion": exactly one of {present, absent, possible, historical}
               present = the report affirms it now; absent = the report denies it ("no free air");
               possible = hedged ("may", "likely", "cannot be excluded", "versus");
               historical = stated as previously known / prior surgery / s/p, not a new observation ("known ulcerative colitis").
  "kind":      exactly one of {observation, interpretation}
               observation = what is seen; interpretation = the radiologist's diagnostic conclusion
               ("consistent with pancolitis", "most likely due to barotrauma").
  "value_unit": measurement with unit if given ("8.5 mm"), else null.
  "qualifier":  extent, severity, laterality or comparison not captured above ("mild", "interval decreased",
               "diffuse"), else null.
  "quote":     the shortest verbatim span of the report that supports this fact.
}

RULES
1. ONE fact per (anatomy, attribute, state). "Sigmoid wall thickening with fat stranding" = 2 facts.
2. A normal sweep ("liver, spleen and adrenals are unremarkable") = one normal fact PER named structure.
3. A negated list ("no free air, free fluid or abscess") = one absent fact per item.
4. Keep interpretations as their own facts with kind=interpretation; do NOT fold them into observations.
5. Repeat facts that the IMPRESSION restates; do not deduplicate across sections.
6. Use ONLY what the text says. Do not infer unstated findings. Ignore technique, comparison and recommendation text.

Return STRICT JSON: {"facts": [ {...}, ... ]}"""


# ---------------------------------------------------------------- sample
def sample(n_random):
    r = pd.read_parquet(RAW / "radiology", columns=["note_id", "hadm_id", "charttime", "exam_name", "text"])
    c = pd.read_parquet(RAW / "cohort")
    prim = r[r.hadm_id.isin(c[c.is_primary_ibd_diagnosis].hadm_id)]
    pilot = prim[prim.hadm_id.isin(PILOT)]
    rest = prim[~prim.hadm_id.isin(PILOT)].sample(n_random, random_state=0)
    s = pd.concat([pilot, rest]).reset_index(drop=True)
    s["finding_text"] = s.text.map(lambda t: "\n".join(c for k, c in sections(t or "") if k == "finding"))
    return s[s.finding_text.str.strip().ne("")]


# ---------------------------------------------------------------- method: concept pairing
STATUS_TO_ASSERTION = {"affirmed": "present", "negated": "absent", "uncertain": "possible", "hypothetical": "possible",
                       "historical": "historical", "family": "historical"}
NORMAL_WORD = re.compile(r"\b(normal|unremarkable|patent|within normal limits)\b", re.I)


def concept_facts(s):
    m = pd.read_parquet(ROOT / "runs/term_concepts/mentions.parquet")
    m = m[m.note_id.isin(s.note_id) & m.section.eq("finding")]
    out = []
    for (note, sent), g in m.groupby(["note_id", "sentence"], sort=False):
        pos = lambda x: sent.lower().find(x.lower())
        anat = [(pos(r.mention), r) for r in g[g.group.eq("anatomy")].itertuples()]
        finds = [r for r in g[g.group.eq("finding")].itertuples()]
        for f in finds:  # finding concept -> nearest anatomy concept in the same sentence
            p = pos(f.mention)
            near = min(anat, key=lambda a: abs(a[0] - p))[1] if anat else None
            out.append({"note_id": note, "anatomy": near.concept if near else None, "attribute": None,
                        "state": f.concept, "finding_status": "abnormal" if f.status == "affirmed" else "normal",
                        "assertion": STATUS_TO_ASSERTION.get(f.status, "present"), "kind": None,
                        "value_unit": None, "qualifier": None, "quote": sent})
        if not finds and NORMAL_WORD.search(sent):  # "liver and spleen are normal": anatomy-only sentence
            for _, a in anat:
                out.append({"note_id": note, "anatomy": a.concept, "attribute": "overall", "state": "normal",
                            "finding_status": "normal", "assertion": "present", "kind": None,
                            "value_unit": None, "qualifier": None, "quote": sent})
    return out


# ---------------------------------------------------------------- method: LLM
def llm_one(row):
    from experiments.annotation.annotate import call_json
    user = f"RADIOLOGY REPORT: {row.exam_name}\n\n{row.finding_text}"
    try:  # gpt-5 models take only temperature=1; max_tokens also covers their reasoning tokens
        res = call_json(SYSTEM, user, model=MODEL, temperature=1, max_tokens=16000)
    except Exception as e:  # one failed report should not stop the batch
        print(f"  {row.note_id}: {type(e).__name__}: {e}")
        return [], {}
    facts = []
    for f in (res.get("parsed") or {}).get("facts") or []:
        if not isinstance(f, dict):
            continue
        f = {k: (str(f[k]).strip() if f.get(k) not in (None, "") else None) for k in FIELDS}
        f["anatomy"] = (f["anatomy"] or "").lower() or None
        for k, allowed in (("finding_status", FINDING_STATUS), ("assertion", ASSERTION), ("kind", KIND)):
            if (f[k] or "").lower() not in allowed:
                f[k + "_raw"], f[k] = f[k], None
            else:
                f[k] = f[k].lower()
        facts.append({"note_id": row.note_id, **f})
    return facts, res.get("usage")


def llm_facts(s, workers):
    out, usage = [], []
    with ThreadPoolExecutor(workers) as ex:
        for facts, u in ex.map(llm_one, s.itertuples()):
            out += facts
            usage.append(u or {})
    tok_in = sum(u.get("prompt_tokens", 0) for u in usage)
    tok_out = sum(u.get("completion_tokens", 0) for u in usage)
    cost = sum(u.get("cost") or 0 for u in usage)
    print(f"{MODEL}: tokens in {tok_in:,} out {tok_out:,}, cost ${cost:.2f} (OpenRouter)")
    return out


# ---------------------------------------------------------------- side-by-side view
def compare(s):
    load = lambda m: pd.read_json(OUT / m / "facts.jsonl", lines=True) if (OUT / m / "facts.jsonl").exists() else None
    a, b = load("concept"), load("llm")
    lines = ["# Atomic facts: concept pairing vs LLM", "",
             "| | reports | facts | facts / report | present | absent | possible | historical |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, f in (("concept", a), ("llm", b)):
        if f is None:
            continue
        v = f.assertion.value_counts()
        lines.append(f"| {name} | {f.note_id.nunique()} | {len(f)} | {len(f) / max(1, f.note_id.nunique()):.1f} | "
                     + " | ".join(str(v.get(k, 0)) for k in ("present", "absent", "possible", "historical")) + " |")
    show = lambda f, cols: "\n".join(
        "| " + " | ".join(str(x if x is not None else "") .replace("|", "/") for x in r) + " |"
        for r in f[cols].itertuples(index=False))
    for r in s.itertuples():
        lines += ["", f"## {r.exam_name} · hadm {r.hadm_id} · note {r.note_id}", "", "```", r.finding_text.strip(), "```"]
        for name, f, cols in (("concept", a, ["anatomy", "state", "assertion"]),
                              ("llm", b, ["anatomy", "attribute", "state", "finding_status", "assertion", "kind", "qualifier"])):
            if f is None:
                continue
            g = f[f.note_id.eq(r.note_id)]
            lines += ["", f"**{name}** ({len(g)})", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            lines.append(show(g, cols))
    (OUT / "compare.md").write_text("\n".join(lines) + "\n")
    print("->", OUT / "compare.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["concept", "llm"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    s = sample(args.n_random)
    if args.method:
        (OUT / args.method).mkdir(parents=True, exist_ok=True)
        facts = concept_facts(s) if args.method == "concept" else llm_facts(s, args.workers)
        pd.DataFrame(facts).to_json(OUT / args.method / "facts.jsonl", orient="records", lines=True, force_ascii=False)
        print(f"{args.method}: {len(facts)} facts from {len(s)} reports")
    if args.compare:
        compare(s)


if __name__ == "__main__":
    main()
