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
  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/atomic_facts.py --method llm_v2 --patients
           v2 schema (no attribute; change / known / based_on / repeats), every report of 11 sampled patients
  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/atomic_facts.py --method llm_v3 --patients
           v3: state -> observation; anatomy only from the text (null otherwise); retry on empty answers

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

# v2 (2026-10-16): RadGraph-like. No attribute slot (the part goes into anatomy: "colon wall"); change, known
# and based_on added; each fact has an id so an interpretation can point at the observations it rests on.
FIELDS_V2 = ["id", "anatomy", "state", "certainty", "change", "known", "finding_status", "kind", "based_on",
             "repeats", "modifier", "section", "quote"]
CERTAINTY = {"present", "absent", "possible"}
CHANGE = {"new", "worse", "better", "unchanged", "resolved"}
SECTION = {"findings", "impression"}

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


SYSTEM_V2 = """You split the FINDINGS and IMPRESSION of a radiology report into ATOMIC FACTS about the patient.
An atomic fact is the smallest statement that can be true or false on its own.

Return STRICT JSON: {"facts": [ {...}, ... ]}. Each fact has exactly these keys:
  "id":        "F1", "F2", ... in reading order
  "anatomy":   the structure the fact is about, lower-case, most specific part included, from organ down to
               sub-part: "sigmoid colon wall", "colon surrounding fat", "pancreatic tail", "left basilic vein",
               "peritoneal cavity", "liver". Resolve references ("this lesion", "it") to the structure they mean.
  "state":     the finding itself in 1-4 words, lower-case: "thickened", "fat stranding", "free air", "dilated",
               "nonocclusive thrombus", "hypodense", "normal", "not visualized". Never a whole sentence. Severity,
               size and extent do NOT go here.
  "certainty": "present" | "absent" | "possible"
               absent = denied ("no free air"); possible = hedged ("may", "likely", "versus", "cannot be excluded").
  "change":    "new" | "worse" | "better" | "unchanged" | "resolved" | null
               only when the text compares with before ("again seen" = unchanged, "no longer evident" = resolved,
               "decreased" = better, "increased" = worse). A resolved finding has certainty "absent".
  "known":     true if stated as already known / history of / s/p / prior surgery ("known ulcerative colitis"), else false
  "finding_status": "abnormal" | "normal" | "not_evaluated" | "equivocal"
  "kind":      "observation" = what is seen; "interpretation" = the radiologist's reading of what it means
               ("consistent with pancolitis", "likely due to collapse", "may represent a cyst")
  "based_on":  for an interpretation: ids of the observations it explains, e.g. ["F1", "F2"]; else []
  "repeats":   id of an earlier fact this one restates (typically the IMPRESSION repeating the FINDINGS), else null
  "modifier":  severity / size / extent / measurement if given ("mild", "diffuse", "8.5 mm", "large amount"), else null
  "section":   "findings" | "impression"
  "quote":     the shortest verbatim span of the report that supports the fact

RULES
1. One fact per (anatomy, state). Split lists and compounds: "sigmoid wall thickening with adjacent fat stranding" ->
   {sigmoid colon wall, thickened} and {sigmoid colon surrounding fat, fat stranding}.
2. A normal sweep ("liver, spleen and adrenals are unremarkable") = one normal fact per structure.
   A negated list ("no free air, free fluid or abscess") = one absent fact per item.
3. Never emit the same content twice as observation and interpretation. An interpretation is a separate fact only
   when it adds a reading (a diagnosis, a cause, a differential), and it lists its observations in based_on.
   Example: "colon wall thickening and fat stranding consistent with pancolitis" ->
   F1 {colon wall, thickened, observation}, F2 {colon surrounding fat, fat stranding, observation},
   F3 {colon, pancolitis, interpretation, based_on ["F1","F2"]}.
   Example: "apparent thickening of the descending colon, likely due to collapse" ->
   F1 {descending colon wall, thickened, possible, observation},
   F2 {descending colon, collapse, possible, interpretation, based_on ["F1"]}.
4. When the IMPRESSION restates a FINDINGS fact, emit it again with section "impression" and repeats = that id.
5. A measurement is not its own fact: put it in "modifier" of the fact it measures.
6. Use ONLY what the text says. Do not infer unstated findings. Ignore technique, comparison and recommendations."""


# v3 (2026-10-16): "state" renamed "observation" (what is seen, never present/absent -- that is certainty);
# anatomy only from the text and its context, null when the report names none (no medical-knowledge fill:
# the standard site is decided once per observation in the organ layer, see RESEARCH_ANCHOR.md).
FIELDS_V3 = [f if f != "state" else "observation" for f in FIELDS_V2]
SYSTEM_V3 = """You split the FINDINGS and IMPRESSION of a radiology report into ATOMIC FACTS about the patient.
An atomic fact is the smallest statement that can be true or false on its own.

Return STRICT JSON: {"facts": [ {...}, ... ]}. Each fact has exactly these keys:
  "id":          "F1", "F2", ... in reading order
  "anatomy":     the structure the fact is about, lower-case, most specific part included, from organ down to
                 sub-part: "sigmoid colon wall", "colon surrounding fat", "pancreatic tail", "left basilic vein",
                 "peritoneal cavity", "liver".
                 Take it ONLY from the report: the same sentence, or its context (an earlier sentence, an organ
                 heading such as "LIVER:", a reference such as "this lesion" / "it" -> resolve it).
                 If the report names no structure for the fact, use null. Do NOT fill it from medical knowledge
                 ("No pneumothorax" -> anatomy null, not "pleural space").
  "observation": what the radiologist sees, 1-4 words, lower-case. Never "present", "absent", "seen", "noted"
                 or a whole sentence -- whether it is there or not goes in "certainty". Severity, size and extent
                 go in "modifier". An observation is one of three things:
                   a property of the structure:      "thickened", "dilated", "hypodense", "enhancing"
                   something found in / at it:       "air-fluid level", "free air", "thrombus", "stone",
                                                     "fluid collection", "catheter"
                   an overall verdict on it:         "normal", "unremarkable", "not visualized"
  "certainty":   "present" | "absent" | "possible"
                 absent = denied ("no free air"); possible = hedged ("may", "likely", "versus", "cannot be excluded").
  "change":      "new" | "worse" | "better" | "unchanged" | "resolved" | null
                 only when the text compares with before ("again seen" = unchanged, "no longer evident" = resolved,
                 "decreased" = better, "increased" = worse). A resolved finding has certainty "absent".
  "known":       true if stated as already known / history of / s/p / prior surgery ("known ulcerative colitis"), else false
  "finding_status": "abnormal" | "normal" | "not_evaluated" | "equivocal"
  "kind":        "observation" = what is seen; "interpretation" = the radiologist's reading of what it means
                 ("consistent with pancolitis", "likely due to collapse", "may represent a cyst")
  "based_on":    for an interpretation: ids of the observations it explains, e.g. ["F1", "F2"]; else []
  "repeats":     id of an earlier fact this one restates (typically the IMPRESSION repeating the FINDINGS), else null
  "modifier":    severity / size / extent / measurement if given ("mild", "diffuse", "8.5 mm", "large amount"), else null
  "section":     "findings" | "impression"
  "quote":       the shortest verbatim span of the report that supports the fact

RULES
1. One fact per (anatomy, observation). Split lists and compounds: "sigmoid wall thickening with adjacent fat
   stranding" -> {sigmoid colon wall, thickened} and {sigmoid colon surrounding fat, fat stranding}.
2. A normal sweep ("liver, spleen and adrenals are unremarkable") = one normal fact per structure.
   A negated list ("no free air, free fluid or abscess") = one absent fact per item.
3. Statements that only say whether something is there:
   "No pneumothorax."                  -> {anatomy null, observation "pneumothorax", certainty "absent"}
   "No air-fluid levels in the bowel." -> {anatomy "bowel", observation "air-fluid level", certainty "absent"}
   "A Foley catheter is present."      -> {anatomy null, observation "foley catheter", certainty "present"}
   "Loss of haustral markings in the left colon." -> {anatomy "left colon", observation "loss of haustral
                                           markings", certainty "present"}
4. Never emit the same content twice as observation and interpretation. An interpretation is a separate fact only
   when it adds a reading (a diagnosis, a cause, a differential), and it lists its observations in based_on.
   Example: "colon wall thickening and fat stranding consistent with pancolitis" ->
   F1 {colon wall, thickened, observation}, F2 {colon surrounding fat, fat stranding, observation},
   F3 {colon, pancolitis, interpretation, based_on ["F1","F2"]}.
   Example: "apparent thickening of the descending colon, likely due to collapse" ->
   F1 {descending colon wall, thickened, possible, observation},
   F2 {descending colon, collapse, possible, interpretation, based_on ["F1"]}.
5. When the IMPRESSION restates a FINDINGS fact, emit it again with section "impression" and repeats = that id.
6. A measurement is not its own fact: put it in "modifier" of the fact it measures.
7. Use ONLY what the text says. Do not infer unstated findings. Ignore technique, comparison and recommendations."""


# ---------------------------------------------------------------- sample
TRANSFER_SAMPLE = ROOT / "runs/ibd_service_transfer_patterns/sample.csv"


def sample_patients(per_kind=2):
    """every report of the 3 pilot admissions + the first `per_kind` admissions of each transfer type
    in the (already random) 14-case transfer sample."""
    t = pd.read_csv(TRANSFER_SAMPLE).groupby("kind", sort=False).head(per_kind)
    r = pd.read_parquet(RAW / "radiology", columns=["note_id", "hadm_id", "charttime", "exam_name", "text"])
    s = r[r.hadm_id.isin(PILOT + t.hadm_id.tolist())].reset_index(drop=True)
    s["finding_text"] = s.text.map(lambda t: "\n".join(c for k, c in sections(t or "") if k == "finding"))
    return s[s.finding_text.str.strip().ne("")]


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


def llm_one_v2(row, system=SYSTEM_V2, fields=FIELDS_V2, max_tokens=16000, tries=1):
    from experiments.annotation.annotate import call_json
    user = f"RADIOLOGY REPORT: {row.exam_name}\n\n{row.finding_text}"
    usage = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
    for attempt in range(tries):  # an empty result is usually a truncated / unparsable answer on a long report
        try:
            res = call_json(system, user, model=MODEL, temperature=1, max_tokens=max_tokens)
        except Exception as e:
            print(f"  {row.note_id} try {attempt + 1}: {type(e).__name__}: {e}")
            continue
        for k in usage:
            usage[k] += (res.get("usage") or {}).get(k) or 0
        if (res.get("parsed") or {}).get("facts"):
            break
        print(f"  {row.note_id} try {attempt + 1}: no facts parsed")
    else:
        return [], usage
    facts = []
    for f in (res.get("parsed") or {}).get("facts") or []:
        if not isinstance(f, dict):
            continue
        f = {k: f.get(k) for k in fields}
        f["anatomy"] = (str(f["anatomy"] or "")).strip().lower() or None
        if f["anatomy"] in ("null", "none", "n/a"):
            f["anatomy"] = None
        f["based_on"] = [str(x) for x in (f["based_on"] or [])] if isinstance(f["based_on"], list) else []
        f["known"] = bool(f["known"])
        for k, allowed in (("certainty", CERTAINTY), ("change", CHANGE), ("finding_status", FINDING_STATUS),
                           ("kind", KIND), ("section", SECTION)):
            v = str(f[k]).strip().lower() if f[k] not in (None, "") else None
            if v is not None and v not in allowed:
                f[k + "_raw"], v = f[k], None
            f[k] = v
        facts.append({"note_id": row.note_id, "hadm_id": int(row.hadm_id), "t": str(row.charttime), **f})
    return facts, usage


def llm_one_v3(row):
    return llm_one_v2(row, system=SYSTEM_V3, fields=FIELDS_V3, max_tokens=32000, tries=3)


def llm_facts(s, workers, one=llm_one):
    out, usage = [], []
    with ThreadPoolExecutor(workers) as ex:
        for facts, u in ex.map(one, s.itertuples()):
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
    ap.add_argument("--method", choices=["concept", "llm", "llm_v2", "llm_v3"])
    ap.add_argument("--patients", action="store_true", help="sample by patient (all reports), see sample_patients")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    s = sample_patients() if args.patients else sample(args.n_random)
    if args.method:
        (OUT / args.method).mkdir(parents=True, exist_ok=True)
        facts = (concept_facts(s) if args.method == "concept" else llm_facts(s, args.workers) if args.method == "llm"
                 else llm_facts(s, args.workers, one=llm_one_v2 if args.method == "llm_v2" else llm_one_v3))
        pd.DataFrame(facts).to_json(OUT / args.method / "facts.jsonl", orient="records", lines=True, force_ascii=False)
        print(f"{args.method}: {len(facts)} facts from {len(s)} reports")
    if args.compare:
        compare(s)


if __name__ == "__main__":
    main()
