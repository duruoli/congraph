#!/usr/bin/env python3
"""Atomic fact units for the density terrain, in two text granularities: word and concept.

A unit is one atomic observation; nothing is merged here (no synonym, organ or
problem grouping -- those are later, separate layers):

  lab      one lab result            unit = lab:<itemid>          status = high / low / abnormal / normal
                                     dev = distance outside the reference range in range widths (+ high, - low)
  micro    one specimen x test       unit = micro:<specimen>|<test>  status = positive / negative, detail = organism
           (cancelled / not performed / redacted results dropped: not observations)
  imaging  one mention in the FINDINGS / IMPRESSION part of a report:
             --unit word     unit = w:<word>   (stop words dropped)   status by a clause-scope rule
             --unit concept  unit = c:<CUI>    (scispaCy + UMLS + radiology lexicon, see term_concepts.py)
                                                status from medspaCy ConText
           status = affirmed / negated / uncertain (/ historical / hypothetical / family for concepts)

Labs and micro are coded data and identical in both versions; only imaging text differs.
Time = when the result became available (lab/micro: charttime; imaging: storetime, else charttime).
day = -1 before admission, 0 = first 24 h, ...  sent = hash of the normalised sentence, for copy detection.
Cohort = primary-IBD admissions (the sea-level population). Restricted MIMIC-derived output: runs/ only.

  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/terrain_units.py --unit word
  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/terrain_units.py --unit concept
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.term_concepts import HEDGE_BACKWARD, HEDGE_FORWARD, sections
from scripts.transfer_bow_model import STOP

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw_data/ibd_mimiciv_3_1"
OUT = ROOT / "runs/terrain_units"
CONCEPT_CACHE = ROOT / "runs/term_concepts"  # shared with term_concepts.py

NEG_CUE = re.compile(r"\b(no|not|without|negative for|free of|absence of|resolved|resolution of|rather than)\b", re.I)
HEDGE_CUE = re.compile(r"\b(" + "|".join(re.escape(w) for w in HEDGE_FORWARD + ["possible", "possibly"]) + r")\b", re.I)
POST_NEG = re.compile(r"\b(no longer|has resolved|have resolved|not (seen|identified|evident|present))\b", re.I)
POST_HEDGE = re.compile(r"\b(" + "|".join(re.escape(w) for w in HEDGE_BACKWARD) + r")\b", re.I)
BREAK = re.compile(r";|:|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\bwhich\b|\bexcept\b|\bwith\b", re.I)
CUE_WORDS = {"no", "not", "nor", "without", "none", "never", "neither", "cannot", "possible", "possibly", "likely",
             "may", "might", "could", "probably", "suggest", "suggests", "suggesting", "suggestive", "versus", "vs"}
WORD = re.compile(r"\b[a-z][a-z]+\b")
STOPSET = set(STOP) | CUE_WORDS


def cohort():
    c = pd.read_parquet(RAW / "cohort")
    return c[c.is_primary_ibd_diagnosis][["hadm_id", "admittime"]].drop_duplicates("hadm_id")


def norm_sentence(s):
    return hashlib.md5(" ".join(WORD.findall(s.lower())).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- coded sources
def lab_units(hadms):
    l = pd.read_parquet(RAW / "labs", columns=["hadm_id", "itemid", "label", "fluid", "charttime", "valuenum",
                                                "ref_range_lower", "ref_range_upper", "flag"])
    l = l[l.hadm_id.isin(hadms)]
    hi = l.valuenum > l.ref_range_upper
    lo = l.valuenum < l.ref_range_lower
    ab = l.flag.eq("abnormal")
    status = np.select([ab & hi, ab & lo, ab], ["high", "low", "abnormal"], "normal")
    name = l.label.where(l.fluid.eq("Blood"), l.label + " (" + l.fluid.fillna("?") + ")")
    # how far outside the reference range, in range widths (+ above, - below), to pick the worst result in a window
    width = (l.ref_range_upper - l.ref_range_lower).where(lambda w: w > 0)
    dev = np.select([hi, lo], [(l.valuenum - l.ref_range_upper) / width, (l.valuenum - l.ref_range_lower) / width], 0.0)
    return pd.DataFrame({"hadm_id": l.hadm_id, "t": l.charttime, "source": "lab", "unit": "lab:" + l.itemid.astype(str),
                         "name": name, "status": status, "group": "lab", "sent": None, "dev": dev, "detail": None})


def micro_units(hadms):
    m = pd.read_parquet(RAW / "microbiology", columns=["hadm_id", "charttime", "chartdate", "spec_type_desc",
                                                        "test_name", "org_name", "comments"])
    m = m[m.hadm_id.isin(hadms)].assign(t=lambda d: d.charttime.fillna(d.chartdate))
    com = m.comments.fillna("").str.strip()
    # not an observation: cancelled / not performed ("CANCELLED" sits in org_name), or the result is redacted
    gone = (m.org_name.eq("CANCELLED") | com.str.match(r"TEST (CANCELLED|NOT PERFORMED)", case=False)
            | (m.org_name.isna() & com.eq("___")))
    m, com = m[~gone], com[~gone]
    # without an organism the result is the comment's first sentence; later ones are method boilerplate
    # ("most cases of Giardia are detected by routine O+P" on a negative O+P)
    first = com.str.split(r"\.\s", n=1, regex=True).str[0]
    said = (first.str.contains(r"\bpositive\b|\bdetected\b|\breactive\b", case=False)
            & ~first.str.contains(r"negative|not detected|\bno\b", case=False))
    gram = m.test_name.str.contains("gram stain", case=False, na=False) & com.str.contains(
        r"(?<!NO )GRAM[- ](?:POSITIVE|NEGATIVE|VARIABLE)", case=False)  # organisms seen on the stain
    pos = m.org_name.notna() | said | gram
    # one unit per specimen x TEST, whatever the result: keyed by organism, a positive and a negative of the
    # same test became two units with one outcome each. The organism goes to `detail`.
    spec, test = m.spec_type_desc.fillna("?"), m.test_name.fillna("?")
    u = pd.DataFrame({"hadm_id": m.hadm_id, "t": m.t, "source": "micro", "unit": "micro:" + spec + "|" + test,
                      "name": spec + " · " + test, "status": np.where(pos, "positive", "negative"),
                      "group": "micro", "sent": None, "dev": np.nan, "detail": m.org_name})
    # antibiotic-sensitivity rows repeat the same specimen x organism
    return u.drop_duplicates(["hadm_id", "t", "unit", "status", "detail"])


# ---------------------------------------------------------------- imaging text
def reports(hadms):
    r = pd.read_parquet(RAW / "radiology", columns=["note_id", "hadm_id", "charttime", "storetime", "text"])
    r = r[r.hadm_id.isin(hadms)]
    return r.assign(t=r.storetime.fillna(r.charttime))


def split_sentences(chunk):
    parts = re.split(r"(?<=[.])\s+|\n\s*\d+\.\s+|\n{2,}", chunk)
    return [" ".join(p.split()) for p in parts if len(p.strip()) > 3]


def word_status(clause_before, clause_after):
    if NEG_CUE.search(clause_before) or POST_NEG.search(clause_after):
        return "negated"
    if HEDGE_CUE.search(clause_before) or POST_HEDGE.search(clause_after):
        return "uncertain"
    return "affirmed"


def word_units(hadms):
    rows = []
    for r in reports(hadms).itertuples():
        for kind, chunk in sections(r.text or ""):
            if kind != "finding":
                continue
            for s in split_sentences(chunk):
                sid, low = norm_sentence(s), s.lower()
                for m in WORD.finditer(low):
                    w = m.group()
                    if w in STOPSET:
                        continue
                    st = word_status(BREAK.split(low[: m.start()])[-1], BREAK.split(low[m.end():])[0])
                    rows.append((r.hadm_id, r.t, "imaging", "w:" + w, w, st, "word", sid))
    u = pd.DataFrame(rows, columns=["hadm_id", "t", "source", "unit", "name", "status", "group", "sent"])
    return u.drop_duplicates()  # same word, same sentence, same status = one mention


def concept_units(hadms):
    from scripts.term_concepts import build_nlp, extract
    rep = reports(hadms)
    cache, done_file = CONCEPT_CACHE / "mentions.parquet", CONCEPT_CACHE / "extracted_notes.parquet"
    done = set(pd.read_parquet(done_file).note_id) if done_file.exists() else set()
    new = rep[~rep.note_id.isin(done)]
    if len(new):
        print(f"extracting concepts from {len(new)} new reports ...")
        m = extract(build_nlp(), new[["note_id", "hadm_id", "t", "text"]])
        if cache.exists():
            m = pd.concat([pd.read_parquet(cache), m], ignore_index=True)
        m.to_parquet(cache, index=False)
        pd.DataFrame({"note_id": sorted(done | set(new.note_id))}).to_parquet(done_file, index=False)
    m = pd.read_parquet(cache)
    m = m[m.note_id.isin(rep.note_id) & m.section.eq("finding")]
    return pd.DataFrame({"hadm_id": m.hadm_id, "t": m.t, "source": "imaging", "unit": "c:" + m.cui,
                         "name": m.concept, "status": m.status, "group": m.group,
                         "sent": m.sentence.map(norm_sentence)}).drop_duplicates()


# ---------------------------------------------------------------- summary
def summary(u, unit):
    lines = [f"# Atomic units — imaging as **{unit}**", "",
             "| source | rows | distinct units | admissions |", "|---|---:|---:|---:|"]
    for s, g in u.groupby("source"):
        lines.append(f"| {s} | {len(g):,} | {g.unit.nunique():,} | {g.hadm_id.nunique():,} |")
    lines += ["", "## status", "", "| source | status | rows |", "|---|---|---:|"]
    lines += [f"| {s} | {st} | {n:,} |" for (s, st), n in u.groupby(["source", "status"]).size().items()]
    for s, keep in (("imaging", "affirmed"), ("lab", None), ("micro", "positive")):
        g = u[u.source.eq(s)]
        g = g[g.status.eq(keep)] if keep else g[g.status.ne("normal")]
        top = g.groupby(["unit", "name"]).hadm_id.nunique().sort_values(ascending=False).head(30)
        lines += ["", f"## top 30 {s} units ({keep or 'abnormal'}) by admissions", "", "| unit | name | admissions |",
                  "|---|---|---:|"] + [f"| `{a}` | {b} | {n:,} |" for (a, b), n in top.items()]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", choices=["word", "concept"], required=True)
    args = ap.parse_args()
    out = OUT / args.unit
    out.mkdir(parents=True, exist_ok=True)
    coh = cohort()
    hadms = set(coh.hadm_id)
    img = word_units(hadms) if args.unit == "word" else concept_units(hadms)
    u = pd.concat([lab_units(hadms), micro_units(hadms), img], ignore_index=True).merge(coh, on="hadm_id")
    u["day"] = ((u.t - u.admittime).dt.total_seconds() // 86400).clip(lower=-1).astype("Int64")
    u = u.drop(columns="admittime").sort_values(["hadm_id", "t"]).reset_index(drop=True)
    u.to_parquet(out / "units.parquet", index=False)
    (out / "summary.md").write_text(summary(u, args.unit))
    print(f"{args.unit}: {len(u):,} unit rows, {u.unit.nunique():,} distinct units, "
          f"{u.hadm_id.nunique():,} admissions -> {out}")


if __name__ == "__main__":
    main()
