#!/usr/bin/env python3
"""Sameness ① 同义: different wordings of the same imaging observation / anatomy -> one canonical name.

Input  runs/atomic_facts/llm_v3/facts.jsonl (11 patients, 79 reports, 1,495 facts).
Labs / micro need no ① step: they are already one code per thing (lab itemid; micro specimen x test).

Each distinct raw string is mapped ONCE (lookup table, not per fact), so the same wording always lands
in the same place and the whole table can be reviewed line by line. Nothing is thrown away: the string is
split into slots, and how coarse "the same" is gets decided later, at counting time.

  anatomy string     -> structure (organ-level named structure) + part (sub-region) + side + type
                        "superior subsegment of the left lower lobe" -> lung | lower lobe superior segment | left
  observation string -> core (the finding itself, word-form variants merged) + site (site words written
                        inside the observation) + qualifier (size / severity / acuity, removed from core) + type
                        "small pleural effusions" -> effusion | pleural space | small

Site is taken only from words in the text (decision 2026-10-16); an observation without site words keeps
site empty -- its standard site is the ② organ table's job.

Method: Sonnet (OpenRouter), one call per batch of distinct strings, answers cached in llm_*.jsonl.
Manual corrections go in overrides.csv (kind, raw, field, value) and win over the model.

  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/sameness.py
Outputs runs/sameness/v1/: anatomy_map.csv, observation_map.csv, facts.jsonl (+ normalized columns), summary.md
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

FACTS = "runs/atomic_facts/llm_v3/facts.jsonl"
OUT = "runs/sameness/v1"
MODEL = "anthropic/claude-sonnet-4-6"
BATCH = 40
MERGE = "scripts/sameness_merge.csv"  # manual second pass, see manual_merge

ANAT_TYPES = {"anatomy", "region", "device", "finding", "other"}
OBS_TYPES = {"finding", "normal", "device", "procedure", "technical", "vague"}

COMMON = """You normalize phrases taken from radiology reports so that different wordings of the same thing
get exactly the same canonical name. Use standard radiology / anatomy terminology (RadLex or SNOMED CT
preferred terms), lowercase, singular, no articles. Be consistent across the whole batch: if two phrases
mean the same thing, give them the identical canonical string. Different things must stay different.
Each phrase comes with up to two report sentences for context. Return STRICT JSON:
{"items": [{"raw": <phrase exactly as given>, ...fields...}, ...]} with one item per phrase, same order."""

ANAT_SYSTEM = COMMON + """

The phrases are ANATOMY slots (where a statement is about). Fields:
  structure  the organ-level named structure: an organ, a named vessel, bone, joint, cavity / space, muscle,
             or lymph node group. Remove side and sub-region. Examples:
               "left lower lobe" -> "lung"; "pancreatic tail" -> "pancreas"; "sigmoid colon" -> "colon";
               "left internal jugular vein" -> "internal jugular vein"; "lung bases" -> "lung";
               "visualized colon" -> "colon"; "intraperitoneal" -> "peritoneal cavity";
               "segmental pulmonary artery of apical segment of right upper lobe" -> "pulmonary artery";
               "left cervical lymph nodes" -> "cervical lymph node"; "c6-7" -> "cervical spine"
  part       the sub-region or component inside the structure, canonical, or null. Keep named segments,
             lobes, walls, ducts, levels and proximal / distal / mid. Drop words that only describe what was
             seen in the image (visualized, imaged, partially, remainder of, all, dependent portion).
               "sigmoid colon" -> "sigmoid"; "pancreatic tail" / "tail of the pancreas" -> "tail";
               "lung bases" / "bibasilar lungs" -> "base"; "colonic wall" -> "wall"; "c6-7" -> "c6-c7"
  side       "left" / "right" / "bilateral" / null
  type       anatomy | region (broad body region: abdomen, pelvis, chest, groin, quadrant, extremity)
             | device (tube, line, catheter, drain, hardware, its tip) | finding (the phrase is really a
             finding, e.g. "pleural effusion", "lymphadenopathy"; put its literal site in structure)
             | other (measurement or not a place, e.g. "heart size", "lung volumes": put the structure measured)"""

OBS_SYSTEM = COMMON + """

The phrases are OBSERVATION slots (what was seen, without present / absent: certainty is stored elsewhere).
Fields:
  core       the finding itself as a canonical noun phrase. Merge word-form variants into one noun:
               thickened / thickening / wall thickening -> "thickening"; dilated / dilation / dilatation /
               ductal dilatation -> "dilatation"; emboli / embolus / embolism -> "embolism";
               stones / calculi / gallstones / cholelithiasis -> "calculus"; free air / pneumoperitoneum /
               free intraperitoneal air -> "pneumoperitoneum"; stranding / fat stranding -> "fat stranding".
             Remove site words, side, size, severity, acuity, number and certainty words ("acute fracture"
             -> "fracture"; "small pleural effusions" -> "effusion"; "pathologically enlarged" -> "enlargement").
             Generic normal words (normal, unremarkable, clear, within normal limits, intact, preserved,
             normal appearance, normal study) -> "normal". A normal statement about one property keeps the
             property ("normal caliber", "normal size", "normal flow", "normal waveform"). Keep
             test-specific signs as their own name ("compressible", "patent", "triphasic waveform").
             Keep single-word disease or sign names whole when no site word is written
             (pneumothorax, atelectasis, ascites, cardiomegaly, pneumonia, colitis, hydronephrosis).
             Distinct diseases stay distinct ("ulcerative colitis" is not "colitis").
  site       the structure named by site words WRITTEN in the phrase, as an organ-level canonical structure
             (same style as above: "pleural effusion" -> "pleural space", "pulmonary embolus" ->
             "pulmonary artery", "mesenteric lymphadenopathy" -> "mesentery", "renal stones" -> "kidney",
             "colonic perforation" -> "colon", "intrahepatic biliary dilatation" -> "bile duct"), or null.
             Never add a site that is not written ("pneumothorax" -> null, "ascites" -> null).
  qualifier  the removed size / severity / acuity / number words joined by spaces, or null
  type       finding (an abnormality or disease) | normal (states something is normal / absent-type normal
             descriptor) | device (tube, line, catheter, hardware, their position) | procedure (post-operative
             state, surgery done, drain placed, paracentesis) | technical (exam quality or report meta:
             not visualized, difficult to assess, excluded from the image, discussion) | vague (too unspecific
             to be a finding on its own: abnormality, involvement, appearance, size, increase)"""


def load_facts():
    df = pd.read_json(FACTS, lines=True)
    df["anatomy_raw"] = df.anatomy.fillna("").str.strip().str.lower()
    df["observation_raw"] = df.observation.fillna("").str.strip().str.lower()
    return df


def distinct(df, col):
    g = df[df[col] != ""].groupby(col)
    return pd.DataFrame({"n": g.size(), "examples": g.quote.apply(lambda q: list(dict.fromkeys(q))[:2])}) \
        .sort_values("n", ascending=False).reset_index().rename(columns={col: "raw"})


def ask(system, batch):
    from experiments.annotation.annotate import call_json
    user = "\n\n".join(f"PHRASE: {r.raw}\n" + "\n".join(f"  context: {q}" for q in r.examples)
                       for r in batch.itertuples())
    for attempt in range(3):
        try:
            res = call_json(system, user, model=MODEL, temperature=0, max_tokens=16000)
        except Exception as e:
            print(f"  try {attempt + 1}: {type(e).__name__}: {e}")
            continue
        items = (res.get("parsed") or {}).get("items") or []
        got = {str(i.get("raw", "")).strip().lower(): i for i in items if isinstance(i, dict)}
        if all(r in got for r in batch.raw):
            return [got[r] | {"raw": r} for r in batch.raw], (res.get("usage") or {}).get("cost") or 0
        print(f"  try {attempt + 1}: {sum(r not in got for r in batch.raw)} phrases missing")
    return [got.get(r, {}) | {"raw": r} for r in batch.raw], 0  # partial: missing ones stay unmapped


def map_strings(kind, system, table):
    """Cached: only strings without an answer in llm_{kind}.jsonl are sent."""
    cache_path = f"{OUT}/llm_{kind}.jsonl"
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            r = json.loads(line)
            cache[r["raw"]] = r
    todo = table[~table.raw.isin(cache)]
    if len(todo):
        batches = [todo.iloc[i:i + BATCH] for i in range(0, len(todo), BATCH)]
        cost = 0.0
        with ThreadPoolExecutor(6) as ex, open(cache_path, "a") as f:
            for items, c in ex.map(lambda b: ask(system, b), batches):
                cost += c
                for it in items:
                    if len(it) > 1:
                        f.write(json.dumps(it) + "\n")
                        cache[it["raw"]] = it
        print(f"{kind}: {len(todo)} strings sent, ${cost:.2f}")
    return cache


def apply_overrides(kind, m):
    path = f"{OUT}/overrides.csv"
    if not os.path.exists(path):
        return m
    o = pd.read_csv(path).query("kind == @kind")
    for r in o.itertuples():
        m.loc[m.raw == r.raw, r.field] = None if pd.isna(r.value) else r.value
    return m


def build_map(kind, system, table, fields, types):
    cache = map_strings(kind, system, table)
    m = table[["raw", "n"]].copy()
    for f in fields:
        m[f] = [(str(cache.get(r, {}).get(f) or "").strip().lower() or None) for r in m.raw]
    bad = ~m.type.isin(types)
    m.loc[bad, "type"] = None
    m = apply_overrides(kind, m)
    m.to_csv(f"{OUT}/{kind}_map.csv", index=False)
    return m


def manual_merge(am, om):
    """Second pass, by hand: batches were mapped in parallel and could not see each other, so the same meaning
    sometimes got two names. scripts/sameness_merge.csv merges them (class -> class), fixes single wrong
    strings (raw -> class); the model's name is kept in *_llm for comparison."""
    mg = pd.read_csv(MERGE)
    for m, kind, col in ((am, "anatomy", "structure"), (om, "observation", "core")):
        m[col + "_llm"] = m[col]
        for r in mg[(mg.kind == kind) & (mg.level == "class")].itertuples():
            m.loc[m[col + "_llm"] == r[3], col] = r.to
        for r in mg[(mg.kind == kind) & (mg.level == "raw")].itertuples():
            m.loc[m.raw == r[3], col] = r.to
    return am, om


def pair_merge(df):
    """Same finding split over the two slots: 'cervical lymph node' + 'enlarged' = 'lymphadenopathy'."""
    mg = pd.read_csv(MERGE).query("level == 'pair'")
    for r in mg.itertuples():
        structure, core = r[3].split("|")
        hit = df.anat_structure.fillna("").str.contains(structure) & (df.obs_core == core)
        df.loc[hit, "obs_core"] = r.to
    return df


def singular(s):
    return " ".join(w[:-1] if w.endswith("s") and not w.endswith(("ss", "us", "is", "ites")) else w for w in s.split())


def keep_negated_qualifier(df):
    """Under negation every qualifier narrows what is denied: 'no acute fracture' is not 'no fracture'.
    The string map cannot see certainty (one wording serves both 'acute fracture' and 'no acute fracture'),
    so absent facts with a removed qualifier get their written wording back (plural only merged)."""
    hit = (df.certainty == "absent") & df.obs_qualifier.notna()
    df["obs_core_merged"] = df.obs_core
    df.loc[hit, "obs_core"] = df.loc[hit, "observation_raw"].map(singular)
    return df


def summary(df, am, om):
    def groups(m, key, k=25):
        g = m[m[key].notna()].groupby(key).agg(n=("n", "sum"), variants=("raw", list))
        g = g[g.variants.str.len() > 1].sort_values("n", ascending=False).head(k)
        return "\n".join(f"| {i} | {r.n} | {' · '.join(r.variants[:8])}{' …' if len(r.variants) > 8 else ''} |"
                         for i, r in g.iterrows())

    has_a = df.anatomy_raw != ""
    obs_site = df.obs_site.notna()
    lines = [
        "# Sameness ① 同义 — imaging, 11 patients",
        "",
        f"Facts {len(df):,}. Distinct strings before → after (canonical):",
        "",
        "| slot | raw | canonical | merged away |",
        "|---|---|---|---|",
        f"| observation → core | {om.raw.nunique()} | {om.core.nunique()} | {1 - om.core.nunique() / om.raw.nunique():.0%} |",
        f"| anatomy → structure | {am.raw.nunique()} | {am.structure.nunique()} | {1 - am.structure.nunique() / am.raw.nunique():.0%} |",
        f"| anatomy → structure + part + side | {am.raw.nunique()} | {am[['structure', 'part', 'side']].astype(str).agg('|'.join, axis=1).nunique()} | |",
        "",
        f"Site: anatomy slot filled {has_a.mean():.0%}; site words inside the observation add {(~has_a & obs_site).sum()} facts;"
        f" still no site {(~has_a & ~obs_site).sum()} (→ ② table).",
        "",
        "Observation type (facts): " + ", ".join(f"{k} {v}" for k, v in df.obs_type.value_counts(dropna=False).items()),
        "",
        "Anatomy type (facts with anatomy): " + ", ".join(f"{k} {v}" for k, v in df[has_a].anat_type.value_counts(dropna=False).items()),
        "",
        "## Largest merged observation groups", "", "| core | facts | variants |", "|---|---|---|", groups(om, "core"),
        "", "## Largest merged anatomy groups", "", "| structure | facts | variants |", "|---|---|---|", groups(am, "structure"),
        "", "Full tables: `anatomy_map.csv`, `observation_map.csv` (sorted by canonical name for review).",
        "Corrections: add rows to `overrides.csv` (kind = anatomy / observation, raw, field, value) and re-run.",
    ]
    open(f"{OUT}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = load_facts()
    am = build_map("anatomy", ANAT_SYSTEM, distinct(df, "anatomy_raw"), ["structure", "part", "side", "type"], ANAT_TYPES)
    om = build_map("observation", OBS_SYSTEM, distinct(df, "observation_raw"), ["core", "site", "qualifier", "type"], OBS_TYPES)
    am, om = manual_merge(am, om)

    df = df.merge(am.drop(columns="n").add_prefix("anat_").rename(columns={"anat_raw": "anatomy_raw"}),
                  on="anatomy_raw", how="left")
    df = df.merge(om.drop(columns="n").add_prefix("obs_").rename(columns={"obs_raw": "observation_raw"}),
                  on="observation_raw", how="left")
    df = keep_negated_qualifier(pair_merge(df))
    df.to_json(f"{OUT}/facts.jsonl", orient="records", lines=True, force_ascii=False)
    # review order: canonical groups together
    am.sort_values(["structure", "part", "side"]).to_csv(f"{OUT}/anatomy_map.csv", index=False)
    om.sort_values(["core", "site"]).to_csv(f"{OUT}/observation_map.csv", index=False)
    summary(df, am, om)


if __name__ == "__main__":
    main()
