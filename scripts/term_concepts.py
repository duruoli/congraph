"""Radiology text -> UMLS concepts with assertion status (step 'thing = concept').

Replaces the raw unigram/bigram split: scispaCy finds medical mentions and links each
to a UMLS concept (CUI), so synonyms merge ("free air" / "pneumoperitoneum" -> one CUI);
medspaCy ConText marks each mention negated / uncertain / hypothetical / historical.

Sections: FINDINGS (+ organ subsections) and IMPRESSION = 'finding'; INDICATION/HISTORY =
'indication' (the question asked); EXAMINATION/TECHNIQUE/DOSE/COMPARISON/... dropped.

Runs in the congraph env (scispacy/medspacy live there):
  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/term_concepts.py   (cohort)
  ... scripts/term_concepts.py --pilot 30                                               (eyeball)
First run downloads the scispaCy UMLS KB (~1 GB, cached in ~/.scispacy).
"""
import argparse
import random
import re
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw_data/ibd_mimiciv_3_1"
HEADER = re.compile(r"(?m)^[ \t]*([A-Z][A-Z /&]{2,40}):")
FINDING_START = {"FINDINGS", "IMPRESSION"}
INDICATION = {"INDICATION", "HISTORY", "CLINICAL HISTORY", "REASON FOR EXAM"}
# UMLS semantic types kept, grouped; everything else (spatial, temporal, qualitative, people, genes...) dropped
GROUPS = {
    "finding": {"T019", "T020", "T033", "T034", "T037", "T046", "T047", "T049", "T184", "T190", "T191"},
    "anatomy": {"T017", "T018", "T022", "T023", "T024", "T025", "T026", "T029", "T030"},
    "procedure_device": {"T059", "T060", "T061", "T074"},
}
TUI_GROUP = {t: g for g, ts in GROUPS.items() for t in ts}
NEG_PREFIX = re.compile(r"^(no|not|without)\b", re.I)  # ConText misses negation swallowed into the span
LIST_WORDS, LIST_BREAK = {"or", "and", "nor"}, {"but", "however", "although", "though", "which", "with", "except",
                                                 "is", "are", "was", "were", "there"}
# radiology hedges missing from ConText's defaults (it only has 'possible'); 'consistent with' stays affirmed
HEDGE_FORWARD = ["could", "may", "might", "probably", "likely", "suggest", "suggests", "suggesting", "suggestive of",
                 "concerning for", "worrisome for", "compatible with", "questionable", "cannot exclude",
                 "can not exclude", "differential includes", "versus", "vs"]
HEDGE_BACKWARD = ["cannot be excluded", "can not be excluded", "not excluded", "is not excluded", "not ruled out",
                  "is possible", "are possible", "cannot be ruled out"]
NON_FINDING = {"EXAMINATION", "TECHNIQUE", "DOSE", "COMPARISON", "NOTIFICATION", "PROCEDURE", "EXAM",
               "RECOMMENDATION", "RECOMMENDATIONS", "ADDENDUM"}


def sections(text):
    """[(section_kind, chunk)] with kind in {'finding', 'indication'}; organ subheaders
    (LIVER:, PANCREAS:, ...) inherit 'finding' once FINDINGS/IMPRESSION has started."""
    out, kind, pos = [], None, 0
    for m in list(HEADER.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        if kind and text[pos:end].strip():
            out.append((kind, text[pos:end]))
        if m is None:
            break
        h = m.group(1).strip()
        if h in FINDING_START:
            kind = "finding"
        elif h in INDICATION:
            kind = "indication"
        elif h in NON_FINDING:
            kind = None
        pos = m.end()  # other headers (organ names) keep the current kind
    # MIMIC hard-wraps at ~80 chars: rejoin single line breaks so sentences (ConText's scope) stay whole
    return [(k, re.sub(r"(?<!\n)\n(?!\n)", " ", c)) for k, c in out]


def build_nlp():
    import medspacy  # noqa: F401  registers medspacy_context
    import scispacy.linking  # noqa: F401  registers scispacy_linker
    import spacy
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": False, "linker_name": "umls",
                                            "threshold": 0.85, "max_entities_per_mention": 1,
                                            "filter_for_definitions": False})
    from medspacy.context import ConTextRule
    ctx = nlp.add_pipe("medspacy_context")
    ctx.add([ConTextRule(w, "POSSIBLE_EXISTENCE", direction="FORWARD") for w in HEDGE_FORWARD]
            + [ConTextRule(w, "POSSIBLE_EXISTENCE", direction="BACKWARD") for w in HEDGE_BACKWARD])
    return nlp


def list_continues(gap):
    """Text between two mentions is just a list joint (', ', ' or ', ' air or ' after a clipped span)."""
    words = set(re.findall(r"[a-z]+", gap.lower()))
    return len(re.findall(r"[a-z]+", gap.lower())) <= 3 and ("," in gap or words & LIST_WORDS) \
        and not words & LIST_BREAK


def status(ent):
    for attr in ("is_negated", "is_hypothetical", "is_uncertain", "is_historical", "is_family"):
        if getattr(ent._, attr):
            return attr[3:]
    return "affirmed"


def extract(nlp, reports):
    """reports: DataFrame with note_id, hadm_id, t (available time), text. One row per linked mention."""
    kb = nlp.get_pipe("scispacy_linker").kb
    rows, chunks = [], [(r.note_id, r.hadm_id, r.t, kind, chunk)
                        for r in reports.itertuples() for kind, chunk in sections(r.text or "")]
    for (note_id, hadm_id, t, kind, _), doc in zip(chunks, nlp.pipe((c[-1] for c in chunks), batch_size=64)):
        prev = None  # (status, end_char) of the previous mention, linked or not
        for ent in doc.ents:
            st = "negated" if NEG_PREFIX.match(ent.text) else status(ent)
            # "No free air or free fluid": the 'No' was swallowed into the first span, so ConText
            # misses it for the rest of the list -> inherit negation across list connectors
            if st == "affirmed" and prev and prev[0] == "negated" and list_continues(doc.text[prev[1]:ent.start_char]):
                st = "negated"
            prev = (st, ent.end_char)
            if not ent._.kb_ents:
                continue
            cui, score = ent._.kb_ents[0]
            c = kb.cui_to_entity[cui]
            group = next((TUI_GROUP[t] for t in c.types if t in TUI_GROUP), None)
            if group is None:
                continue
            rows.append((note_id, hadm_id, t, kind, ent.text, cui, c.canonical_name, group, ",".join(c.types),
                         round(score, 2), st, ent.sent.text.strip()))
    return pd.DataFrame(rows, columns=["note_id", "hadm_id", "t", "section", "mention", "cui",
                                       "concept", "group", "tuis", "score", "status", "sentence"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=int, default=0, help="N random reports from all IBD admissions")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/term_concepts")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rad = ds.dataset(SOURCE / "radiology", format="parquet").to_table(
        columns=["note_id", "hadm_id", "charttime", "storetime", "exam_name", "text"]).to_pandas()
    rad["t"] = rad.storetime.fillna(rad.charttime)  # same availability time as transfer_bow_model.events
    nlp = build_nlp()
    if not args.pilot:  # all reports of the transfer-study admissions; windowing happens at aggregation
        from scripts.transfer_bow_model import build_samples
        samples, _ = build_samples(3, 20261011)
        reports = rad[rad.hadm_id.isin(set(samples[samples.hours >= 48].hadm_id))]
        m = extract(nlp, reports)
        m.to_parquet(args.out / "mentions.parquet", index=False)
        print(len(reports), "reports;", len(m), "mentions;", m.cui.nunique(), "concepts")
        print(m.status.value_counts().to_string())
        return

    reports = rad.iloc[random.Random(0).sample(range(len(rad)), args.pilot)]
    m = extract(nlp, reports)
    m.to_csv(args.out / "pilot_mentions.csv", index=False)

    # side-by-side view for eyeballing: report text, then what was extracted from it
    lines = [f"# 概念抽取试跑（{args.pilot} 份报告）", ""]
    for r in reports.head(6).itertuples():
        lines += [f"## {r.exam_name}（note {r.note_id}）", "", "```", (r.text or "")[:3000], "```", "",
                  "| 节 | 原文 | → 概念 | 组 | 状态 |", "|---|---|---|---|---|"]
        lines += [f"| {x.section} | {x.mention} | {x.concept} | {x.group} | {x.status} |"
                  for x in m[m.note_id == r.note_id].itertuples()]
        lines.append("")
    (args.out / "pilot.md").write_text("\n".join(lines))
    print(len(m), "mentions;", m.cui.nunique(), "concepts")
    print(m.status.value_counts().to_string())


if __name__ == "__main__":
    main()
