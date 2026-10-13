"""Concept mentions -> per-sample concept-day counts + label-blind screening sheet.

Input  runs/term_concepts/mentions.parquet (term_concepts.py) + scripts/term_concept_overrides.csv.
Window as in transfer_recurrence.py: admit - 24 h up to the decision point t0, per sample
(a control admission matched to several cases appears once per match).

A feature = concept x polarity x section:
  'Pleural effusion [+]'  affirmed in findings/impression
  'Pleural effusion [-]'  negated   ('no pleural effusion')
  'Pleural effusion [?]'  uncertain / hypothetical ('possible ...', 'evaluate for ...')
  'Pleural effusion [hx]' historical
  'ask: Pleural effusion' named in INDICATION (the question asked), any polarity
  anatomy carries no polarity (always [+]): negation in its sentence is about the finding
Count = number of distinct days with >= 1 mention (same-day repeats collapse).

Outputs (runs/term_concepts/):
  sample_features.parquet  long table: sample, hadm_id, label, feature, days  (for the AUC step)
  samples.parquet          the 792 samples
  screen.csv / screen.md   label-blind sheet: `auto` = first pass (keep/maybe/drop), `kind` fact/action,
                           `note` action_trace; fill `final` by hand (kept across rebuilds)
"""
from pathlib import Path

import pandas as pd

from scripts.transfer_bow_model import H, build_samples

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/term_concepts"
POLARITY = {"affirmed": "+", "negated": "-", "uncertain": "?", "hypothetical": "?", "historical": "hx"}
MIN_SAMPLES = 20  # ~2.5% of samples

# ---- label-blind first pass (by concept name; applies to every polarity unless the feature is listed)
# findings / asks default keep, anatomy default maybe, devices keep; below are the exceptions
DROP_CONCEPTS = {  # template phrasing or generic words
    "heart size", "heart size normal", "Homogenous", "Abnormal", "Abnormality", "Symmetrical",
    "Small middle ear cavities", "Medical History", "Subcortical", "Parenchymal cell", "Pancreatic ductal cell",
    "Structure of lumen of body system",
    # bare imaging modality = 'compared with prior MRI / radiograph' -> which study, not the patient
    "Magnetic Resonance Imaging", "Plain x-ray", "Chest radiograph", "Ultrasonography", "X-Ray Computed Tomography",
    "Chest CT", "Diagnostic Imaging", "Radionuclide Imaging",
}
MAYBE_CONCEPTS = {  # course / severity words, generic or mostly-normal statements
    "No status change", "Improved", "physiologic resolution", "Reduced", "Worse", "Disease Progression",
    "Severe (severity modifier)", "Echogenic", "Contrast enhancement", "Abnormal degeneration",
    "Atrophic condition of skin", "Complication", "Lesion", "Lysis", "Sclerosis", "Implantation procedure",
    "removal technique",
}
DROP_FEATURES = {"No status change [-]", "ask: No status change", "ask: Severe (severity modifier)"}
# facts that exist because of an earlier intervention (kind stays 'fact'; flagged so analyses can exclude them)
ACTION_TRACE = {"Anastomosis", "Colectomy", "Surgical Clips", "Clip", "Stent, device"}


def first_pass(feature, concept, group, polarity):
    if feature in DROP_FEATURES or concept in DROP_CONCEPTS:
        return "drop"
    if concept in MAYBE_CONCEPTS or group == "anatomy":
        return "maybe"
    return "keep"


def kind_note(concept, group, polarity):
    kind = "action" if polarity == "ask" else "fact"
    trace = kind == "fact" and (group == "procedure_device" or concept in ACTION_TRACE)
    return kind, "action_trace" if trace else ""


def apply_overrides(m):
    ov = pd.read_csv(ROOT / "scripts/term_concept_overrides.csv", dtype=str).fillna("")
    m = m.copy()
    ml = m.mention.str.lower().str.strip()
    for o in ov.itertuples():
        hit = (m.cui == o.cui) & ((ml == o.mention) if o.mention else True)
        if o.new_cui == "DROP":
            m = m[~hit]
            ml = ml[~hit]
            continue
        if o.new_cui:
            m.loc[hit, ["cui", "concept"]] = [o.new_cui, o.new_concept]
        if o.new_group:
            m.loc[hit, "group"] = o.new_group
    return m


def feature_names(m):
    pol = m.status.map(POLARITY)
    m = m[pol.notna()].copy()  # drops 'family'
    pol = pol[pol.notna()]
    pol[m.group == "anatomy"] = "+"  # 'no osseous lesions' negates the lesion, not the bone
    m["feature"] = m.concept + " [" + pol + "]"
    m["polarity"] = pol
    ask = m.section == "indication"
    m.loc[ask, "feature"] = "ask: " + m.loc[ask, "concept"]
    m.loc[ask, "polarity"] = "ask"
    return m


def main():
    samples, _ = build_samples(3, 20261011)
    samples = samples[samples.hours >= 48].reset_index(drop=True)
    samples.to_parquet(OUT / "samples.parquet", index=False)

    m = feature_names(apply_overrides(pd.read_parquet(OUT / "mentions.parquet")))
    m["t"] = pd.to_datetime(m.t)
    m["day"] = m.t.dt.normalize()

    rows = []
    for i, s in enumerate(samples.itertuples()):
        w = m[(m.hadm_id == s.hadm_id) & (m.t >= s.admittime - 24 * H) & (m.t < s.t0)]
        for f, d in w.groupby("feature").day.nunique().items():
            rows.append((i, s.hadm_id, s.label, f, d))
    sf = pd.DataFrame(rows, columns=["sample", "hadm_id", "label", "feature", "days"])
    sf.to_parquet(OUT / "sample_features.parquet", index=False)

    # label-blind sheet: no case/control columns on purpose
    meta = m.drop_duplicates("feature").set_index("feature")[["concept", "group", "polarity"]]
    examples = (m.assign(ml=m.mention.str.lower().str.strip()).groupby("feature").ml
                .agg(lambda x: " | ".join(x.value_counts().head(3).index)))
    sheet = sf.groupby("feature").agg(samples_mentioning=("sample", "nunique"), total_days=("days", "sum"))
    sheet["days_per_mentioning_sample"] = (sheet.total_days / sheet.samples_mentioning).round(2)
    sheet = sheet.join(meta).join(examples.rename("example_mentions"))
    sheet = sheet[sheet.samples_mentioning >= MIN_SAMPLES].reset_index()
    sheet["auto"] = [first_pass(*r) for r in zip(sheet.feature, sheet.concept, sheet.group, sheet.polarity)]
    sheet[["kind", "note"]] = [kind_note(*r) for r in zip(sheet.concept, sheet.group, sheet.polarity)]
    old = OUT / "screen.csv"  # keep any hand edits in `final` across rebuilds
    prev = pd.read_csv(old).set_index("feature")["final"].dropna() if old.exists() else pd.Series(dtype=str)
    sheet["final"] = sheet.feature.map(prev).fillna("")
    sheet = sheet.sort_values(["group", "samples_mentioning"], ascending=[True, False])
    cols = ["feature", "group", "polarity", "kind", "note", "samples_mentioning", "days_per_mentioning_sample",
            "example_mentions", "auto", "final"]
    sheet[cols].to_csv(OUT / "screen.csv", index=False)

    lines = [f"# 概念初筛表（{len(samples)} 个样本；≥{MIN_SAMPLES} 个样本提到；不看标签）", "",
             "[+] 肯定 · [-] 否定 · [?] 不确定/待查 · [hx] 既往 · ask: 出现在检查原因里", ""]
    for (lab, g), d in sheet.groupby(["auto", "group"]):
        lines += [f"## {lab} · {g}（{len(d)}）", "",
                  " · ".join(f"{f}({n})" for f, n in zip(d.feature, d.samples_mentioning)), ""]
    (OUT / "screen.md").write_text("\n".join(lines))
    print(len(sf), "sample-feature rows;", sf.feature.nunique(), "features;", len(sheet), "on sheet")
    print(sheet.groupby(["group", "auto"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
