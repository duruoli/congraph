#!/usr/bin/env python3
"""Does adding ruled-out problems to s make sense? A small check (BAYES_ENV.md 7.10).

Step 6 found stool / infection tests ordered often although their EV for the 10 discharge problems is ~0:
they are almost always negative, and a negative leaves no discharge code. Hypothesis: they rule out an
infection BEFORE escalating immunosuppression, so "infection" belongs in s (a problem that is mostly ruled
out) and its value is the treatment it unlocks. Checked here:

  1 labels    how many admissions are positive (micro result, or an ICD code), per infection group
  2 who       testing rate and positivity among the tested
  3 gate      escalation (first IV steroid / biologic / cyclosporine): are the tests sent before the first
              dose, and is the dose given before or after the result is back?

Outputs runs/bayes_env/v0/ruleout/summary.md

  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/bayes_ruleout_check.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.bayes_data import OUT as V0, RAW, cohort

OUT = V0 / "ruleout"
GROUPS = {  # micro tests by what they rule out
    "enteric": r"FECAL CULTURE|CAMPYLOBACTER|OVA \+ PARASITES|Cryptosporidium|CYCLOSPORA|MICROSPORIDIA|SHIGA",
    "cdiff": r"C\. difficile|CLOSTRIDIUM DIFFICILE",
    "cmv": r"CMV",
}
ICD = {  # prefixes: (ICD-9, ICD-10)
    "enteric": (("001", "002", "003", "004", "005", "006", "007", "0080", "0081", "0082", "0083", "0084", "0085",
                 "0086", "0088", "009"), ("A00", "A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09")),
    "cdiff": (("00845",), ("A047",)),
    "cmv": (("0785",), ("B25",)),
}
ESCALATE = (r"methylpred.*succ|hydrocortisone na succ|infliximab|cyclosporine|ustekinumab|vedolizumab")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coh = cohort()[["hadm_id", "admittime"]]
    n = len(coh)

    m = pd.read_parquet(RAW / "microbiology")
    m = m[m.hadm_id.isin(coh.hadm_id)].assign(t=lambda d: d.charttime.fillna(d.chartdate))
    com = m.comments.fillna("")
    gone = m.org_name.eq("CANCELLED") | com.str.match(r"TEST (CANCELLED|NOT PERFORMED)", case=False)
    m = m[~gone]
    first = com[~gone].str.split(r"\.\s", n=1, regex=True).str[0]
    pos = m.org_name.notna() | (first.str.contains(r"\bpositive\b|\bdetected\b|\breactive\b", case=False)
                                & ~first.str.contains(r"negative|not detected|\bno\b", case=False))
    m = m.assign(pos=pos.values)
    # CMV "detected" in a viral-load comment counts; the CMV IgG of a past infection does not
    m.loc[m.test_name.str.contains("IgG", na=False), "pos"] = False

    d = pd.read_parquet(RAW / "diagnoses")
    d = d[d.hadm_id.isin(coh.hadm_id)]
    p = pd.read_parquet(RAW / "prescriptions", columns=["hadm_id", "drug", "route", "starttime"])
    p = p[p.hadm_id.isin(coh.hadm_id) & p.drug.str.contains(ESCALATE, case=False, na=False)
          & p.route.str.contains("IV", na=False)]
    esc = p.groupby("hadm_id").starttime.min()

    L = ["# Rule-out check: does 'infection' belong in s?", "",
         f"Primary-IBD admissions n = {n:,}; escalated (first IV steroid / infliximab / cyclosporine / ustekinumab / "
         f"vedolizumab) {len(esc):,} ({len(esc) / n:.0%}).", "",
         "## 1-2. labels, testing, positivity", "",
         "| group | admissions tested | positive among tested | positive (micro) | ICD-coded | either |",
         "|---|---:|---:|---:|---:|---:|"]
    gate = []
    for g, pat in GROUPS.items():
        mg = m[m.test_name.str.contains(pat, case=False, na=False)]
        tested = mg.hadm_id.unique()
        pos_adm = mg[mg.pos].hadm_id.unique()
        i9, i10 = ICD[g]
        hit = (d.icd_version.eq(9) & d.icd_code.str.startswith(i9)) | (d.icd_version.eq(10) & d.icd_code.str.startswith(i10))
        if g == "enteric":  # A04 / 0084 also hold the C. diff codes, which are their own group
            hit &= ~d.icd_code.str.startswith(ICD["cdiff"][0] + ICD["cdiff"][1])
        icd = d[hit].hadm_id.unique()
        either = set(pos_adm) | set(icd)
        L.append(f"| {g} | {len(tested):,} ({len(tested) / n:.0%}) | {len(pos_adm) / max(len(tested), 1):.1%} | "
                 f"{len(pos_adm):,} | {len(icd):,} | {len(either):,} |")
        # 3. gate: escalated admissions
        sent = mg.groupby("hadm_id").t.min()
        back = mg.groupby("hadm_id").storetime.max()
        e = esc.to_frame("dose").assign(sent=sent, back=back)
        te = e.dropna(subset=["sent"])
        noesc = coh.hadm_id[~coh.hadm_id.isin(esc.index)]
        gate.append({"group": g,
                     "tested | escalated": len(te) / len(e),
                     "tested | not escalated": noesc.isin(tested).mean(),
                     "sent before 1st dose": (te.sent <= te.dose).mean(),
                     "median h sent -> dose": ((te.dose - te.sent).dt.total_seconds() / 3600).median(),
                     "dose before result back": (te.dose < te.back).mean(),
                     "positive | escalated & tested": te.index.isin(pos_adm).mean()})
    L += ["", "Positive = organism grown, or the result says positive / detected (CMV IgG excluded: past infection).",
          "", "## 3. gate: tests around the first escalation dose", "",
          "| group | tested, escalated | tested, not escalated | sent before 1st dose | median h sent → dose | "
          "1st dose before result back | positive (escalated, tested) |", "|---|---:|---:|---:|---:|---:|---:|"]
    L += [f"| {r['group']} | {r['tested | escalated']:.0%} | {r['tested | not escalated']:.0%} | "
          f"{r['sent before 1st dose']:.0%} | {r['median h sent -> dose']:.0f} | {r['dose before result back']:.0%} | "
          f"{r['positive | escalated & tested']:.1%} |" for r in gate]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
