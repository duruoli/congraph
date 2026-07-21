"""Map every evidence piece into the STANDARDIZED (canonical) format using the
three converged vocabularies (results/vocab/{anatomy,state,attribute}_map.json).

Each raw piece has (anatomy, attribute, state, finding_status, source_test,
value_unit, qualifier, section). We add the canonical projection of the three
narrative axes plus the attribute AXIS, without dropping the raw values (audit).

Two ingestion regimes (rubric_update.md §6g):
  * NARRATIVE (physical_exam / history / radiology): the three hand-authored
    maps apply. Fallback = raw (self-map), flagged residual per axis.
  * LAB / MICRO (tabular): NOT in the narrative maps by design. attribute is
    ALREADY canonical (analyte label / culture id -> keep as-is, axis=lab_analyte
    or microbiology); state is a numeric value or free text (-> __NUMERIC__ when a
    value_unit is present, else raw); anatomy mapped opportunistically.

The finding_status enum (abnormal|normal|not_evaluated|equivocal) is the gate-
readable signal and is passed through unchanged.

Output: results/evidence_pieces_std/{disease}.jsonl  (same per-patient envelope,
each piece gains anatomy_canonical / attribute_canonical / attribute_axis /
state_canonical / std_residual). Prints coverage stats.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter

EV_DIR = "results/evidence_pieces"
OUT_DIR = "results/evidence_pieces_std"
VOCAB_DIR = "results/vocab"
LAB_SOURCES = {"laboratory", "microbiology"}
_NUMERIC = re.compile(r"^\s*[<>]?\s*\d+(\.\d+)?\s*[a-z%/µ]*.*$")


def load_map(field):
    d = json.load(open(os.path.join(VOCAB_DIR, f"{field}_map.json")))
    return d["map"]


def load_axis():
    """canonical attribute -> axis (system), from attribute_vocab.json."""
    v = json.load(open(os.path.join(VOCAB_DIR, "attribute_vocab.json")))
    return {x["canonical"]: x["system"] for x in v}


def load_residual_set(field):
    """canonicals flagged residual (rule-miss, self-mapped) in {field}_vocab.json."""
    v = json.load(open(os.path.join(VOCAB_DIR, f"{field}_vocab.json")))
    return {x["canonical"] for x in v if x.get("residual")}


def std_piece(p, amap, smap, tmap, axis_of, a_resid, s_resid):
    src = p.get("source_test", "")
    raw_a = str(p.get("anatomy", "")).strip().lower()
    raw_s = str(p.get("state", "")).strip().lower()
    raw_t = str(p.get("attribute", "")).strip().lower()
    resid = {}

    if src in LAB_SOURCES:
        # tabular: attribute already canonical (analyte / culture id)
        anatomy_c = amap.get(raw_a, p.get("anatomy"))
        attribute_c = p.get("attribute")  # keep as-is, already canonical
        attr_axis = "lab_analyte" if src == "laboratory" else "microbiology"
        if p.get("value_unit") and _NUMERIC.match(raw_s):
            state_c = "__NUMERIC__"
        else:
            state_c = smap.get(raw_s, p.get("state"))
    else:
        anatomy_c = amap.get(raw_a, p.get("anatomy"))
        attribute_c = tmap.get(raw_t, p.get("attribute"))
        attr_axis = axis_of.get(attribute_c, "__residual__")
        state_c = smap.get(raw_s, p.get("state"))
        # per-axis residual = canonical flagged residual in the vocab (rule-miss)
        resid = {
            "anatomy": anatomy_c in a_resid,
            "attribute": attr_axis == "__residual__",
            "state": state_c in s_resid,
        }

    out = dict(p)
    out["anatomy_canonical"] = anatomy_c
    out["attribute_canonical"] = attribute_c
    out["attribute_axis"] = attr_axis
    out["state_canonical"] = state_c
    flagged = {k: True for k, v in resid.items() if v}
    if flagged:
        out["std_residual"] = flagged
    return out


def main():
    amap, smap, tmap = load_map("anatomy"), load_map("state"), load_map("attribute")
    axis_of = load_axis()
    a_resid, s_resid = load_residual_set("anatomy"), load_residual_set("state")
    os.makedirs(OUT_DIR, exist_ok=True)

    n_pieces = n_narr = 0
    axis_hist = Counter()
    narr_resid = Counter()  # per-axis residual among narrative pieces
    for fp in sorted(glob.glob(os.path.join(EV_DIR, "*.jsonl"))):
        disease = os.path.basename(fp)
        out_lines = []
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            std = []
            for p in rec.get("pieces", []):
                sp = std_piece(p, amap, smap, tmap, axis_of, a_resid, s_resid)
                std.append(sp)
                n_pieces += 1
                if p.get("source_test") not in LAB_SOURCES:
                    n_narr += 1
                    axis_hist[sp["attribute_axis"]] += 1
                    for k in sp.get("std_residual", {}):
                        narr_resid[k] += 1
            rec["pieces"] = std
            out_lines.append(json.dumps(rec, ensure_ascii=False))
        with open(os.path.join(OUT_DIR, disease), "w") as fh:
            fh.write("\n".join(out_lines) + "\n")
        print(f"[{disease}] {len(out_lines)} patients")

    print(f"\ntotal pieces standardized: {n_pieces}  (narrative {n_narr}, "
          f"lab/micro {n_pieces - n_narr})")
    print("narrative attribute-axis distribution:")
    for ax, c in axis_hist.most_common():
        print(f"  {c:6d}  {ax}")
    print("narrative per-axis residual (self-mapped, rule-miss):")
    for k in ("anatomy", "attribute", "state"):
        print(f"  {k:10s} {narr_resid[k]:5d} / {n_narr} = "
              f"{narr_resid[k]/n_narr*100:.1f}%")
    print(f"\nwrote {OUT_DIR}/*.jsonl")


if __name__ == "__main__":
    main()
