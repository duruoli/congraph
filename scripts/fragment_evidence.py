"""Direction-2 fragmentation: turn each patient's raw data into `observed(S)` =
a list of structured EVIDENCE PIECES conforming to the v1 schema locked in
rubric_update.md §0.

evidence_piece := {
  anatomy, attribute, state, finding_status(enum), source_test,
  value_unit?, qualifier?, section?
}

THREE ingestion paths, split by perception method (rubric_update.md §6g):
  A. tabular parse, NO LLM  -> Laboratory Tests + Microbiology + vitals line
     (finding_status is FREE from the CSV's Reference Range columns).
  B. narrative fragmentation, LLM -> Radiology reports + Physical-Exam narrative.
  C. light LLM -> Patient History symptom pieces (present/absent; demand side).

Scope = the tree-building corpus (the ~293 hadm_ids annotated in
results/annotation_experiment/full), NOT the full 2400-row raw CSVs.

Untimed for now: lab/micro have no per-test charttime yet (DEFERRED, §6f); pieces
are built admission-global. Reuses experiments.annotation.annotate.call_json.

Output: results/evidence_pieces/{disease}.jsonl, one row per hadm (resumable).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from experiments.annotation.annotate import call_json

csv.field_size_limit(10 ** 8)

RAW_DIR = "data/raw_data"
FULL_DIR = "results/annotation_experiment/full"
LAB_MAP_PATH = os.path.join(RAW_DIR, "lab_test_mapping.csv")
OUT_DIR = "results/evidence_pieces"
DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]

FINDING_STATUS = {"abnormal", "normal", "not_evaluated", "equivocal"}

# ---------------------------------------------------------------------------
# corpus + raw joins
# ---------------------------------------------------------------------------

def corpus_hadms(disease: str) -> set[int]:
    """hadm_ids we actually build trees for (annotated set)."""
    out = set()
    for f in glob.glob(os.path.join(FULL_DIR, f"{disease}_*.json")) + \
             glob.glob(os.path.join(FULL_DIR, "*.json")):
        try:
            obj = json.load(open(f))
        except Exception:
            continue
        for d in (obj if isinstance(obj, list) else [obj]):
            if isinstance(d, dict) and d.get("disease") == disease and "hadm_id" in d:
                out.add(int(d["hadm_id"]))
    return out


def load_raw(disease: str) -> dict[int, dict]:
    path = os.path.join(RAW_DIR, f"{disease}_hadm_info_first_diag.csv")
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows[int(r["hadm_id"])] = r
            except (KeyError, ValueError):
                pass
    return rows


def load_lab_map() -> dict[str, dict]:
    m = {}
    with open(LAB_MAP_PATH) as f:
        for r in csv.DictReader(f):
            m[str(r["itemid"])] = {"label": r.get("label", ""),
                                   "fluid": r.get("fluid", ""),
                                   "category": r.get("category", "")}
    return m


def loads_lenient(s: str):
    """Lab/ref columns are python-dict-ish JSON with bare NaN -> json accepts NaN."""
    s = (s or "").strip()
    if not s or s in ("{}", "nan", "NaN"):
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Path A -- tabular, no LLM
# ---------------------------------------------------------------------------

_NUM = re.compile(r"[-+]?\d*\.?\d+")


def _num(s):
    m = _NUM.search(str(s))
    return float(m.group()) if m else None


def _neg_pos_status(val: str):
    v = str(val).strip().upper()
    if v.startswith(("NEG", "NONE", "NORMAL")) or "NO GROWTH" in v:
        return "normal"
    if v.startswith(("POS", "PRESENT")) or "GROWTH" in v:
        return "abnormal"
    return None


def lab_pieces(row, lab_map):
    labs = loads_lenient(row.get("Laboratory Tests"))
    lo = loads_lenient(row.get("Reference Range Lower"))
    hi = loads_lenient(row.get("Reference Range Upper"))
    out = []
    for itemid, raw_val in labs.items():
        meta = lab_map.get(str(itemid), {})
        v = _num(raw_val)
        rlo, rhi = _num(lo.get(itemid)), _num(hi.get(itemid))
        # finding_status: numeric vs ref range first, else NEG/POS string, else unknown
        status = "not_evaluated"
        if v is not None and (rlo is not None or rhi is not None):
            ok = (rlo is None or v >= rlo) and (rhi is None or v <= rhi)
            status = "normal" if ok else "abnormal"
        else:
            s = _neg_pos_status(raw_val)
            status = s if s else ("equivocal" if v is None else "not_evaluated")
        out.append({
            "anatomy": (meta.get("fluid") or "serum").lower(),
            "attribute": meta.get("label") or f"lab:{itemid}",
            "state": str(raw_val).strip(),
            "finding_status": status,
            "source_test": "laboratory",
            "value_unit": (str(raw_val).strip() if v is not None else None),
            "qualifier": (f"ref {rlo}-{rhi}" if (rlo is not None or rhi is not None) else None),
            "section": meta.get("category") or None,
        })
    return out


def _micro_status(val):
    v = str(val).strip().upper()
    if not v:
        return "not_evaluated"
    if v.startswith(("NO GROWTH", "NEGATIVE", "NONE", "NOT DETECTED")) or "NO GROWTH" in v:
        return "normal"
    return "abnormal"  # a named organism / quantitation = something grew


def micro_pieces(row):
    micro = loads_lenient(row.get("Microbiology"))
    spec = loads_lenient(row.get("Microbiology Spec"))
    out = []
    for itemid, result in micro.items():
        status = _micro_status(result)
        out.append({
            "anatomy": f"specimen:{spec.get(itemid, '?')}",
            "attribute": f"culture:{itemid}",
            "state": str(result).strip(),
            "finding_status": status,
            "source_test": "microbiology",
            "value_unit": None, "qualifier": None, "section": None,
        })
    return out


# ---------------------------------------------------------------------------
# Path B/C -- LLM narrative fragmentation
# ---------------------------------------------------------------------------

FRAG_SYSTEM = """You fragment a clinical text into atomic EVIDENCE PIECES for a structured patient-state store.

Split the text into the SMALLEST factual observations. For EACH observation emit one object:
{
  "anatomy":   organ / region / structure the finding is ABOUT (e.g. appendix, gallbladder wall, common bile duct, pancreatic tail, adnexa, aorta, terminal ileum, RLQ, liver). Lower-case noun phrase.
  "attribute": the property assessed (e.g. wall, diameter, stones, fat-stranding, fluid, enhancement, compressibility, tenderness, distension, size).
  "state":     the raw descriptor as stated (e.g. thickened, dilated, absent, non-visualized, necrosis, pneumatosis, soft, non-tender, normal, unremarkable, patent).
  "finding_status": one of exactly {abnormal, normal, not_evaluated, equivocal}.
                    normal = explicitly normal/unremarkable/patent/no abnormality.
                    abnormal = a positive pathologic finding.
                    not_evaluated = structure named but NOT assessed / not visualized / limited.
                    equivocal = hedged/indeterminate ("possible", "cannot exclude", "borderline").
  "value_unit": numeric measurement WITH unit if given (e.g. "9 mm", "1.4 cm"), else null.
  "qualifier":  hedge or protocol limitation if any ("not fully visualized", "limited by bowel gas", "non-contrast"), else null.
}

STRICT RULES
1. ONE piece per (anatomy, attribute). If an organ is assessed on several attributes, emit several pieces.
2. STRIP the implication/hypothesis tail. "wall thickening INDICATING cholecystitis" -> keep only
   {anatomy: gallbladder, attribute: wall, state: thickened}. Never put the suspected disease in a field.
3. STRIP temporal/comparative framing; fold any comparison into state ("interval-increased"), do not add a time field.
4. A "normal/unremarkable" sweep ("liver, spleen, adrenals normal") -> one normal piece PER named structure.
5. Use ONLY what the text says. Do not infer unstated findings.

Return STRICT JSON: {"pieces": [ {...}, ... ]}"""


def build_frag_user(kind, meta, text):
    head = {"radiology": f"RADIOLOGY REPORT ({meta}). Fragment FINDINGS and IMPRESSION.",
            "physical_exam": ("PHYSICAL EXAMINATION narrative. Fragment organ-system findings AND "
                              "vital signs (anatomy=vital_sign, attribute=temperature/heart_rate/"
                              "blood_pressure/respiratory_rate/spo2; abnormal if outside normal "
                              "physiologic range; 'stable'/'VSS' = normal, no numbers = not_evaluated)."),
            "history": ("PATIENT HISTORY. Emit symptom/PMH pieces only; state = present/absent; "
                        "anatomy = symptom site, attribute = symptom.")}[kind]
    return f"{head}\n\n{text}"


def llm_pieces(kind, meta, text, source_test, section, model, temperature):
    if not (text or "").strip():
        return []
    res = call_json(FRAG_SYSTEM, build_frag_user(kind, meta, text),
                    model=model, temperature=temperature, max_tokens=2000)
    parsed = res.get("parsed") or {}
    out = []
    for p in (parsed.get("pieces") or []):
        if not isinstance(p, dict):
            continue
        st = str(p.get("finding_status", "")).strip().lower()
        out.append({
            "anatomy": str(p.get("anatomy", "")).strip().lower(),
            "attribute": str(p.get("attribute", "")).strip().lower(),
            "state": str(p.get("state", "")).strip(),
            "finding_status": st if st in FINDING_STATUS else "equivocal",
            "source_test": source_test,
            "value_unit": (str(p["value_unit"]).strip() if p.get("value_unit") else None),
            "qualifier": (str(p["qualifier"]).strip() if p.get("qualifier") else None),
            "section": section,
        })
    return out


def radiology_pieces(row, model, temperature):
    reports = loads_lenient(row.get("Radiology"))
    if isinstance(reports, dict):
        reports = [reports]
    out = []
    for rep in (reports or []):
        if not isinstance(rep, dict):
            continue
        meta = f"{rep.get('Modality','')} {rep.get('Region','')} / {rep.get('Exam Name','')}".strip()
        src = (rep.get("Exam Name") or f"{rep.get('Modality','')} {rep.get('Region','')}").strip()
        pieces = llm_pieces("radiology", meta, rep.get("Report", ""), src, None,
                            model, temperature)
        for p in pieces:
            p["note_id"] = rep.get("Note ID")
        out.extend(pieces)
    return out


# ---------------------------------------------------------------------------
# per-hadm assembly
# ---------------------------------------------------------------------------

def build_hadm(disease, hadm, row, lab_map, sources, model, temperature):
    pieces = []
    if "labs" in sources:
        pieces += lab_pieces(row, lab_map)
    if "micro" in sources:
        pieces += micro_pieces(row)
    if "radiology" in sources:
        pieces += radiology_pieces(row, model, temperature)
    if "pe" in sources:
        # PE handled whole (incl. the highly-variable vitals prefix) by the LLM;
        # regex vitals proved too format-fragile to be reliable (see doc §6g note).
        pieces += llm_pieces("physical_exam", "",
                             row.get("Physical Examination", ""),
                             "physical_exam", None, model, temperature)
    if "history" in sources:
        pieces += llm_pieces("history", "", row.get("Patient History", ""),
                             "history", None, model, temperature)
    return {"disease": disease, "hadm_id": hadm, "n_pieces": len(pieces),
            "pieces": pieces, "error": None}


def load_and_compact(path):
    """Return {hadm: good_record} keeping only error-free records, and REWRITE the
    file with just those (drops errored/duplicate lines so they get retried cleanly)."""
    good = {}
    if not os.path.exists(path):
        return good
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error"):
            continue
        good[int(r["hadm_id"])] = r  # last good wins
    with open(path, "w") as fh:
        for r in good.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return good


def run_disease(disease, sources, model, temperature, limit, workers):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{disease}.jsonl")
    raw = load_raw(disease)
    lab_map = load_lab_map()
    corpus = corpus_hadms(disease) & set(raw)
    done = set(load_and_compact(out_path))  # compacts file, drops errored for retry
    todo = sorted(corpus - done)
    if limit:
        todo = todo[:limit]
    print(f"[{disease}] corpus={len(corpus)} done={len(done)} todo={len(todo)} sources={sources}")
    if not todo:
        return
    n_ok = n_err = 0
    with open(out_path, "a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_hadm, disease, h, raw[h], lab_map, sources, model,
                          temperature): h for h in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            h = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                rec = {"disease": disease, "hadm_id": h, "n_pieces": 0,
                       "pieces": [], "error": f"{type(e).__name__}: {e}"}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_err += bool(rec.get("error"))
            n_ok += not rec.get("error")
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} ok={n_ok} err={n_err}")
    print(f"[{disease}] done -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", choices=DISEASES + ["all"], default="all")
    ap.add_argument("--sources", default="labs,micro,radiology,pe,history",
                    help="comma list of: labs micro radiology pe history "
                         "(vitals folded into the pe LLM pass)")
    ap.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="cap hadm this run (0=all)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    sources = {s.strip() for s in args.sources.split(",") if s.strip()}
    diseases = DISEASES if args.disease == "all" else [args.disease]
    for d in diseases:
        run_disease(d, sources, args.model, args.temperature, args.limit, args.workers)


if __name__ == "__main__":
    main()
