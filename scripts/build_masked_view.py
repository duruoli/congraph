"""
Build the per-step "causally masked" context that an LLM-as-doctor would see.

Framing (HANDOFF §3, option B):
  - step 0 = HPI + Physical Exam + Lab tests (merged baseline, always visible)
  - radiology reports are ordered by RR-N (Note ID), a time proxy (NOT list order)
  - management / therapeutic imaging is DROPPED entirely:
        Exam Name ~ PORTABLE / PRE-OP / LINE PLACEMENT / PICC, or Modality ERCP / Drainage
  - DECISION STEPS = rubric-relevant abdominal diagnostic imaging only
        (US / CT / MRI / MRCP / CTU abdomen) -> these are the steps the LLM must
        rationalize, and the only steps deviation labels attach to.
  - everything else that survives filtering (chest X-ray, abdominal plain film,
        head CT, ...) is CONTEXT, not a decision step: its *result* becomes visible
        to later decisions once passed in RR-N order, but the LLM is never asked to
        invent a motive for ordering it (avoids fabricated belief-trajectory noise).
  - causal masking: at decision step i, visible = baseline + every report with
        RR-N strictly less than i; the result of step i and all later reports are masked.

Outcome fields (Discharge Diagnosis / ICD / Procedures) are NEVER shown (leakage).

Usage:
  python3 scripts/build_masked_view.py <disease> [hadm_id ...] [--json]
  --json  also writes structured per-step context to data/masked_views/<disease>_<hadm>.json
"""
import sys, os, re, json
import pandas as pd

RAW = {
    "cholecystitis": "data/raw_data/cholecystitis_hadm_info_first_diag.csv",
    "appendicitis": "data/raw_data/appendicitis_hadm_info_first_diag.csv",
    "diverticulitis": "data/raw_data/diverticulitis_hadm_info_first_diag.csv",
    "pancreatitis": "data/raw_data/pancreatitis_hadm_info_first_diag.csv",
}

# rubric-relevant abdominal diagnostic imaging = the only DECISION steps
DECISION_SET = {
    "CT_Abdomen", "Ultrasound_Abdomen", "MRCP_Abdomen", "MRI_Abdomen", "CTU_Abdomen",
}
# management / therapeutic -> dropped from the sequence entirely
DROP_EXAM_KW = ("PORTABLE", "PRE-OP", "LINE PLACEMENT", "PICC")
DROP_MODALITY = {"ERCP", "Drainage"}


def load_lab_map():
    m = pd.read_csv("data/raw_data/lab_test_mapping.csv")
    return {str(int(r.itemid)): r.label for r in m.itertuples() if pd.notna(r.itemid)}


def decode_labs(lab_json, lo_json, hi_json, labmap, max_items=60):
    try:
        vals = json.loads(lab_json)
    except Exception:
        return "(no labs)"
    lo = json.loads(lo_json) if isinstance(lo_json, str) else {}
    hi = json.loads(hi_json) if isinstance(hi_json, str) else {}
    out = []
    for itemid, v in list(vals.items())[:max_items]:
        name = labmap.get(str(itemid), f"item{itemid}")
        rng = ""
        if itemid in lo and itemid in hi and lo[itemid] == lo[itemid] and hi[itemid] == hi[itemid]:
            rng = f" (ref {lo[itemid]}-{hi[itemid]})"
        out.append(f"  {name}: {v}{rng}")
    return "\n".join(out)


def rrn(note_id):
    m = re.search(r"RR-(\d+)", str(note_id))
    return int(m.group(1)) if m else None


def classify(r):
    """Return one of: 'decision', 'context', 'dropped'."""
    exam = str(r.get("Exam Name", "")).upper()
    if r.get("Modality") in DROP_MODALITY:
        return "dropped"
    if any(k in exam for k in DROP_EXAM_KW):
        return "dropped"
    mr = f"{r.get('Modality')}_{r.get('Region')}"
    return "decision" if mr in DECISION_SET else "context"


def ordered_reports(rad):
    """RR-N sorted [(rrn, role, item)], dropped items removed."""
    items = []
    for r in rad:
        n = rrn(r.get("Note ID"))
        if n is None:
            continue
        role = classify(r)
        if role == "dropped":
            continue
        items.append((n, role, r))
    items.sort(key=lambda x: x[0])
    return items


def build_record(disease, hadm, row, labmap):
    rad = json.loads(row["Radiology"]) if isinstance(row["Radiology"], str) else []
    items = ordered_reports(rad)

    rec = {
        "hadm_id": int(hadm),
        "disease": disease,
        "baseline": {
            "patient_history": str(row["Patient History"]).strip(),
            "physical_examination": str(row["Physical Examination"]).strip(),
            "laboratory_tests": decode_labs(
                row["Laboratory Tests"], row.get("Reference Range Lower"),
                row.get("Reference Range Upper"), labmap),
        },
        "radiology_order": [
            {"rrn": n, "role": role, "modality": r.get("Modality"),
             "region": r.get("Region"), "exam": r.get("Exam Name")}
            for n, role, r in items
        ],
        "decision_points": [],
    }

    n_decisions = sum(1 for _, role, _ in items if role == "decision")
    step = 0
    for idx, (n, role, r) in enumerate(items):
        if role != "decision":
            continue
        step += 1
        visible_prior = [
            {"modality": pr.get("Modality"), "region": pr.get("Region"),
             "exam": pr.get("Exam Name"), "role": prole, "report": str(pr.get("Report", "")).strip()}
            for pn, prole, pr in items[:idx]
        ]
        rec["decision_points"].append({
            "step": step,
            "ordered": f"{r.get('Modality')} {r.get('Region')} ({r.get('Exam Name')})",
            "visible_prior_imaging": visible_prior,
            "masked_result_of_this_test": str(r.get("Report", "")).strip(),
            "n_later_masked": len(items) - idx - 1,
        })
    rec["n_decision_steps"] = n_decisions
    return rec


def print_record(rec):
    print("=" * 90)
    print(f"PATIENT {rec['hadm_id']}  ({rec['disease']})   decision steps: {rec['n_decision_steps']}")
    print("RR-N ordered imaging (role):")
    for it in rec["radiology_order"]:
        print(f"   RR-{it['rrn']:<4} [{it['role']:8}] {it['modality']} {it['region']} ({it['exam']})")
    print("=" * 90)

    b = rec["baseline"]
    print("\n----- STEP 0  (baseline, always visible) -----")
    print("[Patient History]\n" + b["patient_history"])
    print("\n[Physical Examination]\n" + b["physical_examination"])
    print("\n[Laboratory Tests]\n" + b["laboratory_tests"])

    for dp in rec["decision_points"]:
        print(f"\n----- DECISION STEP {dp['step']}: doctor ordered  {dp['ordered']} -----")
        if dp["visible_prior_imaging"]:
            print("  visible prior imaging results:")
            for v in dp["visible_prior_imaging"]:
                print(f"    - ({v['role']}) {v['modality']} {v['region']} ({v['exam']})")
        else:
            print("  visible prior imaging results: NONE (only baseline)")
        print(f"  MASKED: result of this test + {dp['n_later_masked']} later report(s)")
    print()


def main():
    args = [a for a in sys.argv[1:] if a != "--json"]
    dump_json = "--json" in sys.argv
    disease = args[0] if args else "cholecystitis"
    raw = pd.read_csv(RAW[disease])
    labmap = load_lab_map()
    ids = [int(x) for x in args[1:]] if len(args) > 1 else raw["hadm_id"].head(3).tolist()

    if dump_json:
        os.makedirs("data/masked_views", exist_ok=True)

    for hadm in ids:
        row = raw[raw["hadm_id"] == hadm]
        if row.empty:
            print(f"### {hadm}: NOT FOUND\n")
            continue
        rec = build_record(disease, hadm, row.iloc[0], labmap)
        print_record(rec)
        if dump_json:
            out = f"data/masked_views/{disease}_{hadm}.json"
            with open(out, "w") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            print(f"  -> wrote {out}\n")


if __name__ == "__main__":
    main()
