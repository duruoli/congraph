"""Phase-1 extraction: re-ground each decision step's OPEN CLINICAL QUESTION from
the free-text `information_gap` (+ `expected_finding`), because the existing
`action_role` taxonomy is incomplete (it has no `etiology`/`complication` and
mis-files them under `localize_source`).

This pass is DISCOVERY, not production: the label set is a *seed* and the prompt
keeps an open `other:<phrase>` escape so a missing 6th target can surface (avoids
re-committing the closed-set mistake of `action_role`). Two things come out:

  1. `targets`            – multi-label question set (existence/etiology/severity/
                            complication/broaden/other) with an evidence span each.
  2. `sought_dimensions`  – free-text of the anatomy/pathology the doctor wanted
                            to see (raw material for the disease-level `required(S)`
                            table; deliberately modality-agnostic).

Outputs JSONL (one row per step, resumable) + a per-disease aggregation
(counts, co-occurrence, `other` list) to results/question_targets/.

Reuses experiments.annotation.annotate.call_json (OpenRouter client).
"""
from __future__ import annotations

import argparse
import collections
import glob
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from experiments.annotation.annotate import call_json

FULL_DIR = "results/annotation_experiment/full"
OUT_DIR = "results/question_targets"
EXTRACT_PATH = os.path.join(OUT_DIR, "extractions.jsonl")
SUMMARY_PATH = os.path.join(OUT_DIR, "summary.json")

SEED_LABELS = {"existence", "etiology", "severity", "complication", "broaden"}

SYSTEM = """You are annotating the OPEN CLINICAL QUESTION behind one imaging-decision step.

You are given, for a single decision step:
- disease: the working-diagnosis context
- information_gap: what the physician still does not know at this point
- expected_finding: what they hope the ordered study will show

Identify which diagnostic question(s) are CURRENTLY OPEN -- i.e. the thing this
imaging is meant to RESOLVE. Use this controlled vocabulary (multi-label):

- existence   : whether the primary diagnosis is present at all, or where the
                source of the syndrome is -- diagnosis NOT yet established.
- etiology    : diagnosis (largely) established; the open question is WHAT is
                CAUSING it (stone, obstruction, mass/malignancy, source of
                perforation, gallstone-vs-other, etc.).
- severity    : how bad / extent / grade of the ESTABLISHED diagnosis.
- complication: whether a specific downstream complication is present
                (abscess, perforation, necrosis, obstruction, fistula, ischemia).
- broaden     : a SPECIFIC leading diagnosis was ALREADY established/strongly
                favored, and it now fails to explain the picture, so the search
                re-opens to an ALTERNATIVE or ADDITIONAL source. Do NOT use
                broaden for an initial undifferentiated work-up that considers
                several candidates with no prior leading diagnosis -- that is
                `existence` over multiple sources.

CRITICAL RULES
1. Label a category ONLY if it is the question this step is TRYING TO RESOLVE.
   Do NOT label something merely MENTIONED as background/context. Example: a gap
   that says "worried for necrosis, but the outside CT was low-quality" whose real
   open question is confirming the diagnosis on a better study = existence, NOT
   severity.
2. Multiple labels are allowed if several questions are genuinely open together.
3. If NONE of the five fit, use "other" and put a short free-text phrase naming
   the question in `evidence_span` prefixed with "OTHER: ". Do not force-fit.
4. For every label, quote the MINIMAL span from information_gap that justifies it.

Also extract:
- sought_dimensions: a short phrase for the SPECIFIC finding/dimension the
  physician wants to see (from expected_finding). Describe anatomy/pathology,
  NOT the imaging modality (no "CT"/"ultrasound"/"MRCP").

Return STRICT JSON:
{
  "targets": [{"label": "existence", "evidence_span": "..."}, ...],
  "sought_dimensions": "..."
}"""


def build_user(disease: str, gap: str, expected: str) -> str:
    return (f"disease: {disease}\n"
            f"information_gap: {gap}\n"
            f"expected_finding: {expected}")


def iter_records():
    for f in sorted(glob.glob(os.path.join(FULL_DIR, "*.json"))):
        obj = json.load(open(f))
        for d in (obj if isinstance(obj, list) else [obj]):
            if isinstance(d, dict) and "disease" in d and "steps" in d:
                yield d


def iter_steps():
    for d in iter_records():
        for s in d.get("steps", []):
            ea = s.get("representative_ex_ante", {}) or {}
            yield {
                "disease": d["disease"],
                "hadm_id": d["hadm_id"],
                "step": s.get("step"),
                "n_visible_prior": s.get("n_visible_prior"),
                "action_role": ea.get("action_role"),
                "information_gap": (ea.get("information_gap") or "").strip(),
                "expected_finding": (ea.get("expected_finding") or "").strip(),
            }


def key_of(row) -> str:
    return f"{row['disease']}|{row['hadm_id']}|{row['step']}"


def load_done(path: str) -> set[str]:
    done = set()
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["key"])
                except Exception:
                    pass
    return done


def norm_targets(parsed: dict) -> list[dict]:
    out = []
    for t in (parsed.get("targets") or []):
        if not isinstance(t, dict):
            continue
        lab = str(t.get("label", "")).strip().lower()
        span = str(t.get("evidence_span", "")).strip()
        if lab in SEED_LABELS:
            out.append({"label": lab, "evidence_span": span})
        elif lab == "other" or lab.startswith("other"):
            out.append({"label": "other", "evidence_span": span})
    return out


def extract_one(row, model, temperature):
    if not row["information_gap"]:
        return {**row, "key": key_of(row), "targets": [], "sought_dimensions": "",
                "error": "empty_gap"}
    user = build_user(row["disease"], row["information_gap"], row["expected_finding"])
    try:
        res = call_json(SYSTEM, user, model=model, temperature=temperature,
                        max_tokens=600)
        parsed = res.get("parsed") or {}
        return {**row, "key": key_of(row),
                "targets": norm_targets(parsed),
                "sought_dimensions": str(parsed.get("sought_dimensions", "")).strip(),
                "error": None if parsed else "parse_fail"}
    except Exception as e:  # noqa: BLE001
        return {**row, "key": key_of(row), "targets": [], "sought_dimensions": "",
                "error": f"{type(e).__name__}: {e}"}


def run_extract(model: str, temperature: float, limit: int, workers: int):
    os.makedirs(OUT_DIR, exist_ok=True)
    done = load_done(EXTRACT_PATH)
    todo = [r for r in iter_steps() if key_of(r) not in done]
    if limit:
        todo = todo[:limit]
    print(f"[extract] {len(done)} done, {len(todo)} to do (model={model}, workers={workers})")
    if not todo:
        return
    n_ok = n_err = 0
    with open(EXTRACT_PATH, "a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(extract_one, r, model, temperature): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if rec.get("error"):
                n_err += 1
            else:
                n_ok += 1
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  ok={n_ok} err={n_err}")
    print(f"[extract] done. ok={n_ok} err={n_err} -> {EXTRACT_PATH}")


def summarize():
    rows = [json.loads(l) for l in open(EXTRACT_PATH) if l.strip()]
    by_disease = collections.defaultdict(collections.Counter)      # disease -> label counts
    cooc = collections.defaultdict(collections.Counter)            # disease -> pair counts
    others = collections.defaultdict(list)                         # disease -> [other spans]
    role_vs_target = collections.Counter()                         # (action_role, label)
    n_multi = n_labeled = 0
    for r in rows:
        labs = sorted({t["label"] for t in r.get("targets", [])})
        if labs:
            n_labeled += 1
        if len(labs) > 1:
            n_multi += 1
        dis = r["disease"]
        for lab in labs:
            by_disease[dis][lab] += 1
            role_vs_target[(r.get("action_role"), lab)] += 1
        for a, b in itertools.combinations(labs, 2):
            cooc[dis][f"{a}+{b}"] += 1
        for t in r.get("targets", []):
            if t["label"] == "other" and t.get("evidence_span"):
                others[dis].append(t["evidence_span"])

    print(f"\nrows={len(rows)}  labeled={n_labeled}  multi-label={n_multi}")
    print("\n=== per-disease question_target counts ===")
    for dis, c in by_disease.items():
        print(f"## {dis}")
        for lab, n in c.most_common():
            print(f"   {n:4d}  {lab}")
    print("\n=== label co-occurrence (per disease, top) ===")
    for dis, c in cooc.items():
        print(f"## {dis}: " + ", ".join(f"{k}={v}" for k, v in c.most_common(6)))
    print("\n=== how the NEW target relates to the OLD action_role ===")
    role_tab = collections.defaultdict(collections.Counter)
    for (role, lab), n in role_vs_target.items():
        role_tab[role][lab] += n
    for role, c in role_tab.items():
        print(f"   action_role={role}: " + ", ".join(f"{k}={v}" for k, v in c.most_common()))
    print("\n=== 'other' spans (candidate 6th target) ===")
    for dis, spans in others.items():
        print(f"## {dis} ({len(spans)})")
        for s in spans[:15]:
            print(f"   - {s[:120]}")

    json.dump({
        "n_rows": len(rows), "n_labeled": n_labeled, "n_multi": n_multi,
        "by_disease": {d: dict(c) for d, c in by_disease.items()},
        "cooccurrence": {d: dict(c) for d, c in cooc.items()},
        "other_spans": dict(others),
    }, open(SUMMARY_PATH, "w"), ensure_ascii=False, indent=2)
    print(f"\n[summary] -> {SUMMARY_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="cap steps this run (0=all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    if not args.summarize_only:
        run_extract(args.model, args.temperature, args.limit, args.workers)
    summarize()


if __name__ == "__main__":
    main()
