"""Emergence analysis for the b-NEW pre-RL traces (bottom-up reward design).

For each trace, classify the categorical reasoning components against the three axes
(variance / gold-verifiable / headroom) so we reward only what the base does SOMETIMES and
OFTEN WRONG. Joins traces -> meta (gold how_modality / action_role / rubric_recommended / y)
by split index.

Usage: python analyze_emergence.py <generations.jsonl> <meta.jsonl> [<generations2> <meta2> ...]
"""
import json, re, sys, statistics
from collections import Counter, defaultdict

CUE = "\n\nAnswer: "

# ---- modality extraction: map free-text mentions to the 4 gold classes ----
MOD_PATS = {
    "CT_Abdomen":        re.compile(r'\b(ct scan|ct[- ]?abd|computed tomograph|\bct\b)', re.I),
    "Ultrasound_Abdomen":re.compile(r'\b(ultrasound|sonograph|\bus\b|\bruq us\b)', re.I),
    "MRCP_Abdomen":      re.compile(r'\b(mrcp)', re.I),
    "MRI_Abdomen":       re.compile(r'\b(mri)\b', re.I),
}
# MRI pattern must not fire on MRCP; handle order below.
RUBRIC_PAT = re.compile(r'(rubric|recommend|guideline|first[- ]?line|standard of care|pathway)', re.I)
HYP_PAT = re.compile(r'(appendicitis|cholecystitis|diverticulitis|pancreatitis|\bhypothes|differential|suspect|working diagnosis)', re.I)
# free-text markers of the why-trace components
DIFF_PAT = re.compile(r'(other (hypothes|possibilit|cause|diagnos)|alternativ|rather than|instead of|rule out|as opposed|vs\.|versus|could also be|differential)', re.I)
INFOGAP_PAT = re.compile(r'(information gap|to (confirm|clarify|determine|distinguish|establish|rule)|unclear|uncertain|need to know|to identify|to assess whether|question is whether)', re.I)
EXPFIND_PAT = re.compile(r'(expect(ed)? (to (see|find|show)|finding)|would (show|reveal|demonstrate|confirm)|look(ing)? for|if (the )?(ct|us|mri|scan|imaging) shows|presence of|to visualize)', re.I)

RECO_PAT = re.compile(r'(next step|should (order|get|obtain|proceed)|warrant|recommend|appropriate|order(ing)? (a |an )?|proceed to|indicated|is to (get|obtain|order)|would (order|get|obtain))', re.I)


def mentioned_modalities(text):
    out = set()
    for name, pat in MOD_PATS.items():
        if pat.search(text):
            out.add(name)
    # MRI vs MRCP disambiguation: 'mri' pattern uses \bmri\b so 'mrcp' won't match; ok.
    return out


def primary_modality(body):
    """Heuristic guess at the study the model PREDICTS the physician will order.
    Look at the last recommendation-context sentence; fall back to last modality mentioned."""
    sents = re.split(r'(?<=[.!?])\s+', body)
    # scan from the end for a sentence that both names a modality and reads as a recommendation
    for s in reversed(sents):
        mods = mentioned_modalities(s)
        if mods and RECO_PAT.search(s):
            # if multiple, prefer the one closest to a reco keyword; just pick deterministically
            return sorted(mods)[0] if len(mods) == 1 else _pick(s, mods)
    # fallback: last sentence naming any modality
    for s in reversed(sents):
        mods = mentioned_modalities(s)
        if mods:
            return sorted(mods)[0] if len(mods) == 1 else _pick(s, mods)
    return None


def _pick(sentence, mods):
    """When a sentence names several modalities, pick the one whose keyword appears LAST
    (usually the concluded choice, e.g. 'US is less sensitive so CT is warranted')."""
    last, best = -1, None
    for name in mods:
        m = list(MOD_PATS[name].finditer(sentence))
        if m and m[-1].start() > last:
            last, best = m[-1].start(), name
    return best


def parse_answer(gen):
    if CUE in gen:
        after = gen.rsplit(CUE, 1)[1]
        m = re.match(r'\s*"?(follow|deviate)"?', after, re.I)
        if m:
            return m.group(1).lower()
    return None


def analyze(traces, meta):
    rows = []
    for t in traces:
        i = t["i"]
        m = meta[i]["meta"]
        gen = t["generation"]
        body = gen.rsplit(CUE, 1)[0] if CUE in gen else gen
        pred_ans = parse_answer(gen)
        gold_ans = "deviate" if m["y"] == 1 else "follow"
        pm = primary_modality(body)
        rows.append({
            "i": i,
            "gold_y": m["y"], "gold_ans": gold_ans, "pred_ans": pred_ans,
            "ans_correct": (pred_ans == gold_ans) if pred_ans else None,
            "cue_hit": CUE in gen,
            "gold_mod": m.get("how_modality"),
            "pred_mod": pm,
            "mod_correct": (pm == m.get("how_modality")) if pm else None,
            "mods_mentioned": sorted(mentioned_modalities(body)),
            "n_mods_mentioned": len(mentioned_modalities(body)),
            "gold_role": m.get("action_role"),
            "gold_rubric_rec": m.get("rubric_recommended"),
            "rubric_ref": bool(RUBRIC_PAT.search(body)),
            "hyp_named": bool(HYP_PAT.search(body)),
            "differential": bool(DIFF_PAT.search(body)),
            "info_gap": bool(INFOGAP_PAT.search(body)),
            "exp_finding": bool(EXPFIND_PAT.search(body)),
            "chars": len(gen),
        })
    return rows


def summarize(rows, label):
    n = len(rows)
    print(f"\n{'='*70}\n{label}  (n={n})\n{'='*70}")
    # format / answer
    cue = sum(r["cue_hit"] for r in rows)
    ans = [r for r in rows if r["pred_ans"]]
    acc = sum(r["ans_correct"] for r in ans) / len(ans) if ans else 0
    print(f"FORMAT   cue-hit(answer emitted): {cue}/{n} = {cue/n:.2f}")
    print(f"ANSWER   parseable: {len(ans)}/{n};  acc vs gold: {acc:.3f}  "
          f"(base_rate deviate={sum(r['gold_y'] for r in rows)/n:.3f})")
    print(f"         pred dist: {Counter(r['pred_ans'] for r in rows)}")
    # modality
    pm = [r for r in rows if r["pred_mod"]]
    macc = sum(r["mod_correct"] for r in pm) / len(pm) if pm else 0
    print(f"\nMODALITY primary-modality extracted: {len(pm)}/{n} = {len(pm)/n:.2f}")
    print(f"         pred primary dist: {Counter(r['pred_mod'] for r in rows)}")
    print(f"         gold dist:         {Counter(r['gold_mod'] for r in rows)}")
    print(f"         primary-modality ACC vs gold: {macc:.3f}  (headroom = {1-macc:.3f})")
    print(f"         #modalities mentioned per trace: {Counter(r['n_mods_mentioned'] for r in rows)}")
    # structure presence (the saturated keyword-style terms)
    for k in ("rubric_ref", "hyp_named", "differential", "info_gap", "exp_finding"):
        c = sum(r[k] for r in rows)
        print(f"STRUCT   {k:14s}: {c}/{n} = {c/n:.2f}")
    # correlation of components with answer-correctness
    print("\nCORRELATION with correct answer (among parseable):")
    for k in ("rubric_ref", "hyp_named", "differential", "info_gap", "exp_finding", "mod_correct"):
        present = [r for r in ans if r.get(k)]
        absent = [r for r in ans if not r.get(k)]
        pa = sum(r["ans_correct"] for r in present)/len(present) if present else float('nan')
        ab = sum(r["ans_correct"] for r in absent)/len(absent) if absent else float('nan')
        print(f"         {k:14s}: present acc={pa:.3f} (n={len(present)})  absent acc={ab:.3f} (n={len(absent)})")
    return rows


if __name__ == "__main__":
    args = sys.argv[1:]
    all_rows = []
    pairs = list(zip(args[0::2], args[1::2]))
    for gen_f, meta_f in pairs:
        traces = [json.loads(l) for l in open(gen_f)]
        meta = [json.loads(l) for l in open(meta_f)]
        rows = analyze(traces, meta)
        summarize(rows, f"{gen_f.split('/')[-1]}")
        all_rows.extend(rows)
    if len(pairs) > 1:
        summarize(all_rows, "COMBINED")
    json.dump(all_rows, open("/private/tmp/claude-501/-Users-duruoli-A-A----1----PhD-0-1-code-congraph/3da92298-84f7-472d-8347-2afd109dba18/scratchpad/emergence_rows.json", "w"), indent=1)
