"""Approach C data: SFT target = reasoning trace + a deviation-prediction tail.

Same task as build_deviation_cls.py (predict P(doctor deviates | input)) but instead of a
bare label word, the assistant emits the FULL clinical reasoning trace (belief / gap /
expected / grounding, exactly as build_sft_examples.py) and then, as the LAST key,
"deviation": follow|deviate. Loss is on the whole assistant turn -> reasoning dominates the
token budget and the deviation label is a small tail ("even just a consequence"): the
probability is a readout CONDITIONED on the model's own generated reasoning, not a shortcut.

Everything except the target/system is imported from build_sft_examples.py (single source of
truth) so the split (seed-0, patient-stratified) and the INPUT rendering are byte-identical to
data/training_set/sft and data/training_set/cls -> the three approaches (a/b/c) are compared on
the same patients, same inputs, same binary label. See HANDOFF_pred_dev.md.

Binary label (LOCKED): follow->"follow"(0); {deviate,off_rubric}->"deviate"(1).
EX-ANTE ONLY: the target carries NO verification/appropriateness/outcome — else the prob would
leak the future.

Usage:
  /opt/anaconda3/envs/congraph/bin/python scripts/build_devreason.py [--preview N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "build_sft_examples", ROOT / "scripts" / "build_sft_examples.py")
_bse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bse)

# rubric traversal (CPU-only, no torch) — used to VERBALIZE the rubric's own routing as a
# deterministic reasoning field the model learns to reproduce. Keyed on eff_branch (the branch
# the deviation label is judged against, incl. the biliary post-hoc rescue) so the emitted
# rubric_recommended/state/rationale stay consistent with the "deviation" tail in the same trace.
from pipeline.rubric_graph import DISEASE_GRAPHS  # noqa: E402
from pipeline.traversal_engine import traverse_graph  # noqa: E402
from experiments.annotation.deviation import (  # noqa: E402
    rubric_recommended_imaging, rubric_state, IMAGING_KEYS)

POSITIVE = {"deviate", "off_rubric"}  # y=1

DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]


# --- pre_features reconstruction (mirrors scripts/eval_certainty_agent.py) -------------------
def _load_features() -> dict[str, dict]:
    return {d: json.load(open(ROOT / "data/rubric_features" / f"{d}_features.json"))["results"]
            for d in DISEASES}


def _prefeatures_by_case(FE: dict) -> dict[tuple, list[dict]]:
    """(disease, hadm) -> [pre_features for the k-th IMAGING decision step], accumulative
    idx_{k-1}: the state before imaging-step k is the feature record just before the k-th
    imaging test_key in RAW record order. Guard idx==0 -> {} (root)."""
    by_case: dict[tuple, list[dict]] = {}
    for disease, fe_disease in FE.items():
        for hadm_s, fe_steps in fe_disease.items():
            img_idx = [i for i, s in enumerate(fe_steps) if s["test_key"] in IMAGING_KEYS]
            by_case[(disease, int(hadm_s))] = [
                (fe_steps[i - 1]["features"] if i - 1 >= 0 else {}) for i in img_idx]
    return by_case


def _pick_node(r) -> str:
    """The rubric frontier node that explains the recommendation, mirroring rubric_state's
    priority: terminal > pending-with-imaging > pending > blocked."""
    ns = r.node_statuses
    if r.terminal_node:
        return r.terminal_node
    img = [nid for nid in r.frontier if ns[nid].status == "pending"
           and set(ns[nid].missing_tests) & IMAGING_KEYS]
    if img:
        return img[0]
    pend = [nid for nid in r.frontier if ns[nid].status == "pending"]
    if pend:
        return pend[0]
    blk = [nid for nid in r.frontier if ns[nid].status == "blocked"]
    if blk:
        return blk[0]
    return r.frontier[0] if r.frontier else r.disease


def build_rubric_reason(eff_branch: str, pre: dict, node_meta: dict, edges: dict):
    """DETERMINISTIC verbalization of the rubric's own routing for the leading hypothesis.
    Returns (rubric_recommended: list[str], rubric_state: str, rubric_rationale: str).
    No LLM, no annotation — a pure function of (eff_branch, causally-masked pre_features, rubric).
    """
    pre = pre or {}
    rec = rubric_recommended_imaging(eff_branch, pre)
    state = rubric_state(eff_branch, pre)
    if eff_branch == "biliary":
        return rec, state, ("Leading hypothesis is routed to the biliary axis (duct / gallstone "
                            "source); the radiation-free duct tools are US / MRCP.")
    if eff_branch not in DISEASE_GRAPHS:
        return rec, state, ("Leading hypothesis is outside the four rubric diseases ('other'); "
                            "the rubric offers no recommendation here.")
    r = traverse_graph(DISEASE_GRAPHS[eff_branch], pre)
    node = _pick_node(r)
    nm = node_meta.get(eff_branch, {}).get(node, {})
    label, ntype = nm.get("label", node), nm.get("type", "")
    conds = [e["when"] for e in edges.get(eff_branch, [])
             if e["to"] == node and (e.get("when") or "").strip().lower() != "always"]
    route = f", reached when: {'; '.join(conds)}" if conds else ""
    if state == "recommends_imaging":
        why = (f"Leading hypothesis '{eff_branch}' sits at rubric node '{label}' [{ntype}]{route}; "
               f"per this decision point the rubric calls for {rec} next.")
    elif state in ("terminal_confirmed", "terminal_excluded", "terminal_low_risk"):
        why = (f"Leading hypothesis '{eff_branch}' has reached rubric terminal '{label}' "
               f"({state.replace('terminal_', '')}){route}; the rubric requests no further "
               f"imaging here.")
    elif state == "wants_nonimaging":
        miss = r.node_statuses[node].missing_tests
        why = (f"Leading hypothesis '{eff_branch}' is at rubric node '{label}'{route}; it first "
               f"requires non-imaging test(s) {miss} before any image.")
    elif state == "blocked":
        why = (f"Leading hypothesis '{eff_branch}' is at rubric node '{label}'{route}, a dead-end "
               f"for this patient's recorded features (no rubric path forward) — an "
               f"incompleteness gap.")
    else:
        why = f"Leading hypothesis '{eff_branch}': rubric state '{state}'."
    return rec, state, why

# reasoning system prompt + rubric-reference + deviation keys appended to the schema ---------
# The rubric_* keys make the "deviate from what?" reference EXPLICIT and self-derived: the model
# states the rubric's recommendation (+ why) for its OWN leading hypothesis, THEN the deviation
# tail is a grounded comparison against a stated reference, not a free-floating guess.
_DEV_KEY = (
    '  "other_hypothesis": when belief argmax is \'other\', name the leading non-rubric '
    'process; else "".\n'
    '  "rubric_recommended": the imaging study/studies the rubric recommends NEXT for your '
    'leading hypothesis at this point, as a JSON array of modality keys (e.g. '
    '["Ultrasound_Abdomen"] or ["MRCP_Abdomen","Ultrasound_Abdomen"]); [] if the rubric is at a '
    'diagnostic terminal, wants a non-imaging test first, or your leading hypothesis is outside '
    'the rubric.\n'
    '  "rubric_state": one of recommends_imaging | terminal_confirmed | terminal_excluded | '
    'terminal_low_risk | wants_nonimaging | blocked | biliary | off_rubric — WHY the rubric does '
    'or does not want an image here.\n'
    '  "rubric_rationale": ONE sentence naming which rubric decision-point your leading '
    'hypothesis sits at and why it recommends (or withholds) that study — this is the reference '
    'you are agreeing or disagreeing with.\n'
    '  "deviation": output "follow" if the "modality" you chose is in "rubric_recommended", else '
    '"deviate" (your study departs from the rubric, OR your leading hypothesis is outside the '
    'rubric). This is the LAST key.\n\nRoute yourself:')
DEVREASON_SYSTEM_TEMPLATE = _bse.SYSTEM_TEMPLATE.replace(
    '  "other_hypothesis": when belief argmax is \'other\', name the leading non-rubric '
    'process; else "".\n\nRoute yourself:', _DEV_KEY)
assert DEVREASON_SYSTEM_TEMPLATE != _bse.SYSTEM_TEMPLATE, "deviation-key injection failed"


def label_word(when_action: str) -> str:
    return "deviate" if when_action in POSITIVE else "follow"


def devreason_target(rec: dict, reason: tuple) -> str:
    # reuse the exact reasoning trace, then append the rubric-reference + deviation keys (order:
    # ...other_hypothesis, rubric_recommended, rubric_state, rubric_rationale, deviation).
    obj = json.loads(_bse.assistant_target(rec))
    rrec, rstate, rwhy = reason
    obj["rubric_recommended"] = rrec
    obj["rubric_state"] = rstate
    obj["rubric_rationale"] = rwhy
    obj["deviation"] = label_word(rec["TARGET"]["when_action"])
    return json.dumps(obj, ensure_ascii=False)


def to_example(rec: dict, system: str, reason: tuple) -> dict:
    when = rec["TARGET"]["when_action"]
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _bse.user_prompt(rec)},
            {"role": "assistant", "content": devreason_target(rec, reason)},
        ],
        "meta": {
            "id": rec["id"], "disease": rec["disease"], "hadm": rec["hadm"], "step": rec["step"],
            "when_action": when,
            "label": label_word(when),
            "y": 1 if label_word(when) == "deviate" else 0,
            "effective_branch": rec["TARGET"]["effective_branch"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/training_set/train_steps.jsonl")
    ap.add_argument("--rubric", default="data/training_set/rubric_library.json")
    ap.add_argument("--out", default="data/training_set/cls_reason")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=0)
    args = ap.parse_args()

    inp = ROOT / args.inp if not Path(args.inp).is_absolute() else Path(args.inp)
    rubric_p = ROOT / args.rubric if not Path(args.rubric).is_absolute() else Path(args.rubric)
    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in inp.read_text().splitlines() if l.strip()]
    lib = json.loads(rubric_p.read_text())
    rubric_text = _bse.compact_rubric(lib)
    system = DEVREASON_SYSTEM_TEMPLATE.replace("__RUBRIC__", rubric_text)

    # rubric-routing verbalization context: pre_features per (disease,hadm) step + node/edge text
    PRE = _prefeatures_by_case(_load_features())
    NODE_META = {d: {n["id"]: n for n in lib[d]["nodes"]} for d in DISEASES}
    EDGES = {d: lib[d]["edges"] for d in DISEASES}
    no_pre = 0

    def reason_for(r: dict) -> tuple:
        nonlocal no_pre
        pres = PRE.get((r["disease"], int(r["hadm"])), [])
        si = int(r["step"]) - 1
        pre = pres[si] if 0 <= si < len(pres) else None
        if pre is None:  # guard: no reconstructable pre -> minimal reason from stored META
            no_pre += 1
            m = r["META"]
            recs = m.get("rubric_recommended")
            recs = recs if isinstance(recs, list) else ([recs] if recs and recs != "-" else [])
            return (recs, m.get("rubric_state", ""),
                    "Rubric decision-point could not be reconstructed for this step.")
        return build_rubric_reason(r["TARGET"]["effective_branch"], pre, NODE_META, EDGES)

    assign = _bse.split_cases(rows, args.val_frac, args.test_frac, args.seed)  # SAME split as sft/cls
    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in rows:
        buckets[assign[(r["disease"], r["hadm"])]].append(to_example(r, system, reason_for(r)))

    for name, exs in buckets.items():
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for e in exs:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def stats(exs):
        cases = {(e["meta"]["disease"], e["meta"]["hadm"]) for e in exs}
        pos = sum(e["meta"]["y"] for e in exs)
        return {"rows": len(exs), "cases": len(cases), "pos(deviate)": pos,
                "neg(follow)": len(exs) - pos,
                "base_rate": round(pos / len(exs), 3) if exs else 0.0,
                "when_action": dict(Counter(e["meta"]["when_action"] for e in exs))}
    overlap = ({(e["meta"]["disease"], e["meta"]["hadm"]) for e in buckets["train"]}
               & {(e["meta"]["disease"], e["meta"]["hadm"]) for e in buckets["test"]})
    manifest = {
        "task": "approach C: reasoning trace + rubric-reference + deviation tail; read P(deviate) "
                "from the tail token conditioned on the generated reasoning",
        "label_map": "follow->follow(0); {deviate,off_rubric}->deviate(1)",
        "rubric_reference": "rubric_recommended/rubric_state/rubric_rationale are DETERMINISTIC "
                            "verbalizations of the rubric's routing on eff_branch (biliary rescue "
                            "included) — the explicit 'deviate from what' reference; no LLM/annotation",
        "rows_without_reconstructable_pre": no_pre,
        "source": str(inp.relative_to(ROOT)),
        "system_prompt_chars": len(system),
        "split": {k: stats(v) for k, v in buckets.items()},
        "patient_leak_train_test": len(overlap),
        "config": vars(args),
    }
    (out_dir / "cls_reason_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"system prompt: {len(system)} chars")
    for k in ("train", "val", "test"):
        s = manifest["split"][k]
        print(f"{k:5s}: {s['rows']:3d} rows / {s['cases']:3d} cases  "
              f"pos={s['pos(deviate)']} neg={s['neg(follow)']} base_rate={s['base_rate']}  "
              f"when={s['when_action']}")
    print(f"patient leak (train∩test cases): {len(overlap)}")
    print(f"rows without reconstructable pre (minimal reason fallback): {no_pre}")
    print(f"wrote {out_dir}/(train|val|test).jsonl + cls_reason_manifest.json")

    for e in buckets["train"][:args.preview]:
        print("\n" + "=" * 80)
        print("SYSTEM (schema tail):",
              e["messages"][0]["content"].split("=== RUBRIC ===")[0][-500:])
        print("\nASSISTANT:", e["messages"][2]["content"][:900], "...")
        print("  meta.y =", e["meta"]["y"], " when =", e["meta"]["when_action"])


if __name__ == "__main__":
    main()
