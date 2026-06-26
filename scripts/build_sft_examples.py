"""Compile the step-level training set (data/training_set/train_steps.jsonl) into
chat-format SFT examples for direction-B fine-tuning (Qwen LoRA).

This is the BRIDGE from the structured INPUT/TARGET/REWARD records to (messages)
pairs a causal-LM SFT trainer (trl SFTTrainer / peft) can consume. It adds NO new
information — it only renders + splits.

Design decisions (locked 2026-06, see agent_training_plan / HANDOFF §3, §9):
  - TARGET granularity = FULL reasoning trace. The assistant generates a JSON object
        {belief (5+other dist), modality, information_gap, expected_finding,
         action_role, grounding (+ other_hypothesis when belief argmax='other')}.
  - deviate is NOT generated. when_action (follow/deviate/off_rubric) is a deterministic
        function of the generated belief argmax vs the rubric -> computed at EVAL time
        from the model's belief output, never a training target. So it is absent here.
  - direction B INTERNALIZES the §2 behavior spec into weights -> the §2 rules are NOT
        injected into the prompt (HANDOFF §3). The system prompt gives only role +
        output schema + the rubric (agent stage GIVES the rubric). The rubric is the
        COMPACT serialization of all 4 disease sub-rubrics + open 'other' (full graph
        logic, trimmed prose) so the agent self-routes, exactly as at inference.
  - REWARD (verification / appropriateness / alarm-resolve / sample_weight) is held
        ENTIRELY out of the messages — it is RL/eval material. It is carried in `meta`
        (not seen by the SFT loss) so the same files double as the eval set.
  - Split is BY PATIENT (hadm), stratified by disease, so multiple steps of one patient
        never straddle train/val/test (would leak the trajectory).

Usage:
  /opt/anaconda3/bin/python scripts/build_sft_examples.py
      [--in data/training_set/train_steps.jsonl]
      [--rubric data/training_set/rubric_library.json]
      [--out data/training_set/sft]
      [--val-frac 0.15] [--test-frac 0.15] [--seed 0]
      [--preview N]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANON_BRANCHES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis", "other"]
MODALITIES = ["CT_Abdomen", "Ultrasound_Abdomen", "MRCP_Abdomen", "MRI_Abdomen", "CTU_Abdomen"]


# --- rubric: compact serialization of the library (constant across rows) -----
def compact_rubric(lib: dict, desc_len: int = 140) -> str:
    """Readable, training-lean rendering of all 4 disease graphs + open 'other'.
    Keeps node decision logic (required_tests + trimmed description) and edge
    conditions; drops the verbose prose of the full rubric_library.json."""
    out = []
    for disease in CANON_BRANCHES:
        g = lib.get(disease)
        if g is None:
            continue
        if g.get("kind") == "open_differential":
            out.append("### other (open differential)\n" + g.get("note", "") +
                       "\nBiliary axis: " + g.get("biliary_axis", ""))
            continue
        lines = [f"### {disease} (root={g.get('root')})", "nodes:"]
        for n in g["nodes"]:
            tests = ", ".join(n.get("required_tests") or []) or "—"
            d = (n.get("description") or "").strip().replace("\n", " ")
            if len(d) > desc_len:
                d = d[:desc_len] + "…"
            lines.append(f"  - {n['id']} [{n.get('label','')}] tests:[{tests}] {d}")
        lines.append("edges:")
        for e in g["edges"]:
            w = (e.get("when") or "").strip().replace("\n", " ")
            lines.append(f"  - {e['from']} -> {e['to']}" + (f"  ({w})" if w else ""))
        out.append("\n".join(lines))
    return "\n\n".join(out)


SYSTEM_TEMPLATE = """You are a clinical reasoning agent for adult acute abdominal pain. \
At each decision step you are given the patient's current state (baseline + the imaging \
reports available SO FAR, with this step's result hidden) and a rubric covering four \
index diseases plus an open 'other' slot. You decide the next imaging move and explain it.

Output a single JSON object, no prose around it, with EXACTLY these keys:
  "belief": object mapping each of {appendicitis, cholecystitis, diverticulitis, \
pancreatitis, other} to a probability summing to ~1 (your current differential).
  "modality": one of {CT_Abdomen, Ultrasound_Abdomen, MRCP_Abdomen, MRI_Abdomen, \
CTU_Abdomen} — the imaging you order next.
  "information_gap": the specific question this study must answer.
  "expected_finding": what you expect to see if your leading hypothesis holds.
  "action_role": one of {localize_source, rule_in, rule_out, assess_severity, broaden_search}.
  "grounding": a list of short quotes/values from the patient state that justify the above.
  "other_hypothesis": when belief argmax is 'other', name the leading non-rubric process; \
else "".

Route yourself: your belief argmax selects which disease path you are on. Choosing 'other' \
means leaving the rubric. Ground every claim in the patient state.

=== RUBRIC ===
__RUBRIC__"""


# --- patient-state rendering -------------------------------------------------
def render_patient(ps: dict) -> str:
    parts = ["## Baseline", json.dumps(ps.get("baseline", {}), ensure_ascii=False, indent=1)]
    vpi = ps.get("visible_prior_imaging") or []
    if vpi:
        parts.append("\n## Imaging available so far (this step's result is NOT shown)")
        for i, v in enumerate(vpi, 1):
            head = f"[{i}] {v.get('modality','')} {v.get('exam','') or v.get('region','')}".strip()
            parts.append(head + "\n" + str(v.get("report", "")).strip())
    else:
        parts.append("\n## Imaging available so far: none (this is the first imaging decision)")
    return "\n".join(parts)


def user_prompt(rec: dict) -> str:
    ps = rec["INPUT"]["patient_state"]
    active = rec["INPUT"].get("active_path")
    active_line = (f"Current working hypothesis (your belief argmax last step): {active}"
                   if active else "No prior step — route the differential from scratch.")
    return (render_patient(ps) + "\n\n## Your task\n" + active_line +
            "\nDecide the next imaging move and output the JSON object specified.")


def assistant_target(rec: dict) -> str:
    wt = rec["TARGET"]["why_trace"]
    belief = wt.get("belief") or {}
    obj = {
        "belief": {k: belief.get(k) for k in CANON_BRANCHES if k in belief} or belief,
        "modality": rec["TARGET"]["how_modality"],
        "information_gap": wt.get("information_gap", ""),
        "expected_finding": wt.get("expected_finding", ""),
        "action_role": wt.get("action_role", ""),
        "grounding": wt.get("grounding", []),
        "other_hypothesis": wt.get("other_hypothesis", "") or "",
    }
    return json.dumps(obj, ensure_ascii=False)


def to_example(rec: dict, system: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt(rec)},
            {"role": "assistant", "content": assistant_target(rec)},
        ],
        "meta": {                                    # NOT seen by the SFT loss; eval/RL use
            "id": rec["id"], "disease": rec["disease"], "hadm": rec["hadm"], "step": rec["step"],
            "when_action": rec["TARGET"]["when_action"],          # derived label, for eval
            "effective_branch": rec["TARGET"]["effective_branch"],
            "REWARD": rec["REWARD"],
            "CERTAINTY": rec["CERTAINTY"],
            "sample_weight": rec["REWARD"].get("sample_weight", 1.0),
        },
    }


# --- patient-level, disease-stratified split --------------------------------
def split_cases(rows: list[dict], val_frac: float, test_frac: float, seed: int) -> dict[tuple, str]:
    by_dis: dict[str, list[tuple]] = defaultdict(list)
    seen = set()
    for r in rows:
        key = (r["disease"], r["hadm"])
        if key not in seen:
            seen.add(key)
            by_dis[r["disease"]].append(key)
    rng = random.Random(seed)
    assign: dict[tuple, str] = {}
    for disease, cases in by_dis.items():
        cases = sorted(cases)
        rng.shuffle(cases)
        n = len(cases)
        n_test = max(1, round(n * test_frac)) if n >= 4 else 0
        n_val = max(1, round(n * val_frac)) if n >= 4 else 0
        for i, c in enumerate(cases):
            assign[c] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return assign


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/training_set/train_steps.jsonl")
    ap.add_argument("--rubric", default="data/training_set/rubric_library.json")
    ap.add_argument("--out", default="data/training_set/sft")
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
    rubric_text = compact_rubric(lib)
    system = SYSTEM_TEMPLATE.replace("__RUBRIC__", rubric_text)

    assign = split_cases(rows, args.val_frac, args.test_frac, args.seed)
    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in rows:
        buckets[assign[(r["disease"], r["hadm"])]].append(to_example(r, system))

    for name, exs in buckets.items():
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for e in exs:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # split provenance: rows + unique cases + per-disease, to confirm no patient leak
    def stats(exs):
        cases = {(e["meta"]["disease"], e["meta"]["hadm"]) for e in exs}
        return {"rows": len(exs), "cases": len(cases),
                "by_disease": dict(Counter(e["meta"]["disease"] for e in exs)),
                "when_action": dict(Counter(e["meta"]["when_action"] for e in exs))}
    overlap = (set((e["meta"]["disease"], e["meta"]["hadm"]) for e in buckets["train"])
               & set((e["meta"]["disease"], e["meta"]["hadm"]) for e in buckets["test"]))
    manifest = {
        "source": str(inp.relative_to(ROOT)),
        "system_prompt_chars": len(system),
        "rubric_chars": len(rubric_text),
        "target": "full reasoning trace (belief 5+other / modality / gap / expected / "
                  "action_role / grounding / other_hypothesis); deviate DERIVED at eval, "
                  "not generated",
        "behavior_spec_in_prompt": False,
        "split": {k: stats(v) for k, v in buckets.items()},
        "patient_leak_train_test": len(overlap),
        "config": vars(args),
    }
    (out_dir / "sft_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"system prompt: {len(system)} chars (rubric {len(rubric_text)})")
    for k in ("train", "val", "test"):
        s = manifest["split"][k]
        print(f"{k:5s}: {s['rows']:3d} rows / {s['cases']:3d} cases  "
              f"disease={s['by_disease']}  when={s['when_action']}")
    print(f"patient leak (train∩test cases): {len(overlap)}")
    print(f"wrote {out_dir}/(train|val|test).jsonl + sft_manifest.json")

    for e in buckets["train"][:args.preview]:
        print("\n" + "=" * 80)
        print("SYSTEM:", e["messages"][0]["content"][:300], "...")
        print("\nUSER:", e["messages"][1]["content"][:600], "...")
        print("\nASSISTANT:", e["messages"][2]["content"][:600])


if __name__ == "__main__":
    main()
