"""Systematic 5-ARM panel for the certainty-trigger agent (direction-A context vs
direction-B SFT), settled in [[sft-eval-design]].

ARMS (each = a (base_model, delivery) tuple; the base is a SWAPPABLE flag — default is now
google/medgemma-27b-text-it (Gemma-3), Qwen2.5-7B was the earlier placeholder). The three
model arms below share the SAME base; the arm ids keep the historical "-qwen" suffix but track
whatever --base-model is passed:
  doctor      real trajectory (gold annotation) — a REFERENCE, NOT a correctness standard.
  base        base model, plain system prompt.                                 [floor]
  sft         base + LoRA adapter, plain system prompt.                        [dir-B weights]
  ctx-qwen    base (NO adapter) + configs/context_block.md in system.          [dir-A, SAME base
              as sft => ctx-qwen-vs-sft is a clean MECHANISM ablation]
  ctx-sonnet  Sonnet + context_block.md.  [best-achievable context product; the Sonnet-vs-Qwen
              capability gap is a NAMED CONFOUND — printed in the header, do not over-read]

COMMON YARDSTICK = the rubric, run as a policy: each arm's belief-argmax picks the sub-rubric,
traverse it on the step's causally-masked pre-decision features, and label the arm's ordered
modality follow / deviate / off_rubric via experiments.annotation.deviation.belief_step_deviation
(the SAME function that labelled the doctor). The doctor's own follow/deviate is precomputed in
train_steps.jsonl META.dev_belief — we RECOMPUTE it here from reconstructed pre_features and
ASSERT it matches, which validates the pre_features alignment before trusting any arm's number.

MODALITY IS SCORED ON THE DISTRIBUTION, not a single greedy argmax (the debug run showed greedy
collapses SFT to CT while ~0.4 US mass remains). For model arms we also report follow-PROBABILITY
= sum of P(modality) over the rubric-recommended set, robust to greedy fragility.

CORRECTNESS ANCHOR (doctor-free) = deviation x alarm landing: does each arm's deviate concentrate
on alarm-FIRED steps and its follow on alarm-SILENT steps? Alarm channels = study_inadequacy (A2)
and discordance (A3), which are the two situational alarms the context block names. NOTE A1
(diagnosis-settled / question-moved) has NO alarm channel, so alarm-landing cannot capture it.

HONEST LIMITS (also printed): N=56 clean test -> DIRECTIONAL, not significant; the brake side is
near-empty in this data; the rubric is NOT ground truth (agent != rubric can be a justified patch
OR a hallucinated over-test — this panel measures BEHAVIOR/direction, not correctness).

Systematic metrics run on the CLEAN random test split ONLY. The alarm_probe subset (--data ...
alarm_probe.jsonl) is BEHAVIORAL/qualitative and HAS LEAK — never read its numbers as fidelity.

  # CPU-only, no model — validates the spine + doctor arm + all metrics:
  python scripts/eval_certainty_agent.py --arms doctor
  # full panel on Quest:
  python scripts/eval_certainty_agent.py --arms doctor base sft ctx-qwen ctx-sonnet
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.annotation.deviation import belief_step_deviation, IMAGING_KEYS  # noqa: E402
from experiments.agent.chat_compat import assert_roundtrip  # noqa: E402

DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]
MODALITIES = ["CT_Abdomen", "Ultrasound_Abdomen", "MRCP_Abdomen", "MRI_Abdomen", "CTU_Abdomen"]
ALL_ARMS = ["doctor", "base", "sft", "ctx-qwen", "ctx-sonnet"]
CONTEXT_FILE = ROOT / "configs/context_block.md"
DEFAULT_TEST = ROOT / "data/training_set/sft/test.jsonl"
TRAIN_STEPS = ROOT / "data/training_set/train_steps.jsonl"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["doctor"], choices=ALL_ARMS)
    ap.add_argument("--data", default=str(DEFAULT_TEST),
                    help="test.jsonl (clean, fidelity) or alarm_probe.jsonl (behavioral, HAS LEAK)")
    ap.add_argument("--base-model", default="google/medgemma-27b-text-it",
                    help="swappable base; medgemma-27b (Gemma-3) default, Qwen-7B was the placeholder")
    ap.add_argument("--adapter", default="runs/medgemma-27b-lora-certainty",
                    help="LoRA adapter (HF repo id or local dir) — must match --base-model")
    ap.add_argument("--sonnet-model", default="anthropic/claude-sonnet-4-6")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0, help="cap rows (debug); 0 = all")
    ap.add_argument("--out", default="results/agent_inspection/eval_panel.txt")
    return ap.parse_args()


# --- pre_features reconstruction (mirrors scripts/analyze_belief_deviation.py) --------------
def _load_features() -> dict[str, dict]:
    out = {}
    for d in DISEASES:
        out[d] = json.load(open(ROOT / "data/rubric_features" / f"{d}_features.json"))["results"]
    return out


def _prefeatures_by_case(FE: dict) -> dict[tuple, list[dict]]:
    """(disease, hadm) -> [pre_features for the k-th IMAGING decision step], accumulative
    idx_{k-1}. Mirrors analyze_belief_deviation: the state before imaging-step k is the feature
    record just before the k-th imaging test_key in RAW record order. Guard idx==0 -> {} (root)."""
    by_case: dict[tuple, list[dict]] = {}
    for disease, fe_disease in FE.items():
        for hadm_s, fe_steps in fe_disease.items():
            img_idx = [i for i, s in enumerate(fe_steps) if s["test_key"] in IMAGING_KEYS]
            pres = [(fe_steps[i - 1]["features"] if i - 1 >= 0 else {}) for i in img_idx]
            by_case[(disease, int(hadm_s))] = pres
    return by_case


# --- agent output abstraction --------------------------------------------------------------
class AgentOut:
    __slots__ = ("belief", "modality", "modality_dist", "other_hyp", "parsed_ok", "raw")

    def __init__(self, belief, modality, other_hyp, modality_dist=None, parsed_ok=True, raw=""):
        self.belief = belief or {}
        self.modality = modality
        self.modality_dist = modality_dist   # dict|None (None for doctor / unparsed)
        self.other_hyp = other_hyp or ""
        self.parsed_ok = parsed_ok
        self.raw = raw

    @property
    def belief_argmax(self):
        if not self.belief:
            return "other"
        return max(self.belief, key=self.belief.get)


def status_for(out: AgentOut, pre: dict) -> tuple[str, str, list[str], float | None]:
    """(effective_branch, follow/deviate/off_rubric, rubric_rec, follow_prob|None)."""
    if out.modality is None:
        return ("", "unparsed", [], None)
    eff, dev, rec = belief_step_deviation(out.belief_argmax, pre, out.modality, out.other_hyp)
    fprob = None
    if out.modality_dist is not None and rec:
        fprob = sum(p for m, p in out.modality_dist.items() if m in rec)
    elif out.modality_dist is not None:
        fprob = 0.0
    return (eff, dev, rec, fprob)


# --- arms ----------------------------------------------------------------------------------
def _extract_json(text: str) -> dict | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _out_from_json(p: dict | None, with_dist=None) -> AgentOut:
    if not p:
        return AgentOut(None, None, None, parsed_ok=False)
    return AgentOut(p.get("belief"), p.get("modality"), p.get("other_hypothesis"),
                    modality_dist=with_dist, parsed_ok=True)


def run_doctor(rows) -> list[AgentOut]:
    """Gold annotation = the doctor's real action. No model."""
    outs = []
    for r in rows:
        g = json.loads(r["messages"][-1]["content"])
        outs.append(AgentOut(g.get("belief"), g.get("modality"), g.get("other_hypothesis")))
    return outs


def run_qwen_arms(rows, arms, args):
    """base / sft / ctx-qwen share ONE loaded Qwen (adapter toggled; system prompt swapped).
    Each arm -> per-row AgentOut with a teacher-forced modality distribution over the 5 legal
    modalities, computed on the model's OWN generated belief (greedy-robust follow-prob)."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    print(f"[hf] base={args.base_model} device={device} dtype={dtype}")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # fail loud if the swapped base's template can't render [system, user] (e.g. rejects system)
    if rows:
        assert_roundtrip(tok, rows[0]["messages"][:2])
    load_kwargs = dict(torch_dtype=dtype)
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    if device != "cuda":
        model = model.to(device)
    need_adapter = "sft" in arms
    if need_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    ctx = CONTEXT_FILE.read_text() if any(a.startswith("ctx-qwen") for a in arms) else None
    cand_ids = {m: tok(m + '"', add_special_tokens=False).input_ids for m in MODALITIES}

    def sysmsg(row, with_ctx):
        base = row["messages"][0]["content"]
        return base + "\n\n=== ADDITIONAL CLINICAL OVERRIDE KNOWLEDGE ===\n" + ctx if with_ctx else base

    def gen_and_probe(row, with_ctx):
        messages = [{"role": "system", "content": sysmsg(row, with_ctx)}, row["messages"][1]]
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                  do_sample=False, pad_token_id=tok.pad_token_id)
        text = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        p = _extract_json(text)
        dist = None
        if p and isinstance(p.get("belief"), dict):
            prefix_txt = '{"belief": ' + json.dumps(
                {k: float(p["belief"].get(k, 0.0)) for k in
                 ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis", "other"]}
            ) + ', "modality": "'
            pref = tok(prefix_txt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            prefix_ids = torch.cat([enc["input_ids"][0], pref.to(model.device)])
            totals = {}
            s0 = prefix_ids.shape[0] - 1
            for m, ids in cand_ids.items():
                cont = torch.tensor(ids, device=prefix_ids.device)
                full = torch.cat([prefix_ids, cont]).unsqueeze(0)
                with torch.no_grad():
                    logits = model(full).logits[0]           # [seq, vocab] (bf16)
                    # slice the answer rows BEFORE .float() — upcasting the full seq x 262k-vocab
                    # tensor is an ~8GB alloc that OOMs the 27B on an 80GB card.
                    lp = F.log_softmax(logits[s0:s0 + len(ids)].float(), dim=-1)
                totals[m] = sum(lp[j, t].item() for j, t in enumerate(ids))
                del logits, lp
            probs = F.softmax(torch.tensor([totals[m] for m in MODALITIES]), dim=0)
            dist = {m: probs[i].item() for i, m in enumerate(MODALITIES)}
        return _out_from_json(p, with_dist=dist), text

    results: dict[str, list[AgentOut]] = {a: [] for a in arms if a in ("base", "sft", "ctx-qwen")}
    for row in rows:
        if "base" in results:
            if need_adapter:
                with model.disable_adapter():
                    o, _ = gen_and_probe(row, with_ctx=False)
            else:
                o, _ = gen_and_probe(row, with_ctx=False)
            results["base"].append(o)
        if "sft" in results:
            o, _ = gen_and_probe(row, with_ctx=False)   # adapter ON
            results["sft"].append(o)
        if "ctx-qwen" in results:
            # ctx-qwen = BASE weights (adapter off) + context in system
            if need_adapter:
                with model.disable_adapter():
                    o, _ = gen_and_probe(row, with_ctx=True)
            else:
                o, _ = gen_and_probe(row, with_ctx=True)
            results["ctx-qwen"].append(o)
    return results


def run_ctx_sonnet(rows, args) -> list[AgentOut]:
    """Sonnet via the house OpenRouter client + context_block. No logprobs -> modality argmax only."""
    from openai import OpenAI
    from experiments.llm_experiment.env_loader import load_openrouter_key
    client = OpenAI(api_key=load_openrouter_key(),
                    base_url="https://openrouter.ai/api/v1", max_retries=5, timeout=90.0)
    ctx = CONTEXT_FILE.read_text()
    outs = []
    for row in rows:
        sys_prompt = (row["messages"][0]["content"]
                      + "\n\n=== ADDITIONAL CLINICAL OVERRIDE KNOWLEDGE ===\n" + ctx)
        resp = client.chat.completions.create(
            model=args.sonnet_model, temperature=0.0, max_tokens=args.max_new_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": row["messages"][1]["content"]}])
        outs.append(_out_from_json(_extract_json(resp.choices[0].message.content or "")))
    return outs


# --- metrics -------------------------------------------------------------------------------
def profile(statuses: list[str]) -> str:
    c = Counter(statuses)
    n = sum(c.values()) or 1
    return (f"follow {c['follow']:3d} ({100*c['follow']/n:3.0f}%)  "
            f"deviate {c['deviate']:3d} ({100*c['deviate']/n:3.0f}%)  "
            f"off_rub {c['off_rubric']:3d} ({100*c['off_rubric']/n:3.0f}%)"
            + (f"  unparsed {c['unparsed']}" if c['unparsed'] else ""))


def main():
    args = parse_args()
    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    TS = {}
    for line in TRAIN_STEPS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            TS[r["id"]] = r
    PRE = _prefeatures_by_case(_load_features())

    # attach per-row spine: pre_features, doctor META, alarm cell, judged flag
    spine = []
    for r in rows:
        m = r["meta"]
        ts = TS.get(m["id"], {})
        meta_ts = ts.get("META", {})
        pres = PRE.get((m["disease"], int(m["hadm"])), [])
        step_i = int(m["step"]) - 1
        pre = pres[step_i] if 0 <= step_i < len(pres) else None
        al = m.get("CERTAINTY", {}).get("alarm", {})
        spine.append({
            "id": m["id"], "disease": m["disease"], "step": m["step"],
            "when_action": m.get("when_action"),
            "doctor_dev": meta_ts.get("dev_belief"),
            "top_branch": meta_ts.get("top_branch"),
            "rrn_aligned": meta_ts.get("rrn_aligned", False),
            "pre": pre,
            "alarm_any": bool(al.get("any_trigger")),
            "study_inadeq": bool(al.get("study_inadequacy", {}).get("present")),
            "discordance": bool(al.get("discordance", {}).get("present")),
            "source_split": m.get("source_split"), "leak": m.get("leak"),
        })

    judged = [s for s in spine if s["rrn_aligned"] and s["pre"] is not None]

    # run requested arms
    arm_outs: dict[str, list[AgentOut]] = {}
    if "doctor" in args.arms:
        arm_outs["doctor"] = run_doctor(rows)
    qwen_arms = [a for a in args.arms if a in ("base", "sft", "ctx-qwen")]
    if qwen_arms:
        arm_outs.update(run_qwen_arms(rows, qwen_arms, args))
    if "ctx-sonnet" in args.arms:
        arm_outs["ctx-sonnet"] = run_ctx_sonnet(rows, args)

    # index outs by row id
    outs_by_arm_id = {a: {rows[i]["meta"]["id"]: o for i, o in enumerate(outs)}
                      for a, outs in arm_outs.items()}

    buf = []

    def emit(s=""):
        print(s)
        buf.append(s)

    is_probe = "alarm_probe" in args.data
    emit("### 5-arm certainty-agent eval panel")
    emit(f"data={args.data}  rows={len(rows)}  judged(rrn_aligned&pre)={len(judged)}  "
         f"arms={args.arms}  base={args.base_model}")
    if is_probe:
        emit("!! ALARM_PROBE subset: BEHAVIORAL/qualitative only, HAS LEAK — do NOT read as fidelity.")
    emit("CAVEATS: rubric is NOT ground truth (agent!=rubric = patch OR over-test, undecidable here); "
         "N small => directional; brake side near-empty; ctx-sonnet's Sonnet base is a capability "
         "CONFOUND vs the Qwen arms; modality read on DISTRIBUTION (follow-prob) not greedy argmax.\n")

    # -- pre_features validation against stored doctor dev_belief --
    if "doctor" in arm_outs:
        mism = []
        for s in judged:
            o = outs_by_arm_id["doctor"][s["id"]]
            _, dev, _, _ = status_for(o, s["pre"])
            # recompute with the doctor's stored top_branch too (belief argmax may differ slightly)
            if s["doctor_dev"] and dev != s["doctor_dev"]:
                mism.append((s["id"], dev, s["doctor_dev"]))
        emit(f"[validation] pre_features vs stored META.dev_belief: "
             f"{len(judged)-len(mism)}/{len(judged)} match"
             + (f"  MISMATCHES={mism[:5]}" if mism else "  ✓"))
        emit("")

    # -- Block 1+2: follow/deviate/off_rubric profile per arm (overall / disease / when_action) --
    emit("=" * 92)
    emit("BLOCK 1-2 — vs-rubric follow/deviate/off_rubric profile (judged steps)")
    for arm in args.arms:
        if arm not in outs_by_arm_id:
            continue
        st = [status_for(outs_by_arm_id[arm][s["id"]], s["pre"])[1] for s in judged]
        emit(f"\n[{arm}]  OVERALL  {profile(st)}")
        by_d = defaultdict(list)
        for s in judged:
            by_d[s["disease"]].append(status_for(outs_by_arm_id[arm][s["id"]], s["pre"])[1])
        for d in DISEASES:
            if by_d[d]:
                emit(f"    {d:15s} {profile(by_d[d])}")
        by_w = defaultdict(list)
        for s in judged:
            by_w[s["when_action"]].append(status_for(outs_by_arm_id[arm][s["id"]], s["pre"])[1])
        for w in ["follow", "deviate", "off_rubric"]:
            if by_w[w]:
                emit(f"    doctor={w:11s} {profile(by_w[w])}")
        # soft follow-prob (model arms only)
        fps = [status_for(outs_by_arm_id[arm][s["id"]], s["pre"])[3] for s in judged]
        fps = [x for x in fps if x is not None]
        if fps:
            emit(f"    mean follow-PROB (distribution, greedy-robust) = {sum(fps)/len(fps):.3f}")

    # -- Block 3: deviation x alarm landing (doctor-free correctness anchor) --
    emit("\n" + "=" * 92)
    emit("BLOCK 3 — deviation x alarm landing (anchor): P(deviate|alarm) should exceed "
         "P(deviate|silent); P(follow|silent) should exceed P(follow|alarm)")
    for channel in ["alarm_any", "study_inadeq", "discordance"]:
        emit(f"\n  channel = {channel}")
        for arm in args.arms:
            if arm not in outs_by_arm_id:
                continue
            fired = [s for s in judged if s[channel]]
            silent = [s for s in judged if not s[channel]]

            def rate(sub, want):
                if not sub:
                    return float("nan"), 0
                sts = [status_for(outs_by_arm_id[arm][s["id"]], s["pre"])[1] for s in sub]
                return sum(x == want for x in sts) / len(sub), len(sub)

            dvf, nf = rate(fired, "deviate")
            dvs, ns = rate(silent, "deviate")
            flf, _ = rate(fired, "follow")
            fls, _ = rate(silent, "follow")
            emit(f"    [{arm:10s}] deviate|fired(n={nf})={dvf:.2f} |silent(n={ns})={dvs:.2f} "
                 f"Δ={dvf-dvs:+.2f}   follow|fired={flf:.2f} |silent={fls:.2f} Δ={flf-fls:+.2f}")

    # write outputs
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(buf) + "\n")
    rec_path = out_path.with_suffix(".jsonl")
    with rec_path.open("w") as f:
        for s in judged:
            rec = {**{k: s[k] for k in ("id", "disease", "step", "when_action", "doctor_dev",
                                        "alarm_any", "study_inadeq", "discordance")}}
            for arm in args.arms:
                if arm in outs_by_arm_id:
                    o = outs_by_arm_id[arm][s["id"]]
                    eff, dev, rec_img, fp = status_for(o, s["pre"])
                    rec[arm] = {"belief_argmax": o.belief_argmax if o.modality else None,
                                "modality": o.modality, "status": dev,
                                "rubric_rec": rec_img, "follow_prob": fp,
                                "modality_dist": o.modality_dist}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    emit(f"\n[wrote] {out_path}  (+ {rec_path.name}, {len(judged)} judged rows)")


if __name__ == "__main__":
    main()
