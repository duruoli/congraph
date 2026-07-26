"""Plan-T eval: read a calibrated P(deviate|input) from a Tinker checkpoint (or the bare base).

Same readout + metrics as scripts/eval_deviation_cls.py (the metric functions are IMPORTED, so
the panel stays comparable across venues) — only the model backend changes from HF/peft to
Tinker's SamplingClient, plus the reporting additions described under "What we report" below.

For each row we teacher-force the two candidate answer words after the generation prompt:
  z = logprob("deviate") - logprob("follow")   ->  raw P(deviate) = sigmoid(z)
arm a: the answer slot is right after the prompt (target is one word).
arm c (--generate-first): SAMPLE the reasoning trace first, cut at the "deviation" key, then
  score follow/deviate CONDITIONED on the model's own generated reasoning.
Platt (a,b) is fit on VAL and applied to TEST. Reads the ORIGINAL cls/cls_reason JSONL (keeps meta.y).

  # arm a (best-val checkpoint = epoch 1)
  python scripts/tinker/eval_prob_tinker.py --arm-name a \
      --checkpoint tinker://c000136e-.../sampler_weights/000034 \
      --data data/training_set/cls       --out results/agent_inspection/tinker_deviation_a
  # arm c (best-val checkpoint = epoch 2), reasoning generated first
  python scripts/tinker/eval_prob_tinker.py --arm-name c --generate-first \
      --checkpoint tinker://6621e009-.../sampler_weights/000068 \
      --data data/training_set/cls_reason --out results/agent_inspection/tinker_deviation_c
  # no --checkpoint => the bare instruct base (b-NEW pre-RL reference)

RENDERER (critical): the prompt is built with the cookbook renderer imported from sft_common
(`qwen3_5_disable_thinking`), NOT `tokenizer.apply_chat_template`. Qwen3.6 is hybrid: the chat
template defaults to THINKING mode and ends the prompt at `<think>\\n`, whereas SFT trained on the
disable-thinking prefix `<think>\\n\\n</think>\\n\\n`. Scoring off the wrong prefix would measure
the model off-distribution. See sft_common.py's module docstring, point 1.

What we report (and why), for a deliverable that is a CALIBRATED PROBABILITY, not a classifier:
  PRIMARY
    Brier + 95% CI  - the total score. Proper scoring rule: minimised only by reporting the true
                      probability, so it penalises both bad ranking and overconfidence.
    BSS   + 95% CI  - Brier Skill Score = 1 - Brier/Brier(const base rate). "What fraction of the
                      baseline uncertainty did the model remove." 0 = no better than predicting
                      the train base rate for everyone; negative = worse than doing nothing.
                      Its ceiling is Var(q)/base < 1, NOT 1 — do not read it as a percentage score.
    AUROC + 95% CI  - discrimination only (scale-free). Unchanged by Platt (a monotone map).
    ECE (5 equal-FREQUENCY bins) + the reliability table - calibration only. Equal-frequency
                      because with n=56 the equal-width bins are near-empty at the extremes; ECE is
                      biased upward at small n either way, so it is a cross-arm comparator, not an
                      absolute "x% miscalibrated".
  SECONDARY (appendix): AUPRC, log-loss, acc@0.5, ECE(10 equal-width) = the exact number the Quest
                      panel prints, kept for cross-venue comparability.
  DIAGNOSTIC: the spread of the predicted probabilities. A model can be perfectly calibrated and
                      useless (everyone gets 0.571), and RLVR fails the opposite way (everything
                      collapses to 0/1). Neither shows up in ECE alone.
  BASELINE ROW: constant train-base-rate predictor, so every number has an anchor.

LAYOUT: the main panel shows the BASELINE row and the CALIBRATED arm row only. The RAW
(uncalibrated) row is pushed to an APPENDIX at the bottom of the .txt — Platt is monotone so its
AUROC/AUPRC are identical to the calibrated row, and its Brier/ECE level is partly a tokenisation
artefact (see the note in `score_rows`), so it earns no space in the main read. Both are always
kept in the .json under metrics.{baseline,calibrated,raw}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for _p in (str(ROOT), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from experiments.llm_experiment.env_loader import load_tinker_key  # noqa: E402

# reuse the EXACT metric + Platt code from the HF eval (single source of truth)
from eval_deviation_cls import (  # noqa: E402
    CANDIDATES, acc_at, apply_platt, auprc, auroc, boot_ci, brier, ece, fit_platt, logloss,
)

# the renderer/base-model names the SFT actually used — never hardcode them twice
from sft_common import BASE_MODEL, RENDERER_NAME  # noqa: E402


# --------------------------- extra metrics (Tinker panel only) ---------------------------
def ece_equal_freq(y, p, bins=5):
    """ECE over EQUAL-COUNT, TIE-SAFE bins. Returns (ece, rows[(lo,hi,n,conf,acc)]).

    Equal-width bins (the imported `ece`) leave most of the 10 bins empty at n=56, which makes the
    summary number jumpy. Equal-frequency keeps every bin populated. Both are reported.

    Rows with the SAME predicted probability are forced into the SAME bin. Without that, a
    constant predictor gets chopped into 5 arbitrary groups whose realised rates differ by chance,
    and the ECE comes out ~0.11 instead of ~0.01 — pure binning artefact. This is not a corner
    case here: it is exactly the shape of an RLVR model whose predictions have collapsed onto
    0/1, i.e. the failure mode this panel exists to catch.
    """
    n = len(y)
    if n == 0:
        return float("nan"), []
    order = sorted(range(n), key=lambda i: p[i])
    e, rows, start = 0.0, [], 0
    for b in range(bins):
        cut = round((b + 1) * n / bins)
        while cut < n and p[order[cut]] == p[order[cut - 1]]:
            cut += 1                      # never split a tied group across a boundary
        idx = order[start:cut]
        start = cut
        if not idx:
            continue
        conf = sum(p[i] for i in idx) / len(idx)
        acc = sum(y[i] for i in idx) / len(idx)
        e += len(idx) / n * abs(acc - conf)
        rows.append((min(p[i] for i in idx), max(p[i] for i in idx), len(idx), conf, acc))
    return e, rows


def make_bss(base_rate):
    """Brier Skill Score against a constant `base_rate` predictor, as an (y, p) -> float fn.

    Built as a closure so `boot_ci` can resample it: on each bootstrap replicate the baseline is
    recomputed on the SAME resampled labels, which is what keeps the comparison honest.
    The base rate itself comes from TRAIN (never from test) so the baseline cannot peek.
    """
    def _bss(y, p):
        b_model = brier(y, p)
        b_base = brier(y, [base_rate] * len(y))
        if b_base == 0:
            return float("nan")
        return 1.0 - b_model / b_base
    return _bss


def spread_line(p):
    """One-line description of how far the predictions dare to move off the middle."""
    q = sorted(p)
    n = len(q)

    def pct(f):
        return q[min(n - 1, max(0, int(f * n)))]

    mid = sum(1 for x in p if 0.4 <= x <= 0.6) / n
    extreme = sum(1 for x in p if x < 0.1 or x > 0.9) / n
    return (f"min={q[0]:.3f} p25={pct(0.25):.3f} med={statistics.median(p):.3f} "
            f"p75={pct(0.75):.3f} max={q[-1]:.3f}  sd={statistics.pstdev(p):.3f}  "
            f"frac in [0.4,0.6]={mid:.2f}  frac <0.1 or >0.9={extreme:.2f}")


def report_arm(name, y, p, base_rate, lines):
    """Primary block + secondary block + reliability table. Returns the primary metrics dict."""
    b = brier(y, p)
    b_lo, b_hi = boot_ci(y, p, brier)
    bss_fn = make_bss(base_rate)
    s = bss_fn(y, p)
    s_lo, s_hi = boot_ci(y, p, bss_fn)
    a = auroc(y, p)
    a_lo, a_hi = boot_ci(y, p, auroc)
    e5, rows5 = ece_equal_freq(y, p, bins=5)
    e10, _ = ece(y, p, bins=10)

    lines.append(f"  [{name}]")
    lines.append(f"    PRIMARY    Brier={b:.4f} (95%CI {b_lo:.4f}-{b_hi:.4f})   "
                 f"BSS={s:+.4f} (95%CI {s_lo:+.4f}-{s_hi:+.4f})")
    lines.append(f"               AUROC={a:.4f} (95%CI {a_lo:.4f}-{a_hi:.4f})   "
                 f"ECE(5 eq-freq)={e5:.4f}")
    lines.append(f"    SECONDARY  AUPRC={auprc(y,p):.4f}  logloss={logloss(y,p):.4f}  "
                 f"acc@0.5={acc_at(y,p):.4f}  ECE(10 eq-width)={e10:.4f}")
    lines.append(f"    SPREAD     {spread_line(p)}")
    lines.append("    reliability (5 equal-frequency bins):")
    for lo, hi, cnt, conf, acc in rows5:
        lines.append(f"      p in [{lo:.3f},{hi:.3f}]  n={cnt:2d}  conf={conf:.3f}  "
                     f"actual={acc:.3f}  gap={acc - conf:+.3f}")
    return {"brier": b, "brier_ci": [b_lo, b_hi], "bss": s, "bss_ci": [s_lo, s_hi],
            "auroc": a, "auroc_ci": [a_lo, a_hi], "ece_eqfreq5": e5, "ece_eqwidth10": e10,
            "auprc": auprc(y, p), "logloss": logloss(y, p), "acc@0.5": acc_at(y, p)}


# --------------------------- Tinker scoring -----------------------------------
async def _score_one(row, sem, sampling_client, renderer, tokenizer, cand_ids,
                     generate_first, max_new_tokens, stop):
    """z = logprob(deviate) - logprob(follow) for one row."""
    from tinker import types

    msgs = row["messages"][:2]          # system + user; the assistant turn is what we score
    async with sem:
        prefill = None
        gen_text = None
        if generate_first:
            prompt = renderer.build_generation_prompt(msgs)
            params = types.SamplingParams(max_tokens=max_new_tokens, temperature=0.0, stop=stop)
            res = await sampling_client.sample_async(
                prompt=prompt, num_samples=1, sampling_params=params)
            gen_text = tokenizer.decode(res.sequences[0].tokens, skip_special_tokens=True)
            # condition on the model's OWN reasoning, cut just before the label it would emit
            if '"deviation"' in gen_text:
                prefill = gen_text.split('"deviation"')[0] + '"deviation": "'
            else:  # malformed: graft the key onto whatever JSON body it produced
                prefill = gen_text.rstrip().rstrip('}').rstrip().rstrip(',') + ', "deviation": "'

        prefix_ids = renderer.build_generation_prompt(msgs, prefill=prefill).to_ints()

        totals = {}
        for w, ids in cand_ids.items():
            full = types.ModelInput.from_ints(tokens=prefix_ids + ids)
            lp = await sampling_client.compute_logprobs_async(full)   # per-token over the WHOLE input
            tail = lp[-len(ids):]                                     # the candidate tokens
            if any(x is None for x in tail):
                raise RuntimeError(
                    f"compute_logprobs returned None for candidate {w!r} tokens {ids} — the "
                    f"candidate landed at position 0, i.e. the prompt is empty. Check the renderer.")
            totals[w] = float(sum(tail))
    return totals["deviate"] - totals["follow"], gen_text


async def score_rows(rows, sampling_client, renderer, tokenizer, generate_first,
                     max_new_tokens, concurrency, tag):
    # NOTE (Qwen3.6 tokenizer): 'follow' is ONE token [18000] but 'deviate' is TWO [3464, 6290].
    # Summing token logprobs gives the correct sequence probability of each WORD, so the
    # comparison is well-posed — but the 2-token word pays an extra sub-1 factor, biasing raw z
    # toward 'follow' by a roughly constant amount. Platt's intercept `b` absorbs exactly that,
    # so read the CALIBRATED row; the RAW row's absolute level is not meaningful across arms
    # (its ranking, hence AUROC, is unaffected).
    cand_ids = {w: tokenizer.encode(w, add_special_tokens=False) for w in CANDIDATES}
    stop = renderer.get_stop_sequences() or None
    sem = asyncio.Semaphore(concurrency)
    print(f"[{tag}] scoring {len(rows)} rows (concurrency {concurrency}); "
          f"token ids follow->{cand_ids['follow']} deviate->{cand_ids['deviate']}")
    out = await asyncio.gather(*[
        _score_one(r, sem, sampling_client, renderer, tokenizer, cand_ids,
                   generate_first, max_new_tokens, stop) for r in rows])
    return [z for z, _ in out], [g for _, g in out]


async def amain(args):
    import tinker
    from tinker_cookbook.renderers import get_renderer

    data = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    def _load(split):
        return [json.loads(l) for l in (data / f"{split}.jsonl").read_text().splitlines() if l.strip()]
    val, test = _load("val"), _load("test")
    yv = [r["meta"]["y"] for r in val]
    yt = [r["meta"]["y"] for r in test]

    # baseline prior comes from TRAIN only — the const baseline must not peek at test
    train_rows = _load("train")
    base_rate = sum(r["meta"]["y"] for r in train_rows) / len(train_rows)

    load_tinker_key()   # reads .tinker_env into TINKER_API_KEY (never `source` it)
    service = tinker.ServiceClient()
    sampling_client = service.create_sampling_client(
        base_model=args.base, model_path=args.checkpoint)   # model_path=None => bare base model
    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer(args.renderer, tokenizer)       # MUST match the SFT renderer

    zv, _ = await score_rows(val, sampling_client, renderer, tokenizer, args.generate_first,
                             args.max_new_tokens, args.concurrency, "val")
    zt, gen_t = await score_rows(test, sampling_client, renderer, tokenizer, args.generate_first,
                                 args.max_new_tokens, args.concurrency, "test")

    a, b = fit_platt(zv, yv)
    raw_t = [1.0 / (1.0 + math.exp(-z)) for z in zt]
    cal_t = apply_platt(zt, a, b)

    lines = [
        f"### Tinker P(deviate|input)   arm = {args.arm_name}",
        f"base={args.base}   checkpoint={args.checkpoint or '(none: bare base model)'}",
        f"renderer={args.renderer}   generate_first={args.generate_first}",
        f"val n={len(yv)} (pos {sum(yv)})   test n={len(yt)} (pos {sum(yt)})   "
        f"train base_rate={base_rate:.4f}",
        "",
        "CAVEATS: n=56 => DIRECTIONAL, CIs wide. Platt (a,b) fit on VAL, applied to TEST — and the",
        "checkpoint was ALSO selected on val NLL, so val is used twice (mild at 2 params, but state",
        "it). Everything below is Platt-CALIBRATED; the raw scale is in the appendix at the bottom.",
        "BSS ceiling is Var(q)/base < 1, so a 'low' BSS is not a failed model: for",
        "behavioural prediction with causally-masked inputs, 0.05-0.15 is real signal, ~0 is not.",
        "",
    ]

    # ---- baseline: predict the train base rate for everyone -------------------
    lines.append("TEST — BASELINE (knows only the overall deviation rate; the anchor for BSS=0)")
    p_const = [base_rate] * len(yt)
    m_const = report_arm("const-base-rate", yt, p_const, base_rate, lines)
    lines.append("")

    lines.append(f"TEST — ARM {args.arm_name}   (Platt-calibrated; a={a:.4f} b={b:.4f}, fit on val)")
    m_cal = report_arm(args.arm_name, yt, cal_t, base_rate, lines)

    # ---- appendix: the model's NATIVE scale, kept out of the main read ---------
    # Platt is monotone => AUROC/AUPRC are identical to the calibrated row above; only
    # Brier/BSS/ECE/logloss differ. And the raw level is partly an artefact of 'deviate' costing
    # two tokens vs 'follow' one, so it is not comparable across arms. It is here to show how far
    # off the model's native scale was, nothing more.
    appendix = ["", "-" * 78,
                f"APPENDIX — arm {args.arm_name}, RAW (uncalibrated) scale. NOT for the main panel:",
                "  AUROC/AUPRC are identical to the calibrated row (Platt is monotone). The raw",
                "  Brier/BSS/ECE level is inflated by 'deviate' being a 2-token word vs 'follow'",
                "  1 token, a near-constant offset Platt removes — do not compare it across arms.",
                ""]
    m_raw = report_arm(f"{args.arm_name} RAW", yt, raw_t, base_rate, appendix)
    lines.extend(appendix)

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps({
        "arm": args.arm_name, "base": args.base, "checkpoint": args.checkpoint,
        "renderer": args.renderer, "generate_first": args.generate_first,
        "train_base_rate": base_rate, "platt": {"a": a, "b": b},
        "metrics": {"baseline": m_const, "raw": m_raw, "calibrated": m_cal},
        "test_z": zt, "test_raw": raw_t, "test_cal": cal_t, "test_y": yt,
        "val_z": zv, "val_y": yv,
    }, indent=2))
    if args.dump_generations and args.generate_first:
        (out.parent / f"{out.name}_generations.jsonl").write_text(
            "\n".join(json.dumps({"i": i, "y": yt[i], "p": cal_t[i], "generation": g},
                                 ensure_ascii=False) for i, g in enumerate(gen_t)) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}.txt / .json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--checkpoint", default=None,
                    help="tinker:// sampler_path of the trained LoRA; omit to score the bare base")
    ap.add_argument("--renderer", default=RENDERER_NAME,
                    help="MUST match the renderer SFT used (see sft_common docstring)")
    ap.add_argument("--arm-name", default="a", help="label for the report (a / c / c_rl / b_new)")
    ap.add_argument("--data", default="data/training_set/cls",
                    help="cls for arm a; cls_reason for arm c (+ --generate-first)")
    ap.add_argument("--generate-first", action="store_true",
                    help="arm c/c_rl/b_new: sample the reasoning, then score the label after it")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="traces reach ~700 tok before the deviation key; 512 truncates ~half")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dump-generations", action="store_true",
                    help="also write <out>_generations.jsonl for qualitative trace audit")
    ap.add_argument("--out", default="results/agent_inspection/tinker_deviation")
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
