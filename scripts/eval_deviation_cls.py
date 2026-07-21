"""Evaluate + calibrate the binary deviation classifier P(deviate | input).

Reads data/training_set/cls/{val,test}.jsonl (built by build_deviation_cls.py), loads a
base medgemma + LoRA adapter, and for each row reads a probability by TEACHER-FORCING the
two candidate answer words after the generation prompt (mirrors eval_certainty_agent.py's
candidate-scoring): score(w) = sum token log-prob of w; raw P(deviate) = softmax over
{follow, deviate} = sigmoid(z), z = score_deviate - score_follow.

Calibration = Platt scaling: fit sigmoid(a*z + b) on VAL (minimize NLL), apply to TEST.
Reports AUROC / AUPRC / Brier / log-loss / acc@0.5 / ECE (reliability bins), with bootstrap
CIs, for arms {base (adapter off), sft (adapter on)} plus a constant base-rate baseline.
Small N (test=56) => everything is DIRECTIONAL; CIs are wide by construction.

Run ON A GPU node (medgemma-27b is ~54 GB bf16). Example:
  python scripts/eval_deviation_cls.py \
      --base google/medgemma-27b-text-it \
      --adapter runs/medgemma-27b-lora-deviate-cls \
      --data data/training_set/cls
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ["follow", "deviate"]   # index 1 = positive class (deviate)


# --------------------------- metrics (no sklearn dep) ------------------------
def auroc(y, p):
    pos = [pi for pi, yi in zip(p, y) if yi == 1]
    neg = [pi for pi, yi in zip(p, y) if yi == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranks = [0.0] * len(p)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and p[order[j]] == p[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0  # 1-based average rank for ties
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    sum_pos = sum(ranks[i] for i in range(len(p)) if y[i] == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auprc(y, p):
    order = sorted(range(len(p)), key=lambda i: -p[i])
    tp = fp = 0
    n_pos = sum(y)
    if n_pos == 0:
        return float("nan")
    prev_recall, area, prec = 0.0, 0.0, 1.0
    for i in order:
        if y[i] == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        recall = tp / n_pos
        area += prec * (recall - prev_recall)
        prev_recall = recall
    return area


def brier(y, p):
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(y)


def logloss(y, p, eps=1e-12):
    return -sum(yi * math.log(max(pi, eps)) + (1 - yi) * math.log(max(1 - pi, eps))
               for yi, pi in zip(y, p)) / len(y)


def acc_at(y, p, thr=0.5):
    return sum((pi >= thr) == bool(yi) for pi, yi in zip(p, y)) / len(y)


def ece(y, p, bins=10):
    edges = [i / bins for i in range(bins + 1)]
    tot, e = len(y), 0.0
    rows = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        idx = [i for i in range(tot) if (p[i] >= lo and (p[i] < hi or (b == bins - 1 and p[i] <= hi)))]
        if not idx:
            rows.append((lo, hi, 0, None, None))
            continue
        conf = sum(p[i] for i in idx) / len(idx)
        acc = sum(y[i] for i in idx) / len(idx)
        e += len(idx) / tot * abs(acc - conf)
        rows.append((lo, hi, len(idx), conf, acc))
    return e, rows


def boot_ci(y, p, fn, n=2000, seed=0):
    rng = random.Random(seed)
    N = len(y)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(N) for _ in range(N)]
        v = fn([y[i] for i in idx], [p[i] for i in idx])
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    if not vals:
        return (float("nan"), float("nan"))
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1])


# --------------------------- Platt scaling -----------------------------------
def fit_platt(z, y, iters=500, lr=0.1):
    """sigmoid(a*z + b) fit on val by NLL (simple full-batch gradient descent)."""
    a, b = 1.0, 0.0
    n = len(z)
    for _ in range(iters):
        ga = gb = 0.0
        for zi, yi in zip(z, y):
            pi = 1.0 / (1.0 + math.exp(-(a * zi + b)))
            ga += (pi - yi) * zi
            gb += (pi - yi)
        a -= lr * ga / n
        b -= lr * gb / n
    return a, b


def apply_platt(z, a, b):
    return [1.0 / (1.0 + math.exp(-(a * zi + b))) for zi in z]


# --------------------------- model scoring -----------------------------------
def score_rows(rows, base_model, adapter, arms, generate_first=False, max_new_tokens=512):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    print(f"[hf] base={base_model} device={device} dtype={dtype}")
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    load_kwargs = dict(torch_dtype=dtype)
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    if device != "cuda":
        model = model.to(device)
    need_adapter = "sft" in arms
    if need_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    cand_ids = {w: tok(w, add_special_tokens=False).input_ids for w in CANDIDATES}
    print(f"[tok] follow->{cand_ids['follow']}  deviate->{cand_ids['deviate']}")

    def z_of(row):
        # z = logprob(deviate) - logprob(follow) at the answer slot.
        # approach a (default): the answer slot is right after the generation prompt (target = 1 word).
        # approach c (--generate-first): first GENERATE the reasoning trace, then read the deviate/
        #   follow token CONDITIONED on the model's own generated reasoning (up to the "deviation" key).
        enc = tok.apply_chat_template(row["messages"][:2], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prefix = enc["input_ids"][0]
        if generate_first:
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            text = tok.decode(gen[0, prefix.shape[0]:], skip_special_tokens=True)
            # cut the model's own generation at the deviation key; re-score the value ourselves
            if '"deviation"' in text:
                reason = text.split('"deviation"')[0] + '"deviation": "'
            else:  # malformed: append the key onto whatever JSON body it produced
                reason = text.rstrip().rstrip('}').rstrip().rstrip(',') + ', "deviation": "'
            pref = tok(reason, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            prefix = torch.cat([prefix, pref.to(model.device)])
        totals = {}
        for w, ids in cand_ids.items():
            cont = torch.tensor(ids, device=prefix.device)
            full = torch.cat([prefix, cont]).unsqueeze(0)
            with torch.no_grad():
                lp = F.log_softmax(model(full).logits[0].float(), dim=-1)
            s0 = prefix.shape[0] - 1
            totals[w] = sum(lp[s0 + j, t].item() for j, t in enumerate(ids))
        return totals["deviate"] - totals["follow"]

    out = {a: [] for a in arms}
    for row in rows:
        if "base" in out:
            if need_adapter:
                with model.disable_adapter():
                    out["base"].append(z_of(row))
            else:
                out["base"].append(z_of(row))
        if "sft" in out:
            out["sft"].append(z_of(row))   # adapter ON
    return out


# --------------------------- reporting ---------------------------------------
def report(name, y, p, lines):
    a_lo, a_hi = boot_ci(y, p, auroc)
    e, bins = ece(y, p)
    lines.append(f"  [{name}]  AUROC={auroc(y,p):.3f} (95%CI {a_lo:.3f}-{a_hi:.3f})  "
                 f"AUPRC={auprc(y,p):.3f}  Brier={brier(y,p):.3f}  "
                 f"logloss={logloss(y,p):.3f}  acc@0.5={acc_at(y,p):.3f}  ECE={e:.3f}")
    return bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/medgemma-27b-text-it")
    ap.add_argument("--adapter", default="runs/medgemma-27b-lora-deviate-cls")
    ap.add_argument("--data", default="data/training_set/cls")
    ap.add_argument("--arms", nargs="+", default=["base", "sft"])
    ap.add_argument("--generate-first", action="store_true",
                    help="approach C: generate the reasoning trace, then read the deviate/follow "
                         "token conditioned on it (use with the cls_reason data + devreason adapter)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", default="results/agent_inspection/deviation_cls_eval")
    args = ap.parse_args()

    data = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    val = [json.loads(l) for l in (data / "val.jsonl").read_text().splitlines() if l.strip()]
    test = [json.loads(l) for l in (data / "test.jsonl").read_text().splitlines() if l.strip()]
    yv = [r["meta"]["y"] for r in val]
    yt = [r["meta"]["y"] for r in test]
    train_base_rate = None
    tr = data / "train.jsonl"
    if tr.exists():
        ytr = [json.loads(l)["meta"]["y"] for l in tr.read_text().splitlines() if l.strip()]
        train_base_rate = sum(ytr) / len(ytr)

    zv = score_rows(val, args.base, args.adapter, args.arms,
                    generate_first=args.generate_first, max_new_tokens=args.max_new_tokens)
    zt = score_rows(test, args.base, args.adapter, args.arms,
                    generate_first=args.generate_first, max_new_tokens=args.max_new_tokens)

    lines = ["### binary deviation classifier — P(deviate|input)",
             f"base={args.base}  adapter={args.adapter}",
             f"val n={len(yv)} (pos {sum(yv)})  test n={len(yt)} (pos {sum(yt)})  "
             f"train base_rate={train_base_rate}",
             "CAVEAT: N small => DIRECTIONAL; CIs wide. Calibration (a,b) fit on VAL, applied to TEST.",
             ""]
    dump = {"base": args.base, "adapter": args.adapter, "arms": {}}

    # constant base-rate baseline (predict train pos rate for everyone)
    if train_base_rate is not None:
        p_const = [train_base_rate] * len(yt)
        lines.append("TEST — baselines")
        report("const-base-rate", yt, p_const, lines)
        lines.append("")

    for arm in args.arms:
        raw_v = [1.0 / (1.0 + math.exp(-z)) for z in zv[arm]]
        raw_t = [1.0 / (1.0 + math.exp(-z)) for z in zt[arm]]
        a, b = fit_platt(zv[arm], yv)
        cal_t = apply_platt(zt[arm], a, b)
        lines.append(f"ARM = {arm}   (platt a={a:.3f} b={b:.3f} fit on val)")
        report(f"{arm} test RAW", yt, raw_t, lines)
        bins = report(f"{arm} test CALIBRATED", yt, cal_t, lines)
        lines.append("    reliability (calibrated, test):")
        for lo, hi, cnt, conf, acc in bins:
            if cnt:
                lines.append(f"      [{lo:.1f},{hi:.1f}) n={cnt:2d} conf={conf:.2f} acc={acc:.2f}")
        lines.append("")
        dump["arms"][arm] = {
            "platt": {"a": a, "b": b},
            "test_z": zt[arm], "test_raw": raw_t, "test_cal": cal_t, "test_y": yt,
            "metrics_cal": {"auroc": auroc(yt, cal_t), "auprc": auprc(yt, cal_t),
                            "brier": brier(yt, cal_t), "logloss": logloss(yt, cal_t),
                            "acc@0.5": acc_at(yt, cal_t), "ece": ece(yt, cal_t)[0]},
        }

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text("\n".join(lines))
    out.with_suffix(".json").write_text(json.dumps(dump, indent=2))
    print("\n".join(lines))
    print(f"\nwrote {out}.txt / .json")


if __name__ == "__main__":
    main()
