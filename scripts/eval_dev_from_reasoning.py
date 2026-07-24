"""Approach B — DERIVE a calibrated P(doctor deviates | input) from the EXISTING reasoning SFT
adapter (runs/medgemma-27b-lora-certainty). NO new training.

Deviation is treated as a pure consequence of the reasoning model's own belief + modality
distribution: P(deviate) = 1 - follow_prob, where
    follow_prob = sum_m P(modality=m) over m in the rubric-recommended set,
teacher-forced on the model's OWN generated belief. That follow_prob is already produced,
per judged row, by scripts/eval_certainty_agent.py (the `sft` arm column of its
results/agent_inspection/eval_panel*.jsonl dump; off_rubric -> follow_prob 0 -> P(deviate)=1,
consistent with the binary {follow=0, deviate|off_rubric=1} label of approaches a/c).

So this script does NO model inference. It just reads two eval_certainty_agent panel dumps
(a VAL pass and a TEST pass), maps follow_prob -> raw P(deviate)=1-follow_prob, Platt-calibrates
in logit space on VAL, applies to TEST, and prints the SAME metric block as
scripts/eval_deviation_cls.py so approach b sits arm-by-arm next to a and c.

Prereqs:
  # TEST pass already exists (results/agent_inspection/eval_panel_medgemma.jsonl).
  # VAL pass (run once on a GPU node, produces the val panel dump):
  python scripts/eval_certainty_agent.py --arms doctor sft \
      --data data/training_set/sft/val.jsonl \
      --out results/agent_inspection/eval_panel_medgemma_val.txt
Then (CPU, no model):
  python scripts/eval_dev_from_reasoning.py \
      --val-panel  results/agent_inspection/eval_panel_medgemma_val.jsonl \
      --test-panel results/agent_inspection/eval_panel_medgemma.jsonl \
      --arms sft
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Single source of truth for metrics + Platt (sklearn-verified there). eval_deviation_cls only
# imports torch lazily inside score_rows, so importing it here pulls in NO heavy deps.
from eval_deviation_cls import (  # noqa: E402
    auroc, auprc, brier, logloss, acc_at, ece, boot_ci, fit_platt, apply_platt, report,
)

EPS = 1e-6


def y_of(row) -> int:
    """Binary label from the doctor's realized action. follow -> 0 ; {deviate, off_rubric} -> 1.
    (Verified identical to data/training_set/cls/*.meta.y for the test panel.)"""
    return 0 if row["when_action"] == "follow" else 1


def z_of(follow_prob: float) -> float:
    """raw P(deviate) = 1 - follow_prob ; z = its logit (the Platt input, log-odds space)."""
    p = min(max(1.0 - follow_prob, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def load_panel(path: Path, arm: str):
    """-> (ids, y list, z list, raw-P(deviate) list) for one arm, over the panel's judged rows."""
    ids, ys, zs, raws = [], [], [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cell = r.get(arm)
        if cell is None or cell.get("follow_prob") is None:
            raise SystemExit(f"[error] row {r.get('id')} in {path.name} has no {arm}.follow_prob "
                             f"— was the panel produced with --arms including {arm!r}?")
        fp = float(cell["follow_prob"])
        ids.append(r["id"])
        ys.append(y_of(r))
        zs.append(z_of(fp))
        raws.append(min(max(1.0 - fp, 0.0), 1.0))
    return ids, ys, zs, raws


def cross_check_labels(ids, ys, labels_dir: Path, split: str):
    f = labels_dir / f"{split}.jsonl"
    if not f.exists():
        return
    cmap = {}
    for line in f.read_text().splitlines():
        if line.strip():
            m = json.loads(line)["meta"]
            cmap[m["id"]] = m["y"]
    mism = [i for i, y in zip(ids, ys) if i in cmap and cmap[i] != y]
    missing = [i for i in ids if i not in cmap]
    print(f"[label-check {split}] {len(ids)-len(mism)}/{len(ids)} match cls/{split}.jsonl.meta.y"
          + (f"  MISMATCH={mism[:5]}" if mism else "  ✓")
          + (f"  (not-in-cls: {len(missing)})" if missing else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-panel", default="results/agent_inspection/eval_panel_qwen3_val.jsonl")
    ap.add_argument("--test-panel", default="results/agent_inspection/eval_panel_qwen3.jsonl")
    ap.add_argument("--arms", nargs="+", default=["sft"],
                    help="which panel arm's follow_prob to derive from (sft = the reasoning adapter "
                         "= approach b; base = base-model floor for the same derivation)")
    ap.add_argument("--labels-dir", default="data/training_set/cls",
                    help="cross-check y against cls/{val,test}.meta.y (optional)")
    ap.add_argument("--out", default="results/agent_inspection/deviation_dev_from_reasoning_eval")
    args = ap.parse_args()

    def _p(x):
        return Path(x) if Path(x).is_absolute() else ROOT / x

    val_path, test_path = _p(args.val_panel), _p(args.test_panel)
    if not val_path.exists():
        raise SystemExit(
            f"[error] val panel {val_path} missing. Approach b needs a VAL pass of the reasoning "
            f"SFT to calibrate. Run on a GPU node:\n"
            f"  python scripts/eval_certainty_agent.py --arms doctor sft "
            f"--data data/training_set/sft/val.jsonl "
            f"--out results/agent_inspection/eval_panel_medgemma_val.txt")
    labels_dir = _p(args.labels_dir)

    lines = ["### approach B — P(deviate)=1-follow_prob derived from the reasoning SFT (NO training)",
             f"val_panel={args.val_panel}  test_panel={args.test_panel}",
             "raw P(deviate)=1-follow_prob (follow_prob = P(modality in rubric_rec) on the model's "
             "OWN belief); Platt(a,b) fit on VAL logit, applied to TEST.",
             "CAVEAT: N small => DIRECTIONAL; CIs wide. Same split/label/metrics as approaches a/c.",
             ""]
    dump = {"source": "derived-from-reasoning-sft", "val_panel": str(val_path),
            "test_panel": str(test_path), "arms": {}}

    for arm in args.arms:
        _, yv, zv, _ = load_panel(val_path, arm)
        ids_t, yt, zt, raw_t = load_panel(test_path, arm)
        cross_check_labels(ids_t, yt, labels_dir, "test")

        # base-rate baseline (train prior) if the cls train split is around
        if arm == args.arms[0]:
            tr = labels_dir / "train.jsonl"
            if tr.exists():
                ytr = [json.loads(l)["meta"]["y"] for l in tr.read_text().splitlines() if l.strip()]
                base_rate = sum(ytr) / len(ytr)
                lines.append(f"val n={len(yv)} (pos {sum(yv)})  test n={len(yt)} (pos {sum(yt)})  "
                             f"train base_rate={base_rate:.3f}")
                lines.append("TEST — baseline")
                report("const-base-rate", yt, [base_rate] * len(yt), lines)
                lines.append("")

        a, b = fit_platt(zv, yv)
        cal_t = apply_platt(zt, a, b)
        lines.append(f"ARM = {arm} (derived)   (platt a={a:.3f} b={b:.3f} fit on val)")
        report(f"{arm} test RAW (1-follow_prob)", yt, raw_t, lines)
        bins = report(f"{arm} test CALIBRATED", yt, cal_t, lines)
        lines.append("    reliability (calibrated, test):")
        for lo, hi, cnt, conf, acc in bins:
            if cnt:
                lines.append(f"      [{lo:.1f},{hi:.1f}) n={cnt:2d} conf={conf:.2f} acc={acc:.2f}")
        lines.append("")
        dump["arms"][arm] = {
            "platt": {"a": a, "b": b},
            "test_ids": ids_t, "test_z": zt, "test_raw": raw_t, "test_cal": cal_t, "test_y": yt,
            "metrics_raw": {"auroc": auroc(yt, raw_t), "auprc": auprc(yt, raw_t),
                            "brier": brier(yt, raw_t), "logloss": logloss(yt, raw_t),
                            "acc@0.5": acc_at(yt, raw_t), "ece": ece(yt, raw_t)[0]},
            "metrics_cal": {"auroc": auroc(yt, cal_t), "auprc": auprc(yt, cal_t),
                            "brier": brier(yt, cal_t), "logloss": logloss(yt, cal_t),
                            "acc@0.5": acc_at(yt, cal_t), "ece": ece(yt, cal_t)[0]},
        }

    out = _p(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text("\n".join(lines))
    out.with_suffix(".json").write_text(json.dumps(dump, indent=2))
    print("\n".join(lines))
    print(f"\nwrote {out}.txt / .json")


if __name__ == "__main__":
    main()
