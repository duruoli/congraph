"""Descriptive look at the bag of words: which terms recur most (no model).

Recurrence = number of distinct days a term is mentioned (same-day repeats collapse).
Population and window as in transfer_recurrence.py (MED→SURG cases + matched
MED-only controls, from admit − 24 h up to the decision point).
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.transfer_bow_model import build_samples, events
from scripts.transfer_recurrence import encode, window_units

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {"report word": None, "report phrase (2 words)": None, "lab:": "lab:", "drug:": "drug:",
              "order:": "order:", "rad:": "rad:", "micro:": "micro:", "org:": "org:"}


def category(name):
    if ":" in name and name.split(":")[0] in ("lab", "drug", "order", "rad", "micro", "org"):
        return name.split(":")[0] + ":"
    return "report phrase (2 words)" if " " in name else "report word"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/term_frequency")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    samples, _ = build_samples(3, 20261011)
    samples = samples[samples.hours >= 48].reset_index(drop=True)
    ev = events(set(samples.hadm_id))
    tu, ku, nrec = window_units(samples, ev, 0, facts_only=False)
    X, names = encode(tu, ku, len(samples))
    days = X["distinct"].copy(); days.data = np.expm1(days.data).round()
    y = samples.label.values

    df = pd.DataFrame({
        "term": names,
        "category": [category(n) for n in names],
        "mention_days_total": np.asarray(days.sum(axis=0)).ravel(),
        "patients_mentioning": np.asarray((days > 0).sum(axis=0)).ravel(),
        "share_cases": np.asarray((days[y == 1] > 0).mean(axis=0)).ravel().round(3),
        "share_controls": np.asarray((days[y == 0] > 0).mean(axis=0)).ravel().round(3),
    })
    df["days_per_mentioning_patient"] = (df.mention_days_total / df.patients_mentioning.clip(lower=1)).round(2)
    df.sort_values("mention_days_total", ascending=False).to_csv(args.out / "terms.csv", index=False)

    lines = [f"# 词袋最高频词（{len(samples)} 次住院；频率 = 提到的天数）", ""]
    for cat in CATEGORIES:
        top = df[df.category == cat].nlargest(args.top, "mention_days_total")
        lines += [f"## {cat}", "", "| 词 | 总天数 | 提到的人数 | 转科组提到比例 | 对照组提到比例 | 每人平均天数 |",
                  "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {r.term} | {int(r.mention_days_total)} | {r.patients_mentioning} | {r.share_cases} | "
                  f"{r.share_controls} | {r.days_per_mentioning_patient} |" for r in top.itertuples()]
        lines.append("")

    # two individual terrains: one case, one control
    rng = random.Random(7)
    for lab, who in [(1, "转科病例"), (0, "对照")]:
        i = rng.choice(list(np.flatnonzero(y == lab)))
        row = np.asarray(days[i].todense()).ravel()
        top = np.argsort(-row)[:20]
        lines += [f"## 单个病人的地形：{who}（住院 {samples.hadm_id[i]}，0 点前 {samples.hours[i]:.0f} 小时）", "",
                  " · ".join(f"{names[j]}×{int(row[j])}" for j in top if row[j] > 0), ""]
    (args.out / "overview.md").write_text("\n".join(lines))
    print(args.out / "overview.md")


if __name__ == "__main__":
    main()
