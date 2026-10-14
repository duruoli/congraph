"""Who is in a sample definition: background facts per group and their link to the outcome.

Background = facts about the person, not the current illness (age, sex, IBD type), plus the
population marker (IBD is the primary discharge diagnosis; post hoc, never a feature) and
whether any / abdominal CT imaging exists before t0 (shows how much report-derived features
are driven by imaging being done at all).

Per variable: value in cases vs controls, one-variable AUC, test (Mann-Whitney / Fisher).
Output runs/term_auc/[cohort/]profile.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import fisher_exact, mannwhitneyu

from scripts.term_auc import auc_cols
from scripts.transfer_bow_model import COHORTS, SOURCE, H, build_samples, load

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="all", help="sample definition, see transfer_bow_model.COHORTS")
    args = parser.parse_args()
    out = ROOT / "runs/term_auc" / ("" if args.cohort == "all" else args.cohort)
    out.mkdir(parents=True, exist_ok=True)

    s, _ = build_samples(3, 20261011, **COHORTS[args.cohort])
    s = s[s.hours >= 48].reset_index(drop=True)
    co = ds.dataset(SOURCE / "cohort", format="parquet").to_table(
        columns=["hadm_id", "age", "gender", "ibd_type", "is_primary_ibd_diagnosis"]).to_pandas()
    d = s.merge(co, on="hadm_id", how="left")
    rad = load("radiology", set(d.hadm_id), ["hadm_id", "charttime", "storetime", "exam_name"])
    rad["t"] = rad.storetime.fillna(rad.charttime)
    by = dict(tuple(rad.groupby("hadm_id")))

    def imaged(pattern=None):
        v = []
        for r in d.itertuples():
            w = by.get(r.hadm_id)
            w = w[(w.t >= r.admittime - 24 * H) & (w.t < r.t0)] if w is not None else []
            if pattern is not None and len(w):
                w = w[w.exam_name.str.contains(pattern, case=False, na=False)]
            v.append(len(w) > 0)
        return np.array(v)

    y = d.label.values
    binary = {
        "female": (d.gender == "F").values,
        "ulcerative colitis (vs Crohn)": (d.ibd_type == "ulcerative_colitis").values,
        "IBD = primary diagnosis (post hoc)": d.is_primary_ibd_diagnosis.astype(bool).values,
        "any imaging before t0": imaged(),
        "abdominal CT before t0": imaged("CT ABD|CT PELVIS|CTA ABD"),
    }
    lines = [f"# 样本画像：{args.cohort}", "",
             f"样本 {len(d)}（转科 {int(y.sum())} / 对照 {int((1 - y).sum())}；对照来自 {d[y == 0].hadm_id.nunique()} 次不同住院）。",
             "AUC < 0.5 = 对照组更高。主诊断是出院编码，只用来看人群构成，不能当预测因子。", "",
             "| 变量 | 转科组 | 对照组 | AUC | p |", "|---|---|---|---:|---:|"]
    a = d.age.values
    lines.append(f"| 年龄（中位，IQR） | {np.median(a[y == 1]):.0f}（{np.percentile(a[y == 1], 25):.0f}–"
                 f"{np.percentile(a[y == 1], 75):.0f}） | {np.median(a[y == 0]):.0f}（{np.percentile(a[y == 0], 25):.0f}–"
                 f"{np.percentile(a[y == 0], 75):.0f}） | {auc_cols(y, a[:, None])[0]:.3f} | "
                 f"{mannwhitneyu(a[y == 1], a[y == 0]).pvalue:.2g} |")
    for name, x in binary.items():
        x = x.astype(int)
        p = fisher_exact(pd.crosstab(x, y).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values)[1]
        lines.append(f"| {name} | {x[y == 1].mean():.0%} | {x[y == 0].mean():.0%} | "
                     f"{auc_cols(y, x[:, None].astype(float))[0]:.3f} | {p:.2g} |")
    lines += ["", "注：一个对照住院可能匹配给多个病例而重复出现，p 值略偏乐观。"]
    (out / "profile.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
