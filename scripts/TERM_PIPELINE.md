# 词 / 概念探索流水线（东西 × 重复 × 转科）

一条命令跑完一个样本定义：

```bash
PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/term_pipeline.py --cohort ibd_primary_age
```

环境：全部在 `congraph` conda 环境里（含 scispaCy、medspaCy、pyarrow）。

## 上游：IBD 原始数据从哪来

`data/raw_data/ibd_mimiciv_3_1/` 由仓库根目录的 `MIMIC_DISEASE_EXTRACTION_GUIDE.md` 在 BigQuery 上提取（§4 建 cohort，§5 起提取各表；DRG、echo/POE 详情另有 `scripts/extract_ibd_*.sh`）。cohort 定义：本次住院的诊断编码里**任何位置**出现 ICD-9 `555*`/`556*` 或 ICD-10 `K50*`/`K51*` 即纳入；`is_primary_ibd_diagnosis` = 最靠前的 IBD 编码排第 1 位。按指南 §11 的原则，cohort 保留所有 IBD 住院，主诊断、首次住院等条件放到具体分析里（即这里的 `COHORTS`）。

## 新增一个对照组 / 样本定义

1. 在 `scripts/transfer_bow_model.py` 的 `COHORTS` 里加一项。现有参数：
   - `primary_ibd`：病例和对照都限定为"IBD 是出院主诊断"（只用来划定人群，不作特征）
   - `age_caliper`：对照与病例年龄差不超过几岁
   - 需要新的限定条件时，在 `build_samples` 里加参数；默认值必须保持旧行为不变。
2. 运行 `term_pipeline.py --cohort <新名字>`。
3. 结果在 `runs/term_auc/<新名字>/`，与 `runs/term_auc/`（cohort = all）并排对比。

## 步骤与产物

| 步骤 | 脚本 | 读 | 写 | 按组还是共享 |
|---|---|---|---|---|
| 样本 | `transfer_bow_model.build_samples` | `data/raw_data/ibd_mimiciv_3_1/{cohort,services,...}` | （各脚本内存中重建，确定性） | 按组 |
| profile | `cohort_profile.py` | 样本 + cohort 表 + 影像 | `runs/term_auc/[组/]profile.md` | 按组 |
| extract | `term_concepts.py` | 影像报告 | `runs/term_concepts/mentions.parquet`，`extracted_notes.parquet` | **共享缓存**，只处理新报告 |
| table | `term_concept_table.py` | 缓存 + `term_concept_overrides.csv` | `runs/term_concepts/[组/]{samples,sample_features}.parquet`，`screen.csv/.md` | 按组 |
| auc_word | `term_auc.py --unit word` | 样本 + 事件 + `runs/term_frequency/screen.csv` | `runs/term_auc/[组/]words.{csv,md}` | 按组（词表共享） |
| auc_concept | `term_auc.py --unit concept` | `runs/term_concepts/[组/]` | `runs/term_auc/[组/]concepts.{csv,md}` | 按组 |

固定的设定：时间窗 = 入院前 24 小时至决定点；只保留决定点距入院 ≥ 48 小时的病例；每个病例匹配 3 个对照；随机种子 20261011；bootstrap 按病人重抽 1000 次；多重比较用 BH。

## 人工环节（改了之后要重跑的步骤）

| 人工环节 | 文件 | 改完重跑 |
|---|---|---|
| 词表初筛（keep / maybe / drop，不看分组） | `runs/term_frequency/screen.csv` 的 `final` 列；规则在 `term_screen.py` | auc_word |
| 概念初筛 | `runs/term_concepts/[组/]screen.csv` 的 `final` 列（重建时保留）；规则在 `term_concept_table.py` 的 `DROP_CONCEPTS` / `MAYBE_CONCEPTS` | auc_concept |
| 概念纠错（缩写错配、泛词） | `scripts/term_concept_overrides.csv` | table，auc_concept |
| 放射科口语词表 | `term_concepts.py` 的 `RADIOLOGY_LEXICON` | **清空缓存**后 extract（见下），table，auc_concept |
| 否定 / 不确定的触发词 | `term_concepts.py` 的 `HEDGE_FORWARD` / `HEDGE_BACKWARD` | 同上 |

改了抽取规则（词表、触发词）后，缓存里的旧结果已经过时，需要删掉 `runs/term_concepts/mentions.parquet` 和 `extracted_notes.parquet`，再对每个用到的组跑 extract。

## 输出怎么读

- `profile.md`：两组是什么人。先看这个——年龄、主诊断构成差很多时，后面的词信号可能只是人群差异。
- `words.md` / `concepts.md`：
  - `auc_presence`：只看"出现过"；> 0.5 偏转科，< 0.5 偏对照。
  - `auc_days` / `auc_count`：按天数 / 次数；`days_minus_presence`、`count_minus_presence` 已按方向校正，正值 = 区分得更好。
  - `q_presence`：多重比较校正后的显著性。
  - `auc_presence_imaged`：只在决定点前做过影像的样本里。
  - 概念的 `[any]` = 提到过，不分有无（"在看哪里"）；`[+]` / `[-]` / `[?]` = 看到了 / 查了没有 / 可能。
  - `note = action_trace`：由之前的干预造成的事实（置管、术后状态）。
