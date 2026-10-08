# IBD MIMIC-IV 事件时间线

运行 `python scripts/build_ibd_full_timeline.py`，从本地 `data/raw_data/ibd_mimiciv_3_1` 构造所有 IBD 队列住院的事件表。输出在 `runs/ibd_full_timeline/events.parquet`，汇总在同目录的 `manifest.json`。这两个文件包含受限 MIMIC 衍生数据，不应提交到公开仓库。

## 覆盖范围

时间线包含住院开始、出院、院内死亡，以及本地已提取的 `labs`、`microbiology`、`prescriptions`、`procedures`、`radiology`、`discharge_notes`、`services`、`poe`。每条源记录对应一条事件；相同 `poe_id` 的不同 `poe_seq` 仍是不同记录。脚本不会复制临床笔记正文，只保存来源 ID 和少量结构化字段。

`diagnoses_icd` 没有事件时间，不会被放在住院过程的某个假想时刻。它仍在原始 `diagnoses` 表中，可作为回顾性的住院标签与时间线连接。尚未提取的 `transfers`、`emar`、ICU 模块等自然不在这份时间线中，因此它不是完整的诊疗过程。

## 时间字段

| 字段 | 含义 |
|---|---|
| `event_at` | 事件时间；优先使用源表的 `charttime`、`ordertime`、`transfertime` 或 `starttime` |
| `time_precision` | `minute`、`date` 或 `unknown`；`date` 表示仅知道日期 |
| `event_end_at` / `event_end_kind` | 日期事件的次日 00:00 排他上界，或处方的预定停止时间；通过 `event_end_kind` 区分 |
| `recorded_at` / `recorded_precision` | 源表有 `storetime` 时的记录／验证时间；微生物表在缺少 `storetime` 时可用 `storedate` |
| `patient_event_index` | 单个病人的展示顺序，从 0 开始；不能用它断言同一天仅有日期的事件与其他事件的先后 |
| `admission_event_index` | 单次住院的展示顺序，从 0 开始 |
| `source_table` / `source_row_ordinal` | 来源表及按文件名排序扫描时的行序号，用于追溯原始 Parquet；原始行在重新导出后可能改变序号 |
| `detail_json` | 少量结构化字段；如需完整信息，应回到原始源表 |

`chartdate` 被表示为该日 `[00:00, 次日 00:00)` 的范围，**00:00 不是实际发生时刻**。排序仅为阅读方便；同一天的日期精度事件不能与分钟精度事件建立确定先后。`storetime` 不是通用的“医生首次获知结果时间”，而是记录／验证的时间。跨病人的绝对日期经过独立偏移，不能据此比较真实日历时间。

处方 `starttime` / `stoptime` 表示预定用药时段，不证明已经给药；`poe.ordertime` 表示医嘱记录时间，不证明检查或治疗已执行；`services.transfertime` 表示负责服务的归属或变动，不证明诊断假设发生变化。出院笔记是回顾性文档，即使有 `charttime` 也不应被当作当时医生已经知道的事实。

## 查询一次住院

```python
import pyarrow.dataset as ds

events = ds.dataset("runs/ibd_full_timeline/events.parquet", format="parquet")
one_admission = events.to_table(
    filter=ds.field("hadm_id") == YOUR_HADM_ID,
    columns=["event_at", "time_precision", "source_table", "event_kind", "detail_json"],
).to_pandas().sort_values("event_at")
```

围绕一次 service transfer，可用相同 `hadm_id` 取 `event_at` 落在 `transfertime` 前后指定窗口内的记录。解释时应区分医嘱、测量、结果记录、报告和处方，并检查事件的时间精度与回顾性性质。
