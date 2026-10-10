# MIMIC-IV 疾病数据提取指南

目标：从 MIMIC-IV 中按疾病诊断建立住院队列，提取结构化数据与临床文本，并下载为本地 Parquet 数据集。

> 默认保留该疾病的**所有住院记录**。首次住院、主要诊断、是否有影像等均作为标记保存，具体任务再筛选。

## 1. 准备访问权限

在 PhysioNet 分别获得：

- [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)：诊断、住院、检验、处方、操作等。
- [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/)：出院记录和影像报告。
- [MIMIC-IV-Echo](https://www.physionet.org/content/mimic-iv-echo/1.0.1/)：心脏超声结构化测量与部分检查的 DICOM 文件；需要单独授权，仅提取心超时需要。

在 Google Cloud 中：

1. 创建或选择一个启用 Billing 的项目。
2. 通过 PhysioNet 将 BigQuery 权限授予同一个 Google 账号。
3. 确认可查询以下数据集：

```text
physionet-data.mimiciv_3_1_hosp
physionet-data.mimiciv_3_1_icu       # 仅 ICU 研究需要
physionet-data.mimiciv_3_1_derived   # 可选衍生变量
physionet-data.mimiciv_note
physionet-data.mimiciv_echo     # 可选；心超
```

基础流程只要求 `hosp` 和 `mimiciv_note`，不限制患者是否进入 ICU。心超扩展需另有 `mimiciv_echo` 权限。

## 2. 定义疾病

这里的“确诊”表示：该次住院在 `diagnoses_icd` 中存在对应 ICD 编码。它是回顾性的住院诊断，不代表疾病在某个具体时间点已经被确认。

先用诊断字典检查候选编码：

```sql
SELECT icd_code, icd_version, long_title
FROM `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses`
WHERE LOWER(long_title) LIKE '%disease keyword%'
ORDER BY icd_version, icd_code;
```

再由临床知识确定 ICD-9/10 编码或前缀。不要只依靠疾病名称模糊匹配建立最终队列。

IBD 示例：

| 类型 | ICD-9 | ICD-10 |
|---|---|---|
| Crohn's disease | `555*` | `K50*` |
| Ulcerative colitis | `556*` | `K51*` |

## 3. 创建输出 dataset

在 BigQuery Editor 中运行；替换 `{PROJECT}`、`{DATASET}`：

```sql
CREATE SCHEMA IF NOT EXISTS `{PROJECT}.{DATASET}`
OPTIONS(location = 'US');
```

建议命名：

```text
GCP project: disease-specific-project
BigQuery dataset: disease_extract
```

## 4. 建立疾病 cohort

只需修改 `disease_codes`。编码应去掉小数点并使用大写。

```sql
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.cohort` AS
WITH disease_codes AS (
  -- 替换为目标疾病的编码；可增加更多行
  SELECT 9 AS icd_version, '555' AS code_prefix, 'crohn' AS disease_type
  UNION ALL SELECT 10, 'K50', 'crohn'
  UNION ALL SELECT 9, '556', 'ulcerative_colitis'
  UNION ALL SELECT 10, 'K51', 'ulcerative_colitis'
),
matched AS (
  SELECT
    d.subject_id,
    d.hadm_id,
    MIN(d.seq_num) AS first_disease_seq_num,
    STRING_AGG(DISTINCT c.disease_type, '+' ORDER BY c.disease_type) AS disease_type,
    STRING_AGG(DISTINCT d.icd_code, ', ' ORDER BY d.icd_code) AS matched_icd_codes
  FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` AS d
  JOIN disease_codes AS c
    ON d.icd_version = c.icd_version
   AND STARTS_WITH(
         REPLACE(UPPER(d.icd_code), '.', ''),
         c.code_prefix
       )
  GROUP BY d.subject_id, d.hadm_id
),
base AS (
  SELECT
    m.subject_id,
    m.hadm_id,
    a.admittime,
    a.dischtime,
    a.deathtime,
    a.hospital_expire_flag,
    p.gender,
    p.anchor_age
      + EXTRACT(YEAR FROM a.admittime)
      - p.anchor_year AS age,
    m.disease_type,
    m.matched_icd_codes,
    m.first_disease_seq_num = 1 AS is_primary_disease_diagnosis,
    ROW_NUMBER() OVER (
      PARTITION BY m.subject_id
      ORDER BY a.admittime, m.hadm_id
    ) AS disease_admission_number
  FROM matched AS m
  JOIN `physionet-data.mimiciv_3_1_hosp.admissions` AS a
    USING (subject_id, hadm_id)
  JOIN `physionet-data.mimiciv_3_1_hosp.patients` AS p
    USING (subject_id)
)
SELECT
  *,
  disease_admission_number = 1 AS is_first_disease_admission
FROM base;
```

关键字段：

- `is_first_disease_admission`：该患者在 MIMIC 中时间最早的该疾病住院；不是必须筛选条件。
- `is_primary_disease_diagnosis`：该疾病是本次住院的第一顺位诊断。
- `disease_admission_number`：保留重复住院的顺序。

## 5. 提取各类数据

所有表均通过 `subject_id + hadm_id` 限制在 cohort 内。以下 SQL 可以作为一个 BigQuery script 一次运行。

默认不按 `hadm_id` 分区；其取值过多。需要优化时，可按日期分区并按 `hadm_id` cluster。

```sql
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.diagnoses` AS
SELECT d.*, dictionary.long_title
FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` AS d
JOIN `{PROJECT}.{DATASET}.cohort` AS c USING (subject_id, hadm_id)
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses` AS dictionary
  USING (icd_code, icd_version);

CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.labs` AS
SELECT l.*, dictionary.label, dictionary.fluid, dictionary.category
FROM `physionet-data.mimiciv_3_1_hosp.labevents` AS l
JOIN `{PROJECT}.{DATASET}.cohort` AS c USING (subject_id, hadm_id)
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.d_labitems` AS dictionary
  USING (itemid);

CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.microbiology` AS
SELECT m.*
FROM `physionet-data.mimiciv_3_1_hosp.microbiologyevents` AS m
JOIN `{PROJECT}.{DATASET}.cohort` AS c USING (subject_id, hadm_id);

CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.prescriptions` AS
SELECT p.*
FROM `physionet-data.mimiciv_3_1_hosp.prescriptions` AS p
JOIN `{PROJECT}.{DATASET}.cohort` AS c USING (subject_id, hadm_id);

CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.procedures` AS
SELECT p.*, dictionary.long_title
FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd` AS p
JOIN `{PROJECT}.{DATASET}.cohort` AS c USING (subject_id, hadm_id)
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.d_icd_procedures` AS dictionary
  USING (icd_code, icd_version);

CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.services` AS
SELECT s.*
FROM `physionet-data.mimiciv_3_1_hosp.services` AS s
JOIN `{PROJECT}.{DATASET}.cohort` AS c USING (subject_id, hadm_id);

CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.poe` AS
SELECT p.*
FROM `physionet-data.mimiciv_3_1_hosp.poe` AS p
JOIN `{PROJECT}.{DATASET}.cohort` AS c USING (subject_id, hadm_id);

CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.discharge_notes` AS
SELECT n.*
FROM `physionet-data.mimiciv_note.discharge` AS n
JOIN `{PROJECT}.{DATASET}.cohort` AS c USING (subject_id, hadm_id);

CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.radiology` AS
SELECT n.*
FROM `physionet-data.mimiciv_note.radiology` AS n
JOIN `{PROJECT}.{DATASET}.cohort` AS c USING (subject_id, hadm_id);
```

建议首先保存原始粒度的数据。病史/查体段落、影像类型、时间窗和模型输入等任务相关变量，之后再生成衍生表。

`services` 的一行记录一次负责临床服务的变动或初始归属，`transfertime` 可作为研究交接前后的时间锚点；它本身不能证明诊断假设发生了变化。`poe` 记录医嘱及其状态，并不等同于检查已经完成或药物已经实际给入。若需医嘱的更多细节，可提取 `poe_detail`，用 `subject_id + poe_id + poe_seq` 与 `poe` 精确连接。`poe_detail` 本身没有 `hadm_id` 或事件时间，应沿用所连医嘱的住院号和 `ordertime`。

### IBD 队列的心超与医嘱细节扩展

获得 MIMIC-IV-Echo 权限后，可运行 [提取脚本](scripts/extract_ibd_echo_poe_detail.sh)。脚本使用已有 IBD `cohort` 和 `poe`，在 BigQuery 建立四个附加表，并导出至 `data/raw_data/ibd_mimiciv_3_1/` 下同名目录：

| 附加表 | 选择和链接方法 | 时间含义 |
|---|---|---|
| `echo_measurements` | `structured_measurement` 按 `subject_id` 匹配，检查时间位于入院至出院区间；增加 `hadm_id` | `measurement_datetime` 是检查时间，不是报告可用时间；同一 `measurement_id` 有多行测量 |
| `echo_studies` | `echo_study_list` 按患者及 `study_datetime` 匹配住院 | DICOM 子集的检查索引，不是全部结构化心超 |
| `echo_record_list` | 根据已匹配的 `subject_id + study_id` 取文件索引 | 仅有 DICOM 路径及元数据，脚本没有下载影像文件 |
| `poe_detail` | 与已提取的 `poe` 按 `subject_id + poe_id + poe_seq` 连接 | 继承 `poe.ordertime`；不是执行或结果时间 |

2026-10-08 对当前 IBD 队列的提取结果：`echo_measurements` 183,544 行（1,377 项检查、1,090 次住院）；`echo_studies` 53 行；`echo_record_list` 3,616 行；`poe_detail` 183,238 行。前者的检查数与 DICOM 索引数不能直接比较覆盖率，因为 DICOM 是受年份及存储限制的子集。脚本核对了各本地 Parquet 行数与 BigQuery 表行数；使用唯一的 GCS 导出前缀，导出副本保留。

### DRG 扩展

另可运行 [DRG 提取脚本](scripts/extract_ibd_drgcodes.sh)：按 `subject_id + hadm_id` 将 `mimiciv_3_1_hosp.drgcodes` 与现有 IBD `cohort` 匹配，保留源表全部字段，导出到 `data/raw_data/ibd_mimiciv_3_1/drgcodes/`。2026-10-09 已提取并核对 17,167 行，覆盖 9,161 次住院。一个住院可能同时有不同 `drg_type`（如 APR 与 HCFA）的编码，分析时应连同类型和描述一起报告。DRG 是住院层面的回顾性分类，不应赋予某个临床事件时间或直接当作转科原因。

## 6. 检查数据量与覆盖率

避免使用 `rows` 作为列名；使用 `row_count`。

```sql
SELECT 'cohort' AS table_name, COUNT(*) AS row_count
FROM `{PROJECT}.{DATASET}.cohort`
UNION ALL
SELECT 'diagnoses', COUNT(*) FROM `{PROJECT}.{DATASET}.diagnoses`
UNION ALL
SELECT 'labs', COUNT(*) FROM `{PROJECT}.{DATASET}.labs`
UNION ALL
SELECT 'microbiology', COUNT(*) FROM `{PROJECT}.{DATASET}.microbiology`
UNION ALL
SELECT 'prescriptions', COUNT(*) FROM `{PROJECT}.{DATASET}.prescriptions`
UNION ALL
SELECT 'procedures', COUNT(*) FROM `{PROJECT}.{DATASET}.procedures`
UNION ALL
SELECT 'services', COUNT(*) FROM `{PROJECT}.{DATASET}.services`
UNION ALL
SELECT 'poe', COUNT(*) FROM `{PROJECT}.{DATASET}.poe`
UNION ALL
SELECT 'discharge_notes', COUNT(*) FROM `{PROJECT}.{DATASET}.discharge_notes`
UNION ALL
SELECT 'radiology', COUNT(*) FROM `{PROJECT}.{DATASET}.radiology`;
```

不同表覆盖不同患者是正常现象。不要为了获得“完整病例”而提前删除缺少某种模态的患者；根据下游任务单独筛选。

## 7. 导出至 Google Cloud Storage

创建与 BigQuery dataset 同区域的 bucket。bucket 名称只能使用小写字母、数字、短横线、下划线和点。

每张表运行一个 `EXPORT DATA`：

```sql
EXPORT DATA OPTIONS (
  uri = 'gs://{BUCKET}/cohort/part-*.parquet',
  format = 'PARQUET',
  compression = 'SNAPPY',
  overwrite = TRUE
) AS
SELECT * FROM `{PROJECT}.{DATASET}.cohort`;
```

依次将 `cohort` 替换为：

```text
diagnoses
labs
microbiology
prescriptions
procedures
services
poe
discharge_notes
radiology
```

如有 `admission_inputs` 等衍生表，也用相同方式导出。

BigQuery 会自动将大表分成多个 `part-*.parquet`。这是正常的分布式导出；本地读取整个目录即可。Snappy 会被 Pandas/PyArrow 自动解压。

## 8. 下载到本地

macOS：

```bash
brew install --cask gcloud-cli
gcloud auth login
gcloud config set project {PROJECT}
```

下载或增量同步：

```bash
gcloud storage rsync --recursive \
  gs://{BUCKET} \
  data/raw_data/{DISEASE}_mimiciv_3_1
```

重复运行是安全的：已下载且未变化的文件会被跳过。

## 9. 验证本地数据

```bash
find data/raw_data/{DISEASE}_mimiciv_3_1 -name '*.gstmp'
find data/raw_data/{DISEASE}_mimiciv_3_1 -name '*.parquet' | wc -l
du -sh data/raw_data/{DISEASE}_mimiciv_3_1
```

第一条命令应无输出。随后用 PyArrow 打开所有文件并统计行数：

```python
from pathlib import Path
import pyarrow.parquet as pq

root = Path("data/raw_data/{DISEASE}_mimiciv_3_1")

for folder in sorted(p for p in root.iterdir() if p.is_dir()):
    files = list(folder.glob("*.parquet"))
    rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    print(folder.name, "files=", len(files), "rows=", rows)
```

确认本地文件可读、行数与 BigQuery 一致后，可删除 GCS 中用于导出下载的 Parquet 副本。例如：

```bash
gcloud storage rm 'gs://{BUCKET}/services/part-*.parquet'
gcloud storage rm 'gs://{BUCKET}/poe/part-*.parquet'
```

这只清理 GCS 导出文件；BigQuery 中的提取表和本地文件仍会保留。`scripts/extract_ibd_services_poe.sh` 在验证本地文件可读后会自动清理这两类 GCS 副本。

## 10. 本地读取

小表可以直接读入 Pandas：

```python
import pandas as pd

cohort = pd.read_parquet(
    "data/raw_data/{DISEASE}_mimiciv_3_1/cohort"
)
```

大表建议用 PyArrow 按住院号或时间过滤：

```python
import pyarrow.dataset as ds

labs = ds.dataset(
    "data/raw_data/{DISEASE}_mimiciv_3_1/labs",
    format="parquet",
)

one_admission = labs.to_table(
    filter=ds.field("hadm_id") == 12345678
).to_pandas()
```

## 11. 复用到新疾病时只需修改

1. GCP project、BigQuery dataset、bucket 和本地目录名称。
2. `disease_codes` 中的 ICD-9/10 定义。
3. 重新运行 cohort、各数据表、检查、导出和下载步骤。
4. 保留所有疾病住院记录；在具体任务中再使用首次住院、主要诊断、影像类型或模态完整性条件。

## 成本与数据安全

- BigQuery 查询可能按扫描字节收费；运行前查看 Console 中的 estimated bytes。
- GCS 存储和下载可能产生费用；下载验证后可按研究需要决定是否保留云端副本。
- 可为临时 BigQuery 表设置 expiration，避免长期存储费用。
- MIMIC 数据受 PhysioNet DUA 约束，不要提交到 Git、公开网盘或公开数据仓库。

## 本次 IBD 示例

```text
GCP project: congraph-mimic-ibd
BigQuery dataset: ibd_extract
GCS bucket: congraph-mimic-ibd-export-duruoli-2026
Local directory: data/raw_data/ibd_mimiciv_3_1
```

最终 cohort：10,815 次住院、4,397 位患者。

## 官方参考

- [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/)
- [MIMIC-IV-Note v2.2](https://physionet.org/content/mimic-iv-note/2.2/)
- [BigQuery 导出数据](https://cloud.google.com/bigquery/docs/exporting-data)
- [gcloud storage rsync](https://cloud.google.com/sdk/gcloud/reference/storage/rsync)
