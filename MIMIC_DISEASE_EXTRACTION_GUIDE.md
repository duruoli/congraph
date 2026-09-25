# MIMIC-IV 疾病数据提取指南

目标：从 MIMIC-IV 中按疾病诊断建立住院队列，提取结构化数据与临床文本，并下载为本地 Parquet 数据集。

> 默认保留该疾病的**所有住院记录**。首次住院、主要诊断、是否有影像等均作为标记保存，具体任务再筛选。

## 1. 准备访问权限

在 PhysioNet 分别获得：

- [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)：诊断、住院、检验、处方、操作等。
- [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/)：出院记录和影像报告。

在 Google Cloud 中：

1. 创建或选择一个启用 Billing 的项目。
2. 通过 PhysioNet 将 BigQuery 权限授予同一个 Google 账号。
3. 确认可查询以下数据集：

```text
physionet-data.mimiciv_3_1_hosp
physionet-data.mimiciv_3_1_icu       # 仅 ICU 研究需要
physionet-data.mimiciv_3_1_derived   # 可选衍生变量
physionet-data.mimiciv_note
```

本流程只要求 `hosp` 和 `mimiciv_note`，不限制患者是否进入 ICU。

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
