# MIMIC IBD action scene：八轴人工标注草案

状态：讨论稿，依据 `unified_axes_v1.yml` 与六次影像检查的 toy case。标注单位是**一个有时间锚点的记录 action**，输出同时描述该步的临床场景和实际 action。这不是预测下一步 action 的输入卡片。

## 0. 每一步先记录的非八轴字段

| 字段 | 定义 |
|---|---|
| `action_raw`、`source_event_id` | 原始行为及来源；保留粒度和原文名称。 |
| `action_type` | 每个原始 action 优先归为 `observation`（获取信息）或 `intervention`（改变管理/治疗状态）。兼具两者时仍选主类，并另记 `secondary_action_type`；无法合理定主类则标人工复核。入院/转科等照护转换不进入这两类的主标注队列。 |
| `action_status` | `ordered`、`performed_or_collected`、`reported`、`prescribed`、`administered` 等，按数据实际支持的阶段填写。处方开始不等于给药；影像 charttime 不等于医嘱时间；操作只有日期时不虚构时分。 |
| `clinical_question` | 本次要解决的具体问题，以简短文字记录，例如“结肠是否继续扩张？”它比 management_domain 更具体，不属于八轴。 |
| `evidence_scope` | 对每个值注明来自检查名称、该次 indication、此前已可用记录、该次结果、回顾性编码中的哪一层，以及证据时间。 |
| `assertion`、`mapping_status` | 区分 documented / inferred / suspected / retrospectively known；区分 `mapped`、`unmapped`、`unknown`、`n_a`。候选词不能伪装成 YAML 现有值。 |

场景标注允许读取该次 action 名称及其 indication，以描述该次行为及它面对的问题。报告中的 indication 是行动目的的**回顾性代理**，不能保证医嘱时原样可知。对随时间变化的状态（尤其 `clinical_phase`）分别保存 `as_known_at_action` 与 `retrospective_at_action`：后者允许全程资料帮助重建**该时点**的状态，但必须记录支持它的后续证据与时间定位，不能把后来才发生的状态提前，也不能声称医生当时已知。**该次结果进入“结果/下一状态”；若被用于回顾判断该时点已存在的状态，需明确标记为后见证据。** 全序列资料也可用于单独的病例表型标签，但不能无声地替换该步的证据状态。

## 1. 八轴在 action scene 中的定义

| 轴 | 本任务标注对象 | 主要判据与边界 |
|---|---|---|
| `disease` | 该步所服务的临床问题所属疾病；另存全程 cohort/phenotype 标签 | UC/CD 明确时使用 YAML 中 `ibd_ulcerative_colitis` / `ibd_crohns`。仅知 IBD 时记录 `unmapped:ibd_unspecified`，并保留原文；不能用 `disease:all` 代替 IBD。感染等竞争问题可另列其疾病与角色。 |
| `management_domain` | **为什么**做这一步：决策任务类别 | 新问题的鉴别或并发症评估可为 `diagnosis_workup`；既有异常的复查可为 `monitoring`；因风险分层或治疗选择而做检查，要据 indication/上下文判断，不能由 modality 自动推出。一次可多值，但须有独立理由。 |
| `intervention_class` | 本步已记录的改变状态或治疗行为的类别 | 给药/处方/手术等按实际 action 映射。纯检查填 `n_a`。既诊断又治疗的操作可同时填此轴与 modality。不能把先前用过的药写成本步 intervention。 |
| `diagnostic_modality` | 本步已记录的信息获取或疾病表征方式 | MR enterography → `cross_sectional_imaging_mri`。腹部平片 → `unmapped:abdominal_radiograph`（候选扩词）；纯治疗填 `n_a`。`unknown` 只用于检查方式不可辨认，不能表示“做了平片但 YAML 没有值”。普通微生物培养不能随意归入 `serology`。 |
| `clinical_phase` | 行动时患者在病程/治疗轨迹中的位置，按时点标注两种证据视角 | `as_known_at_action` 用当时/此前证据和该次 indication 代理；`retrospective_at_action` 可用全程证据重建同一时点。`diagnosis_workup` 属 management，不等于 `pre_diagnosis`。疑似急性并发症可标 `complication_acute` 加 `suspected`；治疗失败要有明确证据及可定位的发生时间。活跃 flare 的一般复查在现有 phase 词表中无精确值，不硬套 `monitoring_remission`。 |
| `population` | 当步相关患者亚组 | 成人等可用结构化年龄；免疫抑制等需额外证据。背景字段不因每次检查重复推断。 |
| `setting` | 当步所在照护环境 | 根据 event time 与 encounter 时间定位住院/急诊/门诊；住院 cohort 身份本身不能证明住院前的检查也发生在住院。 |
| `anatomy` | 本步**临床问题聚焦的器官** | 首先用 indication 和既有上下文，检查覆盖部位只作辅助。结肠扩张问题 → `colon`；MR enterography 名称本身不能证明当步只聚焦小肠。另存 `exam_region` 与 `finding_site`，避免把扫描范围或事后结果混为决策焦点。 |

统一空值语义：`n_a` = 此 action 不激活该轴；`unknown` = 应有值但证据不足；`unmapped:<raw concept>` = 概念已知而 YAML 无对应值。不要将数据库空值、`n_a` 与词表缺口混用。每个可争议标签须留证据、推断状态及可信程度。

### 两个 action 轴的适配边界

YAML 将 `diagnostic_modality` 定义为疾病的 detection / characterization **方式**，所以本任务把它用于诊断、监测、筛查、风险评估等目的下的信息获取；目的另由 `management_domain` 指出。`intervention_class` 是所选干预的**类别**，不是具体药名、剂量、手术细节或真实执行状态。这两轴原本服务文献主题检索，不能当作完整的 MIMIC 事件分类法。

因此一级 `action_type` 优先在 `observation` / `intervention` 中选主类。诊疗兼具的操作可记录次要类别；入院/转科以及单纯记录、等待不应被强塞进 diagnostic modality。即使主类清楚，YAML 仍可能没有相应 value：腹部平片、常规微生物培养等须保留原始 action 和 `unmapped`。每步应同时存 `raw_action`、`action_type`、`management_domain`、可映射的八轴值及 `action_status`。

## 2. 六步影像 toy 的人工标注摘要

基于私有病例配置 `data/raw_data/ibd_mimiciv_3_1/toy_deviation/case.json` 中的 exam 名称、临床问题及证据说明；此表不包含患者标识。六步 `action_type` 均为 `observation`，影像记录支持检查已实施，但不能反推准确医嘱时间；`intervention_class=n_a`，`disease=ibd_ulcerative_colitis`，`population=adults`，`setting=hospitalized_inpatient`，`anatomy=colon`。其中 disease 为该次 indication / 既往报告支持的 UC，并非仅由事后 ICD 推断。

| 步 | 实际 action / `diagnostic_modality` | `management_domain` | `clinical_phase`（断言） | 具体问题 |
|---|---|---|---|---|
| 1 | 腹部平片 / `unmapped:abdominal_radiograph` | `diagnosis_workup` | `complication_acute`（suspected） | 疑似结肠扩张或 toxic megacolon？ |
| 2 | MR enterography / `cross_sectional_imaging_mri` | `diagnosis_workup` | `complication_acute`（suspected，依据之前的扩张疑虑） | 病情恶化后进一步表征肠道情况？ |
| 3 | 腹部平片 / `unmapped:abdominal_radiograph` | `monitoring` | `treatment_failure_refractory`（该次 indication 记载）；`complication_acute`（suspected） | 激素难治背景下扩张/并发症如何变化？ |
| 4 | 腹部平片 / `unmapped:abdominal_radiograph` | `monitoring`（由连续检查与既往扩张推断） | `complication_acute`（suspected） | 已见扩张是否继续变化？ |
| 5 | 腹部平片 / `unmapped:abdominal_radiograph` | `monitoring` | `complication_acute`（suspected） | 最近加重的扩张是否变化？ |
| 6 | 腹部平片 / `unmapped:abdominal_radiograph` | `monitoring` | `unknown`（有活跃病程线索，但当前材料不足以定 phase；`active_flare` 是候选词表缺口） | 扩张部分改善后是否继续变化？ |

表中的 phase 是**当步 indication / 此前报告支持的暂定视角**，不是已经完成的全程回顾性 phase 重建。尚未把同期化验、用药执行、病程记录整合进每步证据；下一轮应为每一步另标 `retrospective_at_action`。第 4 步管理任务主要是从序列推断；第 6 步 phase 暂不硬定。检查结果若证明或排除并发症，应记录在行动后的观察/状态；只有证据足以定位该时点的状态时，才进入回顾性 phase，并标明是后见证据。

## 3. 与 deviation 的接口

这一层给出 `scene_t`、`observed_action_t`、`result_after_t` 三个对象。文献检索可用 scene 加 observed action 找到相关文章；判断“本来应做什么”时，必须从适用文献中另抽取**有条件的建议/证据主张**。索引匹配本身不生成建议。若要评估行动前可选项，须另构造只含当时可用信息的 `scene_before_action_t`；不能把本层的 observed modality 当作预测输入。每个 literature claim 还需记录适用条件、证据类型、方向/强度与其对应的行动，否则不同来源的冲突或偏差无法解释。

## 4. 需要下一轮人工裁决的词表问题

1. 是否在共享 YAML 增加 `ibd_unspecified`（或单独 cohort-family 字段），以支持仅有泛称 IBD 的病人；当前六步已有 UC 证据，所以可保持更具体的 UC 值。
2. 是否增加 `abdominal_radiograph`，以及如何覆盖普通实验室检查/培养；在裁决前用 `unmapped` 保存明确动作。
3. `active_flare`、`suspected_complication` 是否应成为 phase 值，或保持现有 phase 加 assertion 字段。此草案倾向后者处理“疑似”，前者保留为候选缺口。
