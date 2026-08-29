# 新对话任务 Prompt：建立第一版正式 A/Q/C 分类规则

请在仓库 `congraph` 中继续完成 A/Q/C Track B 的第一层工作：**建立并审计第一版正式的
A/Q/C 分类规则**。这一步的目标是把“标注尺子”制定清楚，不是现在就统计最终 pattern、
批量标注全部患者或进行下一影像预测。

## 开始前必须完整阅读

1. `rubric_update.md`
2. `HANDOFF_acr_aqc_schema_extraction.md`
3. `aqc_annotation_design.md`
4. `data/aqc_development/README.md`
5. `data/aqc_development/open_coding_memos.md`
6. `data/acr_normative/README.md`
7. `HANDOFF_annotation_pipeline.md`（仅用于理解旧经验标注是如何产生的）

同时检查以下实现和数据结构：

- `scripts/build_aqc_discovery_sample.py`
- `scripts/validate_aqc_development.py`
- `experiments/aqc/prompts.py`
- `results/annotation_experiment/full/*.json`
- `results/annotation_experiment/full/timing_roles.csv`

## 不可改变的边界

- 正式经验语料是 `results/annotation_experiment/full/` 中的293位患者、542个决策步。
- ACR v1.1 已经完成，是独立、只读的规范知识 `N`。不要用 ACR 定义 A/Q/C 类型，也不要
  为了适配患者订单而修改 ACR。
- 保留旧 annotation 原文件不变。新的 open coding、codebook 和 A/Q/C 数据必须作为单独
  的分析层保存。
- 当前 `data/aqc_development` 中基于38位患者子集、16条轨迹形成的结果只是原型，不是正式
  codebook。
- 不得使用当前影像结果、之后的事件、verification、deviation、最终诊断正确性或 ACR
  rating 来发现分类或选择数据。
- 疾病标签可以用于保证抽样覆盖，但不得作为 A/Q/C 标注答案展示给标注模型，也不要为
  四种疾病建立四套互不相通的本体。
- 第一层只建立分类规则，不要启动全量批处理、最终 pattern 统计或最终测试。

## 本次需要完成的工作

### 1. 建立患者级开发区与最终测试区

只从 `full/*.json` 读取合法患者轨迹，排除 `manifest.json` 等非患者文件，并连接
`full/timing_roles.csv`。

建立稳定、可复现的约80/20患者级划分：

- 约80%为 development；
- 约20%为 final test；
- 在每种疾病内部划分，以保证四种疾病在两边都有覆盖并大致保留原始比例；
- 同一患者的所有步骤只能属于一个 partition；
- final test 的患者在 codebook、prompt、pattern 定义和模型冻结前不得打开或用于修改。

生成清晰的 split manifest，记录算法、seed或稳定hash规则、患者数、步骤数和各疾病计数。

### 2. 从 development 选择第一批 codebook 样本

选择约24条轨迹**总计**，不是每种疾病24条；目标约为每病6条。

24只是第一轮起点。它的目的不是估计 pattern 频率，而是尽可能覆盖不同决策结构。抽样应
包含可复现的分层/随机成分，并有计划地补足预先定义的少见结构。至少审计：

- 单步与多步轨迹；
- repeat 与 modality switch；
- 不同 modality sequence；
- 初次与后续影像；
- prior study limited、nonvisualized、indeterminate 或未覆盖目标；
- 疾病/发现是否存在、来源定位、病因、严重程度/病程、并发症、其他来源；
- post-intervention/device 状态作为明确的次级 stratum；
- differential 集中与分散、`other` 较高的复杂病例；
- 四种疾病和主要 timing strata。

不要用 verification、deviation、ACR、当前订单的结果或之后的结局来优化样本的“多样性”。
可以使用决策前已经可见的 prior-study limitation，因为它正是需要覆盖的因果前置信息。
输出正式的 development sample manifest 和 diversity audit。

### 3. 用两种阅读视图进行 open coding

对选中的 development 样本逐步、按因果时间进行 open coding：

1. 先只看旧 annotation 的自由文本 `reasoning` 内容，但隐藏 `differential`、
   `information_gap`、`expected_finding`、`action_role` 等字段名和结构提示；
2. 再看完整的旧 schema-light、ex-ante annotation；
3. 比较哪些类型在两种视图中都自然出现，哪些可能是旧字段结构诱导出来的。

这一阶段不要把 ACR 加入输入。

### 4. 建立三个相互连接的正式 codebook

产出可审计、机器可读并有人类说明的第一版规则：

#### Assumption

- 每个 assumption 必须是原子命题；
- 同一步可以同时存在多个不同层级的命题；
- 每个命题分别记录原文、normalized type、level、status、证据和支持强度；
- 明确定义每个 type 的纳入、排除、边界案例；
- 保留 `other` 和 `unclear`；
- 不要把“胰腺炎已确诊但胆源性病因仍被怀疑”压成一个统一置信度。

#### Question

- 定义 recurring target/type；
- 区分一个 primary question 和可选 secondary questions；
- Question 不能只是复述检查名称；
- 每个 question 必须记录 positive/negative answer 会改变什么决定；
- 为每个 recurrent question open-code `answer_requirements[]`：什么信息维度被覆盖后才算
  回答了这个问题；answer requirement 不是推荐的检查方式。

#### Coverage

- Coverage 是当前检查之前全部可用证据相对于当前 question requirements 的覆盖情况；
- 每项 requirement 分别记录：
  `unaddressed | partially_addressed | sufficiently_addressed`；
- 同时记录支持证据和方向：支持、反驳、混合或无方向；
- 可增加总体汇总，但不能用单一汇总替代逐 requirement 记录；
- 严格区分 study adequacy、test-question capability、result status 和 aggregate coverage；
- valid negative 不等于 nondiagnostic、nonvisualized 或 not assessed。

为每个 codebook 保留来源例句、反例、冲突案例和决策规则。不要从直觉直接冻结现有 seed
类型。

### 5. 判断第一批是否达到定性饱和

24条不是固定上限，也不能保证代表性。使用仍未参与 coding 的 development 患者，按新的、
不重叠的小批次（建议每批8–16条）检查：

- 是否继续出现新的 assumption 顶层类型；
- 是否继续出现新的 question 顶层类型；
- 是否出现反复发生但 codebook 没有表达的 answer-requirement 维度；
- `other/unclear` 是否形成稳定、可命名的新簇；
- 某种疾病、timing 或 sequence stratum 是否持续暴露结构缺口。

如果出现实质性新结构，扩展样本、修改 codebook，并用另一批全新的 development 病例再次
检查。记录每轮新增病例、修改原因和结果。只有新病例不再导致实质性 schema 修改时，才能
报告第一层达到定性饱和。

不要打开 final test 来完成这个迭代。

## 本次交付物

至少应交付：

1. 可复现的 development/final-test split manifest；
2. 初始约24条 development codebook 样本 manifest 和 diversity audit；
3. 两种阅读视图下的逐步 open-coding 数据，且来源文本保持可追溯；
4. 第一版正式 assumption codebook；
5. 第一版正式 question + answer-requirement codebook；
6. requirement-level coverage codebook/contract；
7. 所有 `other/unclear`、边界案例、两种视图差异和可能过度合理化的审计记录；
8. 定性饱和检查记录；如果尚未饱和，明确下一批需要补什么，不得虚报冻结；
9. 更新 `experiments/aqc/prompts.py` 的建议或实现，使输出结构支持
   `answer_requirements[]` 和逐 requirement coverage，但本次不要启动全量标注；
10. 更新验证脚本，检查患者不跨 partition、final test 未进入开发样本、源文本完整、枚举合法
    和因果遮蔽边界。

所有文件修改后运行适当的本地验证。最后用简明中文汇报：实际样本规模、每轮是否出现新
类型、最终得到哪些 assumption/question 类型、answer requirements 如何组织、是否已达到
饱和、仍有哪些问题，以及下一步是否可以进入独立框架检查。不要只给计划；请在仓库中完成
能够安全完成的实现和分析。
