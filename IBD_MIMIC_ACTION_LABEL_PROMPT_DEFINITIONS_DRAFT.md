# MIMIC IBD action label prompt：定义段（待逐条讨论）

本段由输入组装器原样放入 LLM prompt；八轴允许值由运行时 `unified_axes_v1.yml` 自动附在本段之后。当前是供人工迭代的 v0.1，不是已批准的最终标注规范。

## 标注对象

每次只标一个**已记录的原始 action**，即 `target_action`。同一时间的多个原始 action 各自标注；不要将同时发生误作因果顺序。`action_text` 保留原始检查名、药名/途径、标本名或操作描述。`action_type` 是它的一级功能类别，优先只选：

- `observation`：主要为获取或更新患者信息的行动，包括初次诊断、筛查、病情监测、随访检查、风险评估。这里指**做检查/采样**，不是后来产生的结果。
- `intervention`：主要意图改变病情或管理状态的行动，包括药物、治疗性操作、手术、营养治疗。根据记录区分处方、执行与实际给药；不能把处方说成已给药。

若一项操作同时有两种功能，仍给一个主类 `action_type`，另填 `secondary_action_type` 与依据；如果无法合理选主类，标 `requires_review=true`。入院、转科等纯照护转换不进入此二分类的主标注队列。勿把“医生做了某个诊断/治疗判断”当成已记录的 action。

## 八轴含义

- `disease`：本 action 所服务的疾病/临床问题；区分背景诊断、当前焦点、竞争假设及回顾性 cohort 编码。UC/CD 有现有值；只有泛称 IBD 时保留 `unmapped:ibd_unspecified`，不能用 `all`。
- `management_domain`：本次**为何**行动，属于诊断评估、监测、筛查、治疗选择、并发症管理等哪类任务。不能由检查/药名独自推断目的。具体临床问题另写 `clinical_question`。
- `diagnostic_modality`：本次 observation 用来获取/表征信息的方式，涵盖诊断、监测、筛查等目的；纯 intervention 填 `n_a`。若方式已知但词表无值（如腹部平片），标 `unmapped` 并保留原词，绝不标 `unknown`。
- `intervention_class`：本次 intervention 的治疗/管理类别；纯 observation 填 `n_a`。之前或计划中的治疗不能当成本 action；兼具诊疗作用时两个轴都可激活。
- `clinical_phase`：该时点所处病程/治疗轨迹。先标 `as_known_at_action`，使用当时可用的证据及该次报告 indication（注明其为回顾性代理）；全程回顾重建另起一遍 `retrospective_at_action`。疑似与确立须分开；不能因进行监测而填 `monitoring_remission`。
- `population`：当步相关的患者亚组。年龄可用结构化资料，其他特征需证据；不要自动假设免疫抑制。
- `setting`：行动发生时的照护环境，以时间和住院区间判断；不能仅凭 cohort 入选判断为住院。
- `anatomy`：行动面对的**临床问题聚焦的器官**，优先据 indication 和此前证据，再参考检查范围；不能将后见影像发现当作行动时的焦点。

## 证据和空值

输入中的 `target_report_indication` 可辅助还原行动目的，可能由报告后来记录，因此视为 retrospective proxy。`prior_available_reports` 只有报告可用时间严格早于 action 时间者；可用不等于医生实际读过。`prior_actions` 只是记录动作，处方不等于给药。`cohort_ibd_type_retrospective` 仅作全程背景线索，不证明当时已确诊。

每个轴返回 YAML 中的零或多个合法 value，并为每个 value 写 `assertion`（documented / inferred / suspected / historical / retrospective）、`evidence_ids` 和一句简短依据。另允许：`n_a`（此轴不激活）、`unknown`（应有值但证据不足）、`unmapped`（概念明确但 YAML 无值，须给 `raw_concept`）。不要填无证据的值，也不要用 `n_a` 掩盖未知。当前报告 findings/impression 与后续记录不在本轮 scene 输入中；不得假装知道检查结果。输出只需严格 JSON。
