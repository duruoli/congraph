# IBD MIMIC action labeling：第一版输入与 prompt 管线

## 当前范围

每个原始 action 单独形成一个待标注对象，即使它与其他操作拥有同一个记录时间。`action_type` 优先选 `observation` / `intervention`，再给八轴中的适用值。`action_text` 原样保留检查名、药名、标本或操作名称；它是具体内容，不是类别。纯入院/转科不在两类行动主队列。

输入组装器：`scripts/build_ibd_action_label_inputs.py`。可讨论的 prompt **定义段**：`IBD_MIMIC_ACTION_LABEL_PROMPT_DEFINITIONS_DRAFT.md`。八轴取值由脚本在运行时从 `unified_axes_v1.yml` 读取，不手写在 prompt 文件中。当前脚本只生成本地 prompt JSONL，**不调用 LLM，也不产生自动标签**；用户讨论定义后再接模型调用与输出校验。

## LLM 输入：按用途选择，不默认放整份报告

| 输入 | 当前是否包含 | 作用与证据边界 |
|---|---|---|
| 目标 action 名称及元数据、来源 ID、记录时间/精度、执行状态 | 是 | 决定 `action_type` 和本次 modality/intervention 的主要依据。处方开始并非给药，影像 charttime 并非医嘱时间，ICD 操作只有日期。 |
| 年龄、性别、行动是否位于住院区间 | 是 | population/setting 的基础。住院区间由结构化时间计算，不向 prompt 提供未来出院时间。 |
| cohort IBD 类型 | 是，明确标为 retrospective | 可作全程疾病背景线索，不能伪称当时已确诊。 |
| 该次影像报告 `INDICATION` / `CLINICAL INFORMATION` / `HISTORY` | 有该字段时包含 | 用于该步的临床问题、management、anatomy、phase；是后来写入报告的行动目的代理，不是证实的医嘱时信息。 |
| 此前已可用影像报告的 indication + impression | 最多最近 4 份 | 辅助判断已知异常与监测轨迹。只用 `storetime < action_at`；不是医生确已阅读的证据。 |
| 此前已可用的选定化验值 | 最多最近 10 条 | 为 phase/问题提供结构化趋势线索；目前仅选择少数炎症和常见病情相关项目。 |
| 此前处方开始、操作日期 | 最多最近 8 条 | 治疗轨迹上下文；不推断处方已经执行。日期操作只在整日早于该次 action 时纳入。 |
| 目标报告 `FINDINGS`、`IMPRESSION`、全文，以及更晚报告 | 本轮不包含 | 这些属于行动后结果或后见证据。`retrospective_at_action` 的 phase 重建应单独建立第二遍输入，不污染当前 scene 标注。 |
| 病程记录、出院总结、微生物培养结果值、精确医嘱 | 暂未包含 | 现有 ledger 未保留足够内容或当前 extract 缺源；药物/操作的管理目的可能因此必须填 unknown。 |

输入有显式 `context_omitted_older_counts`。超过上限的旧资料不会悄悄消失；标签若依赖被省略的内容，应暂记不足证据并复查。一个报告缺 indication 时，绝不退回读取全文来猜目的。

## 原六步人工标注实际看过什么

人工 toy 使用了：影像检查名、该次报告 indication（若存在）、此前已完成报告的发现/结论、病例级 UC/结肠背景、年龄和住院时间。当前检查的结果与全程出院结论**没有**作为该步管理目的或当时 phase 的直接依据。因此，单给 exam name 通常足够标影像方式，但不足以区分“初次评估并发症”与“连续监测扩张”。六步试标本身还缺同期化验、处方执行与进展记录，不能视为金标准。

## 运行与检查

```bash
/opt/anaconda3/bin/python3.12 scripts/build_ibd_action_label_inputs.py
/opt/anaconda3/bin/python3.12 scripts/build_ibd_action_label_inputs.py --source-table prescriptions --limit 2 --output runs/ibd_action_label_inputs_pilot_20/prescription_scene_prompts.jsonl
/opt/anaconda3/bin/python3.12 scripts/build_ibd_action_label_inputs.py --source-table procedures --limit 2 --output runs/ibd_action_label_inputs_pilot_20/procedure_scene_prompts.jsonl
/opt/anaconda3/bin/python3.12 tests/test_ibd_action_label_inputs.py
```

输出严格限制在 gitignored `runs/`，其中包含私有患者上下文；Git 跟踪的文档和测试不含患者原文。20 人 pilot 的默认 radiology 构建产生 37 个**原始 action** prompt（来自 35 个含 imaging 的 action group）；36 个识别出目标 indication，1 个未识别。对该病例不可因缺 indication 而臆测 clinical question。

## 尚待与用户迭代

1. 定义段中每一轴的对象、空值和证据门槛，尤其 `clinical_phase` 的 `as_known_at_action` 与以后独立的 `retrospective_at_action`。
2. 是否先在 radiology 35 个 group 上人工审核若干 prompt，再扩展到药物、化验、培养、操作；这些来源的目的常缺失，可能需要额外临床记录。
3. 后续增加 LLM 输出 JSON schema 校验、YAML 值校验、证据 ID 核对和人工抽查；在定义段确认前暂不批量调用 LLM。
