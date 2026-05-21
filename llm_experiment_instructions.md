# LLM next-test recommendation experiment

## 目的

测试加入 LLM 这个新信息源后，对 actual deviation pattern 的解释/复现能力。
两个并行问题：

1. **复现能力**：LLM 推荐 vs actual sequence 的吻合度，对比 rubric-only
2. **机制验证**：LLM 的推荐行为是否和 CertaintyScore 一致（高 CS 时倾向 shortcut，低 CS 时倾向加检查）。

这是描述性 paper 的 supplementary experiment，不是 prescriptive claim。

## 实验设计

### Conditions（消融）

每个 patient 在每个决策节点跑以下 condition，记录 LLM 推荐的 next test：

- `llm_features_only`：只给 patient features
- `llm_full`：patient features + rubric recommendation + KNN similar patients

KNN 信息呈现方式：取 top-5 相似患者，按 similarity 降序排列，
每个患者展示其完整 test sequence（不只是 next test，提供 context）。

### Baselines（已有，不需要重跑）

- `rubric_recommendation`：来自 rubric simulator 的下一步推荐

### Order-sensitivity 对照

选 3 个 patient（不同 disease、不同 CertaintyScore 高/中/低），对每个 patient 跑：
- info 顺序 A：rubric 在前、KNN 在后
- info 顺序 B：KNN 在前、rubric 在后

看 LLM 推荐是否稳定。这个对照先做，结果不稳定的话主实验需要 average over 多次顺序。

## LLM 设置

用 GPT-4o。temperature 设 0 求稳定。API key 从环境变量读取（OPENAI_API_KEY）。
每个 prompt 让 LLM 输出 JSON：
```json
{
  "next_test": "CT_Abdomen",
  "reasoning": "1-2 句话解释为什么"
}
```

## Prompt 结构

System prompt 说明任务：临床医生角色，根据 patient 当前状态推荐下一个诊断 test。
可选 test 集合限定在已有的 vocabulary：Lab_Panel, Ultrasound_Abdomen, CT_Abdomen,
MRI_Abdomen, MRCP_Abdomen, Radiograph_Chest, HIDA_Scan。

User prompt 包含（按 condition 不同包含不同部分）：
- Patient features（step-k 的 feature dict，只展示 True 的 binary feature 和有值的非 binary feature）
- 已完成的 tests
- **整个 sub-rubric 的 graph 结构**（不只是下一步推荐）。把 sub-rubric 序列化成 LLM 可读的格式：
  节点列表（id, label, type, required_tests）+ 边列表（source, target, condition 的自然语言描述）。
  condition 是 lambda 表达式，需要转成可读文本（比如 `alvarado_score >= 7` 而不是代码）。
  这个序列化你看 rubric_graph.py 决定怎么处理最干净。
- Top-5 相似患者的 test sequences（按 similarity 排序）

prompt 设计上保持简洁，但 sub-rubric 这部分必须完整传达。

## 评估

### 主指标：gap descriptive analysis

对每个 condition 的 LLM 推荐序列，和 actual sequence 比较，计算 Part 1 那套指标：
- exact match rate
- commission rate
- omission rate
- order swap rate

按 disease 分层。和已有的 rubric-only / KNN-only baseline 放在同一张表里对比。

### 机制验证

按 CertaintyScore 的 tercile 分层（用 part3_certainty_score.csv 里的
certainty_score_oracle），看：
- 高 CS 患者中，LLM 是否倾向 shortcut（推荐序列更短、omission 更多）
- 低 CS 患者中，LLM 是否倾向加检查（commission 更多）
- 和 actual 在同样分层下的行为对比

### 输出 LLM reasoning 分析

把所有 LLM 的 reasoning 文本保存。事后人工 sample 看 LLM 在做决策时是基于 rubric、
KNN、还是自己的 medical knowledge。这个不需要全自动分析，存下来即可。

## 实现注意

- **先跑 30 个 patient**（不是全部 300）。从 4 个 disease 各抽 7-8 个，按 CertaintyScore tercile 覆盖
  高/中/低（保证机制验证那部分有分层数据）。具体采样策略你决定，但要保证 disease 和 CS 都有 spread。
  跑完确认 pipeline 没问题、初步信号合理后，再决定是否扩到全量。
- 决策节点：跟 rubric_sim 一样的逻辑，到 terminal node 或 max_steps 停
- step-k feature 用 KNN counterfactual engine（和 rubric_sim 用的同一个）
- LLM call 数量预估（30 patient 规模）：30 × 平均 2 steps × 2 conditions ≈ 120 次。
  先跑 order-sensitivity 的 3 个 patient × 2 顺序 × 平均 2 step = 12 次确认稳定性，再跑主实验。

## 输出文件

```
results/llm_experiment/
    order_sensitivity_check.csv      # 3 个 patient 的对照结果
    llm_recommendations.csv          # 所有 condition × patient × step 的 LLM 推荐
    gap_comparison_table.csv         # 各 condition 的 gap descriptive stats 对比
    certainty_stratified.csv         # 按 CS tercile 分层的行为对比
    llm_reasoning_samples.md         # LLM reasoning 文本（选 20-30 个 sample）
    findings.md                      # 描述性 findings，无规范性声明
```

## 诚实性约束

- LLM 的 medical knowledge 可能与 MIMIC 训练数据有重叠，"LLM common sense" 不是
  独立 information source。在 findings 里标注。
- 不裁决任何 strategy "更好"。只描述谁的推荐更接近 actual、谁的行为模式和 CS 更一致。

## Handoff

写一个简短的 markdown 衔接 summary（接到现有的 handoff_summary.md 末尾或新建），
说明本实验的哪些结果在 main-triage 重做后需要重算（涉及 oracle routing 的部分）。
