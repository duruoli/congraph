# Handoff: Doctor-Reasoning Annotation Pipeline & Certainty-Trigger Agent

> 本文档是一次长对话的总结，作为下一个对话的执行context。
> 配套memory: `~/.claude/.../memory/annotation_agent_design.md`（同一套设计的精简版）。

---

## 0. 一句话目标

从 MIMIC 医生数据中**反向还原医生的决策推理**（when/how/why 偏离 rubric），
用这些标注训练一个 agent：它内部积累一个类 certainty 的指标，
当指标跌破阈值时**提示偏离 rubric**，并给出**如何偏离**（加什么 test / 跳过什么）+ **理由**。

核心动机（也是论文立论）：**朴素的 LLM+rubric 比真实医生更僵化（over-attachment，已被既往实验证明）**。
医生的偏离里携带 rubric 没有的信息；我们把它抽取成一个可学习的 certainty-trigger，
用循证方式给过度 attach rubric 的 LLM 松绑。

---

## 1. 已确定的整体设计（不要再推翻，已逐点讨论收敛）

### 1.1 两条 LLM pass = 两个角色，不是二选一
- **思路2 / policy reconstruction（ex-ante，不给 test 结果）** → 产出 agent 要模仿的 reasoning trace（when/how/why、belief 轨迹）。
  - **必须用 Mode A =「解释已发生的医生动作」**，绝不让 LLM 自己选下一步 —— 锚定真实动作能压制 LLM 的教科书/rubric attachment。
  - （对比 Mode B = LLM 自己决定下一步 = 那是 agent 本身，attachment 是敌人。标注阶段绝不用 Mode B。）
- **思路1 / vindication（ex-post，给结果）** → 只当 reward/质量标签，**绝不回灌进 agent 输入**。

### 1.2 验证是 LOCAL 不是 GLOBAL
判断「医生这步合不合理」要看**是否验证了他自己上一步的预想**，不是看最终诊断对不对
（最终对可能只是靠维持先验蒙对，这一步的 test 仍可能是多余/错的）。拆成：
- **(a) appropriateness**（事前）：这个 test 是否针对医生自己声明的某个真实信息 gap。→ 可进 agent 训练。
- **(b) vindication**（事后）：结果 confirmed / disconfirmed / uninformative（相对医生预想的 finding）。→ 只当 reward。
- **关键连接：(b) 的 local prediction error = certainty score 每一步的更新增量**。
  disconfirmed → certainty 下跌 → 触发下一次偏离。这把「积累一个量」具体化成可计算的东西。

### 1.3 differential 输出形式
**Figure 0 的 5 个 triage 分支（appendicitis / cholecystitis / diverticulitis / pancreatitis / other）+ 一个 open「其他」槽**，
用于接住 rubric 外病因（妇科/泌尿 —— commission 多开 US/CT 最常见的真实动机）。给不确定性一个受约束、可跨病人比较的坐标系。

### 1.4 rubric 的分阶段放置（由 over-attachment 发现驱动）
- **标注阶段**：LLM 输入里**彻底剔除 rubric + disease 标签**。rubric 一旦出现就会让 LLM 把医生还原成「在 follow rubric」，抹掉偏离信号。让 LLM 用通用临床知识扮演医生。
- **agent 阶段**：rubric = default path（省思考）；一个**独立的 certainty-trigger 模块**（用 rubric-free 医生数据训）在 certainty 跌破阈值时 override rubric。**解耦本身就是对抗 attachment 的机制** —— 把 rubric 和决策塞进同一个 prompt 会得到 rubric 克隆体。

### 1.5 deviation 是 LABEL 不是 FILTER
- 不要只标偏离步！要标**所有步**（或 偏离步 + 匹配采样的非偏离步对照），否则学不出 deviate vs follow 的决策边界（只有正样本）。
- deviation 定义是上帝视角（disease-conditioned rubric 算出来的），但**这个泄漏正是信号**（医生的不确定 vs rubric 假装的确定 之间的 gap），只要 disease/deviation 标签**不进思路2输入**。

### 1.6 标注质量保障措施
- **因果遮蔽（causal masking）**：每步只暴露决策前可见信息（见 §3 序列定义）。
- **~~ensemble disagreement 当 certainty proxy~~ — ⚠️ 已被 pilot 证伪，见 §7 Q4**。强 action-anchoring（§1.1 Mode A 必须做）会让多次采样高度一致，分歧度塌缩（偏离步 0.018 vs 非偏离 0.013，无区分度）。**certainty 信号改从 vindication / local prediction-error 读（§1.2 (b)），不从采样方差读。** 采样多次仍可保留，但仅用于 grounding 稳健性/捞 parse 失败，不当不确定度。
- **extractive grounding**：每个 claim 必须引用具体特征字段，不能引用就不许断言。（pilot 验证：质量很高，单步可引 9 个具体字段。）
- **整条序列联合标注**：保证 belief 轨迹连贯（医生 belief 应随信息单调演化）。（pilot Q1 通过。）
- **交叉验证锚点**：用已有特征算 Alvarado/BISAP，与 LLM 还原的不确定度比对，不一致的 flag 人工复核。

### 1.7 偏离类型（从 differential 形状读出，不是硬分类）
- **Type A 替代假设**：differential 分散到其他病 → 怀疑不是 rubric 那个病。
- **Type B 不确定性**：differential 集中在 rubric 病、但 action 是 rule_in → 信本病但信息不足。
- **Type C 无理由/次优**：raw data 找不到支撑 → 由 ensemble disagreement 涌现，不让 LLM 主动判定（它知识有限易误判）。

---

## 2. 数据规模与三层数据（本次已核实）

**deviation 规模**（`results/gap_analysis/joined_data.csv`，300 患者）：
- 235/300 有偏离；~338 个 deviation events（235 commission + 103 omission）；4 个病。
- 按病：pancreatitis 92+44、cholecystitis 70+37、appendicitis 49+22、diverticulitis 24+0。
- 对 LLM 标注完全可行（成本约 $10–30）；fine-tune 偏少，够 few-shot / LoRA。

**三层数据（2026-06 已重整目录，见下）：**
| 层 | 路径 | 内容 | 用途 |
|---|---|---|---|
| ① Raw 报告 | `data/raw_data/{disease}_hadm_info_first_diag.csv`(及 .pkl) | 自由文本 HPI / Physical Exam / Lab(itemid字典) / **Radiology(每份报告全文)** / + Discharge Dx·ICD·Procedures(=泄漏字段,禁用) | **还原医生(用这层)** |
| ② Rubric-specific feature | `data/rubric_features/{disease}_features.json`（4 个 json，原 `results/features_extraction/`） | 逐 step **accumulative** 的 rubric 二值特征（idx_k 已含 test-k 结果）；由 `scripts/run_feature_extraction.py` 生成 | 跑 rubric / gap analysis / belief-deviation traversal（**还原医生不要用,循环/泄漏**） |
| ③ State trajectory | ~~`data/data_{d}/state_trajectories_denoised_{d}.json`~~ → 已 **archive** 到 `_archive/old_data/` | 旧实验的逐 step 抽象状态；**无任何 live 代码引用，已弃用** | 不再使用（层②的 `{disease}_features.json` 取代了它做 step 对齐） |

`joined_data.csv` = 层② + deviation 标签（gap_analysis.py 采样 300 建的）。

**目录重整记录（2026-06）：** 之前 §2 把层② 误标为 `data/data_{d}/patient_features_{d}.csv`，实际 live 的 rubric 特征是上表的 4 个 `{disease}_features.json`（已从 `results/features_extraction/` 移入 `data/rubric_features/`，并修正所有代码引用）。`data/data_chole/`、`data/data_diver/`、`data/data_pan/`（含 patient_features_、state_text_、state_trajectories_、extracted_sequences_、conformance_results_ 等）均为旧实验产物、无 live 引用，已整体移到 `_archive/old_data/`。`data/` 现仅含三个 live 目录：`raw_data/`（层①）、`rubric_features/`（层②）、`masked_views/`（标注阶段的因果遮蔽视图）。

**关键已核实事实：**
- ID 对齐：`joined_data.patient_id` == raw `hadm_id`，chole 93/93 完全对上。
- Lab 可解码：`data/raw_data/lab_test_mapping.csv`（itemid→label）。
- **Python 环境**：`/opt/anaconda3/bin/python3.12` 有 pandas；系统 `python3`(3.14) 没有。
- 泄漏字段（必须硬删，绝不进 LLM 输入）：Discharge Diagnosis / ICD Diagnosis / Procedures*。

---

## 3. 序列定义（本次讨论最终收敛，因果遮蔽地基已验证 ✅）

**按用户思路：step 0 = HPI+PE+Lab 合并为起始步；之后只有腹部诊断影像才算 decision step。**

```
诊断决策序列 =
   step 0: HPI + Physical Exam + Lab tests (baseline, 对所有后续决策永远可见)
   + 影像报告, 按 Note ID 的 RR-N 编号升序排列 (时序 proxy, 不是列表顺序!), 分三类:
       · DROPPED  — 管理期/治疗性, 整条剔除不进序列:
                    Exam Name 含 PORTABLE / PRE-OP / LINE PLACEMENT / PICC, 或 Modality = ERCP / Drainage
       · DECISION — rubric 相关腹部诊断影像 (US/CT/MRI/MRCP/CTU-abdomen):
                    唯一需要 LLM 解释「医生为何下这个 order」+ 更新 belief 的步, 也是 deviation 标签唯一落点
       · CONTEXT  — 其余存活影像 (chest X-ray / 腹部平片 / head CT ...):
                    其报告「结果」按 RR-N 并入后续 decision 的可见上下文, 但不单独成步、不让 LLM 编造下单动机
   因果遮蔽: 决定第 i 个 decision 步时, 只暴露 baseline + 所有 RR-N 更小的报告(含 context 与更早 decision 的结果),
             遮蔽该 decision 步自身结果及所有更晚报告
```

**为何 chest X-ray 降级为 CONTEXT(option B，本次数据驱动决定）**：
527 份 diagnostic `CHEST (PA & LAT)`（已排除 portable/pre-op）里，41% 排在腹部影像**之后**（纯术前/并发症筛查）；
59% 在之前的，报告原文也几乎全是例行阴性片（"lungs clear, no free air"），仅 ~19% 真正在问 free-air/肺炎这种鉴别问题。
让 LLM 给一份例行阴性胸片**编造鉴别动机 = 污染 belief 轨迹**。而 chest 步本来就不带 deviation 标签，
其唯一价值（医生"已知胸片阴性"这个信息）用 CONTEXT 身份即可保留 → **保留结果、不逼 LLM 解释下单动机**。
（彻底删 chest = option A 也可；保留少数命中 free-air 的升格 = option C，工程脆弱，未采用。）

**验证地基时发现并已解决：**
- baseline 信息量足够：chole 93 例 PE 中位 321 字符(min133/max1979)，HPI 丰富，Lab 完整。之前担心 PE 太短(109)被证伪。
- **列表顺序 ≠ 时序**：8/93 例 Radiology list 顺序与 RR-N 不一致 → **必须按 RR-N 排序**。
- **管理期影像污染**：43/93 例、87/336 报告是 PORTABLE/PRE-OP/ERCP → 关键词可干净识别。位置验证：PRE-OP 中位位置=1.00(永远最后)、PORTABLE=0.67、普通chest=0.50，说明 RR-N 排序抓住了真实时序结构。
- raw radiology 比 rubric 序列丰富得多（chest X-ray 最常见 123 次 > 腹部影像）。包含这些非腹部影像让 belief 轨迹更真实，但 deviation 标签只在腹部诊断影像步上有。

**⚠️ RR-N 是 proxy 不是 ground truth**：提取数据丢了 MIMIC 原始 `charttime`（pkl 也没有）。RR-N 是当前唯一时序信号、且规律自洽、大概率正确。
**Rigor TODO**：回 MIMIC 源 `radiology` 表拉 `charttime` 替换 RR-N（需 MIMIC 访问），可顺带修正 8/93 乱序。先用 RR-N 跑通，这个作为待补的严谨性步骤挂着。

**已有工具**：`scripts/build_masked_view.py` —— 给定 disease + hadm_id，打印每步因果遮蔽后的可见 context。
**已升级 ✅**：RR-N 排序 + DROPPED/DECISION/CONTEXT 三分类（见上）。`--json` 落盘到 `data/masked_views/<disease>_<hadm>.json`
（结构：baseline / radiology_order[role] / decision_points[visible_prior_imaging 含 context 报告全文]），可直接喂 LLM prompt。

---

## 4. 下一步：6-8 case 小实验（验证组合设计，未开始）

挑 6-8 个 case（4 病 × deviation/non-deviation），同时跑两条 pipeline，测 4 件事：
1. **belief 轨迹连贯性**：思路2 自回归还原的 differential 随 step 演化是否合理、有无前后矛盾。
2. **ex-ante 猜测 vs ex-post 验证 一致性**：思路2 猜的「医生想确认 X」，思路1 用结果看是否确实是 X；分叉处是否 informative。
3. **deviation 中 triage-artifact 占比**：多少「偏离」其实是医生还没分诊、撒大网的 triage 行为（验证 §1.5 那个「泄漏即信号」reframe）。
4. **ensemble disagreement 是否 track deviation**：偏离步的多次采样还原是否比非偏离步更分歧（→ disagreement 当无监督 certainty proxy）。

实验前要做的工程：
- ~~在 `build_masked_view.py` 基础上补 RR-N 排序 + 过滤~~ **已完成 ✅**（含 chest→CONTEXT 降级，见 §3）。
- 写思路2 的 Mode A prompt（rubric-free、输出 5+open differential、extractive grounding、action_role）。
- 写思路1 的 vindication 判定（用被遮蔽的那份报告结果，对比上一步预想 → confirmed/disconfirmed/uninformative）。
- LLM client：参考既有 `experiments/llm_experiment/llm_client.py`；key 在 `.openrouter_env` / `.openai_env`。

---

## 5. 阶段路线（远期）
Prompting（验证框架，含本标注） → Fine-tune（几百标注后，专化 deviation reasoning） → RL（可选，reward 是难点：local imitation + prediction-error，offline RL 有 distribution shift 风险）。
RL 非必须，fine-tune + 好 prompting 框架已是 solid 贡献。

---

## 6. 给下一个对话的第一步建议
1. 读本文件 + memory `annotation_agent_design.md`。
2. 用 `/opt/anaconda3/bin/python3.12`。
3. ~~升级 `build_masked_view.py`~~ **已完成**（§3）。
4. ~~挑 6-8 个 case 跑思路2 Mode A~~ **已完成,见 §7**。下一步见 §7 末「待办」。

---

## 7. Pilot 结果（8 case × 4 病 × dev/non-dev，30 决策步，已完成 ✅）

**代码**：思路2 Mode A + 思路1 vindication 在 `experiments/annotation/{prompts,annotate}.py`；
runner `scripts/run_annotation_experiment.py`（含 8-case 列表 + 4 问聚合）；
模型 `anthropic/claude-sonnet-4-6` via OpenRouter；成本 ~$1.5。
输出 `results/annotation_experiment/{disease}_{hadm}.json`（完整 ensemble+grounding+vindication）+ `summary_steps.csv`。
（查 OpenRouter 花费：`scripts/check_openrouter_usage.py`。）

**四个问题结论：**
- **Q1 belief 轨迹连贯性 ✅**：differential 演化干净、无矛盾。如 diverticulitis 26371704 dive `0.50→0.74→0.74→0.89` 单调；appendicitis 23202997 起手 chole 最高→CT 后 appe `0.20→0.30→0.72` 正确翻盘。
- **Q2 ex-ante vs ex-post 一致性 ✅ 有区分度**：vindication 18 confirmed / 9 disconfirmed / 3 uninformative（~30% 被推翻，非退化）。**pancreatitis 21282967 是关键示例**：s1↓s3↓s4↓s6↓ 反复 disconfirmed → 医生反复加做影像(7 步) = 真实数据里「local prediction-error 驱动持续偏离」的直接证据，坐实 §1.2。
- **Q3 triage-artifact ⚠️ 部分**：8/13 偏离步 `other`>0.25，但 `other` 均值偏离(0.42) vs 非偏离(0.43) 几乎相同 → `other` 反映**病人复杂度**(post-ERCP/妇科/泌尿)，不是 step 级「偏离 vs 不偏离」的干净判据。
- **Q4 ensemble disagreement ❌ 证伪**：偏离 0.018 vs 非偏离 0.013，都极小。强 action-anchoring 让采样高度一致，分歧度塌缩。**淘汰 §1.6 的 disagreement-as-certainty 分支。**

**方法论收获（最重要）**：certainty 信号来自 **vindication / local prediction-error（Q2 有效）**，不是 ensemble disagreement（Q4 失效）。不确定度从 `confirmed/disconfirmed` 序列 + differential 形状(entropy/`other`)读。Mode A 质量本身很高（extractive grounding 严格，vindication 是真正 LOCAL）。

**待办（下一步）：**
1. **deviation 标签按 event 对齐**（当前 runner 按 modality 匹配 commission 会过标：一个 case 多次 CT 全被标 dev）。
2. 把 certainty score 形式化：用 vindication 的 confirmed(+)/disconfirmed(−)/uninformative(0) 当每步增量，跑出一条 certainty 轨迹，对照「医生何时停 / 何时继续加做」。
3. 扩样到几十例做 fine-tune 前的标注集；differential entropy + `other` 作为辅助不确定度特征。
4. （Rigor）回 MIMIC 源补 `charttime` 替换 RR-N（§3 挂着）。
