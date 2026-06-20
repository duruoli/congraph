# Handoff: Certainty-Trigger Agent — Training Goal Decomposition

> 本文档把"用 LLM annotation 材料训练 deviation/certainty agent"这个目标拆成可执行的工作流。
> 上游：`HANDOFF_annotation_pipeline.md`（数据怎么来）、`llm_annotation_analysis.md`（抽出来的医生知识 = 行为规范）。
> 配套 memory：`annotation_agent_design.md`、`belief_step_deviation_traversal.md`、本文件对应的 `agent_training_plan.md`。

---

## 0. 一句话目标

给 agent 提供 rubric + 病人当前状态，让它在每个决策点：
**(1) 维护一个 certainty 信号；(2) 决定 follow / deviate / stop；(3) 给出 how + why。**
方向是双向的——既要**循证地松开** rubric over-attachment（deviate+confirmed 抽出的医生知识），
又要**循证地刹车**（disconfirmed 两组抽出的「该停 / 负结果不是失败」约束）。
朴素 LLM+rubric 既会太僵（不敢偏离），松开过头又会变成乱加检查——agent 要同时学会两侧。

---

## 1. 两个交付方向（共用同一套训练材料与行为规范，载体不同）

| | 方向 A：Context-injection（闭源大模型） | 方向 B：Fine-tune（小开源模型） |
|---|---|---|
| 目标模型 | sonnet-4.6 / gpt-4o 等 | qwen 等（LoRA/SFT） |
| 机制 | 在 prompt 里注入 rubric + 检索到的 annotation exemplar + certainty 规则 | 把 annotation trace 编译成 SFT 样本，把行为规范内化进权重 |
| 优势 | 0 训练成本、快速迭代、可解释 | 部署便宜、低延迟、隐私可控、可放规则进权重 |
| 风险 | exemplar 检索质量 = 上限；长 context 成本；LLM 自带知识与 MIMIC 重叠（非独立信息源，需标注） | 样本量偏少（~338 deviation events）→ 只够 LoRA/few-hundred SFT；过拟合 4 病 |
| 主要瓶颈 | retrieval + prompt 结构 | 样本构造 + 防泄漏 + eval |

**两方向同源**：行为规范（§2）、训练样本字段（§3）、certainty 形式化（§4）、eval（§5）完全共用；
只有"如何把样本喂给模型"不同（A=in-context exemplar，B=SFT target）。先把共用层做扎实，再分叉。

---

## 2. 行为规范（policy spec —— agent 该学的目标行为，全部来自 `llm_annotation_analysis.md`）

这是 agent 的"应然"对照表。把它当成 eval rubric，也当成 prompt 里的规则草稿。

**A. 何时该松开 / 偏离 rubric（deviate+confirmed，119 步）**
- A1 诊断已定 → 把影像重新瞄准 severity / etiology / complication（按当前**临床问题**路由，不按病名）。
- A2 study-adequacy gate：检查"不充分"≠"阴性"；不充分/技术受限的片子不该关闭 workup。
- A3 双流不一致（临床 vs 检查）本身就是警报 → 降 certainty、触发下一步影像，正好在 rubric 会停的地方。

**B. 何时该刹车 / 不该再走（disconfirmed 两组）**
- B1 unavoidable / calibration（FD-1、DD-2，胆道排查阴性）：检查是对的，**阴性就是答案**。
  → 预登记现实的 P(阴性)；用 disconfirm 更新**信念**（P(胆石病因)↓）而非决策信心；**第一次干净胆管后停止升级**。
- B2 current-state gate（FD-2 陈旧解剖）：成像前先核对器官是否已切除/引流/吸收 → 跳过徒劳检查。
- B3 redundancy down-weight（FD-3）：CT 已刻画的器官，再换模态"确认"低产出；"扩张⟹结石"要按胆道干预史 condition。
- B4 respect valid STOP（DD-1，过度复扫）：rubric 处于 terminal/blocked 时的防御性复扫，并发症多半不存在 → **抑制**，不模仿。

**C. reward 设计的硬约束（disconfirmed ≠ wrong action）**
- C1 区分 **disconfirmed-prediction** 与 **wrong-action**：广撒网命中真相（DD-3：28238173、25444703、28672604）必须**不被惩罚**——这就是 outlier-patch。
- C2 certainty 更新作用于**信念**不作用于**决策质量**：一次 appropriate 检查的阴性该降 belief，不该降"这步做得对不对"。

---

## 3. 训练样本构造（共用层；从 `results/annotation_experiment/full/*.json` 编译）

每个 decision step → 一条样本。字段三分：

- **INPUT（agent 推理时可见）**：
  - baseline + 因果遮蔽后的可见 prior 影像结果（已有：`build_masked_view.py` 的 `decision_points`）。
  - **rubric**：agent 阶段要给（与标注阶段相反！标注剔除 rubric，agent 提供 rubric——见 annotation_agent_design §1.4）。给整张 sub-rubric 图（节点+边+可读 condition），不只下一步。
  - （方向 A 额外）检索到的 top-k annotation exemplar；（方向 B）不放，目标内化。
- **TARGET（监督信号，来自 ex-ante 重建）**：
  - 动作：follow / deviate / stop + 具体 modality。
  - reasoning trace：differential(5+other)、information_gap、expected_finding、action_role、extractive grounding。
  - dev_belief 三态（follow/deviate/off_rubric）+ rubric_state（为什么）当结构化标签。
- **REWARD/质量（ex-post，绝不进 INPUT）**：
  - vindication（confirmed/disconfirmed/uninformative）+ certainty_update(±/0)。
  - appropriateness（yes/partial/no）。
  - 按 §2-C 的约束使用（disconfirmed≠惩罚）。

**防泄漏（硬规则，沿用 annotation pipeline）**：Discharge Dx / ICD / Procedures 永不进任何样本；disease/deviation 上帝视角标签不进 INPUT（只当 eval/label）；vindication 不进 INPUT。

**样本量现状**：~338 deviation events + 匹配的 follow/non-deviation 对照（§1.5 deviation 是 label 不是 filter，必须含负类/follow 步，否则学不出边界）。430 judged steps 里 follow 181 / deviate 169 / off_rubric 80，可直接当四类决策样本。

---

## 4. Certainty 信号形式化（共用层；目前 memory 标为"待做"）

- 每步增量 = vindication 的 confirmed(+) / disconfirmed(−) / uninformative(0)（annotation_agent_design §1.2(b)）。
- 跑出每个 case 的 certainty 轨迹，对照"医生何时停 / 何时继续加做"。
- 辅助不确定度特征：differential entropy + `other` mass（Q3：是病人复杂度信号，弱 step 级判据，只作辅助）。
- ⚠️ ensemble-disagreement 已证伪（Q4），不用采样方差当不确定度。
- **关键**：要把 §2-B1/C2 编码进去——一次 appropriate 检查的阴性是**预期内**下跌（calibrated），不是触发恐慌的"意外"；区分 belief-drop 与 alarm。

---

## 5. 评估（共用层；扩展现有 `results/llm_experiment/` 框架）

- **复现力**：LLM/agent 推荐序列 vs actual，算 exact-match / commission / omission / order-swap，按病分层，和 rubric-only / KNN-only baseline 同表（`gap_comparison_table.csv` 已有骨架）。
- **机制验证**：按 certainty tercile 分层——高 CS 是否倾向 shortcut（更短/omission↑），低 CS 是否倾向加检查（commission↑），对照 actual。
- **行为规范命中率（新，用 §2 当 rubric）**：在 held-out 的 deviate+confirmed 上是否复现 A1-A3；在 disconfirmed 上是否触发 B1-B4 的正确刹车（尤其 DD-1 该 stop、DD-3 该 keep）。
- 诚实性约束（沿用）：LLM 自带知识与 MIMIC 重叠 → 标注；不裁决 strategy"更好"，只描述谁更接近 actual / 谁和 CS 更一致。

---

## 6. 里程碑顺序

1. **编译共用训练集**（§3）：从 full/*.json + belief_deviation_analysis.csv 生成 step 级样本（INPUT/TARGET/REWARD 三分，含 follow 负类）。
2. **Certainty 形式化**（§4）：vindication→轨迹，落 certainty 列。
3. **方向 A 先行（便宜快）**：prompt = rubric 图 + top-k exemplar + §2 规则；在 held-out 上跑 §5。这步同时验证行为规范是否可学。
4. **方向 B**：把样本编译成 SFT 格式，LoRA 微调 qwen；同一 held-out 对比方向 A。
5. （可选 RL）reward = local imitation + prediction-error；offline RL 有 distribution-shift 风险，非必须（annotation_agent_design §5）。

---

## 7. 开放问题 / 风险

- 样本量对方向 B 偏小（4 病、~338 events）→ 优先 LoRA/few-shot，警惕过拟合；可能需把 follow/off_rubric 步也充分纳入扩容。
- exemplar 检索（方向 A）的相似度定义 = 上限，复用现有 KNN engine 但要按"当前临床问题"而非纯特征相似。
- RR-N 仍是时序 proxy（charttime rigor TODO 挂着）；51/293 patient rrn_aligned=False 被跳过，扩样前考虑补 charttime。
- biliary 是 post-hoc 派生 belief，不是重建的 disease（不会被 over-attribute），但其 vindication 质量较杂（~56% confirmed）——当训练材料时按 §2-C 谨慎用。
- §2 的行为规范源自 4 病的 542 步，外推到新病种前需重标。
