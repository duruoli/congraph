**任务：完成描述性 paper 的 Part 3、Part 4,并产出衔接 summary**

**背景**

这是一个关于临床医生偏离诊断 rubric 的描述性研究。Part 1-2 已完成:gap 的形态(commission/omission/order swap)已量化,logistic regression 已识别出 "certainty-driven deviation" 现象(高 certainty 跳步、低 certainty 加检查)。结果在 `results/gap_analysis/`。

现在做 Part 3(机制解释)和 Part 4(efficiency tradeoff 量化)。**这是描述性研究,不做任何 "新策略更好" 的规范性声明。**

注意:当前 rubric simulation 使用 oracle routing(用 ground truth 直接选 sub-rubric),这是已知的设计简化,后续会有独立的一轮用 main triage 重做。本轮分析在 oracle 前提下进行,但所有受 oracle 影响的结论必须显式标注。

---

**Part 3:Edge condition 诊断价值,作为 gap 的机制解释**

目标:解释 Part 2 为什么某些 feature 能预测 gap。假设是——能预测 gap 的 feature,本质上是诊断区分能力强的 feature,它们触发时诊断 certainty 跳升,医生因此跳步。

- 从 `rubric_graph.py` 提取所有 edge 的 condition 涉及的 feature。具体提取方式你看代码决定(condition 是 lambda,可能需要静态分析或按 feature schema 枚举)。
- 用训练集(1044 患者)估计每个 condition feature `c` 对每个疾病 `d` 的诊断价值,log-likelihood ratio:`w(c,d) = log[ P(c=1|d) / P(c=1|¬d) ]`。注意小样本 condition 的估计稳定性,需要做平滑或标注不可靠的项。
- 对每个测试患者,用其 step-0 feature 计算一个 CertaintyScore(已触发 conditions 的 `w` 之和,按其 oracle 疾病或按 top diagnosis,你判断哪个更合理并说明选择)。
- 把 CertaintyScore 作为新变量加入 Part 2 的 commission/omission regression,对比加入前后的预测力(AUC、likelihood)。核心问题:CertaintyScore 是否比单个 feature 更好地、更统一地解释 gap?

---

**Part 4:Efficiency tradeoff 量化(speed + cost)**

目标:量化 deviation 带来的 efficiency 变化的大小和分布。**只报告 tradeoff,不裁决好坏。**

- 用 counterfactual engine,对每个测试患者跑 rubric 轨迹(oracle routing)与 actual 轨迹。
- 计算 Δlength = actual_length − rubric_length,以及 Δcost(用已有的 cost mapping,Radiograph_Chest 按你 Part 1 的处理方式一致对待)。
- 按 Part 2 的 certainty 分层汇总:高 certainty 患者的 deviation 平均省/多做了多少 test 和多少 cost;低 certainty 患者同理。
- 关键诚实性约束:omission 必然在 speed/cost 上更优(少做检查),这是定义决定的,不是发现。报告时必须把这点写清楚,不能暗示 "省了就是好"。

---

**衔接 summary(重要)**

后续会有一轮近乎 parallel 的重做(用 main triage 替代 oracle routing)。写一个简短 markdown summary(`results/gap_analysis/handoff_summary.md`),包含:

- 本轮所有结论中,哪些依赖 oracle routing、会在重做后改变;哪些不依赖、可直接复用
- Part 3 的 `w(c,d)` 表是否依赖 oracle(应该不依赖,因为基于训练集疾病标签),可否直接复用
- Part 4 的哪些数字会因 routing 改变而失效
- 重做时需要保持一致的方法选择(如 cost mapping、Radiograph_Chest 处理、certainty 分层阈值)

---

**输出**

```
results/gap_analysis/
    part3_condition_weights.csv      # w(c,d) 表
    part3_certainty_score.csv        # 每患者 CertaintyScore
    part3_regression_comparison.txt  # 加入 CertaintyScore 前后对比
    part4_efficiency_tradeoff.csv    # 按 certainty 分层的 Δlength/Δcost
    part4_figures/
    handoff_summary.md
    part3_part4_findings.md          # 这两部分的文字发现,描述性语言,无规范性声明
```

实现方式、模型选择、统计细节由你根据数据和代码决定。遇到方法论上的判断(比如 CertaintyScore 按哪个疾病算、平滑方式),在 findings 里说明你的选择和理由。

