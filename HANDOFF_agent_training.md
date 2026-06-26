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

## 2.5 "显而易见失误"的客观判据（mistake detectors —— 两阶段共用，SFT 降权 + RL reward 都用）

> 核心原则：**判据绝不建立在单次 outcome（confirmed/disconfirmed）上**，只建立在事前可知 / 结构性 / 跨步可离线算的条件上。outcome 退回它唯一的位置——更新下一步 belief（`certainty_update`），不参与"这步动作好不好"。
> 这四条是探测器，不是 RL 专属：四条都不需要 agent rollout，所以 SFT 阶段就能用（hybrid target 的降权/重标），RL 阶段再用同一判据做 reward shaping + 反事实。按客观性排序：

1. **Ex-ante 逻辑失败（最干净，单步、结果无关）**：`appropriateness="no"` —— 事前就没有真实 gap，或这个 test 根本 address 不了 gap。结果出来前就能判定。→ SFT 直接降权最干净。
2. **结构性过度成像（DD-1，⚠️ 已重定义——不再用 rubric terminal/disconfirmed 当判据）**：
   - 旧（错）判据 = `deviate + disconfirmed + rubric terminal/blocked`。**问题**：rubric 到 terminal 恰恰是最该让医生知识介入的地方；同一结构位置，confirmed=patch(松)、disconfirmed=over-imaging(刹)，用 outcome 区分=依赖单次结果，且和 C1/C2 自相矛盾，agent 决策时也拿不到结果。
   - 新（对）判据 = **`alarm 静默 且 belief 集中（低 entropy / 低 other_mass）且 同模态重复` → over-imaging（刹）**。反之 **`alarm 触发 或 belief 弥散` → terminal 过早 = rubric limitation → override 是有价值的 deviation（松），无论结果**。即用 alarm+belief 两通道（A2/A3）在**事前**区分"真终点"vs"rubric 局限"，把 outcome 彻底踢出 action-quality。
3. **跨步不更新（轨迹级）**：单次 disconfirm 无意义；但连续多次 expected X / 结果 not X，医生却不更新 belief、不改策略 = 不从 prediction-error 学习。annotation 是离线全序列 → 可离线预计算该步权重（SFT 可用），但天然更贴 RL。
4. **客观锚点排除徒劳检查（FD-2）**：给已切除/已处理器官成像 → 用 procedure history + charttime（确定性事实，非 LLM 判断）认定徒劳。需真 charttime，目前 deferred；数据到了当硬过滤或降权。

**SFT vs RL 分工**：SFT 能做的极限 = 对**数据里已存在的步**改 `sample_weight`/重标 stop（判据1/2 最适合，3 可离线预算，4 待数据）。**只有 RL 能做** = 采样医生没做过的反事实动作并打分（"没停但停更好"）+ 跨步 credit assignment。判据4 的 timing、判据2 的 alarm 目前都是 provisional，待 clarify / charttime。

### 2.5.1 两通道分工 + 决策表（alarm = override 开关；belief = 默认路径的 continue/stop 开关）

两通道**正交**：**alarm 只管"要不要 override 默认路径"；belief 管"默认是 follow-rubric 的下一步 还是 stop"。** 所以 **"无 alarm" 不对应固定动作**，它=「别 override，回默认」；默认走还是停由 belief 的 gap 开/闭决定：

| | override（偏离 / 加检查 / 重复） | defer 回默认（follow rubric / stop） |
|---|---|---|
| **有 alarm** | ✅ 最高学习价值（动作精准裁决了矛盾=patch，需 `addresses_alarm=yes`）/ ⚠️ 选错检查则负 | ⚠️ 漏掉该处理的 alarm（`unaddressed_alarm=true`） |
| **无 alarm** | ❌ 过度成像 → 降权（§2.5 判据2） | ✅ 正确：gap 开→follow rubric 下一步，gap 闭（诊断+severity 已清）→stop（由 belief 定） |

一句话：**没 alarm 不是"该 follow"也不是"该 stop"，而是"不该 override"；follow vs stop 归 belief 管，不归 alarm 管。** §2-A（松）= alarm 响时奖励果断 override；§2-B（刹）= alarm 静时奖励克制。

### 2.5.2 ALARM 标注 pass（已实现：两步两调用，平行于 verification，但都 ex-ante）

⚠️ **A1 不在 LLM pass 里（2026-06 再修订，且已落代码）**：A1(advanced_question) 不是客观红旗、没有"独立于医生"的盲看版——它就是 belief 通道的属性（dx 已立 + severity/etiology gap），盲看 LLM 去再判一次"dx clear"会和 belief 打架（实测 5/12 误判，全发生在 belief 弥散处）。且 A1 的"解决质量"已被 Mode-A 的 `appropriateness` 覆盖。**故 A1 改为 compiler 算法派生**（`belief 集中在某 rubric 病 max_p≥τ` ∧ `gap/action_role ∈ severity/etiology/complication`；resolution=该步 appropriateness），`'other'` 占优的步不算 A1。**LLM pass 只剩 A2/A3 两类客观红旗**。两步**分两次调 API**，理由是**独立性**：

- **step 1 `alarm_detect`（盲看材料，不给医生 order/reasoning）**：只读因果遮蔽后的 baseline + 已出 prior imaging，独立判断有无两类——`study_inadequacy`(A2：前片技术受限/未显影，问题仍开着，≠阴性)、`discordance`(A3：两路证据互相矛盾无法调和)。盲看靠**不传 order 进 input** 强制（prompt 不啰嗦提醒），避免被医生 reasoning 循环污染。
- **step 2 `alarm_resolve`（给 step1 两类 + 医生 ordered + 重建 reasoning）**：同时权衡两轴——**Capability**（test 技术上能否解决）+ **Intent**（reasoning 是否真在追这个问题）。**不要求显式点名**，aim 覆盖即可；test 机械能解决但 reasoning 奔着别处、只顺带覆盖 = `partial` 而非 `yes`（= C1 aimed-vs-incidental）。**ex-ante，绝不看结果**。
- **API 门控（省钱）**：step1 后算法判有无 trigger；**无 trigger 跳过 step2**（`addresses_alarm=not_applicable, api_skipped=true`）。pass = step1 API + 门控 + 条件 step2 API。
- **健壮性（2026-06 修复）**：step1 用 `max_tokens=1000`+一次重试（长 grounded 证据 + prior 报告会在 500 处**截断**→JSON 不可解析；旧版把截断静默当"无 trigger"会污染数据）；解析失败**显式记 `_parse_error`+raw**，runner `_has_error()` 在 re-run 时**自愈重跑**该 case，绝不当 fake-clean。
- **过度成像（无 trigger + 偏离/重复）不在此 pass**：确定性的（belief 集中度 + 偏离算法 + 同模态计数），留给 compiler。

**代码**：`experiments/annotation/prompts.py`（`ALARM_DETECT_SYSTEM`/`build_alarm_detect_user`、`ALARM_RESOLVE_SYSTEM`/`build_alarm_resolve_user`）、`experiments/annotation/alarm.py`（`annotate_alarm_case`，复用 `annotate.call_json` + `build_masked_view.build_record`）、runner `scripts/run_alarm_pass.py`（resume-safe + workers，离线跑在已有 `full/*.json` 上）→ `results/annotation_experiment/alarm/<disease>_<hadm>.json`，每步 `{alarm_detect, alarm_resolve}`，temp=0。

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
  - verification（confirmed/disconfirmed/uninformative）+ certainty_update(±/0)。
  - appropriateness（yes/partial/no）。
  - 按 §2-C 的约束使用（disconfirmed≠惩罚）。

**防泄漏（硬规则，沿用 annotation pipeline）**：Discharge Dx / ICD / Procedures 永不进任何样本；disease/deviation 上帝视角标签不进 INPUT（只当 eval/label）；verification 不进 INPUT。

**样本量现状**：~338 deviation events + 匹配的 follow/non-deviation 对照（§1.5 deviation 是 label 不是 filter，必须含负类/follow 步，否则学不出边界）。430 judged steps 里 follow 181 / deviate 169 / off_rubric 80，可直接当四类决策样本。

---

## 4. Certainty 信号形式化（共用层；目前 memory 标为"待做"）

- 每步增量 = verification 的 confirmed(+) / disconfirmed(−) / uninformative(0)（annotation_agent_design §1.2(b)）。
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
2. **Certainty 形式化**（§4）：verification→轨迹，落 certainty 列。
3. **方向 A 先行（便宜快）**：prompt = rubric 图 + top-k exemplar + §2 规则；在 held-out 上跑 §5。这步同时验证行为规范是否可学。
4. **方向 B**：把样本编译成 SFT 格式，LoRA 微调 qwen；同一 held-out 对比方向 A。
5. （可选 RL）reward = local imitation + prediction-error；offline RL 有 distribution-shift 风险，非必须（annotation_agent_design §5）。

---

## 7. 开放问题 / 风险

- 样本量对方向 B 偏小（4 病、~338 events）→ 优先 LoRA/few-shot，警惕过拟合；可能需把 follow/off_rubric 步也充分纳入扩容。
- exemplar 检索（方向 A）的相似度定义 = 上限，复用现有 KNN engine 但要按"当前临床问题"而非纯特征相似。
- RR-N 仍是时序 proxy（charttime rigor TODO 挂着）；51/293 patient rrn_aligned=False 被跳过，扩样前考虑补 charttime。
- biliary 是 post-hoc 派生 belief，不是重建的 disease（不会被 over-attribute），但其 verification 质量较杂（~56% confirmed）——当训练材料时按 §2-C 谨慎用。
- §2 的行为规范源自 4 病的 542 步，外推到新病种前需重标。

---

## 8. Step 级数据集编译器（milestone 1 的具体落地）

> 用户决策（2026-06，已锁）：① 现在就编译（用 provisional timing filter）② 混合 target（模仿为主 + should-suppress 步重标/降权）③ 双通道 certainty（belief + alarm）④ 一路做到 RL。

**它是什么**：一个**打包脚本**（`scripts/compile_training_set.py`，待写），不产生新信息。把已有 annotation（`results/annotation_experiment/full/*.json` 每步的 ex-ante 重建）+ `belief_deviation_filtered.csv`（标签 + timing 过滤）**重塑成一行行训练样本**，每行 = 一个保留的决策步，固定三段 `INPUT → TARGET → REWARD`。542 步 → 506 行（timing 过滤后）。

**when / how / why 已经显式写在 annotation 里**，编译器只做分流：

| 要学的 | 来自 annotation 的字段 | 在样本里的角色 |
|---|---|---|
| **WHEN**（何时偏离/触发） | `differential`(belief 形状)、`information_gap`、`reasoning` 里的 discordance/adequacy 语言、`certainty_update` 序列 | → 算 **alarm 通道** + `dev_belief`(follow/deviate/off_rubric) 当决策 target |
| **HOW**（怎么偏离） | `ordered`(实际动作)、`action_role`(localize/stage/etiology…)、modality | → **动作 target**：follow/deviate/stop + 具体 modality |
| **WHY**（为何偏离） | `information_gap`、`expected_finding`、`grounding`(引用具体特征)、`appropriateness_reason` | → **reasoning trace target**（agent 要生成的 NL 理由 + 引用） |

**编译器的 5 个转换（非简单拷贝）：**
1. **因果遮蔽**：INPUT 只含该步之前可见的影像结果（复用 `scripts/build_masked_view.py`），后续结果 + verification 挡掉。
2. **挂 rubric 库（全 4 + other，不是单个子图）**：⚠️ **2026-06 修订，下方单子图示例已被取代。** 早先方案是按 `top_branch` 取单个 sub-rubric 进 INPUT——但 belief argmax 是 agent 这一步要**输出**的 target（在 `why_trace.belief` 里），单独喂它的 sub-rubric = **泄漏 belief target** 且与推理不符（`eff_branch` 并非 ground truth，`top_branch==disease` 仅 59%，但仍是 this-step target，照样泄漏）。**现在 INPUT 每步给全 4 疾病 sub-rubric + open 'other'**（`rubric_library.json` 单独写一份，每行用 `INPUT.rubric_library_ref` 引用，jsonl 从 ~13MB 降到 ~3.5MB），再加 `INPUT.active_path` = **上一步的 belief argmax**（当前工作假设；过去信息，非泄漏；teacher-forcing 的循环输入，推理时换 agent 自己上一步输出；step1=None）。agent **自己 self-route**：输出 differential → argmax 即隐式选定 rubric。TARGET 保留 `belief_branch`(=top_branch，agent 要输出的 4+other argmax) + `effective_branch`(=eff_branch，deviate 实际对照的 rubric，可经 post-hoc 变 'biliary')。**deviate 是确定性算法不是学出来的**：argmax(agent belief) → traverse 该疾病图于因果遮蔽 features(idx_{k-1}) → rubric 此处想要的影像 ∩ imaging → 与 agent 的 ordered modality 比 → in=follow/否=deviate/argmax='other'=off_rubric（= `belief_step_deviation`，eval 用 agent belief、train 用重建 belief）。
3. **派生双通道 certainty**：`belief` 从 differential 直接拿（max_p/entropy/other_mass）；`alarm` 从 information_gap/reasoning 的 discordance/adequacy/conflict 信号 + 前序步 disconfirm 计数算。⚠️ 当前是 **provisional 关键词版**（`method:provisional_keyword`）= 待 clarify（纯规则关键词 vs 轻量 LLM 标一列）。
4. **混合 target 重标**：默认 target=医生动作；should-suppress 步默认 `sample_weight=0.3` 降权（`--suppress-mode stop` 改成 `when_action→stop`，`off` 纯模仿）。⚠️ **代码现状仍是旧 DD-1 proxy（`deviate+disconfirmed+terminal/blocked`），已被 §2.5 判据2 判定为错（用了单次 outcome + 和 C1/C2 矛盾）——TODO 改成 `alarm 静默 且 belief 集中 且 同模态重复`（事前、不看结果）。** 其余判据见 §2.5：判据1（appropriateness="no"）可同样 SFT 降权；FD-2 陈旧解剖需真 charttime → 暂缓。
5. **防泄漏剥离**：Discharge Dx / ICD / Procedures / disease 上帝视角 / verification 一律不进 INPUT。

**实现状态（2026-06）：`scripts/compile_training_set.py` 已写并跑通** → `data/training_set/{train_steps.jsonl(404 行), train_manifest.json, rubric_library.json}`。行 = `rrn_aligned & not excluded_monitoring`（542→404）。验证：INPUT 内 0 个序列化疾病图（belief 不泄漏）、0 个 this-step 结果；step1 的 231 行 active_path 全 None。

**一条真实样本（`20279299` step2，deviate+confirmed）—— ⚠️ `rubric_subgraph` 单子图字段已被转换2修订取代，现为 `rubric_library_ref`(全4+other) + `active_path`；TARGET 增 `belief_branch`/`effective_branch`。下方保留以示 TARGET/REWARD 语义：**
```yaml
INPUT:                                  # 推理时可见
  patient_state: <baseline + 因果遮蔽后只含 step1 CT 结果>
  # 旧: rubric_subgraph(单 appendicitis 子图) —— 已改为下面两行（防 belief 泄漏）
  rubric_library_ref: rubric_library.json   # 全 4 疾病图 + open 'other'
  active_path: appendicitis                  # 上一步 belief argmax（过去信息）
TARGET:                                 # 监督信号（医生做了什么）
  belief_branch: appendicitis           # ← top_branch，agent 要输出的 4+other argmax
  effective_branch: appendicitis        # ← eff_branch，deviate 实际对照的 rubric（可 post-hoc=biliary）
  when_action: deviate                  # ← dev_belief（vs effective_branch 算）
  how_modality: CT_Abdomen              # ← ordered（rubric 本要 US）
  why_trace:
    belief: {appendicitis:0.75, other:0.15, ...}
    information_gap: "已知穿孔，但 WBC18 vs 腹软 discordant，需 interval 变化定手术/引流"
    expected_finding: "脓腔增大/新积液 or 稳定→保守"
    grounding: [WBC 18.2, lactate 2.1, "soft non-tender abdomen", 前次 5.7cm 脓腔]
    action_role: stage_complication
REWARD:                                 # 仅 RL/eval，绝不进 INPUT
  verification: confirmed
  appropriateness: partial              # eGFR 低，重复造影顾虑
  certainty_update: up
  sample_weight: 1.0
```
教 agent：**当** belief 已定但双流不一致(alarm↑) → **怎么** 弃 rubric 的 US 改 CT → **为何** 要分期并发症(US 看不了脓腔间隔)。= §2-A1/A3。

---

## 9. INPUT / TARGET / REWARD 到底怎么用于训练（概念澄清）

> 用户常见困惑：「目标是让 agent 输出贴近 target 吗？reward 是 measure f(x) 和 y 的差距吗？」

**核心：三者分属两种范式。INPUT/TARGET 是监督学习；REWARD 是 RL，不是 loss、不衡量和 y 的距离。**

**(a) SFT（监督学习 / behavior cloning）= INPUT + TARGET**
- 就是 x=INPUT, y=TARGET, agent=f。loss = **标准 cross-entropy**（逐 token 算 f(x) 与 y 差距），**现成的、不用自己设计**；你只要把数据做成 (x,y) 对。
- 产出：会模仿医生动作+推理的 warm-start policy。

**(b) RL = INPUT + REWARD（不用 TARGET）**
- reward **不是 loss，不衡量和 y 的距离**。它是给"一个动作"打的**独立质量分**。
- 流程：agent 自己采样动作 `a ~ f(INPUT)`（不看 y）→ reward `R(a)` 打分 → 调权重让高分动作更可能输出。
- reward **可以和 target 唱反调** —— 这是关键。

**(c) 为什么必须 reward（与 target 的根本区别）**：医生动作(TARGET)不总是对的，纯 SFT 会把过度成像也学进去。reward 让你说"动作好/不好"哪怕它就是医生做的：

| 真实步 | TARGET（模仿医生） | REWARD（动作质量） | 后果 |
|---|---|---|---|
| `27993727 s2` 术后冗余 CT (DD-1) | "做 CT" | **低**(disconfirmed+inappropriate+suppress) | 纯 SFT 会学冗余 CT；reward 让 RL 不强化 |
| `25444703` 偏离撞中肿物 (DD-3) | "做 CT" | **高**(循证+瞄准对器官) | reward 若错设成"是否命中预测"会误罚；正确设计要奖 |

→ reward 若是 f(x) vs y 的距离，两个都会"贴近医生=高分"，就永远学不会刹车。reward 是从 ex-post 字段(verification/appropriateness/§2-C)**独立判出的好坏**。

**(d) reward 要自己 design 吗**：要，但 design 的不是"和 y 的距离"，而是"什么动作算好"。草稿就是 §2 行为规范，例：
`R = +1 if deviate且confirmed且appropriate ; −1 if DD-1过度复扫 ; DD-3撒网命中真相不罚(C1) ; certainty更新只作用 belief 不作用决策分(C2)`。

**(e) 三阶段串联（已选一路到 RL）**：① SFT/behavior-clone 用 (INPUT,TARGET)，cross-entropy → warm-start。② 混合重标：SFT 阶段就把 should-suppress 步 target→stop / 降 weight（把少量 reward 信息提前掺进监督）。③ offline RL：在 warm-start 上用 REWARD 采样-打分-更新，纠正过度成像；reward=local imitation+prediction-error，**绝不用 global outcome**；防 distribution-shift 用 CQL/IQL/AWR。

**一句话**：TARGET=「医生做了什么」(监督，现成 loss)；REWARD=「这动作好不好」(自设质量分，RL 用，能和医生唱反调)。项目核心——循证刹车——只能靠 reward。
