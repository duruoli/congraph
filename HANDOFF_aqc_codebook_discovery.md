# A/Q/C development annotation handoff

## 2026-09-02 latest checkpoint: bridge 006 fully adjudicated and quality-cleared

> **新对话从本节开始；本节覆盖下方 bridge 006 pending-authorization checkpoint。**
> final test 临床内容仍未读取。主 annotation prompt 未修改，SHA-256 仍为
> `697923b99721c21edd474848a816423fab20d3a65c1e0388c938dbf24a72d5c1`；validator 为 `3.1.0`。

- bridge 006 冻结 manifest、causal-mask review、主生成和 manifest-scoped invalid-step repairs 的细节见下一节。
  主 annotation 共 73 calls，记录费用 `$1.896766`。
- 经第二次明确授权，7 个 dry-run targeted payload 仅发送至 OpenRouter `openai/gpt-5.1`，用于局部 A/Q/C
  temporal/discordance repair：7 calls / 7 request IDs / 7 usage objects，18,233 prompt tokens、1,535
  completion tokens，记录费用 `$0.038141`。2 个 step 产生通过门禁的修补：
  `cholecystitis:25217286:s2` 的伪 discordance 改为 `not_applicable`；
  `cholecystitis:24636219:s2` 增加 temporal requirement 及对应 coverage。其余 5 个保留原判断。
- 主 annotation + targeted 的记录总费用为 `$1.934907`；OpenRouter billing 仍为权威值。
- 剩余 12 个 algorithmic `nonverbatim_unresolved` 已逐项人工裁决：合并 lab 行缩减为逐字 source line，
  absence/technique summaries 删除，拼错的 lab name 修正，非连续的院外 CT 摘要恢复为完整逐字句；
  `appendicitis:25547534:s2` 将“假设 vs 阴性检查”伪 discordance 改为 `not_applicable`。
- 最终 non-destructive overlay：
  `results/aqc_direct/development/697923b99721/manual_adjudication_bridge_006.json`，共覆盖 20 steps；原始模型
  输出未覆盖。最终 algorithmic 重扫：31 steps、0 issues、0 unresolved nonverbatim、0 remaining operations、
  0 invalid。最终 batch audit：20/20 patients、31/31 steps、0 validator invalid、0 low-evidence-fidelity items。
- 唯一 exact repeat `appendicitis:22881737:s2` 已人工核对：它是实际第二次 CT，报告与第一次不同，不是重复
  数据。repeat 本身不足以强制 temporal requirement；由于可见材料没有明确说明当前 Q 是 interval response，
  保留原 Q，不作猜测性改写。
- `scripts/aqc_algorithmic_auditor.py` 的空-assumption-evidence 修补已回归 bridge 004 makeup、005、006，三批
  normalization 后均为 0 invalid。
- Development 当前共完成 84 位互不重复患者；尚余 51 位。继续排除 final test。

### 下一步

1. 若继续 annotation，按同一规则冻结下一批全新 development manifest，排除已完成 84 位患者与 final test。
2. 先 causal-mask preflight + 人工裁决；外发前对新冻结批次、OpenRouter、`openai/gpt-5.1`、A/Q/C 用途和
   cost-stop 策略重新取得明确授权。
3. 生成后继续 validator → algorithmic auditor → targeted dry-run；targeted 的实际候选 payload 另行授权。

## 2026-09-02 latest checkpoint: bridge 006 generated; targeted payload awaiting authorization

> **新对话从本节开始；本节覆盖下方 checkpoint 的“下一步”。**
> final test 临床内容仍未读取。主 annotation prompt 未修改，SHA-256 仍为
> `697923b99721c21edd474848a816423fab20d3a65c1e0388c938dbf24a72d5c1`；validator 为 `3.1.0`。

- 冻结 manifest：`data/aqc_direct/bridge_697923b99721_006.json`；SHA-256
  `10698BD5A2F362DBF3557B23139AF32809FE4DFAD490635F437EFC319FEDEAEC`。共 20 位全新 development
  患者 / 31 steps：appendicitis 4、cholecystitis 8、pancreatitis 8。
- causal-mask preflight 命中 3 个候选：`cholecystitis:22023307:s2` 确认为当前 CT 结果泄漏并做精确、
  hash-bound 的内存 redaction；另 2 个确认为院外既往检查。review 位于
  `data/aqc_direct/bridge_697923b99721_006_leakage_review.json`；复验 `blocking=false`。
- 经明确授权，冻结批次仅发送至 OpenRouter `openai/gpt-5.1`，用于 A/Q/C development annotation，采用
  `--no-cost-stop`。普通生成后 5 位患者含无效 step，第一次 manifest-scoped repair 后仍余
  `cholecystitis:24308410:s1` 不可解析；第二次仅修复该 step 后，31/31 均通过 validator 3.1.0。
- 共保留 73 attempts / 73 request IDs / 73 usage objects，285,677 prompt tokens、170,239 completion tokens，
  记录累计费用 `$1.896766`；OpenRouter billing 仍为权威值。
- batch audit：20/20 patients、31/31 steps、0 current-validator invalid、0 low-evidence-fidelity items；有 1 个
  exact repeat：`appendicitis:22881737:s2`。人工检查确认它是实际第二次 CT（报告与第一次不同），不是重复
  数据文件；是否应把 Q 改写为 temporal follow-up 仍属于语义判断，不作机械改写。
- algorithmic auditor 初次在 `appendicitis:22881737:s1` 删除唯一 scaffolding evidence 后留下空 evidence，
  自己制造 validator 错误。`scripts/aqc_algorithmic_auditor.py` 已修正：若 assumptions 超过最少 3 项，整体
  删除失去全部合法 evidence 的 assumption；否则回退该操作并留给 adjudication。bridge 004 makeup、005、
  006 回归均为 normalization 后 0 invalid。bridge 006 当前为 42 issues / 18 steps、29 个确定性 operations、
  12 个 unresolved nonverbatim；proposed overlay 尚未合并为最终 adjudication。
- targeted auditor dry-run 形成 7-step 实际 payload，尚未外发：2 temporal
  (`cholecystitis:24636219:s2`, `pancreatitis:21775506:s1`)；5 discordance
  (`cholecystitis:25217286:s2`, `pancreatitis:28920003:s2`, `appendicitis:25547534:s2`,
  `cholecystitis:22023307:s2`, `pancreatitis:24197495:s2`)。prompt hashes 保持 temporal
  `c3a7693366a61fece4ad14837a69565d5fa2a660ea67901e1ab06f0ad9b3a66b`、discordance
  `3b9c140b3b828e1c5a04fd7a98ddddd1a30fc3b0182e49c11888f61ec016e88d`。

### 下一步

1. 先取得上述 7-step targeted payload 发送至 OpenRouter `openai/gpt-5.1`、用于局部 temporal/discordance
   A/Q/C repair 的明确授权；此前只保留 dry-run。
2. 获授权后运行 targeted `--execute`，然后重跑 algorithmic auditor + validator；若 targeted correction
   引入新 nonverbatim evidence 或硬错误则拒绝该 correction。
3. 人工裁决剩余 12 个 nonverbatim evidence 和 exact-repeat 的 temporal 语义；只将通过复验的 overlay
   提升为最终 adjudication，原始模型输出不得覆盖。

## 2026-09-02 latest checkpoint: post-generation quality gate implemented through targeted-auditor dry-run

> **新对话从本节开始；本节覆盖下方 checkpoint 中关于 active validator 和下一步流程的旧说明。**
> `HANDOFF_annotation_pipeline.md` 是第一轮 annotation 的 archive，不再记录当前 A/Q/C pipeline 进度。
> 继续禁止读取 final test 临床内容。主 annotation prompt 未修改，活动 prompt SHA-256 仍为
> `697923b99721c21edd474848a816423fab20d3a65c1e0388c938dbf24a72d5c1`。

当前完整控制流：

```text
causal-mask preflight（生成前）
  → LLM 完整 step annotation
  → validator（结构与跨字段硬规则）
  → algorithmic auditor（clinical-only evidence 检查 + 确定性机械修补）
  → targeted LLM auditor（temporal / discordance 局部语义修补）
  → algorithmic auditor + validator 复验
  → batch auditor（跨 step / 批次统计）
```

### 三层职责边界

- `validator` 不只是筛选给 LLM auditor。它逐 step 检查 source-independent 的结构/逻辑硬规则，并在
  修补后作为最终门禁；不判断复杂临床语义，也不直接改写输出。
- `algorithmic auditor` 同时筛选和修改，但只自动修复答案唯一明确的机械错误，例如 evidence 中的
  scaffolding、`(none)`、空项、首步 previous fields，以及确定性的 coverage 状态矛盾。无法唯一恢复的
  evidence 只 flag，不猜测。
- `targeted LLM auditor` 只处理算法不能可靠判断的局部临床语义问题，目前为 temporal 与 discordance；
  只重判对应子块，不重新生成整个 step。

因此不是“validator 将所有问题交给 LLM”，而是 validator 定位硬错误，algorithmic auditor 确定性修补
并产生语义候选，targeted LLM auditor 局部修补，最后再由 algorithmic auditor + validator 复验。
HPI leakage preflight 是生成前门禁，与上述生成后三层独立；主 annotation prompt 不因这些门禁而修改。

### Validator 3.1.0

- Active implementation：`scripts/aqc_validator.py`；DIRECT runner、framework check 和 batch audit 共用。
- 规则按 assumptions、question/requirements、coverage、previous-order update/discordance structure、
  sequence state、current-order fit 分区，只纳入确定性规则。
- 新硬规则包括 coverage status/direction/aggregate 一致性、first-step previous fields 必须为
  `not_applicable`，以及 `materially_discordant/indeterminate` 必须有两条非空 evidence streams 和 reason。
- 旧 repeat-order temporal marker 启发式已退出硬 validator。只有 Q 真正在问相对 earlier study/treatment
  state 的改善、恶化、进展、稳定或 response 时，才需要 `temporal_course_or_response`；repeat order 本身
  不充分。该语义对应关系交给 targeted temporal auditor。
- bridge 005 旧输出回归命中 6 个机械错误；makeup 5 步无 validator 3.1.0 硬错误。

### Algorithmic auditor 1.0.0

- Implementation：`scripts/aqc_algorithmic_auditor.py`。
- 合法 evidence source 仅为 filtered HPI、PE、labs 和当前决策前已出结果 imaging；排除 current order、
  prior A/Q/C、section headers、`(none)` 和 output template。
- 检查 assumptions、current question、coverage supporting evidence 与 discordance evidence streams。
  只有可以唯一决定的修复才写入 non-destructive proposed overlay；原 result JSON 不覆盖。
- Evidence matcher 允许一条 evidence 合并或重排多个逐字 source sentences，但每个分句都必须在合法
  clinical source 中逐字存在；语义相似 paraphrase 不会自动放行。
- bridge 005：49 issues / 22 steps；43 个确定性 operations；6 个 unresolved nonverbatim；normalization
  后 validator 3.1.0 为 0 invalid。
- bridge 004 makeup：1 issue / 1 step；0 个确定性 operations；剩余 1 个 nonverbatim absence summary；
  normalization 后 validator 3.1.0 为 0 invalid。
- 稳定产物位于 `results/aqc_direct/development/697923b99721/`：
  `algorithmic_audit_bridge_*.json`、`algorithmic_proposed_overlay_bridge_*.json`。`proposed` 表示尚未合并
  为最终 adjudication overlay。

### Targeted LLM auditor 1.0.0（仅 dry-run，尚未外发）

- Prompts：`experiments/aqc/targeted_repair_prompts.py`；orchestrator：
  `scripts/aqc_targeted_auditor.py`。
- Temporal prompt 区分影像/病变相对 prior state 的变化与仅描述 worsening symptoms/current status；只可
  返回 aligned、添加 temporal requirement + matching coverage、删除不当 temporal wording、或 unclear。
- Discordance prompt 仅在两条 evidence streams 针对同一临床命题且方向相反时判冲突；明确排除 technique/
  adequacy 差异、nonvisualization→visualization、可共存 findings 和 limited reassurance。
- LLM 只能返回 typed local repair。合并后重跑 algorithmic auditor + validator；如引入新的 nonverbatim
  evidence 或硬错误，则拒绝 proposed correction。
- Prompt hashes：temporal
  `c3a7693366a61fece4ad14837a69565d5fa2a660ea67901e1ab06f0ad9b3a66b`；discordance
  `3b9c140b3b828e1c5a04fd7a98ddddd1a30fc3b0182e49c11888f61ec016e88d`。
- Dry-run candidates：bridge 005 为 1 temporal + 4 discordance；makeup 为 1 temporal
  (`pancreatitis:26486125:s1`)。未发生 targeted LLM call；运行 `--execute` 前需针对该 targeted payload
  取得明确外发授权。

### 新对话的下一步

1. 读取本节，重新计算并确认主 annotation prompt hash 与 validator `3.1.0`；若不一致先报告。
2. 为下一批全新 development 患者冻结 manifest；继续排除 final test 和所有已标注患者。
3. 对冻结批次先运行 causal-mask preflight 并人工裁决 blocking hits。
4. 在发送任何新临床文本前，取得该冻结批次、OpenRouter、目标模型、A/Q/C 用途及费用停止策略的明确授权。
5. 标注后依次运行 validator → algorithmic auditor → targeted auditor；targeted 外发另以实际候选 payload
   为界确认授权。保存原始输出、所有 attempts/usage/cost 和 non-destructive repair provenance。

## 2026-09-01 bridge batch 005 completed: causal-mask-gated 20-patient GPT-5.1 annotation

- Frozen manifest: `data/aqc_direct/bridge_697923b99721_005.json` (20 patients/32 steps; 7 appendicitis, 7 cholecystitis, 6 pancreatitis).
- Prompt SHA-256 remained `697923b99721c21edd474848a816423fab20d3a65c1e0388c938dbf24a72d5c1`; validator remained `3.0.0`; the annotation prompt was not modified.
- Preflight found one confirmed HPI/current-CT restatement (`pancreatitis:27929956:s1`). Exact reviewed redaction made the preflight non-blocking before external transmission.
- With explicit authorization, the filtered frozen batch was sent only to OpenRouter `openai/gpt-5.1` for A/Q/C annotation using `--no-cost-stop`. No final-test clinical content was read.
- Two steps exhausted their ordinary attempts without parsed JSON; manifest-scoped `--repair-invalid-steps` repaired only those steps and retained the failed attempts. Final audit: 20/20 patients, 32/32 steps valid, 0 low-evidence-fidelity alerts.
- One semantic wording mismatch at `pancreatitis:28226418:s2` was minimally corrected in `results/aqc_direct/development/697923b99721/manual_adjudication_bridge_005.json`; the original model output remains intact.
- Recorded cumulative batch cost after repair: `$1.493984`. OpenRouter billing is authoritative.
- All 32 steps generated exactly five assumptions: 122/160 `well_supported`, 38/160 `weakly_supported`. Treat this as persistent cap-filling behavior, not proof that every assumption was necessary.
- Development coverage is now 64 unique patients; 71 development patients remain unannotated.

## 2026-09-01 bridge batch 004 completed: frozen 12-patient GPT-5.1 annotation

> **Post-hoc causal-mask gate correction:** structure/content validation passed, but the later
> HPI-to-current-report preflight found 3 confirmed current-result leaks. This batch is therefore
> not quality-cleared until those trajectories are regenerated. Exact reviewed HPI redactions passed
> the integrated preflight, and the three trajectories have now been regenerated and re-audited. Detection is in
> `scripts/audit_aqc_input_leakage.py`; manual decisions are in
> `data/aqc_direct/bridge_697923b99721_004_leakage_review.json`. The annotation prompt was not changed.

> **Completed frozen work:** bridge 004 makeup contains 3 patients/5 valid steps. Bridge 005 contains
> 20 new patients/32 valid steps; its single algorithmic hit was manually confirmed and exact-redacted.
> Both were sent only after explicit authorization and pass causal-mask, validator, and filtered-input
> evidence audits. Bridge 005 has one non-destructive semantic correction overlay.

- The active prompt SHA-256 was re-derived as `697923b99721c21edd474848a816423fab20d3a65c1e0388c938dbf24a72d5c1`; validator remained `3.0.0` and retry protocol remained `1.0.0-validator-feedback`.
- The explicit frozen manifest is `data/aqc_direct/bridge_697923b99721_004.json`: 12 new development patients / 18 steps, with 4 appendicitis, 4 cholecystitis, and 4 pancreatitis patients. Manifest file SHA-256: `ED15D138E61A106A38A849FDFD07B9E29BEF5E93F90035A0D84F9F717AEEAEAC`.
- With explicit user authorization, the frozen batch was sent only to OpenRouter model `openai/gpt-5.1` for A/Q/C annotation, using `--no-cost-stop`. No final-test clinical content was read.
- The initial run left 2/18 steps structurally invalid after their ordinary attempts. A manifest-scoped `--repair-invalid-steps` run repaired only those steps, reused all valid outputs, and retained the superseded attempts.
- Final validator 3.0.0 scan: 12/12 patients present, 18/18 steps valid, 0 exact repeats, and 0 steps with `question_grounding != well_supported`.
- Evidence-fidelity screening flagged one step. Manual review found three non-verbatim absence summaries in `supporting_evidence`; the original model output remains unchanged and the correction overlay is `results/aqc_direct/development/697923b99721/manual_adjudication_bridge_004.json`. Re-audit with the overlay has 0 evidence-fidelity alerts.
- Audit trail retained: 37 attempts, 37 request IDs, 37 usage objects, 154,255 prompt tokens, 86,122 completion tokens, and recorded cost `$0.962743`. OpenRouter billing remains authoritative.
- At the time this batch completed, development had 44 unique patients covered (the earlier 32 plus this frozen batch of 12). After bridge 005, the current total is 64 and 71 remain. Do not expand without a separately frozen manifest and new external-transmission authorization.

## 2026-09-01 final prompt checkpoint：新 schema 已冻结候选，下一对话开始 bridge 标注

> **下一对话从本节开始；本节覆盖下方较早的“下一任务”说明。** 不要重做 codebook discovery、framework comparison、模型选择或旧患者标注，不要读取 final test 临床内容。当前任务是使用下述新 prompt，对全新的 development 患者继续进行 A/Q/C 标注。

### 当前范围

- Development 共235位、433步；已完成132位互不重复患者，尚余103位。
- Final test 共58位、109步，仍未读取，继续严格排除。
- 最近完成的32位/62步历史批次及人工 overlay 保持原样；其模型输出不覆盖，仍按旧 schema 和 validator `2.2.0` 解释。

### 本轮 prompt/schema 决策记录

- 活动 prompt 已改写成一个独立、自包含的 clinical reasoning annotation 任务，不再向模型提及 DIRECT/RECODE、research arm 或旧 schema-free reconstruction。角色是 clinical reasoning annotator；主任务始终是综合医嘱前病历、既往已出结果影像、前一 A/Q/C 状态和当前医嘱，重建 Assumptions、Question、Coverage 及其轨迹。
- `output_contract()` 是在生成前随每个病例一起提供的 JSON 输出模板，不是生成后才做分类，也不是 filled clinical example。模型替换模板说明为病例特异标注；生成后 validator 再检查枚举、字段结构、question/coverage requirement 对齐和轨迹规则。
- 医嘱是推断 Q 的重要线索，但不证明怀疑的疾病为真。Q 表示医生希望解决的核心不确定性；Coverage 表示当前医嘱前的证据已经回答 Q 的多少。二者不得混淆。
- 为保持顶层 schema 连续性，保留 `current_order_fit`，但它只是一个轻量的两轴复核包，不是 appropriateness：

```json
"current_order_fit": {
  "question_grounding": "well_supported | weakly_supported | unclear",
  "test_question_capability": "capable | partially_capable | not_capable | uncertain"
}
```

  - `question_grounding`：Record → Q；可见病历在多大程度上支持“这是医生关心的问题”。它不判断 Q 的阳性答案是否成立。
  - `test_question_capability`：Test → Q；当前检查是否有能力回答 Q。
  - 删除旧的 `intent_support`、`why_this_order_could_answer`、`unsupported_residual` 和额外 gap；不标注 normative appropriateness。
- evidence/supporting-evidence/evidence-stream 字段只放可见临床原文；概括、推论和由缺失得出的判断放 explanation/reason。
- 重复检查只有在 Q 真正询问较前变化时才加入 `temporal_course_or_response`；不能仅因检查重复而机械添加。旧影像可提供 baseline，但不能单独回答其后的变化。
- 旧 `intent_support` 语义曾发生漂移，不能在分析时无条件重命名为新 `question_grounding`。跨版本提取必须保留 prompt/schema/validator 版本；旧批次不作静默迁移。

### 冻结候选版本与本地验证

- 活动 system prompt：`experiments/aqc/prompts.py::ANNOTATION_SYSTEM`
- 输出 wrapper：`2.0.0-development`
- Validator：`3.0.0`
- Retry protocol：`1.0.0-validator-feedback`
- GPT-5.1 prompt SHA-256：`697923b99721c21edd474848a816423fab20d3a65c1e0388c938dbf24a72d5c1`
- Python 编译与新 schema 的内存转换验证通过。
- Development dry-run 仍选出12位全新患者、18步：appendicitis 4位、cholecystitis 4位、pancreatitis 4位；diverticulitis 的未标注 development 候选已耗尽。
- 尚未使用该 hash 调用 OpenRouter，也没有产生新标注。

### 下一对话的具体任务

1. 先读取本节并确认活动 hash 仍为 `697923b99721c21edd474848a816423fab20d3a65c1e0388c938dbf24a72d5c1`、validator 仍为 `3.0.0`；如 hash 不同，先报告而不要执行。
2. 将 dry-run 的12位/18步新患者冻结为显式 patient manifest，确保网络中断后恢复时名单不漂移；继续排除 final test 和所有已标注患者。
3. 在实际向 OpenRouter 发送这批临床文本前，取得用户对该冻结名单、OpenRouter、`openai/gpt-5.1`、A/Q/C annotation 用途和 `--no-cost-stop` 的本批明确授权。授权前只做本地操作。
4. 获授权后使用 `--patient-manifest` 逐患者落盘、可恢复执行；只对结构无效步骤做定向 repair，保留所有 attempts、request id、usage 和 cost。
5. 完成后用 validator `3.0.0` 全扫18步，并人工复核所有 repeat、`question_grounding != well_supported`、evidence-fidelity alerts；其余步骤抽查。若 bridge 通过，再决定是否提交剩余 development 患者的 Quest/SLURM 批次。

## 2026-09-01 later checkpoint：132位已完成，repeat 规则进入 validator 2.2 bridge

- 当前共完成132位互不重复的 development 患者；尚余103位。final test 58位、109步仍未读取。
- 扩大批次 `data/aqc_direct/batch_c7f7ffae_003.json` 完成32位、62步；一次网络中断后用新增的 `--patient-manifest` 精确恢复原获授权名单，没有动态扩展患者范围。记录费用 `$2.082012`，最终 validator 2.1 为62/62有效。
- 批次审计发现 exact repeat、证据字段混入概括和 order-driven over-rationalization 边界。13步人工纠正位于 `results/aqc_direct/development/c7f7ffae2271/manual_adjudication_batch_003.json`；应用后 validator 2.2 为62/62有效，低证据忠实度候选为0，原模型输出不覆盖。
- 历史中间状态（已被顶部 final prompt checkpoint 覆盖）：曾得到 prompt hash `ca9e5be6060aa40099adb947b8be59aa817e12039eac0d4ec020e78beb4e306d`，但随后仍有本地文字精简，因此该 hash 从未执行且不再使用。
- 当时的 bridge dry run 已选12位、18步：appendicitis、cholecystitis、pancreatitis 各4位；diverticulitis 的 development 候选已全部标注。实际执行必须使用顶部记录的最终候选 hash、冻结 manifest，并重新取得本批外部发送授权。

## 2026-09-01 continuation checkpoint：新版 bridge 通过，可扩大批次

- Development 仍为235位、433步；final test 58位、109步未读取。
- 当前共完成100位互不重复的 development 患者：旧 prompt 76位，加两轮新患者 bridge 各12位。
- 第一轮 bridge 使用 prompt hash `d9f0c01a505663454371289ea7f45995b2ea8f897a72053d8e90ad1bd738deb9`，12位、18步、18/18最终结构有效，记录费用 `$0.518180`。人工审计发现 `appendicitis:29310170:s1` 把当前医嘱名称作为 question evidence 并过度锁定胆道目标；原模型输出保持不变，人工纠正 overlay 位于 `results/aqc_direct/development/d9f0c01a5056/manual_adjudication.json`。
- 因上述问题，prompt 增加“当前医嘱是决策上下文而非临床证据”的规则；validator 升至 `2.1.0`，禁止 literal current order 作为 question evidence。当前冻结候选 hash 为 `c7f7ffae2271dc9305d0473ae4509b1deab21ee2257341f87694dcc949796ce9`。
- 第二轮 bridge 使用该 hash，12位、22步。一次 repair-only 运行只修复了1个失败步骤；最终22/22步有效。含 superseded attempts 的记录费用为 `$0.762669`。语义审计确认未再出现依赖医嘱名称锁定具体 question 的错误；repeat、低支持度和复合 assumption 检查通过。审计见 `results/aqc_direct/development/c7f7ffae2271/bridge_batch_audit.md`。
- 尚余135位 development 患者。下一批 dry run 已选32位、62步，疾病分布为 appendicitis 9、cholecystitis 9、diverticulitis 5、pancreatitis 9；diverticulitis 较少是因为其未标注候选先耗尽。该批仍需单独取得对 OpenRouter、GPT-5.1、DIRECT 用途和无费用停止线的明确授权后才能加 `--execute`。
- 继续保持：不读取 final test；同一 prompt hash 下可恢复执行；全量 validator 复扫；只定向 repair 无效步骤；所有 repeat 和低支持度输出100%人工复核，其他步骤至少抽查20%。

> 2026-08-31 状态：development split、codebook discovery、两轮 saturation check、
> DIRECT/RECODE framework check 和 DIRECT 模型小样本 pilot 已完成。新对话不要重做 discovery、
> framework check 或模型海选，也不要打开 final test；当前任务是冻结 GPT-5.1
> DIRECT 工作流后，分批完成 development 标注。

## 2026-08-31 continuation checkpoint：DIRECT 中途 prompt 修订

> **下一对话从本节开始；本节覆盖下方较早的“下一任务”说明。** 不要重做 discovery、
> framework check、模型选择或旧患者标注，也不要读取 final test 临床内容。

### 当前完成范围

- Development 共235位患者、433步；final test 58位、109步仍未打开。
- GPT-5.1 DIRECT 已覆盖76位互不重复的 development 患者、142步，当前 validator 复扫为
  142/142步有效：
  - 旧 pilot：6位、12步；
  - 后续 development：70位、130步；
  - 结果均使用旧 prompt hash `26ee973ad4d741310c5cbf29682e15891b70d9fd7195e6fe774df54d88adb536`，
    位于 `results/aqc_direct/{pilot,development}/26ee973ad4d7/`。
- 尚未标注：159位 development 患者、291步。
- 上述 DIRECT 已记录费用合计约 `$4.102132`；网络中断时未返回 usage 的调用不一定包含在内，
  账户扣费以 OpenRouter 账单为准。
- 当前 validator 为 `2.0.0`；retry protocol 为 `1.0.0-validator-feedback`。运行器支持逐步修复、
  保留 superseded step、跨恢复累计费用，以及跨 prompt hash 按患者去重。

### 为什么修改 prompt

对旧版本结果进行5例定向人工抽查，并对候选问题进行扩展复核后，发现三项需要在继续扩批前修正：

1. 重复检查有时没有把“较前变化/新触发因素”纳入 A/Q/C 推理；
2. 当病历只支持较粗的检查目的时，模型可能仅凭医嘱名称重建过于具体的 question；
3. assumption 偶尔把已建立的临床事实与仍在推测的病因/机制/并发症合成一个高置信命题。

因此 `experiments/aqc/prompts.py` 的 DIRECT/COMMON 规则已整理为七个单一职责部分：
Assumptions、Question、Requirements、Coverage、Trajectory、Repeats、Order fit。关键修改是：

- 不同置信度的事实与推测必须拆为原子 propositions；通常1--3条限制主要针对不确定假设，
  只额外保留决策所需的 established facts，所有 propositions 总计仍不超过5条；
- evidence fidelity 优先于 question specificity；精确目标或触发因素无证据时，使用证据支持的
  较粗共同问题、降低 `intent_support`，并把歧义写入 `unsupported_residual`；不得只凭检查名称
  锁定具体目标；
- 重复当前检查或检查目标时，必须在 A/Q/C 中考虑较前变化与新触发因素，但不机械增加一个
  独立 `temporal_course_or_response` requirement。

最终新 prompt hash 是
`d9f0c01a505663454371289ea7f45995b2ea8f897a72053d8e90ad1bd738deb9`。
中间编辑 hash `c22b05201972...` 和 `f6069ad7b1a3...` 没有产生模型标注。输出 contract、因果遮蔽、
模型选择和 validator 枚举没有改变。

### 版本边界

- 旧 hash 的76位/142步仍是可审计的 development 探索材料，不删除、不覆盖、不静默改写。
- prompt 内容已经改变，旧 pilot 不能充当新 prompt 的正式 pilot，也不能把两个 hash 的输出
  当作同一无版本数据直接汇总频率。后续分析必须记录 prompt hash，并考虑分版本报告、敏感性
  检查或在冻结后形成统一处理方案。
- `scripts/run_aqc_direct.py --stratified-new` 已跨所有 prompt-hash 目录排除已完成患者，因此新
  hash 不会重新发送上述76位患者。
- final test 仍不可读取；此次修改完全来自 development 审计。

### 下一对话应从哪里开始

1. 阅读本 checkpoint、`experiments/aqc/prompts.py`、`scripts/run_aqc_direct.py` 和
   `scripts/run_aqc_framework_check.py`；运行本地编译和 dry run，确认 prompt hash 精确为
   `d9f0c01a505663454371289ea7f45995b2ea8f897a72053d8e90ad1bd738deb9`。
2. 不要立即标完剩余159位。先用新 prompt 选12位 bridge batch（四种疾病各3位），且不得重复
   已完成76位：

   ```powershell
   & '.\.venv\Scripts\python.exe' scripts/run_aqc_direct.py `
     --scope development --model openai/gpt-5.1 `
     --stratified-new 12 --retries 2 --no-cost-stop
   ```

3. dry run 后，在任何新临床文本发往 OpenRouter 前，必须重新取得用户对该12位批次、
   OpenRouter、GPT-5.1、DIRECT用途及不设费用停止线的明确授权；获得授权后才加 `--execute`。
4. bridge batch 完成后用 validator 2.0 全量复扫；只用
   `--repair-invalid-steps --repair-existing-only` 定向补失败步骤，不得重跑成功步骤。
5. 人工重点复核：所有 repeat；所有 `weakly_supported/unclear`；question 比病历证据更具体的候选；
   以及包含 established 事实和 speculative 病因的复合 assumption。确认三项 prompt 修正生效后，
   才扩大新版本批次。

用户已表示后续运行不需要费用停止线，但外部临床文本授权仍按新批次单独确认；不要把上一批授权
自动扩展到新的患者集合。

## 已完成状态

- 正式语料：293位患者、542个决策步。
- Development：235位患者、433步。
- 未触碰的 final test：58位患者、109步。另有1位曾在检查数据结构时意外暴露，已移入
  development，不得补回 final test。
- Codebook discovery：初始24条轨迹/83步，加两轮各12条轨迹/31步和23步；总计48条轨迹、
  137步。24条只是初始 open-coding 样本，不代表单独完成饱和验证。
- 当前正式 development artifacts 位于 `data/aqc_development/`；本地验证已通过。
- 独立 framework check：从未参与上述48条的 development 中抽取16位患者、28步；DIRECT
  和 RECODE 使用同一模型 `anthropic/claude-sonnet-4.6`、同一A/Q/C输出契约，仅输入不同。
- framework check 结果位于 `results/aqc_framework_check/`，汇总见
  `results/aqc_framework_check/summary.json`。

## DIRECT/RECODE framework check 的发现

原始机械比较结果：

- DIRECT结构有效27/28步；RECODE结构有效25/28步；
- assumption-type平均Jaccard 0.5736；
- question-type名称一致率0.6429；
- answer-requirement type平均Jaccard 0.6412；
- aggregate coverage一致率0.8214；
- 旧规则把20/28步送入人工复核。

逐步重新复核后，不应把这20步理解为20个真实临床冲突：

- 14步的临床含义实质一致，可由更合理的自动规则处理；
- 4步的检查目的相同，差异来自相邻类别边界，例如 existence/identity 与 source
  localization、complication 与 severity/extent、remedy 与 advance；
- 2步的差异会影响 coverage 或 test capability，DIRECT的判断更符合当前定义；
- 没有一步显示两臂对检查目的存在完全不同且无法调和的理解。

具体规则发现：

1. `answer_requirements[]` 中同一个type可以重复，只要具体信息对象不同；不能只凭type相同
   判为重复。
2. `alternative_source_discrimination` 是 answer-requirement type；作为核心question时应
   对应 `alternative_source`。
3. `temporal_course_or_response` 暂时保留为必要时出现的answer requirement，而不是新增
   核心question type。重复检查若不说明“与之前相比”就无法解释目的，应在问题正文、
   requirement和continuity中明确时间关系。
4. 核心question types描述“在问什么”，并非严格互斥；影像常同时回答存在、身份、来源、
   范围和并发症。自动复核不能把名称不一致直接当作语义冲突。
5. 真正需要升级人工复核的是：两臂关注完全不同的疾病/解剖目标；coverage结论不同；
   test capability结论不同；或差异会改变对检查在诊疗轨迹中作用的理解。
6. DIRECT偶尔比证据允许的程度更确定；不能预设DIRECT逐项都正确。RECODE适合发现这种
   过度推断，但不适合作为主要数据层，因为它只能重整旧annotation已经保留的信息。

当前方法学决定：**DIRECT作为主要经验标注层，RECODE作为敏感性分析和审计参照**。

## DIRECT 模型 pilot 与选择

从未参与48例 discovery 或16例 framework check的 development 中，用稳定hash选取
6位患者、12个决策步，覆盖四种疾病、单/多步轨迹、repeat、modality switch、
prior-study limitation、nonvisualization 和 post-intervention。样本见
`data/aqc_direct/pilot_manifest.json`，结果见
`results/aqc_direct/pilot/26ee973ad4d7/`；prompt hash 为
`26ee973ad4d741310c5cbf29682e15891b70d9fd7195e6fe774df54d88adb536`。

后续模型决策只需关注：

| 模型 | 有效步 | 调用 | 已记录费用 | 质量结论 |
|---|---:|---:|---:|---|
| `openai/gpt-5.1` | 12/12 | 12 | `$0.295309` | 更克制，能在重复检查中保留不确定性和真实 residual |
| `anthropic/claude-sonnet-4.6` | 12/12 | 12 | `$0.653970` | 覆盖更完整，但倾向写满结构、把每个医嘱都判为 `well_supported` |

人工逐步复核后，**选择 `openai/gpt-5.1` 作为 development DIRECT 主标注模型**。
Sonnet 4.6 不作全量双标；它仅用于高风险步的定向复核，例如 `weakly_supported`、
`unclear`、重试/验证失败、repeat、post-intervention、coverage 或 test-capability 冲突。

这一决定不是因为字段名称一致率更高，而是 GPT-5.1 更少用无依据的确定性去合理化
已观察到的检查。例如 pancreatitis 的第4步重复 CT，GPT-5.1 正确保留了“未见新的
恶化触发证据”的 residual，而 Sonnet 仍将 intent 判为 `well_supported`。

## 已记录费用

最终保存的56次framework-check调用共花费 `$2.976552`：

- DIRECT：28次，输入138,067 tokens，输出74,468 tokens，`$1.531221`；
- RECODE：28次，输入102,377 tokens，输出75,880 tokens，`$1.445331`。

这不包含最初防截断修正前、或网络中断时未留下usage对象的调用；账户总扣费应以
OpenRouter账单为准。

## 新对话的下一任务：冻结并分批完成 GPT-5.1 DIRECT development 标注

1. 阅读本文件顶部、`aqc_annotation_design.md`、三个正式 codebook/contract、
   `experiments/aqc/prompts.py`、`scripts/run_aqc_direct.py` 和上述 pilot 结果。不要重做模型比较。
2. 在不改变已保存 pilot 输出的前提下，补强 validator：
   - 首个决策步的 `question_continuity`、`assumption_change.label` 和
     `derived_transition` 必须为 `initial`；
   - 当问题要求新的时间点/变化时，旧影像不能单独使 coverage 成为
     `sufficiently_answered`；
   - `established`/`excluded` 必须有对应的强证据；
   - 区分“患者文件存在”与“整条轨迹的所有步都验证成功”。
3. 保持当前因果遮蔽：只给决策前病历、已出结果的旧影像、前一A/Q/C状态和
   当前医嘱；不得输入当前结果、后续事件、verification、deviation、ACR或疾病答案标签。
4. 运行本地验证和 dry run，记录最终 prompt hash、validator 版本、质量门槛和成本停止线。
   如果 prompt 内容改变，不得沿用旧hash的pilot输出作为新版正式标注。
5. 建议先用 GPT-5.1 跑一个可恢复的 development 批次，人工抽查20%--30%；上述高风险步
   100%复核，必要时才调用 Sonnet。通过门槛后再扩大批次，不要一次启动235人。
6. 每步保存 request id、所有 attempts、token usage、cost、model、prompt hash和validation结果；
   每批结束后汇总有效步、重试、失败、费用和人工复核队列。
7. pilot授权只覆盖已完成的6位患者模型比较。**在向 OpenRouter 发送任何新的
   development 临床文本前，必须获得用户对新批次、OpenRouter、GPT-5.1 和 DIRECT 用途的
   明确授权。**
8. final test 58人/109步仍未打开；在pattern、prompt、validator、模型和统计方案冻结前，
   不得读取其临床内容。

## 这些DIRECT标注能支持什么研究结论

DIRECT development标注可以用于：

- 发现候选A/Q/C轨迹pattern；
- 检查核心research ideas是否在经验材料中形成可观察、可重复的结构；
- 修改pattern定义、codebook、prompt和分析方法；
- 估计标注可靠性并形成预先规定的最终分析方案。

但它们属于**发现与方法开发材料**，不能同时作为核心想法的无偏最终验证。正式验证应在
pattern定义、prompt、模型、过滤规则和统计方案冻结后，使用仍未打开的58位final-test患者。
如果根据development结果反复修改研究想法，最终论文中应明确区分探索性发现与held-out
验证。

---

# 原始任务说明：建立第一版正式 A/Q/C 分类规则

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
