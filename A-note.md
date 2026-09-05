Annotation Chain:

A：医嘱前病历支持哪些临床假设？
          ↓
Q：结合病历和医嘱，医生最可能想问什么？
          ↓
R：回答这个问题需要哪些信息？
          ↓
C：医嘱前已有证据回答了多少？
          ↓
Order fit：
  这项检查技术上能否回答 Q？
  病历是否解释了为什么现在选择它？

## 重要数据发现：HPI 不等于严格的 pre-order snapshot

`Patient History` 虽然是 HPI/入院叙事，但并不天然早于所有 recorded radiology orders。它可能在
正式入院时回顾 ED 中已经完成的影像，因此包含我们原先以为位于后续的 current-test preliminary
或 final findings。

```text
pre-admission imaging ≠ pre-order information for that same imaging
```

当前 causal masking 正确遮蔽了结构化 `Radiology[].Report`，却没有遮蔽 HPI 中对同一结果的提取。
已确认 development 反例：`pancreatitis:26486125`、`cholecystitis:24115267`、
`pancreatitis:29581468`。旧 Mode-A 和新 A/Q/C 共用同一 `build_record()`，所以这是既有输入认识的
漏洞，不是新 prompt、prior A/Q/C 或 LLM hallucination。

含义：模型可以忠实地从 given record 提取信息，但 given record 未必是正确决策时点的 record。
缺少 HPI note-level charttime 时无法完美自动恢复；后续应在模型发送前比较 HPI 与当前被遮蔽影像，
筛查同模态结果复述，明确泄漏则清理/排除/按原始 MIMIC 时间重建，标注后人工复核作为第二道防线。

已实现并接入 runner 的 preflight：`scripts/audit_aqc_input_leakage.py`。它不修改 prompt；候选通过句子
SHA-256 进入人工 review。只有人工确认的 exact sentence 才在内存中删除，原始 CSV 不变，并记录原始/
过滤后 HPI hash、review hash 和 redaction provenance。bridge 004 的 3 个 confirmed leak 已完成
GPT-5.1 makeup 重标：3 人/5 步均通过 validator 与过滤后输入的 evidence audit；旧输出保留为
`superseded_runs`，本轮增量记录成本 `$0.234268`。

下一批 bridge 005 已冻结 20 人/32 步。preflight 命中 `pancreatitis:27929956:s1`：HPI 对当前 CT 的
“无 free air + 稠密钡剂导致后十二指肠壁评估受限”几乎逐项复述，已人工确认并精确删除；复核后
`blocking=false`。该批次已完成 GPT-5.1 标注：20 人/32 步，2 个无效 JSON 步骤经 scoped repair 后
32/32 有效，0 个低 evidence-fidelity 告警。`pancreatitis:28226418:s2` 有 1 个 temporal wording/
requirement misalignment，已用人工 overlay 最小修正且不覆盖模型原输出。批次累计记录成本 `$1.493984`。

新的稳定性观察：32/32 步仍全部恰好生成 5 条 assumptions（160 条：122 `well_supported`、38
`weakly_supported`）。这不违反 “at most five”，但说明 GPT-5.1 持续把上限当作目标；它不是
`established` assumptions 无限导致的，而是上限、固定数组模板与“覆盖完整”倾向共同造成的。

### temporal role
有时重复的modality是因为时间正常的推移，需要观察是否吸收了