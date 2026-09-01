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
