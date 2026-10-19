# sameness v1：基于旧器官 / 问题分类的三层 sameness（已搁置，2026-10-19）

问题层改由出院 ICD 编码作隐藏状态 s（见 `RESEARCH_ANCHOR.md` 第二圈"数学化框架"）后搁置。下面是搁置前 anchor 第三圈里这一步的原文。

仍在用的产出：② 器官表 `scripts/organ_map.csv`（地形页器官层）；① 同义合并 `scripts/sameness.py`、`scripts/sameness_merge.csv`、`runs/sameness/v1/`（以后可作影像 o 的词表）。被取代的：③ 问题表 `scripts/problem_map.csv`。

---

- [ ] **sameness（新对话从这里开始）。** 先在 11 个病人上做通，再决定全量。
  - **输入：**
    - 影像：`runs/atomic_facts/llm_v3/facts.jsonl`。11 个病人（3 个 pilot + 14 例转科抽样里每类前 2 例）的全部 79 份报告，1,495 条。槽位：`anatomy`（可为空）· `observation` · `certainty` · `change` · `known` · `finding_status` · `kind` · `based_on` · `repeats` · `modifier` · `section` · `quote`，外加 `note_id` / `hadm_id` / `t`。
    - 化验 / 微生物：`runs/terrain_units/word/units.parquet` 里 `source` = lab / micro 的行，已是 atomic（化验带 high / low / normal）。
  - **① 同义：** 归一 `observation` 和 `anatomy` 的写法：复数、thickened / thickening / wall thickening、形容词部位（intraperitoneal → 腹腔，pleural effusion → 胸膜 + 积液）。可复用 `scripts/normalize_vocab.py` 的思路（急腹症那批：部位 48 类、状态 64 类）。
    - **10-16 第一版已跑**（`scripts/sameness.py` → `runs/sameness/v1/`，Sonnet，每个写法问一次，$0.92）：观察 610 → 382 种，部位 562 → 208 个结构（侧别 / 子部位单独存，不丢）。
    - **10-17 人工第二轮**（LLM 分批并行、批间看不到，同义被起了不同名字）：合并表 `scripts/sameness_merge.csv`（类 → 类 / 单个写法改归 / 跨槽位"淋巴结 + 肿大 = 淋巴结病"），每行写了理由。原则：①只合并"同一个观察换说法"；临床相关但不同的东西（血栓 vs 栓塞、转移 vs 腹膜癌病）留给③。结果观察 382 → 264，部位 208 → 170；异常事实"同一结构 × 同一观察、≥2 份报告"14 → 54 组。
    - **10-17 对照原句复查**：撤销所有"要靠医学判断才成立"的合并（囊性病变 ≠ 囊肿、变窄 ≠ 狭窄、临界 ≠ 增大、灌肠 ≠ 口服造影、血流 / 波形各说法等）——那是②③的事，不是①。合并表现 153 行。
    - **否定规则**：否定事实里限定词不能去（"无急性骨折" ≠ "无骨折"）。原因不是 LLM 误解否定，而是按写法映射看不到肯定 / 否定。处理：否定 + 有限定词 → 类名保留原写法（46 条事实）。最终观察 318 类、部位 170 类。化验 / 微生物不需要这一步（本来就一个编码一件事）。
  - **② 器官：** 部位 → 10 个器官 + "未定位"。部位为空的事实，按"观察"查一张**每种观察只定一次**的标准部位表（LLM 每种问一次 + 人工逐行审阅；拿到 SNOMED 后用"发生部位"属性核对）；没有唯一部位的（积液、脓肿、淋巴结肿大）进"未定位"。化验按"直接测哪里"归器官。
  - **③ 问题：** 人工定义、单独一张表，可审阅。
  - **10-17 ②③ 手工版，3 个 pilot**（不用 LLM；`scripts/organ_map.csv`、`scripts/problem_map.csv` → `scripts/organ_problem.py` → `runs/sameness/v1/organ_problem/summary.md`）。表都按"观察"写一次、不看肯定 / 否定：极性由提取时每条事实的 `finding_status` 决定（"没有血流" = 异常，"没有单相波" = 正常），所以②③不会在否定句上出错。器官在 10 个之外加了"其他"（脾、骨、淋巴结、上消化道等）和"肠（未分）"；问题在 12 个之外加了"肿瘤 / 占位"。只计数，不加权：影像里最多的问题正好是转科去向（穿孔 13 条 / 血栓 36 条），但加上化验后，三人排第一的都是"出血 / 贫血"（一次血常规 = 7 条异常，每天抽一次）→ 必须先有海平面和同源去重，才能把化验和影像放在一起看。CMED 那例的"心脏问题"只来自肌钙蛋白，影像里没有。待查：抗凝期间 PTT / INR 高是治疗痕迹；淋巴结肿大归"肿瘤"还是"感染"。
  - **然后：** weight（见下）→ 用 atomic fact 重建这 11 人的地形页（替代规则版），3 个 pilot 病例和规则版对比。
  - **weight 的候选规则（事先定，不调）：** 新出现 1、同源再现 0.5、抄写 0；`repeats` 非空（结论复述所见）= 0；判断通过 `based_on` 挂在观察上，不另算一份；`change` = 不变 → 监测性重复，`known` = 背景；判断可以给它依据的观察减分（"增厚，可能是塌陷所致"）。
  - **注意：**
    - 11 人里 7 人不是 IBD 主诊断，不在海平面人群里。
    - 影像的海平面需要全队列提取（约 5.3k 份报告，估 $50），等方法验证后再决定；在那之前，影像只看绝对密度。
