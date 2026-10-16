# 地形图映射表：事实 → 器官 / 问题

由 `scripts/terrain_pilot.py` 每次运行时自动生成，规则以代码为准；不要手改本文件。

## 原则

- **只计事实**：异常化验、影像阳性所见、微生物阳性。用药、医嘱、会诊、检查理由、医生判断不计。
- **单位不是 UMLS 概念**：化验按检查项目名（结构化，精确）；影像按手写关键词规则（正则）把同义说法归成一个“发现”；微生物按标本类型。
- **器官** = 这个检查直接测的是哪里的状态；**问题** = 它通常意味着什么（一个事实可喂多个问题，也可不喂任何问题）。
- **权重**：同一（来源, 发现）第一次 = 1，之后每天再出现 = 0.5，同一天重复或影像原句抄写 = 0。
- **影像否定**：否定词（no / without / negative for …）作用到转折词（but / however / though / which …）或句末为止；后置否定（no longer seen / resolved / cannot be excluded …）也算否定；否定的不计。只读 FINDINGS 与 IMPRESSION。
- **微生物阳性**：有菌名，或备注含 positive / detected 且不含 negative / not detected / no growth。

计数列 = 全 IBD 队列（入院前至第 15 天）里这条规则产生的事实数 / 涉及住院数。

## 化验（结果被标为 abnormal）

| 发现 | 器官 | 问题 | 检查项目（正则） | 事实 / 住院 |
|---|---|---|---|---:|
| 血红蛋白低 | 血液 | 出血 / 贫血 | `^(Hemoglobin\|Hematocrit\|Red Blood Cells)$` | 40,360 / 9,060 |
| 铁 / 溶血指标 | 血液 | 出血 / 贫血 | `^(Iron\|Ferritin\|Transferrin\|Iron Binding Capacity, Total\|Reticulocyte Count, Automated\|Haptoglobin)$` | 1,779 / 1,461 |
| 血小板异常 | 血液 | 出血 / 贫血 | `^Platelet Count$` | 15,259 / 3,802 |
| 凝血异常 | 血液 | 出血 / 贫血 | `^(INR\(PT\)\|PT\|PTT\|Fibrinogen, Functional)$` | 13,580 / 4,340 |
| D-二聚体高 | 心脏 / 血管 | 血栓 | `^D-Dimer$` | 166 / 127 |
| 炎症指标高 | 全身 | 肠道炎症活动 | `^(C-Reactive Protein\|Sedimentation Rate)$` | 5,558 / 2,616 |
| 粪钙卫蛋白高 | 结肠 | 肠道炎症活动 | `(?i)calprotectin` | 0 / 0 |
| 白细胞异常 | 血液 | 全身感染 | `^(White Blood Cells\|Neutrophils\|Absolute Neutrophil Count\|Bands\|Immature Granulocytes\|Metamyelocytes\|Myelocytes)$` | 21,387 / 6,388 |
| 乳酸高 | 全身 | 全身感染 | `^Lactate$` | 1,395 / 913 |
| 肾功能异常 | 肾 / 泌尿 | 肾 / 泌尿问题 | `^(Creatinine\|Urea Nitrogen)$` | 18,106 / 5,058 |
| 尿检异常 | 肾 / 泌尿 | 肾 / 泌尿问题 | `^(RBC\|WBC\|Protein\|Bacteria\|Leukocytes\|Blood\|Nitrite)$`（尿） | 2,606 / 2,029 |
| 肝功能异常 | 肝胆 | 肝胆异常 | `^(Alanine Aminotransferase \(ALT\)\|Asparate Aminotransferase \(AST\)\|Alkaline Phosphatase\|Bilirubin, Total\|Bilirubin, Direct)$` | 8,327 / 2,570 |
| 脂肪酶高 | 胰腺 | — | `^Lipase$` | 565 / 356 |
| 营养指标低 | 全身 | 营养 / 电解质 | `^(Albumin\|Protein, Total\|25-OH Vitamin D\|Vitamin B12)$` | 5,524 / 2,576 |
| 电解质异常 | 全身 | 营养 / 电解质 | `^(Potassium\|Magnesium\|Phosphate\|Calcium, Total\|Sodium\|Chloride\|Bicarbonate\|Anion Gap)$` | 30,307 / 7,778 |
| 心肌指标高 | 心脏 / 血管 | 心脏问题 | `^(Troponin T\|NTproBNP)$` | 1,102 / 712 |
| 血气异常 | 肺 | — | `^(pO2\|pCO2)$` | 2,966 / 1,244 |

未列出的化验项目不计。血糖（常是激素副作用）、药物浓度是治疗留下的痕迹，有意排除。

## 影像（报告句子里的阳性所见）

| 发现 | 器官 | 问题 | 关键词规则（正则，不分大小写） | 事实 / 住院 |
|---|---|---|---|---:|
| 结肠炎症 | 结肠 | 肠道炎症活动 | `colitis\|proctitis\|loss of (the )?haustra\|(colon\|colonic\|rectal\|rectum\|sigmoid)[^.]{0,40}(thicken\|inflam\|hyperenhanc)\|(thicken\|inflam)[^.]{0,40}(colon\|rectum\|sigmoid)` | 1,600 / 1,318 |
| 结肠扩张 | 结肠 | 肠道炎症活动、梗阻 / 狭窄 | `megacolon\|(colon\|colonic)[^.]{0,30}(dilat\|distension\|distention)\|distension of the colon` | 191 / 149 |
| 小肠炎症 | 小肠 | 肠道炎症活动 | `ileitis\|enteritis\|(ileum\|ileal\|small bowel\|jejun)[^.]{0,40}(thicken\|inflam\|hyperenhanc)` | 710 / 657 |
| 小肠梗阻 | 小肠 | 梗阻 / 狭窄 | `small bowel obstruction\|\bSBO\b\|dilated (loops of )?small bowel\|small bowel[^.]{0,30}dilat\|air-fluid levels` | 1,397 / 1,035 |
| 狭窄 | 小肠 | 梗阻 / 狭窄 | `stricture` | 300 / 276 |
| 瘘 | 小肠 | 脓肿 / 瘘 | `fistula\|sinus tract` | 576 / 480 |
| 游离气 / 穿孔 | 腹腔 | 穿孔 | `free (intraperitoneal )?air\|pneumoperitoneum\|perforat\|extraluminal (contrast\|air\|gas)\|pneumatosis` | 786 / 588 |
| 脓肿 / 积液 | 腹腔 | 脓肿 / 瘘 | `abscess\|(fluid )?collection` | 1,467 / 995 |
| 腹水 | 腹腔 | — | `ascites\|free fluid` | 1,516 / 1,193 |
| 肝胆影像异常 | 肝胆 | 肝胆异常 | `steatosis\|fatty liver\|hepatomegaly\|cirrho\|biliary (ductal )?dilat\|(intrahepatic\|bile) duct[^.]{0,30}dilat\|cholangitis\|cholelithiasis\|gallstone\|cholecystitis\|sclerosing` | 1,332 / 1,019 |
| 胰腺炎表现 | 胰腺 | — | `pancreatitis\|peripancreatic` | 144 / 120 |
| 肾 / 尿路异常 | 肾 / 泌尿 | 肾 / 泌尿问题 | `hydronephrosis\|nephrolithiasis\|renal (stone\|calculus\|calculi)\|pyelonephritis` | 395 / 324 |
| 肺炎 / 实变 | 肺 | 全身感染 | `pneumonia\|consolidation` | 1,758 / 1,043 |
| 肺部其他异常 | 肺 | — | `pleural effusion\|atelectasis\|pulmonary edema\|opacit` | 5,751 / 2,909 |
| 血栓 | 心脏 / 血管 | 血栓 | `(pulmonary )?embol\|thromb\|\bDVT\b\|deep vein\|\bclot\b` | 789 / 582 |
| 心脏影像异常 | 心脏 / 血管 | 心脏问题 | `pericardial effusion\|cardiomegaly\|reduced (systolic\|ejection)\|hypokinesis` | 921 / 626 |

## 微生物（阳性）

| 发现 | 器官 | 问题 | 标本类型（正则） | 事实 / 住院 |
|---|---|---|---|---:|
| 粪便病原阳性 | 结肠 | 肠道感染 | `^STOOL` | 1,589 / 1,131 |
| 血培养阳性 | 血液 | 全身感染 | `^BLOOD CULTURE` | 272 / 175 |
| 尿培养阳性 | 肾 / 泌尿 | 肾 / 泌尿问题 | `^URINE` | 590 / 490 |
| 腹腔 / 伤口培养阳性 | 腹腔 | 脓肿 / 瘘 | `^(ABSCESS\|PERITONEAL FLUID\|TISSUE\|FLUID,OTHER\|SWAB)` | 575 / 509 |
| 胆汁培养阳性 | 肝胆 | 肝胆异常 | `^BILE` | 22 / 19 |
| 呼吸道培养阳性 | 肺 | 全身感染 | `^(SPUTUM\|BRONCHOALVEOLAR\|BRONCHIAL\|Mini-BAL)` | 277 / 193 |
| 导管培养阳性 | 全身 | 全身感染 | `^CATHETER TIP` | 23 / 23 |

## 问题 ← 发现（反查）

| 问题 | 喂它的发现 |
|---|---|
| 肠道炎症活动 | 炎症指标高、粪钙卫蛋白高、结肠炎症、结肠扩张、小肠炎症 |
| 肠道感染 | 粪便病原阳性 |
| 穿孔 | 游离气 / 穿孔 |
| 脓肿 / 瘘 | 瘘、脓肿 / 积液、腹腔 / 伤口培养阳性 |
| 梗阻 / 狭窄 | 结肠扩张、小肠梗阻、狭窄 |
| 出血 / 贫血 | 血红蛋白低、铁 / 溶血指标、血小板异常、凝血异常 |
| 全身感染 | 白细胞异常、乳酸高、肺炎 / 实变、血培养阳性、呼吸道培养阳性、导管培养阳性 |
| 肝胆异常 | 肝功能异常、肝胆影像异常、胆汁培养阳性 |
| 肾 / 泌尿问题 | 肾功能异常、尿检异常、肾 / 尿路异常、尿培养阳性 |
| 血栓 | D-二聚体高、血栓 |
| 营养 / 电解质 | 营养指标低、电解质异常 |
| 心脏问题 | 心肌指标高、心脏影像异常 |
