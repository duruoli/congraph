# Narrative vocabulary (名词库 / 状态库 / 属性库) — v1 converged

Built by `scripts/normalize_vocab.py` (API-free, hand-authored medical rules) from `results/evidence_pieces/*.jsonl`, NARRATIVE pieces only (labs/micro excluded — their attribute IS the analyte name, already canonical via itemid→label). Raw→canonical maps live in `results/vocab/{anatomy,state,attribute}_map.json`.

## 名词库 (anatomy): 48 canonical concepts, 18682 occurrences

Residual (rule-miss, self-mapped): 47 occ (0.3%). Core diagnostic mass: 10105/18682 = 54%.

| canonical | system | core | freq | surface forms |
|---|---|:-:|--:|--:|
| abdomen/pelvis region | gastrointestinal | ✓ | 1996 | 76 |
| gallbladder | hepatobiliary | ✓ | 1000 | 21 |
| stomach | gastrointestinal | ✓ | 969 | 34 |
| liver | hepatobiliary | ✓ | 723 | 46 |
| kidney | urinary | ✓ | 671 | 69 |
| venous | vascular | ✓ | 534 | 38 |
| pancreas | pancreas | ✓ | 495 | 30 |
| female_pelvis | gynecologic | ✓ | 308 | 38 |
| common bile duct | hepatobiliary | ✓ | 302 | 5 |
| colon | gastrointestinal | ✓ | 272 | 22 |
| spleen | spleen | ✓ | 271 | 9 |
| arterial | vascular | ✓ | 250 | 56 |
| bladder | urinary | ✓ | 227 | 11 |
| bowel | gastrointestinal | ✓ | 224 | 19 |
| vasculature | vascular | ✓ | 188 | 33 |
| appendix | gastrointestinal | ✓ | 185 | 11 |
| intrahepatic bile ducts | hepatobiliary | ✓ | 182 | 7 |
| peritoneum | peritoneum_mesentery | ✓ | 151 | 7 |
| biliary tree | hepatobiliary | ✓ | 139 | 20 |
| lymph nodes | lymphatic | ✓ | 128 | 31 |
| small bowel | gastrointestinal | ✓ | 125 | 11 |
| rectum | gastrointestinal | ✓ | 108 | 11 |
| pancreatic duct | pancreas | ✓ | 107 | 11 |
| adrenal gland | adrenal | ✓ | 102 | 10 |
| pelvis | gynecologic | ✓ | 91 | 10 |
| abdominal wall | peritoneum_mesentery | ✓ | 79 | 33 |
| male_pelvis | male_gu | ✓ | 79 | 18 |
| esophagus | gastrointestinal | ✓ | 43 | 4 |
| duodenum | gastrointestinal | ✓ | 36 | 10 |
| peripancreatic region | pancreas | ✓ | 35 | 9 |
| ureter | urinary | ✓ | 25 | 7 |
| hepatic duct | hepatobiliary | ✓ | 25 | 9 |
| cecum | gastrointestinal | ✓ | 12 | 3 |
| cystic duct | hepatobiliary | ✓ | 12 | 3 |
| mesentery | peritoneum_mesentery | ✓ | 5 | 2 |
| terminal ileum | gastrointestinal | ✓ | 4 | 1 |
| diverticulum | gastrointestinal | ✓ | 2 | 2 |
| systemic/constitutional | systemic |  | 2558 | 36 |
| lungs | pulmonary |  | 2316 | 162 |
| musculoskeletal | musculoskeletal |  | 1148 | 189 |
| heart | cardiac |  | 947 | 32 |
| heent | heent |  | 545 | 75 |
| device/line | device_line |  | 323 | 138 |
| nervous system | neurologic |  | 314 | 35 |
| integument | integument |  | 265 | 12 |
| psychiatric | psychiatric |  | 54 | 3 |
| thyroid | endocrine |  | 38 | 3 |
| breast | breast |  | 22 | 6 |

## 状态库 (state): 64 canonical descriptors, 18682 occurrences

POLARITY preserved (X vs `no X` are distinct). `__NUMERIC__` (numeric vitals, read via finding_status+value_unit): 562 occ. `__DROP__` (degree-only / process words): 436 occ. Residual: 1262 occ (6.8%).

| canonical | freq | surface forms |
|---|--:|--:|
| absent | 4746 | 227 |
| normal | 4688 | 406 |
| present | 3342 | 327 |
| increased | 278 | 58 |
| tender | 251 | 104 |
| non-visualized | 239 | 78 |
| fluid/collection | 229 | 166 |
| decreased | 189 | 56 |
| dilated | 172 | 53 |
| stones/sludge | 157 | 119 |
| distended | 140 | 29 |
| mass/lesion | 128 | 113 |
| opacity/consolidation | 120 | 105 |
| patent/flow | 115 | 11 |
| atelectasis | 107 | 64 |
| thickened | 106 | 52 |
| prominent | 106 | 52 |
| enlarged | 95 | 39 |
| no mass/lesion | 89 | 74 |
| echogenic | 84 | 56 |
| edema | 76 | 52 |
| no edema | 75 | 16 |
| no thickened | 74 | 12 |
| no fluid/collection | 68 | 44 |
| no dilated | 62 | 18 |
| no enlarged | 56 | 12 |
| calcified | 56 | 46 |
| inflammatory stranding | 55 | 43 |
| contracted/small | 46 | 21 |
| vital-abnormal | 45 | 12 |
| hypoattenuating | 41 | 39 |
| no tender | 41 | 12 |
| enhancing | 35 | 24 |
| cardiomegaly | 27 | 17 |
| no stones/sludge | 27 | 22 |
| obese/habitus | 26 | 3 |
| heterogeneous | 23 | 12 |
| guarding | 20 | 10 |
| no guarding | 17 | 3 |
| no distended | 15 | 7 |
| jaundice | 14 | 10 |
| no increased | 14 | 8 |
| no opacity/consolidation | 13 | 10 |
| no rebound | 13 | 1 |
| equivocal | 11 | 7 |
| no obstruction | 10 | 5 |
| compressible | 10 | 3 |
| no peritoneal signs | 8 | 6 |
| no calcified | 8 | 4 |
| no prominent | 7 | 1 |
| no necrosis/nonenhancement | 6 | 4 |
| necrosis/nonenhancement | 6 | 6 |
| rebound | 5 | 5 |
| no enhancing | 5 | 5 |
| no inflammatory stranding | 5 | 3 |
| peritoneal signs | 4 | 4 |
| non-compressible | 4 | 2 |
| no perforation | 3 | 3 |
| no jaundice | 3 | 2 |
| no echogenic | 3 | 3 |
| obstruction | 2 | 2 |
| no vital-abnormal | 1 | 1 |
| perforation | 1 | 1 |
| no patent/flow | 1 | 1 |

## 属性库 (attribute): 63 canonical dims, 18682 occurrences

`attribute` is the CORE axis of the adequacy gate — the atomic covered/required unit is the `(anatomy, attribute)` PAIR, and `required(S)` / `sought_dimensions` ARE attribute dims. **core = `morphology` axis** = the imaging/exam characterization dims a study COVERS (the required(S)/covered(S) dims the gate reads); the other axes (symptom / sign / vital / constitutional / history / device) are the demand-trigger side, slotted present/absent.

Morphology (adequacy-covered) mass: 9775/18682 = 52%. Residual (rule-miss, self-mapped — mostly the rare comorbidity/disease-name tail): 821 occ (4.4%).

| canonical | axis | core | freq | surface forms |
|---|---|:-:|--:|--:|
| appearance/contour | morphology | ✓ | 2184 | 152 |
| size/caliber | morphology | ✓ | 1998 | 63 |
| fluid/collection | morphology | ✓ | 886 | 69 |
| mass/lesion | morphology | ✓ | 714 | 112 |
| echogenicity/density | morphology | ✓ | 597 | 82 |
| patency/flow | morphology | ✓ | 547 | 37 |
| stones/calculi | morphology | ✓ | 460 | 34 |
| opacity/consolidation | morphology | ✓ | 331 | 27 |
| wall/thickness | morphology | ✓ | 275 | 25 |
| edema | morphology | ✓ | 274 | 8 |
| air/gas | morphology | ✓ | 259 | 32 |
| enhancement/vascularity | morphology | ✓ | 227 | 40 |
| pneumothorax | morphology | ✓ | 202 | 1 |
| hydronephrosis | morphology | ✓ | 132 | 5 |
| lymphadenopathy | morphology | ✓ | 130 | 3 |
| atelectasis | morphology | ✓ | 115 | 3 |
| fat-stranding/inflammation | morphology | ✓ | 89 | 17 |
| pulmonary-edema/congestion | morphology | ✓ | 82 | 17 |
| hemorrhage/thrombus | morphology | ✓ | 71 | 24 |
| obstruction | morphology | ✓ | 66 | 11 |
| hernia | morphology | ✓ | 52 | 11 |
| compressibility | morphology | ✓ | 41 | 3 |
| necrosis/ischemia | morphology | ✓ | 22 | 11 |
| perforation | morphology | ✓ | 14 | 8 |
| organomegaly | morphology | ✓ | 7 | 2 |
| general-appearance | constitutional |  | 160 | 5 |
| hydration | constitutional |  | 116 | 3 |
| mental-status | constitutional |  | 101 | 20 |
| body-habitus | constitutional |  | 33 | 6 |
| device-position | device |  | 321 | 50 |
| device/hardware | device |  | 205 | 95 |
| foreign-body | device |  | 14 | 4 |
| comorbidity/pmh | history |  | 1242 | 500 |
| surgical-history | history |  | 43 | 31 |
| social-history | history |  | 40 | 17 |
| tenderness | sign |  | 421 | 13 |
| lung-auscultation | sign |  | 340 | 32 |
| heart-auscultation | sign |  | 229 | 21 |
| skin/wound | sign |  | 196 | 35 |
| jaundice/icterus | sign |  | 135 | 4 |
| bowel-sounds | sign |  | 132 | 9 |
| guarding/rigidity | sign |  | 131 | 5 |
| rebound | sign |  | 126 | 4 |
| musculoskeletal-exam | sign |  | 74 | 10 |
| special-sign | sign |  | 70 | 9 |
| pulses | sign |  | 29 | 3 |
| peritoneal-signs | sign |  | 14 | 3 |
| pain | symptom |  | 665 | 54 |
| nausea/vomiting | symptom |  | 476 | 20 |
| fever/chills | symptom |  | 417 | 18 |
| bowel-habit | symptom |  | 285 | 38 |
| neuro-symptom | symptom |  | 156 | 45 |
| respiratory-symptom | symptom |  | 134 | 14 |
| appetite/intake | symptom |  | 127 | 26 |
| urinary-symptom | symptom |  | 126 | 24 |
| gi-bleeding | symptom |  | 117 | 14 |
| constitutional-symptom | symptom |  | 35 | 9 |
| temperature | vital |  | 259 | 2 |
| heart-rate | vital |  | 238 | 7 |
| blood-pressure | vital |  | 236 | 6 |
| respiratory-rate | vital |  | 218 | 6 |
| oxygen-saturation | vital |  | 218 | 5 |
| cardiac-rhythm | vital |  | 207 | 4 |
