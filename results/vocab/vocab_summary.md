# Narrative vocabulary (名词库 / 状态库) — v1 converged

Built by `scripts/normalize_vocab.py` (API-free, hand-authored medical rules) from `results/evidence_pieces/*.jsonl`, NARRATIVE pieces only (labs/micro excluded). Raw→canonical maps live in `results/vocab/{anatomy,state}_map.json`.

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

POLARITY preserved (X vs `no X` are distinct). `__NUMERIC__` (numeric vitals, read via finding_status+value_unit): 562 occ. `__DROP__` (degree-only / process words): 436 occ. Residual: 1264 occ (6.8%).

| canonical | freq | surface forms |
|---|--:|--:|
| absent | 4746 | 227 |
| normal | 4686 | 405 |
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
