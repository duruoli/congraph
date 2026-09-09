# Manual extraction review notes

These notes record the main semantic judgments made during manual extraction.
They are not additional annotations and are not part of the JSON contract.

| Step | Main judgment retained in the manual extraction |
|---|---|
| `appendicitis:20123918:s2` | Pregnancy and Crohn disease are material attributes. The complete abdominal US challenges biliary disease but does not evaluate the appendix; no appendicitis diagnosis is manufactured from RLQ pain alone. |
| `appendicitis:20123918:s3` | The second US reports no RLQ inflammatory focus but does not explicitly document appendix visualization. It is therefore recorded as negative for RLQ inflammation, not as a definitive negative appendix study. |
| `appendicitis:20276429:s2` | A 1-cm noncompressible, hyperemic appendix with surrounding fat edema is sufficient to encode established acute appendicitis despite a nontender examination. |
| `appendicitis:20689999:s2` | Pelvic US is limited for right-ovarian pathology because the right ovary is not visualized. The record does not explicitly establish appendicitis, so only native symptoms, signs, and imaging findings are retained. |
| `cholecystitis:20334898:s2` | Stones plus only possible minimal pericholecystic stranding and no wall thickening are encoded as equivocal acute cholecystitis. Severe respiratory disease/support is retained because it materially constrains the clinical context. |
| `cholecystitis:20660601:s2` | Extensive pericholecystic inflammation establishes acute cholecystitis; the report's “probable perforation” remains suspected/equivocal rather than established. The temporary pacing wire is preserved through the open intervention/device dimension. |
| `cholecystitis:21948836:s2` | Per study direction, MRCP text already present in the rendered HPI is treated as visible. The earlier pancreatitis label is marked challenged by the later “No MR evidence of pancreatitis”; choledocholithiasis is excluded, while cholangitis remains suspected. |
| `diverticulitis:20180280:s2` | Sigmoid diverticular disease, wall thickening, free air, and multiple evolving pelvic collections support established complicated diverticulitis. Resolved bowel dilation is kept distinct from the still-evolving collections. |
| `diverticulitis:21292285:s2` | The examination supports suspected acute peritonitis and hemodynamic instability. Noncontrast CT shows no free gas but is limited for the inflammatory source, so diverticulitis remains equivocal rather than being inferred from the pilot stratum. |
| `pancreatitis:20001800:s2` | Classic epigastric-to-back pain and recent alcohol exposure support suspected, not established, acute pancreatitis. RUQ US challenges a biliary cause and is limited for direct pancreatic assessment. |
| `pancreatitis:20720063:s3` | Peripancreatic inflammatory change supports established acute pancreatitis. Ductal dilation without a visualized stone is retained as equivocal biliary obstruction rather than converted to choledocholithiasis. |
| `pancreatitis:25133113:s6` | Acute pancreatitis and renal failure are established. Serial noncontrast imaging shows minimally progressive inflammation but still cannot assess necrosis or vascular complications. The biliary stent and contrast limitation are retained as open-dimension bridge candidates. |

## Mechanical comparison with the API run

The API run produced 3 strictly valid outputs out of 12. Two responses were not
JSON objects; the other invalid outputs primarily used evidence text that was
not an exact source substring. It emitted 605 items across the 12 steps. The
manual baseline emits 277 more selective items, and all 309 evidence spans pass
the exact-substring validator.

This count difference is descriptive, not proof that every omitted API item is
clinically irrelevant. The next useful debugging pass is a per-item semantic
diff: manual-only, API-only, and shared propositions with different status or
scope.
