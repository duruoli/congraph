# Stratified manual audit against ACR source pages

Audit date: 2026-08-13 (Asia/Shanghai). Twelve actions were selected deterministically: three per
topic, spanning variants, initial/next imaging, appropriateness categories, diagnostic modalities,
an intervention, `n/a` medians, and unusual evidence values. Source PDFs were rendered to PNG and
visually inspected; the local source files are hash-locked in `sources/manifest.json`.

For every sample, the audit checked exact variant and procedure wording, category, SOE, rating,
median, adult/pediatric RRL, all nine final-tabulation counts, listed reference/study-quality pairs,
appendix page, rationale-family link, narrative page range, and the absence of A/Q/C fields.

| Action ID | Source locator | Deliberate edge covered | Result |
|---|---|---|---|
| `acr_21_v1_a01` | RLQ appendix p.1, V1 / CT with IV contrast; narrative V1/A pp.4-5 | Usually appropriate; Strong; rating/median 9; ten references including `Good` | Pass |
| `acr_21_v2_a02` | RLQ appendix p.6, V2 / US abdomen; narrative V2/E pp.13-14 | `n/a` median and zero tabulations retained | Pass |
| `acr_21_v3_a08` | RLQ appendix p.10, V3 / MRI without and with IV contrast; narrative V3/C pp.15-17 | Pregnant population; usually not appropriate; cross-page PDF row | Pass |
| `acr_132_v1_a03` | RUQ appendix p.2, V1 / radiography; narrative V1/D p.5 | Only `May be appropriate (Disagreement)` action; Expert Opinion | Pass |
| `acr_132_v4_a03` | RUQ appendix p.7, V4 / CT with IV contrast; narrative V4/A p.10 | Negative/equivocal prior US; next imaging; rating 7 | Pass |
| `acr_132_v5_a02` | RUQ appendix p.10, V5 / image-guided cholecystostomy; narrative V5/D pp.12-13 | Interventional action; source heading mismatch documented | Pass |
| `acr_20_v1_a01` | LLQ appendix p.1, V1 / CT with IV contrast; narrative V1/A pp.3-4 | Initial nonspecific LLQ context; rating 9; Strong | Pass |
| `acr_20_v2_a05` | LLQ appendix p.5, V2 / US transabdominal; narrative V2/E p.7 | Usually not appropriate US; Limited; rating 3 | Pass |
| `acr_20_v3_a07` | LLQ appendix p.8, V3 / CT cystography; narrative V3/B p.8 | Complication context; specialized protocol and bladder target | Pass |
| `acr_126_v1_a06` | Pancreatitis appendix p.3, V1 / contrast-enhanced US; narrative V1/D p.5 | Usually not appropriate; off-label rationale; cross-page variant boundary | Pass |
| `acr_126_v5_a02` | Pancreatitis appendix p.8, V5 / US abdomen; narrative V5/C p.14 | Deteriorating necrotizing pancreatitis; Expert Consensus | Pass |
| `acr_126_v6_a02` | Pancreatitis appendix p.10, V6 / CT with IV contrast; narrative V6/A pp.15-16 | >4 weeks; known fluid collections; rating 9 | Pass |

## Findings

- **12/12 action records passed** all table-field and provenance checks.
- **12/12 rationale links passed** at the ACR procedure-family level.
- Render review confirmed that PDF table layout, especially rows split across pages, was not safely
  recoverable from naïve reading-order text alone. The extraction therefore uses the official HTML
  table for values and the rendered/PDF page for page-level provenance.
- The audit found no introduced A/Q/C field or forced diagnostic-sequence interpretation.
- The RUQ variant 5 narrative-heading mismatch, negative parenthesized identifiers, `Good` /
  `Inadequate` study qualities, and `n/a` median behavior are recorded in `ambiguities.md`.

No sampled discrepancy required changing an ACR value. Parser fixes made during audit concerned
lossless capture of unclassed reference cells and page location of visually split rows; the corpus
was regenerated and revalidated afterward.
