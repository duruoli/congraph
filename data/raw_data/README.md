# MIMIC-derived abdominal disease data

## Overview

[MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) is a large, deidentified electronic health record database from Beth Israel Deaconess Medical Center. This project uses a small subset focused on four abdominal diseases: appendicitis, cholecystitis, diverticulitis, and pancreatitis. Free-text radiology reports are obtained from the linked [MIMIC-IV-Note v2.2](https://physionet.org/content/mimic-iv-note/2.2/) database.

Access to both databases requires a **credentialed** PhysioNet account, completion of the required CITI training, and acceptance of the Data Use Agreement for each project. See the official [PhysioNet access instructions](https://physionet.org/about/citi-course/) for details. The derived CSVs in this folder are stored for authorized internal research only and must not be made public.

```text
MIMIC-docs/
├── README.md
├── appendicitis_hadm_info_first_diag.csv
├── cholecystitis_hadm_info_first_diag.csv
├── diverticulitis_hadm_info_first_diag.csv
├── pancreatitis_hadm_info_first_diag.csv
└── scripts/
    ├── download_mimic_source.sh
    ├── build_timing_table.py
    └── timing.py
```

## Files in this directory

### Disease-specific extracts

The four CSV files are **derived, admission-level extracts**, not raw MIMIC tables:

| File | Admissions |
|---|---:|
| `appendicitis_hadm_info_first_diag.csv` | 957 |
| `cholecystitis_hadm_info_first_diag.csv` | 648 |
| `diverticulitis_hadm_info_first_diag.csv` | 257 |
| `pancreatitis_hadm_info_first_diag.csv` | 538 |

Each row represents one hospital admission (`hadm_id`). The original extraction selected admissions associated with the four diseases using ICD diagnosis information and aggregated the following dimensions:

| Dimension | CSV columns | Content |
|---|---|---|
| Admission identifier | `hadm_id` | Deidentified identifier for one hospital admission. |
| Patient history | `Patient History` | Integrated history/HPI narrative. |
| Physical examination | `Physical Examination` | Examination findings and documented vital signs. |
| Laboratory tests | `Laboratory Tests` | JSON dictionary of laboratory `itemid` and recorded values. |
| Laboratory reference ranges | `Reference Range Lower`, `Reference Range Upper` | Lower and upper reference values corresponding to the laboratory items. |
| Microbiology | `Microbiology`, `Microbiology Spec` | Aggregated microbiology results and specimen information when available. |
| Radiology | `Radiology` | JSON list containing `Note ID`, modality, body region, exam name, and report text. |
| Diagnoses | `Discharge Diagnosis`, `ICD Diagnosis` | Retrospective diagnosis information from the completed admission. |
| Procedures | `Procedures Discharge`, `Procedures ICD9`, `Procedures ICD9 Title`, `Procedures ICD10`, `Procedures ICD10 Title` | Documented procedures, codes, and descriptions from the admission. |

The exact cohort-generation query and ICD-9/ICD-10 inclusion rules are not available in this repository, so the disease selection cannot be reproduced from these CSV files alone.

### MIMIC source tables

The large official source tables are **not included** in this folder. After PhysioNet access is approved, run:

```bash
bash scripts/download_mimic_source.sh <physionet_username>
```

The script downloads the following tables to `mimic_source/`. These are the raw source tables needed for **temporal reconstruction**, not the complete MIMIC database.

| File | Source | Purpose |
|---|---|---|
| `admissions.csv.gz` | MIMIC-IV v3.1 | Provides `admittime` for each `hadm_id`. |
| `radiology.csv.gz` | MIMIC-IV-Note v2.2 | Provides each report's `note_id` and `charttime`. |
| `procedures_icd.csv.gz` | MIMIC-IV v3.1 | Provides procedure dates and ICD codes. |
| `d_icd_procedures.csv.gz` | MIMIC-IV v3.1 | Maps procedure codes to procedure names. |

## Reconstructing the radiology timeline

The disease-specific CSV files aggregate all radiology reports from an admission but do not include report timestamps. Their stored list order should therefore not be treated as chronological.

Timing is reconstructed as follows:

1. Match `Radiology[].Note ID` in a disease CSV to `radiology.note_id` to recover the report `charttime`.
2. Match `hadm_id` to `admissions.csv.gz` to recover `admittime`.
3. Match `hadm_id` to `procedures_icd.csv.gz`, then decode the procedure through `d_icd_procedures.csv.gz`.
4. Identify the earliest major therapeutic intervention relevant to the abdominal disease, such as appendectomy, cholecystectomy, therapeutic ERCP, or abdominal drainage.
5. Sort reports by `charttime` and compare them with the admission and intervention anchors.

These temporal categories are **not fields provided by MIMIC**. They are derived by this project from `charttime`, `admittime`, and the date of the first qualifying intervention.

| Project-derived category | Rule | Interpretation |
|---|---|---|
| `pre_admission` | `charttime < admittime` | Report charted before formal hospital admission, usually during an ED or outpatient period. |
| `pre_intervention` | Before the first qualifying intervention date | Report charted before a major treatment event. This may include both pre-admission and post-admission imaging. |
| `same_day_as_intervention` | On the first intervention date | Exact order is uncertain because the procedure has no time of day. |
| `post_intervention` | After the first intervention date | Usually monitoring or management imaging. |

These categories are not mutually exclusive: a pre-admission report is also pre-intervention when it precedes the first intervention.

Because MIMIC procedure data provide a date (`chartdate`) rather than an exact time, `same_day_as_intervention` takes precedence whenever a report and intervention have the same calendar date.

The download and reconstruction code is available in `scripts/download_mimic_source.sh`, `scripts/timing.py`, and `scripts/build_timing_table.py`.

## Important interpretation notes

### Admission, intervention, and diagnosis are different time anchors

- **Pre-admission** means `radiology.charttime < admissions.admittime`. It only indicates that the report was charted before formal hospital admission, often during the emergency-department period.
- **Pre-intervention** means the report was charted before the first identified major therapeutic procedure. This window may include both pre-admission and post-admission imaging. An intervention is a treatment event, such as surgery, ERCP with therapy, or drainage; it is not the diagnosis itself.
- **Pre-diagnosis** **cannot** be determined from the available files. MIMIC does not provide a reliable timestamp for the moment when the clinical diagnosis was established, and discharge/ICD diagnoses are retrospective.

Therefore:

```text
pre-admission != pre-intervention != pre-diagnosis
```

These labels describe different questions and must not be used interchangeably.

### `Patient History` is not a guaranteed pre-order snapshot

`Patient History` is an integrated clinical narrative. It may summarize events that occurred earlier in the emergency-department or hospital course, including results from imaging already performed. In some cases, it can restate findings from the radiology report being treated as the current decision target.

Consequently, `Patient History` should not automatically be assumed to represent information available before every imaging order. Decision-sequence analyses should screen for same-test result leakage and redact or exclude affected cases when the original note timing cannot be recovered.

## Citations

- Johnson A, Bulgarelli L, Pollard T, et al. [MIMIC-IV, version 3.1](https://doi.org/10.13026/kpb9-mt58). PhysioNet; 2024.
- Johnson A, Pollard T, Horng S, Celi LA, Mark RG. [MIMIC-IV-Note, version 2.2](https://doi.org/10.13026/1n74-ne17). PhysioNet; 2023.
