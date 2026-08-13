# Track A - faithful ACR normative extraction

This directory contains an ACR-native extraction for the four study topics. It is deliberately
independent of the empirical A/Q/C representation and does not encode a mandatory diagnostic path.

## Corpus and versions

| Topic | ACR topic ID | Topic version printed in narrative | Variants | Actions | Rationale sections |
|---|---:|---|---:|---:|---:|
| Right Lower Quadrant Pain | 21 | Revised 2022 | 3 | 30 | 21 |
| Right Upper Quadrant Pain | 132 | Revised 2022 | 5 | 34 | 19 |
| Left Lower Quadrant Pain | 20 | Revised 2023 | 3 | 29 | 20 |
| Acute Pancreatitis | 126 | New 2019 | 6 | 48 | 30 |
| **Total** | | | **17** | **141** | **90** |

The source manifest records the UTC acquisition time separately. ACR dynamically generates the
download files, so their generation date is not treated as the topic revision date.

## Files

- `acr_topics.json`: canonical nested topic -> variant -> action corpus, plus narrative rationale.
- `acr_actions.jsonl`: denormalized action-level view for analysis and modeling.
- `schema/acr_extraction.schema.json`: formal schema for the canonical JSON.
- `native_vocabulary.json`: context phrases and action vocabulary induced from this corpus only.
- `sources/`: official narrative, appendix, and evidence-table snapshots plus HTML appendix/evidence
  snapshots and a SHA-256 manifest.
- `audit/sample_audit.md`: stratified manual comparison against rendered source pages.
- `ambiguities.md`: source inconsistencies and extraction decisions that must not be silently erased.

Rebuild without changing existing source snapshots:

```bash
python3 scripts/extract_acr_normative.py
python3 scripts/validate_acr_normative.py
```

Pass `--refresh` only when intentionally creating a new source snapshot; ACR source files are
dynamic and refreshing can change hashes or content.

## Schema principles

The primary unit is an ACR `topic / variant / procedure`. The exact variant and procedure wording
is retained. Each action separately stores:

- appropriateness category, numeric rating, median, and full 1-9 final tabulations;
- strength of evidence (SOE), not conflated with appropriateness;
- adult and pediatric relative radiation levels;
- the appendix references and study-quality values exactly as printed;
- a link to the corresponding ACR narrative procedure-family rationale;
- PDF page, semantic locator, official URL, local file, and stable HTML locator.

`context` and `action_components` are post-extraction views derived from literal source phrases.
They do not replace the original wording. No A/Q/C fields occur in this corpus.

The reviewed context has four top-level parts:

```text
clinical_state   = presentation + condition + severity/complication
imaging_history  = prior test + prior result
modifiers        = population + timing + constraints/confounders
decision_stage   = initial/next/unspecified imaging + encounter status
```

`first time presentation` is encounter status, not population. Prior imaging remains distinct from
presentation because it is the key normative signal for repeat/switch/next-imaging decisions.

## Action ranking

`final_rating` (1-9, higher is more appropriate) is the single primary ranking metric. Equal ratings
remain tied: the extraction does not fabricate a unique path. `appropriateness_category`, SOE,
median, and vote distribution remain for interpretation and audit but do not alter the primary
rank. The category preserves the one explicit panel-disagreement case; SOE describes evidentiary
support, not action preference. When ACR explicitly calls procedures `equivalent alternatives` or
`complementary`, the variant stores that source-backed relationship in `action_relationships`.

## Fidelity boundary

The narrative rationale is stored by ACR's own procedure-family headings (for example, `A. CT
Abdomen and Pelvis`), because the narrative generally discusses protocol alternatives together.
The action table remains procedure-specific. Thus, a rationale link means “this action belongs to
the source's rationale family,” not that every sentence in the paragraph applies uniquely to that
protocol.

This extraction preserves the complete official topic source snapshots. It structures all variant
tables and all procedure-family rationale sections. Introductory narrative, panel membership,
references, and full evidence tables remain available verbatim in `sources/` rather than being
duplicated into the JSON.
