# ACR source ambiguities and extraction notes

These cases are preserved rather than silently normalized.

- **RUQ variant 5 heading mismatch:** the appendix action is `Image-guided cholecystostomy`, while
  the narrative heading is `D. Image-Guided Biopsy Liver`. The paragraphs under that heading
  discuss percutaneous cholecystostomy, bile aspiration, and gallbladder drainage. The action keeps
  the appendix wording and links to the narrative heading exactly as printed.
- **HIDA naming:** the current appendix uses `HIDA scan`; the narrative table uses `Nuclear medicine
  scan gallbladder`. They are retained verbatim in their respective source fields and mapped to the
  shared native action family `nuclear_medicine_gallbladder`.
- **Non-PMID parenthesized identifiers:** several appendix references print negative identifiers,
  such as `-3194377` and `-3188535`. `pmid_as_printed` preserves these strings and does not claim
  that they are valid PubMed identifiers.
- **Study-quality value types:** individual references may print numeric categories, `Good`, or
  `Inadequate`. These remain strings; action-level SOE is a separate field.
- **`n/a` medians and zero tabulations:** 20 RLQ actions print a numeric final rating but `n/a` for
  median and all-zero final tabulations. These are preserved exactly, not imputed.
- **Procedure-family rationale:** narrative headings often combine several action rows (for example,
  contrast and noncontrast CT). Links are family-level and do not imply that every sentence applies
  to every protocol.
- **ACR wording/typographical issues:** capitalization (`Initial Imaging`), `pelvic` versus `pelvis`,
  and narrative typographical errors are not used to rewrite the action or variant text.
- **Derived context vocabulary:** terms are open-coded only when their literal phrase occurs in a
  variant. The reviewed four-part grouping (`clinical_state`, `imaging_history`, `modifiers`,
  `decision_stage`) is an indexing schema, not A/Q/C or a claim that all clinical context reduces to
  these 17 variants.
- **Action ranking:** `final_rating` is used to order actions within a variant. Ties are retained.
  Appropriateness category and SOE are not used as numeric tie-breakers.
