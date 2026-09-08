# Manual open patient-context pilot v1

This directory contains a hand-annotated calibration pilot for the Stage-1
open patient-context extraction schema.

## Unit of annotation

- 12 imaging decision steps
- 11 unique patients
- `appendicitis:20123918` contributes two sequential decisions (`s2`, `s3`)

The same patient's steps are intentionally kept separate. At `s3`, the
resulted focused RLQ ultrasound becomes causally available and changes the
open context.

## Files

- `inputs.json`: blinded records used for manual extraction.
- `case_map.json`: anonymous case IDs to source decision-step IDs; kept
  separate during review.
- `manual_context_items_v1.tsv`: auditable, hand-authored atomic context
  items and verbatim evidence spans.
- `manual_case_notes_v1.json`: case-level ambiguity notes.
- `manual_extractions_v1.jsonl`: compiled output in the Stage-1 prompt
  contract (one decision step per line).
- `manual_acr_vocab_mapping_audit_v1.tsv`: Stage-2 item-to-ACR mapping audit;
  one row per patient-item/ACR-value relation.
- `manual_acr_vocab_mappings_v1.jsonl`: machine-readable Stage-2 output
  retaining each complete patient item, all ACR links, source Variant IDs,
  and the explicit `unmapped_value_within_dimension` state.
- `manual_acr_vocab_mapping_case_summary_v1.tsv`: decision-step summary of
  entailed, partial, judgment-dependent, contradicted, and unmapped mappings.

Regenerate and validate the compiled output with:

```bash
python3 scripts/compile_manual_open_context_pilot.py
python3 scripts/map_manual_open_context_to_acr_vocab.py
```

The compiler checks case coverage, dimensions, epistemic sources, item
references, prior-imaging indices, and exact substring agreement between
every evidence quote and the blinded input.

Stage-2 mapping currently contains 175 patient items and 206 item-to-vocabulary
relations. Of the 175 items, 116 have at least one ACR link and 59 are retained
as `unmapped_value_within_dimension`. Multiple links are intentional when a
conjunctive patient item instantiates several ACR values or when synonymous
ACR-native values come from different Variants.

The mapping stage reads only the frozen open-context extraction and the
audited 50-value ACR vocabulary. It does not load A/Q/C, the current imaging
order or result, action ratings, or later events. Variant-level candidate
generation and adjudication remain a separate next step.

## Interpretation

This is a **method-calibration set**, not an independent blind gold standard.
The pilot cases were previously involved in bridge exploration, so the output
is labeled `manual_contaminated_calibration_v1`. It is suitable for checking
schema coverage, annotation boundaries, and prompt behavior. A fresh sample
should be used for unbiased evaluation after the codebook is frozen.
