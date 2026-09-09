# Manual patient-item to ACR mapping: pilot v1

This directory maps every item in `manual_extraction_v1` to the compiled ACR
predicate-instance registry. The mapping is a separate pass from extraction and
does not assign an ACR Variant.

## Mapping rules

- Direction is always the patient item relative to the ACR predicate.
- One patient item may link to equivalent instances in multiple Variants.
- Component findings use `related_judgment_required` when a clinical synthesis
  is needed to reach an ACR diagnostic or aggregate predicate.
- Missing evidence is not mapped as contradiction.
- A direct negative at compatible scope can use `contradicted`.
- A seed-dimension item with no comparable ACR instance is retained under
  `unmapped_value_within_dimension`.
- An item extracted through an open proposed dimension remains under
  `other_proposed_dimension`; it is not relabelled as an unmapped seed value.

The hand-authored semantic rules are implemented in
`scripts/map_manual_patient_context_to_acr.py`. Output links include the exact
ACR predicate-instance ID, source surface text, normalized predicate, Variant
key/text, relation, and a case-level rationale.

## Boundary

These outputs are item-to-predicate mappings only. They do not evaluate full
Boolean Variant satisfaction, resolve aggregate hierarchies, rank candidates,
or reveal observed actions and ratings.
