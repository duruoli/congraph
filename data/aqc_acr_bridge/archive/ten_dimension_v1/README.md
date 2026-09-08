# Archived ten-dimension patient extraction (v1)

This folder freezes the superseded calibration artifacts built around a closed set of ten patient
dimensions and a 50-value ACR vocabulary. They are retained for provenance only and are not active
inputs to the current predicate-based ACR representation or the open patient-dimension workflow.

Contents:

- `acr_context_value_dimension_audit_v1.csv`: the historical 50-value/ten-dimension audit;
- `open_context_manual_v1/`: the 12-case manual extraction and mapping outputs based on that audit;
- `scripts/`: frozen preparation, compilation, mapping, and validation scripts for those artifacts.

The archived scripts preserve the historical implementation and paths. They are not maintained as
active pipeline entry points.
