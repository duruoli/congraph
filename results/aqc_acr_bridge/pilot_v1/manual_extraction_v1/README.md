# Manual text-complement extraction: pilot v1

This directory contains a manually curated extraction baseline for the frozen
12-step AQC--ACR bridge pilot. It is intentionally separate from the API model
outputs under `openai__gpt-5.1/`.

## Scope

- Unit: one frozen imaging decision step.
- Input: the rendered history, physical-examination prose, and visible prior
  imaging reports in the corresponding API-run `inputs/*.json` artifact.
- Output: the same `3.0.0-hybrid-text-context` extraction contract used by the
  API run, stored directly as JSON for field-level comparison.
- Extraction is sparse and clinically selective. Incidental findings with no
  material role in the decision context are generally omitted.
- Every evidence `support` is an exact substring of the named visible source.
- Every extracted item is justified only by the rendered patient-record input;
  no A/Q/C, ACR, current-order, current-report, or later-outcome field is copied
  into the output or accepted as evidence.

Per explicit study-direction for this debugging pass, the HPI leakage preflight
is not applied. Text present in the rendered HPI is treated as visible even when
it may restate information that a later causal audit would remove. Consequently,
these files are a manual extraction/debugging baseline, not yet a causally
validated study artifact.

The deterministic laboratory/vital/prior-test stream is not merged here.
Laboratory or vital facts are included only when they occur in the visible text
given to the text-complement extractor.
