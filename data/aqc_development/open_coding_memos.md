# Track B first-cycle open-coding memos

## Source boundary

The coding source is the existing rubric-free, order-aware annotation corpus only. ACR v1.1 was not
loaded, matched, or used to define a type. The copied fields in `discovery_open_coding.jsonl` remain
verbatim; `open_type_codes` are a new, provisional analytical layer and do not replace the source.

## Sample

The fixed maximum-variation sample contains 16 trajectories and 48 decision steps, exactly four
trajectories from each disease stratum. It contains 1–7-step trajectories, repeats and switches,
and all five old action roles. Local verification outcomes include 28 confirmed, 17 disconfirmed,
and 3 uninformative steps. Those outcome labels were used to check sample diversity only; they are
not assumption codes and must not enter the ex-ante pilot input.

## Empirical clusters

Five seed families recurred: disease/finding identity, etiology/mechanism, severity, complication,
and alternative source. Three changes were needed after reading the source wording:

1. **Broad syndrome or source frame.** Several early CT decisions were organized by “there is a
   serious intra-abdominal process, but its organ source is unknown.” Coding these as a named
   disease would invent specificity.
2. **Intervention or device state.** Post-ERCP and drained-abscess trajectories repeatedly asked
   whether a stent or drain was positioned, patent, decompressing, or effective. This is not fully
   captured by disease etiology or biological complication.
3. **Finding identity.** Some workups asked what an already observed collection, lymph-node burden,
   or echogenic structure represented. The disease-existence seed was therefore broadened to
   `disease_or_finding_identity`.

Longitudinal response and progression recur frequently, but they are retained as attributes within
`severity_extent_or_course`, not promoted to another top-level type. A distinct new adverse entity
such as perforation, necrosis, or abscess is additionally coded as `complication`.

## Important non-equivalences

- A broad differential is not automatically `alternative_source`; before a focal frame exists it
  may be `syndrome_or_source_frame`.
- A nested cause of a retained disease is `etiology_or_mechanism`, not `alternative_source`.
- Stent failure is `intervention_or_device_state`; a bile leak or post-procedural abscess is a
  `complication`. Both may be present.
- Known disease plus suspected complication requires two propositions with two statuses. One
  certainty label for the whole step loses the hierarchy.
- A valid negative can challenge or exclude a proposition. Nondiagnostic imaging, nonvisualization,
  and anatomy not assessed cannot do so by themselves.

## Over-rationalization audit target

The old source was generated with the actual action visible. It is intentionally useful for
reconstructing intent, but may make an order appear more coherent than the chart supports. The
paired pilot therefore keeps two independent arms:

- direct A/Q/C from the masked chart plus actual order, without the old reasoning;
- recoding of the old schema-free reasoning, without the current result or verification.

Disagreement is not automatically error. Reviewers should distinguish missing information,
different granularity, and unsupported rationale. `other`, `unclear`, weak support, alternative
reconstructions, and an explicit unsupported residual must remain available.

## Freeze criterion

This codebook is not frozen. The existing 16-trajectory sample is a preliminary prototype and the
formal first codebook batch should start from approximately 24 trajectories total selected from the
full-corpus development partition. Twenty-four is not a fixed stopping number: add fresh,
non-overlapping development batches if they reveal new top-level types, recurrent answer
requirements, or systematic `other/unclear` clusters. Check revisions on further unused development
cases. Freeze only after fresh-case qualitative saturation, residual review, and causal-leakage
checks. Do not use the final-test partition to revise the codebook.

Before bulk annotation, compare the two independent annotation routes on unused development cases:
direct construction from the causally masked chart plus actual order, and conversion of the old
schema-light reasoning. Compare atomic propositions, type/status, question, requirements, and
coverage, and audit over-rationalization. Close agreement may justify reusing the old annotations as
the main bulk source for A and Q, but requirement-level C still needs causally available patient
evidence and the original annotations must remain unchanged.
