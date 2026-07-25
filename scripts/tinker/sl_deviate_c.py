"""Plan-T arm **c** — the target design: base -> SFT structured reasoning trace + deviation tail.

Target = the full JSON reasoning trace (belief / modality / information_gap / expected_finding /
grounding / other_hypothesis) + the rubric-reference block (rubric_recommended / rubric_state /
rubric_rationale) + `"deviation": follow|deviate` as the LAST key. Loss covers the whole assistant
turn (~574 tokens), so the reasoning dominates and the label is a small tail — the prob is a
readout on top of clinical reasoning, grounded against a STATED rubric reference rather than a
free-floating guess. Trained FROM BASE (not warm-started from a reasoning adapter) so that a↔c
differ only in target content.

  /opt/anaconda3/envs/tinker/bin/python scripts/tinker/sl_deviate_c.py
  # overrides (chz): ... learning_rate=1e-4 num_epochs=2 batch_size=4
  # smoke test     : ... max_steps=2 save_every=2 eval_every=2

Data: data/training_set/tinker/cls_reason/{train,val}.jsonl (278/67, same LOCKED split as arm a).
Read the prob afterwards with (note --generate-first — the model must produce its OWN reasoning,
which is then cut at the `"deviation"` key and the label scored conditioned on it):
  scripts/tinker/eval_prob_tinker.py --checkpoint <tinker://...> --arm-name c --generate-first \
      --data data/training_set/cls_reason --out results/agent_inspection/tinker_deviation_c
"""

from __future__ import annotations

import sys
from pathlib import Path

import chz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sft_common import DeviateSFTConfig, run  # noqa: E402


@chz.chz
class ArmCConfig(DeviateSFTConfig):
    arm: str = "c"
    data_dir: str = "data/training_set/tinker/cls_reason"


if __name__ == "__main__":
    run(chz.entrypoint(ArmCConfig))
