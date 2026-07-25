"""Plan-T arm **a** — the FLOOR: base -> SFT the deviation label directly, NO reasoning.

Target = ONE word (`follow` / `deviate`), so the model learns a direct input->label shortcut and
the resulting probability has no reasoning behind it. Contrast a↔c isolates "does putting
structured reasoning IN the SFT target help the calibrated deviation prob?".

  /opt/anaconda3/envs/tinker/bin/python scripts/tinker/sl_deviate_a.py
  # overrides (chz): ... learning_rate=1e-4 num_epochs=2 batch_size=4
  # smoke test     : ... max_steps=2 save_every=2 eval_every=2

Data: data/training_set/tinker/cls/{train,val}.jsonl (278/67, LOCKED seed-0 patient-level split).
Read the prob afterwards with:
  scripts/tinker/eval_prob_tinker.py --checkpoint <tinker://...> --arm-name a \
      --data data/training_set/cls --out results/agent_inspection/tinker_deviation_a
(no --generate-first: for arm a the answer slot is immediately after the generation prompt).
"""

from __future__ import annotations

import sys
from pathlib import Path

import chz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sft_common import DeviateSFTConfig, run  # noqa: E402


@chz.chz
class ArmAConfig(DeviateSFTConfig):
    arm: str = "a"
    data_dir: str = "data/training_set/tinker/cls"


if __name__ == "__main__":
    run(chz.entrypoint(ArmAConfig))
