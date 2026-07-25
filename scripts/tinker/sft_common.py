"""Plan-T shared SFT machinery for the pred_dev arms (a = direct label, c = reasoning+label).

Everything that must be IDENTICAL across arms lives here, so a↔c isolates exactly one factor:
whether the SFT target contains reasoning. Only the data directory differs between the two
recipes (`sl_deviate_a.py` / `sl_deviate_c.py`).

Two things this module fixes that the cookbook defaults get wrong for us:

1. **Renderer = `qwen3_5_disable_thinking`, NOT the "recommended" `qwen3_5`.**
   `model_info.get_recommended_renderer_name("Qwen/Qwen3.6-35B-A3B")` returns `qwen3_5`
   (thinking mode). Qwen3.6 is a HYBRID model, and the two variants build the SAME supervised
   example but DIFFERENT generation prompts:
       qwen3_5                  -> prompt ends `<|im_start|>assistant\\n<think>\\n`
       qwen3_5_disable_thinking -> prompt ends `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`
   Training always renders the assistant turn with the EMPTY think block (our targets carry no
   reasoning_content), i.e. the trained prefix is the disable-thinking one. Sampling/scoring with
   the thinking renderer would therefore condition on a prefix the model never saw after SFT, and
   the teacher-forced P(deviate) readout would be measured off-distribution. So both training and
   `eval_prob_tinker.py` MUST use `qwen3_5_disable_thinking`.

2. **`train_on_what = LAST_ASSISTANT_MESSAGE`.** Our conversations are single-turn
   (system, user, assistant), so this is numerically identical to ALL_ASSISTANT_MESSAGES but
   avoids the cookbook's extension-property warning (qwen3_5 has `has_extension_property=False`).

Verified against the data (Qwen3.6 tokenizer, both arms):
  arm a  -> 3 trained tokens: 'deviate<|im_end|>'      (prompt ~6.2K tok)
  arm c  -> ~574 trained tokens: the full JSON trace ending '..."deviation": "deviate"}<|im_end|>'
  lengths: cls max 9157 tok / cls_reason max 10372 tok  => max_length 16384 never truncates.
  (Truncation would be silently fatal here: the label is the LAST token of the target.)
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import chz
import datasets

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.llm_experiment.env_loader import load_tinker_key  # noqa: E402

from tinker_cookbook import cli_utils, model_info  # noqa: E402
from tinker_cookbook.hyperparam_utils import get_lr  # noqa: E402
from tinker_cookbook.renderers import TrainOnWhat  # noqa: E402
from tinker_cookbook.supervised import train  # noqa: E402
from tinker_cookbook.supervised.data import (  # noqa: E402
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (  # noqa: E402
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)

BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
RENDERER_NAME = "qwen3_5_disable_thinking"  # see module docstring, point 1
MAX_LENGTH = 16384
TRAIN_ON_WHAT = TrainOnWhat.LAST_ASSISTANT_MESSAGE


def _load_conversations(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "messages" not in row:
                raise ValueError(f"{path}:{i} has no 'messages' key (got {list(row)})")
            rows.append({"messages": row["messages"]})
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


@chz.chz
class TwoFileConversationBuilder(ChatDatasetBuilder):
    """Like `FromConversationFileBuilder`, but reads train and val from SEPARATE files.

    The cookbook builder carves a held-out set out of one file with its own shuffle; we already
    have a LOCKED patient-level, disease-stratified seed-0 split (278/67/56, shared with
    `data/training_set/{cls,cls_reason,sft}`) and must not re-split it — a random re-split would
    leak patients across train/val. The returned val dataset is picked up by `train.main` as an
    `NLLEvaluator`, logging `test/nll` + `test/bpb` every `eval_every` steps.

    val_batch_size defaults to "all val rows in one batch": `NLLEvaluator.from_dataset`
    materialises every batch into a single forward call anyway, and `SupervisedDataset.__len__`
    floor-divides, so a smaller batch size would silently DROP the remainder (67 is prime).
    """

    train_path: str
    val_path: str | None = None
    val_batch_size: int | None = None

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )
        renderer = self.renderer

        def map_fn(row: dict):
            return conversation_to_datum(
                row["messages"], renderer, self.common_config.max_length, train_on_what
            )

        train_rows = _load_conversations(Path(self.train_path))
        train_ds = SupervisedDatasetFromHFDataset(
            datasets.Dataset.from_list(train_rows),
            batch_size=self.common_config.batch_size,
            map_fn=map_fn,
        )
        val_ds = None
        if self.val_path:
            val_rows = _load_conversations(Path(self.val_path))
            val_ds = SupervisedDatasetFromHFDataset(
                datasets.Dataset.from_list(val_rows),
                batch_size=self.val_batch_size or len(val_rows),
                map_fn=map_fn,
            )
        return train_ds, val_ds


@chz.chz
class DeviateSFTConfig:
    """CLI knobs shared by arm a and arm c. Defaults are the LOCKED Plan-T settings."""

    arm: str = "a"  # "a" (cls, direct label) | "c" (cls_reason, reasoning + label)
    data_dir: str = "data/training_set/tinker/cls"
    log_path: str | None = None

    model_name: str = BASE_MODEL
    renderer_name: str = RENDERER_NAME
    lora_rank: int = 32

    learning_rate: float = 2e-4
    lr_schedule: str = "linear"
    num_epochs: int = 3
    batch_size: int = 8
    max_length: int = MAX_LENGTH

    # 0 => auto (once per epoch for saves, twice per epoch for eval, incl. step 0)
    save_every: int = 0
    eval_every: int = 0
    max_steps: int | None = None

    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: str = "raise"


def run(cfg: DeviateSFTConfig) -> None:
    load_tinker_key()  # reads repo-root .tinker_env into TINKER_API_KEY (never `source` it)

    data_dir = Path(cfg.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    train_path, val_path = data_dir / "train.jsonl", data_dir / "val.jsonl"
    n_train = sum(1 for line in train_path.open() if line.strip())
    n_val = sum(1 for line in val_path.open() if line.strip())

    batches_per_epoch = n_train // cfg.batch_size  # cookbook floor-divides; remainder is dropped
    if batches_per_epoch == 0:
        raise ValueError(f"batch_size {cfg.batch_size} > n_train {n_train}")
    save_every = cfg.save_every or batches_per_epoch  # a checkpoint at each epoch boundary
    eval_every = cfg.eval_every or max(1, batches_per_epoch // 2)

    log_path = cfg.log_path or (
        f"runs/tinker/pred_dev_{cfg.arm}_"
        f"{cfg.lora_rank}rank-{cfg.learning_rate}lr-{cfg.batch_size}batch-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    if not Path(log_path).is_absolute():
        log_path = str(ROOT / log_path)

    recommended_renderer = model_info.get_recommended_renderer_name(cfg.model_name)
    print(
        "\n=== Plan-T pred_dev SFT ===\n"
        f"arm            : {cfg.arm}\n"
        f"data           : {data_dir}  (train {n_train} / val {n_val})\n"
        f"model          : {cfg.model_name}   lora_rank={cfg.lora_rank}\n"
        f"renderer       : {cfg.renderer_name}"
        f"   [cookbook 'recommended' = {recommended_renderer}; overridden on purpose,"
        f" see sft_common docstring]\n"
        f"train_on_what  : {TRAIN_ON_WHAT}   max_length={cfg.max_length}\n"
        f"lr             : {cfg.learning_rate}"
        f"   [cookbook get_lr(lora) = {get_lr(cfg.model_name, is_lora=True):.2e}]\n"
        f"schedule       : {cfg.lr_schedule}  epochs={cfg.num_epochs}  batch_size={cfg.batch_size}\n"
        f"steps          : {batches_per_epoch}/epoch x {cfg.num_epochs}"
        f" = {batches_per_epoch * cfg.num_epochs}"
        f"   (drops {n_train - batches_per_epoch * cfg.batch_size} rows/epoch to the floor)\n"
        f"save_every     : {save_every}   eval_every={eval_every}\n"
        f"log_path       : {log_path}\n"
    )

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.renderer_name,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        train_on_what=TRAIN_ON_WHAT,
    )
    dataset_builder = TwoFileConversationBuilder(
        common_config=common_config,
        train_path=str(train_path),
        val_path=str(val_path),
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    config = train.Config(
        log_path=log_path,
        model_name=cfg.model_name,
        recipe_name=f"pred_dev_sft_{cfg.arm}",
        renderer_name=cfg.renderer_name,
        dataset_builder=dataset_builder,
        learning_rate=cfg.learning_rate,
        lr_schedule=cfg.lr_schedule,
        num_epochs=cfg.num_epochs,
        lora_rank=cfg.lora_rank,
        save_every=save_every,
        eval_every=eval_every,
        max_steps=cfg.max_steps,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
    )
    asyncio.run(train.main(config))
    print(
        f"\nDONE. Checkpoints -> {log_path}/checkpoints.jsonl "
        "(grab the final `sampler_path` = the tinker:// path for eval_prob_tinker.py)"
    )
