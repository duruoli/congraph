"""Plan-T b-NEW RLVR entrypoint — GRPO on the 4-term process reward, warm-started from the BARE base.

Adapted from recipes/math_rl/train.py. Builds a cookbook rl.train.Config around DeviationDatasetBuilder
and runs the standard GRPO loop. No checkpoint => the policy starts from the bare instruct base, which
IS the b-NEW pre-RL reference — so pre↔post is the informative RL contrast (see the memory
bnew-reward-emergence-design). Post-RL is expected to collapse the verdict to 0/1 (it's a lookup on the
model's own rec+study fields); recover the graded probability with the SC readout at eval, not here.

Run with the TINKER env (see run-env-congraph memory):
  /opt/anaconda3/envs/tinker/bin/python scripts/tinker/rl_bnew.py \
      group_size=16 groups_per_batch=32 learning_rate=1e-5 max_tokens=768 \
      log_path=runs/tinker/rl_bnew wandb_project=congraph-preddev

Smoke (2 steps, tiny): add  max_steps=2 groups_per_batch=4 group_size=4  save_every=1 eval_every=1
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import chz

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "tinker")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from experiments.llm_experiment.env_loader import load_tinker_key  # noqa: E402
from deviation_env import DeviationDatasetBuilder  # noqa: E402
from sft_common import BASE_MODEL, RENDERER_NAME  # noqa: E402

from tinker_cookbook import cli_utils  # noqa: E402
from tinker_cookbook.rl.train import AsyncConfig, Config, main  # noqa: E402


@chz.chz
class CLIConfig:
    # model — bare base = b-NEW pre-RL; omit load_checkpoint_path to warm-start from it
    model_name: str = BASE_MODEL
    renderer_name: str = RENDERER_NAME
    lora_rank: int = 32
    load_checkpoint_path: str | None = None

    # data / reward
    data_dir: str = "data/training_set/cls_free"
    seed: int = 0
    n_epochs: int = 1        # train dataset passes (278 rows / groups_per_batch = batches per epoch)
    w_ans: float = 1.0
    w_dx: float = 0.25
    w_rec: float = 0.25
    w_std: float = 0.25
    w_len: float = 0.0002
    len_budget: int = 512

    # GRPO — group_size rollouts per prompt for in-group advantage; groups_per_batch prompts/step
    group_size: int = 16
    groups_per_batch: int = 32
    learning_rate: float = 1e-5
    max_tokens: int = 768        # reasoning + 4 fields; base median ~370 tok
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1
    max_steps: int | None = None
    max_steps_off_policy: int | None = None

    # logging / checkpointing
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    eval_every: int = 10
    save_every: int = 10
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cfg: CLIConfig):
    load_tinker_key()   # reads .tinker_env into TINKER_API_KEY (never `source` it)

    model_tag = cfg.model_name.replace("/", "-")
    run_name = (f"bnew-{model_tag}-{cfg.lora_rank}rank-{cfg.learning_rate}lr-"
                f"{cfg.group_size}group-{cfg.groups_per_batch}batch-seed{cfg.seed}-"
                f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}")
    log_path = cfg.log_path or f"runs/tinker/rl_bnew/{run_name}"

    dataset_builder = DeviationDatasetBuilder(
        batch_size=cfg.groups_per_batch, group_size=cfg.group_size,
        model_name_for_tokenizer=cfg.model_name, renderer_name=cfg.renderer_name,
        data_dir=cfg.data_dir, seed=cfg.seed, n_epochs=cfg.n_epochs,
        w_ans=cfg.w_ans, w_dx=cfg.w_dx, w_rec=cfg.w_rec, w_std=cfg.w_std,
        w_len=cfg.w_len, len_budget=cfg.len_budget)

    config = Config(
        learning_rate=cfg.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cfg.model_name,
        recipe_name="recipe_bnew_rl",
        renderer_name=cfg.renderer_name,
        lora_rank=cfg.lora_rank,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name or run_name,
        log_path=log_path,
        load_checkpoint_path=cfg.load_checkpoint_path,
        kl_penalty_coef=cfg.kl_penalty_coef,
        num_substeps=cfg.num_substeps,
        eval_every=cfg.eval_every,
        save_every=cfg.save_every,
        max_steps=cfg.max_steps,
        async_config=AsyncConfig(
            max_steps_off_policy=cfg.max_steps_off_policy,
            groups_per_batch=cfg.groups_per_batch,
        ) if cfg.max_steps_off_policy is not None else None,
    )
    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
