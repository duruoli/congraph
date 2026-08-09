"""Plan-T b-NEW RL environment (Tinker cookbook ProblemEnv) — the 4-term process reward.

A single-turn env: system(rubric+instructions) + user(patient state) -> the model emits free
reasoning + the four labelled lines, and we score it with rl_reward.bnew_reward (verdict DOMINANT +
diagnosis + rubric-rec + predicted-study, each correctness-checked; strict format gate). See
results/tinker/RESULTS_bnew_reward_emergence.md for the design.

Mirrors recipes/math_rl/math_env.py (DeviationEnv ~ MathEnv, DeviationDataset ~ MathDataset,
DeviationDatasetBuilder ~ MathDatasetBuilder) but:
  - the prompt is our two-message {system, user} from data/training_set/cls_free (system carried as
    convo_prefix so build_generation_prompt reproduces the eval/generation prompt exactly);
  - step() is OVERRIDDEN to return the composite 4-term reward instead of ProblemEnv's binary
    correct+format. check_answer/check_format/get_reference_answer stay (abstract) for logging.

Run with the TINKER env (see run-env-congraph memory):
  /opt/anaconda3/envs/tinker/bin/python scripts/tinker/rl_bnew.py ...
"""
from __future__ import annotations

import json
import math
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Literal

import chz

from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, Metrics, RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter

import tinker

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "tinker")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from rl_reward import (  # noqa: E402
    ANSWER_CUE_DEFAULT, bnew_reward, gold_rec_set, parse_dx, parse_prediction, parse_rec_set,
    parse_study,
)
from sft_common import BASE_MODEL, RENDERER_NAME  # noqa: E402


class DeviationEnv(ProblemEnv):
    """One b-NEW decision step. `meta` holds the four golds (y / effective_branch /
    rubric_recommended / how_modality) the reward reads."""

    def __init__(
        self,
        system: str,
        user: str,
        meta: dict,
        renderer: renderers.Renderer,
        w_ans: float = 1.0,
        w_dx: float = 0.25,
        w_rec: float = 0.25,
        w_std: float = 0.25,
        w_len: float = 0.0002,
        len_budget: int = 512,
    ):
        # carry the system message as the conversation prefix so the generation prompt matches
        # exactly what eval_prob_tinker / gen_bnew_traces build from messages[:2].
        super().__init__(renderer, convo_prefix=[{"role": "system", "content": system}])
        self.user = user
        self.meta = meta
        self.weights = dict(w_ans=w_ans, w_dx=w_dx, w_rec=w_rec, w_std=w_std,
                            w_len=w_len, len_budget=len_budget)

    def get_question(self) -> str:
        return self.user

    # --- the three abstract hooks: used ONLY for logging metrics, not for the reward total ---
    def check_format(self, sample_str: str) -> bool:
        return ANSWER_CUE_DEFAULT in sample_str and parse_prediction(sample_str, ANSWER_CUE_DEFAULT) is not None

    def check_answer(self, sample_str: str) -> bool:
        pred = parse_prediction(sample_str, ANSWER_CUE_DEFAULT) if ANSWER_CUE_DEFAULT in sample_str else None
        gold_word = "deviate" if int(self.meta["y"]) == 1 else "follow"
        return pred == gold_word

    def get_reference_answer(self) -> str:
        m = self.meta
        return (f"answer={'deviate' if int(m['y']) == 1 else 'follow'} dx={m.get('effective_branch')} "
                f"rec={m.get('rubric_recommended')} study={m.get('how_modality')}")

    async def step(self, action, *, extra=None) -> StepResult:
        """Score with the composite 4-term reward (overrides ProblemEnv's binary correct+format)."""
        convo = self.convo_prefix + [{"role": "user", "content": self.get_question()}]
        message, _termination = self.renderer.parse_response(action)
        content = renderers.get_text_content(message) or ""   # never None -> reward parsers are str-safe
        n_tokens = len(action) if hasattr(action, "__len__") else None

        total, bd = bnew_reward(content, self.meta, n_tokens=n_tokens, return_breakdown=True,
                                **self.weights)

        with logtree.scope_header("Prompt"):
            logtree.log_formatter(ConversationFormatter(messages=convo))
        with logtree.scope_header("Policy Response"):
            logtree.log_formatter(ConversationFormatter(messages=[message]))
        with logtree.scope_header("Reward"):
            logtree.table_from_dict(
                {"reference": self.get_reference_answer(), **{k: bd.get(k) for k in (
                    "pred", "gated", "ans_match", "dx", "dx_match", "rec", "rec_match",
                    "study", "std_match")}, "reward": f"{total:.3f}"},
                caption="4-term reward")

        metrics: Metrics = {
            "reward": total, "gated": float(bd.get("gated", False)),
            "ans_match": float(bd.get("ans_match", 0.0)), "dx_match": float(bd.get("dx_match", 0.0)),
            "rec_match": float(bd.get("rec_match", 0.0)), "std_match": float(bd.get("std_match", 0.0)),
        }
        return StepResult(
            reward=total, episode_done=True, next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition, metrics=metrics)


class DeviationDataset(RLDataset):
    """Batches cls_free rows into groups. Each row -> one ProblemGroupBuilder of `group_size`
    identical envs (GRPO samples `group_size` rollouts of the same prompt for in-group advantage).

    Supports MULTIPLE epochs: __len__ = n_epochs * batches_per_epoch, and get_batch wraps around,
    RESHUFFLING with a per-epoch seed. Without this the dataset yields only ceil(N/batch_size)
    batches (1 epoch) and max_steps beyond that has no effect."""

    def __init__(self, rows: list[dict], batch_size: int, group_size: int,
                 renderer: renderers.Renderer, weights: dict, shuffle: bool, seed: int,
                 n_epochs: int = 1):
        self.base_rows = list(rows)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.weights = weights
        self.shuffle = shuffle
        self.seed = seed
        self.n_epochs = n_epochs
        self.batches_per_epoch = math.ceil(len(self.base_rows) / batch_size)

    def __len__(self) -> int:
        return self.n_epochs * self.batches_per_epoch

    def _epoch_rows(self, epoch: int) -> list[dict]:
        rows = list(self.base_rows)
        if self.shuffle:
            import random
            random.Random(self.seed + epoch).shuffle(rows)   # a fresh order each epoch
        return rows

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        epoch, within = divmod(index, self.batches_per_epoch)
        rows = self._epoch_rows(epoch)
        lo = within * self.batch_size
        hi = min(lo + self.batch_size, len(rows))
        assert lo < hi, "empty batch — check batch_size vs dataset length"
        out = []
        for r in rows[lo:hi]:
            sys_msg, user_msg = r["messages"][0]["content"], r["messages"][1]["content"]
            out.append(ProblemGroupBuilder(
                env_thunk=partial(DeviationEnv, sys_msg, user_msg, r["meta"], self.renderer,
                                  **self.weights),
                num_envs=self.group_size,
                dataset_name="deviation_bnew"))
        return out


@chz.chz
class DeviationDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str = BASE_MODEL
    renderer_name: str = RENDERER_NAME
    data_dir: str = "data/training_set/cls_free"
    seed: int = 0
    n_epochs: int = 1                    # train dataset cycles this many reshuffled passes
    # reward weights (flow through to every env)
    w_ans: float = 1.0
    w_dx: float = 0.25
    w_rec: float = 0.25
    w_std: float = 0.25
    w_len: float = 0.0002
    len_budget: int = 512

    def _load(self, split: str) -> list[dict]:
        p = ROOT / self.data_dir / f"{split}.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    async def __call__(self) -> tuple[DeviationDataset, DeviationDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        weights = dict(w_ans=self.w_ans, w_dx=self.w_dx, w_rec=self.w_rec, w_std=self.w_std,
                       w_len=self.w_len, len_budget=self.len_budget)
        train = DeviationDataset(self._load("train"), self.batch_size, self.group_size, renderer,
                                 weights, shuffle=True, seed=self.seed, n_epochs=self.n_epochs)
        # test: group_size 1 (eval samples once per prompt), always a single pass
        test = DeviationDataset(self._load("test"), self.batch_size, 1, renderer, weights,
                                shuffle=False, seed=self.seed, n_epochs=1)
        logger.info(f"DeviationDataset: train {len(train.base_rows)} rows / {len(train)} batches "
                    f"({self.n_epochs} epochs), test {len(test.base_rows)} rows; "
                    f"group_size {self.group_size}")
        return train, test
