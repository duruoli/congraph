#!/bin/bash

# 5-arm certainty-agent eval panel (scripts/eval_certainty_agent.py).
# Submit from the repo root on Quest:
#   sbatch scripts/eval_certainty_agent.slurm                         # default 4 offline arms
#   sbatch scripts/eval_certainty_agent.slurm "doctor base sft"       # custom arm list (quote it)
# Inputs:  data/training_set/sft/test.jsonl  +  runs/qwen2.5-7b-lora-certainty/ (local adapter)
# Output:  results/agent_inspection/eval_panel.{txt,jsonl}  (git-tracked -> push/pull home)
#
# DEFAULT ARMS = doctor base sft ctx-qwen  (all run OFFLINE from cached base + local adapter).
# ctx-sonnet is NOT in the default: it calls OpenRouter (external net), which Quest compute
# nodes usually lack. To include it you must (a) have net on the node OR run that arm at home,
# (b) comment out HF_HUB_OFFLINE below, and (c) have the OpenRouter key available to
# experiments.llm_experiment.env_loader.load_openrouter_key.

#SBATCH --account=p31777
#SBATCH --partition=gengpu                       # Quest general GPU partition (A100s)
#SBATCH --gres=gpu:a100:1                         # one A100; eval is inference-only
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00                           # ~10-20 min for 56 rows x 3 model arms; headroom
#SBATCH --job-name=eval_certainty_agent
#SBATCH --output=runs/eval_agent_%j.out
#SBATCH --error=runs/eval_agent_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=duruoli2024@u.northwestern.edu

# ============================================
# Environment Setup  (identical to train_lora_qwen.slurm)
# ============================================
module purge all
# do NOT `module load cuda` — the pip torch wheel bundles its own CUDA runtime; the system
# cuda module drags in an ancient gcc whose libstdc++.so.6 lacks CXXABI_1.3.8 -> import torch dies.
source "$HOME/miniconda/bin/activate"
conda activate congraph
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /home/dlf8982/AAA/congraph
mkdir -p runs results/agent_inspection

echo "============================================"
echo "Job started at: $(date)"
echo "Node: $(hostname)   Job ID: $SLURM_JOB_ID"
echo "============================================"
python --version
nvidia-smi
echo "============================================"

# ============================================
# Hugging Face cache / offline weights
# ============================================
# Pre-download the base ONCE on a login node (see train_lora_qwen.slurm notes), then run offline.
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1                            # comment out if the node HAS net OR you add ctx-sonnet
export TOKENIZERS_PARALLELISM=false

# ============================================
# Eval
# ============================================
# Uses the LOCAL adapter dir (no HF network needed for the LoRA). Arms overridable via $1.
ARMS="${1:-doctor base sft ctx-qwen}"
echo "arms = $ARMS"

python scripts/eval_certainty_agent.py \
    --arms $ARMS \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter runs/qwen2.5-7b-lora-certainty \
    --data data/training_set/sft/test.jsonl \
    --out results/agent_inspection/eval_panel.txt

echo "============================================"
echo "Job completed at: $(date)"
echo "============================================"
