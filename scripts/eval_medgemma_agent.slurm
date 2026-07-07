#!/bin/bash

# 5-arm certainty-agent eval panel for the MEDGEMMA-27B SFT (scripts/eval_certainty_agent.py).
# Twin of scripts/eval_certainty_agent.slurm but pointed at the medgemma base+adapter and
# writing to *_medgemma.{txt,jsonl} so the committed Qwen panel (eval_panel.{txt,jsonl}) is
# preserved for a Qwen-vs-medgemma comparison.
# Submit from the repo root on Quest:
#   sbatch scripts/eval_medgemma_agent.slurm                     # default: doctor base sft
#   sbatch scripts/eval_medgemma_agent.slurm "doctor base sft"   # custom arm list (quote it)
# Inputs:  data/training_set/sft/test.jsonl  +  runs/medgemma-27b-lora-certainty/ (local adapter)
# Output:  results/agent_inspection/eval_panel_medgemma.{txt,jsonl}  (git-tracked -> push/pull home)
#
# ctx-sonnet is NOT in the default (external OpenRouter net; compute nodes usually offline).
# NOTE the arm named "ctx-qwen" in the code means "base weights + context block" — with
# --base-model=medgemma it injects context on the medgemma base, despite the qwen-era name.

#SBATCH --account=p31777
#SBATCH --partition=gengpu                        # ⚠️ CONFIRM this partition has the GPU below
#SBATCH --gres=gpu:h100:1                          # 27B bf16 base ~54GB -> needs an 80GB card
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                                 # 27B checkpoint shards ~54GB + cache
#SBATCH --time=01:30:00                            # inference-only; headroom for 27B load + 3 arms
#SBATCH --job-name=eval_medgemma_agent
#SBATCH --output=runs/eval_medgemma_%j.out
#SBATCH --error=runs/eval_medgemma_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=duruoli2024@u.northwestern.edu

# ============================================
# Environment Setup  (identical to train_medgemma_sft.slurm)
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
# Hugging Face cache / offline weights  (medgemma is GATED — pre-download on a login node)
# ============================================
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1                            # comment out if the node HAS net
export TOKENIZERS_PARALLELISM=false

# ============================================
# Eval
# ============================================
ARMS="${1:-doctor base sft}"
echo "arms = $ARMS"

python scripts/eval_certainty_agent.py \
    --arms $ARMS \
    --base-model google/medgemma-27b-text-it \
    --adapter runs/medgemma-27b-lora-certainty \
    --data data/training_set/sft/test.jsonl \
    --out results/agent_inspection/eval_panel_medgemma.txt

echo "============================================"
echo "Job completed at: $(date)"
echo "============================================"
