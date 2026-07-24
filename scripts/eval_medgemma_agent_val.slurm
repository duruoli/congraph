#!/bin/bash

# VAL pass of the QWEN3-30B reasoning SFT — the calibration input for pred_dev approach B.
# Twin of scripts/eval_medgemma_agent.slurm, pointed at the VAL split and writing to
# *_qwen3_val.{txt,jsonl}. Approach B derives P(deviate)=1-follow_prob from this adapter and
# needs val follow_probs to fit Platt (test follow_probs come from eval_panel_qwen3.jsonl).
# NOTE: train_medgemma_sft.slurm ALREADY auto-runs the val pass + the B deriver after training —
# use this standalone slurm only to RE-RUN the val panel without retraining.
# After this finishes, run (CPU, no model):
#   python scripts/eval_dev_from_reasoning.py \
#       --val-panel  results/agent_inspection/eval_panel_qwen3_val.jsonl \
#       --test-panel results/agent_inspection/eval_panel_qwen3.jsonl --arms sft base
# Submit from the repo root on Quest:
#   sbatch scripts/eval_medgemma_agent_val.slurm                 # default: doctor base sft
# Inputs:  data/training_set/sft/val.jsonl  +  runs/qwen3-30b-a3b-lora-certainty/ (local adapter)
# Output:  results/agent_inspection/eval_panel_qwen3_val.{txt,jsonl}  (git-tracked)

#SBATCH --account=p31777
#SBATCH --partition=gengpu                        # ⚠️ CONFIRM this partition has the GPU below
#SBATCH --gres=gpu:h100:1                          # 30B bf16 base ~61GB -> needs an 80GB card (inference)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G                                 # 30B checkpoint shards ~61GB + cache
#SBATCH --time=01:30:00                            # inference-only; val=67 rows x arms
#SBATCH --job-name=eval_medgemma_val
#SBATCH --output=runs/eval_medgemma_val_%j.out
#SBATCH --error=runs/eval_medgemma_val_%j.err
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
# Hugging Face cache / offline weights  (Qwen3 is NOT gated — pre-download on a login node)
# ============================================
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1                            # comment out if the node HAS net
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # combat fragmentation (see OOM 2026-07-21)

# ============================================
# Eval  (VAL split — produces the follow_prob column used to Platt-calibrate approach B)
# ============================================
ARMS="${1:-doctor base sft}"
echo "arms = $ARMS"

python scripts/eval_certainty_agent.py \
    --arms $ARMS \
    --base-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapter runs/qwen3-30b-a3b-lora-certainty \
    --data data/training_set/sft/val.jsonl \
    --out results/agent_inspection/eval_panel_qwen3_val.txt

echo "============================================"
echo "Job completed at: $(date)"
echo "============================================"
