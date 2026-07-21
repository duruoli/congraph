#!/bin/bash

# VAL pass of the MEDGEMMA-27B reasoning SFT — the calibration input for pred_dev approach B.
# Twin of scripts/eval_medgemma_agent.slurm, pointed at the VAL split and writing to
# *_medgemma_val.{txt,jsonl}. Approach B derives P(deviate)=1-follow_prob from this adapter and
# needs val follow_probs to fit Platt (test follow_probs already exist in eval_panel_medgemma.jsonl).
# After this finishes, run (CPU, no model):
#   python scripts/eval_dev_from_reasoning.py \
#       --val-panel  results/agent_inspection/eval_panel_medgemma_val.jsonl \
#       --test-panel results/agent_inspection/eval_panel_medgemma.jsonl --arms sft base
# Submit from the repo root on Quest:
#   sbatch scripts/eval_medgemma_agent_val.slurm                 # default: doctor base sft
# Inputs:  data/training_set/sft/val.jsonl  +  runs/medgemma-27b-lora-certainty/ (local adapter)
# Output:  results/agent_inspection/eval_panel_medgemma_val.{txt,jsonl}  (git-tracked)

#SBATCH --account=p31777
#SBATCH --partition=gengpu                        # ⚠️ CONFIRM this partition has the GPU below
#SBATCH --gres=gpu:h100:1                          # 27B bf16 base ~54GB -> needs an 80GB card
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                                 # 27B checkpoint shards ~54GB + cache
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
# Hugging Face cache / offline weights  (medgemma is GATED — pre-download on a login node)
# ============================================
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1                            # comment out if the node HAS net
export TOKENIZERS_PARALLELISM=false

# ============================================
# Eval  (VAL split — produces the follow_prob column used to Platt-calibrate approach B)
# ============================================
ARMS="${1:-doctor base sft}"
echo "arms = $ARMS"

python scripts/eval_certainty_agent.py \
    --arms $ARMS \
    --base-model google/medgemma-27b-text-it \
    --adapter runs/medgemma-27b-lora-certainty \
    --data data/training_set/sft/val.jsonl \
    --out results/agent_inspection/eval_panel_medgemma_val.txt

echo "============================================"
echo "Job completed at: $(date)"
echo "============================================"
