#!/bin/bash

# Approach C (pred_dev) eval: read a calibrated P(deviate) from the REASONING+deviation-tail
# Qwen3-30B, then metrics vs the observed binary label. Uses --generate-first (generate the
# reasoning trace, then score the follow/deviate token conditioned on it). Platt (a,b) fit on
# VAL, applied to TEST. See HANDOFF_pred_dev.md.
# NOTE: train_devreason_sft.slurm ALREADY auto-runs this eval after training — use this
# standalone slurm only to RE-RUN eval without retraining.
# Submit from the repo root on Quest:  sbatch scripts/eval_devreason.slurm
# Inputs:  data/training_set/cls_reason/{val,test}.jsonl  +  runs/qwen3-30b-a3b-lora-devreason/
# Output:  results/agent_inspection/deviation_cls_eval_devreason.{txt,json}  (git-tracked)

#SBATCH --account=p31777
#SBATCH --partition=gengpu                          # ⚠️ CONFIRM this partition has the GPU below
#SBATCH --gres=gpu:h100:1                            # 30B bf16 base ~61GB -> needs an 80GB card (inference)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00                              # inference + per-row generation (val+test=123)
#SBATCH --job-name=eval_devreason
#SBATCH --output=runs/eval_devreason_%j.out
#SBATCH --error=runs/eval_devreason_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=duruoli2024@u.northwestern.edu

# ============================================
# Environment Setup  (identical to train_medgemma_sft.slurm)
# ============================================
module purge all
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

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1                              # comment out if the node HAS net
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # combat fragmentation (see OOM 2026-07-21)

# ============================================
# Eval  (approach C: --generate-first; arm = sft = the devreason adapter)
# ============================================
python scripts/eval_deviation_cls.py \
    --base Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapter runs/qwen3-30b-a3b-lora-devreason \
    --data data/training_set/cls_reason \
    --arms sft \
    --generate-first --max-new-tokens 512 \
    --out results/agent_inspection/deviation_cls_eval_devreason

echo "============================================"
echo "Job completed at: $(date)"
echo "============================================"
