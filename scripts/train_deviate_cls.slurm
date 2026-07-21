#!/bin/bash

# Approach A (pred_dev): FULL-QUALITY (bf16, no quantization) LoRA fine-tune of
# google/medgemma-27b-text-it on the BARE deviation label (target = one word follow|deviate,
# NO reasoning). The floor / ablation for HANDOFF_pred_dev.md's a-vs-b-vs-c comparison.
#
# Submit from the repo root on Quest:   sbatch scripts/train_deviate_cls.slurm
# Inputs: data/training_set/cls/{train,val}.jsonl (built by scripts/build_deviation_cls.py).
# Output: runs/medgemma-27b-lora-deviate-cls/  (the LoRA adapter + tokenizer).

#SBATCH --account=p31777
#SBATCH --partition=gengpu                          # ⚠️ CONFIRM H100s here: sinfo -p gengpu -o "%P %G %N"
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=medgemma_deviate_cls
#SBATCH --output=runs/medgemma_deviate_cls_%j.out
#SBATCH --error=runs/medgemma_deviate_cls_%j.err
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
mkdir -p runs

echo "============================================"
echo "Job started at: $(date)"
echo "Node: $(hostname)   Job ID: $SLURM_JOB_ID"
echo "============================================"
python --version
nvidia-smi
echo "============================================"

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ============================================
# Train  (FULL QUALITY: no --load-4bit)
# ============================================
# Target is a single word, so the assistant span is tiny (assistant_only_loss still masks it
# correctly; watch for [chat-compat] assistant mask OK). Prompt still dominates -> --max-len 12288.
python scripts/train_lora_qwen.py \
    --base google/medgemma-27b-text-it \
    --data data/training_set/cls \
    --out runs/medgemma-27b-lora-deviate-cls \
    --epochs 3 --batch 1 --grad-accum 16 --lr 2e-4 \
    --max-len 12288

echo "============================================"
echo "Job completed at: $(date)"
echo "============================================"
