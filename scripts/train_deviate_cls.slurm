#!/bin/bash

# Approach A (pred_dev): FULL-QUALITY (bf16, no quantization) LoRA fine-tune of
# Qwen/Qwen3-30B-A3B-Instruct-2507 (MoE) on the BARE deviation label (target = one word
# follow|deviate, NO reasoning). The floor / ablation for HANDOFF_pred_dev.md's a-vs-b-vs-c.
#
# Submit from the repo root on Quest:   sbatch scripts/train_deviate_cls.slurm
# Inputs: data/training_set/cls/{train,val}.jsonl (built by scripts/build_deviation_cls.py).
# Output: runs/qwen3-30b-a3b-lora-deviate-cls/  (the LoRA adapter + tokenizer).

#SBATCH --account=p31777
#SBATCH --partition=gengpu                          # ⚠️ CONFIRM H100s here: sinfo -p gengpu -o "%P %G %N"
#SBATCH --gres=gpu:h100:1                           # 80GB H100: ~61GB bf16 weights fit; --load-4bit if OOM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=qwen3_deviate_cls
#SBATCH --output=runs/qwen3_deviate_cls_%j.out
#SBATCH --error=runs/qwen3_deviate_cls_%j.err
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

# Qwen3 is NOT gated (medgemma was). One-time on a LOGIN node if compute nodes are offline:
#   export HF_HOME=$HOME/.cache/huggingface
#   python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-30B-A3B-Instruct-2507')"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1                             # comment out if compute nodes have net
export TOKENIZERS_PARALLELISM=false

# ============================================
# Train  (FULL QUALITY: bf16, no --load-4bit; add --load-4bit if the 30B MoE OOMs)
# ============================================
# Target is a single word, so the assistant span is tiny (assistant_only_loss still masks it
# correctly; watch for [chat-compat] swapped in ... + assistant mask OK). MoE -> LoRA on
# attention only ([lora] target_modules = [q/k/v/o_proj]). Prompt dominates -> --max-len 12288.
python scripts/train_lora_qwen.py \
    --base Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --data data/training_set/cls \
    --out runs/qwen3-30b-a3b-lora-deviate-cls \
    --epochs 3 --batch 1 --grad-accum 16 --lr 2e-4 \
    --max-len 12288

echo "============================================"
echo "Job completed at: $(date)"
echo "============================================"
