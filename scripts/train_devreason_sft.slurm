#!/bin/bash

# Approach C (pred_dev): FULL-QUALITY (bf16, no quantization) LoRA fine-tune of
# Qwen/Qwen3-30B-A3B-Instruct-2507 (MoE) on the REASONING + deviation-tail target.
# Target = clinical reasoning trace + a final "deviation": follow|deviate key -> the deviation
# probability is a readout conditioned on the model's own generated reasoning (not a shortcut).
# FROM BASE (not from the reasoning-tuned adapter) -> clean a-vs-c ablation. See HANDOFF_pred_dev.md.
#
# Submit from the repo root on Quest:   sbatch scripts/train_devreason_sft.slurm
# Inputs: data/training_set/cls_reason/{train,val}.jsonl (built by scripts/build_devreason.py).
# Output: runs/qwen3-30b-a3b-lora-devreason/  (the LoRA adapter + tokenizer).

#SBATCH --account=p31777
#SBATCH --partition=gengpu                         # ⚠️ CONFIRM H100s here: sinfo -p gengpu -o "%P %G %N"
#SBATCH --gres=gpu:h100:2                           # 2x80GB H100: ~61GB bf16 weights shard across both
                                                   # (device_map=auto) -> comfortable headroom, guaranteed
                                                   # fit. Single-GPU alt: set :1 and add --load-4bit.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                         # ONE process (model-parallel), NOT torchrun/DDP
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G                                  # host RAM: 30B shards ~61GB materialized on CPU at load
#SBATCH --time=08:00:00
#SBATCH --job-name=qwen3_devreason
#SBATCH --output=runs/qwen3_devreason_%j.out
#SBATCH --error=runs/qwen3_devreason_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=duruoli2024@u.northwestern.edu

# ============================================
# Environment Setup  (identical to train_medgemma_sft.slurm)
# ============================================
module purge all
# Do NOT `module load cuda` — the pip torch wheel bundles its own CUDA runtime; the system cuda
# module drags in an ancient gcc whose libstdc++.so.6 lacks CXXABI_1.3.8 -> import torch dies.
source "$HOME/miniconda/bin/activate"
conda activate congraph                            # torch(+cuda) trl>=0.12 peft>=0.13 transformers>=4.50
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

# ============================================
# Hugging Face — offline compute nodes (Qwen3 is NOT gated; no license/login needed)
# ============================================
# One-time on a LOGIN node if compute nodes are offline, pre-download:
#   export HF_HOME=$HOME/.cache/huggingface
#   python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-30B-A3B-Instruct-2507')"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1                             # comment out if compute nodes have net
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # combat fragmentation across the 2 GPUs

# ============================================
# Train  (FULL QUALITY: bf16, no --load-4bit; add --load-4bit if the 30B MoE OOMs)
# ============================================
# Same trainer/flags as the reasoning SFT — only --data and --out differ. The assistant target is
# LONGER than the reasoning-only trace (adds the deviation tail) but still << the ~10.2k prompt, so
# --max-len 12288 stays safe; the script's len-guard will confirm. MoE -> LoRA on attention only.
# Watch for:  [chat-compat] swapped in ... + assistant mask OK — N/M tokens ...
python scripts/train_lora_qwen.py \
    --base Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --data data/training_set/cls_reason \
    --out runs/qwen3-30b-a3b-lora-devreason \
    --epochs 3 --batch 1 --grad-accum 16 --lr 2e-4 \
    --max-len 12288
status=$?
if [ $status -ne 0 ]; then echo "TRAIN FAILED (exit $status) — skipping eval"; exit $status; fi

# ============================================
# Eval  (auto-chained: runs in THIS job right after training, same 2-GPU allocation)
# approach C: --generate-first (generate the reasoning trace, THEN score follow/deviate
# conditioned on it); Platt on VAL -> TEST.
# ============================================
echo "=== training done; starting approach-C eval ==="
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
