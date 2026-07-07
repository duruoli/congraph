#!/bin/bash

# Direction-B SFT: FULL-QUALITY (bf16, no quantization) LoRA fine-tune of
# google/medgemma-27b-text-it (Gemma-3, medical-tuned) on the certainty-trigger traces.
# Submit from the repo root on Quest:   sbatch scripts/train_medgemma_sft.slurm
# Inputs: data/training_set/sft/{train,val}.jsonl (built by scripts/build_sft_examples.py).
# Output: runs/medgemma-27b-lora-certainty/  (the LoRA adapter + tokenizer).
# Rationale + budget: see HANDOFF_medgemma_sft_quest.md.

#SBATCH --account=p31777
#SBATCH --partition=gengpu                        # ⚠️ CONFIRM this partition has H100s:
                                                  #   sinfo -p gengpu -o "%P %G %N"
                                                  # (the Qwen run used gengpu A100s; if H100s live
                                                  #  in another partition/constraint, update here)
#SBATCH --gres=gpu:h100:1                          # one 80GB H100: fits full bf16 LoRA of 27B
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8                          # dataloading is trivial (278 rows)
#SBATCH --mem=128G                                 # host RAM: 27B checkpoint shards ~54GB + cache
#SBATCH --time=08:00:00                            # compute ~2-4h; 8h = headroom (download + eval)
#SBATCH --job-name=medgemma_lora_certainty
#SBATCH --output=runs/medgemma_lora_%j.out
#SBATCH --error=runs/medgemma_lora_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=duruoli2024@u.northwestern.edu

# ============================================
# Environment Setup
# ============================================
module purge all
# Do NOT `module load cuda` — the pip torch wheel bundles its own CUDA runtime; the system
# cuda module drags in an ancient gcc whose libstdc++.so.6 lacks CXXABI_1.3.8 -> import torch
# dies in <10s. (Same trap as the Qwen run.)
source "$HOME/miniconda/bin/activate"
conda activate congraph                           # must have: torch(+cuda) trl>=0.12 peft>=0.13
                                                  #   transformers>=4.50 (Gemma-3!) accelerate datasets
                                                  # one-time install on a LOGIN node:
                                                  #   pip install -U "transformers>=4.50" "trl>=0.12" \
                                                  #     "peft>=0.13" accelerate datasets
# Put the conda env's modern libstdc++ ahead of any stale system gcc, else torch fails with
# libstdc++.so.6: version `CXXABI_1.3.8' not found.
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
# Hugging Face — GATED model + offline compute nodes
# ============================================
# medgemma is a GATED repo. ON A LOGIN NODE (which has internet), ONE TIME, do:
#   1) In a browser, sign in to HF and click "Agree and access" on
#      https://huggingface.co/google/medgemma-27b-text-it  (accept the Health AI license).
#   2) Create a READ token: HF Settings -> Access Tokens.
#   3) On the Quest LOGIN node terminal:   huggingface-cli login   (paste the token; it is
#      saved to ~/.cache/huggingface/token and persists across jobs).
#   4) Pre-download the weights while online (compute nodes have NO net):
#        export HF_HOME=$HOME/.cache/huggingface
#        python -c "from huggingface_hub import snapshot_download; \
#                   snapshot_download('google/medgemma-27b-text-it')"
# Then this job runs fully offline:
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1                            # comment out if compute nodes DO have net
export TOKENIZERS_PARALLELISM=false

# ============================================
# Train  (FULL QUALITY: no --load-4bit)
# ============================================
# Full bf16 base + LoRA with gradient checkpointing (peak ~64-68GB, fits the 80GB H100).
# assistant_only_loss + the Gemma-3 generation-marked chat template are auto-handled by the
# script (see HANDOFF_medgemma_sft_quest.md "Why the template swap"); watch the log for
#   [chat-compat] assistant mask OK — N/M tokens ...
# If (unlikely) OOM: try  --max-len 10496  first, then 8-bit Adam, then --load-4bit as last resort.
python scripts/train_lora_qwen.py \
    --base google/medgemma-27b-text-it \
    --data data/training_set/sft \
    --out runs/medgemma-27b-lora-certainty \
    --epochs 3 --batch 1 --grad-accum 16 --lr 2e-4 \
    --max-len 12288

echo "============================================"
echo "Job completed at: $(date)"
echo "============================================"
