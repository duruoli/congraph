#!/bin/bash

# Direction-B SFT: FULL-QUALITY (bf16, no quantization) LoRA fine-tune of
# Qwen/Qwen3-30B-A3B-Instruct-2507 (MoE) on the certainty-trigger reasoning traces.
# This produces the REASONING adapter that pred_dev approach b derives from and c' warm-starts
# from — keep its base identical to a/c (Qwen3) or the a/b/c comparison mixes bases.
# Submit from the repo root on Quest:   sbatch scripts/train_medgemma_sft.slurm
# Inputs: data/training_set/sft/{train,val}.jsonl (built by scripts/build_sft_examples.py).
# Output: runs/qwen3-30b-a3b-lora-certainty/  (the LoRA adapter + tokenizer).
# Rationale + budget: see HANDOFF_medgemma_sft_quest.md.

#SBATCH --account=p31777
#SBATCH --partition=gengpu                        # ⚠️ CONFIRM this partition has H100s:
                                                  #   sinfo -p gengpu -o "%P %G %N"
                                                  # (the Qwen run used gengpu A100s; if H100s live
                                                  #  in another partition/constraint, update here)
#SBATCH --gres=gpu:h100:2                          # 2x80GB H100: ~61GB bf16 weights shard across both
                                                  # (device_map=auto) -> comfortable headroom, guaranteed
                                                  # fit. Single-GPU alt: set :1 and add --load-4bit.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                        # ONE process (model-parallel), NOT torchrun/DDP
#SBATCH --cpus-per-task=8                          # dataloading is trivial (278 rows)
#SBATCH --mem=192G                                 # host RAM: 30B shards ~61GB materialized on CPU at load
#SBATCH --time=08:00:00                            # compute ~2-4h; 8h = headroom (download + eval)
#SBATCH --job-name=qwen3_lora_certainty
#SBATCH --output=runs/qwen3_lora_%j.out
#SBATCH --error=runs/qwen3_lora_%j.err
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
                                                  #   transformers>=4.51 (Qwen3 MoE!) accelerate datasets
                                                  # one-time install on a LOGIN node:
                                                  #   pip install -U "transformers>=4.51" "trl>=0.12" \
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
# Hugging Face — offline compute nodes (Qwen3 is NOT gated; no license/login needed)
# ============================================
# ON A LOGIN NODE (which has internet), ONE TIME, pre-download the weights (compute nodes
# have NO net):
#        export HF_HOME=$HOME/.cache/huggingface
#        python -c "from huggingface_hub import snapshot_download; \
#                   snapshot_download('Qwen/Qwen3-30B-A3B-Instruct-2507')"
# Then this job runs fully offline:
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1                            # comment out if compute nodes DO have net
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # combat fragmentation across the 2 GPUs

# ============================================
# Train  (FULL QUALITY: bf16, no --load-4bit; add --load-4bit if the 30B MoE OOMs)
# ============================================
# Full bf16 base + LoRA with gradient checkpointing. MoE -> LoRA on attention only
# ([lora] target_modules = [q/k/v/o_proj]); the 128 experts stay frozen (cheap forward).
# assistant_only_loss + the Qwen3 generation-marked chat template are auto-handled by the
# script (configs/qwen3_assistant_loss_template.jinja); watch the log for
#   [chat-compat] swapped in a generation-marked template ... + assistant mask OK — N/M tokens
# If OOM: try  --max-len 10496  first, then 8-bit Adam, then --load-4bit as last resort.
python scripts/train_lora_qwen.py \
    --base Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --data data/training_set/sft \
    --out runs/qwen3-30b-a3b-lora-certainty \
    --epochs 3 --batch 1 --grad-accum 16 --lr 2e-4 \
    --max-len 12288
status=$?
if [ $status -ne 0 ]; then echo "TRAIN FAILED (exit $status) — skipping eval"; exit $status; fi

# ============================================
# Eval  (auto-chained: pred_dev approach B end-to-end, same 2-GPU allocation).
# B derives P(deviate)=1-follow_prob from THIS reasoning adapter. It needs a TEST pass and a
# VAL pass of the certainty-agent panel (for Platt), then a CPU deriver. All three run here so
# one sbatch yields B's calibrated result with no manual step.
# ============================================
echo "=== training done; starting approach-B eval (test panel) ==="
python scripts/eval_certainty_agent.py \
    --arms doctor base sft \
    --base-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapter runs/qwen3-30b-a3b-lora-certainty \
    --data data/training_set/sft/test.jsonl \
    --out results/agent_inspection/eval_panel_qwen3.txt

echo "=== approach-B eval (val panel, for Platt calibration) ==="
python scripts/eval_certainty_agent.py \
    --arms doctor base sft \
    --base-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapter runs/qwen3-30b-a3b-lora-certainty \
    --data data/training_set/sft/val.jsonl \
    --out results/agent_inspection/eval_panel_qwen3_val.txt

echo "=== approach-B deriver (CPU: 1-follow_prob -> Platt on val -> test metrics) ==="
python scripts/eval_dev_from_reasoning.py \
    --val-panel  results/agent_inspection/eval_panel_qwen3_val.jsonl \
    --test-panel results/agent_inspection/eval_panel_qwen3.jsonl \
    --arms sft base

echo "============================================"
echo "Job completed at: $(date)"
echo "============================================"
