#!/bin/bash

# Direction-B SFT: LoRA fine-tune Qwen2.5-7B-Instruct on the certainty-trigger traces.
# Submit from the repo root on Quest:   sbatch scripts/train_lora_qwen.slurm
# Inputs: data/training_set/sft/{train,val}.jsonl (built by scripts/build_sft_examples.py).
# Output: runs/qwen2.5-7b-lora-certainty/  (the LoRA adapter + tokenizer).

#SBATCH --account=p31777
#SBATCH --partition=gengpu                       # Quest general GPU partition (A100s)
#SBATCH --gres=gpu:a100:1                         # one A100 is plenty for 7B LoRA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8                         # dataloading is trivial (285 rows); 8 is ample
#SBATCH --mem=64G                                 # host RAM for model load + HF cache
#SBATCH --time=02:00:00                           # actual ~30-45 min; headroom for model download
#SBATCH --job-name=qwen_lora_certainty
#SBATCH --output=runs/qwen_lora_%j.out
#SBATCH --error=runs/qwen_lora_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=duruoli2024@u.northwestern.edu

# ============================================
# Environment Setup
# ============================================
module purge all
# NOTE: do NOT `module load cuda` here — the pip torch wheel bundles its own CUDA
# runtime, and the system cuda module drags in an ancient gcc whose libstdc++.so.6
# lacks CXXABI_1.3.8 -> `import torch` ImportError (job dies in <10s). See below.
source "$HOME/miniconda/bin/activate"
conda activate congraph                           # must have: torch(+cuda) trl peft transformers
                                                  #            accelerate bitsandbytes datasets
                                                  # one-time install (login node, see notes below):
                                                  #   pip install "trl>=0.12" "peft>=0.13" \
                                                  #     "transformers>=4.45" accelerate bitsandbytes datasets

# Put the conda env's modern libstdc++ AHEAD of any stale system gcc on the library
# path, else torch fails with: libstdc++.so.6: version `CXXABI_1.3.8' not found.
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
# Hugging Face cache / offline weights
# ============================================
# Quest compute nodes usually have NO internet. Pre-download the base model ONCE on a
# login node (which does have net):
#   export HF_HOME=$HOME/.cache/huggingface
#   python -c "from huggingface_hub import snapshot_download; \
#              snapshot_download('Qwen/Qwen2.5-7B-Instruct')"
# then run the job offline:
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1                            # comment out if compute nodes DO have net
export TOKENIZERS_PARALLELISM=false

# ============================================
# Train
# ============================================
# Default: full bf16 LoRA with gradient checkpointing (fits a 40GB A100 at max-len 12288).
# If you land on a smaller GPU or hit OOM, add  --load-4bit  (QLoRA).
python scripts/train_lora_qwen.py \
    --base Qwen/Qwen2.5-7B-Instruct \
    --data data/training_set/sft \
    --out runs/qwen2.5-7b-lora-certainty \
    --epochs 3 --batch 1 --grad-accum 16 --lr 2e-4

echo "============================================"
echo "Job completed at: $(date)"
echo "============================================"
