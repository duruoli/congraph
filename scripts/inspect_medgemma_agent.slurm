#!/bin/bash

# Manual 3-way inspector (GOLD vs BASE vs SFT) for the MEDGEMMA-27B agent
# (scripts/inspect_agent_outputs.py). The Qwen inspector ran on the Mac's MPS, but 27B
# does not fit MPS, so medgemma must inspect on a Quest GPU.
# Submit from the repo root on Quest:
#   sbatch scripts/inspect_medgemma_agent.slurm                  # first 6 steps, greedy
#   sbatch scripts/inspect_medgemma_agent.slurm "--n 10 --first" # extra flags (quote them)
# Output:  results/agent_inspection/inspect_outputs_medgemma.{txt,jsonl}  (preserves the Qwen files)

#SBATCH --account=p31777
#SBATCH --partition=gengpu                        # ⚠️ CONFIRM this partition has the GPU below
#SBATCH --gres=gpu:h100:1                          # 27B bf16 ~54GB -> needs an 80GB card
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00                            # a handful of steps, generation is short
#SBATCH --job-name=inspect_medgemma_agent
#SBATCH --output=runs/inspect_medgemma_%j.out
#SBATCH --error=runs/inspect_medgemma_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=duruoli2024@u.northwestern.edu

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
export HF_HUB_OFFLINE=1                            # comment out if the node HAS net
export TOKENIZERS_PARALLELISM=false

# Extra CLI flags overridable via $1 (base/adapter default to medgemma inside the script).
EXTRA="${1:---n 6 --first}"
echo "extra flags = $EXTRA"

python scripts/inspect_agent_outputs.py \
    $EXTRA \
    --out results/agent_inspection/inspect_outputs_medgemma.txt

echo "============================================"
echo "Job completed at: $(date)"
echo "============================================"
