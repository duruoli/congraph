#!/bin/bash

# Approach A (pred_dev) eval: read a calibrated P(deviate) from the bare-label medgemma
# (target = one word; NO --generate-first). Platt (a,b) fit on VAL, applied to TEST.
# Submit from the repo root on Quest:  sbatch scripts/eval_deviate_cls.slurm
# Inputs:  data/training_set/cls/{val,test}.jsonl  +  runs/medgemma-27b-lora-deviate-cls/
# Output:  results/agent_inspection/deviation_cls_eval.{txt,json}  (git-tracked)

#SBATCH --account=p31777
#SBATCH --partition=gengpu                          # ⚠️ CONFIRM this partition has the GPU below
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00                              # no generation, just teacher-forced scoring
#SBATCH --job-name=eval_deviate_cls
#SBATCH --output=runs/eval_deviate_cls_%j.out
#SBATCH --error=runs/eval_deviate_cls_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=duruoli2024@u.northwestern.edu

module purge all
source "$HOME/miniconda/bin/activate"
conda activate congraph
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /home/dlf8982/AAA/congraph
mkdir -p runs results/agent_inspection

echo "Job started at: $(date)   Node: $(hostname)   Job ID: $SLURM_JOB_ID"
python --version
nvidia-smi

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python scripts/eval_deviation_cls.py \
    --base google/medgemma-27b-text-it \
    --adapter runs/medgemma-27b-lora-deviate-cls \
    --data data/training_set/cls \
    --arms base sft \
    --out results/agent_inspection/deviation_cls_eval

echo "Job completed at: $(date)"
