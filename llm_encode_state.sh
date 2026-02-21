#!/bin/bash

#SBATCH --account=p31777
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=05:00:00
#SBATCH --mem=20G
#SBATCH --job-name=llm_encode_state
#SBATCH --output=experiments/congraph/main_py_%j.out
#SBATCH --error=experiments/congraph/main_py_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=duruoli2024@u.northwestern.edu

# ============================================
# Environment Setup
# ============================================
module purge all
source "$HOME/miniconda/bin/activate"
conda activate congraph

echo "============================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"
python --version
echo "============================================"

# ============================================
# Run your script
# ============================================
cd /home/dlf8982/AAA/congraph
python main.py

echo "============================================"
echo "Job completed at: $(date)"
echo "============================================"