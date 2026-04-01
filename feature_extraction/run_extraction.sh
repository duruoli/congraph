#!/bin/bash

#SBATCH --account=p31777
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=20G
#SBATCH --job-name=feature_extraction
#SBATCH --output=experiments/traversal_engine/feature_extraction_%j.out
#SBATCH --error=experiments/traversal_engine/feature_extraction_%j.err
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
# OpenAI API Key
# ============================================
# Option A: load from a key file (recommended — keep this file private)
# export OPENAI_API_KEY=$(cat "$HOME/.openai_api_key")
#
# Option B: set directly here (less secure)
# export OPENAI_API_KEY="sk-..."
#
# The key must be available before the Python script is called.
# Uncomment and edit one of the options above if not set in your environment.

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set. Aborting."
    exit 1
fi

# ============================================
# Run feature extraction
# ============================================
cd /home/dlf8982/AAA/congraph

# Process all four diseases (default).
# Add --disease <name> to run a single disease.
# Add --limit <n>     to process only the first n patients (for testing).
# Add --no-llm        to run algo-only extraction (no API calls).
python run_feature_extraction.py \
    --disease all \
    --output-dir results/feature_extraction

echo "============================================"
echo "Job completed at: $(date)"
echo "============================================"
