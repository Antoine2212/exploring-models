#!/bin/bash
#SBATCH --job-name=ruler-qwen
#SBATCH --output=/home/antoine_df/gemma_explore/logs/ruler_%j.out
#SBATCH --error=/home/antoine_df/gemma_explore/logs/ruler_%j.err
#SBATCH --partition=ialab
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

set -euo pipefail

pwd
date

START_TIME=$(date +%s)

export HF_TOKEN=$(tr -d '\n\r ' < submissions/HF_token.txt)
echo "Token length: ${#HF_TOKEN}"
echo "Token starts with: ${HF_TOKEN:0:10}"

export PYTHONUNBUFFERED=1

source /home/antoine_df/gemma_explore/.venv/bin/activate



# Exécuter avec python direct (pas uv run)
python scripts/build_cache.py

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

printf "Temps ecoule : %02dh %02dm %02ds\n" \
    $((ELAPSED / 3600)) \
    $(((ELAPSED % 3600) / 60)) \
    $((ELAPSED % 60))

echo "Toutes les experiences terminees."