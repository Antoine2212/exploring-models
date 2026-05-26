#!/bin/bash
#SBATCH --job-name=explore
#SBATCH --output=/home/antoine_df/gemma_explore/logs/job_%j.out
#SBATCH --error=/home/antoine_df/gemma_explore/logs/job_%j.err
#SBATCH --partition=ialab

#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB


pwd


START_TIME=$(date +%s)

export HF_TOKEN=$(cat submissions/HF_token.txt)
uv run python scripts/collect_qk.py

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

printf "⏱️  Temps écoulé : %02dh %02dm %02ds\n" \
    $((ELAPSED / 3600)) \
    $(( (ELAPSED % 3600) / 60 )) \
    $((ELAPSED % 60))

echo "✅ Toutes les expériences terminées."
