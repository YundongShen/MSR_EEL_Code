#!/bin/bash
#SBATCH --partition=alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -A NAISS2026-3-104
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=ebd_baseline_bm25
#SBATCH --output=logs/baseline_bm25_%j.out
#SBATCH --error=logs/baseline_bm25_%j.err

set -euo pipefail

PROJECT=/mimer/NOBACKUP/groups/llm-readability/EditBoundaryDetection
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"

python baselines_bm25.py \
    --instances data/processed/instances_full.jsonl \
    --tier3     data/cache/tier3_hunks.jsonl \
    --repo-pool \
    --output    logs/baseline_bm25.json

echo "Done: $(date)"
