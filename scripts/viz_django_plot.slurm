#!/bin/bash
#SBATCH --partition=alvis
#SBATCH --gpus-per-node=A100fat:1
#SBATCH -A NAISS2026-3-104
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --job-name=viz_plot
#SBATCH --output=logs/viz_plot_%j.out
#SBATCH --error=logs/viz_plot_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT=/mimer/NOBACKUP/groups/llm-readability/EditBoundaryDetection
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"

python scripts/viz_django.py \
    --out-prefix logs/django_viz \
    --plot-only

echo "3D/2D viz done: $(date)"

python scripts/plot_django_regions.py \
    --embs       logs/django_viz_embs.npz \
    --out-prefix logs/django

echo "Combined figure done: $(date)"
