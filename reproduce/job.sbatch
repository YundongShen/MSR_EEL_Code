#!/bin/bash
# Berzelius job template. Usage (from the repository root):
#   sbatch reproduce/job.sbatch python train.py ...
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export PYTHONHASHSEED=42

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
"$@"
