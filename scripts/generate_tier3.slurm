#!/bin/bash
#SBATCH --partition=berzelius-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=08:00:00
#SBATCH --job-name=ebd_tier3
#SBATCH --output=logs/tier3_%j.out
#SBATCH --error=logs/tier3_%j.err

set -euo pipefail

PROJECT=${PROJECT:-$SLURM_SUBMIT_DIR}
cd "$PROJECT"
source gpu_env/bin/activate

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set. Submit with:"
    echo "  sbatch --export=ANTHROPIC_API_KEY=\$ANTHROPIC_API_KEY scripts/generate_tier3.slurm"
    exit 1
fi

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Model: claude-haiku-4-5-20251001"

python scripts/generate_tier3.py \
    --input  data/raw/swebench_full_instances.jsonl \
    --output data/cache/tier3_hunks.jsonl \
    --model  claude-haiku-4-5-20251001

echo "Done: $(date)"
if [ -f data/cache/tier3_hunks.jsonl ]; then
    echo "Tier-3 instances written: $(wc -l < data/cache/tier3_hunks.jsonl)"
fi
