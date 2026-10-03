#!/bin/bash
# Environment + data. Run from the repository root.
set -euo pipefail

# 1. Python environment
python3.12 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r reproduce/requirements.txt

# 2. Data (https://github.com/YundongShen/MSR_EEL_Data)
#    DATASET=eel_dataset_paper_counts  -> 743 / 82 / 48 (default)
#    DATASET=eel_dataset               -> Haiku only, no selection
DATASET=${DATASET:-eel_dataset_paper_counts}
[ -d MSR_EEL_Data ] || git clone https://github.com/YundongShen/MSR_EEL_Data.git
mkdir -p data/raw data/processed data/cache logs checkpoints
zcat MSR_EEL_Data/swebench/swebench_full_instances.jsonl.gz      > data/raw/swebench_full_instances.jsonl
zcat MSR_EEL_Data/eel_dataset/instances_full.jsonl.gz            > data/processed/instances_full.jsonl
cp   MSR_EEL_Data/eel_dataset/splits.json                          data/processed/splits.json
cp   MSR_EEL_Data/$DATASET/llm_t12_hunks.jsonl                     data/cache/llm_t12_hunks.jsonl
cp   MSR_EEL_Data/$DATASET/tier3_hunks.jsonl                       data/cache/tier3_hunks.jsonl

# 3. Repository subsets for the leave-Django-out experiment
python - <<'EOF'
import json
with open("data/processed/instances_full.jsonl") as f, \
     open("data/processed/instances_django_only.jsonl", "w") as d, \
     open("data/processed/instances_non_django.jsonl", "w") as n:
    for line in f:
        (d if json.loads(line)["instance_id"].startswith("django__") else n).write(line)
EOF
