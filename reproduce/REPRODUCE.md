# Reproduction

All commands run from the repository root. Prefix GPU commands with `sbatch reproduce/job.sbatch` on Berzelius.

## 1. Environment

| | |
|---|---|
| Python | 3.12 |
| CUDA | 12.1 (torch 2.5.1+cu121, cuDNN 9.1) |
| Packages | `reproduce/requirements.txt` |
| Setup | `bash reproduce/setup_env.sh` (venv + data + Django subsets) |
| API keys | `ANTHROPIC_API_KEY` (candidate generation, region analysis), `GEMINI_API_KEY` (edit-type labels) |

## 2. Hardware

| Job | GPU |
|---|---|
| Training (M1–M4, ablations) | 1× A100 80GB |
| Evaluation, baselines | 1× A100 or A40 |
| BM25, data pipeline | CPU |

## 3. Random seeds

| What | Seed |
|---|---|
| Training (`random`, `numpy`, `torch`) | 42 (`config.py: train.seed`) |
| `PYTHONHASHSEED` | 42 |
| Train/val/test split (`splits.json`) | 42 |
| T0 distractor sampling + score tie-breaks | `random.Random(42)` |
| Test-set selection (`eel_dataset_paper_counts`) | 20261003 |
| Candidate generation (Haiku) | none (API sampling, temperature 0.7) |

## 4. Hyperparameters

| | |
|---|---|
| Backbone | `microsoft/unixcoder-base` |
| Projection dim | 256, dropout 0.1 |
| Max tokens per entity | 512; requirement cut to 2000 chars |
| Batch size | 32 |
| Optimizer | AdamW, lr 2e-5, weight decay 1e-4, warmup 200 steps, grad clip 1.0 |
| Epochs | 10 (250,000 pair-type-balanced samples per epoch) |
| InfoNCE temperature | learnable, init 0.07 |
| Pair types | REQ–TEST, REQ–HUNK, ORIG–HUNK, REQ–ORIG (weight 1.0 each) |
| M2 hard negatives | Tier-3 hunks of the same instance |
| M3 extra negatives | 3 same-repo hunks per anchor |
| M4 tier weights | T1 1.0, T2 0.67 |
| Score | α·sim(h,r) + β·sim(h,t̄) + γ·sim(h,ō), α=1.0, β=0.5, γ=0.5 |
| Evaluation | up to 50 T0 per instance; nDCG gains T1 3 / T2 2 / T3 0 / T0 0; k = #retained |
| Candidate generation | `claude-haiku-4-5-20251001`, temperature 0.7, max tokens 4096, 1 sample per issue |
| Tier rule | retained = same file and start line within 20 lines of a reference hunk; T1 if file linked to a fail-to-pass test, else T2; otherwise T3 |

## 5. Commands (in order)

### 5.1 Data (skip if `setup_env.sh` was used)
```
python data_pipeline/download_swebench.py --dataset princeton-nlp/SWE-bench --output data/raw/swebench_full_instances.jsonl
python data_pipeline/parse_instances.py --input data/raw/swebench_full_instances.jsonl --output data/processed/instances_full.jsonl
python data_pipeline/generate_candidates.py --input data/raw/swebench_full_instances.jsonl --generations data/cache/haiku_generations.jsonl --model claude-haiku-4-5-20251001
python data_pipeline/build_dataset.py
```
`data/processed/splits.json` is fixed (from MSR_EEL_Data); do not regenerate it.

### 5.2 Training
```
python train.py --instances data/processed/instances_full.jsonl --encoder microsoft/unixcoder-base --epochs 10 --checkpoint-dir checkpoints/m1 --pair-types req_test req_hunk orig_hunk req_orig
python train.py --instances data/processed/instances_full.jsonl --encoder microsoft/unixcoder-base --epochs 10 --checkpoint-dir checkpoints/m2 --pair-types req_test req_hunk orig_hunk req_orig --tier3 data/cache/tier3_hunks.jsonl
python train.py --instances data/processed/instances_full.jsonl --encoder microsoft/unixcoder-base --epochs 10 --checkpoint-dir checkpoints/m3 --pair-types req_test req_hunk orig_hunk req_orig --tier3 data/cache/tier3_hunks.jsonl --same-repo
python train.py --instances data/processed/instances_full.jsonl --encoder microsoft/unixcoder-base --epochs 10 --checkpoint-dir checkpoints/m4 --pair-types req_test req_hunk orig_hunk req_orig --tier3 data/cache/tier3_hunks.jsonl --tier-aware-loss

python train.py --instances data/processed/instances_full.jsonl --encoder microsoft/unixcoder-base --epochs 10 --checkpoint-dir checkpoints/m2_abl_no_orig --pair-types req_test req_hunk --tier3 data/cache/tier3_hunks.jsonl
python train.py --instances data/processed/instances_full.jsonl --encoder microsoft/unixcoder-base --epochs 10 --checkpoint-dir checkpoints/m2_abl_no_test --pair-types req_hunk orig_hunk req_orig --tier3 data/cache/tier3_hunks.jsonl
python train.py --instances data/processed/instances_full.jsonl --encoder microsoft/unixcoder-base --epochs 10 --checkpoint-dir checkpoints/m2_abl_no_req --pair-types orig_hunk --tier3 data/cache/tier3_hunks.jsonl
python train.py --instances data/processed/instances_full.jsonl --encoder microsoft/unixcoder-base --epochs 10 --batch-size 32 --checkpoint-dir checkpoints/exp2_req_only --pair-types req_hunk --tier3 data/cache/tier3_hunks.jsonl
python train.py --instances data/processed/instances_full.jsonl --encoder microsoft/unixcoder-base --epochs 10 --batch-size 32 --checkpoint-dir checkpoints/exp2_test_only --pair-types req_hunk req_test --tier3 data/cache/tier3_hunks.jsonl

python train.py --instances data/processed/instances_non_django.jsonl --encoder microsoft/unixcoder-base --epochs 10 --batch-size 32 --checkpoint-dir checkpoints/exp4_ldo --pair-types req_hunk req_test orig_hunk req_orig --tier3 data/cache/tier3_hunks.jsonl
```

### 5.3 Evaluation
Shared arguments: `A="--instances data/processed/instances_full.jsonl --tier3 data/cache/tier3_hunks.jsonl"`
```
python evaluate.py --checkpoint none --no-projection --exp all --umap --repo-pool $A           # M0
python evaluate.py --checkpoint checkpoints/m1/best.pt --exp all --umap --repo-pool $A         # M1
python evaluate.py --checkpoint checkpoints/m2/best.pt --exp all --umap --repo-pool $A         # M2 = EEL
python evaluate.py --checkpoint checkpoints/m3/best.pt --exp all --umap --repo-pool $A         # M3
python evaluate.py --checkpoint checkpoints/m4/best.pt --exp all --umap --repo-pool $A         # M4

python evaluate.py --checkpoint checkpoints/m2_abl_no_orig/best.pt --exp all --repo-pool --score-alpha 1.0 --score-beta 0.5 --score-gamma 0.0 $A
python evaluate.py --checkpoint checkpoints/m2_abl_no_test/best.pt --exp all --repo-pool --score-alpha 1.0 --score-beta 0.0 --score-gamma 0.5 $A
python evaluate.py --checkpoint checkpoints/m2_abl_no_req/best.pt  --exp all --repo-pool --score-alpha 0.0 --score-beta 0.0 --score-gamma 0.5 $A
python evaluate.py --checkpoint checkpoints/exp2_req_only/best.pt  --exp retrieval --repo-pool --score-alpha 1.0 --score-beta 0.0 --score-gamma 0.0 $A
python evaluate.py --checkpoint checkpoints/exp2_test_only/best.pt --exp retrieval --repo-pool --score-alpha 0.0 --score-beta 1.0 --score-gamma 0.0 $A
python evaluate.py --checkpoint checkpoints/m2_abl_no_req/best.pt  --exp retrieval --repo-pool --score-alpha 0.0 --score-beta 0.0 --score-gamma 1.0 $A
```

### 5.4 Baselines
```
python baselines/baselines_bm25.py --repo-pool $A --output logs/baseline_bm25.json
python baselines/baselines_pretrained.py --model microsoft/codebert-base      --repo-pool $A --output logs/baseline_codebert.json
python baselines/baselines_pretrained.py --model microsoft/graphcodebert-base --repo-pool $A --output logs/baseline_graphcodebert.json
python baselines/baselines_extra.py --mode codet5        --repo-pool $A --output logs/baseline_codet5.json
python baselines/baselines_extra.py --mode cross_encoder --repo-pool $A --output logs/baseline_cross_encoder.json
```

### 5.5 Analysis and figures
```
python evaluate.py --checkpoint checkpoints/m2/best.pt --exp geometry $A --save-hunk-scores logs/exp2_hunk_scores_m2.jsonl
python analysis/exp2_label_hunks.py                     # -> data/processed/hunk_edit_types.jsonl (GEMINI_API_KEY)
python analysis/exp2_analysis.py                        # -> logs/exp2_analysis.json, logs/exp2_component_violin.png
python analysis/exp3_analysis.py --m1 checkpoints/m1/best.pt --m2 checkpoints/m2/best.pt $A --output logs/exp3_analysis.json --analysis all
python analysis/viz_django.py --checkpoint-m2 checkpoints/m2/best.pt $A --out-prefix logs/django_viz --encode-only
python analysis/viz_django.py --out-prefix logs/django_viz --plot-only
python analysis/plot_django_regions.py --embs logs/django_viz_embs.npz --out-prefix logs/django
python analysis/analyze_regions.py                      # -> logs/region_analysis.json (ANTHROPIC_API_KEY)
```

### 5.6 Generalization
```
python evaluate.py --exp retrieval --checkpoint checkpoints/exp4_ldo/best.pt --instances data/processed/instances_django_only.jsonl --tier3 data/cache/tier3_hunks.jsonl --all-instances
python evaluate.py --exp retrieval --checkpoint checkpoints/m2/best.pt       --instances data/processed/instances_django_only.jsonl --tier3 data/cache/tier3_hunks.jsonl --all-instances
python evaluate.py --exp retrieval --no-projection                            --instances data/processed/instances_django_only.jsonl --tier3 data/cache/tier3_hunks.jsonl --all-instances
```
Cross-generator sets: `slurm/data_pipeline/generate_candidates_{gemini,qwen,deepseek}.slurm`, then `slurm/generalization/eval_exp4_crossllm*.slurm`.

## 6. Tables and figures → commands

| Paper item | Commands |
|---|---|
| Training variants M0–M4 (nDCG@k, T2-Recall, PSR) | 5.2 M1–M4, 5.3 M0–M4 |
| Baselines | 5.4 |
| View ablation: Full EEL / w/o ORIG / w/o TEST / w/o REQ | 5.3 M2 + three `m2_abl_*` evaluations |
| Single-view: REQ-only / TEST-view / ORIG-only | 5.3 last three lines |
| 3D UMAP, M0 vs M2 | 5.3 M0 and M2 (`--umap`), `logs/umap_*` |
| Django embedding scatter and regions R1–R5 | 5.5 `viz_django.py`, `plot_django_regions.py`, `analyze_regions.py` |
| Edit-type component analysis | 5.5 `exp2_*` |
| Geometric separation statistics (silhouette, Mann–Whitney) | 5.3 `--exp all` / 5.5 `exp3_analysis.py` |
| Leave-Django-out, cross-generator | 5.6 |

## 7. Outputs

| | |
|---|---|
| Checkpoints | `checkpoints/<variant>/best.pt` |
| Retrieval metrics | job log lines `nDCG@k`, `T2-Recall`, `Perfect-Sep` |
| Baselines | `logs/baseline_*.json` |
| Figures / analysis | `logs/` |
