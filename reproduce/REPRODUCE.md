# Reproduction

Use the final submission repository, `MSR_EEL_SUBMIT`. Commands below run from
its `Code` directory. Data comes from the adjacent `Data` directory; no older
`MSR_EEL_Data` selection is downloaded or substituted.

## Environment and data

```bash
git clone https://github.com/YundongShen/MSR_EEL_SUBMIT.git
cd MSR_EEL_SUBMIT/Code
bash reproduce/setup_env.sh
source env/bin/activate
export PYTHONHASHSEED=42
```

The supported environment is Python 3.11 or 3.12 with the dependencies in
`requirements.txt`. The default interpreter is `python3.12`; override it with
`EEL_PYTHON_BIN`. Linux GPU setup uses PyTorch 2.5.1 CUDA 12.1 wheels. Set
`EEL_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu` for CPU-only setup.
These package pins describe a supported runtime, rather than a complete freeze
of the historical Alvis environment.

`bash reproduce/setup_env.sh --check` checks the published data without writing
files or installing packages. `--data-only` prepares data without installing an
environment. For a separate Code checkout, set `EEL_DATA_ROOT` to the final
submission's `Data` directory. The setup checks:

| Item | Count |
|---|---:|
| Issues; train / validation / test | 2291; 1833 / 229 / 229 |
| Generated retained hunks; T1 / T2 | 9223; 3354 / 5869 |
| Generated T3 hunks / issues | 1257 / 743 |
| Core evaluation issues / T2-Recall issues | 82 / 47 |
| Django issues / eligible Django issues | 849 / 204 |

Setup decompresses the parsed instances into `data/processed/`, copies the fixed
splits and generated candidates, and creates Django and non-Django instance
files. It does not regenerate labels or modify the published files. The instance
loader excludes reference-patch hunks; model candidates are read from
`llm_t12_hunks.jsonl` and `tier3_hunks.jsonl`.

For Slurm, prefix a command with `sbatch reproduce/job.sbatch`. Supply the
account, partition, GPU type and time limit required by your cluster as `sbatch`
options. Setup creates `logs/` before submission. The wrapper uses one GPU and
eight CPU cores. Training and encoder evaluation benefit from a GPU; BM25 and
data preparation can run on CPU. Resource settings are runtime choices, not
claims about the hardware of every historical job.

## Protocol and training

Training uses seed 42, the fixed issue splits, UniXCoder, a 256-dimensional
projection, dropout 0.1, and at most 512 tokens per entity. Requirements are
truncated to 2000 characters. AdamW uses learning rate 2e-5, weight decay 1e-4,
200 warmup steps, cosine decay and gradient clipping at 1.0. Temperature starts
at 0.07, is learned and is clamped to [0.01, 1.0]. Batch size is 32; each epoch
draws 250000 pair-type-balanced samples. The four pair types are REQ–TEST,
REQ–HUNK, ORIG–HUNK and REQ–ORIG, with unit weights.

Ten epochs is the maximum training budget. The historical REQ-only and TEST-view
runs completed six and seven epochs before timing out; their evaluations used
the best validation checkpoint, saved at epoch one. A fresh ten-epoch run is a
new experiment and must report its own selected checkpoint and results.

```bash
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/m1 --pair-types req_test req_hunk orig_hunk req_orig
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/m2 --pair-types req_test req_hunk orig_hunk req_orig --tier3 data/cache/tier3_hunks.jsonl
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/m3 --pair-types req_test req_hunk orig_hunk req_orig --tier3 data/cache/tier3_hunks.jsonl --same-repo
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/m4 --pair-types req_test req_hunk orig_hunk req_orig --tier3 data/cache/tier3_hunks.jsonl --same-repo --tier-aware-loss
```

M1 uses in-batch negatives. M2 adds same-issue T3 negatives. M3 retains M2's
negatives and adds up to three retained hunks from other issues in the same
repository per anchor. M4 uses this M3 configuration with T1/T2 loss weights
1.0/0.67, matching the submitted training log. M0 is the untrained backbone
without the projection for the ranking evaluation.

```bash
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/m2_abl_no_orig --pair-types req_test req_hunk --tier3 data/cache/tier3_hunks.jsonl
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/m2_abl_no_test --pair-types req_hunk orig_hunk req_orig --tier3 data/cache/tier3_hunks.jsonl
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/m2_abl_no_req --pair-types orig_hunk --tier3 data/cache/tier3_hunks.jsonl
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/exp2_req_only --pair-types req_hunk --tier3 data/cache/tier3_hunks.jsonl
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/exp2_test_only --pair-types req_hunk req_test --tier3 data/cache/tier3_hunks.jsonl
```

TEST-view retains REQ as a training anchor but scores with TEST alone. The
ORIG-only model uses the ORIG–HUNK relation; removing REQ yields that same
relation and is not an additional row in the main ablation table.

## Core ranking and repository-view ablations

Define these Bash arrays once for the evaluation commands below:

```bash
A=(--instances data/processed/instances_full.jsonl --llm-t12 data/cache/llm_t12_hunks.jsonl --tier3 data/cache/tier3_hunks.jsonl --repo-pool)
W=(--score-alpha 1.0 --score-beta 0.5 --score-gamma 0.5)
```

The retention score is `alpha * sim(h, REQ) + beta * max_t sim(h, t) + gamma *
sim(h, o(h))`. TEST takes the maximum over individual tests. ORIG uses the
normalized mean of matched source-unit embeddings and is zero when no unit
matches. The reported weights are (1.0, 0.5, 0.5); the commands specify them
explicitly. The evaluator also supports validation-only weight selection with
`--select-weights`.

Core ranking uses the 82 eligible test issues. `--repo-pool` adds up to 50
same-repository retained hunks from other test issues as T0 candidates, with
sampling and tie breaks seeded at 42. All baseline commands use the same
candidate iterator. nDCG uses gains 3/2/0/0 for T1/T2/T3/T0 and
`k = |T1 union T2|`. T2-Recall is averaged over the 47 eligible issues with T2;
PSR compares T1/T2 against T3 and excludes T0.

```bash
python evaluate.py --checkpoint none --no-projection --exp retrieval "${A[@]}" "${W[@]}"
python evaluate.py --checkpoint checkpoints/m1/best.pt --exp retrieval "${A[@]}" "${W[@]}"
python evaluate.py --checkpoint checkpoints/m2/best.pt --exp retrieval "${A[@]}" "${W[@]}"
python evaluate.py --checkpoint checkpoints/m3/best.pt --exp retrieval "${A[@]}" "${W[@]}"
python evaluate.py --checkpoint checkpoints/m4/best.pt --exp retrieval "${A[@]}" "${W[@]}"

python evaluate.py --checkpoint checkpoints/m2_abl_no_orig/best.pt --exp retrieval "${A[@]}" --score-alpha 1.0 --score-beta 0.5 --score-gamma 0.0
python evaluate.py --checkpoint checkpoints/m2_abl_no_test/best.pt --exp retrieval "${A[@]}" --score-alpha 1.0 --score-beta 0.0 --score-gamma 0.5
python evaluate.py --checkpoint checkpoints/exp2_req_only/best.pt --exp retrieval "${A[@]}" --score-alpha 1.0 --score-beta 0.0 --score-gamma 0.0
python evaluate.py --checkpoint checkpoints/exp2_test_only/best.pt --exp retrieval "${A[@]}" --score-alpha 0.0 --score-beta 1.0 --score-gamma 0.0
python evaluate.py --checkpoint checkpoints/m2_abl_no_req/best.pt --exp retrieval "${A[@]}" --score-alpha 0.0 --score-beta 0.0 --score-gamma 1.0
```

Checkpoints are produced by training and are not supplied by environment setup.
The default loader requires a generated-data provenance stamp. The
`--allow-legacy-checkpoint` option is for inspecting unstamped historical
checkpoints; it does not establish agreement with the current dataset.

## Baseline implementations

```bash
python baselines/baselines_bm25.py "${A[@]}" --output logs/baseline_bm25.json
python baselines/baselines_pretrained.py --model microsoft/codebert-base "${A[@]}" --output logs/baseline_codebert.json
python baselines/baselines_pretrained.py --model microsoft/graphcodebert-base "${A[@]}" --output logs/baseline_graphcodebert.json
python baselines/baselines_extra.py --mode codet5 --model Salesforce/codet5p-110m-embedding "${A[@]}" --output logs/baseline_codet5.json
python baselines/baselines_extra.py --mode coderetriever --model microsoft/graphcodebert-base "${A[@]}" --output logs/baseline_pooler_retrieval.json
python baselines/baselines_extra.py --mode cross_encoder --model cross-encoder/ms-marco-MiniLM-L-6-v2 "${A[@]}" --output logs/baseline_cross_encoder.json
```

BM25 uses lexical matching. CodeBERT and GraphCodeBERT use normalized mean
token embeddings. CodeT5 uses the CodeT5+ 110M embedding checkpoint.
Pooler-based Retrieval uses GraphCodeBERT's pooler output; the pooler is
randomly initialized and receives no task-specific training. The existing
`coderetriever` CLI mode is used for this configuration and is not an
implementation of the published CodeRetriever model. The historical JSON with
that filename records the actual GraphCodeBERT model ID.

The supplied Cross-Encoder command evaluates the pretrained MS MARCO ranker
without retention-label training. A retention-trained Cross-Encoder requires
separate training code and its trained checkpoint, neither supplied here. The
REQ-only command above is the supervised requirement–hunk baseline.

## RQ3 diagnostics and Django figures

```bash
python analysis/exp3_analysis.py --m1 checkpoints/m1/best.pt --m2 checkpoints/m2/best.pt --instances data/processed/instances_full.jsonl --llm-t12 data/cache/llm_t12_hunks.jsonl --tier3 data/cache/tier3_hunks.jsonl --output logs/exp3_analysis.json --analysis all
python analysis/viz_django.py --checkpoint-m2 checkpoints/m2/best.pt --instances data/processed/instances_full.jsonl --llm-t12 data/cache/llm_t12_hunks.jsonl --tier3 data/cache/tier3_hunks.jsonl --out-prefix logs/django_viz --encode-only
python analysis/viz_django.py --out-prefix logs/django_viz --plot-only
python analysis/plot_django_regions.py --embs logs/django_viz_embs.npz --out-prefix logs/django --scatter-only
```

Table VI uses the core test split. Semantic difficulty groups the 82 eligible
issues into 41/41 using M0's mean requirement–T3 cosine similarity. The ORIG
comparison includes test hunks with matched source units and excludes T0.

The Django scripts select all 849 Django issues. Figure 4 fits UMAP on the
untrained projected hunk representation and transforms M2 hunks into that
space; its reference background remains the M0 background. This visualization
uses an untrained projection, unlike the M0 ranking and difficulty baselines.
Figure 5 and the five-region statistics use diagnostic similarities: TEST is a
normalized mean test representation, and missing matched ORIG falls back to
the normalized mean of the issue's first 16 source units. These diagnostic
coordinates differ from the ranking score above. Region percentages are
computed among retained T1/T2 hunks, not all hunks.

Optional region classification must explicitly use the Django data:

```bash
python analysis/analyze_regions.py --npz logs/django_viz_data.npz --instances data/processed/instances_full.jsonl --out logs/django_region_analysis.json
```

Optional edit-type analysis labels generated retained hunks and requires newly
exported scores with `hunk_key`; historical score files without this key cannot
be joined by this script. It does not change T1/T2 labels.

```bash
python evaluate.py --checkpoint checkpoints/m2/best.pt --exp geometry --instances data/processed/instances_full.jsonl --llm-t12 data/cache/llm_t12_hunks.jsonl --tier3 data/cache/tier3_hunks.jsonl "${W[@]}" --save-hunk-scores logs/exp2_hunk_scores_m2.jsonl
python analysis/exp2_label_hunks.py --instances data/processed/instances_full.jsonl --splits data/processed/splits.json --llm-t12 data/cache/llm_t12_hunks.jsonl
python analysis/exp2_analysis.py
```

## RQ4: leave-Django-out and available evidence

```bash
python train.py --instances data/processed/instances_non_django.jsonl --epochs 10 --checkpoint-dir checkpoints/exp4_ldo --pair-types req_hunk req_test orig_hunk req_orig --tier3 data/cache/tier3_hunks.jsonl
D=(--instances data/processed/instances_django_only.jsonl --llm-t12 data/cache/llm_t12_hunks.jsonl --tier3 data/cache/tier3_hunks.jsonl --all-instances --repo-pool)
python evaluate.py --checkpoint checkpoints/exp4_ldo/best.pt --exp retrieval "${D[@]}" "${W[@]}"
```

The leave-Django-out model is trained without Django in either training or
validation and evaluated on the 204 eligible Django issues with T0 distractors.

`../Data/Evaluation/generalization/results/rq4_results.json` summarizes the
historical RQ4 results: EEL 82 issues; leave-Django-out 204; Gemini 41; Qwen 43;
DeepSeek 21; multilingual 109; and 17 GitHub PRs (9 Python, 8 non-Python).
GitHub PRs use nDCG@5 with a different candidate construction and no PSR.
The included logs support the recorded numeric results. Complete historical
candidate inputs for the cross-generator, multilingual and PR experiments are
not included; see `artifact_manifest.json` in the same directory. Their result
records can be inspected, but this guide does not provide a complete rerun from
the available count summaries or partial candidate files.

## Candidate sampling and rebuilding

Training and evaluation above use the published candidates without rebuilding
them. For a new sampling experiment, set `ANTHROPIC_API_KEY` and use a separate
generation output. Samples are batched until the number of matched generated
hunks reaches the reference-hunk count. Duplicate hunks are removed; later
samples are merged into the issue record. If matches exceed that count, the
nearest matches are retained. Later full responses are not all preserved in
the compact records. Reusing a run ID resumes the corresponding sampling round.

```bash
python data_pipeline/sample_haiku_merge.py --input data/raw/swebench_full_instances.jsonl.gz --generations data/cache/new_haiku_generations.jsonl.gz --model claude-haiku-4-5-20251001 --temperature 0.7 --max-tokens 4096 --samples 1 --run-id new-sampling --dry-run
python data_pipeline/sample_haiku_merge.py --input data/raw/swebench_full_instances.jsonl.gz --generations data/cache/new_haiku_generations.jsonl.gz --model claude-haiku-4-5-20251001 --temperature 0.7 --max-tokens 4096 --samples 1 --run-id new-sampling
python data_pipeline/build_dataset.py --generations data/cache/new_haiku_generations.jsonl.gz --instances data/processed/instances_full.jsonl --splits data/processed/splits.json --retained-out data/cache/new_llm_t12_hunks.jsonl --tier3-out data/cache/new_tier3_hunks.jsonl --manifest data/cache/new_dataset_manifest.json
```

New API samples, rebuilt labels and retrained models may yield different
counts or scores; they are not replacements for the recorded historical
results. `GEMINI_API_KEY` is needed for optional edit-type labeling, and
`ANTHROPIC_API_KEY` for optional region classification and sampling.

## Submitted result records

| Paper experiment | Result directory under `../Data/Evaluation/` |
|---|---|
| RQ1 baseline JSONs | `baselines/` |
| EEL, M0–M4, RQ3 training and Table VI analysis | `training_variants/` |
| RQ2 repository-view ablations | `view_ablation/` |
| RQ4 transfer | `generalization/` |
| Human evaluation | `Discussion/` |

Evaluation commands write fresh outputs to `logs/`; training writes selected
weights to `checkpoints/<variant>/best.pt`. Submitted historical records remain
under `Data/Evaluation/`. Numeric agreement alone does not verify a historical
run's candidate construction or checkpoint provenance.
