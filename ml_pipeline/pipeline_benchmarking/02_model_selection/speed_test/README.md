# Speed benchmarking (Phase 1)

Scripts to measure training and inference time before the full model screen.
All benchmarks use the same **5 fixed pilot miRNAs** (`SPEED_TARGETS` in `constants.py`):

- `hsa-mir-125a-5p`
- `hsa-mir-301b-5p`
- `hsa-mir-411-5p`
- `hsa-mir-487a-3p`
- `hsa-mir-99b-3p`

Train pool: Stage00 train split with runtime inner 85/15 split (`shared/data.py`).
Inference timing uses the Stage00 **outer_val** holdout (bulk + K1 + pseudobulk K2–K10).

## Scripts

| Script | Models | Targets | Output |
|--------|--------|---------|--------|
| `speed_benchmark.py` | 7 screen models + FT-Transformer | 5 fixed | `results/speed_benchmark/` |
| `tabpfn3_speed_benchmark.py` | TabPFN-3 (full train) | 5 fixed | `results/tabpfn3_speed/` |
| `fttransformer_speed_50.py` | FT-Transformer (train only) | all 50 pilot | `results/fttransformer_speed_50/` |
| `speed_benchmark_candidates.py` | LassoNet, GANDALF | 5 fixed | `results/speed_candidates/` |
| `build_speed_comparison.py` | merge main + TabPFN-3 | — | `tables/speed_comparison_all_01.tsv` |

Committed summary table: `tables/speed_comparison_all_01.tsv` (from the original run).

## Run (local)

```bash
export PYTHONPATH=/path/to/ml_pipeline:/path/to/ml_pipeline/pipeline_benchmarking/model_selection
cd ml_pipeline/pipeline_benchmarking/model_selection

python speed_test/speed_benchmark.py
python speed_test/tabpfn3_speed_benchmark.py   # needs HF_TOKEN
python speed_test/build_speed_comparison.py
```

FT-Transformer on all 50 targets (GPU, long run):

```bash
bash speed_test/run_fttransformer_speed_50.sh
```

LassoNet / GANDALF candidates (GPU + extra deps):

```bash
bash speed_test/run_speed_candidates.sh
```

## TabPFN-3 deps

Use an isolated env — see `requirements-tabpfn3.txt` (`tabpfn==8.0.8`).
Set `HF_TOKEN` (HuggingFace read token with TabPFN-3 license accepted).

## Notes

- `speed_benchmark.py` includes `fttransformer` alongside the 7 screen models; TabPFN was moved to a separate script after import failures in the main stack.
- `fttransformer_speed_50.py` extrapolates total training time for 50/327 miRNAs (train only, no inference).
- `build_speed_comparison.py` expects `results/speed_benchmark/speed_results.csv` and `results/tabpfn3_speed/speed_results.csv`.
