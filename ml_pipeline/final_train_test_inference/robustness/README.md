# robustness v3

Estimates per–miRNA **`max_absent_frac`**: largest fraction of the SC panel that can be
globally zeroed while masked eval-half R² stays above **`retention_ratio × r2_reference`**.

## Method

- Eval half of `sc_TEST` (Optimal_K split).
- **Binary search** on `p ∈ [0, 0.95]` until interval width **&lt; 0.01**.
- **200** random masks per bisect probe; **1000** masks at the final `p`.
- **Retention statistic** (CLI `--retention-stat`):
  - `median` (default) — `median(R²_masks) ≥ τ`
  - `q25` — `Q25(R²_masks) ≥ τ` (stricter; re-run pipeline with this flag)

Reference `r2_reference`: bootstrap `sc_r2_median` when cohort matches, else `test_sc_r2` from `prediction_config.json`.

## Setup (once per machine / after reboot)

```bash
cd final_train_test_inference/robustness
./setup_venv.sh
conda deactivate   # important on Mac
source ./setup_env.sh
./warm_eval_cache.sh
```

## Run (manual chunks)

```bash
source ./setup_env.sh
./run_next_chunk.sh 10
```

Progress: `grep -c ',done,' results/tables/_progress.csv`

### Q25 instead of median

```bash
export ROBUSTNESS_RETENTION_STAT=q25
# optional: fresh results dir or delete _progress.csv to recompute all
./run_robustness.py --retention-stat q25 --max-targets 5
```

## Outputs (`results/`)

| File | Content |
|------|---------|
| `robustness_config.json` | v3 inference tolerances |
| `tables/tolerance_summary.csv` | one row per completed miRNA |
| `tables/_progress.csv` | resume state |
| `journal.log` | human log |

Optional: `--save-probes` → `tables/bisect_probes.csv`

Legacy grid search pipeline: `../robustness_/`
