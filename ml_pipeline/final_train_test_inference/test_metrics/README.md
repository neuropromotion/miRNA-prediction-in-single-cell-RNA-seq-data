# TEST metrics (eval half)

Final holdout evaluation for **Optimal_K-eligible** miRNAs only.

## Protocol

- Split: `../Optimal_K/results/test_split.json` — **eval** indices only
- Assignments: `../Optimal_K/results/proto_prediction_config.json`
- Predictions: reuse `../Optimal_K/results/pred_cache/`
- Per target:
  - **SC** on assigned K only (no K1–K10 menu)
  - **bulk** on eval bulk half
  - B=1000 bootstrap → R² and MSE distributions
  - Summary stats: mean / median / std / q05 / q25 / q75 / q95  
    plus point estimate on the full eval slice (`*_full`)

## Run

```bash
bash run.sh
```

## Outputs

- `tables/per_target_bootstrap_summary.csv` — one row per eligible target
- `tables/sc_summary.csv`, `bulk_summary.csv`, `overall_summary.csv`
- `figures/*.png` + `*.pdf`
- `results/prediction_config.json` — **production** config for inference  
  (`features` + `test_bulk` / `test_optimal_k` = eval-half bootstrap median R²;  
  no `version` / `assignment_rule` / `thresholds` / `split_path` / `features_source`)

Copy to inference when ready:

```bash
cp results/prediction_config.json ../inference/prediction_config.json
```
