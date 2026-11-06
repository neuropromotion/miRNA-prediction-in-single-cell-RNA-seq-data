## Test metrics 

This is the **final** quality assessment on the **eval** half of the same holdout TEST for which `Optimal_K` fixed the split and K assignments. It produces report tables, figures, and the production **`prediction_config.json`**: the list of miRNAs available for inference, with metrics and feature panels. Metrics (R² and MSE) were calculated using bootstrap (n=1000) generating as median bootstrap-derived values.

## Outputs

- `tables/per_target_bootstrap_summary.csv` — main per-miRNA table.
- Aggregates for sc/bulk/overall; figures under `figures/` if generated.
- `results/prediction_config.json` — copy or sync to **`../inference/prediction_config.json`** as the reference for user inference.