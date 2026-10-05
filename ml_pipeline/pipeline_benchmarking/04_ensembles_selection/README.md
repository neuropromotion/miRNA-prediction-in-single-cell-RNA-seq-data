# Ensemble selection v4

## Protocol

| Role | Data |
|------|------|
| **Tune** weights | `inner_val` **K1 + PB** only (bulk excluded) |
| **Report / rank** | `outer_val` K1 + PB + bulk |
 

Summary also reports:
- `avg_of_medians_K` — mean of per-cohort **medians** over K1 + PB K2–K10
- `avg_of_means_K` — mean of per-cohort **means** over the same cohorts
- `n_best_outer_val_k1` / `n_best_outer_val_bulk` — #targets where this ensemble is best (ties count for all)
- `n_best_unique_*` — sole winners only (no ties)

Win counts are among ensembles in this stage (not vs solos). Per-target winners: `results/best_per_target_outer_val_{k1,bulk}.csv`.

## Base models

| id | Source | Recipe |
|----|--------|--------|
| `xgb_optuna` | `model_selection` | XGB + Optuna |
| `tabpack` | `model_tuning` | TabPack + Muon | 
| `tabm` | `model_selection` | TabM + AdamW |

Sets = all pairs + triple:

- pairs (3): `xgb_tabpack`, `xgb_tabm`, `tabpack_tabm`
- triples (1): `xgb_tabpack_tabm`

→ **4 sets × 3 methods = 12** configs.

## Methods

| id | Meaning |
|----|---------|
| `blend` | non-negative weights on simplex (grid) |
| `avg_uniform` | equal average of predictions |
| `stack` | Ridge meta-learner (`RidgeCV`) |
 

## Benchmarking resutls
Based on evaluation metrics, stacking of XGBoost, TabPack and TabM outperformed the rest of ensembles, despite that differences between ensebmles were subtle. But Tripple stacking is single ensemble that increased r2 above 0.4 for 23 miRNAs (best result). The top models were ranked by their average of means and medians $R^2$ scores among all cohorts (K1-K10):

| Rank | Model | Average of medians $R^2$ | Model | Average of means $R^2$ |
| :---: | :--- | :---: | :--- | :---: |
| **1** | **xgb_tabpack_tabm_stack** | 0.8263 | **xgb_tabpack_tabm_stack** | 0.8116 |
| **2** | **tabpack_tabm_stack** | 0.8256 | **xgb_tabpack_stack** | 0.8102 |
| **3** | **xgb_tabpack_stack** | 0.8237 | **xgb_tabpack_tabm_blend** | 0.81 |
| **4** | **xgb_tabpack_tabm_blend** | 0.8222 | **tabpack_tabm_stack** | 0.8082 |