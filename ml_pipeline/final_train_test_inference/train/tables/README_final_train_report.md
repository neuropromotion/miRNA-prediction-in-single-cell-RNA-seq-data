# Final train report — Stage00 validation

Ensemble: `tabpack_dcnv2_tabm_stack`  
Bases: TabPack Muon + DCNv2 AdamW + TabM AdamW → Ridge stack  
Targets: **312** (15 zero-expressed excluded)  
Stack fallbacks to best solo: **13** / 312  
Tune splits: `val_k1` + `val_pb_K*` (no bulk)

## Median / mean R²

| model | n | median val K1 | mean val K1 | median val PB (all) | mean val PB (all) | median val bulk | mean val bulk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabPack Muon | 312 | 0.373 | 0.4058 | 0.9183 | 0.9037 | 0.8083 | 0.7553 |
| DCNv2 AdamW | 312 | 0.2481 | 0.2895 | 0.8507 | 0.8234 | 0.7598 | 0.6884 |
| TabM AdamW | 312 | 0.228 | 0.2935 | 0.892 | 0.8734 | 0.7862 | 0.7348 |
| Ridge stack | 312 | 0.4021 | 0.4247 | 0.9245 | 0.9115 | 0.794 | 0.7383 |

Stack − best solo on val K1: median **0.0093**, mean **0.0092**, stack wins on **72.1%** targets

## n_best (argmax R² per target)

| split | model | n_best | frac_best |
| --- | --- | --- | --- |
| val K1 | TabPack Muon | 54 | 0.17307692307692307 |
| val K1 | DCNv2 AdamW | 21 | 0.0673076923076923 |
| val K1 | TabM AdamW | 12 | 0.038461538461538464 |
| val K1 | Ridge stack | 225 | 0.7211538461538461 |
| val PB (all) | TabPack Muon | 52 | 0.16666666666666666 |
| val PB (all) | DCNv2 AdamW | 0 | 0.0 |
| val PB (all) | TabM AdamW | 19 | 0.060897435897435896 |
| val PB (all) | Ridge stack | 241 | 0.7724358974358975 |
| val bulk | TabPack Muon | 269 | 0.8621794871794872 |
| val bulk | DCNv2 AdamW | 0 | 0.0 |
| val bulk | TabM AdamW | 22 | 0.07051282051282051 |
| val bulk | Ridge stack | 21 | 0.0673076923076923 |

## Files

- `tables/summary_by_model.csv` — full mean/median/std/quantiles
- `tables/summary_compact.csv` — primary splits only
- `tables/per_target_all_models.csv` — wide join + deltas
- `tables/n_best_by_split.csv`
- `figures/*.pdf` + `*.png`
- weight tables/figures from `plot_stack_weights.py` in the same dirs

True holdout (`sc_TEST` / `bulk_TEST`) is not in this report.
