# Ensemble selection v4

## Protocol

| Role | Data |
|------|------|
| **Tune** weights | `inner_val` **K1 + PB** only (bulk excluded) |
| **Report / rank** | `outer_val` K1 + PB + bulk |
| Primary rank | `median_outer_val_k1_r2` |

Summary also reports:
- `avg_of_medians_K` — mean of per-cohort **medians** over K1 + PB K2–K10
- `avg_of_means_K` — mean of per-cohort **means** over the same cohorts
- `n_best_outer_val_k1` / `n_best_outer_val_bulk` — #targets where this ensemble is best (ties count for all)
- `n_best_unique_*` — sole winners only (no ties)

Win counts are among ensembles in this stage (not vs solos). Per-target winners: `results/best_per_target_outer_val_{k1,bulk}.csv`.

## Base models

| id | Source | Recipe |
|----|--------|--------|
| `xgb_optuna` | `model_selection` | XGB Optuna |
| `tabpack` | `model_tuning` | TabPack Muon (paper) |
| `dcnv2` | `model_selection` | DCNv2 AdamW |
| `tabm` | `model_selection` | TabM AdamW |

Sets = all pairs + triples (full quadruple excluded):

- pairs (6): `xgb_tabpack`, `xgb_dcnv2`, `xgb_tabm`, `tabpack_dcnv2`, `tabpack_tabm`, `dcnv2_tabm`
- triples (4): `xgb_tabpack_dcnv2`, `xgb_tabpack_tabm`, `xgb_dcnv2_tabm`, `tabpack_dcnv2_tabm`

→ **10 sets × 3 methods = 30** configs.

## Methods

| id | Meaning |
|----|---------|
| `blend` | non-negative weights on simplex (grid) |
| `avg_uniform` | equal average of predictions |
| `stack` | Ridge meta-learner (`RidgeCV`) |

TabPack uses cached `preds.npz` (no live re-inference).

## Run

```bash
cd ml_pipeline/pipeline_benchmarking/ensembles_selection_v4
bash run_docker.sh
```

Smoke:

```bash
STAGE04_SETS=xgb_tabpack STAGE04_METHODS=avg_uniform \
  STAGE04_TARGETS=hsa-mir-1180-3p STAGE04_DEVICE=cuda bash run_docker.sh
```
