# Data splitting protocol

This document describes **all** train / validation folds used in `ml_pipeline`, how they relate to each other, and what is **not** included here (held-out `sc_TEST`).

## Overview (three levels)

```
Level 0 — Raw TRAIN corpus (not in git; see data/raw/)
  bulk_TRAIN + sc_TRAIN
        │
        ▼  Stage00 split  [prepare_splits/prepare_stage00_splits.py]
        │  Writes data/splits/*/X_train|X_val.parquet
        │
        ├─ train fold  ──► training pool for benchmarking
        │                      │
        │                      ▼  Inner split at runtime  [shared/data.py → build_modality_bundle()]
        │                      ├─ inner train  (85% of pool, stratified by modality)
        │                      └─ inner_val    (15% of pool) — early stopping, base-model metrics
        │
        └─ val fold (parquet X_val) ──► outer_val  — benchmarking holdout
                                          (ensemble tuning on SC only; reporting on bulk+SC)

Level 1 — Held-out test (separate repo / raw data; NEVER used in pipeline_benchmarking)
  sc_TEST  — final generalization metrics (outside this repository)
```

## Terminology

| Name | Where stored | Role |
|------|--------------|------|
| **train fold** (Stage00) | `data/splits/*/X_train.parquet` | Pool for benchmarking model fitting |
| **inner train** | Created in memory only | Actual fit data after 85/15 split of Stage00 train pool |
| **inner_val** | Created in memory only | Early stopping + `inner_val_r2` for base models |
| **outer_val** | `data/splits/*/X_val.parquet` | Holdout from TRAIN corpus; used for model/ensemble **selection** and benchmarking reports |
| **sc_TEST** | External | True held-out test; **not** loaded by any script in `pipeline_benchmarking/` |

> **Important:** Parquet files are named `X_val.parquet` / `Y_val.parquet` on disk (Stage00 convention). Code loads them via `load_*_outer_val()` and exposes them as `outer_val_*` in bundles. They are **not** the final test set.

---

## Stage00 — train vs outer_val

**Script:** `data/prepare_splits/prepare_stage00_splits.py`  
**Output:** `data/splits/{bulk,sc_k1,sc_pb}/`

| Dataset | Train % | outer_val % | Notes |
|---------|---------|-------------|-------|
| bulk | 85% | 15% | random, seed=42 |
| sc_pb | 85% | 15% | stratified by pseudobulk cohort (K2–K10) |
| sc_k1 | 80% | 20% | random, seed=42 |

Both X and Y are transformed with `log2(x+1)` before saving.

After download from Kaggle, splits are ready under `data/splits/`; re-run the script only if you need to regenerate from `data/raw/`.

---

## Inner split — runtime only (not on disk)

**Code:** `shared/data.py` → `build_modality_bundle()`

Used by:
- `pipeline_benchmarking/model_selection/run_model_screen.py`
- `pipeline_benchmarking/ensembles_selection/run_ensembles.py`
- `pipeline_benchmarking/model_selection/speed_test/*`

Steps:
1. Load Stage00 **train** folds: bulk + K1 (KNN-imputed) + PB.
2. Concatenate into one pool tagged by modality (`bulk`, `k1`, `pb`).
3. `sklearn.train_test_split(test_size=0.15, stratify=modality, random_state=42)`.
4. Load Stage00 **outer_val** folds separately (never mixed into the pool).

Sample weights: inverse modality frequency on **inner train** only.

Inner split IDs are **not** materialized to disk (reproducible given Stage00 parquets + `INNER_VAL_FRAC` + `SEED` in `shared/paths.py`). Re-running `build_modality_bundle()` always yields the same inner train / inner_val partition.

---

## What each stage evaluates on

### Feature selection & imputation screening
Use Stage00 train pool and outer_val via stage-specific loaders (`feature_selection/io_data.py`, etc.).

### Model selection (`model_selection/`)
| Split | Used for |
|-------|----------|
| inner train | Fit base models |
| inner_val | Early stopping (DL), `inner_val_r2` reporting |
| outer_val (bulk, K1, PB K2–K10) | Report `outer_val_*_r2`; compare architectures |

Metrics files: `results/<model>/outer_val_metrics.csv`

### Ensemble selection (`ensembles_selection/`)
| Split | Used for |
|-------|----------|
| inner train | (base models already trained on inner train) |
| inner_val | Reported in `outer_val_metrics.csv` |
| outer_val K1 + PB K2–K10 | **Tune** ensemble weights (`TUNE_SPLITS`) |
| outer_val bulk | Report only (`EVAL_SPLITS`; not used for ensemble tuning) |

Ensemble comparison (median K1 R², etc.) is on **outer_val**, not on `sc_TEST`.

### Final training (`final_train_test_inference/train/`)
Uses Stage00 train + outer_val parquets directly (no inner split):
- **train** = pooled Stage00 train folds
- **val** = pooled Stage00 outer_val folds (named `val_*` in final_train code)
- Ridge stack tuning: `val_k1`, `val_pb_*` (same samples as benchmarking outer_val SC)

Final publication metrics on **sc_TEST** are computed outside this repository.

---

## File / column naming map

| Old name (deprecated) | New name |
|----------------------|----------|
| `load_bulk_outer_val()` | `load_bulk_outer_val()` |
| `x_outer_val_k1` | `x_outer_val_k1` |
| `outer_val_k1_r2` | `outer_val_k1_r2` |
| `outer_val_metrics.csv` | `outer_val_metrics.csv` |
| split key `test_k1` | `outer_val_k1` |

---

## Reproducibility checklist

1. Place raw TRAIN matrices in `data/raw/` (or download prepared `data/splits/` from Kaggle).
2. (Optional) Regenerate Stage00: `python data/prepare_splits/prepare_stage00_splits.py`
3. `export PYTHONPATH=/path/to/ml_pipeline`
4. Run benchmarking stages in order (see root `README.md`).
5. Inner_val is created automatically; no extra download required.

---

## FAQ for reviewers

**Q: Did you tune ensembles on the final test set?**  
No. Ensemble weights are tuned on **outer_val** single-cell folds (Stage00 val = holdout from TRAIN). The final **sc_TEST** set is not accessed during `pipeline_benchmarking/`.

**Q: Why is there an inner_val if Stage00 already has a val fold?**  
Stage00 outer_val is reserved for comparing models and ensembles. Inner_val is carved from the Stage00 **train** pool so base models can use early stopping without peeking at outer_val.

**Q: Where is inner_val defined?**  
Only in `shared/data.py` (`build_modality_bundle()`), at the start of model_selection and ensembles_selection runs.
