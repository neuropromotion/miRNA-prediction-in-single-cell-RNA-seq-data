## Layout

| Subfolder | Role |
|-----------|------|
| `train/` | Train CatBoost + TabM + ResNet and fit Ridge stack |
| `test_metrics/` | Test R² tables + cohort config / figures |
| `inference/` | Single-cell inference (`preprocessor`, `stack_predictor`) |

## Inputs

- `../data/splits/` prepared matrices
- `selected_features.json` (this folder; synced from frozen features)
- Pretrained weights (optional for inference-only): `models/` — see `../data/README.md`

## Run training

```bash
export PYTHONPATH=/path/to/ml_pipeline
cd train
python run_train.py
python run_stack.py
```

## Run held-out TEST evaluation

See `test_metrics/README.md`. Scripts live in `test_metrics/test_evaluating/` and write:

- `test_metrics/bulk_test_metrics.csv`
- `test_metrics/K1_K10_test_metrics.csv`

## Run inference

See `inference/README.md`. Requires weights under `models/` and KNN ref `../data/splits/sc_k1/X_train.parquet`.

## Final Model

The final training, testing, and inference pipeline is based on the best-performing approach identified during model benchmarking.

### Architecture

The final ensemble consists of:

- CatBoost (hyperparameters optimized using Optuna)
- TabM
- ResNet-like neural network

### Ensembling Strategy

Predictions from the individual models are combined using a stacking framework with Ridge Regression as the meta-learner.
