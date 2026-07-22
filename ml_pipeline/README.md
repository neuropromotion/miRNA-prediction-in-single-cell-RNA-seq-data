# ml_pipeline

All steps from feature selection, model benchmarking up to inference. 

Train data preparation: ../prepare_train_data
Inference data preparation: ../scRNA_inference_data_processing

```
ml_pipeline/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md                # splits / models / inference I/O
│   ├── SPLIT_PROTOCOL.md        # full splitting protocole 
│   ├── prepare_splits/          # Split script for splitting sc_TRAIN/bulk_TRAIN 
│   ├── frozen/                  # small committed artifacts
│   ├── splits/                  # train + outer_val parquets (download)
│   ├── inference_inputs/        # scRNA matrices for batch inference (download)
│   └── inference_outputs/       # prediction CSVs written by batch script
    └── sc_TRAIN/                # single cell train data (see prepare_train_data) .ignored
    └── bulk_TRAIN/              # bulk train data (see prepare_train_data) .ignored
├── shared/                      # shared Python helpers (paths, IO, DL trainers)
├── deps/imputation/             # KNN / NE helpers used by imputation + train
├── pipeline_benchmarking/
│   ├── feature_selection/
│   ├── sc_imputation_selection/
│   ├── model_selection/
│   └── ensembles_selection/
└── final_train_test_inference/
    ├── train/
    ├── test_metrics/
    └── inference/
```

## Quick start

```bash
export PYTHONPATH=/path/to/ml_pipeline
pip install -r requirements.txt

# Download / copy (see data/README.md):
#   1) data/splits/              — train/val matrices + KNN ref
#   2) final_train_test_inference/models/  — pretrained weights
#   3) data/inference_inputs/    — scRNA count matrices for batch inference
# all stuff available on Kaggle (see main page)
```

### Pipeline order

1. `pipeline_benchmarking/feature_selection` → ElasticNet features separe for bulk (trimmed) and sc 
2. `pipeline_benchmarking/sc_imputation_selection` → KNN k=5
3. `pipeline_benchmarking/model_selection` → XGB / CatBoost / TabM / ResNet are winners
4. `pipeline_benchmarking/ensembles_selection` → Ridge stack CatBoost+TabM+ResNet - winner  
5. `final_train_test_inference/train` → retrain on all targets   
6. `final_train_test_inference/inference` → scRNA inference (`data/inference_inputs` → `data/inference_outputs`)

Each stage folder has its own README with **Inputs / Run / Outputs**.

**Data splits:** read [`data/SPLIT_PROTOCOL.md`](data/SPLIT_PROTOCOL.md) before interpreting any `outer_val_*` metric.

## Preprocessing note (scaler fit noise)

For deep-learning and some candidate models we fit `StandardScaler` or `QuantileTransformer` on training features. Expression matrices are sparse and zero-inflated, so many features have duplicate or near-constant values. Before **fitting** the scaler we add a tiny Gaussian jitter 𝒩(0, 10⁻⁵) with a fixed seed (`SEED`):

```python
noise = rng.normal(0.0, 1e-5, x_train.shape)
scaler.fit(x_train + noise)
```

This is only for numerical stability of the scaler fit (ties / near-zero variance). It is **not** training-time data augmentation: `transform()` still uses the original matrices, and tree models (XGBoost / CatBoost) do not use this step.

Implemented in `shared/dl_trainers.py`, `shared/tabm_wrapper.py`, `final_train_test_inference/train/dl_trainers.py`, and the LassoNet/GANDALF trainers under `pipeline_benchmarking/model_selection/`.

## Environment

- Python ≥ 3.10  
- GPU recommended for DL models  
- See `requirements.txt`
