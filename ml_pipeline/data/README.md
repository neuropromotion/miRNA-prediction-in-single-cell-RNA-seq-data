# Data layout

Large matrices, pretrained models, and scRNA inference inputs are **not** stored in git.
Place them under this `data/` tree after download.

**Splitting strategy (train / inner_val / outer_val / sc_TEST):** see [`SPLIT_PROTOCOL.md`](SPLIT_PROTOCOL.md).

---

## 0) TRAIN matrices + Stage00 splits

Train and test maticies were generated in prepare_train_data section from publicaly available data. Then splitted into train and validation cohorts for banchmarking and training purposes

See [`prepare_splits/README.md`](prepare_splits/README.md).

---

## 1) Training / benchmarking splits 

```
data/splits/
  bulk/
    X_train.parquet  Y_train.parquet
    X_val.parquet    Y_val.parquet    # outer_val holdout (not sc_TEST)
  sc_k1/                              # K1 - pure single cell
    X_train.parquet  Y_train.parquet
    X_val.parquet    Y_val.parquet
  sc_pb/                                # K2,K3,K4,K5 and K10 pseudobulks
    X_train.parquet  Y_train.parquet  
    X_val.parquet    Y_val.parquet
  split_summary.json
```

Public link: see repository root README (*Training and inference datasets*).

`sc_k1/X_train.parquet` is also the **KNN imputation reference** for inference. It is single-cell K1 Train part; it serves as the KNN imputing reference.

---

## 2) Pretrained models (for inference without re-training)

```
final_train_test_inference/models/
  ensemble/catboost_tabm_resnet_stack/weights/<mirna>.json
  catboost_optuna/...   # base models as produced by train/
  tabm/...
  resnet/...
```

Public link: see repository root README (*Pretrained models*).

---

## 3) scRNA datasets for large-scale inference

Inference script: `final_train_test_inference/inference/total_inference/total_inference.py`

Default paths (override with CLI flags):

```
data/inference_inputs/     # put *.parquet / *.csv count matrices here
data/inference_outputs/    # predict_all writes one CSV per dataset here
```

Public link: see repository root README (*Training and inference datasets*).

Example:

```bash
export PYTHONPATH=/path/to/ml_pipeline
mkdir -p data/inference_inputs data/inference_outputs
python final_train_test_inference/inference/total_inference/total_inference.py \
  --input-dir data/inference_inputs \
  --output-dir data/inference_outputs
```

Processed / annotated scRNA pipelines: see repo folder `scRNA_inference_data_processing/`.

---

## Frozen artifacts (committed in git)

| File | Role |
|------|------|
| `frozen/selected_targets.txt` | 50 pilot miRNAs for benchmarking |
| `frozen/selected_features.json` | Final gene lists per miRNA |
| `frozen/split_summary.json` | Split metadata / full target list |
