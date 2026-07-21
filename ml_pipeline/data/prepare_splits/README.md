# Stage00 split preparation

Builds frozen train / **outer_val** parquet splits from raw TRAIN matrices.

## Inputs (not in git)

```
data/raw/bulk_TRAIN/
  X_BULK_TRAIN.parquet
  Y_BULK_TRAIN.parquet
data/raw/sc_TRAIN/
  X_TRAIN_K1.parquet  Y_TRAIN_K1.parquet
  X_TRAIN_PB.parquet  Y_TRAIN_PB.parquet
```

## Run

```bash
export PYTHONPATH=/path/to/ml_pipeline
python data/prepare_splits/prepare_stage00_splits.py
```

## Outputs

```
data/splits/
  bulk/   sc_k1/   sc_pb/
    X_train.parquet  Y_train.parquet
    X_val.parquet    Y_val.parquet   ← outer_val holdout (see SPLIT_PROTOCOL.md)
    train_ids.txt  val_ids.txt  meta.json
  split_summary.json
```

## Protocol

| Dataset | outer_val fraction | Stratification |
|---------|-------------------|----------------|
| bulk    | 15%               | random         |
| sc_pb   | 15%               | PB cohort (K2–K10) |
| sc_k1   | 20%               | random         |

`seed=42`, transform `log2(x+1)` on X and Y.

Full naming and inner_val logic: [`../SPLIT_PROTOCOL.md`](../SPLIT_PROTOCOL.md).
