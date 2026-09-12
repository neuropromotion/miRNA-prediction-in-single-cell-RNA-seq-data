# Final train — TabPack + TabM + XGB Optuna

Production stack: **`tabpack_tabm_xgb_stack`**.

| Module | Role |
|--------|------|
| `tabpack_trainer.py` | TabPack Muon paper (`n_models=32`) |
| `torch_trainers.py` / `dl_trainers.py` | TabM AdamW |
| `xgb_trainer.py` | XGB Optuna (15 trials, recipe from `02_model_selection`) |
| `model_trainers.py` | Dispatch |
| `run_train.py` | Per-model loop (`FINAL_MODEL=…`) |
| `run_stack.py` | Ridge stack |
| `run_all.sh` | Full 3-model + stack pipeline |

## Layout

```
train/
  results/{tabpack,tabm,xgb_optuna}/models/
  results/ensemble/tabpack_tabm_xgb_stack/
  logs/
```

## Targets

All columns from `Y_train` **minus** `zero_expressed_mirs.txt` (15 miRNAs) → **312** targets.
Exclusion is applied in `io_splits.load_targets()` for both base training and stack.

## Run

```bash
cd ml_pipeline/final_train_test_inference/train
bash run_all.sh
```

XGB only (3 GPU shards):

```bash
N_SHARDS=3
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$i FINAL_SHARD=$i/$N_SHARDS FINAL_METRICS_SUFFIX=_shard$i \
    bash run_xgb.sh > logs/xgb_shard$i.log 2>&1 &
done
wait
…/deps/tabpack/.venv/bin/python -c 'from merge_metrics import merge_model; merge_model("xgb_optuna")'
```

Smoke:

```bash
FINAL_TARGETS=hsa-mir-1180-3p bash run_xgb.sh
FINAL_TARGETS=hsa-mir-1180-3p bash run_tabm.sh
FINAL_TARGETS=hsa-mir-1180-3p CUDA_VISIBLE_DEVICES=0 bash run_tabpack.sh
```

## Protocol

- Fit: `data/splits` (Stage00 / current TRAIN)  
- ES / TabPack patience: val  
- Stack tune: `val_k1` + `val_pb_K*` (no bulk)  
- True holdout: `sc_TEST` / `bulk_TEST` under `../test_metrics/`
