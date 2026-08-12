# Final train — TabPack + DCNv2 + TabM

Winner stack from `ensembles_selection_v4`: **`tabpack_dcnv2_tabm_stack`**.

| Module | Role |
|--------|------|
| `tabpack_trainer.py` | TabPack Muon paper (`n_models=32`) |
| `torch_trainers.py` / `dl_trainers.py` | DCNv2 AdamW + TabM AdamW |
| `model_trainers.py` | Dispatch |
| `run_train.py` | Per-model loop (`FINAL_MODEL=…`) |
| `run_stack.py` | Ridge stack |
| `run_all.sh` | Full 3-model + stack pipeline |

## Layout after prep

```
train/
  results/{tabpack,dcnv2,tabm}/models/
  results/ensemble/tabpack_dcnv2_tabm_stack/
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

Smoke:

```bash
FINAL_TARGETS=hsa-mir-1180-3p bash run_dcnv2.sh
FINAL_TARGETS=hsa-mir-1180-3p bash run_tabm.sh
FINAL_TARGETS=hsa-mir-1180-3p CUDA_VISIBLE_DEVICES=0 bash run_tabpack.sh
```

## Protocol

- Fit: full Stage00 train  
- ES / TabPack patience: Stage00 val  
- Stack tune: `val_k1` + `val_pb_K*` (no bulk)  
- True holdout: `sc_TEST` / `bulk_TEST` under `../test_metrics/` or top-level `test_metrics/`

**TabPack** persists cached `preds.npz` only (no live `predict(x)` for new cells yet).
