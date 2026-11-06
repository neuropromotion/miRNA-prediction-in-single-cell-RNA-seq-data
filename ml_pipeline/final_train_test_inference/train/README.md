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


## Targets

15 miRNAs (`zero_expressed_mirs.txt`) were excluded due to zero expression in GTEx train data. Resulted set is 312 miRNAs (models)
