# Final-train report tables

Ensemble: `tabpack_tabm_xgb_stack`  
Bases: TabPack Muon + TabM AdamW + XGB Optuna → Ridge stack  

Regenerated after XGB Optuna + new Ridge stack (2026-09-03).

## Summary (median R²)

| model | n | val K1 | val PB | val bulk |
|-------|--:|------:|-------:|---------:|
| TabPack Muon | 312 | 0.373 | 0.918 | 0.808 |
| TabM AdamW | 312 | 0.228 | 0.892 | 0.786 |
| XGB Optuna | 312 | 0.161 | 0.805 | 0.791 |
| Ridge stack | 312 | **0.399** | **0.925** | 0.794 |

Stack wins ~77% of targets on val K1. Ridge coef abs-share (non-fallback): TabPack ~0.67, TabM ~0.21, XGB ~0.13.

Figures: `../figures/`  
Tables: this directory.
