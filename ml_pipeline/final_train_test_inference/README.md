# train → test → inference

1. **`train/`** — Train the final ensemble (Ridge TabPack+TabM+XGBoost).
2. **`Optimal_K/`** — On held-out scRNA/bulk **TEST**: eligibility rules, **optimal cohort K** per miRNA
3. **`test_metrics/`** — **Final** evaluation and metrics for eligible models
4. **`robustness/`** — Feature importance and robustness to feature dropout evaluating 
5. **`inference/`** — Prediction miRNAs on independent scRNA-seq datasets

