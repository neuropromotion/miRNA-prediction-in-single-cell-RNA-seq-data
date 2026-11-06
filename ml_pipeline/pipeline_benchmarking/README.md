# Pipeline benchmarking

| Order | Folder | Purpose | Result |
|------:|--------|-------------------|------------|
| 1 | `00_feature_selection/` | feature selection strategies benchmarking | ElasticNet separate for sc-derived and bulk-derived features|
| 2 | `01_sc_imputation_selection/` | imputation methods benchmarking | KNN k=5 |
| 3 | `02_model_selection/` | 12 models benchmarking and selection best 4 | CatBoost / TabM / ResNet + XGBoost |
| 4 | `03_optimizer_selection/` | 13 models benchmarking | TabPack / TabM / XGBoost |
| 5 | `04_ensembles_selection/` | ensemble types benchmarking | Ridge stack TabPack+TabM+XGBoost |
