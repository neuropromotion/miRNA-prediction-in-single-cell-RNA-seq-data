# Pipeline benchmarking

| Order | Folder | Purpose | Result |
|------:|--------|-------------------|------------|
| 1 | `feature_selection/` | feature selection strategies benchmarking | ElasticNet separate for sc and bulk|
| 2 | `sc_imputation_selection/` | imputation methods benchmarking | KNN k=5 |
| 3 | `model_selection/` | 12 models benchmarking and selection best 4 | CatBoost / TabM / ResNet + XGBoost |
| 4 | `ensembles_selection/` | ensemble types benchmarking | Ridge stack CatBoost+TabM+ResNet |
