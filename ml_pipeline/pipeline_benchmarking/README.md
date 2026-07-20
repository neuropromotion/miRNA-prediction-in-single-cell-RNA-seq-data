# Pipeline benchmarking

Four sequential experiment stages. All paths are relative to the `ml_pipeline` root (`GITHUB/` locally).

Set before running:

```bash
export PYTHONPATH=/path/to/ml_pipeline
```

| Order | Folder | Question answered | Key output |
|------:|--------|-------------------|------------|
| 1 | `feature_selection/` | Which feature selector? | ElasticNet (+ bulk trim) |
| 2 | `sc_imputation_selection/` | Which SC imputer? | KNN k=5 |
| 3 | `model_selection/` | Which base models? | CatBoost / TabM / ResNet (+ others screened) |
| 4 | `ensembles_selection/` | How to combine? | Ridge stack CatBoost+TabM+ResNet |

## Shared dependencies

- `../shared/` — IO, splits, DL trainers, path config  
- `../data/splits/` — prepared matrices (download)  
- `../data/frozen/` — pilot targets + feature JSON  

## Reproduce vs inspect

- **Inspect results:** each stage already contains `tables/` and `figures/` used in the paper.  
- **Full re-run:** requires GPU, `data/splits/`, and several hours–days depending on the stage.
