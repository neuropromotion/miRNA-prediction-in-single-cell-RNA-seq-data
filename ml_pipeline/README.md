# ml_pipeline

All steps from feature selection, model benchmarking up to inference. 

Train data preparation: ../prepare_train_data
Inference data preparation: ../scRNA_inference_data_processing

```
ml_pipeline/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md                # splits / models / inference I/O
│   ├── frozen/                  # small committed artifacts
│   ├── splits/                  # train/val parquet splits (download)
│   ├── inference_inputs/        # scRNA matrices for batch inference (download)
│   └── inference_outputs/       # prediction CSVs written by batch script
├── shared/                      # shared Python helpers (paths, IO, DL trainers)
├── deps/imputation/             # KNN / NE helpers used by imputation + train
├── pipeline_benchmarking/
│   ├── feature_selection/
│   ├── sc_imputation_selection/
│   ├── model_selection/
│   └── ensembles_selection/
└── final_train_test_inference/
    ├── train/
    ├── test_metrics/
    └── inference/
```

## Quick start

```bash
export PYTHONPATH=/path/to/ml_pipeline
pip install -r requirements.txt

# Download / copy (see data/README.md):
#   1) data/splits/              — train/val matrices + KNN ref
#   2) final_train_test_inference/models/  — pretrained weights
#   3) data/inference_inputs/    — scRNA count matrices for batch inference
```

### Pipeline order

1. `pipeline_benchmarking/feature_selection` → ElasticNet features  
2. `pipeline_benchmarking/sc_imputation_selection` → KNN k=5  
3. `pipeline_benchmarking/model_selection` → CatBoost / TabM / ResNet among others  
4. `pipeline_benchmarking/ensembles_selection` → Ridge stack CatBoost+TabM+ResNet  
5. `final_train_test_inference/train` → retrain on all targets  
6. `final_train_test_inference/inference` → scRNA inference (`data/inference_inputs` → `data/inference_outputs`)

Each stage folder has its own README with **Inputs / Run / Outputs**.

## Environment

- Python ≥ 3.10  
- GPU recommended for DL models  
- See `requirements.txt`
