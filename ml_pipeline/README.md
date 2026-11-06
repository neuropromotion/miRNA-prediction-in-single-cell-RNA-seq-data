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
│   ├── SPLIT_PROTOCOL.md        # full splitting protocole 
│   ├── prepare_splits/          # Split script for splitting sc_TRAIN/bulk_TRAIN 
│   ├── frozen/                  # small committed artifacts
│   ├── splits/                  # train + outer_val parquets (download)
    ├── sc_TRAIN/                # single cell train data (see prepare_train_data) .ignored
    └── bulk_TRAIN/              # bulk train data (see prepare_train_data) .ignored
├── shared/                      # shared Python helpers (paths, IO, DL trainers)
├── deps/imputation/             # KNN / NE helpers used by imputation + train
├── pipeline_benchmarking/
│   ├── 00_feature_selection/
│   ├── 01_sc_imputation_selection/
│   ├── 02_model_selection/
│   ├── 03_optimizer_selection/
│   └── 04_ensembles_selection/
└── final_train_test_inference/
    ├── train/
    ├── Optimal_K/
    ├── test_metrics/
    ├── robustness/
    └── inference/
```

### Pipeline order

1. `pipeline_benchmarking/00_feature_selection` - feature selection strategy benchmarking
2. `pipeline_benchmarking/01_sc_imputation_selection` - scRNA-seq (K1) data imputation strategy selection
3. `pipeline_benchmarking/02_model_selection` - compition of 12 models and selection best 4 of them
4. `pipeline_benchmarking/03_optimizer_selection` - comparing adamW and Muon for best DL models
5. `pipeline_benchmarking/04_ensembles_selection` - ensemble approaches benchmarking  
6. `final_train_test_inference/train` - final train of selected architecture 
7. `final_train_test_inference/Optimal_K` - target filtering and pseudobulk size (K) optimization
8. `final_train_test_inference/test_metrics` - test metrics for selected models
9. `final_train_test_inference/robustness` - models robustness for target dropout
10. `final_train_test_inference/inference` → inference on scRNA-seq datasets  
