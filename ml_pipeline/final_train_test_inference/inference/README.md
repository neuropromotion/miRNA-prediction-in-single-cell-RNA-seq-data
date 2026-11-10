## Overview

1. **Training splits** → `data/splits/` (includes KNN ref `sc_k1/X_train.parquet`)
2. **Pretrained models** → `final_train_test_inference/models/`


`miRPredictor` — main inference class (`mirpredictor.py`):

```python
from mirpredictor import miRPredictor
import scanpy as sc

mp = miRPredictor(device="cpu")
mp.get_mirs()
mp.get_features()
mp.get_metrics(["hsa-let-7b-5p"])

pred_df = mp.predict_csv("counts.csv", groupby='CellType')   # DataFrame: cells × miRNAs and cluster annotation column
adata = sc.read_h5ad("data.h5ad") 
adata = mp.predict(adata,                          # AnnData with miRNA columns in .obs
                  groupby = 'CellType',            # Cluster annotation
                  embedding_key = 'X_pca_harmony', # Recommended using corrected PCA embeddings for integrated data. UMAP embedings also possible but not recommended
                  embedding_dims=30)         

de = mp.differential_expression(adata, # de analysis
                                 groupby="CellType",       
                                 method="wilcoxon",        
                                 reference=None,            
                                 # reference="Malignant cells",   # possible to use reference cluster 
                                 mirnas=None, # all 
                                 min_cells=3,
                                 adjust="fdr_bh",
                              )

mk = predictor.find_markers(adata,                 # Specific markers 
                           groupby="CellType") 

# also: adata.obsm['X_mirna'], adata.uns['mirna_prediction']
```
 

## Repository Structure

| File | Description |
|------|-------------|
| `prediction_config.json` | Eligible miRNAs, feature panels, bulk/sc R² & MSE, optimal K |
| `robustness_config.json` | Feature dropout robustness config |
| `environment.yml` | Conda env `mir_inference` (scanpy, optuna, torch, xgboost, …) |
| `stack_predictor.py` | TabPack + TabM + XGB ridge stack |
| `mirpredictor.py` | `miRPredictor` (align, ENSG, KNN impute/pseudobulk, TPM, predict) |
| `preprocessor.py` | Back-compat shim → `miRPredictor` |
| `mRNA_names.json` | Reference mRNA feature list |
| `ensembl_gene_mapping.csv` | Gene symbol → ENSG map |
| `df_gene_mapping.parquet` | Gene lengths for TPM |
| `constants.py` | constants |
| `Inference_tutorial.ipynb` | Usage example |


KNN reference (`X_train.parquet` for K1) is also on Kaggle as `X_TRAIN_K1.parquet`:  
https://www.kaggle.com/datasets/ismailovaly/mirna-prediction-project

 