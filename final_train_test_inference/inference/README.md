## Overview

Model performance was evaluated on 327 target miRNAs. A total of 164 targets achieved the predefined performance threshold (R² > 0.4) and were selected for downstream inference.

For each target, the final prediction level was chosen as the lowest pseudobulk aggregation level (including single-cell resolution, K = 1) at which the target achieved R² > 0.4.

## Repository Structure

| File | Description |
|------|-------------|
| `target_config.json` | Configuration file containing all eligible miRNA targets (R² > 0.4), selected mRNA features, test metrics, and optimal pseudobulk level for inference |
| `stack_predictor.py` | Final prediction model. Implements a stacking ensemble of CatBoost, TabM, and a ResNet-like neural network with Ridge Regression as the meta-model |
| `preprocessor.py` | `SingleCell` preprocessing pipeline including gene alignment, ENSG conversion, pseudobulk generation, TPM normalization, and log-normalization |
| `mRNA_names.json` | Reference list of mRNA features required by the prediction models. Missing genes are filled with zeros |
| `ensembl_gene_mapping.csv` | Mapping between gene symbols and Ensembl Gene IDs (ENSG) |
| `df_gene_mapping.parquet` | Gene length annotations used for TPM normalization |
| `constants.py` | Project-wide constants and configuration parameters |
| `Inference_tutorial.ipynb` | Step-by-step tutorial demonstrating inference on five RCC (renal cell carcinoma) single-cell datasets |
| `total_inference/total_inference.py` | Pipeline for large-scale inference across all 121 single-cell datasets analyzed in this study |
| `X_train.parquet` | Reference scRNA-seq dataset for KNN imputing |
