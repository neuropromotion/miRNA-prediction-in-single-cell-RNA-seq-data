# miRNA-prediction-in-single-cell-RNA-seq-data

End-to-end machine learning pipeline for predicting miRNA expression from single-cell RNA-seq data.

The training dataset combines bulk RNA-seq (TCGA, GTEx) with paired single-cell RNA-seq data. The repository includes the complete workflow from data preparation and benchmark experiments to final model training and large-scale inference. During development, multiple feature selection methods, scRNA-seq imputation strategies, model architectures, and ensemble approaches were evaluated. The final predictor is a stacking ensemble consisting of **CatBoost**, **ResNet-like**, and **TabM** models.

Out of 327 miRNAs, 164 achieved acceptable predictive performance on the test set (R² > 0.4), with a median R² of 0.82.

### Processed datasets and models
- **Training and inference datasets:** *[link](https://www.kaggle.com/datasets/ismailovaly/mirna-prediction-project)*
- **Pretrained models:** *[link](https://www.kaggle.com/models/ismailovaly/mirna-prediction-model)*

## Repository Structure

| Directory | Description |
|----------|-------------|
| `ml_pipeline` | **Main ML workspace:** feature selection | data imputation | architecture selection | inference | . |
| `prepare_train_data` | End-to-end preprocessing pipeline for building the final training datasets from bulk and single-cell RNA-seq data. |
| `scRNA_inference_data_processing` | R scripts for preprocessing scRNA-seq datasets, integrating predictions, visualization, and differential expression analysis. |

See [`ml_pipeline/README.md`](ml_pipeline/README.md) for environment setup, pipeline order.
