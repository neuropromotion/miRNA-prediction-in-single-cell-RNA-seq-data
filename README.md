# miRNA-prediction-in-single-cell-RNA-seq-data

End-to-end machine learning pipeline for predicting miRNA expression from single-cell RNA-seq data.

The training dataset combines bulk RNA-seq (TCGA, GTEx) with paired single-cell RNA-seq data. The repository includes the complete workflow from data preparation and benchmark experiments to final model training and large-scale inference. During development, multiple feature selection methods, scRNA-seq imputation strategies, model architectures, and ensemble approaches were evaluated. The final predictor is a stacking ensemble consisting of **CatBoost**, **ResNet-like**, and **TabM** models. 

### Processed datasets and models
- **Training datasets:** *[link](https://www.kaggle.com/datasets/ismailovaly/mirna-prediction-project)*
- **Pretrained models:** *[link](https://www.kaggle.com/models/ismailovaly/mirna-prediction-model)*

## Repository Structure

| Directory | Description |
|----------|-------------|
| `prepare_train_data` | End-to-end preprocessing pipeline for building the final training datasets from bulk and single-cell RNA-seq data. |
| `pipeline_benchmarking` | Benchmarking experiments for feature selection, scRNA-seq imputation, model selection, and ensemble methods. |
| `final_train_test_inference` | Training of the final models, evaluation on the test set, and inference on 121 scRNA-seq datasets using the selected 164 miRNA prediction models. |
| `scRNA_inference_data_processing` | R scripts for preprocessing scRNA-seq datasets, integrating predictions, visualization, and differential expression analysis. |
| `VAE` | Experimental variational autoencoder for feature compression. Retained for completeness; this approach was not included in the final pipeline. |
