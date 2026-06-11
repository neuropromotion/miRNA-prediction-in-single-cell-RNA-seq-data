## Feature Selection 

This repository contains a scalable pipeline designed to handle a severe feature-to-sample ratio of out train data: **17k+ features** against **300+ targets** (miRNAs). 

To identify the most robust and computationally efficient feature selection strategy, we implemented a benchmarking framework using a representative subset of **50 randomly selected miRNAs** before scaling the optimal approach to the entire dataset.

### Pipeline Architecture

Every feature selection strategy in this pipeline follows a **two-step architecture** applied strictly to the training data to prevent data leakage:

1. **Step 1: Filtering (Adaptive Spearman Correlation)** A fast, rank-based filtering step using an adaptive Spearman correlation threshold to rapidly eliminate orthogonal/noisy features and retain those with the strongest monotonic relationship to the target.
2. **Step 2: Embedded/Filter Selection** The remaining features are processed through one of the following competing approaches:
   * **Baseline:** Linear selection via `LassoCV` (non-zero coefficient extraction).
   * **Non-linear Alternative 1:** Feature importance from a **Shallow XGBoost** regressor.
   * **Non-linear Alternative 2:** Information-theoretic ranking via **Mutual Information (MI)** scores.

---

### Benchmarking Strategy

To evaluate and compare the performance of each selection method, we train a downstream **XGBoost Regressor** with default hyperparameters on the selected feature subsets.
