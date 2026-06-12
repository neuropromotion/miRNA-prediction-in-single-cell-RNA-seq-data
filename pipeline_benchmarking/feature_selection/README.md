## Feature Selection 

This repository contains a scalable pipeline designed to handle a severe feature-to-sample ratio of out train data: **17k+ features** against **300+ targets** (miRNAs). 

To identify the most robust and computationally efficient feature selection strategy, we implemented a benchmarking framework using a representative subset of **50 randomly selected miRNAs** before scaling the optimal approach to the entire dataset.

### Pipeline Architecture

Every feature selection strategy in this pipeline follows a **two-step architecture** applied strictly to the training data to prevent data leakage:

1. **Step 1: Filtering (Spearman Correlation)** Select top 1500 correlated features for each miRNA
2. **Step 2: Models** The remaining features are processed through one of the following competing approaches:
   * **Baseline:** Spearman correlation selection itself.
   * Linear selection via `Lasso` and `ElasticNET` 
   * **Non-linear Alternative 1:** Feature importance from a **Shallow XGBoost** regressor.
   * **Non-linear Alternative 2:** Information-theoretic ranking via **Mutual Information (MI)** scores.
   * Pair-wise union of Linean and non-linear approaches

For each method we selected top 800 features from bulk train data and top 800 features for train K1+K2 data (pure single cell + pseudobulk size 2 samples). Final set included 800 features (preferentially from K1+K2 part) - in order to test stability of method to select features in resticted conditions
---

### Benchmarking Strategy

To evaluate and compare the performance of each selection method, we train a downstream **XGBoost Regressor** with default hyperparameters on the selected feature subsets.

### Benchmarking Results

The evaluation of the feature selection strategies revealed a tight competition between the linear methods, while non-linear approaches and their pairwise combinations significantly underperformed, particularly in terms of the performance-to-sparsity trade-off.

The final results, compiled in `summary_by_config.csv`, demonstrate that **ElasticNet** and **Lasso** deliver highly comparable predictive accuracy, but with a noticeable difference in the size of the selected feature space.

#### Key Findings:

* **Lasso vs. ElasticNet Matrix:** * **Lasso** aggressively restricted the feature space to an average of **527.84 features** per target. On the pseudo-single-cell level (`K1+K2` subset), its median $R^2$ score was only **0.01 lower** than that of ElasticNet. 
  * While Lasso could arguably be considered a better immediate trade-off due to its higher sparsity, we strategically selected **ElasticNet** as the definitive winner. This decision is forward-looking: the additional feature capacity provided by ElasticNet (averaging **656 features**) serves as a crucial buffer that is hypothesized to enhance model robustness and stability when scaling the pipeline to the full dataset of 300+ targets.
* **Failure of Non-Linear Approaches:** Pure non-linear methods (Shallow XGBoost, Mutual Information) and their pairwise unions with linear models failed to yield competitive results. They suffered from high computational overhead and poor generalization on the restricted single-cell data, proving that non-linear interaction modeling at this stage introduces more noise than signal.
* **Robustness under Restricted Conditions:** Testing on the `K1+K2` (pseudo-single-cell) subset proved that both linear methods are highly resilient to low-sample-size regimes, ensuring that the selected features capture stable biological signals rather than dataset-specific artifacts.

#### Performance Visualization

The cross-target validation performance across the benchmarked configuration options is illustrated below:

##### 1. Bulk Validation Performance (`r2_by_target_bulk.png`)
![Bulk R2 Performance](r2_by_target_bulk.png)
*This plot displays the distribution of $R^2$ scores across the benchmarked miRNAs evaluated on bulk training data.*

##### 2. Pseudo-Single-Cell Validation Performance (`r2_by_target_sc.png`)
![Single-Cell R2 Performance](r2_by_target_sc.png)
*This plot highlights the stability and generalization of the selection methods on the restricted `K1+K2` single-cell level data, justifying the selection of ElasticNet for scaling.*
