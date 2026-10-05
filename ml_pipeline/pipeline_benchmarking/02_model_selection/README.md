## Model Selection 

### Overview of candidates
An XGBoost model with default parameters was established as the baseline. 

1. **XGBoost** + Optuna
2. **CatBoost** + Optuna
3. **TabNet**
4. **GANDALF**
5. **TabPFN 3.0**
6. **FT-Transformer**
7. **LassoNet**
8. **TabM**
9. **ResNet-like**  
10. **RealMLP**
11. **DCNv2**
12. **TabPack**
13. **TabR**

---

### Phase 1: Efficiency & Speed Benchmarking
In the initial iteration, we benchmarked the training and inference speeds of the models. We specifically focused on **TabPFN** and **FT-Transformer**, as both are notorious for computational overhead. Detailed logs and metrics of this phase are located in the `speed_test/` directory.

*   **TabPFN:** As anticipated, TabPFN exhibited prohibitive inference times that were completely incompatible with the scale of our data. Consequently, it was excluded from subsequent phases.
*   **FT-Transformer:** Although it required substantially more training time compared to the other architectures, we retained it for further evaluation. This decision was based on findings by Gorishniy et al. (arXiv:2106.11959), which demonstrated that FT-Transformer can outperform ResNet-like model on specific tabular tasks.

---

### Phase 2: Evaluation Protocol
All remaining models were systematically evaluated using the following validation pipeline:
*   **Data Composition:** Models were trained and validated using a mixed data combining **bulk RNA-seq**, **single-cell (K1)**, and **pseudobulk** data (K2, K3, K4, K5, K10) as usual.
*   **Validation Split:** Train dataset was split into inner_train and inner_validation to train models and facilitate early stopping / optimal epoch selection, respectively. Model performance was tracked using this validation set (outer_val - used as temporary test cohort).
*   **Scaler fit noise:** For DL / neural candidates that use `StandardScaler` or `QuantileTransformer`, we add N(0, 10e-5) noise (fixed seed) **only when fitting** the scaler, to avoid numerical issues from duplicate / zero-inflated feature values. 

---

### Benchmarking resutls
Based on our evaluation metrics, TabPack and TabM clearly outperformed the rest of the cohort. 
DCNv2, RealMLP, and GANDALF showed comparable mean and median performance. For downstream analysis and ensemble benchmarking, we selected TabPack and TabM as neural network-based models and XGB + Optuna as the best tree-based model. The top models were ranked by their average of means and medians $R^2$ scores among all cohorts (K1-K10):

| Rank | Model | Average of medians $R^2$ | Model | Average of means $R^2$ |
| :---: | :--- | :---: | :--- | :---: |
| **1** | **TabPack** | 0.81 | **TabPack** | 0.79 |
| **2** | **TabM** | 0.8 | **TabM** | 0.78 |
| **3** | **DCNv2** | 0.76 | **TabR** | 0.73 |
| **4** | **RealMLP** | 0.75 | **RealMLP** | 0.73 |

##### FIGUERS
![R2 Performance](figures/mean_median_r2_by_model.png)
![R2 Performance](figures/r2_by_target_k1.png)


> **Conclusion:** **TabPack**, **TabM**, and **XGBoost** architectures have been selected to construct advanced ensemble architectures in the next phase of the project (ensebmle benchmarking)
