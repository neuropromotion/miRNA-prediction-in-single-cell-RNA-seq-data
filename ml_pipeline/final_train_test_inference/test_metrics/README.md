## Overview

Model performance was evaluated across **327 target miRNAs**.  
Only targets achieving a minimum predictive performance threshold (**R² > 0.4**) at any resolution (K = 1, 2–5, 10) or on bulk-level metrics were retained for downstream analysis.

In total, **164 miRNAs** passed this criterion and were defined as *eligible targets*.

---

## Optimal pseudobulk resolution selection

Predictive performance generally improved with increasing pseudobulk aggregation level which was expected gived abundance of bulk data in train set. However, for a subset of miRNAs, performance differences between single-cell resolution (K = 1) and higher pseudobulk levels were minimal.

To fix it we implemented an adaptive selection strategy (see `vizualization_and_config.ipynb`):

1. For each eligible miRNA, the maximum R² across all K values was identified  
2. A tolerance threshold was defined as **−7.5% from the maximum R²**  
3. The **smallest K** satisfying this criterion was selected as the optimal resolution  

This ensures preference for lower K when performance is comparable.

---

## Distribution of selected targets

Final assignment of optimal pseudobulk resolution:

- **K1 (single-cell level):** 11 targets  
- **K2:** 21 targets  
- **K3:** 18 targets  
- **K4:** 20 targets  
- **K5:** 22 targets  
- **K10:** 76 targets  

---

## Model performance summary

### Pseudobulk (optimal K selection)

- Mean R²: **0.7820**  
- Median R²: **0.8212**  
- Max R²: **0.9688**  
- Min R²: **0.4019**

### Bulk-level performance

- Mean R²: **0.7663**  
- Median R²: **0.7813**  
- Max R²: **0.9754**  
- Min R²: **0.4195**

---

## Reproducibility

The selection strategy is implemented in `vizualization_and_config.ipynb`.

![R2 Performance](figures/eligible_vs_rest_k1.png)
![R2 Performance](figures/r2_mean_median_by_k.png)
![R2 Performance](figures/r2_lines_by_target.png)

# Correlation between number of selected features on single-cell and R2 perfomance on K1 (single cell level)
![R2 Performance](figures/n_sc_vs_k1.png)
