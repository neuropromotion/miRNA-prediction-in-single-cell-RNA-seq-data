# Robustness

On the same data as test_metrics (**eval** half of scRNA TEST), each eligible miRNA gets a rule for how much **missing sc signal** (absent or all-zero genes in the panel) is acceptable at inference. In uses **gene importance weights** (leave-one-out [LOO] on eval) and an **importance budget**: a cap on the summed weight of absent genes, calibrated with the same R² retention idea as earlier versions (aligned with `sc_r2_median` from test_metrics). Evaluating carried out using single-cell (sc) derived features only (without bulk features due to their irrelevance to the scRNA-seq data). 

Roughly: **how important is each sc feature**, and **how much missing importance** are we willing to tolerate before we expect performance to collapse?

## Algorithm 

**Step 1 — feature importance (per miRNA)**  
for each eligible miRNA i, we run **leave-one-out (LOO)** the importance of each associated gene feature j (j = 1, ..., mᵢ) was estimated by individually setting that feature to zero while keeping all other features unchanged. The resulting change in predictive performance was calculated as the difference between the R² obtained using all genes and the R² obtained after masking gene j (ΔR²ⱼ = R²_all genes − R²_gene j masked). Feature importance weights were then calculated by normalizing the positive contributions across all associated genes: wᵢⱼ = max(0, ΔR²ⱼ) / sumₖ max(0, ΔR²ₖ).


**Step 2 — random joint masking**  
for each miRNA, gene features were randomly shuffled and sequentially masked until the cumulative importance of the removed features reached the predefined importance budget Bᵢ. This procedure was repeated 200 times using independently randomized feature orders to obtain a distribution of masked-model performance.

**Step 3 - budget calibration**  
the maximum allowable importance budget Bᵢ was determined using binary search (bisection), identifying the largest budget for which the median masked-model R² remained at least 50% of the original test R² for that miRNA. The selected budget was then evaluated using 500 additional independently generated masks. If the confirmation median R² fell below the predefined threshold, the budget was recalibrated

**median R² (masked) ≥ `retention_ratio` × `sc_r2_median`**

**`retention_ratio` = 0.5**, i.e. **half** of that test median R². 

We search for the **largest budget** (sum of weights we allow to be “spent” on absent genes) that still satisfies that inequality on the eval split; that becomes **`importance_budget`** in `robustness_config.json`.
 