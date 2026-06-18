## Overview

Model performance was evaluated on 327 target miRNAs. A total of 168 targets achieved the predefined performance threshold (R² > 0.4) - eligible targets, they were selected for downstream inference.

For each target, the final prediction level was chosen as the lowest pseudobulk aggregation level (including single-cell resolution, K = 1) at which the target achieved R² > 0.4.

![R2 Performance](figures/r2_by_target_all_k.png)
![R2 Performance](figures/eligible_vs_rest_k1.png)
![R2 Performance](figures/r2_k1_vs_n_sc.png)
![R2 Performance](figures/r2_mean_median_by_k.png)
