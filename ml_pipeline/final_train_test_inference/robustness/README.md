# Robustness

On the same data as test_metrics (**eval** half of scRNA TEST), each eligible miRNA gets a rule for how much **missing sc signal** (absent or all-zero genes in the panel) is acceptable at inference. In uses **gene importance weights** (leave-one-out [LOO] on eval) and an **importance budget**: a cap on the summed weight of absent genes, calibrated with the same R² retention idea as earlier versions (aligned with `sc_r2_median` from test_metrics). Evaluating carried out using single-cell (sc) derived features only (without bulk features due to their irrelevance to the scRNA-seq data). 

Roughly: **how important is each sc feature**, and **how much missing importance** are we willing to tolerate before we expect performance to collapse?

## Algorithm 

**Step 1 — feature importance (per miRNA)**  
For each eligible miRNA we run **leave-one-out (LOO)** on the Optimal_K **eval** half of `data/sc_TEST`: zero out one sc gene at a time, predict, and measure how much R² drops. Those drops are turned into **normalized weights** (one weight per sc gene). So for each miRNA we get a table: **gene → importance**.

**Step 2 — calibrate an “importance budget”**  
We need a single number per miRNA: the maximum allowed **sum of weights over genes that are absent** in a new sample.  
To pick that number we simulate **random masks** (turn off several sc genes at once according to the weights), re-predict, and require that **median R² under masking** does not fall below a fixed fraction of the **reference** sc performance from test_metrics.

**The reference and the threshold**  
The reference is **`sc_r2_median`** for that miRNA from `test_metrics` (bootstrap summary on the eval half).  
The **cutoff** we enforce during calibration is:

**median R² (masked) ≥ `retention_ratio` × `sc_r2_median`**

By default **`retention_ratio` = 0.5**, i.e. **half** of that test median R². (This is the same “retention” idea as in older robustness versions.)

We search for the **largest budget** (sum of weights we allow to be “spent” on absent genes) that still satisfies that inequality on the eval split; that becomes **`importance_budget`** in `robustness_config.json`.

**Outputs**

- **`results/robustness_config.json`** — per miRNA: `importance_budget`, pointer to the weight CSV, calibration metadata.
- **`results/feature_weights/<mirna>.csv`** — gene-level weights used both for calibration and at inference.