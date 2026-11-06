# Optimal_K

## Purpose

After training the stack, held-out scRNA/bulk **TEST** is used to answer:

- which miRNAs are **good enough** overall (eligibility)?
- for each such miRNA, which **cohort K** (size of the pseudobulk aggregation) to use for later reporting and inference?

TEST is split into **tune** and **eval** halves with **shared cell positions** across cohorts. That split is saved and **reused** in `test_metrics/` and `robustness/` so all numbers line up.

## Algorithm overview

1. **Bootstrap sampling.** To assess the robustness of the performance metric, 1 000 bootstrap samples were generated from the **tune** half of the TEST cohort for each pseudobulk size separately (K1–K5 and K10).

2. **Model evaluation.** Each miRNA prediction model was evaluated across all 1 000 bootstrap samples, resulting in a bootstrap distribution of R² scores for each pseudobulk size.

3. **Model eligibility.** A model was considered eligible if its median bootstrap R² was ≥ 0.4.

4. **Optimal pseudobulk size selection.** For each eligible model:
   - The median bootstrap R² was calculated for each K.
   - The maximum median R² across all K values was identified.
   - A tolerance margin of Δ = 0.05 was subtracted to define the selection threshold.
   - The threshold had to remain ≥ 0.4.
   - The smallest K with a median R² meeting or exceeding this threshold was selected as the optimal pseudobulk size.

5. **Selection idea.** The tolerance margin prevents minor differences in predictive performance from systematically favoring larger pseudobulk sizes. Instead, it prioritizes the smallest aggregation size that achieves performance close to the best observed value, balancing pseudobulk size and prediction quality.

## Outputs

- Decision / bootstrap summary tables (`tables/`).
- **`results/test_split.json`** — tune vs eval indices (required input for `test_metrics` and `robustness`).
- **`results/proto_prediction_config.json`** — draft with K assignments and feature lists; the **production** config with full eval metrics is built in **`test_metrics/`**.
 