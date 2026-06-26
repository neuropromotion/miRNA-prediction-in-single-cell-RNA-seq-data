## Single cell RNA seq data preparation to prediction miRNA. Pipeline includes standard processign of scRNA-seq: QC, dimentional reduction, clustering and annotation. In order to confirm malignant cluster annotation CopyKAT was used.
### Finally 121 single cell dataset was processed in order to predict miRNA expression, including:

- Renal cell carcinoma: 5 samples
- Breast cancer: 5 samples
- Colorectal cancer: 5 samples
- Ovarian cancer (metastatic): 5 samples
- Cervical cancer: 5 samples
- B cell lymphoma (DLBCL): 6 samples
- Intrahepatic Cholangiocarcinoma (ICC): 5 samples
- Pancreas ductal carcinoma: 5 samples
- Lung adenocarcinoma (metastatic): 7 samples
- Colorectal cancer (metastatic): 5 samples
- Lung adenocarcinoma: 5 samples
- Breast cancer (metastatic): 5 samples
- Hepatocellular carcinoma: 5 samples
- Thyroid cancer: 6 samples
- Cholangiocarcinoma (metastatic): 5 samples
- Gastric cancer: 5 samples
- Thyroid cancer (metastatic): 5 samples
- PBMC: 5 samples # single non malignant dataset 
- Ovarian cancer: 6 samples
- Melanoma: 5 samples
- clear cell renal carcinoma (bone marrow metastasis): 6 samples
- Gastric cancer (metastatic): 5 samples
- cunateus squamous cell carcinoma: 5 samples

N=121

## Repository Structure

| File | Description |
|------|-------------|
| `standard_workflow.Rmd` | Standard preproceccing and preparation data for ML prediction for all 121 datasets |
| `ADD_ALL_PREDS.R` | Script for adding all predicted miRNA values in RDS files |
| `PROCESS_DEG.R` | Evaluates differentially expressed miRNAs, aggregates the results within individual cancer types, and integrates them across all cancers to produce the final comprehensive table|
| `LOAD_AND_PLOT.R` | Scripts for load datasets, plot figures |
| `save_figures.R` | Script for saving all UMAPs and mRNA heatmaps for supplementary matherials |
| `Single_cell_datasets.xlsx` | Table with accession numbers for all 121 scRNA-seq datasets | 
