# TCGA miRNA: reconstructing -3p / -5p expression profiles

Pipeline to build a **mature-level** TCGA count matrix (`hsa-*-3p`, `hsa-*-5p`) from the raw GDC **precursor** expression matrix.

## Overview

The standard GDC product **miRNA Expression Quantification** is aggregated at the **hairpin/precursor** level (e.g. `hsa-let-7a-1`) and does not report -3p/-5p separately.

We do **not** split precursor counts by proportions. Mature isoform counts come from a separate GDC product:

**Isoform Expression Quantification** (BCGSC miRNA-Seq pipeline) — reads are already assigned to `mature,MIMAT*`.

```
precursor matrix  +  isoform files (GDC)  →  mature matrix (hsa-*-3p / *-5p)
                         ↓
              MIMAT → name (miRBase v22.1)
```

## Repository contents

| File | Description |
|---|---|
| `01_query_gdc_isoforms.R` | Query GDC API → isoform file manifest |
| `02_download_isoforms.R` | Download isoform files into `isoform/` |
| `gdc_sample_to_file.tsv` | Sample barcode ↔ file_id mapping |
| `mirbase_v22_mature.tsv` | MIMAT → mature miRNA name |
| `reconstruct_tcga_3p5p_matrix.R` | Build the final matrix |
| `output/` | Results (matrix, logs, mappings) |

## Additional inputs (not included)

- `TCGA_precursor_counts.csv` — source precursor matrix  
- `annotation_TCGA.csv` — sample list for step 01  
- `isoform/` — ~15k GDC files (~6 GB), downloaded with script 02  

## Usage

```bash
Rscript 01_query_gdc_isoforms.R      # build manifest (~30 s)
Rscript 02_download_isoforms.R       # download isoform/ (~1.5 h)
Rscript reconstruct_tcga_3p5p_matrix.R
```

**R packages:** `data.table`, `httr2`, `jsonlite`, `future`, `future.apply`

## Reconstruction logic

1. Parse each isoform file: keep rows with `miRNA_region = mature,MIMAT*`, sum `read_count` per MIMAT per sample.
2. Map MIMAT IDs to miRBase names (`hsa-let-7a-5p`, `hsa-let-7a-3p`, …) using `mirbase_v22_mature.tsv`.
3. If a precursor has a mature mapping in the isoform data → replace the precursor row with mature rows.
4. If no mapping exists → keep the original precursor row unchanged.

## Output (`output/`)

| File | Description |
|---|---|
| `TCGA_mature_raw_matrix.tsv` | Final count matrix (recommended) |
| `TCGA_mature_raw_matrix.csv` | Same data, quoted CSV |
| `reconstruction_log.tsv` | Which miRNAs were split vs kept as precursor |
| `precursor_to_mature_mapping.tsv` | precursor → MIMAT → mature name |
| `sample_isoform_coverage.tsv` | Per-column isoform file coverage |

## Data sources

- [GDC Data Portal](https://portal.gdc.cancer.gov/) — TCGA / CGCI / TARGET miRNA-Seq  
- [miRBase v22.1](https://mirbase.org/)
