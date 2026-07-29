library(dplyr)
library(purrr)
library(tidyr)
library(openxlsx)
library(jsonlite)

n_datasets <- c(
  'RCC' = 5,
  'breast' = 5,
  'col' = 5,
  'ovarian_met' = 5,
  'cervic' = 5,
  'DLBCL' = 6,
  'ICC' = 5,
  'pancreas' = 5,
  'LUAD_metastasis' = 7,
  'colorectal_met' = 5,
  'LUAD' = 5, 
  'breast_met' = 5,
  'HCC' = 5,
  'thyroid' = 6,
  'met_cholangiocarcinoma' = 5,
  'GC' = 5,
  'thyroid_met' = 5,
  'pbmc' = 5,
  'ovarian' = 6,
  'mel' = 5, 
  'ccRCC-BM' = 6, 
  'GCM' = 5,
  'cSCC' = 5
)

change_names <- c(
  'RCC' = 'Renal cell cancer',
  'breast' = 'Breast cancer',
  'col' = 'Colorectal cancer',
  'ovarian_met' = 'Ovarian cancer [Metastatic]',
  'cervic' = 'Cervical cancer',
  'DLBCL' = 'B cell lymhpoma [DLBCL]',
  'ICC' = 'Intrahepatic cholangiocarcinoma',
  'pancreas' = 'Pancreatic cancer',
  'LUAD_metastasis' = 'LUAD [Metastatic]',
  'colorectal_met' = 'Colorectal cancer [Metastatic]',
  'LUAD' = 'Lung adenocarcinoma', 
  'breast_met' = 'Breast cancer [Metastatic]',
  'HCC' = 'Hepatocellular carcinoma',
  'thyroid' = 'Thyroid cancer',
  'met_cholangiocarcinoma' = 'Cholangiocarcinoma [Metastatic]',
  'GC' = 'Gastric cancer',
  'thyroid_met' = 'Thyroid cancer [Metastatic]',
  'pbmc' = 'PBMC',
  'ovarian' = 'Ovarian cancer',
  'mel' = 'Melanoma', 
  'ccRCC-BM' = 'Renal cell cancer [Metastatic]', 
  'GCM' = 'Gastric cancer [Metastatic]',
  'cSCC' = 'cunateus SCC'
)

# Keep only QC-eligible miRNAs from prediction_config.json
prediction_config_path <- 'ml_pipeline/final_train_test_inference/test_metrics/prediction_config.json'
eligible_mirs <- fromJSON(prediction_config_path)$eligible_mirs
if (is.null(eligible_mirs) || length(eligible_mirs) == 0) {
  stop("eligible_mirs is empty or missing in: ", prediction_config_path)
}
cat('Loaded', length(eligible_mirs), 'eligible miRNAs from prediction_config.json\n')

get_cancer_label <- function(type) {
  if (type %in% names(change_names)) unname(change_names[[type]]) else type
}

integrate_samles <- function(root, k, min_datasets, keep_mirnas = eligible_mirs){
  path <- paste0(root, k)
  summary_path <- paste0(path, '_summary')
  dir.create(summary_path, showWarnings = FALSE, recursive = TRUE)
  
  files <- list.files(path, pattern = "_S[0-9]+\\.csv$", full.names = FALSE)
  if (length(files) == 0) {
    stop("No input files found in: ", path)
  }
  
  types <- sub("_S[0-9]+\\.csv$", "", files)
  groups <- split(files, types)
   
  
  for (type in names(groups)){
    cat('Starting process type: ', type, '\n')
    
    current_files <- file.path(path, groups[[type]])
    
    all_degs <- lapply(seq_along(current_files), function(i) {
      df <- read.csv(current_files[i], stringsAsFactors = FALSE)
      
      df$sample_id <- basename(current_files[i])
      
      if ("gene" %in% colnames(df)) {
        df$miRNA <- df$gene
      } else if ("X" %in% colnames(df)) {
        colnames(df)[colnames(df) == "X"] <- "miRNA"
      }
      if (!"miRNA" %in% colnames(df)) {
        stop("miRNA column not found in: ", current_files[i])
      }
      
      df %>%
        dplyr::select(cluster, miRNA, sample_id) %>%
        distinct()
    }) %>% bind_rows() %>%
      filter(miRNA %in% keep_mirnas)
    
    cluster_presence <- all_degs %>%
      group_by(cluster) %>%
      summarise(cluster_n_datasets = n_distinct(sample_id), .groups = "drop")
    
    cat('Total datasets found:', length(current_files), '| Threshold set to >=', min_datasets, '\n')
    
    # Full counts (no min_datasets filter) — used for miRNA_cancer_cluster_matrix
    full_counts <- all_degs %>%
      group_by(cluster, miRNA) %>%
      summarise(
        n_datasets = n_distinct(sample_id),
        .groups = "drop"
      ) %>%
      left_join(cluster_presence, by = "cluster") %>%
      arrange(cluster, desc(n_datasets))
    
    full_file <- file.path(summary_path, paste0(type, '_full.csv'))
    write.csv(full_counts, file = full_file, row.names = FALSE)
    cat('Saved full counts to:', full_file, '\n')
    
    # Consensus markers (>= min_datasets) — used for pan_cancer_miRNA_markers
    consensus_df <- full_counts %>%
      filter(n_datasets >= min_datasets)
    
    valid_clusters <- cluster_presence %>%
      filter(cluster_n_datasets >= min_datasets)
    
    final_robust_markers <- consensus_df %>%
      semi_join(valid_clusters, by = "cluster") %>%
      arrange(cluster, desc(n_datasets))
    
    output_file <- file.path(summary_path, paste0(type, '_markers.csv'))
    write.csv(final_robust_markers, file = output_file, row.names = FALSE)
    
    cat('Saved consensus markers to:', output_file, '\n')
  }
}



assambly_summary <- function(root, k, min_datasets = 3, keep_mirnas = eligible_mirs){
  cancer_total_lookup <- n_datasets
  
  path <- paste0(root, k, '_summary')
  files <- list.files(path, pattern = "_markers\\.csv$", full.names = TRUE)
  
  if (length(files) == 0) {
    stop("No *_markers.csv files found in: ", path)
  }
  
  pan_cancer_list <- list()
  
  for (f in files) {
    type <- sub("_markers\\.csv$", "", basename(f))
    cat('Reading consensus for:', f, '\n')
    
    df <- read.csv(f, stringsAsFactors = FALSE)
    
    if (nrow(df) > 0) {
      df <- df %>%
        mutate(Cancer_Type = type)
      pan_cancer_list <- append(pan_cancer_list, list(df))
    }
  }
  
  
  pan_cancer_df <- bind_rows(pan_cancer_list) %>%
    filter(miRNA %in% keep_mirnas)
  if (nrow(pan_cancer_df) == 0) {
    stop("All consensus files are empty in: ", path)
  }
  if (!"Cancer_Type" %in% colnames(pan_cancer_df)) {
    stop("Cancer_Type column missing — check input files")
  }
  if (!"cluster_n_datasets" %in% colnames(pan_cancer_df)) {
    stop("cluster_n_datasets column missing — rerun integrate_samles first")
  }
  
  
  all_cancer_types <- sort(sub("_markers\\.csv$", "", basename(files)))
  
  # Full counts (needed for Cluster_Cancer_Count and the wide matrix)
  full_files <- list.files(path, pattern = "_full\\.csv$", full.names = TRUE)
  if (length(full_files) == 0) {
    stop("No *_full.csv files found in: ", path, " — rerun integrate_samles")
  }
  
  full_list <- list()
  for (f in full_files) {
    type <- sub("_full\\.csv$", "", basename(f))
    df <- read.csv(f, stringsAsFactors = FALSE) %>%
      filter(miRNA %in% keep_mirnas) %>%
      mutate(Cancer_Type = type)
    if (nrow(df) > 0) {
      full_list <- append(full_list, list(df))
    }
  }
  full_df <- bind_rows(full_list)
  
  # Cluster present in a cancer type only if seen in >= min_datasets datasets
  cluster_cancer_n <- full_df %>%
    dplyr::distinct(Cancer_Type, cluster, cluster_n_datasets) %>%
    filter(cluster_n_datasets >= min_datasets) %>%
    group_by(cluster) %>%
    summarise(Cluster_Cancer_Count = n_distinct(Cancer_Type), .groups = "drop")
  
  cat('=== Cluster presence threshold: >=', min_datasets, 'datasets per cancer type ===\n') 
  pan_cancer_df <- pan_cancer_df %>% 
    mutate(
      Total_Datasets = unname(cancer_total_lookup[Cancer_Type]),
      Cluster_Datasets = cluster_n_datasets,
      Score = n_datasets / Cluster_Datasets,
      Cancer_Type = vapply(Cancer_Type, get_cancer_label, character(1))
    ) %>%
    dplyr::select(
      miRNA, cluster, Cancer_Type, n_datasets,
      Total_Datasets, Cluster_Datasets, Score
    ) %>%
    arrange(miRNA, cluster, desc(n_datasets))
  
  cat('\n=== Всего найдено уникальных комбинаций:', nrow(pan_cancer_df), '===\n')
  cat('=== Kept eligible miRNAs:', length(keep_mirnas), '===\n')
  
  all_cancer_labels <- vapply(all_cancer_types, get_cancer_label, character(1))
  
  # 1. Pan-cancer summary: signal breadth vs cluster presence across tissue types
  pan_cancer_summary <- pan_cancer_df %>%
    group_by(miRNA, cluster) %>%
    summarise(
      Cancer_Count = n_distinct(Cancer_Type),
      Cancers_With_Signal = paste(sort(unique(Cancer_Type)), collapse = "; "),
      .groups = "drop"
    ) %>%
    left_join(cluster_cancer_n, by = "cluster") %>%
    mutate(
      Score = Cancer_Count / Cluster_Cancer_Count,
      Cancers_Without_Signal = vapply(
        Cancers_With_Signal,
        function(x) {
          present <- strsplit(x, "; ", fixed = TRUE)[[1]]
          paste(setdiff(all_cancer_labels, present), collapse = "; ")
        },
        character(1)
      )
    ) %>%
    dplyr::select(
      miRNA, cluster, Cancer_Count, Cluster_Cancer_Count, Score,
      Cancers_With_Signal, Cancers_Without_Signal
    ) %>%
    arrange(desc(Cancer_Count), miRNA, cluster)
  
  # miRNA linked to exactly one cluster (may still span many cancer types)
  single_cluster_mirnas <- pan_cancer_summary %>%
    group_by(miRNA) %>%
    filter(n_distinct(cluster) == 1) %>%
    ungroup() %>%
    arrange(desc(Cancer_Count), miRNA, cluster)
  
  cat('=== miRNA только в одном кластере:', nrow(single_cluster_mirnas), '===\n')
  
  # 2. Full matrix (no min_datasets cutoff): DE counts + cluster presence per tissue
  # Disambiguate display labels if several types share the same change_names value
  type_keys <- sort(unique(full_df$Cancer_Type))
  raw_labels <- vapply(type_keys, get_cancer_label, character(1))
  label_counts <- table(raw_labels)
  type_display <- setNames(
    vapply(type_keys, function(t) {
      lab <- get_cancer_label(t)
      total_n <- unname(cancer_total_lookup[t])
      if (is.na(total_n)) total_n <- NA_integer_
      if (!is.na(label_counts[lab]) && label_counts[lab] > 1) {
        lab <- paste0(lab, " [", t, "]")
      }
      sprintf("%s (n=%d)", lab, total_n)
    }, character(1)),
    type_keys
  )
  
  cat('=== Building full matrix from', length(full_files), 'tissue types ===\n')
  
  # All miRNA x cluster pairs that appear anywhere
  pairs <- full_df %>% dplyr::distinct(miRNA, cluster)
  
  # Cluster presence per tissue (independent of miRNA)
  cluster_lookup <- full_df %>%
    dplyr::distinct(Cancer_Type, cluster, cluster_n_datasets)
  
  # DE counts per miRNA x cluster x tissue
  de_lookup <- full_df %>%
    dplyr::select(miRNA, cluster, Cancer_Type, n_datasets)
  
  # Build wide matrix: for each tissue type → DE column + cluster column
  pivot_matrix <- pairs
  for (t in type_keys) {
    base <- unname(type_display[t])
    de_col <- paste0(base, " | DE")
    cl_col <- paste0(base, " | cluster")
    
    de_t <- de_lookup %>%
      filter(Cancer_Type == t) %>%
      dplyr::select(miRNA, cluster, n_datasets)
    
    cl_t <- cluster_lookup %>%
      filter(Cancer_Type == t) %>%
      dplyr::select(cluster, cluster_n_datasets)
    
    pivot_matrix <- pivot_matrix %>%
      left_join(de_t, by = c("miRNA", "cluster")) %>%
      left_join(cl_t, by = "cluster") %>%
      mutate(
        !!de_col := tidyr::replace_na(n_datasets, 0L),
        !!cl_col := tidyr::replace_na(cluster_n_datasets, 0L)
      ) %>%
      dplyr::select(-n_datasets, -cluster_n_datasets)
  }
  
  pivot_matrix <- pivot_matrix %>%
    arrange(miRNA, cluster)
  
  wb_base <- createWorkbook()
  addWorksheet(wb_base, "Summary by datasets")
  addWorksheet(wb_base, "Long Format Summary")
  addWorksheet(wb_base, "Only specific miRNAs")
  
  writeData(wb_base, "Summary by datasets", pan_cancer_summary)
  writeData(wb_base, "Long Format Summary", pan_cancer_df)
  writeData(wb_base, "Only specific miRNAs", single_cluster_mirnas)
  
  saveWorkbook(wb_base, file = paste0(root, "pan_cancer_miRNA_markers.xlsx"), overwrite = TRUE)
  write.xlsx(pivot_matrix, file = paste0(root, "miRNA_cancer_cluster_matrix.xlsx"), overwrite = TRUE)
  
  cat('Все результаты успешно сохранены в папку:', root, '\n')
}

root <- 'tables/'
log_fc <- '1'
min_datasets <- 3 # threshold for miRNA to be considered as a marker

integrate_samles(root, log_fc, min_datasets)
assambly_summary(root, log_fc, min_datasets)
