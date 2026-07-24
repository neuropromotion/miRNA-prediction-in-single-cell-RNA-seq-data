library(dplyr)
library(purrr)
library(tidyr)
library(openxlsx)

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
  'breast_met' = 'Breast cancer',
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

# miRNAs with r2 < 0.4 on bulk test metrics
to_remove <- c('hsa-mir-29a-3p', 'hsa-mir-556-3p', 'hsa-mir-585-5p', 'hsa-mir-7-1-3p')

get_cancer_label <- function(type) {
  if (type %in% names(change_names)) unname(change_names[[type]]) else type
}

integrate_samles <- function(root, k, min_datasets, exclude_mirnas = to_remove){
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
      filter(!miRNA %in% exclude_mirnas)
    
    cluster_presence <- all_degs %>%
      group_by(cluster) %>%
      summarise(cluster_n_datasets = n_distinct(sample_id), .groups = "drop")
    
    cat('Total datasets found:', length(current_files), '| Threshold set to >=', min_datasets, '\n')
    
    # count for each miRNA how many times gene was statistically significant upregulated within cluster and chose >= min_datasets
    consensus_df <- all_degs %>%
      group_by(cluster, miRNA) %>%
      summarise(
        n_datasets = n_distinct(sample_id),
        .groups = "drop"
      ) %>%
      filter(n_datasets >= min_datasets)
    
    valid_clusters <- all_degs %>%
      group_by(cluster) %>%
      summarise(n_datasets = n_distinct(sample_id), .groups = "drop") %>%
      filter(n_datasets >= min_datasets)
    
    consensus_df <- consensus_df %>%
      semi_join(valid_clusters, by = "cluster")
    
    final_robust_markers <- consensus_df %>%
      left_join(cluster_presence, by = "cluster") %>%
      arrange(cluster, desc(n_datasets))
    
    output_file <- file.path(summary_path, paste0(type, '_markers.csv'))
    write.csv(final_robust_markers, file = output_file, row.names = FALSE)
    
    cat('Saved consensus markers to:', output_file, '\n')
  }
}

assambly_summary <- function(root, k, exclude_mirnas = to_remove){
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
    filter(!miRNA %in% exclude_mirnas)
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
  cat('=== Исключено miRNA из to_remove:', length(exclude_mirnas), '===\n')
  
  all_cancer_labels <- vapply(all_cancer_types, get_cancer_label, character(1))
  
  # 1. Создаем глобальную метрику: в скольки ВИДАХ РАКА эта miRNA является маркером для данного кластера?
  pan_cancer_summary <- pan_cancer_df %>%
    group_by(miRNA, cluster) %>%
    summarise(
      Cancer_Count = n_distinct(Cancer_Type),
      Cancers_With_Signal = paste(sort(unique(Cancer_Type)), collapse = "; "),
      .groups = "drop"
    ) %>%
    mutate(
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
      miRNA, cluster, Cancer_Count,
      Cancers_With_Signal, Cancers_Without_Signal
    ) %>%
    arrange(desc(Cancer_Count), miRNA, cluster)
  
  # miRNA, встречающиеся только в одном кластере (но возможно во многих видах рака)
  single_cluster_mirnas <- pan_cancer_summary %>%
    group_by(miRNA) %>%
    filter(n_distinct(cluster) == 1) %>%
    ungroup() %>%
    arrange(desc(Cancer_Count), miRNA, cluster)
  
  cat('=== miRNA только в одном кластере:', nrow(single_cluster_mirnas), '===\n')
  
  # 2. Создаем широкую сводную матрицу (Pivot Table)
  pivot_matrix <- pan_cancer_df %>%
    dplyr::select(miRNA, cluster, Cancer_Type, n_datasets) %>%
    mutate(n_datasets = as.numeric(n_datasets)) %>%
    group_by(miRNA, cluster, Cancer_Type) %>%
    summarise(n_datasets = max(n_datasets), .groups = "drop") %>%
    pivot_wider(
      names_from = Cancer_Type,
      values_from = n_datasets,
      values_fill = 0
    )
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


root <- '/tables/'
log_fc <- '1'
min_datasets <- 3 # threshold for miRNA to be considered as a marker

integrate_samles(root, log_fc, min_datasets)
assambly_summary(root, log_fc)
