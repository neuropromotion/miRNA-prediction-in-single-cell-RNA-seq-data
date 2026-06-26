library(dplyr)
library(purrr)
library(tidyr)
library(openxlsx)

integrate_samles <- function(root, k, min_datasets){
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
    }) %>% bind_rows()
    
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
      arrange(cluster, desc(n_datasets))
    
    output_file <- file.path(summary_path, paste0(type, '_markers.csv'))
    write.csv(final_robust_markers, file = output_file, row.names = FALSE)
    
    cat('Saved consensus markers to:', output_file, '\n')
  }
}



assambly_summary <- function(root, k){
  
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
  
  
  pan_cancer_df <- bind_rows(pan_cancer_list)
  if (nrow(pan_cancer_df) == 0) {
    stop("All consensus files are empty in: ", path)
  }
  if (!"Cancer_Type" %in% colnames(pan_cancer_df)) {
    stop("Cancer_Type column missing — check input files")
  }
  
  
  all_cancer_types <- sort(sub("_markers\\.csv$", "", basename(files)))
  
  pan_cancer_df <- pan_cancer_df %>% 
    dplyr::select(miRNA, cluster, Cancer_Type, n_datasets) %>%
    arrange(miRNA, cluster, desc(n_datasets))
  
  cat('\n=== Всего найдено уникальных комбинаций:', nrow(pan_cancer_df), '===\n')
  
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
        function(x) paste(setdiff(all_cancer_types, strsplit(x, "; ", fixed = TRUE)[[1]]), collapse = "; "),
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
  addWorksheet(wb_base, "Long_Format_All_Data")
  addWorksheet(wb_base, "Summary_By_Cancer_Count")
  addWorksheet(wb_base, "Single_Cluster_miRNAs")
  
  writeData(wb_base, "Long_Format_All_Data", pan_cancer_df)
  writeData(wb_base, "Summary_By_Cancer_Count", pan_cancer_summary)
  writeData(wb_base, "Single_Cluster_miRNAs", single_cluster_mirnas)
  
  saveWorkbook(wb_base, file = paste0(root, k, '_', "pan_cancer_miRNA_markers.xlsx"), overwrite = TRUE)
  write.xlsx(pivot_matrix, file = paste0(root, k, '_', "miRNA_cancer_cluster_matrix.xlsx"), overwrite = TRUE)
  
  cat('Все результаты успешно сохранены в папку:', root, '\n')
}


root <- '/mnt/jack-5/amismailov/DEGs_MIR/'
log_fc <- '1'
min_datasets <- 3

integrate_samles(root, log_fc, min_datasets)
assambly_summary(root, log_fc)
