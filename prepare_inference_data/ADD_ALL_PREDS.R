library(Seurat)
library(dplyr)

path <- 'PATH_TO_RDS_FILES' 
# RDS files are not presented in Git and Kaggle, but can be requested from author through email on main page
types <- list.dirs(path, recursive=F, full.names = F)


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

FAM <- function(obj, logfc=0.5){
  res <- FindAllMarkers(obj,
                          logfc.threshold = logfc, 
                          min.pct = 0.25,
                          only.pos = TRUE,
                          test.use = 'wilcox',
                          layer = 'data')
  
  res <- res[res$p_val_adj<0.05,]
  res
}

log_progress <- function(current_type, current_sample, types_list, n_datasets_vector) {
  script_dir <- "./"
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", cmd_args, value = TRUE)
  
  if (length(file_arg) > 0) {
    script_dir <- dirname(sub("--file=", "", file_arg))
  } else if (rstudioapi::isAvailable()) {
    script_dir <- dirname(rstudioapi::getSourceEditorContext()$path)
  }
  
  log_file <- file.path(script_dir, "script_progress.log")
  
  # 2. Считаем, сколько всего задач и сколько осталось
  # Создаем таблицу всех возможных комбинаций (type + sample)
  all_tasks <- data.frame()
  for (t in types_list) {
    if (t %in% names(n_datasets_vector)) {
      all_tasks <- rbind(all_tasks, data.frame(type = t, sample = 1:n_datasets_vector[[t]]))
    }
  }
  
  total_tasks <- nrow(all_tasks)
  
  # Находим индекс текущей задачи в общем списке
  current_index <- which(all_tasks$type == current_type & all_tasks$sample == current_sample)
  
  if (length(current_index) == 0) {
    remaining_tasks <- "Unknown"
  } else {
    remaining_tasks <- total_tasks - current_index
  }
  
  # 3. Формируем строку лога
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  log_message <- sprintf(
    "[%s] Processing: %s (Sample %d) | Done: %d/%d | Remaining: %s\n",
    timestamp, current_type, current_sample, current_index, total_tasks, remaining_tasks
  )
  
  # 4. Записываем в файл (append = TRUE, чтобы не затирать старое)
  cat(log_message, file = log_file, append = TRUE)
  
  # Дублируем в консоль, чтобы видеть процесс живьем
  cat(log_message)
}

types_to_process <- intersect(names(n_datasets), types)
types_to_process <- 'RCC'

for (type in types_to_process){
  cat('Starting process type: ', type, '\n')
  
  for (sample_id in seq_len(n_datasets[[type]])){
    log_progress(
      current_type = type, 
      current_sample = sample_id, 
      types_list = types_to_process, 
      n_datasets_vector = n_datasets
    )
    
    obj <- readRDS(file = paste0(path, '/', type, '/rds/', sample_id, '.rds'))
    
    DefaultAssay(obj) <- 'RNA'
    obj@assays <- list(RNA = obj@assays$RNA)
    
    path_predicted <- paste0('ml_pipeline/data/inference_outputs', type, '_S', sample_id, '.csv')
    
    preds <- read.csv(path_predicted, row.names = 1, check.names = FALSE)
    barcodes_val <- colnames(obj)
    preds_filtered <- preds[barcodes_val, ,drop=FALSE]
    
    mir_mat <- pmax(as.matrix(preds_filtered), 0)
    
    mir_assay <- CreateAssayObject(data = t(mir_mat))
    obj[["miRNA"]] <- mir_assay
    
    DefaultAssay(obj) <- "miRNA"
    obj <- ScaleData(obj, assay = "miRNA", do.center = TRUE, do.scale = TRUE)
    
    res_1 <- FAM(obj, logfc=1)
    res_0_5 <- FAM(obj, logfc=0.5)
    
    
    write.csv(res_1, 
              file = paste0('/tables/1/', type, '_S', sample_id, '.csv'),
              row.names = TRUE)
    
    write.csv(res_0_5, 
              file = paste0('/tables/0_5/', type, '_S', sample_id, '.csv'), # logfc=0.5 eventually excluded. We used logfc=1
              row.names = TRUE)
    
    saveRDS(obj, file = paste0(path, '/', type, '/rds/', sample_id, '.rds')) # save predictions in RDS
    
    gc()
  }
}
  
  
