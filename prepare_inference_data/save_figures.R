library(Seurat)
library(dplyr)
library(ggplot2)
library(patchwork) # Для удобной склейки графиков в один лейаут

path <- '/mnt/jack-5/amismailov/miRNA_study/cancers' 
types <- list.dirs(path, recursive = FALSE, full.names = FALSE)

n_datasets <- c(
  'RCC' = 5, 'breast' = 5, 'col' = 5, 'ovarian_met' = 5, 'cervic' = 5,
  'DLBCL' = 6, 'ICC' = 5, 'pancreas' = 5, 'LUAD_metastasis' = 7,
  'colorectal_met' = 5, 'LUAD' = 5, 'breast_met' = 5, 'HCC' = 5,
  'thyroid' = 6, 'met_cholangiocarcinoma' = 5, 'GC' = 5,
  'thyroid_met' = 5, 'pbmc' = 5, 'ovarian' = 6, 'mel' = 5, 
  'ccRCC-BM' = 6, 'GCM' = 5, 'cSCC' = 5
)

# Создаем общую папку для финальных отчетов, если её нет
output_report_dir <- file.path("/mnt/jack-5/amismailov/miRNA_study/supplementary_plots")
if(!dir.exists(output_report_dir)) dir.create(output_report_dir)



for (type in names(n_datasets)){
  cat('\nStarting process type: ', type, '\n')
  
  for (sample_id in 1:n_datasets[[type]]){
    cat('Sample ', sample_id, '/', n_datasets[[type]], '\n')
    main_path <- paste0('/mnt/jack-5/amismailov/miRNA_study/cancers/', type, '/')
    
    # 1. Загрузка данных
    seu <- readRDS(file = paste0(main_path, 'rds/', sample_id, '.rds'))
    DefaultAssay(seu) <- "RNA" 
    
    # 2. Поиск маркерных генов
    cat("Finding markers...\n")
    FAM <- FindAllMarkers(seu,
                          logfc.threshold = 0.5, 
                          min.pct = 0.25,
                          only.pos = TRUE,
                          test.use = 'wilcox',
                          layer = 'data',
                          verbose = FALSE)
 
    
    # Выбираем топ-5 маркера для каждого кластера для DotPlot
    top_genes <- FAM %>%
      group_by(cluster) %>%
      slice_max(n = 5, order_by = avg_log2FC) %>%
      pull(gene) %>%
      unique()
  
    # 3. Визуализация
    title_text <- paste0("Cancer: ", type, " | Sample: ", sample_id, 
                         " (Cells: ", ncol(seu), ", Clusters: ", length(unique(Idents(seu))), ")")
    
    # UMAP (важно: raster = TRUE сжимает клетки в png внутри векторного контейнера)
    p1 <- DimPlot(seu, reduction = "umap", label = TRUE, repel = TRUE, raster = TRUE, pt.size=2) + 
      labs(title = "Cell Clusters") +
      theme_minimal() +
      theme(legend.position = "none") # Убираем легенду, если кластеры подписаны на графике
    
    # DotPlot
    p2 <- DotPlot(seu, features = top_genes, assay = "RNA") + 
      RotatedAxis() +
      labs(title = "Top Marker Genes per Cluster") +
      theme(axis.text.x = element_text(size = 8))
    
    # Склеиваем графики с помощью patchwork
    combined_plot <- (p1 + p2) + 
      plot_annotation(
        title = title_text,
        theme = theme(plot_title = element_text(size = 14, face = "bold", hjust = 0.5))
      ) +
      plot_layout(widths = c(1, 1.3)) # Даем чуть больше места DotPlot под имена генов
    
    # 4. Сохранение файла
    # Сохраняем каждый датасет как отдельный PNG (300 DPI для журнального качества)
    filename <- paste0(output_report_dir, "/", type, "_sample_", sample_id, "_report.png")
    
    png(filename, width = 20, height = 10, units = "in", res = 250)
    print(combined_plot)
    dev.off()
    
    cat("Saved report to:", filename, "\n")
    
    # Очистка памяти, чтобы R не упал на 20-м датасете
    rm(seu, FAM, p1, p2, combined_plot)
    gc()
  }
}
