library(Seurat)
library(ggplot2)
library(ComplexHeatmap)
library(circlize)
library(viridis)
library(grid)

path <- 'PATH_TO_RDS_FILES' 
# RDS files are not presented in Git and Kaggle, but can be requested from author through email on main page

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

load_by_type <- function(type){
  
  if (!(type %in% names(n_datasets))) {
    stop(paste0("Unknown type: ", type))
  }
  
  n_files <- n_datasets[[type]]
  
  lapply(seq_len(n_files), function(id) {
    load_dataset(type, id)
  })
}

load_dataset <- function(type, id){
  current_path <- file.path(path, type, "rds", paste0(id, ".rds"))
  
  if (!file.exists(current_path)) {
    stop(paste0("File not found: ", current_path))
  }
  
  obj <- readRDS(current_path)
  return(obj)
}



dimplot_v2 <- function(seu, 
                         colors = NULL, 
                         comp_1 = 1,
                         comp_2 = 2,
                         group.by = NULL,
                         shape = 21,
                         size = 2,
                         stroke = 0.2,
                         alpha = 0.8) {
  
  library(ggplot2)
  library(Seurat)
  
  if (is.null(group.by)) {
    cell_groups <- Idents(seu)
  } else {
    cell_groups <- seu[[group.by]][, 1]
  }
  
  df <- data.frame(
    dim1 = Embeddings(seu, "umap")[, comp_1],
    dim2 = Embeddings(seu, "umap")[, comp_2],
    cell_type = cell_groups
  )
  
  p <- ggplot(df, aes(x = dim1, y = dim2, fill = cell_type)) +
    geom_point(shape = shape, size = size, stroke = stroke, color = "black", alpha = alpha) +
    theme_classic() +
    labs(x = paste0("UMAP_", comp_1), y = paste0("UMAP_", comp_2)) +
    theme(
      legend.key.size = unit(1.75, "lines"),
      legend.text = element_text(size = 13),
      legend.title = element_blank()
    ) +
    guides(
      fill = guide_legend(override.aes = list(size = 6))
    )
  
  if (!is.null(colors)) {
    p <- p + scale_fill_manual(values = colors)
  } else {
    p <- p + scale_fill_discrete()
  }
  
  return(p)
}

plot_expression <- function(obj, mir, order=F, option='inferno', slot='scale.data', pt.size=1.5){
  p <- FeaturePlot(obj, features = mir, slot = slot,  pt.size=pt.size, order=order)
  p + scale_color_viridis_c(
    option = option,     
    direction = 1
  )
}





make_heatmap <- function(seu, genes,
                               clusters = NULL,  
                               cluster_rows = FALSE,
                               cluster_cols = FALSE,
                               side = "right",
                               layer = 'scale.data',
                               font_size_row = 10,
                               font_size_col = 12,
                               z_limit = 2,
                               lwd = 1,
                               column_title = '',
                               assay = 'RNA') {
  
  avg <- AverageExpression(seu, features = genes, layer = layer)
  mat_raw <- as.matrix(avg[[assay]])
  
  
  if (is.null(clusters)) {
    clusters <- colnames(mat_raw)
  }
  
  mat_full <- matrix(NA, 
                     nrow = length(genes), 
                     ncol = length(clusters), 
                     dimnames = list(genes, clusters))
  
  # Находим пересечение по генам и по кластерам
  existing_genes <- intersect(genes, rownames(mat_raw))
  existing_clusters <- intersect(clusters, colnames(mat_raw))
  
  
  if(length(existing_genes) > 0 && length(existing_clusters) > 0) {
    mat_full[existing_genes, existing_clusters] <- mat_raw[existing_genes, existing_clusters]
  }
  
  mat_scaled <- t(apply(mat_full, 1, function(x) {
    # Если в строке совсем нет данных или нет вариации — оставляем NA
    if (all(is.na(x)) || sd(x, na.rm = TRUE) == 0) {
      return(rep(NA, length(x)))
    }
    # Масштабируем (автоматически сохраняет NA на пустых местах)
    return(as.vector(scale(x)))
  }))
  
  colnames(mat_scaled) <- clusters
  rownames(mat_scaled) <- genes
  
  # Clipping
  mat_scaled[mat_scaled > z_limit] <- z_limit
  mat_scaled[mat_scaled < -z_limit] <- -z_limit
  
  # 5. Цветовая шкала
  col_fun <- colorRamp2(
    seq(-z_limit, z_limit, length.out = 3),
    viridis::plasma(3)
  )
  
  # 6. Heatmap
  ht <- Heatmap(
    mat_scaled,
    name = "Z-score",
    col = col_fun,
    
    cluster_rows = cluster_rows,
    cluster_columns = cluster_cols,
    
    show_row_names = TRUE,
    show_column_names = TRUE,
    column_title = column_title,
    row_names_side = side,
    rect_gp = gpar(col = "black", lwd = lwd),
    row_names_gp = gpar(fontsize = font_size_row),
    column_names_gp = gpar(fontsize = font_size_col),
    border = TRUE
  )
  
  ht_drawn <- draw(ht)
  
  # 7. Возврат порядка генов
  if (cluster_rows) {
    row_idx <- row_order(ht_drawn)
    if (is.list(row_idx)) row_idx <- unlist(row_idx)
    ordered_genes <- rownames(mat_scaled)[row_idx]
  } else {
    ordered_genes <- genes
  }
  
  
  return(list(
    plot = ht_drawn,
    genes = ordered_genes,
    matrix = mat_scaled  # <-- Добавляем это
  ))
}






