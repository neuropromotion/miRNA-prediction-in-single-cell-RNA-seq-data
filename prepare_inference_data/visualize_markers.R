#!/usr/bin/env Rscript
# Visualizations for pan-cancer miRNA marker tables
# Inputs:
#   pan_cancer_miRNA_markers.xlsx
#   miRNA_cancer_cluster_matrix.xlsx
# Outputs → figures/*.pdf + figures/*.png

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readxl)
  library(ggplot2)
  library(stringr)
  library(forcats)
  library(ggalluvial)
  library(circlize)
  library(scales)
  library(viridis)
})

# ── paths ────────────────────────────────────────────────────────────────────
root <- "prepare_inference_data"
fig_dir <- file.path(root, "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

pan_xlsx <- file.path(root, "tables/pan_cancer_miRNA_markers.xlsx")
mat_xlsx <- file.path(root, "tables/miRNA_cancer_cluster_matrix.xlsx")

# ── helpers ──────────────────────────────────────────────────────────────────
save_both <- function(plot, name, width = 10, height = 7) {
  pdf_path <- file.path(fig_dir, paste0(name, ".pdf"))
  png_path <- file.path(fig_dir, paste0(name, ".png"))
  pdf(pdf_path, width = width, height = height)
  print(plot)
  dev.off()
  # cairo_png avoids ragg Graphics API mismatch on some systems
  tryCatch({
    png(png_path, width = width, height = height, units = "in", res = 300, type = "cairo")
    print(plot)
    dev.off()
  }, error = function(e) {
    png(png_path, width = width, height = height, units = "in", res = 300)
    print(plot)
    dev.off()
  })
  cat("  saved:", name, "(.pdf/.png)\n")
}

map_cluster_group <- function(x) {
  dplyr::case_when(
    grepl("Malignant|tumor|Tumour|Cycling tumor", x, ignore.case = TRUE) ~ "Malignant",
    grepl("Fibroblast|Stromal|Pericyte", x, ignore.case = TRUE) ~ "Stromal / Fibroblast",
    grepl("Endothelial", x, ignore.case = TRUE) ~ "Endothelial",
    grepl("TAM|Macrophage", x, ignore.case = TRUE) ~ "Macrophage / TAM",
    grepl("^T cells|CD4|CD8|Treg", x, ignore.case = TRUE) ~ "T cells",
    grepl("B cell", x, ignore.case = TRUE) ~ "B cells",
    grepl("NK", x, ignore.case = TRUE) ~ "NK cells",
    grepl("DC|Dendritic", x, ignore.case = TRUE) ~ "Dendritic",
    grepl("Neutrophil", x, ignore.case = TRUE) ~ "Neutrophils",
    grepl("Epithelial|Keratinocyte|Acinar|Ciliated|Basal|Follicular|Gastric|Bronchial|Luminal|Myoepithelial",
          x, ignore.case = TRUE) ~ "Epithelial",
    TRUE ~ "Other"
  )
}

short_mir <- function(x) sub("^hsa-", "", x)

theme_clean <- theme_bw(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey92", colour = NA),
    strip.text = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    plot.title = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(colour = "grey30", size = 10)
  )

# ── load data ────────────────────────────────────────────────────────────────
cat("Loading tables...\n")
summary_df <- read_excel(pan_xlsx, sheet = "Summary by datasets")
long_df    <- read_excel(pan_xlsx, sheet = "Long Format Summary")
specific_df <- read_excel(pan_xlsx, sheet = "Only specific miRNAs")
matrix_df  <- read_excel(mat_xlsx)

long_df <- long_df %>%
  mutate(
    cluster_group = map_cluster_group(cluster),
    mir_short = short_mir(miRNA),
    Score = as.numeric(Score),
    n_datasets = as.numeric(n_datasets),
    Cancer_Count_lookup = NA_real_
  )

summary_df <- summary_df %>%
  mutate(
    cluster_group = map_cluster_group(cluster),
    mir_short = short_mir(miRNA)
  )

# attach Cancer_Count to long format
long_df <- long_df %>%
  left_join(
    summary_df %>% dplyr::select(miRNA, cluster, Cancer_Count),
    by = c("miRNA", "cluster")
  )

# matrix → long (DE + cluster presence)
de_cols <- grep("\\| DE$", names(matrix_df), value = TRUE)
cl_cols <- grep("\\| cluster$", names(matrix_df), value = TRUE)

matrix_long <- matrix_df %>%
  pivot_longer(
    cols = all_of(c(de_cols, cl_cols)),
    names_to = "col",
    values_to = "value"
  ) %>%
  mutate(
    metric = if_else(str_detect(col, "\\| DE$"), "DE", "cluster_n"),
    tissue = str_replace(col, " \\| (DE|cluster)$", ""),
    tissue_short = str_replace(tissue, " \\(n=\\d+\\)$", "")
  ) %>%
  dplyr::select(miRNA, cluster, tissue, tissue_short, metric, value) %>%
  pivot_wider(names_from = metric, values_from = value) %>%
  mutate(
    cluster_group = map_cluster_group(cluster),
    mir_short = short_mir(miRNA),
    score = if_else(cluster_n > 0, DE / cluster_n, 0)
  )

cat("Data ready. Writing figures to:", fig_dir, "\n\n")

# =============================================================================
# 1. Dot / bubble matrix — faceted by major cell-type groups
#    Top miRNAs by Cancer_Count within each group; size = Score; colour = n_datasets
# =============================================================================
cat("[1/7] Dot plot by cell-type group...\n")

focus_groups <- c("Malignant", "Stromal / Fibroblast", "Endothelial",
                  "Macrophage / TAM", "T cells")

# keep miRNAs that are top by Cancer_Count in at least one focus group
top_per_group <- summary_df %>%
  filter(cluster_group %in% focus_groups) %>%
  group_by(cluster_group) %>%
  slice_max(order_by = Cancer_Count, n = 12, with_ties = FALSE) %>%
  ungroup()

dot_df <- long_df %>%
  semi_join(top_per_group %>% dplyr::select(miRNA, cluster_group),
            by = c("miRNA", "cluster_group")) %>%
  filter(cluster_group %in% focus_groups) %>%
  # collapse multiple raw cluster names inside a group: keep max score per mir×cancer×group
  group_by(mir_short, Cancer_Type, cluster_group) %>%
  summarise(
    Score = max(Score, na.rm = TRUE),
    n_datasets = max(n_datasets, na.rm = TRUE),
    Cancer_Count = max(Cancer_Count, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    mir_short = fct_reorder(mir_short, Cancer_Count, .fun = max, .desc = FALSE),
    Cancer_Type = fct_relevel(Cancer_Type, sort(unique(as.character(Cancer_Type))))
  )

p1 <- ggplot(dot_df, aes(x = Cancer_Type, y = mir_short)) +
  geom_point(aes(size = Score, colour = n_datasets), alpha = 0.9) +
  facet_wrap(~ cluster_group, scales = "free_y", ncol = 1) +
  scale_size_continuous(range = c(1.5, 7), name = "Score\n(DE / cluster present)") +
  scale_colour_viridis_c(option = "C", name = "# datasets\nwith DE") +
  labs(
    title = "Consensus miRNA markers across cancer types",
    subtitle = "Bubble size = Score; colour = n_datasets with DE. Facets = cell-type groups.",
    x = NULL, y = NULL
  ) +
  theme_clean +
  theme(legend.position = "right")

save_both(p1, "01_dotplot_by_celltype", width = 12, height = 16)

# =============================================================================
# 2. Alluvial / Sankey — top pan-cancer miRNAs → cluster → cancer
# =============================================================================
cat("[2/7] Alluvial diagram...\n")

# Top miRNAs by max Cancer_Count (broad pan-cancer markers)
top_mirs <- summary_df %>%
  group_by(miRNA) %>%
  summarise(max_cc = max(Cancer_Count), .groups = "drop") %>%
  slice_max(order_by = max_cc, n = 10, with_ties = FALSE) %>%
  pull(miRNA)

alluvial_df <- long_df %>%
  filter(miRNA %in% top_mirs, cluster_group %in% focus_groups) %>%
  group_by(mir_short, cluster_group, Cancer_Type) %>%
  summarise(freq = sum(n_datasets), .groups = "drop") %>%
  mutate(
    mir_short = factor(mir_short, levels = short_mir(top_mirs)),
    cluster_group = factor(cluster_group, levels = focus_groups)
  )

p2 <- ggplot(alluvial_df,
             aes(axis1 = mir_short, axis2 = cluster_group, axis3 = Cancer_Type, y = freq)) +
  geom_alluvium(aes(fill = cluster_group), width = 1/6, alpha = 0.75, knot.pos = 0.3) +
  geom_stratum(width = 1/6, fill = "grey95", colour = "grey40", linewidth = 0.3) +
  geom_text(stat = "stratum", aes(label = after_stat(stratum)), size = 2.4, min.y = 8) +
  scale_x_discrete(limits = c("miRNA", "Cell type", "Cancer type"), expand = c(0.08, 0.08)) +
  scale_fill_brewer(palette = "Set2", name = "Cell type") +
  labs(
    title = "Alluvial map of top pan-cancer miRNA markers",
    subtitle = "Top 10 miRNAs by Cancer_Count. Flow thickness = sum of n_datasets (consensus).",
    y = "Total dataset detections"
  ) +
  theme_bw(base_size = 11) +
  theme(
    panel.grid = element_blank(),
    axis.title.x = element_blank(),
    plot.title = element_text(face = "bold", size = 13),
    legend.position = "bottom"
  )

save_both(p2, "02_alluvial_top_mirnas", width = 14, height = 9)

# =============================================================================
# 3. Chord diagram — Malignant-cell markers linking miRNA ↔ cancer type
# =============================================================================
cat("[3/7] Chord diagram (Malignant)...\n")

chord_src <- long_df %>%
  filter(cluster_group == "Malignant") %>%
  group_by(miRNA) %>%
  mutate(cc = n_distinct(Cancer_Type)) %>%
  ungroup() %>%
  filter(cc >= 8) %>%   # only broadly shared malignant markers
  group_by(mir_short, Cancer_Type) %>%
  summarise(weight = max(n_datasets), .groups = "drop")

if (nrow(chord_src) > 0) {
  mat <- chord_src %>%
    pivot_wider(names_from = Cancer_Type, values_from = weight, values_fill = 0) %>%
    tibble::column_to_rownames("mir_short") %>%
    as.matrix()

  # limit sectors for readability
  if (nrow(mat) > 15) {
    keep <- names(sort(rowSums(mat), decreasing = TRUE))[1:15]
    mat <- mat[keep, , drop = FALSE]
  }

  pdf(file.path(fig_dir, "03_chord_malignant.pdf"), width = 11, height = 11)
  circos.clear()
  circos.par(start.degree = 90, gap.degree = 2, track.margin = c(0.01, 0.01))
  set.seed(42)
  chordDiagram(
    mat,
    annotationTrack = "grid",
    preAllocateTracks = list(track.height = 0.12),
    transparency = 0.35
  )
  circos.trackPlotRegion(
    track.index = 1, panel.fun = function(x, y) {
      xlim <- get.cell.meta.data("xlim")
      ylim <- get.cell.meta.data("ylim")
      sector <- get.cell.meta.data("sector.index")
      circos.text(mean(xlim), ylim[1] + 0.2, sector,
                  facing = "clockwise", niceFacing = TRUE,
                  adj = c(0, 0.5), cex = 0.55)
    },
    bg.border = NA
  )
  title("Malignant-cell miRNA markers (Cancer_Count ≥ 8)\nLinks = n_datasets with DE", cex.main = 1)
  circos.clear()
  dev.off()

  png(file.path(fig_dir, "03_chord_malignant.png"), width = 11, height = 11, units = "in", res = 300)
  circos.clear()
  circos.par(start.degree = 90, gap.degree = 2, track.margin = c(0.01, 0.01))
  set.seed(42)
  chordDiagram(
    mat,
    annotationTrack = "grid",
    preAllocateTracks = list(track.height = 0.12),
    transparency = 0.35
  )
  circos.trackPlotRegion(
    track.index = 1, panel.fun = function(x, y) {
      xlim <- get.cell.meta.data("xlim")
      ylim <- get.cell.meta.data("ylim")
      sector <- get.cell.meta.data("sector.index")
      circos.text(mean(xlim), ylim[1] + 0.2, sector,
                  facing = "clockwise", niceFacing = TRUE,
                  adj = c(0, 0.5), cex = 0.55)
    },
    bg.border = NA
  )
  title("Malignant-cell miRNA markers (Cancer_Count ≥ 8)\nLinks = n_datasets with DE", cex.main = 1)
  circos.clear()
  dev.off()
  cat("  saved: 03_chord_malignant (.pdf/.png)\n")
} else {
  cat("  skipped chord (insufficient Malignant data)\n")
}

# =============================================================================
# 4. Heatmap — Score for top miRNAs × cancer (Malignant cells)
# =============================================================================
cat("[4/7] Score heatmap (Malignant)...\n")

heat_src <- long_df %>%
  filter(grepl("Malignant", cluster, ignore.case = TRUE)) %>%
  group_by(miRNA) %>%
  mutate(cc = n_distinct(Cancer_Type)) %>%
  ungroup() %>%
  filter(cc >= 5) %>%
  group_by(mir_short, Cancer_Type) %>%
  summarise(Score = max(Score), .groups = "drop")

top_heat_mirs <- heat_src %>%
  group_by(mir_short) %>%
  summarise(mean_score = mean(Score), n_c = n(), .groups = "drop") %>%
  arrange(desc(n_c), desc(mean_score)) %>%
  slice_head(n = 30) %>%
  pull(mir_short)

heat_df <- heat_src %>%
  filter(mir_short %in% top_heat_mirs) %>%
  mutate(mir_short = factor(mir_short, levels = rev(top_heat_mirs)))

p4 <- ggplot(heat_df, aes(x = Cancer_Type, y = mir_short, fill = Score)) +
  geom_tile(colour = "white", linewidth = 0.2) +
  scale_fill_viridis_c(option = "B", limits = c(0, 1), name = "Score") +
  labs(
    title = "Consistency Score — Malignant cell markers",
    subtitle = "Score = n_datasets with DE / n_datasets where cluster is present. Top 30 miRNAs by breadth.",
    x = NULL, y = NULL
  ) +
  theme_clean +
  theme(legend.position = "right", panel.grid = element_blank())

save_both(p4, "04_heatmap_score_malignant", width = 12, height = 9)

# =============================================================================
# 5. Pan-cancer breadth — lollipop of Cancer_Count
# =============================================================================
cat("[5/7] Pan-cancer breadth lollipop...\n")

breadth_df <- summary_df %>%
  filter(Cancer_Count >= 5) %>%
  mutate(
    label = paste0(mir_short, " · ", cluster),
    label = fct_reorder(label, Cancer_Count)
  )

p5 <- ggplot(breadth_df, aes(x = Cancer_Count, y = label, colour = cluster_group)) +
  geom_segment(aes(x = 0, xend = Cancer_Count, y = label, yend = label),
               colour = "grey80", linewidth = 0.6) +
  geom_point(size = 2.5) +
  scale_colour_brewer(palette = "Dark2", name = "Cell-type group") +
  scale_x_continuous(breaks = pretty_breaks(8), expand = expansion(mult = c(0, 0.05))) +
  labs(
    title = "Pan-cancer breadth of consensus markers",
    subtitle = "miRNA–cluster pairs present in ≥ 5 cancer types (Cancer_Count).",
    x = "Cancer_Count", y = NULL
  ) +
  theme_bw(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    plot.title = element_text(face = "bold", size = 13),
    legend.position = "bottom"
  )

save_both(p5, "05_pan_cancer_breadth", width = 11, height = max(6, nrow(breadth_df) * 0.12 + 2))

# =============================================================================
# 6. Cluster-specific miRNAs (Only specific miRNAs sheet)
# =============================================================================
cat("[6/7] Cluster-specific miRNAs...\n")

spec_plot <- specific_df %>%
  mutate(
    mir_short = short_mir(miRNA),
    cluster_group = map_cluster_group(cluster),
    mir_short = fct_reorder(mir_short, Cancer_Count)
  )

p6 <- ggplot(spec_plot, aes(x = Cancer_Count, y = mir_short, fill = cluster)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = cluster), hjust = -0.05, size = 2.6, colour = "grey20") +
  scale_fill_brewer(palette = "Paired", name = "Cluster") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.25))) +
  labs(
    title = "Cluster-specific consensus miRNAs",
    subtitle = "miRNAs linked to exactly one cluster across the pan-cancer summary.",
    x = "Cancer_Count", y = NULL
  ) +
  theme_bw(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    plot.title = element_text(face = "bold", size = 13),
    legend.position = "none"
  )

save_both(p6, "06_cluster_specific_mirnas", width = 10, height = 7)

# =============================================================================
# 7. Consistency vs breadth scatter (+ full-matrix DE overview for Malignant)
# =============================================================================
cat("[7/7] Consistency vs breadth + raw DE overview...\n")

scatter_df <- long_df %>%
  group_by(miRNA, cluster, cluster_group, mir_short) %>%
  summarise(
    Cancer_Count = n_distinct(Cancer_Type),
    mean_Score = mean(Score, na.rm = TRUE),
    median_n = median(n_datasets, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(Cancer_Count >= 2, cluster_group %in% focus_groups)

p7a <- ggplot(scatter_df, aes(x = Cancer_Count, y = mean_Score, colour = cluster_group)) +
  geom_point(aes(size = median_n), alpha = 0.7) +
  geom_text(
    data = scatter_df %>% filter(Cancer_Count >= 12 | mean_Score >= 0.95),
    aes(label = mir_short), size = 2.2, vjust = -0.8, show.legend = FALSE
  ) +
  scale_colour_brewer(palette = "Set1", name = "Cell-type group") +
  scale_size_continuous(range = c(1.5, 6), name = "Median\nn_datasets") +
  labs(
    title = "Consistency vs pan-cancer breadth",
    subtitle = "Each point = miRNA–cluster pair. X = # cancer types; Y = mean Score.",
    x = "Cancer_Count", y = "Mean Score"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    legend.position = "right"
  )

save_both(p7a, "07_consistency_vs_breadth", width = 10, height = 7)

# Raw DE (including 1–2 datasets) for Malignant — from full matrix
raw_mal <- matrix_long %>%
  filter(
    grepl("Malignant", cluster, ignore.case = TRUE),
    DE > 0
  ) %>%
  group_by(mir_short) %>%
  mutate(n_tissues = n_distinct(tissue_short), mean_de = mean(DE)) %>%
  ungroup() %>%
  filter(n_tissues >= 8) %>%
  mutate(mir_short = fct_reorder(mir_short, mean_de))

p7b <- ggplot(raw_mal, aes(x = tissue_short, y = mir_short, fill = DE)) +
  geom_tile(colour = "white", linewidth = 0.15) +
  scale_fill_viridis_c(option = "D", name = "# datasets\nwith DE",
                       breaks = 0:7) +
  labs(
    title = "Full DE counts — Malignant cells (no ≥3 filter)",
    subtitle = "Includes weak signals (DE = 1 or 2). miRNAs present in ≥ 8 tissue types.",
    x = NULL, y = NULL
  ) +
  theme_clean +
  theme(panel.grid = element_blank(), legend.position = "right")

save_both(p7b, "08_full_DE_malignant_matrix", width = 13, height = 9)

# =============================================================================
cat("\nDone. Figures written to:\n  ", fig_dir, "\n")
print(list.files(fig_dir, pattern = "\\.(pdf|png)$"))
