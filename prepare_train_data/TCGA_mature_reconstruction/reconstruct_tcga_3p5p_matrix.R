#!/usr/bin/env Rscript
# =============================================================================
# reconstruct_tcga_3p5p_matrix.R
#
# Восстановление профиля экспрессии зрелых miRNA (-3p / -5p) из сырой
# precursor-матрицы TCGA с использованием isoform-файлов GDC.
#
# ВАЖНО: это НЕ математическое «разрезание» precursor counts на доли.
# Counts для -3p/-5p берутся из официальных файлов GDC типа
# «Isoform Expression Quantification» (BCGSC miRNA-Seq pipeline), где каждый
# read уже отнесён к зрелому isoform (MIMAT ID). Precursor-матрица служит
# только списком образцов / hairpin-строк и источником для miRNA, которые
# нельзя разложить на mature.
# =============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(future)
  library(future.apply)
})

# --- пути относительно каталога скрипта (self-contained) ----------------------
args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg)) {
  SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", file_arg[1])))
} else {
  SCRIPT_DIR <- normalizePath(".", mustWork = TRUE)
}

DATA_DIR      <- file.path(SCRIPT_DIR, "data")
PRECURSOR_CSV <- file.path(DATA_DIR, "raw/TCGA_precursor_counts.csv")
ISO_DIR       <- file.path(DATA_DIR, "isoform")
MAP_FILE      <- file.path(DATA_DIR, "processed/gdc_sample_to_file.tsv")
MIRBASE_TSV   <- file.path(DATA_DIR, "reference/mirbase_v22_mature.tsv")
OUT_DIR       <- file.path(SCRIPT_DIR, "output")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

for (p in c(PRECURSOR_CSV, MAP_FILE, MIRBASE_TSV)) {
  if (!file.exists(p)) stop("Не найден файл: ", p, call. = FALSE)
}
if (!dir.exists(ISO_DIR)) stop("Нет папки isoform: ", ISO_DIR, call. = FALSE)

cat("=== TCGA mature (-3p/-5p) reconstruction ===\n")
cat("Script dir:", SCRIPT_DIR, "\n")

# =============================================================================
# ШАГ 1. Исходная precursor-матрица (miRNA Expression Quantification)
# =============================================================================
# Формат: miRNA_ID × sample_barcode. Строки — hairpin/precursor (hsa-let-7a-1,
# hsa-mir-100), без суффиксов -3p/-5p. Это «сырая» сводная матрица GDC,
# агрегированная по precursor; она НЕ содержит разделения на зрелые isoform.
# =============================================================================
cat("[1/6] Reading precursor matrix ...\n")
prec <- fread(PRECURSOR_CSV, check.names = FALSE)
stopifnot("miRNA_ID" %in% names(prec))
sample_cols <- setdiff(names(prec), "miRNA_ID")
prec[, miRNA_ID := as.character(miRNA_ID)]
precursor_ids <- unique(prec$miRNA_ID)
cat(sprintf("    precursors: %d | samples: %d\n", length(precursor_ids), length(sample_cols)))

# =============================================================================
# ШАГ 2. Справочник miRBase v22.1: MIMAT → mature name (hsa-*-3p / *-5p)
# =============================================================================
mirbase <- fread(MIRBASE_TSV)
mimat_to_name <- setNames(mirbase$mature_name, mirbase$mimat)
name_to_mimat <- setNames(mirbase$mimat, mirbase$mature_name)
cat(sprintf("[2/6] miRBase mature entries: %d\n", nrow(mirbase)))

# =============================================================================
# ШАГ 3. Карта sample ↔ isoform-файл GDC
# =============================================================================
# Для каждого sample_barcode из аннотации найден файл
# *.mirnaseq.isoforms.quantification.txt (data_type = Isoform Expression
# Quantification).
#
# Особенность исходной precursor-матрицы: 250 колонок CPTAC имеют в заголовке
# несколько barcode через запятую (напр. "C3L-00088-01, C3L-00088-02") — одно
# агрегированное значение на группу aliquot. Isoform-файлы на GDC для CPTAC
# отсутствуют; для остальных 15714 колонок barcode = одному isoform-файлу.
# =============================================================================
map <- fread(MAP_FILE)
map <- map[file.exists(file.path(ISO_DIR, file_id, file_name))]
map[, file_path := file.path(ISO_DIR, file_id, file_name)]
map <- unique(map[, .(file_id, file_name, sample_barcode, file_path)])
samples_with_isoform <- unique(map$sample_barcode)

# Карта: имя колонки исходной матрицы → один или несколько isoform barcode
col_to_iso <- setNames(
  lapply(sample_cols, function(sc) {
    if (grepl(",", sc, fixed = TRUE)) trimws(strsplit(sc, ",")[[1]]) else sc
  }),
  sample_cols
)
iso_map_long <- rbindlist(lapply(sample_cols, function(sc) {
  data.table(output_col = sc, iso_sample = col_to_iso[[sc]])
}), use.names = TRUE)
cols_with_iso <- iso_map_long[iso_sample %in% samples_with_isoform, unique(output_col)]
cols_no_iso   <- setdiff(sample_cols, cols_with_iso)

cat(sprintf("[3/6] isoform files: %d | isoform samples: %d\n",
            nrow(map), length(samples_with_isoform)))
cat(sprintf("    output columns with isoform: %d / %d (без isoform: %d CPTAC-колонок)\n",
            length(cols_with_iso), length(sample_cols), length(cols_no_iso)))

# =============================================================================
# ШАГ 4. Парсинг isoform-файлов → counts по MIMAT × sample
# =============================================================================
# В каждом isoform-файле:
#   - miRNA_ID        — precursor (hairpin), напр. hsa-let-7a-1
#   - miRNA_region    — "precursor" или "mature,MIMAT0000062"
#   - read_count      — число reads, отнесённых к этой координате/isoform
#
# Берём только строки miRNA_region, начинающиеся с "mature,".
# Суммируем read_count по MIMAT внутри файла (все isomiR-варианты одного
# mature попадают в один MIMAT). Затем суммируем по sample_barcode, если
# несколько aliquot-файлов на один образец.
# =============================================================================
parse_isoform_file <- function(path) {
  dt <- tryCatch(
    fread(path, sep = "\t", header = TRUE,
          select = c("read_count", "miRNA_region", "miRNA_ID"),
          colClasses = c(read_count = "numeric",
                         miRNA_region = "character",
                         miRNA_ID = "character")),
    error = function(e) NULL
  )
  if (is.null(dt) || nrow(dt) == 0L) return(NULL)
  dt <- dt[startsWith(miRNA_region, "mature,")]
  if (nrow(dt) == 0L) return(NULL)
  dt[, mimat := sub("^mature,", "", miRNA_region)]
  mature <- dt[, .(count = sum(read_count)), by = mimat]
  hairpin <- unique(dt[, .(precursor = miRNA_ID, mimat)])
  list(mature = mature, hairpin_map = hairpin)
}

cat("[4/6] Parsing isoform files (parallel; may take several minutes) ...\n")
t0 <- Sys.time()

det <- tryCatch(parallel::detectCores(logical = TRUE), error = function(e) NA_integer_)
if (is.na(det) || !is.finite(det) || det < 1) det <- 4L
n_workers <- max(1L, det - 1L)
cat(sprintf("    parallel workers: %d\n", n_workers))
plan(multisession, workers = n_workers)

paths   <- map$file_path
samples <- map$sample_barcode

parsed_list <- future_lapply(seq_along(paths), function(i) {
  p <- parse_isoform_file(paths[i])
  if (is.null(p)) return(NULL)
  p$mature[, sample_barcode := samples[i]]
  p$hairpin_map[, sample_barcode := samples[i]]
  p
}, future.seed = NULL)

plan(sequential)

mature_chunks <- lapply(parsed_list, function(p) if (!is.null(p)) p$mature else NULL)
hairpin_maps  <- lapply(parsed_list, function(p) if (!is.null(p)) p$hairpin_map else NULL)
rm(parsed_list)

mature_long <- rbindlist(mature_chunks, fill = TRUE)
hairpin_long <- rbindlist(hairpin_maps, fill = TRUE)

# sample × mimat (сумма aliquots)
mature_agg <- mature_long[, .(count = sum(count)), by = .(mimat, sample_barcode)]
mature_agg[, mature_name := mimat_to_name[mimat]]
mature_agg <- mature_agg[!is.na(mature_name)]

# глобальная карта precursor → mature (по всем isoform-файлам когорты)
prec_to_mature <- unique(hairpin_long[, .(precursor, mimat)])
prec_to_mature[, mature_name := mimat_to_name[mimat]]
prec_to_mature <- prec_to_mature[!is.na(mature_name)]
prec_to_mature <- unique(prec_to_mature[, .(precursor, mature_name, mimat)])

cat(sprintf("    mature long rows: %s | unique mature names: %d\n",
            format(nrow(mature_agg), big.mark = ","),
            uniqueN(mature_agg$mature_name)))
cat(sprintf("    precursor→mature links: %d (unique precursors: %d)\n",
            nrow(prec_to_mature), uniqueN(prec_to_mature$precursor)))

# =============================================================================
# ШАГ 5. Сборка выходной матрицы в ТОЙ ЖЕ структуре колонок, что и precursor
# =============================================================================
# Precursor считается РАЗЛОЖИМЫМ, если в isoform-файлах есть связь
# precursor → ≥1 MIMAT/mature_name.
#
# Mature counts агрегируются в колонки исходной матрицы:
#   - обычная колонка "SAMPLE-01" ← isoform counts для SAMPLE-01;
#   - CPTAC-колонка "A, B" ← sum(isoform A + isoform B), если isoform есть;
#     для CPTAC isoform нет → NA (восстановление невозможно).
#
# НЕразложимые precursor → исходная строка без изменений.
#
# Формат выхода: TSV (не CSV), т.к. 250 заголовков содержат запятую и при
# записи CSV без кавычек ломают разбор (pandas/R «раскалывают» одну колонку
# на несколько → массовые NaN).
# =============================================================================
cat("[5/6] Building output matrix ...\n")

splittable_precursors <- intersect(
  precursor_ids,
  unique(prec_to_mature$precursor)
)
kept_precursors <- setdiff(precursor_ids, splittable_precursors)

recon_log <- rbindlist(list(
  data.table(
    miRNA_ID = splittable_precursors,
    reconstruction = "split_to_mature",
    note = "Counts from GDC isoform files (mature,MIMAT*); precursor row dropped"
  ),
  data.table(
    miRNA_ID = kept_precursors,
    reconstruction = "kept_as_precursor",
    note = "No mature mapping in isoform files; original precursor counts preserved"
  )
), fill = TRUE)

# mature counts → колонки исходной матрицы (sum по группе barcode)
mature_for_out <- merge(
  mature_agg[, .(mature_name, iso_sample = sample_barcode, count)],
  iso_map_long,
  by = "iso_sample"
)
mature_out_agg <- mature_for_out[, .(count = sum(count)), by = .(mature_name, output_col)]
mature_wide <- dcast(mature_out_agg, mature_name ~ output_col,
                     value.var = "count", fill = 0)
setnames(mature_wide, "mature_name", "miRNA_ID")

# колонки без isoform (CPTAC): NA, не 0
for (sc in cols_no_iso) {
  mature_wide[, (sc) := NA_real_]
}
# гарантировать все колонки и порядок
for (sc in sample_cols) {
  if (!sc %in% names(mature_wide)) mature_wide[, (sc) := NA_real_]
}
mature_wide <- mature_wide[, c("miRNA_ID", sample_cols), with = FALSE]

prec_kept <- prec[miRNA_ID %in% kept_precursors, c("miRNA_ID", sample_cols), with = FALSE]
out <- rbindlist(list(mature_wide, prec_kept), use.names = TRUE)

# сортировка строк: mature (-3p/-5p) сверху, затем kept precursors
out[, row_kind := fifelse(miRNA_ID %in% kept_precursors, "precursor", "mature")]
setorder(out, row_kind, miRNA_ID)
out[, row_kind := NULL]

mature_rows <- out[!miRNA_ID %in% kept_precursors]
na_mature_iso <- mean(is.na(as.matrix(mature_rows[, ..cols_with_iso])))
na_mature_no  <- mean(is.na(as.matrix(mature_rows[, ..cols_no_iso])))
na_kept       <- mean(is.na(as.matrix(prec_kept[, ..sample_cols])))

cat(sprintf("    output rows: %d (mature: %d | kept precursor: %d)\n",
            nrow(out),
            nrow(mature_wide),
            nrow(prec_kept)))
cat(sprintf("    QC NA mature (cols with isoform): %.5f (ожидается ~0)\n", na_mature_iso))
cat(sprintf("    QC NA mature (cols без isoform):  %.5f (ожидается 1.0, CPTAC)\n", na_mature_no))
cat(sprintf("    QC NA kept precursor (all cols):  %.5f (ожидается ~0)\n", na_kept))

# =============================================================================
# ШАГ 6. Запись результатов (TSV — безопасно для запятых в именах колонок)
# =============================================================================
out_matrix <- file.path(OUT_DIR, "TCGA_mature_raw_matrix.tsv")
out_matrix_csv <- file.path(OUT_DIR, "TCGA_mature_raw_matrix.csv")
out_log    <- file.path(OUT_DIR, "reconstruction_log.tsv")
out_map    <- file.path(OUT_DIR, "precursor_to_mature_mapping.tsv")
out_cov    <- file.path(OUT_DIR, "sample_isoform_coverage.tsv")

fwrite(out, out_matrix, sep = "\t", quote = FALSE)
# CSV с кавычками для совместимости (без quote=FALSE — файл нечитаем)
fwrite(out, out_matrix_csv, sep = ",", quote = TRUE)

fwrite(recon_log, out_log, sep = "\t")
fwrite(prec_to_mature, out_map, sep = "\t")
fwrite(data.table(
  output_col = sample_cols,
  has_isoform = sample_cols %in% cols_with_iso,
  iso_samples = vapply(col_to_iso[sample_cols], paste, character(1), collapse = ";")
), out_cov, sep = "\t")

cat("[6/6] Wrote:\n")
cat("  ", out_matrix, "  ← основной файл (рекомендуется)\n")
cat("  ", out_matrix_csv, "  ← CSV с quote=TRUE\n")
cat("  ", out_log, "\n")
cat("  ", out_map, "\n")
cat("  ", out_cov, "\n")
cat(sprintf("\nDone in %.1f min.\n",
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))
