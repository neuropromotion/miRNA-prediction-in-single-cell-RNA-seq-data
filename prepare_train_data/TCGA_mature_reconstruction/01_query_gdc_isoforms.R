#!/usr/bin/env Rscript
# =============================================================================
# 01_query_gdc_isoforms.R
#
# Запрос GDC API: для каждого sample barcode из annotation_TCGA.csv найти файл
# «Isoform Expression Quantification» (BCGSC miRNA-Seq) и собрать manifest.
#
# Выход (data/processed/):
#   gdc_isoform_manifest.tsv      — file_id, filename, md5, size
#   gdc_sample_to_file.tsv        — sample_barcode ↔ file_id ↔ aliquot
#   gdc_unmatched_samples.tsv     — образцы без isoform-файла на GDC
#
# Запуск:
#   Rscript 01_query_gdc_isoforms.R
# =============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(httr2)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
SCRIPT_DIR <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
} else {
  normalizePath(".", mustWork = TRUE)
}

DATA_DIR  <- file.path(SCRIPT_DIR, "data")
ANN_PATH  <- file.path(DATA_DIR, "raw/annotation_TCGA.csv")
OUT_DIR   <- file.path(DATA_DIR, "processed")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(ANN_PATH)) {
  stop("Не найдена аннотация: ", ANN_PATH,
       "\nПоложите annotation_TCGA.csv в data/raw/", call. = FALSE)
}

GDC_FILES_URL <- "https://api.gdc.cancer.gov/files"
PAGE_SIZE     <- 1000L
BATCH_SAMPLES <- 500L

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

ann <- fread(ANN_PATH)
sample_ids <- unique(ann$`Sample ID`)
cat(sprintf("[01] annotation: %d rows, %d unique samples\n",
            nrow(ann), length(sample_ids)))
cat(sprintf("[01] querying GDC for %d samples ...\n", length(sample_ids)))

fields <- c(
  "file_id", "file_name", "md5sum", "file_size",
  "data_category", "data_type", "experimental_strategy",
  "analysis.workflow_type",
  "cases.project.project_id",
  "cases.samples.submitter_id",
  "cases.samples.sample_type",
  "cases.samples.portions.analytes.aliquots.submitter_id"
)

make_filter <- function(barcodes_chunk) {
  list(
    op = "and",
    content = list(
      list(op = "in", content = list(field = "data_type",
                                     value = list("Isoform Expression Quantification"))),
      list(op = "in", content = list(field = "experimental_strategy",
                                     value = list("miRNA-Seq"))),
      list(op = "in", content = list(field = "cases.samples.submitter_id",
                                     value = as.list(barcodes_chunk)))
    )
  )
}

query_chunk <- function(barcodes_chunk) {
  body <- list(
    filters = make_filter(barcodes_chunk),
    fields  = paste(fields, collapse = ","),
    format  = "JSON",
    size    = PAGE_SIZE,
    from    = 0L
  )
  results <- list()
  from <- 0L
  repeat {
    body$from <- from
    resp <- request(GDC_FILES_URL) |>
      req_method("POST") |>
      req_headers("Content-Type" = "application/json") |>
      req_body_json(body) |>
      req_retry(max_tries = 5, backoff = ~ 2 ^ .x) |>
      req_perform()
    parsed <- resp_body_json(resp, simplifyVector = FALSE)
    hits <- parsed$data$hits
    if (length(hits) == 0L) break
    results <- c(results, hits)
    pagination <- parsed$data$pagination
    cat(sprintf("    page from=%d, got %d / total %d\n",
                from, length(hits), pagination$total))
    if ((from + length(hits)) >= pagination$total) break
    from <- from + length(hits)
  }
  results
}

flatten_hit <- function(h) {
  out <- list()
  for (case in h$cases) {
    project <- case$project$project_id %||% NA_character_
    for (s in case$samples) {
      sample_barcode <- s$submitter_id %||% NA_character_
      sample_type    <- s$sample_type %||% NA_character_
      aliquots <- character()
      for (p in (s$portions %||% list())) {
        for (a in (p$analytes %||% list())) {
          for (al in (a$aliquots %||% list())) {
            aliquots <- c(aliquots, al$submitter_id %||% NA_character_)
          }
        }
      }
      if (length(aliquots) == 0) aliquots <- NA_character_
      for (alq in aliquots) {
        out[[length(out) + 1L]] <- data.table(
          file_id         = h$file_id,
          file_name       = h$file_name,
          md5sum          = h$md5sum,
          file_size       = h$file_size,
          data_type       = h$data_type,
          workflow        = h$analysis$workflow_type %||% NA_character_,
          project_id      = project,
          sample_barcode  = sample_barcode,
          sample_type     = sample_type,
          aliquot_barcode = alq
        )
      }
    }
  }
  rbindlist(out, fill = TRUE)
}

chunks <- split(sample_ids, ceiling(seq_along(sample_ids) / BATCH_SAMPLES))
all_hits <- list()
for (i in seq_along(chunks)) {
  cat(sprintf("[01] chunk %d/%d (n=%d) ...\n", i, length(chunks), length(chunks[[i]])))
  hits <- tryCatch(query_chunk(chunks[[i]]),
                   error = function(e) {
                     warning("Chunk ", i, " failed: ", conditionMessage(e))
                     list()
                   })
  all_hits <- c(all_hits, hits)
}

if (length(all_hits) == 0L) stop("[01] No hits from GDC.", call. = FALSE)

flat <- rbindlist(lapply(all_hits, flatten_hit), fill = TRUE)
flat_match <- flat[sample_barcode %in% sample_ids]

manifest <- unique(flat_match[, .(
  id = file_id, filename = file_name, md5 = md5sum, size = file_size,
  state = "validated"
)])

fwrite(manifest, file.path(OUT_DIR, "gdc_isoform_manifest.tsv"), sep = "\t")
fwrite(flat_match, file.path(OUT_DIR, "gdc_sample_to_file.tsv"), sep = "\t")

unmatched <- ann[!`Sample ID` %in% flat_match$sample_barcode]
fwrite(unmatched, file.path(OUT_DIR, "gdc_unmatched_samples.tsv"), sep = "\t")

cat(sprintf("[01] manifest: %d files | matched samples: %d | unmatched: %d\n",
            nrow(manifest), uniqueN(flat_match$sample_barcode), nrow(unmatched)))
cat("[01] DONE.\n")
