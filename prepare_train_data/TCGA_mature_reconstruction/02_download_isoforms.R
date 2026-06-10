#!/usr/bin/env Rscript
# =============================================================================
# 02_download_isoforms.R
#
# Скачивает isoform-файлы с GDC в data/isoform/<file_id>/<file_name>.
#
# Источник: manifest из 01_query_gdc_isoforms.R (gdc_isoform_manifest.tsv).
# Метод: POST https://api.gdc.cancer.gov/data батчами (tar.gz архивы).
# Идемпотентно: уже скачанные file_id пропускаются.
#
# Запуск:
#   Rscript 01_query_gdc_isoforms.R   # один раз, если manifest ещё нет
#   Rscript 02_download_isoforms.R    # ~1.5 ч, ~6–10 GB
#
# При ошибках батча: перезапустите скрипт — докачает только недостающее.
# =============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(httr2)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
SCRIPT_DIR <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
} else {
  normalizePath(".", mustWork = TRUE)
}

DATA_DIR <- file.path(SCRIPT_DIR, "data")
MANIFEST <- file.path(DATA_DIR, "processed/gdc_isoform_manifest.tsv")
OUT_DIR  <- file.path(DATA_DIR, "isoform")
TMP_DIR  <- file.path(DATA_DIR, "isoform/_tmp")
LOG_DIR  <- file.path(SCRIPT_DIR, "logs")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TMP_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(LOG_DIR, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(MANIFEST)) {
  stop("Manifest не найден: ", MANIFEST,
       "\nСначала запустите: Rscript 01_query_gdc_isoforms.R", call. = FALSE)
}

GDC_DATA_URL <- "https://api.gdc.cancer.gov/data"
BATCH_FILES  <- 200L

manifest <- fread(MANIFEST)
cat(sprintf("[02] manifest: %d files\n", nrow(manifest)))

already <- list.dirs(OUT_DIR, recursive = FALSE, full.names = FALSE)
already <- already[!already %in% c("_tmp", "")]
cat(sprintf("[02] already present: %d folders\n", length(already)))

todo <- manifest[!id %in% already]
cat(sprintf("[02] to download: %d files\n", nrow(todo)))

if (nrow(todo) == 0L) {
  cat("[02] nothing to do.\n")
  quit(status = 0)
}

batches <- split(todo$id, ceiling(seq_along(todo$id) / BATCH_FILES))
cat(sprintf("[02] %d batches (up to %d files each)\n", length(batches), BATCH_FILES))

t0 <- Sys.time()
ok <- 0L
fails <- character()

for (i in seq_along(batches)) {
  ids <- batches[[i]]
  archive_path <- file.path(TMP_DIR, sprintf("batch_%05d.tar.gz", i))

  resp <- tryCatch({
    request(GDC_DATA_URL) |>
      req_method("POST") |>
      req_headers("Content-Type" = "application/json") |>
      req_body_json(list(ids = as.list(ids))) |>
      req_timeout(600) |>
      req_retry(max_tries = 4, backoff = ~ 5 * .x) |>
      req_perform(path = archive_path)
  }, error = function(e) {
    warning(sprintf("[02] batch %d failed: %s", i, conditionMessage(e)))
    NULL
  })

  if (is.null(resp)) {
    fails <- c(fails, ids)
    next
  }

  con <- file(archive_path, "rb")
  magic <- readBin(con, "raw", n = 4)
  close(con)
  is_gzip <- length(magic) >= 2 &&
    magic[1] == as.raw(0x1f) && magic[2] == as.raw(0x8b)

  if (is_gzip) {
    untar_dir <- file.path(TMP_DIR, sprintf("batch_%05d_extracted", i))
    dir.create(untar_dir, showWarnings = FALSE)
    untar(archive_path, exdir = untar_dir)
    extracted <- list.dirs(untar_dir, recursive = FALSE, full.names = TRUE)
    for (d in extracted) {
      fid <- basename(d)
      if (!fid %in% ids) next
      target <- file.path(OUT_DIR, fid)
      if (dir.exists(target)) unlink(target, recursive = TRUE)
      file.rename(d, target)
      ok <- ok + 1L
    }
    unlink(untar_dir, recursive = TRUE)
  } else {
    fid <- ids[1]
    fname <- manifest[id == fid, filename][1]
    target_dir <- file.path(OUT_DIR, fid)
    dir.create(target_dir, showWarnings = FALSE)
    file.rename(archive_path, file.path(target_dir, fname))
    ok <- ok + 1L
  }

  if (file.exists(archive_path)) unlink(archive_path)

  if (i %% 5L == 0L || i == length(batches)) {
    elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    rate <- ok / max(elapsed, 1)
    eta <- (nrow(todo) - ok) / max(rate, 0.001)
    cat(sprintf("[02] batch %d/%d | ok=%d/%d | %.2f files/s | ETA %.0f s\n",
                i, length(batches), ok, nrow(todo), rate, eta))
  }
}

unlink(TMP_DIR, recursive = TRUE)

cat(sprintf("\n[02] DONE: %d downloaded, %d failed\n", ok, length(fails)))
if (length(fails) > 0L) {
  fail_path <- file.path(DATA_DIR, "processed/gdc_failed_downloads.tsv")
  fwrite(data.table(file_id = fails), fail_path, sep = "\t")
  cat("[02] failed ids:", fail_path, "\n")
  cat("[02] re-run to retry.\n")
}
