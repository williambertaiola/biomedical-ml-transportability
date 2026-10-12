#!/usr/bin/env Rscript

# ============================================================
# Merge robusto dei 4 dataset UCI Heart Disease
# - Carica i file
# - Aggiunge/normalizza la colonna source
# - Rende coerenti colonne e tipi
# - Unisce mantenendo tutti i campioni
# - Salva CSV e RDS finali
# ============================================================

# -----------------------------
# 1) PATH INPUT/OUTPUT (EDITABILI)
# -----------------------------
input_files <- list(
  cleveland  = "data/raw/uci/cleveland.data",
  switzerland = "data/raw/uci/switzerland.data",
  va         = "data/raw/uci/long-beach-va.data",
  hungarian  = "data/raw/uci/hungarian.data"
)

# Se i file hanno intestazione, imposta TRUE; per i file UCI .data di solito FALSE
has_header <- FALSE

output_csv <- "uci_merged.csv"
output_rds <- "uci_merged.rds"


# -----------------------------
# 2) LIBRERIE
# -----------------------------
required_pkgs <- c("dplyr", "readr", "tibble")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(
    sprintf(
      "Pacchetti mancanti: %s\nInstalla prima i pacchetti e rilancia lo script.",
      paste(missing_pkgs, collapse = ", ")
    )
  )
}

library(dplyr)
library(readr)
library(tibble)


# -----------------------------
# 3) FUNZIONI DI SUPPORTO
# -----------------------------

# Parse formato UCI "raw" (record su piu righe, terminato da token "name")
parse_uci_name_terminated <- function(lines, source_name) {
  all_tokens <- unlist(strsplit(paste(lines, collapse = " "), "\\s+"))
  all_tokens <- all_tokens[nzchar(all_tokens)]

  records <- list()
  current <- character()

  for (tok in all_tokens) {
    if (tolower(tok) == "name") {
      if (length(current) > 0) {
        records[[length(records) + 1]] <- current
        current <- character()
      }
    } else {
      current <- c(current, tok)
    }
  }

  # Se il file termina senza "name", conserva comunque l'ultimo record incompleto
  if (length(current) > 0) {
    records[[length(records) + 1]] <- current
  }

  if (length(records) == 0) {
    stop(sprintf("Nessun record valido trovato in formato name-terminated: %s", source_name))
  }

  rec_lengths <- lengths(records)
  target_len <- as.integer(names(which.max(table(rec_lengths))))

  if (any(rec_lengths != target_len)) {
    warning(
      sprintf(
        "[%s] Record con lunghezza non standard rilevati. Lunghezza attesa (modale): %d. Verranno troncati/paddati a questa lunghezza.",
        source_name, target_len
      )
    )
  }

  fixed_records <- lapply(records, function(x) {
    if (length(x) < target_len) {
      c(x, rep(NA_character_, target_len - length(x)))
    } else if (length(x) > target_len) {
      x[seq_len(target_len)]
    } else {
      x
    }
  })

  mat <- do.call(rbind, fixed_records)
  df <- tibble::as_tibble(as.data.frame(mat, stringsAsFactors = FALSE), .name_repair = "minimal")
  names(df) <- paste0("V", seq_len(ncol(df)))
  df
}

# Parse CSV-like o whitespace-like (1 riga = 1 record)
parse_row_delimited <- function(lines, has_header = FALSE) {
  delim_is_comma <- any(grepl(",", lines, fixed = TRUE))
  split_rows <- if (delim_is_comma) {
    strsplit(lines, ",", fixed = TRUE)
  } else {
    strsplit(lines, "\\s+")
  }
  split_rows <- lapply(split_rows, trimws)

  max_cols <- max(lengths(split_rows))
  padded_rows <- lapply(split_rows, function(x) {
    if (length(x) < max_cols) c(x, rep(NA_character_, max_cols - length(x))) else x
  })

  mat <- do.call(rbind, padded_rows)
  df <- tibble::as_tibble(as.data.frame(mat, stringsAsFactors = FALSE), .name_repair = "minimal")

  if (isTRUE(has_header)) {
    header <- make.names(as.character(df[1, ]), unique = TRUE)
    df <- df[-1, , drop = FALSE]
    names(df) <- header
  } else {
    names(df) <- paste0("V", seq_len(ncol(df)))
  }

  df
}

# Carica un file UCI in modo robusto:
# - supporta formato raw "name-terminated" e CSV-like
# - converte "" e "?" in NA
read_uci_like <- function(path, source_name, has_header = FALSE) {
  if (!file.exists(path)) {
    stop(sprintf("File non trovato: %s", path))
  }

  # readLines con latin1/skipNul e pulizia caratteri di controllo:
  # aiuta con file legacy contenenti byte non UTF-8.
  lines <- readLines(path, warn = FALSE, encoding = "latin1", skipNul = TRUE)
  lines <- iconv(lines, from = "latin1", to = "UTF-8", sub = " ")
  lines <- gsub("[[:cntrl:]]", " ", lines)
  lines <- trimws(lines)
  lines <- lines[nzchar(lines)]

  if (length(lines) == 0) {
    stop(sprintf("File vuoto: %s", path))
  }

  is_name_terminated <- any(grepl("\\bname\\b", lines, ignore.case = TRUE))
  df <- if (is_name_terminated) {
    parse_uci_name_terminated(lines, source_name = source_name)
  } else {
    parse_row_delimited(lines, has_header = has_header)
  }

  # Uniforma mancanti tipici UCI
  df <- df %>%
    mutate(across(everything(), ~ na_if(.x, ""))) %>%
    mutate(across(everything(), ~ na_if(.x, "?")))

  # Aggiunge source se assente; se presente riempie solo i valori mancanti/vuoti
  if (!"source" %in% names(df)) {
    df <- df %>% mutate(source = source_name, .before = 1)
  } else {
    df <- df %>%
      mutate(
        source = if_else(
          is.na(source) | trimws(as.character(source)) == "",
          source_name,
          as.character(source)
        )
      )
    # Porta source come prima colonna per chiarezza
    df <- df %>% select(source, everything())
  }

  df
}

is_numeric_like <- function(x, threshold = 0.98) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(TRUE)
  parsed <- suppressWarnings(as.numeric(x))
  mean(!is.na(parsed)) >= threshold
}


# -----------------------------
# 4) CARICAMENTO DEI 4 DATASET
# -----------------------------
datasets <- lapply(names(input_files), function(src) {
  read_uci_like(path = input_files[[src]], source_name = src, has_header = has_header)
})
names(datasets) <- names(input_files)


# -----------------------------
# 5) CHECK COERENZA COLONNE
# -----------------------------
col_sets <- lapply(datasets, names)
col_signatures <- vapply(col_sets, paste, collapse = "|", FUN.VALUE = character(1))
same_structure <- length(unique(col_signatures)) == 1

cat("Struttura colonne identica tra i dataset:", same_structure, "\n")
if (!same_structure) {
  cat("Differenze colonne rilevate: verranno allineate con NA dove necessario.\n")
}


# -----------------------------
# 6) ALLINEAMENTO COLONNE + TIPI
# -----------------------------

# Unione di tutte le colonne presenti; source sempre in testa
all_cols <- unique(unlist(col_sets, use.names = FALSE))
all_cols <- c("source", setdiff(all_cols, "source"))

# Aggiunge colonne mancanti con NA e riallinea ordine
datasets_aligned <- lapply(datasets, function(df) {
  missing_cols <- setdiff(all_cols, names(df))
  if (length(missing_cols) > 0) {
    for (mc in missing_cols) df[[mc]] <- NA_character_
  }
  df <- df[, all_cols, drop = FALSE]
  as_tibble(df)
})

# Rende coerenti i tipi per colonna:
# - numerico se tutti i valori non-NA sono numeric-like
# - altrimenti character
non_source_cols <- setdiff(all_cols, "source")

for (col_name in non_source_cols) {
  pooled_values <- unlist(lapply(datasets_aligned, function(df) df[[col_name]]), use.names = FALSE)
  target_numeric <- is_numeric_like(pooled_values, threshold = 0.98)

  datasets_aligned <- lapply(datasets_aligned, function(df) {
    if (target_numeric) {
      df[[col_name]] <- suppressWarnings(as.numeric(df[[col_name]]))
    } else {
      df[[col_name]] <- as.character(df[[col_name]])
    }
    df
  })
}

# source sempre character
datasets_aligned <- lapply(datasets_aligned, function(df) {
  df$source <- as.character(df$source)
  df
})


# -----------------------------
# 7) MERGE FINALE
# -----------------------------
uci_merged <- dplyr::bind_rows(datasets_aligned)


# -----------------------------
# 8) SALVATAGGIO
# -----------------------------
readr::write_csv(uci_merged, output_csv, na = "")
saveRDS(uci_merged, output_rds)


# -----------------------------
# 9) OUTPUT RIEPILOGO RICHIESTO
# -----------------------------
cat("\n================ RISULTATI ================\n")
cat(sprintf("Dimensioni dataset finale: %d righe x %d colonne\n", nrow(uci_merged), ncol(uci_merged)))

cat("\nNomi delle colonne:\n")
print(names(uci_merged))

cat("\nConteggio righe per source:\n")
print(uci_merged %>% count(source, name = "n_righe"))

cat("\nPrime righe del dataset finale:\n")
print(head(uci_merged))

cat("\nFile salvati:\n")
cat(sprintf("- %s\n", normalizePath(output_csv, winslash = "/", mustWork = FALSE)))
cat(sprintf("- %s\n", normalizePath(output_rds, winslash = "/", mustWork = FALSE)))
