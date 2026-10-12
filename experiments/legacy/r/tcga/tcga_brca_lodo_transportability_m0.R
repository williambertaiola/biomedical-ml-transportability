#!/usr/bin/env Rscript

# Baseline M0 - TCGA-BRCA transportability pipeline
# - Dataset inspection + robust domain/target selection
# - LODO cross-domain with anti-leakage preprocessing
# - ID vs OOD deterioration analysis
# - Training-domain diversity experiment
# - Models: Logistic Regression, Random Forest, XGBoost

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(pROC)
  library(ranger)
  library(xgboost)
  library(jsonlite)
  library(glmnet)
})

options(stringsAsFactors = FALSE)

DEFAULT_FALLBACK_READY <- "data/legacy/tcga_brca_dashi_ready.csv"
DEFAULT_MIN_SOURCE_N <- 25L
DEFAULT_MIN_CLASS_N <- 5L
DEFAULT_MAX_SOURCES <- 14L
DEFAULT_TOP_FEATURES <- 300L

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
clean_names_local <- function(x) {
  x2 <- x %>%
    stringr::str_trim() %>%
    stringr::str_replace_all("[^A-Za-z0-9]+", "_") %>%
    stringr::str_replace_all("_+", "_") %>%
    stringr::str_replace_all("^_|_$", "") %>%
    tolower()
  x2[x2 == ""] <- "x"
  make.unique(x2, sep = "_")
}

find_existing <- function(paths) {
  hit <- paths[file.exists(paths)]
  if (length(hit) == 0) NA_character_ else hit[1]
}

safe_read_csv_robust <- function(path) {
  out <- tryCatch(
    suppressMessages(readr::read_csv(path, show_col_types = FALSE, progress = FALSE)),
    error = function(e) NULL
  )
  if (!is.null(out)) return(as_tibble(out))

  out <- tryCatch(
    suppressMessages(
      readr::read_csv(
        path,
        locale = readr::locale(encoding = "Latin1"),
        show_col_types = FALSE,
        progress = FALSE
      )
    ),
    error = function(e) NULL
  )
  if (!is.null(out)) return(as_tibble(out))

  out <- tryCatch(
    as_tibble(utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE, fileEncoding = "latin1")),
    error = function(e) NULL
  )
  if (is.null(out)) stop("Cannot read CSV file: ", path)
  out
}

safe_read_tsv_robust <- function(path) {
  out <- tryCatch(
    suppressMessages(readr::read_tsv(path, show_col_types = FALSE, progress = FALSE)),
    error = function(e) NULL
  )
  if (!is.null(out)) return(as_tibble(out))

  out <- tryCatch(
    suppressMessages(
      readr::read_tsv(
        path,
        locale = readr::locale(encoding = "Latin1"),
        show_col_types = FALSE,
        progress = FALSE
      )
    ),
    error = function(e) NULL
  )
  if (is.null(out)) stop("Cannot read TSV file: ", path)
  out
}

safe_read_json <- function(path) {
  if (!file.exists(path)) return(NULL)
  tryCatch(jsonlite::read_json(path, simplifyVector = TRUE), error = function(e) NULL)
}

parse_tcga_barcode <- function(sample_id) {
  mat <- stringr::str_match(sample_id, "^TCGA-([A-Z0-9]{2})-([A-Z0-9]{4})-([0-9]{2})")
  tibble::tibble(
    sample_id = sample_id,
    source = mat[, 2],
    case_code = mat[, 3],
    sample_type_code = mat[, 4],
    case_id = ifelse(!is.na(mat[, 2]) & !is.na(mat[, 3]), paste0("TCGA-", mat[, 2], "-", mat[, 3]), NA_character_)
  )
}

encode_stage_binary <- function(x) {
  s <- stringr::str_trim(as.character(x))
  s[s %in% c("", "NA", "NaN", "NULL", "null", "not reported", "--")] <- NA_character_
  out <- rep(NA_real_, length(s))
  late <- stringr::str_detect(s, regex("^Stage\\s*(III|IV)", ignore_case = TRUE))
  early <- stringr::str_detect(s, regex("^Stage\\s*(I(?!I|V)|II)", ignore_case = TRUE))
  out[late] <- 1
  out[early] <- 0
  out
}

safe_auc <- function(y_true_num, p_hat) {
  ok <- !is.na(y_true_num) & !is.na(p_hat)
  y <- y_true_num[ok]
  p <- p_hat[ok]
  if (length(unique(y)) < 2) return(NA_real_)
  roc_obj <- tryCatch(
    pROC::roc(response = y, predictor = p, levels = c(0, 1), direction = "<", quiet = TRUE),
    error = function(e) NULL
  )
  if (is.null(roc_obj)) return(NA_real_)
  as.numeric(pROC::auc(roc_obj))
}

safe_calibration <- function(y_true_num, p_hat) {
  ok <- !is.na(y_true_num) & !is.na(p_hat)
  y <- y_true_num[ok]
  p <- p_hat[ok]
  if (length(y) < 10 || length(unique(y)) < 2) return(list(intercept = NA_real_, slope = NA_real_))

  p_clip <- pmin(pmax(p, 1e-6), 1 - 1e-6)
  lp <- qlogis(p_clip)
  fit_intercept <- tryCatch(suppressWarnings(glm(y ~ offset(lp), family = binomial())), error = function(e) NULL)
  fit_slope <- tryCatch(suppressWarnings(glm(y ~ lp, family = binomial())), error = function(e) NULL)
  intercept <- if (!is.null(fit_intercept)) unname(coef(fit_intercept)[1]) else NA_real_
  slope <- if (!is.null(fit_slope)) unname(coef(fit_slope)[2]) else NA_real_
  list(intercept = intercept, slope = slope)
}

evaluate_metrics <- function(truth, prob, threshold = 0.5) {
  df <- tibble::tibble(
    truth = factor(as.character(truth), levels = c("0", "1")),
    prob = as.numeric(prob)
  ) %>% dplyr::filter(!is.na(truth), !is.na(prob))

  issues <- character(0)
  if (nrow(df) == 0) {
    return(list(
      accuracy = NA_real_,
      auc = NA_real_,
      f1 = NA_real_,
      balacc = NA_real_,
      brier = NA_real_,
      cal_intercept = NA_real_,
      cal_slope = NA_real_,
      n = 0L,
      positives = 0L,
      negatives = 0L,
      issues = "no_complete_cases"
    ))
  }

  y <- ifelse(df$truth == "1", 1, 0)
  pred_class <- ifelse(df$prob >= threshold, 1, 0)
  tp <- sum(pred_class == 1 & y == 1)
  tn <- sum(pred_class == 0 & y == 0)
  fp <- sum(pred_class == 1 & y == 0)
  fn <- sum(pred_class == 0 & y == 1)

  accuracy <- mean(pred_class == y)
  precision <- if ((tp + fp) == 0) NA_real_ else tp / (tp + fp)
  recall <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
  f1 <- if (is.na(precision) || is.na(recall) || (precision + recall) == 0) NA_real_ else 2 * precision * recall / (precision + recall)
  sensitivity <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
  specificity <- if ((tn + fp) == 0) NA_real_ else tn / (tn + fp)
  balacc <- if (is.na(sensitivity) || is.na(specificity)) NA_real_ else (sensitivity + specificity) / 2

  auc <- safe_auc(y_true_num = y, p_hat = df$prob)
  brier <- mean((df$prob - y)^2)
  cal <- safe_calibration(y_true_num = y, p_hat = df$prob)

  if (length(unique(y)) < 2) issues <- c(issues, "single_class_truth")
  if (all(pred_class == 0) || all(pred_class == 1)) issues <- c(issues, "single_class_prediction")

  list(
    accuracy = accuracy,
    auc = auc,
    f1 = f1,
    balacc = balacc,
    brier = brier,
    cal_intercept = cal$intercept,
    cal_slope = cal$slope,
    n = nrow(df),
    positives = sum(y == 1),
    negatives = sum(y == 0),
    issues = if (length(issues) == 0) NA_character_ else paste(unique(issues), collapse = ";")
  )
}

extract_prob_col <- function(pred_df, positive_class = "1") {
  preferred <- paste0(".pred_", positive_class)
  if (preferred %in% names(pred_df)) return(preferred)
  prob_cols <- grep("^\\.pred_", names(pred_df), value = TRUE)
  prob_cols <- setdiff(prob_cols, ".pred_class")
  if (length(prob_cols) == 0) stop("No probability columns found in prediction output.")
  prob_cols[length(prob_cols)]
}

build_calibration_curve <- function(predictions_tbl, n_bins = 10) {
  if (nrow(predictions_tbl) == 0) return(tibble::tibble())
  predictions_tbl %>%
    dplyr::filter(!is.na(prob), !is.na(truth)) %>%
    dplyr::mutate(truth_num = ifelse(as.character(truth) == "1", 1, 0)) %>%
    dplyr::group_by(model, set_type) %>%
    dplyr::mutate(bin = dplyr::ntile(prob, n_bins)) %>%
    dplyr::group_by(model, set_type, bin) %>%
    dplyr::summarise(
      mean_pred = mean(prob, na.rm = TRUE),
      obs_rate = mean(truth_num, na.rm = TRUE),
      n = dplyr::n(),
      .groups = "drop"
    )
}

choose_internal_resamples <- function(train_df, seed = 20260311L) {
  set.seed(seed)
  n_sites <- dplyr::n_distinct(train_df$source)
  if (n_sites >= 2) {
    v <- min(3, n_sites)
    rs <- rsample::group_vfold_cv(train_df, group = source, v = v)
    rs_type <- paste0("group_vfold_cv_source_v", v)
  } else {
    v <- max(2, min(5, nrow(train_df)))
    rs <- rsample::vfold_cv(train_df, v = v, strata = target_bin)
    rs_type <- "vfold_cv_fallback"
  }
  list(resamples = rs, resample_type = rs_type)
}

build_default_train_sizes <- function(n_sources, requested = c(1L, 4L, 8L, 13L)) {
  max_train <- n_sources - 1L
  if (max_train < 1L) return(integer(0))
  out <- sort(unique(as.integer(requested[requested >= 1L & requested <= max_train])))
  if (length(out) == 0) {
    out <- sort(unique(c(1L, floor(max_train / 2), max_train)))
  }
  if (!(max_train %in% out)) out <- sort(unique(c(out, max_train)))
  out
}

inspect_data_directory <- function(data_dir) {
  files <- list.files(data_dir, recursive = TRUE, full.names = TRUE, all.files = FALSE)
  files <- files[file.info(files)$isdir %in% FALSE]
  rel <- gsub(paste0("^", stringr::fixed(paste0(normalizePath(data_dir, mustWork = FALSE), "/"))), "", normalizePath(files, mustWork = FALSE))
  info <- file.info(files)
  tibble::tibble(
    relative_path = rel,
    absolute_path = normalizePath(files, mustWork = FALSE),
    size_mb = round(info$size / (1024^2), 3),
    extension = tolower(tools::file_ext(files)),
    type_guess = dplyr::case_when(
      stringr::str_detect(rel, regex("star_counts|counts", ignore_case = TRUE)) ~ "molecular_features",
      stringr::str_detect(rel, regex("clinical|follow_up|pathology|exposure|family", ignore_case = TRUE)) ~ "clinical_metadata",
      stringr::str_detect(rel, regex("sample", ignore_case = TRUE)) ~ "sample_metadata_or_matrix",
      TRUE ~ "other"
    )
  ) %>% dplyr::arrange(relative_path)
}

read_sidecar_tables <- function(ready_path, data_dir) {
  out <- list(
    preprocessing_summary = tibble::tibble(),
    preprocessing_step_summary = tibble::tibble(),
    source_balance = tibble::tibble(),
    samples_per_source = tibble::tibble(),
    target_distribution_by_source = tibble::tibble(),
    selected_genes = tibble::tibble(),
    sample_audit = tibble::tibble(),
    gene_filter_summary = list()
  )

  candidate_tables_dirs <- c(
    file.path(dirname(dirname(ready_path)), "tables"),
    file.path(data_dir, "analysis_outputs", "tables"),
    file.path(data_dir, "tcga_brca_analysis_output_full", "tables")
  )
  tables_dir <- find_existing(candidate_tables_dirs)
  if (is.na(tables_dir)) return(out)

  read_csv_if <- function(name) {
    p <- file.path(tables_dir, name)
    if (!file.exists(p)) return(tibble::tibble())
    safe_read_csv_robust(p)
  }

  out$preprocessing_summary <- read_csv_if("preprocessing_summary.csv")
  out$preprocessing_step_summary <- read_csv_if("preprocessing_step_summary.csv")
  out$source_balance <- read_csv_if("source_balance_before_source_filter.csv")
  out$samples_per_source <- read_csv_if("samples_per_source.csv")
  out$target_distribution_by_source <- read_csv_if("target_distribution_by_source.csv")
  out$selected_genes <- read_csv_if("selected_genes_top_variance.csv")
  out$sample_audit <- read_csv_if("sample_audit.csv")
  out$gene_filter_summary <- safe_read_json(file.path(tables_dir, "gene_filter_summary.json"))
  out
}

# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------
build_from_matrix_tsv <- function(tsv_path) {
  mat <- safe_read_tsv_robust(tsv_path)
  names(mat) <- clean_names_local(names(mat))
  if (!("ensembl_id" %in% names(mat))) stop("Expected Ensembl_ID in matrix TSV: ", tsv_path)

  gene_ids <- as.character(mat$ensembl_id)
  expr_mat <- mat %>% dplyr::select(-ensembl_id)

  expr_long <- t(expr_mat) %>% as.data.frame(stringsAsFactors = FALSE, check.names = FALSE)
  names(expr_long) <- make.unique(gene_ids, sep = "_")
  expr_long <- tibble::rownames_to_column(expr_long, var = "sample_id")
  expr_long <- expr_long %>% dplyr::mutate(dplyr::across(-sample_id, ~ suppressWarnings(as.numeric(.x))))

  meta <- parse_tcga_barcode(expr_long$sample_id)
  expr_long %>% dplyr::left_join(meta, by = "sample_id")
}

load_data <- function(
  data_dir,
  fallback_ready = DEFAULT_FALLBACK_READY,
  min_source_n = DEFAULT_MIN_SOURCE_N,
  min_class_n = DEFAULT_MIN_CLASS_N,
  max_sources = DEFAULT_MAX_SOURCES
) {
  if (!dir.exists(data_dir)) stop("Data directory does not exist: ", data_dir)

  inventory <- inspect_data_directory(data_dir)

  ready_candidates <- c(
    file.path(data_dir, "tcga_brca_dashi_ready.csv"),
    file.path(data_dir, "analysis_outputs", "data", "tcga_brca_dashi_ready.csv"),
    file.path(data_dir, "tcga_brca_analysis_output_full", "data", "tcga_brca_dashi_ready.csv"),
    fallback_ready
  )
  ready_path <- find_existing(ready_candidates)

  source_mode <- ""
  target_mode <- ""
  target_label <- ""

  if (!is.na(ready_path)) {
    source_mode <- "dashi_ready"
    df <- safe_read_csv_robust(ready_path)
    names(df) <- clean_names_local(names(df))
    sidecars <- read_sidecar_tables(ready_path = ready_path, data_dir = data_dir)
  } else {
    source_mode <- "matrix_tsv"
    sidecars <- read_sidecar_tables(ready_path = "", data_dir = data_dir)
    tsv_path <- find_existing(c(
      file.path(data_dir, "TCGA-BRCA.star_counts.tsv"),
      file.path(data_dir, "sample.tsv")
    ))
    if (is.na(tsv_path)) stop("No suitable matrix file found in: ", data_dir)
    df <- build_from_matrix_tsv(tsv_path)
    names(df) <- clean_names_local(names(df))
  }

  if (!("sample_id" %in% names(df))) {
    df <- df %>% dplyr::mutate(sample_id = paste0("sample_", dplyr::row_number()))
  }

  if (!("source" %in% names(df))) {
    parsed <- parse_tcga_barcode(df$sample_id)
    df$source <- parsed$source
    if (!("sample_type_code" %in% names(df))) df$sample_type_code <- parsed$sample_type_code
  }

  # Target harmonization priority:
  # 1) target column already present in dashi_ready
  # 2) stage-like column in current data
  # 3) sample_type_code fallback (01 vs 11)
  if ("target" %in% names(df)) {
    target_mode <- "provided_target"
    target_label <- "AJCC pathologic stage binary (I/II=0 vs III/IV=1) from dashi_ready"
    df <- df %>% dplyr::mutate(target_bin = factor(ifelse(as.numeric(target) > 0, "1", "0"), levels = c("0", "1")))
  } else if ("ajcc_pathologic_stage" %in% names(df)) {
    target_mode <- "derived_from_stage"
    target_label <- "AJCC pathologic stage binary (I/II=0 vs III/IV=1)"
    df <- df %>%
      dplyr::mutate(target = encode_stage_binary(ajcc_pathologic_stage)) %>%
      dplyr::mutate(target_bin = factor(ifelse(target > 0, "1", "0"), levels = c("0", "1")))
  } else {
    target_mode <- "sample_type_fallback"
    target_label <- "Sample type code fallback (01=0 tumor, 11=1 normal)"
    if (!("sample_type_code" %in% names(df))) {
      parsed <- parse_tcga_barcode(df$sample_id)
      df$sample_type_code <- parsed$sample_type_code
    }
    df <- df %>%
      dplyr::mutate(
        sample_type_code = as.character(sample_type_code),
        target_bin = dplyr::case_when(
          sample_type_code == "01" ~ "0",
          sample_type_code == "11" ~ "1",
          TRUE ~ NA_character_
        ),
        target_bin = factor(target_bin, levels = c("0", "1"))
      )
  }

  df <- df %>% dplyr::filter(!is.na(target_bin))

  df <- df %>%
    dplyr::mutate(source = as.character(source)) %>%
    dplyr::filter(!is.na(source), source != "")

  domain_all <- df %>%
    dplyr::count(source, target_bin, name = "n") %>%
    tidyr::pivot_wider(names_from = target_bin, values_from = n, values_fill = 0, names_prefix = "class_") %>%
    dplyr::mutate(
      n_samples = dplyr::coalesce(class_0, 0L) + dplyr::coalesce(class_1, 0L),
      prevalence_class1 = ifelse(n_samples > 0, dplyr::coalesce(class_1, 0L) / n_samples, NA_real_)
    ) %>%
    dplyr::arrange(dplyr::desc(n_samples), source)

  domain_eligibility <- domain_all %>%
    dplyr::mutate(
      eligible = n_samples >= min_source_n & dplyr::coalesce(class_0, 0L) >= min_class_n & dplyr::coalesce(class_1, 0L) >= min_class_n,
      exclusion_reason = dplyr::case_when(
        eligible ~ "eligible",
        n_samples < min_source_n ~ paste0("n_samples<", min_source_n),
        dplyr::coalesce(class_0, 0L) < min_class_n ~ paste0("class_0<", min_class_n),
        dplyr::coalesce(class_1, 0L) < min_class_n ~ paste0("class_1<", min_class_n),
        TRUE ~ "other"
      )
    )

  selected_sources <- domain_eligibility %>%
    dplyr::filter(eligible) %>%
    dplyr::arrange(dplyr::desc(n_samples), source) %>%
    dplyr::slice_head(n = max_sources)

  # If nothing passes strict criteria, relax gracefully for robustness.
  if (nrow(selected_sources) < 2) {
    selected_sources <- domain_all %>%
      dplyr::arrange(dplyr::desc(n_samples), source) %>%
      dplyr::slice_head(n = min(max_sources, nrow(domain_all)))
  }

  selected_ids <- selected_sources$source
  df <- df %>% dplyr::filter(source %in% selected_ids)

  numeric_cols <- names(df)[vapply(df, is.numeric, logical(1))]
  predictor_cols <- setdiff(numeric_cols, c("target"))

  if (length(predictor_cols) < 20) {
    stop("Too few numeric predictors after filtering: ", length(predictor_cols))
  }

  meta_cols <- c(
    "sample_id", "source", "target", "target_bin", "sample_type_code",
    "ajcc_pathologic_stage", "temporal_date", "case_id"
  )

  out_df <- df %>%
    dplyr::select(dplyr::any_of(c(meta_cols, predictor_cols))) %>%
    dplyr::distinct(sample_id, .keep_all = TRUE) %>%
    dplyr::mutate(row_id = dplyr::row_number())

  site_summary <- out_df %>% dplyr::count(source, name = "n_samples") %>% dplyr::arrange(dplyr::desc(n_samples), source)
  class_summary <- out_df %>% dplyr::count(source, target_bin, name = "n") %>% dplyr::arrange(source, target_bin)

  domain_proxy_candidates <- tibble::tibble(
    proxy = c("barcode_tss_source", "clinical_primary_site", "batch_or_plate"),
    availability = c(
      ifelse("source" %in% names(out_df), "available", "missing"),
      ifelse(any(stringr::str_detect(names(df), regex("primary_site", ignore_case = TRUE))), "partially_available", "not_in_model_table"),
      ifelse(any(stringr::str_detect(names(df), regex("batch|plate|shipment|center", ignore_case = TRUE))), "available", "not_available")
    ),
    selected = c(TRUE, FALSE, FALSE),
    rationale = c(
      "Tissue Source Site (TSS) encoded in TCGA barcode is complete at sample level and best proxy for cross-site transportability.",
      "Clinical primary site is disease-level and not an acquisition domain proxy in this matrix.",
      "No robust batch/plate fields available in final supervised table."
    )
  )

  target_definition <- tibble::tibble(
    target_mode = target_mode,
    target_label = target_label,
    positive_class = "1",
    positive_meaning = ifelse(stringr::str_detect(target_label, "AJCC"), "Late stage (III/IV)", "Fallback class (Normal sample)")
  )

  files_used <- tibble::tibble(
    source_mode = source_mode,
    ready_path = ifelse(is.na(ready_path), "", ready_path),
    data_dir = data_dir,
    min_source_n = min_source_n,
    min_class_n = min_class_n,
    max_sources = max_sources,
    n_samples_final = nrow(out_df),
    n_sources_final = dplyr::n_distinct(out_df$source),
    n_predictors_final = length(predictor_cols)
  )

  list(
    data = out_df,
    files_used = files_used,
    site_summary = site_summary,
    class_summary = class_summary,
    domain_all = domain_all,
    domain_eligibility = domain_eligibility,
    selected_sources = selected_sources,
    domain_proxy_candidates = domain_proxy_candidates,
    target_definition = target_definition,
    dataset_inventory = inventory,
    sidecars = sidecars
  )
}

preprocess_fold <- function(train_df, max_features = DEFAULT_TOP_FEATURES) {
  model_train <- train_df %>%
    dplyr::select(-dplyr::any_of(c("source", "sample_id", "row_id", "target", "sample_type_code", "sample_type_label", "ajcc_pathologic_stage", "temporal_date", "case_id")))

  predictor_cols <- setdiff(names(model_train), "target_bin")
  if (length(predictor_cols) < 2) stop("Not enough predictors for preprocessing.")

  v <- vapply(
    model_train[predictor_cols],
    function(x) {
      x_num <- suppressWarnings(as.numeric(x))
      out <- stats::var(x_num, na.rm = TRUE)
      ifelse(is.finite(out), out, NA_real_)
    },
    numeric(1)
  )

  ranked <- names(sort(v, decreasing = TRUE, na.last = NA))
  if (length(ranked) == 0) ranked <- predictor_cols
  n_keep <- min(as.integer(max_features), length(ranked))
  selected <- ranked[seq_len(n_keep)]

  model_train_small <- model_train %>% dplyr::select(target_bin, dplyr::all_of(selected))

  rec <- recipes::recipe(target_bin ~ ., data = model_train_small) %>%
    recipes::step_impute_median(recipes::all_numeric_predictors()) %>%
    recipes::step_normalize(recipes::all_numeric_predictors()) %>%
    recipes::step_zv(recipes::all_predictors())

  list(
    recipe = rec,
    selected_features = selected,
    n_selected = length(selected)
  )
}

train_logistic <- function() {
  parsnip::logistic_reg(mode = "classification", penalty = 0.01, mixture = 1) %>%
    parsnip::set_engine("glmnet")
}

train_rf <- function(seed = 20260311L) {
  parsnip::rand_forest(mode = "classification", trees = 300, min_n = 5) %>%
    parsnip::set_engine("ranger", probability = TRUE, seed = seed, num.threads = 1)
}

train_xgb <- function(seed = 20260311L) {
  parsnip::boost_tree(
    mode = "classification",
    trees = 140,
    tree_depth = 4,
    learn_rate = 0.05,
    min_n = 5,
    sample_size = 0.8,
    loss_reduction = 0
  ) %>%
    parsnip::set_engine("xgboost", objective = "binary:logistic", eval_metric = "auc", nthread = 1)
}

# -------------------------------------------------------------------------
# Experiments
# -------------------------------------------------------------------------
run_lodo_pipeline <- function(df, seed = 20260311L, save_models = TRUE, max_features = DEFAULT_TOP_FEATURES) {
  sites <- sort(unique(df$source))
  if (length(sites) < 2) stop("LODO requires at least 2 sources.")

  model_specs <- list(
    logistic = train_logistic(),
    random_forest = train_rf(seed = seed),
    xgboost = train_xgb(seed = seed)
  )

  all_results <- list()
  all_predictions <- list()
  fitted_models <- list()
  ctr <- 1L

  for (i in seq_along(sites)) {
    holdout_site <- sites[i]
    train_df <- df %>% dplyr::filter(source != holdout_site)
    test_df <- df %>% dplyr::filter(source == holdout_site)
    rs_obj <- choose_internal_resamples(train_df = train_df, seed = seed + i)

    prep_obj <- preprocess_fold(train_df = train_df, max_features = max_features)

    for (m in seq_along(model_specs)) {
      model_name <- names(model_specs)[m]
      spec <- model_specs[[m]]

      wf <- workflows::workflow() %>%
        workflows::add_recipe(prep_obj$recipe) %>%
        workflows::add_model(spec)

      set.seed(seed + i * 100 + m)
      id_rs <- tryCatch(
        tune::fit_resamples(
          wf,
          resamples = rs_obj$resamples,
          control = tune::control_resamples(save_pred = TRUE, verbose = FALSE, allow_par = FALSE)
        ),
        error = function(e) e
      )

      id_metrics <- list(
        accuracy = NA_real_, auc = NA_real_, f1 = NA_real_, balacc = NA_real_, brier = NA_real_,
        cal_intercept = NA_real_, cal_slope = NA_real_,
        n = 0L, positives = 0L, negatives = 0L, issues = NA_character_
      )
      id_pred_store <- tibble::tibble()

      if (inherits(id_rs, "error")) {
        id_metrics$issues <- paste0("id_resample_error: ", id_rs$message)
      } else {
        id_preds <- tune::collect_predictions(id_rs)
        id_prob_col <- extract_prob_col(id_preds, positive_class = "1")
        id_metrics <- evaluate_metrics(truth = id_preds$target_bin, prob = id_preds[[id_prob_col]])
        id_pred_store <- id_preds %>%
          dplyr::transmute(
            holdout_site = holdout_site,
            model = model_name,
            set_type = "ID",
            truth = factor(as.character(target_bin), levels = c("0", "1")),
            prob = .data[[id_prob_col]],
            pred_class = as.character(.pred_class)
          )
      }

      final_fit <- tryCatch(parsnip::fit(wf, data = train_df), error = function(e) e)
      ood_metrics <- list(
        accuracy = NA_real_, auc = NA_real_, f1 = NA_real_, balacc = NA_real_, brier = NA_real_,
        cal_intercept = NA_real_, cal_slope = NA_real_,
        n = 0L, positives = 0L, negatives = 0L, issues = NA_character_
      )
      ood_pred_store <- tibble::tibble()

      if (inherits(final_fit, "error")) {
        ood_metrics$issues <- paste0("ood_fit_error: ", final_fit$message)
      } else {
        ood_prob <- predict(final_fit, new_data = test_df, type = "prob")
        ood_class <- predict(final_fit, new_data = test_df, type = "class")
        ood_prob_col <- extract_prob_col(ood_prob, positive_class = "1")
        ood_preds <- dplyr::bind_cols(test_df %>% dplyr::select(target_bin), ood_prob, ood_class)
        ood_metrics <- evaluate_metrics(truth = ood_preds$target_bin, prob = ood_preds[[ood_prob_col]])
        ood_pred_store <- ood_preds %>%
          dplyr::transmute(
            holdout_site = holdout_site,
            model = model_name,
            set_type = "OOD",
            truth = factor(as.character(target_bin), levels = c("0", "1")),
            prob = .data[[ood_prob_col]],
            pred_class = as.character(.pred_class)
          )
        if (save_models) fitted_models[[paste(holdout_site, model_name, sep = "__")]] <- final_fit
      }

      all_results[[ctr]] <- tibble::tibble(
        holdout_site = holdout_site,
        model = model_name,
        inner_validation = rs_obj$resample_type,
        n_train = nrow(train_df),
        n_test = nrow(test_df),
        n_selected_features = prep_obj$n_selected,
        pos_rate_train = mean(train_df$target_bin == "1"),
        pos_rate_test = mean(test_df$target_bin == "1"),
        accuracy_id = id_metrics$accuracy,
        accuracy_ood = ood_metrics$accuracy,
        auc_id = id_metrics$auc,
        auc_ood = ood_metrics$auc,
        f1_id = id_metrics$f1,
        f1_ood = ood_metrics$f1,
        balacc_id = id_metrics$balacc,
        balacc_ood = ood_metrics$balacc,
        brier_id = id_metrics$brier,
        brier_ood = ood_metrics$brier,
        cal_intercept_id = id_metrics$cal_intercept,
        cal_slope_id = id_metrics$cal_slope,
        cal_intercept_ood = ood_metrics$cal_intercept,
        cal_slope_ood = ood_metrics$cal_slope,
        accuracy_gap = id_metrics$accuracy - ood_metrics$accuracy,
        auc_gap = id_metrics$auc - ood_metrics$auc,
        f1_gap = id_metrics$f1 - ood_metrics$f1,
        balacc_gap = id_metrics$balacc - ood_metrics$balacc,
        brier_gap = id_metrics$brier - ood_metrics$brier,
        id_issues = id_metrics$issues,
        ood_issues = ood_metrics$issues
      )
      all_predictions[[ctr]] <- dplyr::bind_rows(id_pred_store, ood_pred_store)
      ctr <- ctr + 1L
    }
  }

  list(
    results = dplyr::bind_rows(all_results),
    predictions = dplyr::bind_rows(all_predictions),
    models = fitted_models
  )
}

run_training_domain_diversity_experiment <- function(
  df,
  seed = 20260311L,
  train_sizes = c(1L, 4L, 8L, 13L),
  max_combinations_per_size = 40,
  max_features = DEFAULT_TOP_FEATURES
) {
  all_sites <- sort(unique(df$source))
  model_specs <- list(
    logistic = train_logistic(),
    random_forest = train_rf(seed = seed),
    xgboost = train_xgb(seed = seed)
  )

  rows_out <- list()
  row_idx <- 1L
  sampling_log <- list()
  sampling_idx <- 1L

  for (i in seq_along(all_sites)) {
    test_site <- all_sites[i]
    remaining_sites <- setdiff(all_sites, test_site)
    valid_sizes <- sort(unique(train_sizes[train_sizes > 0 & train_sizes <= length(remaining_sites)]))
    if (length(valid_sizes) == 0) next
    message("Diversity experiment - test source ", test_site, " | sizes: ", paste(valid_sizes, collapse = ", "))
    test_df <- df %>% dplyr::filter(source == test_site)
    combo_counter <- 1L

    for (k in valid_sizes) {
      all_sets <- combn(remaining_sites, k, simplify = FALSE)
      n_total <- length(all_sets)
      train_site_sets <- all_sets
      sampled <- FALSE

      if (is.finite(max_combinations_per_size) && n_total > max_combinations_per_size) {
        set.seed(seed + i * 100 + k)
        idx <- sort(sample.int(n_total, size = max_combinations_per_size, replace = FALSE))
        train_site_sets <- all_sets[idx]
        sampled <- TRUE
      }

      sampling_log[[sampling_idx]] <- tibble::tibble(
        test_site = test_site,
        n_train_sites = k,
        n_combinations_total = n_total,
        n_combinations_used = length(train_site_sets),
        sampled = sampled
      )
      sampling_idx <- sampling_idx + 1L

      message("  Train size ", k, " -> combinations used: ", length(train_site_sets), " (total=", n_total, ")")

      for (j in seq_along(train_site_sets)) {
        train_sites <- train_site_sets[[j]]
        train_df <- df %>% dplyr::filter(source %in% train_sites)
        train_sites_label <- paste(train_sites, collapse = " + ")
        n_train_sites <- length(train_sites)

        prep_obj <- preprocess_fold(train_df = train_df, max_features = max_features)

        for (m in seq_along(model_specs)) {
          model_name <- names(model_specs)[m]
          spec <- model_specs[[m]]

          wf <- workflows::workflow() %>%
            workflows::add_recipe(prep_obj$recipe) %>%
            workflows::add_model(spec)

          set.seed(seed + i * 100000 + combo_counter * 10 + m)
          fit_obj <- tryCatch(parsnip::fit(wf, data = train_df), error = function(e) e)

          acc <- NA_real_
          bal_acc <- NA_real_
          auc <- NA_real_
          f1 <- NA_real_
          brier <- NA_real_
          cal_intercept <- NA_real_
          cal_slope <- NA_real_
          issue_msg <- NA_character_

          if (inherits(fit_obj, "error")) {
            issue_msg <- paste0("fit_error: ", fit_obj$message)
          } else {
            prob_pred <- tryCatch(predict(fit_obj, new_data = test_df, type = "prob"), error = function(e) e)
            cls_pred <- tryCatch(predict(fit_obj, new_data = test_df, type = "class"), error = function(e) e)
            if (inherits(prob_pred, "error") || inherits(cls_pred, "error")) {
              issue_msg <- "predict_error"
            } else {
              prob_col <- extract_prob_col(prob_pred, positive_class = "1")
              metrics <- evaluate_metrics(truth = test_df$target_bin, prob = prob_pred[[prob_col]])
              truth_chr <- as.character(test_df$target_bin)
              pred_chr <- as.character(cls_pred$.pred_class)
              acc <- mean(pred_chr == truth_chr, na.rm = TRUE)
              bal_acc <- metrics$balacc
              auc <- metrics$auc
              f1 <- metrics$f1
              brier <- metrics$brier
              cal_intercept <- metrics$cal_intercept
              cal_slope <- metrics$cal_slope
              issue_msg <- metrics$issues
            }
          }

          rows_out[[row_idx]] <- tibble::tibble(
            test_site = test_site,
            train_sites = train_sites_label,
            n_train_sites = n_train_sites,
            model = model_name,
            n_selected_features = prep_obj$n_selected,
            accuracy = acc,
            balanced_accuracy = bal_acc,
            auc = auc,
            f1 = f1,
            brier = brier,
            cal_intercept = cal_intercept,
            cal_slope = cal_slope,
            n_train_samples = nrow(train_df),
            n_test_samples = nrow(test_df),
            issues = issue_msg
          )
          row_idx <- row_idx + 1L
        }

        combo_counter <- combo_counter + 1L
      }
    }
  }

  results_tbl <- dplyr::bind_rows(rows_out) %>%
    dplyr::arrange(test_site, n_train_sites, train_sites, model)

  summary_by_model <- results_tbl %>%
    dplyr::group_by(model, n_train_sites) %>%
    dplyr::summarise(
      n_experiments = dplyr::n(),
      mean_accuracy = mean(accuracy, na.rm = TRUE),
      sd_accuracy = sd(accuracy, na.rm = TRUE),
      mean_balanced_accuracy = mean(balanced_accuracy, na.rm = TRUE),
      sd_balanced_accuracy = sd(balanced_accuracy, na.rm = TRUE),
      mean_auc = mean(auc, na.rm = TRUE),
      sd_auc = sd(auc, na.rm = TRUE),
      .groups = "drop"
    ) %>% dplyr::arrange(model, n_train_sites)

  summary_by_site <- results_tbl %>%
    dplyr::group_by(test_site, model, n_train_sites) %>%
    dplyr::summarise(
      n_experiments = dplyr::n(),
      mean_accuracy = mean(accuracy, na.rm = TRUE),
      sd_accuracy = sd(accuracy, na.rm = TRUE),
      mean_balanced_accuracy = mean(balanced_accuracy, na.rm = TRUE),
      sd_balanced_accuracy = sd(balanced_accuracy, na.rm = TRUE),
      mean_auc = mean(auc, na.rm = TRUE),
      sd_auc = sd(auc, na.rm = TRUE),
      .groups = "drop"
    ) %>% dplyr::arrange(test_site, model, n_train_sites)

  list(
    results = results_tbl,
    summary_by_model = summary_by_model,
    summary_by_site = summary_by_site,
    combination_sampling_log = dplyr::bind_rows(sampling_log)
  )
}

summarize_results <- function(results_tbl) {
  results_tbl %>%
    dplyr::group_by(model) %>%
    dplyr::summarise(
      folds = dplyr::n(),
      accuracy_ood_mean = mean(accuracy_ood, na.rm = TRUE),
      accuracy_ood_sd = sd(accuracy_ood, na.rm = TRUE),
      auc_ood_mean = mean(auc_ood, na.rm = TRUE),
      auc_ood_sd = sd(auc_ood, na.rm = TRUE),
      f1_ood_mean = mean(f1_ood, na.rm = TRUE),
      f1_ood_sd = sd(f1_ood, na.rm = TRUE),
      balacc_ood_mean = mean(balacc_ood, na.rm = TRUE),
      balacc_ood_sd = sd(balacc_ood, na.rm = TRUE),
      brier_ood_mean = mean(brier_ood, na.rm = TRUE),
      brier_ood_sd = sd(brier_ood, na.rm = TRUE),
      accuracy_gap_mean = mean(accuracy_gap, na.rm = TRUE),
      auc_gap_mean = mean(auc_gap, na.rm = TRUE),
      f1_gap_mean = mean(f1_gap, na.rm = TRUE),
      balacc_gap_mean = mean(balacc_gap, na.rm = TRUE),
      brier_gap_mean = mean(brier_gap, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      rank_acc = dplyr::min_rank(dplyr::desc(accuracy_ood_mean)),
      rank_auc = dplyr::min_rank(dplyr::desc(auc_ood_mean)),
      rank_f1 = dplyr::min_rank(dplyr::desc(f1_ood_mean)),
      rank_balacc = dplyr::min_rank(dplyr::desc(balacc_ood_mean)),
      rank_brier = dplyr::min_rank(brier_ood_mean),
      mean_rank = (rank_acc + rank_auc + rank_f1 + rank_balacc + rank_brier) / 5,
      final_rank = dplyr::min_rank(mean_rank)
    ) %>%
    dplyr::arrange(final_rank, dplyr::desc(auc_ood_mean))
}

# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------
plot_results <- function(results_tbl, predictions_tbl, plots_dir) {
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

  base_theme <- ggplot2::theme_bw(base_size = 12) +
    ggplot2::theme(
      legend.position = "bottom",
      panel.grid.minor = ggplot2::element_blank(),
      strip.background = ggplot2::element_rect(fill = "grey95", color = "grey80")
    )

  ood_long <- results_tbl %>%
    dplyr::select(holdout_site, model, accuracy_ood, auc_ood, f1_ood, balacc_ood, brier_ood) %>%
    tidyr::pivot_longer(cols = c(accuracy_ood, auc_ood, f1_ood, balacc_ood, brier_ood), names_to = "metric", values_to = "value")
  p_ood <- ggplot2::ggplot(ood_long, ggplot2::aes(x = holdout_site, y = value, color = model, group = model)) +
    ggplot2::geom_point(size = 2.8) +
    ggplot2::geom_line(alpha = 0.75) +
    ggplot2::facet_wrap(~ metric, scales = "free_y", ncol = 2) +
    ggplot2::labs(title = "OOD performance by holdout source and model", x = "Holdout source", y = "Metric value") +
    base_theme

  gap_long <- results_tbl %>%
    dplyr::select(holdout_site, model, accuracy_gap, auc_gap, f1_gap, balacc_gap, brier_gap) %>%
    tidyr::pivot_longer(cols = c(accuracy_gap, auc_gap, f1_gap, balacc_gap, brier_gap), names_to = "metric", values_to = "gap")
  p_gap <- ggplot2::ggplot(gap_long, ggplot2::aes(x = holdout_site, y = gap, fill = model)) +
    ggplot2::geom_col(position = ggplot2::position_dodge(width = 0.8), width = 0.72) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "grey35") +
    ggplot2::facet_wrap(~ metric, scales = "free_y", ncol = 2) +
    ggplot2::labs(title = "ID - OOD deterioration gap by holdout source", x = "Holdout source", y = "Gap (ID - OOD)") +
    base_theme

  heatmap_df <- results_tbl %>%
    dplyr::select(holdout_site, model, accuracy_ood, auc_ood, f1_ood, balacc_ood) %>%
    tidyr::pivot_longer(cols = c(accuracy_ood, auc_ood, f1_ood, balacc_ood), names_to = "metric", values_to = "value")
  p_heat <- ggplot2::ggplot(heatmap_df, ggplot2::aes(x = model, y = holdout_site, fill = value)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.6) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.3f", value)), size = 2.9) +
    ggplot2::facet_wrap(~ metric, ncol = 2) +
    ggplot2::scale_fill_viridis_c(option = "C", na.value = "grey85") +
    ggplot2::labs(title = "Heatmap: source x model OOD performance", x = "Model", y = "Holdout source", fill = "Value") +
    base_theme

  dist_df <- results_tbl %>%
    dplyr::select(model, accuracy_ood, auc_ood, f1_ood, balacc_ood, brier_ood) %>%
    tidyr::pivot_longer(cols = c(accuracy_ood, auc_ood, f1_ood, balacc_ood, brier_ood), names_to = "metric", values_to = "value")
  p_dist <- ggplot2::ggplot(dist_df, ggplot2::aes(x = model, y = value, fill = model)) +
    ggplot2::geom_boxplot(alpha = 0.7, outlier.shape = NA) +
    ggplot2::geom_jitter(width = 0.16, alpha = 0.45, size = 1.4) +
    ggplot2::facet_wrap(~ metric, scales = "free_y", ncol = 2) +
    ggplot2::labs(title = "Distribution of OOD metrics by model", x = "Model", y = "OOD metric") +
    base_theme

  calibration_df <- build_calibration_curve(predictions_tbl, n_bins = 10)
  p_cal <- ggplot2::ggplot(calibration_df, ggplot2::aes(x = mean_pred, y = obs_rate)) +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey25") +
    ggplot2::geom_line(ggplot2::aes(color = model), linewidth = 0.9) +
    ggplot2::geom_point(ggplot2::aes(color = model, size = n), alpha = 0.9) +
    ggplot2::facet_grid(set_type ~ model) +
    ggplot2::coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    ggplot2::scale_size_continuous(range = c(1.8, 6)) +
    ggplot2::labs(title = "Reliability diagrams (ID vs OOD)", x = "Mean predicted risk", y = "Observed event rate") +
    base_theme

  ggplot2::ggsave(file.path(plots_dir, "ood_performance_by_site_model.png"), p_ood, width = 12, height = 8, dpi = 320)
  ggplot2::ggsave(file.path(plots_dir, "id_vs_ood_gap.png"), p_gap, width = 12, height = 8, dpi = 320)
  ggplot2::ggsave(file.path(plots_dir, "heatmap_site_model_ood.png"), p_heat, width = 12, height = 7, dpi = 320)
  ggplot2::ggsave(file.path(plots_dir, "ood_metric_distribution_by_model.png"), p_dist, width = 12, height = 8, dpi = 320)
  ggplot2::ggsave(file.path(plots_dir, "calibration_reliability_id_ood.png"), p_cal, width = 13, height = 8, dpi = 320)
}

plot_training_domain_diversity_experiment <- function(summary_by_model, summary_by_site, plots_dir) {
  if (nrow(summary_by_model) == 0 || nrow(summary_by_site) == 0) return(invisible(NULL))

  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

  summary_by_model <- summary_by_model %>%
    dplyr::mutate(
      n_train_sites = factor(n_train_sites, levels = sort(unique(n_train_sites))),
      model = factor(model, levels = sort(unique(model)))
    )
  summary_by_site <- summary_by_site %>%
    dplyr::mutate(
      n_train_sites = factor(n_train_sites, levels = sort(unique(n_train_sites))),
      model = factor(model, levels = sort(unique(model))),
      test_site = factor(test_site, levels = sort(unique(test_site)))
    )

  theme_pub <- ggplot2::theme_bw(base_size = 12) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major = ggplot2::element_blank(),
      strip.background = ggplot2::element_rect(fill = "grey95", color = "grey80"),
      legend.position = "bottom"
    )

  p_main_heatmap <- ggplot2::ggplot(summary_by_model, ggplot2::aes(x = n_train_sites, y = model, fill = mean_accuracy)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.7) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.3f", mean_accuracy)), size = 3.6) +
    ggplot2::scale_fill_viridis_c(option = "C", na.value = "grey85") +
    ggplot2::labs(
      title = "Effect of training domain diversity on transportability",
      x = "Number of training sites",
      y = "Model",
      fill = "Mean accuracy"
    ) + theme_pub

  p_site_heatmap <- ggplot2::ggplot(summary_by_site, ggplot2::aes(x = n_train_sites, y = test_site, fill = mean_accuracy)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.7) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.3f", mean_accuracy)), size = 2.8) +
    ggplot2::facet_wrap(~ model, ncol = 3) +
    ggplot2::scale_fill_viridis_c(option = "C", na.value = "grey85") +
    ggplot2::labs(
      title = "Transportability by held-out source as training domains increase",
      x = "Number of training sites",
      y = "Held-out source",
      fill = "Mean accuracy"
    ) + theme_pub

  line_df <- summary_by_model %>% dplyr::mutate(n_train_sites_num = as.integer(as.character(n_train_sites)))
  p_line <- ggplot2::ggplot(line_df, ggplot2::aes(x = n_train_sites_num, y = mean_accuracy, color = model, group = model)) +
    ggplot2::geom_line(linewidth = 0.95) +
    ggplot2::geom_point(size = 2.5) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.3f", mean_accuracy)), vjust = -0.9, size = 3, show.legend = FALSE) +
    ggplot2::scale_x_continuous(breaks = sort(unique(line_df$n_train_sites_num))) +
    ggplot2::coord_cartesian(ylim = c(0, 1)) +
    ggplot2::labs(
      title = "Training-domain diversity trend in OOD accuracy",
      x = "Number of training sites",
      y = "Mean accuracy",
      color = "Model"
    ) + theme_pub

  ggplot2::ggsave(file.path(plots_dir, "transportability_training_domains_heatmap.png"), p_main_heatmap, width = 10, height = 5.2, dpi = 320)
  ggplot2::ggsave(file.path(plots_dir, "transportability_training_domains_by_site_heatmap.png"), p_site_heatmap, width = 12, height = 8.4, dpi = 320)
  ggplot2::ggsave(file.path(plots_dir, "transportability_training_domains_lineplot.png"), p_line, width = 10, height = 5.2, dpi = 320)
}

plot_training_domain_combinations_heatmap <- function(results_tbl, plots_dir) {
  if (nrow(results_tbl) == 0) return(invisible(NULL))

  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

  combo_map <- results_tbl %>%
    dplyr::distinct(test_site, n_train_sites, train_sites) %>%
    dplyr::arrange(test_site, n_train_sites, train_sites) %>%
    dplyr::group_by(test_site) %>%
    dplyr::mutate(
      combo_rank = dplyr::row_number(),
      combo_label = paste0("Train ", n_train_sites, " site(s) | ", train_sites),
      combo_key = paste0(test_site, " :: ", sprintf("%04d", combo_rank), " :: ", combo_label)
    ) %>%
    dplyr::ungroup()

  plot_df <- results_tbl %>%
    dplyr::left_join(combo_map %>% dplyr::select(test_site, n_train_sites, train_sites, combo_key), by = c("test_site", "n_train_sites", "train_sites")) %>%
    dplyr::mutate(model = factor(model, levels = sort(unique(model))), test_site = factor(test_site, levels = sort(unique(test_site))))

  p_combo <- ggplot2::ggplot(plot_df, ggplot2::aes(x = model, y = combo_key, fill = accuracy)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.65) +
    ggplot2::facet_grid(test_site ~ ., scales = "free_y", space = "free_y") +
    ggplot2::scale_y_discrete(labels = function(x) sub("^.* :: [0-9]+ :: ", "", x)) +
    ggplot2::scale_fill_viridis_c(option = "C", na.value = "grey85") +
    ggplot2::labs(
      title = "Sampled training-source combinations for each held-out source",
      x = "Model",
      y = "Training combination",
      fill = "Accuracy"
    ) +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(
      panel.grid = ggplot2::element_blank(),
      strip.background = ggplot2::element_rect(fill = "grey95", color = "grey80"),
      legend.position = "bottom"
    )

  ggplot2::ggsave(
    file.path(plots_dir, "transportability_training_domains_all_combinations_heatmap.png"),
    p_combo,
    width = 14,
    height = 17,
    dpi = 320
  )
}

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
main <- function(
  data_dir = "data/legacy/TCGA-BRCA",
  output_dir = file.path(getwd(), "outputs_m0_transportability_tcga_brca"),
  seed = 20260311L,
  train_sizes = c(1L, 4L, 8L, 13L),
  max_combinations_per_size = 40,
  max_features = DEFAULT_TOP_FEATURES,
  min_source_n = DEFAULT_MIN_SOURCE_N,
  min_class_n = DEFAULT_MIN_CLASS_N,
  max_sources = DEFAULT_MAX_SOURCES
) {
  set.seed(seed)

  dirs <- list(
    root = output_dir,
    tables = file.path(output_dir, "tables"),
    plots = file.path(output_dir, "plots"),
    objects = file.path(output_dir, "objects"),
    report = file.path(output_dir, "report")
  )
  purrr::walk(dirs, ~ dir.create(.x, recursive = TRUE, showWarnings = FALSE))

  message("Loading TCGA-BRCA data from: ", data_dir)
  ld <- load_data(
    data_dir = data_dir,
    min_source_n = min_source_n,
    min_class_n = min_class_n,
    max_sources = max_sources
  )
  df <- ld$data

  if (nrow(df) == 0 || dplyr::n_distinct(df$source) < 2) {
    stop("Data not suitable for transportability (need >=2 sources with samples).")
  }

  train_sizes_final <- build_default_train_sizes(dplyr::n_distinct(df$source), requested = train_sizes)

  # Core metadata / inspection tables
  readr::write_csv(ld$files_used, file.path(dirs$tables, "files_used.csv"))
  readr::write_csv(ld$dataset_inventory, file.path(dirs$tables, "dataset_file_inventory.csv"))
  readr::write_csv(ld$domain_proxy_candidates, file.path(dirs$tables, "domain_proxy_candidates.csv"))
  readr::write_csv(ld$target_definition, file.path(dirs$tables, "target_definition.csv"))
  readr::write_csv(ld$domain_all, file.path(dirs$tables, "domain_counts_all.csv"))
  readr::write_csv(ld$domain_eligibility, file.path(dirs$tables, "domain_eligibility.csv"))
  readr::write_csv(ld$site_summary, file.path(dirs$tables, "site_summary.csv"))
  readr::write_csv(ld$class_summary, file.path(dirs$tables, "class_summary.csv"))
  readr::write_csv(ld$selected_sources, file.path(dirs$tables, "selected_sources.csv"))

  # Sidecar preprocessing tables from dashi build (if available)
  if (nrow(ld$sidecars$preprocessing_summary) > 0) {
    readr::write_csv(ld$sidecars$preprocessing_summary, file.path(dirs$tables, "preprocessing_summary.csv"))
  }
  if (nrow(ld$sidecars$preprocessing_step_summary) > 0) {
    readr::write_csv(ld$sidecars$preprocessing_step_summary, file.path(dirs$tables, "preprocessing_step_summary.csv"))
  }
  if (nrow(ld$sidecars$source_balance) > 0) {
    readr::write_csv(ld$sidecars$source_balance, file.path(dirs$tables, "source_balance_before_source_filter.csv"))
  }
  if (nrow(ld$sidecars$samples_per_source) > 0) {
    readr::write_csv(ld$sidecars$samples_per_source, file.path(dirs$tables, "samples_per_source_sidecar.csv"))
  }
  if (nrow(ld$sidecars$target_distribution_by_source) > 0) {
    readr::write_csv(ld$sidecars$target_distribution_by_source, file.path(dirs$tables, "target_distribution_by_source_sidecar.csv"))
  }
  if (nrow(ld$sidecars$selected_genes) > 0) {
    readr::write_csv(ld$sidecars$selected_genes, file.path(dirs$tables, "selected_genes_top_variance_sidecar.csv"))
  }
  if (nrow(ld$sidecars$sample_audit) > 0) {
    readr::write_csv(ld$sidecars$sample_audit, file.path(dirs$tables, "sample_audit_sidecar.csv"))
  }

  if (length(ld$sidecars$gene_filter_summary) > 0) {
    gf <- ld$sidecars$gene_filter_summary
    gf_tbl <- tibble::tibble(
      metric = names(gf),
      value = unlist(gf, use.names = FALSE)
    )
    readr::write_csv(gf_tbl, file.path(dirs$tables, "gene_filter_summary_sidecar.csv"))
  }

  analysis_config <- tibble::tibble(
    seed = seed,
    max_features = max_features,
    min_source_n = min_source_n,
    min_class_n = min_class_n,
    max_sources = max_sources,
    max_combinations_per_size = max_combinations_per_size,
    train_sizes_requested = paste(train_sizes, collapse = ","),
    train_sizes_used = paste(train_sizes_final, collapse = ",")
  )
  readr::write_csv(analysis_config, file.path(dirs$tables, "analysis_config.csv"))

  message("Running LODO pipeline across ", dplyr::n_distinct(df$source), " selected sources...")
  out <- run_lodo_pipeline(df = df, seed = seed, save_models = TRUE, max_features = max_features)
  results_tbl <- out$results %>% dplyr::arrange(holdout_site, model)
  summary_tbl <- summarize_results(results_tbl)

  readr::write_csv(results_tbl, file.path(dirs$tables, "lodo_fold_level_results.csv"))
  readr::write_csv(summary_tbl, file.path(dirs$tables, "lodo_model_summary.csv"))
  readr::write_csv(out$predictions, file.path(dirs$tables, "id_ood_predictions.csv"))
  plot_results(results_tbl, out$predictions, plots_dir = dirs$plots)

  message(
    "Running training-domain diversity experiment (train sizes used: ",
    paste(train_sizes_final, collapse = ","),
    ")..."
  )

  td_out <- run_training_domain_diversity_experiment(
    df = df,
    seed = seed,
    train_sizes = as.integer(train_sizes_final),
    max_combinations_per_size = max_combinations_per_size,
    max_features = max_features
  )

  readr::write_csv(td_out$results, file.path(dirs$tables, "training_domain_experiment_results.csv"))
  readr::write_csv(td_out$summary_by_model, file.path(dirs$tables, "training_domain_experiment_summary_by_model.csv"))
  readr::write_csv(td_out$summary_by_site, file.path(dirs$tables, "training_domain_experiment_summary_by_site.csv"))
  readr::write_csv(td_out$combination_sampling_log, file.path(dirs$tables, "training_domain_combination_sampling_log.csv"))

  plot_training_domain_diversity_experiment(td_out$summary_by_model, td_out$summary_by_site, plots_dir = dirs$plots)
  plot_training_domain_combinations_heatmap(td_out$results, plots_dir = dirs$plots)

  saveRDS(
    list(
      config = list(
        data_dir = data_dir,
        output_dir = output_dir,
        seed = seed,
        train_sizes_requested = as.integer(train_sizes),
        train_sizes_used = as.integer(train_sizes_final),
        max_combinations_per_size = max_combinations_per_size,
        max_features = max_features,
        min_source_n = min_source_n,
        min_class_n = min_class_n,
        max_sources = max_sources,
        timestamp = as.character(Sys.time())
      ),
      files_used = ld$files_used,
      dataset_inventory = ld$dataset_inventory,
      domain_proxy_candidates = ld$domain_proxy_candidates,
      target_definition = ld$target_definition,
      domain_all = ld$domain_all,
      domain_eligibility = ld$domain_eligibility,
      selected_sources = ld$selected_sources,
      data = df,
      site_summary = ld$site_summary,
      class_summary = ld$class_summary,
      sidecars = ld$sidecars,
      results_fold_level = results_tbl,
      summary_model_level = summary_tbl,
      predictions = out$predictions,
      models = out$models,
      training_domain_diversity_results = td_out$results,
      training_domain_diversity_summary_by_model = td_out$summary_by_model,
      training_domain_diversity_summary_by_site = td_out$summary_by_site,
      training_domain_combination_sampling_log = td_out$combination_sampling_log
    ),
    file.path(dirs$objects, "m0_transportability_tcga_brca_objects.rds")
  )

  message("Completed TCGA-BRCA transportability analysis. Output: ", output_dir)
  invisible(list(results = results_tbl, summary = summary_tbl, training_diversity = td_out))
}

args <- commandArgs(trailingOnly = TRUE)
arg_data_dir <- if (length(args) >= 1) args[[1]] else "data/legacy/TCGA-BRCA"
arg_output_dir <- if (length(args) >= 2) args[[2]] else file.path(getwd(), "outputs_m0_transportability_tcga_brca")
arg_seed <- if (length(args) >= 3) as.integer(args[[3]]) else 20260311L
arg_max_combinations <- if (length(args) >= 4) as.numeric(args[[4]]) else 40
if (!is.finite(arg_max_combinations) || is.na(arg_max_combinations) || arg_max_combinations <= 0) {
  arg_max_combinations <- Inf
}
arg_train_sizes <- if (length(args) >= 5) as.integer(strsplit(args[[5]], ",")[[1]]) else c(1L, 4L, 8L, 13L)
arg_train_sizes <- arg_train_sizes[!is.na(arg_train_sizes) & arg_train_sizes > 0]
if (length(arg_train_sizes) == 0) arg_train_sizes <- c(1L, 4L, 8L, 13L)
arg_max_features <- if (length(args) >= 6) as.integer(args[[6]]) else DEFAULT_TOP_FEATURES
if (is.na(arg_max_features) || arg_max_features < 30) arg_max_features <- DEFAULT_TOP_FEATURES
arg_min_source_n <- if (length(args) >= 7) as.integer(args[[7]]) else DEFAULT_MIN_SOURCE_N
if (is.na(arg_min_source_n) || arg_min_source_n < 5) arg_min_source_n <- DEFAULT_MIN_SOURCE_N
arg_min_class_n <- if (length(args) >= 8) as.integer(args[[8]]) else DEFAULT_MIN_CLASS_N
if (is.na(arg_min_class_n) || arg_min_class_n < 2) arg_min_class_n <- DEFAULT_MIN_CLASS_N
arg_max_sources <- if (length(args) >= 9) as.integer(args[[9]]) else DEFAULT_MAX_SOURCES
if (is.na(arg_max_sources) || arg_max_sources < 2) arg_max_sources <- DEFAULT_MAX_SOURCES

if (identical(environment(), globalenv())) {
  main(
    data_dir = arg_data_dir,
    output_dir = arg_output_dir,
    seed = arg_seed,
    train_sizes = arg_train_sizes,
    max_combinations_per_size = arg_max_combinations,
    max_features = arg_max_features,
    min_source_n = arg_min_source_n,
    min_class_n = arg_min_class_n,
    max_sources = arg_max_sources
  )
}
