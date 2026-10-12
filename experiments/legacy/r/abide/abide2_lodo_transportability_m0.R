#!/usr/bin/env Rscript

# Baseline M0 - ABIDE II transportability pipeline
# - LODO cross-site with strict anti-leakage preprocessing
# - ID vs OOD comparison
# - Exhaustive training-domain diversity experiment (same design as UCI: 4 sites)
# - Models: Logistic Regression, Random Forest, XGBoost

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(pROC)
  library(ranger)
  library(xgboost)
})

options(stringsAsFactors = FALSE)

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

safe_read_csv_robust <- function(path) {
  # Try UTF-8 first, then Latin-1/cp1252 fallback.
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
  if (length(y) < 10 || length(unique(y)) < 2) {
    return(list(intercept = NA_real_, slope = NA_real_))
  }
  p_clip <- pmin(pmax(p, 1e-6), 1 - 1e-6)
  lp <- qlogis(p_clip)

  fit_intercept <- tryCatch(
    suppressWarnings(glm(y ~ offset(lp), family = binomial())),
    error = function(e) NULL
  )
  fit_slope <- tryCatch(
    suppressWarnings(glm(y ~ lp, family = binomial())),
    error = function(e) NULL
  )

  intercept <- if (!is.null(fit_intercept)) unname(coef(fit_intercept)[1]) else NA_real_
  slope <- if (!is.null(fit_slope)) unname(coef(fit_slope)[2]) else NA_real_

  list(intercept = intercept, slope = slope)
}

evaluate_metrics <- function(truth, prob, threshold = 0.5) {
  df <- tibble::tibble(
    truth = factor(as.character(truth), levels = c("0", "1")),
    prob = as.numeric(prob)
  ) %>%
    dplyr::filter(!is.na(truth), !is.na(prob))

  issues <- character(0)
  if (nrow(df) == 0) {
    return(list(
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

  precision <- if ((tp + fp) == 0) NA_real_ else tp / (tp + fp)
  recall <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
  f1 <- if (is.na(precision) || is.na(recall) || (precision + recall) == 0) {
    NA_real_
  } else {
    2 * precision * recall / (precision + recall)
  }

  sensitivity <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
  specificity <- if ((tn + fp) == 0) NA_real_ else tn / (tn + fp)
  balacc <- if (is.na(sensitivity) || is.na(specificity)) NA_real_ else (sensitivity + specificity) / 2

  auc <- safe_auc(y_true_num = y, p_hat = df$prob)
  brier <- mean((df$prob - y)^2)
  cal <- safe_calibration(y_true_num = y, p_hat = df$prob)

  if (length(unique(y)) < 2) issues <- c(issues, "single_class_truth")
  if (all(pred_class == 0) || all(pred_class == 1)) issues <- c(issues, "single_class_prediction")

  list(
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
  n_sites <- dplyr::n_distinct(train_df$site)
  if (n_sites >= 2) {
    # For ABIDE this keeps compute feasible while preserving grouped split logic.
    v <- min(5, n_sites)
    rs <- rsample::group_vfold_cv(train_df, group = site, v = v)
    rs_type <- paste0("group_vfold_cv_site_v", v)
  } else {
    v <- max(2, min(5, nrow(train_df)))
    rs <- rsample::vfold_cv(train_df, v = v, strata = target_bin)
    rs_type <- "vfold_cv_fallback"
  }
  list(resamples = rs, resample_type = rs_type)
}

generate_all_train_site_combinations <- function(candidate_sites) {
  combos <- list()
  idx <- 1L
  for (k in seq_len(length(candidate_sites))) {
    cmb_k <- combn(candidate_sites, k, simplify = FALSE)
    for (cset in cmb_k) {
      combos[[idx]] <- cset
      idx <- idx + 1L
    }
  }
  combos
}

read_qap_summary <- function(path, prefix) {
  if (!file.exists(path)) return(tibble::tibble(site_id = character(), sub_id = numeric()))
  q <- safe_read_csv_robust(path)
  names(q) <- clean_names_local(names(q))
  if (!("site_id" %in% names(q)) || !("sub_id" %in% names(q))) {
    return(tibble::tibble(site_id = character(), sub_id = numeric()))
  }

  q <- q %>%
    dplyr::mutate(
      site_id = as.character(site_id),
      sub_id = suppressWarnings(as.numeric(sub_id))
    )

  numeric_cols <- names(q)[vapply(q, is.numeric, logical(1))]
  numeric_cols <- setdiff(numeric_cols, c("sub_id"))
  if (length(numeric_cols) == 0) return(tibble::tibble(site_id = character(), sub_id = numeric()))

  out <- q %>%
    dplyr::group_by(site_id, sub_id) %>%
    dplyr::summarise(
      dplyr::across(
        dplyr::all_of(numeric_cols),
        ~ {
          z <- mean(.x, na.rm = TRUE)
          if (is.nan(z)) NA_real_ else z
        }
      ),
      .groups = "drop"
    ) %>%
    dplyr::rename_with(~ paste0(prefix, "_", .x), .cols = -c(site_id, sub_id))
  out
}

find_phenotypic_file <- function(data_dir, fallback_phen_path = "data/legacy/ABIDEII_Composite_Phenotypic.csv") {
  candidates <- list.files(
    data_dir,
    pattern = "ABIDEII.*(Composite_Phenotypic|BNI_1).*\\.csv$",
    full.names = TRUE,
    ignore.case = TRUE
  )

  # Prefer composite file in target folder.
  if (length(candidates) > 0) {
    composite <- candidates[grepl("Composite_Phenotypic", basename(candidates), ignore.case = TRUE)]
    if (length(composite) > 0) return(composite[1])
    return(candidates[1])
  }

  if (file.exists(fallback_phen_path)) {
    warning("Composite phenotypic file not found in data_dir; using fallback: ", fallback_phen_path)
    return(fallback_phen_path)
  }

  stop("No ABIDE II phenotypic CSV found in data_dir and no fallback available.")
}

# -------------------------------------------------------------------------
# Required modular functions
# -------------------------------------------------------------------------
load_data <- function(
  data_dir,
  n_sites_select = 14L,
  min_per_class = 15L,
  fallback_phen_path = "data/legacy/ABIDEII_Composite_Phenotypic.csv"
) {
  if (!dir.exists(data_dir)) stop("Data directory does not exist: ", data_dir)

  phen_path <- find_phenotypic_file(data_dir = data_dir, fallback_phen_path = fallback_phen_path)
  phen <- safe_read_csv_robust(phen_path)
  names(phen) <- clean_names_local(names(phen))

  required_cols <- c("site_id", "sub_id", "dx_group")
  miss_req <- setdiff(required_cols, names(phen))
  if (length(miss_req) > 0) {
    stop("Phenotypic file is missing required columns: ", paste(miss_req, collapse = ", "))
  }

  phen <- phen %>%
    dplyr::mutate(
      site_id = as.character(site_id),
      sub_id = suppressWarnings(as.numeric(sub_id)),
      dx_group = suppressWarnings(as.numeric(dx_group))
    ) %>%
    dplyr::filter(!is.na(site_id), !is.na(sub_id), dx_group %in% c(1, 2)) %>%
    dplyr::mutate(
      target_bin = factor(ifelse(dx_group == 1, "1", "0"), levels = c("0", "1")),
      diagnosis_label = ifelse(dx_group == 1, "ASD", "Control")
    )

  # If local phenotypic file is single-site (e.g., ABIDEII-BNI_1.csv), switch to composite fallback.
  if (dplyr::n_distinct(phen$site_id) < 2 && file.exists(fallback_phen_path) && normalizePath(phen_path) != normalizePath(fallback_phen_path)) {
    warning(
      "Phenotypic file in data_dir has <2 sites; switching to composite fallback: ",
      fallback_phen_path
    )
    phen_path <- fallback_phen_path
    phen <- safe_read_csv_robust(phen_path)
    names(phen) <- clean_names_local(names(phen))
    miss_req_fb <- setdiff(required_cols, names(phen))
    if (length(miss_req_fb) > 0) {
      stop("Fallback phenotypic file is missing required columns: ", paste(miss_req_fb, collapse = ", "))
    }
    phen <- phen %>%
      dplyr::mutate(
        site_id = as.character(site_id),
        sub_id = suppressWarnings(as.numeric(sub_id)),
        dx_group = suppressWarnings(as.numeric(dx_group))
      ) %>%
      dplyr::filter(!is.na(site_id), !is.na(sub_id), dx_group %in% c(1, 2)) %>%
      dplyr::mutate(
        target_bin = factor(ifelse(dx_group == 1, "1", "0"), levels = c("0", "1")),
        diagnosis_label = ifelse(dx_group == 1, "ASD", "Control")
      )
  }

  qap_dir <- file.path(data_dir, "ABIDEII_MRI_Quality_Metrics")
  anat <- read_qap_summary(file.path(qap_dir, "anat_qap.csv"), "anat")
  dti <- read_qap_summary(file.path(qap_dir, "dti_qap.csv"), "dti")
  func <- read_qap_summary(file.path(qap_dir, "functional_qap.csv"), "func")

  df <- phen %>%
    dplyr::left_join(anat, by = c("site_id", "sub_id")) %>%
    dplyr::left_join(dti, by = c("site_id", "sub_id")) %>%
    dplyr::left_join(func, by = c("site_id", "sub_id")) %>%
    dplyr::mutate(site = site_id, row_id = dplyr::row_number())

  # Keep sites with both classes; select top sites by size (default: 14 sources).
  site_summary_all <- df %>%
    dplyr::group_by(site) %>%
    dplyr::summarise(
      n_samples = dplyr::n(),
      n_asd = sum(target_bin == "1"),
      n_ctrl = sum(target_bin == "0"),
      .groups = "drop"
    ) %>%
    dplyr::arrange(dplyr::desc(n_samples))

  eligible <- site_summary_all %>%
    dplyr::filter(n_asd >= min_per_class, n_ctrl >= min_per_class)

  if (nrow(eligible) < n_sites_select) {
    warning(
      "Only ", nrow(eligible),
      " sites pass min_per_class=", min_per_class,
      "; selecting top sites with both classes available."
    )
    eligible <- site_summary_all %>% dplyr::filter(n_asd > 0, n_ctrl > 0)
  }
  if (nrow(eligible) < 2) stop("Not enough multi-class sites for cross-site analysis.")

  selected_sites <- eligible %>%
    dplyr::arrange(dplyr::desc(n_samples)) %>%
    dplyr::slice_head(n = min(n_sites_select, nrow(.))) %>%
    dplyr::pull(site)

  df <- df %>% dplyr::filter(site %in% selected_sites)

  # Fixed feature set (predefined, no leakage from fold-specific feature selection).
  candidate_base <- c("age_at_scan", "sex", "fiq", "viq", "piq", "handedness_scores")
  qap_feats <- names(df)[grepl("^(anat|dti|func)_", names(df))]
  keep_feats <- intersect(c(candidate_base, qap_feats), names(df))

  if (!("sex" %in% keep_feats) && ("sex" %in% names(df))) keep_feats <- c("sex", keep_feats)
  if (length(keep_feats) < 5) {
    stop("Too few predictors available after ABIDE merge. Check source files.")
  }

  modeling_df <- df %>%
    dplyr::select(
      dplyr::all_of(c("site", "sub_id", "target_bin", "diagnosis_label", "row_id", keep_feats))
    ) %>%
    dplyr::mutate(
      sex = if ("sex" %in% names(.)) factor(as.character(sex)) else NULL
    )

  num_cols <- names(modeling_df)[vapply(modeling_df, is.numeric, logical(1))]
  num_cols <- setdiff(num_cols, c("sub_id", "row_id"))
  if (length(num_cols) > 0) {
    modeling_df <- modeling_df %>%
      dplyr::mutate(dplyr::across(dplyr::all_of(num_cols), ~ suppressWarnings(as.numeric(.x))))
  }

  site_summary <- modeling_df %>%
    dplyr::count(site, name = "n_samples") %>%
    dplyr::arrange(dplyr::desc(n_samples))

  class_summary <- modeling_df %>%
    dplyr::count(site, target_bin, name = "n") %>%
    dplyr::arrange(site, target_bin)

  list(
    data = modeling_df,
    site_summary = site_summary,
    class_summary = class_summary,
    files_used = tibble::tibble(
      phenotypic_file = phen_path,
      anat_qap = file.path(qap_dir, "anat_qap.csv"),
      dti_qap = file.path(qap_dir, "dti_qap.csv"),
      functional_qap = file.path(qap_dir, "functional_qap.csv")
    ),
    selected_sites = tibble::tibble(site = selected_sites)
  )
}

preprocess_fold <- function(train_df, test_df = NULL) {
  model_train <- train_df %>%
    dplyr::select(-dplyr::any_of(c("site", "sub_id", "diagnosis_label", "row_id")))

  rec <- recipes::recipe(target_bin ~ ., data = model_train) %>%
    recipes::step_novel(recipes::all_nominal_predictors()) %>%
    recipes::step_impute_median(recipes::all_numeric_predictors()) %>%
    recipes::step_impute_mode(recipes::all_nominal_predictors()) %>%
    recipes::step_normalize(recipes::all_numeric_predictors()) %>%
    recipes::step_dummy(recipes::all_nominal_predictors(), one_hot = TRUE) %>%
    recipes::step_zv(recipes::all_predictors())

  if (is.null(test_df)) {
    return(list(recipe = rec, train_processed = NULL, test_processed = NULL))
  }

  model_test <- test_df %>%
    dplyr::select(-dplyr::any_of(c("site", "sub_id", "diagnosis_label", "row_id")))

  rec_prep <- recipes::prep(rec, training = model_train, retain = TRUE)
  train_baked <- recipes::bake(rec_prep, new_data = NULL)
  test_baked <- recipes::bake(rec_prep, new_data = model_test)
  list(recipe = rec, train_processed = train_baked, test_processed = test_baked)
}

train_logistic <- function() {
  parsnip::logistic_reg(mode = "classification") %>%
    parsnip::set_engine("glm")
}

train_rf <- function(seed = 20260311L) {
  parsnip::rand_forest(mode = "classification", trees = 600, min_n = 5) %>%
    parsnip::set_engine("ranger", probability = TRUE, seed = seed, num.threads = 1)
}

train_xgb <- function(seed = 20260311L) {
  parsnip::boost_tree(
    mode = "classification",
    trees = 400,
    tree_depth = 4,
    learn_rate = 0.05,
    min_n = 5,
    sample_size = 0.8,
    loss_reduction = 0
  ) %>%
    parsnip::set_engine(
      "xgboost",
      objective = "binary:logistic",
      eval_metric = "auc",
      nthread = 1
    )
}

run_lodo_pipeline <- function(df, seed = 20260311L, save_models = TRUE) {
  sites <- sort(unique(df$site))
  if (length(sites) < 2) stop("LODO requires at least 2 sites/domains.")

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
    train_df <- df %>% dplyr::filter(site != holdout_site)
    test_df <- df %>% dplyr::filter(site == holdout_site)

    rs_obj <- choose_internal_resamples(train_df = train_df, seed = seed + i)
    inner_resamples <- rs_obj$resamples
    inner_scheme <- rs_obj$resample_type

    for (m in seq_along(model_specs)) {
      model_name <- names(model_specs)[m]
      spec <- model_specs[[m]]

      prep_obj <- preprocess_fold(train_df)
      wf <- workflows::workflow() %>%
        workflows::add_recipe(prep_obj$recipe) %>%
        workflows::add_model(spec)

      set.seed(seed + i * 100 + m)
      id_rs <- tryCatch(
        tune::fit_resamples(
          wf,
          resamples = inner_resamples,
          control = tune::control_resamples(save_pred = TRUE, verbose = FALSE, allow_par = FALSE)
        ),
        error = function(e) e
      )

      id_metrics <- list(
        auc = NA_real_, f1 = NA_real_, balacc = NA_real_, brier = NA_real_,
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
        auc = NA_real_, f1 = NA_real_, balacc = NA_real_, brier = NA_real_,
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
        inner_validation = inner_scheme,
        n_train = nrow(train_df),
        n_test = nrow(test_df),
        pos_rate_train = mean(train_df$target_bin == "1"),
        pos_rate_test = mean(test_df$target_bin == "1"),
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

summarize_results <- function(results_tbl) {
  results_tbl %>%
    dplyr::group_by(model) %>%
    dplyr::summarise(
      folds = dplyr::n(),
      auc_ood_mean = mean(auc_ood, na.rm = TRUE),
      auc_ood_sd = sd(auc_ood, na.rm = TRUE),
      f1_ood_mean = mean(f1_ood, na.rm = TRUE),
      f1_ood_sd = sd(f1_ood, na.rm = TRUE),
      balacc_ood_mean = mean(balacc_ood, na.rm = TRUE),
      balacc_ood_sd = sd(balacc_ood, na.rm = TRUE),
      brier_ood_mean = mean(brier_ood, na.rm = TRUE),
      brier_ood_sd = sd(brier_ood, na.rm = TRUE),
      auc_gap_mean = mean(auc_gap, na.rm = TRUE),
      f1_gap_mean = mean(f1_gap, na.rm = TRUE),
      balacc_gap_mean = mean(balacc_gap, na.rm = TRUE),
      brier_gap_mean = mean(brier_gap, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      rank_auc = dplyr::min_rank(dplyr::desc(auc_ood_mean)),
      rank_f1 = dplyr::min_rank(dplyr::desc(f1_ood_mean)),
      rank_balacc = dplyr::min_rank(dplyr::desc(balacc_ood_mean)),
      rank_brier = dplyr::min_rank(brier_ood_mean),
      mean_rank = (rank_auc + rank_f1 + rank_balacc + rank_brier) / 4,
      final_rank = dplyr::min_rank(mean_rank)
    ) %>%
    dplyr::arrange(final_rank, dplyr::desc(auc_ood_mean))
}

plot_results <- function(results_tbl, predictions_tbl, plots_dir) {
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
  base_theme <- ggplot2::theme_bw(base_size = 12) +
    ggplot2::theme(
      legend.position = "bottom",
      panel.grid.minor = ggplot2::element_blank(),
      strip.background = ggplot2::element_rect(fill = "grey95", color = "grey80")
    )

  ood_long <- results_tbl %>%
    dplyr::select(holdout_site, model, auc_ood, f1_ood, balacc_ood, brier_ood) %>%
    tidyr::pivot_longer(
      cols = c(auc_ood, f1_ood, balacc_ood, brier_ood),
      names_to = "metric",
      values_to = "value"
    )
  p_ood <- ggplot2::ggplot(ood_long, ggplot2::aes(x = holdout_site, y = value, color = model, group = model)) +
    ggplot2::geom_point(size = 2.8) +
    ggplot2::geom_line(alpha = 0.75) +
    ggplot2::facet_wrap(~ metric, scales = "free_y", ncol = 2) +
    ggplot2::labs(title = "OOD performance by holdout site and model", x = "Holdout site", y = "Metric value") +
    base_theme

  gap_long <- results_tbl %>%
    dplyr::select(holdout_site, model, auc_gap, f1_gap, balacc_gap, brier_gap) %>%
    tidyr::pivot_longer(cols = c(auc_gap, f1_gap, balacc_gap, brier_gap), names_to = "metric", values_to = "gap")
  p_gap <- ggplot2::ggplot(gap_long, ggplot2::aes(x = holdout_site, y = gap, fill = model)) +
    ggplot2::geom_col(position = ggplot2::position_dodge(width = 0.8), width = 0.72) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "grey35") +
    ggplot2::facet_wrap(~ metric, scales = "free_y", ncol = 2) +
    ggplot2::labs(title = "ID - OOD deterioration gap by holdout site", x = "Holdout site", y = "Gap (ID - OOD)") +
    base_theme

  heatmap_df <- results_tbl %>%
    dplyr::select(holdout_site, model, auc_ood, f1_ood, balacc_ood) %>%
    tidyr::pivot_longer(cols = c(auc_ood, f1_ood, balacc_ood), names_to = "metric", values_to = "value")
  p_heat <- ggplot2::ggplot(heatmap_df, ggplot2::aes(x = model, y = holdout_site, fill = value)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.6) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.3f", value)), size = 3) +
    ggplot2::facet_wrap(~ metric, ncol = 3) +
    ggplot2::scale_fill_viridis_c(option = "C", na.value = "grey85") +
    ggplot2::labs(title = "Heatmap: site x model OOD performance", x = "Model", y = "Holdout site", fill = "Value") +
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
  ggplot2::ggsave(file.path(plots_dir, "heatmap_site_model_ood.png"), p_heat, width = 12, height = 5.8, dpi = 320)
  ggplot2::ggsave(file.path(plots_dir, "calibration_reliability_id_ood.png"), p_cal, width = 13, height = 8, dpi = 320)
}

run_training_domain_diversity_experiment <- function(
  df,
  seed = 20260311L,
  train_sizes = c(1L, 4L, 8L, 13L),
  max_combinations_per_size = Inf
) {
  all_sites <- sort(unique(df$site))
  if (length(all_sites) < 2) stop("Training-domain diversity experiment requires at least 2 sites.")

  model_specs <- list(
    logistic = train_logistic(),
    random_forest = train_rf(seed = seed),
    xgboost = train_xgb(seed = seed)
  )

  rows_out <- list()
  row_idx <- 1L

  for (i in seq_along(all_sites)) {
    test_site <- all_sites[i]
    remaining_sites <- setdiff(all_sites, test_site)
    valid_sizes <- sort(unique(train_sizes[train_sizes > 0 & train_sizes <= length(remaining_sites)]))
    if (length(valid_sizes) == 0) {
      warning("No valid train_sizes for test site: ", test_site)
      next
    }

    message(
      "Diversity experiment - test site ",
      test_site,
      " | valid train sizes: ",
      paste(valid_sizes, collapse = ", ")
    )

    test_df <- df %>% dplyr::filter(site == test_site)
    combo_counter <- 1L

    for (k in valid_sizes) {
      train_site_sets <- combn(remaining_sites, k, simplify = FALSE)
      if (is.finite(max_combinations_per_size) && length(train_site_sets) > max_combinations_per_size) {
        set.seed(seed + i * 100 + k)
        idx <- sort(sample.int(length(train_site_sets), size = max_combinations_per_size, replace = FALSE))
        train_site_sets <- train_site_sets[idx]
      }

      message("  Train size ", k, " -> combinations: ", length(train_site_sets))

      for (j in seq_along(train_site_sets)) {
        train_sites <- train_site_sets[[j]]
        train_df <- df %>% dplyr::filter(site %in% train_sites)
        train_sites_label <- paste(train_sites, collapse = " + ")
        n_train_sites <- length(train_sites)

        for (m in seq_along(model_specs)) {
          model_name <- names(model_specs)[m]
          spec <- model_specs[[m]]
          rec_obj <- preprocess_fold(train_df)
          wf <- workflows::workflow() %>%
            workflows::add_recipe(rec_obj$recipe) %>%
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
        if (combo_counter %% 200 == 0) {
          message("    Processed ", combo_counter, " combinations for test site ", test_site)
        }
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
    ) %>%
    dplyr::arrange(model, n_train_sites)

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
    ) %>%
    dplyr::arrange(test_site, model, n_train_sites)

  list(results = results_tbl, summary_by_model = summary_by_model, summary_by_site = summary_by_site)
}

plot_training_domain_diversity_experiment <- function(summary_by_model, summary_by_site, plots_dir) {
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
    ) +
    theme_pub

  p_site_heatmap <- ggplot2::ggplot(summary_by_site, ggplot2::aes(x = n_train_sites, y = test_site, fill = mean_accuracy)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.7) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.3f", mean_accuracy)), size = 3.1) +
    ggplot2::facet_wrap(~ model, ncol = 3) +
    ggplot2::scale_fill_viridis_c(option = "C", na.value = "grey85") +
    ggplot2::labs(
      title = "Transportability improvement by held-out site as training domains increase",
      x = "Number of training sites",
      y = "Held-out test site",
      fill = "Mean accuracy"
    ) +
    theme_pub

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
    ) +
    theme_pub

  ggplot2::ggsave(file.path(plots_dir, "transportability_training_domains_heatmap.png"), p_main_heatmap, width = 10, height = 5.2, dpi = 320)
  ggplot2::ggsave(file.path(plots_dir, "transportability_training_domains_by_site_heatmap.png"), p_site_heatmap, width = 12, height = 5.8, dpi = 320)
  ggplot2::ggsave(file.path(plots_dir, "transportability_training_domains_lineplot.png"), p_line, width = 10, height = 5.2, dpi = 320)
}

plot_training_domain_combinations_heatmap <- function(results_tbl, plots_dir) {
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

  combo_map <- results_tbl %>%
    dplyr::distinct(test_site, n_train_sites, train_sites) %>%
    dplyr::arrange(test_site, n_train_sites, train_sites) %>%
    dplyr::group_by(test_site) %>%
    dplyr::mutate(
      combo_rank = dplyr::row_number(),
      combo_label = paste0("Train ", n_train_sites, " site(s) | ", train_sites),
      combo_key = paste0(test_site, " :: ", sprintf("%02d", combo_rank), " :: ", combo_label)
    ) %>%
    dplyr::ungroup()

  plot_df <- results_tbl %>%
    dplyr::left_join(combo_map %>% dplyr::select(test_site, n_train_sites, train_sites, combo_key), by = c("test_site", "n_train_sites", "train_sites")) %>%
    dplyr::mutate(model = factor(model, levels = sort(unique(model))), test_site = factor(test_site, levels = sort(unique(test_site))))

  p_combo <- ggplot2::ggplot(plot_df, ggplot2::aes(x = model, y = combo_key, fill = accuracy)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.65) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.3f", accuracy)), size = 2.9) +
    ggplot2::facet_grid(test_site ~ ., scales = "free_y", space = "free_y") +
    ggplot2::scale_y_discrete(labels = function(x) sub("^.* :: [0-9]+ :: ", "", x)) +
    ggplot2::scale_fill_viridis_c(option = "C", na.value = "grey85") +
    ggplot2::labs(
      title = "Exhaustive training-site combinations for each held-out test site",
      x = "Model",
      y = "Training combination",
      fill = "Accuracy"
    ) +
    ggplot2::theme_bw(base_size = 11.5) +
    ggplot2::theme(
      panel.grid = ggplot2::element_blank(),
      strip.background = ggplot2::element_rect(fill = "grey95", color = "grey80"),
      legend.position = "bottom"
    )

  ggplot2::ggsave(
    file.path(plots_dir, "transportability_training_domains_all_combinations_heatmap.png"),
    p_combo,
    width = 13,
    height = 10.5,
    dpi = 320
  )
}

main <- function(
  data_dir = "data/legacy/ABIDE II",
  output_dir = file.path(getwd(), "outputs_m0_transportability_abide2"),
  seed = 20260311L,
  train_sizes = c(1L, 4L, 8L, 13L),
  max_combinations_per_size = Inf
) {
  set.seed(seed)
  dirs <- list(
    root = output_dir,
    tables = file.path(output_dir, "tables"),
    plots = file.path(output_dir, "plots"),
    objects = file.path(output_dir, "objects")
  )
  purrr::walk(dirs, ~ dir.create(.x, recursive = TRUE, showWarnings = FALSE))

  message("Loading ABIDE II data from: ", data_dir)
  ld <- load_data(data_dir = data_dir)
  df <- ld$data

  readr::write_csv(ld$files_used, file.path(dirs$tables, "files_used.csv"))
  readr::write_csv(ld$site_summary, file.path(dirs$tables, "site_summary.csv"))
  readr::write_csv(ld$class_summary, file.path(dirs$tables, "class_summary.csv"))
  readr::write_csv(ld$selected_sites, file.path(dirs$tables, "selected_sites.csv"))

  message("Running LODO pipeline across ", dplyr::n_distinct(df$site), " selected ABIDE II sites...")
  out <- run_lodo_pipeline(df = df, seed = seed, save_models = TRUE)

  results_tbl <- out$results %>% dplyr::arrange(holdout_site, model)
  summary_tbl <- summarize_results(results_tbl)

  readr::write_csv(results_tbl, file.path(dirs$tables, "lodo_fold_level_results.csv"))
  readr::write_csv(summary_tbl, file.path(dirs$tables, "lodo_model_summary.csv"))
  readr::write_csv(out$predictions, file.path(dirs$tables, "id_ood_predictions.csv"))
  plot_results(results_tbl, out$predictions, plots_dir = dirs$plots)

  message(
    "Running exhaustive training-domain diversity experiment (train sizes: ",
    paste(train_sizes, collapse = ","),
    ")..."
  )
  td_out <- run_training_domain_diversity_experiment(
    df = df,
    seed = seed,
    train_sizes = as.integer(train_sizes),
    max_combinations_per_size = max_combinations_per_size
  )
  readr::write_csv(td_out$results, file.path(dirs$tables, "training_domain_experiment_results.csv"))
  readr::write_csv(td_out$summary_by_model, file.path(dirs$tables, "training_domain_experiment_summary_by_model.csv"))
  readr::write_csv(td_out$summary_by_site, file.path(dirs$tables, "training_domain_experiment_summary_by_site.csv"))
  plot_training_domain_diversity_experiment(td_out$summary_by_model, td_out$summary_by_site, plots_dir = dirs$plots)
  plot_training_domain_combinations_heatmap(td_out$results, plots_dir = dirs$plots)

  saveRDS(
    list(
      config = list(
        data_dir = data_dir,
        output_dir = output_dir,
        seed = seed,
        train_sizes = as.integer(train_sizes),
        max_combinations_per_size = max_combinations_per_size,
        timestamp = as.character(Sys.time())
      ),
      selected_sites = ld$selected_sites,
      data = df,
      site_summary = ld$site_summary,
      class_summary = ld$class_summary,
      results_fold_level = results_tbl,
      summary_model_level = summary_tbl,
      predictions = out$predictions,
      models = out$models,
      training_domain_diversity_results = td_out$results,
      training_domain_diversity_summary_by_model = td_out$summary_by_model,
      training_domain_diversity_summary_by_site = td_out$summary_by_site
    ),
    file.path(dirs$objects, "m0_transportability_abide2_objects.rds")
  )

  message("Completed ABIDE II transportability analysis. Output: ", output_dir)
  invisible(list(results = results_tbl, summary = summary_tbl, training_diversity = td_out))
}

args <- commandArgs(trailingOnly = TRUE)
arg_data_dir <- if (length(args) >= 1) args[[1]] else "data/legacy/ABIDE II"
arg_output_dir <- if (length(args) >= 2) args[[2]] else file.path(getwd(), "outputs_m0_transportability_abide2")
arg_seed <- if (length(args) >= 3) as.integer(args[[3]]) else 20260311L
arg_max_combinations <- if (length(args) >= 4) as.numeric(args[[4]]) else Inf
if (!is.finite(arg_max_combinations) || is.na(arg_max_combinations) || arg_max_combinations <= 0) {
  arg_max_combinations <- Inf
}
arg_train_sizes <- if (length(args) >= 5) as.integer(strsplit(args[[5]], ",")[[1]]) else c(1L, 4L, 8L, 13L)
arg_train_sizes <- arg_train_sizes[!is.na(arg_train_sizes) & arg_train_sizes > 0]
if (length(arg_train_sizes) == 0) arg_train_sizes <- c(1L, 4L, 8L, 13L)

if (identical(environment(), globalenv())) {
  main(
    data_dir = arg_data_dir,
    output_dir = arg_output_dir,
    seed = arg_seed,
    train_sizes = arg_train_sizes,
    max_combinations_per_size = arg_max_combinations
  )
}
