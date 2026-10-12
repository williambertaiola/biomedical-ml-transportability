"""Transportability experiment: tcga m0 uniform."""

from __future__ import annotations

# Locate the repository without depending on the current working directory.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys

_REPOSITORY_ROOT = next(
    parent
    for parent in _RepositoryPath(__file__).resolve().parents
    if (parent / ".thesis-root").is_file()
)
_repository_sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))
from thesis_tools.paths import configured_path as _configured_path


import json
import logging
import math
import os
import re
import shutil
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from html import escape
from itertools import combinations, product
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")


@dataclass
class ExperimentConfig:
    dataset_name: str
    input_path: str
    output_dir: str
    target_col: str
    domain_col: str
    id_cols: list[str] = field(default_factory=list)
    extra_exclude_cols: list[str] = field(default_factory=list)
    force_categorical_cols: list[str] = field(default_factory=list)
    force_numeric_cols: list[str] = field(default_factory=list)
    positive_label: Any | None = None
    random_seed: int = 42
    n_jobs: int = -1
    enable_xgboost: bool = True
    enable_ece: bool = True
    enable_calibration_plots: bool = True
    calibration_bins: int = 10
    min_domain_size: int = 20
    save_pdf_plots: bool = True
    low_cardinality_as_categorical: bool = True
    max_low_cardinality_unique: int = 10
    auto_exclude_id_like: bool = True
    feature_selection_method: str = "none"
    feature_selection_top_k: int = 1000
    variance_threshold: float = 0.0
    pca_n_components: int | None = None
    enable_pairwise_site_transfer: bool = True
    within_domain_cv_splits: int = 5

    @staticmethod
    def from_json(path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return ExperimentConfig(**payload)


@dataclass
class ColumnSpec:
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    dropped_cols: list[dict[str, str]]
    warnings: list[str]


def ensure_directory(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def init_logger(log_path: Path) -> logging.Logger:
    ensure_directory(log_path.parent)
    logger = logging.getLogger(f"baseline_m0::{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def read_table(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(input_path, sep=None, engine="python")
    raise ValueError(f"Unsupported input format: {input_path}")


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return cleaned.lower() or "value"


def is_integer_like(series: pd.Series) -> bool:
    non_missing = series.dropna()
    if non_missing.empty:
        return False
    if not pd.api.types.is_numeric_dtype(non_missing):
        return False
    return np.all(
        np.isclose(
            non_missing.to_numpy(dtype=float),
            np.round(non_missing.to_numpy(dtype=float)),
        )
    )


def load_dataset(
    config: ExperimentConfig, logger: logging.Logger
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    df = read_table(config.input_path)
    warnings_out: list[str] = []

    required = {config.target_col, config.domain_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[config.domain_col] = df[config.domain_col].astype("string")
    target_series = df[config.target_col]

    non_missing_classes = pd.Series(target_series.dropna().unique()).tolist()
    if len(non_missing_classes) != 2:
        raise ValueError(
            f"Baseline M0 currently expects a binary target. Found {len(non_missing_classes)} classes in {config.target_col}."
        )

    target_mapping: dict[str, Any] = {}
    if set(non_missing_classes) == {0, 1}:
        df[config.target_col] = target_series.astype(int)
        target_mapping = {"negative": 0, "positive": 1}
    else:
        if config.positive_label is not None:
            positive_label = config.positive_label
            negative_candidates = [
                value for value in non_missing_classes if value != positive_label
            ]
            if len(negative_candidates) != 1:
                raise ValueError(
                    "positive_label does not identify a unique positive class."
                )
            negative_label = negative_candidates[0]
        else:
            ordered = sorted(non_missing_classes, key=lambda value: str(value))
            negative_label, positive_label = ordered[0], ordered[1]
            warnings_out.append(
                f"positive_label_not_provided_auto_mapping_used:{negative_label}->0,{positive_label}->1"
            )
        df[config.target_col] = target_series.map(
            {negative_label: 0, positive_label: 1}
        ).astype(int)
        target_mapping = {"negative": negative_label, "positive": positive_label}

    logger.info(
        "Loaded dataset %s with shape %s and %s domains",
        config.dataset_name,
        df.shape,
        df[config.domain_col].nunique(dropna=True),
    )
    return df, target_mapping, warnings_out


def prepare_columns(train_df: pd.DataFrame, config: ExperimentConfig) -> ColumnSpec:
    dropped_cols: list[dict[str, str]] = []
    warnings_out: list[str] = []
    explicit_exclude = {
        config.target_col,
        config.domain_col,
        *config.id_cols,
        *config.extra_exclude_cols,
    }

    auto_excluded: set[str] = set()
    if config.auto_exclude_id_like:
        id_like_patterns = [
            r"(^|_)id$",
            r"^id_",
            r"^sample(_|$)",
            r"^subject(_|$)",
            r"^patient(_|$)",
            r"^case(_|$)",
            r"^barcode(_|$)",
        ]
        for column in train_df.columns:
            if column in explicit_exclude:
                continue
            if any(
                re.search(pattern, column, flags=re.IGNORECASE)
                for pattern in id_like_patterns
            ):
                auto_excluded.add(column)

    candidate_cols = [
        column
        for column in train_df.columns
        if column not in explicit_exclude and column not in auto_excluded
    ]
    feature_cols: list[str] = []
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for column in candidate_cols:
        series = train_df[column]
        if series.isna().all():
            dropped_cols.append(
                {"column_name": column, "reason": "all_missing_in_training"}
            )
            continue
        non_missing_unique = series.dropna().nunique()
        if non_missing_unique <= 1:
            dropped_cols.append(
                {"column_name": column, "reason": "constant_in_training"}
            )
            continue

        feature_cols.append(column)
        if column in config.force_categorical_cols:
            categorical_cols.append(column)
            continue
        if column in config.force_numeric_cols:
            numeric_cols.append(column)
            continue

        if pd.api.types.is_bool_dtype(series):
            categorical_cols.append(column)
            continue
        if pd.api.types.is_numeric_dtype(series):
            if (
                config.low_cardinality_as_categorical
                and is_integer_like(series)
                and non_missing_unique <= config.max_low_cardinality_unique
            ):
                categorical_cols.append(column)
            else:
                numeric_cols.append(column)
            continue
        categorical_cols.append(column)

    for column in sorted(auto_excluded):
        dropped_cols.append({"column_name": column, "reason": "auto_excluded_id_like"})

    if not feature_cols:
        warnings_out.append("no_usable_features_after_training_only_column_checks")
    if not numeric_cols:
        warnings_out.append("no_numeric_features_available")
    if not categorical_cols:
        warnings_out.append("no_categorical_features_available")

    return ColumnSpec(
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        dropped_cols=dropped_cols,
        warnings=warnings_out,
    )


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    spec: ColumnSpec, *, scale_numeric: bool = True
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if spec.numeric_cols:
        numeric_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median"))
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(
            (
                "numeric",
                Pipeline(steps=numeric_steps),
                spec.numeric_cols,
            )
        )
    if spec.categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", make_one_hot_encoder()),
                    ]
                ),
                spec.categorical_cols,
            )
        )
    return ColumnTransformer(
        transformers=transformers, remainder="drop", sparse_threshold=0.0
    )


def build_selector(config: ExperimentConfig):
    method = config.feature_selection_method.lower()
    if method in {"none", "passthrough"}:
        return "passthrough"
    if method == "variance":
        return VarianceThreshold(threshold=config.variance_threshold)
    if method == "selectkbest":
        return SelectKBest(score_func=f_classif, k=config.feature_selection_top_k)
    if method == "pca":
        if config.pca_n_components is None:
            raise ValueError(
                "pca_n_components must be set when feature_selection_method='pca'."
            )
        return PCA(n_components=config.pca_n_components, svd_solver="auto")
    raise ValueError(
        f"Unknown feature_selection_method: {config.feature_selection_method}"
    )


def build_models(config: ExperimentConfig, logger: logging.Logger) -> dict[str, Any]:
    models: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            solver="saga",
            max_iter=2000,
            random_state=config.random_seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=75,
            random_state=config.random_seed,
            n_jobs=config.n_jobs,
        ),
    }

    if config.enable_xgboost:
        try:
            from xgboost import XGBClassifier

            models["XGBoost"] = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=config.random_seed,
                n_jobs=config.n_jobs,
            )
            logger.info("Using XGBoost baseline.")
            return models
        except Exception as exc:
            logger.warning(
                "XGBoost not available, falling back to HistGradientBoostingClassifier: %s",
                exc,
            )

    try:
        models["HistGradientBoosting"] = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            max_iter=75,
            random_state=config.random_seed,
        )
    except Exception:
        models["GradientBoosting"] = GradientBoostingClassifier(
            random_state=config.random_seed,
        )
    return models


def build_pipeline(
    spec: ColumnSpec, estimator: Any, config: ExperimentConfig
) -> Pipeline:
    is_tcga_numeric_only = (
        "TCGA" in config.dataset_name.upper() and not spec.categorical_cols
    )
    scale_before_selection = not is_tcga_numeric_only
    steps: list[tuple[str, Any]] = [
        (
            "preprocessor",
            build_preprocessor(spec, scale_numeric=scale_before_selection),
        ),
        ("selector", build_selector(config)),
    ]
    if is_tcga_numeric_only and isinstance(estimator, LogisticRegression):
        steps.append(("post_selector_scaler", StandardScaler()))
    steps.append(("model", clone(estimator)))
    return Pipeline(steps=steps)


def fit_with_warnings(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[Pipeline, list[str]]:
    fit_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        pipeline.fit(X_train, y_train)
    for item in captured:
        message = str(item.message)
        if "Could not find the number of physical cores" in message:
            continue
        fit_warnings.append(f"{item.category.__name__}:{message}")
    return pipeline, fit_warnings


def get_transformed_feature_count(
    fitted_pipeline: Pipeline, X_fit_reference: pd.DataFrame
) -> int | float:
    try:
        transformed = fitted_pipeline[:-1].transform(
            X_fit_reference.iloc[: min(5, len(X_fit_reference))]
        )
        return int(transformed.shape[1])
    except Exception:
        return math.nan


def safe_predict_proba(fitted_pipeline: Pipeline, X_data: pd.DataFrame) -> np.ndarray:
    model = fitted_pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return fitted_pipeline.predict_proba(X_data)[:, 1]
    if hasattr(model, "decision_function"):
        scores = fitted_pipeline.decision_function(X_data)
        return 1.0 / (1.0 + np.exp(-scores))
    raise AttributeError("Model does not expose predict_proba or decision_function.")


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error:
    sum_b (n_b / N) * |acc(b) - conf(b)| over equally spaced probability bins.
    """
    if len(y_true) == 0:
        return math.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    ece = 0.0
    n_total = len(y_true)
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        accuracy = float(np.mean(y_true[mask]))
        confidence = float(np.mean(y_prob[mask]))
        ece += (np.sum(mask) / n_total) * abs(accuracy - confidence)
    return float(ece)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    config: ExperimentConfig,
) -> tuple[dict[str, float], list[str]]:
    metrics: dict[str, float] = {}
    metric_warnings: list[str] = []

    if len(y_true) == 0:
        metric_warnings.append("empty_evaluation_set")
        for metric_name in [
            "auc",
            "accuracy",
            "f1",
            "balacc",
            "precision",
            "recall",
            "brier",
            "ece",
        ]:
            metrics[metric_name] = math.nan
        return metrics, metric_warnings

    y_pred = (y_prob >= 0.5).astype(int)
    unique_classes = np.unique(y_true)

    try:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    except Exception as exc:
        metrics["accuracy"] = math.nan
        metric_warnings.append(f"accuracy_failed:{exc}")

    if len(unique_classes) < 2:
        metric_warnings.append("single_class_in_evaluation_labels")
        metrics["auc"] = math.nan
        metrics["f1"] = math.nan
        metrics["balacc"] = math.nan
        metrics["precision"] = math.nan
        metrics["recall"] = math.nan
    else:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception as exc:
            metrics["auc"] = math.nan
            metric_warnings.append(f"auc_failed:{exc}")
        try:
            metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        except Exception as exc:
            metrics["f1"] = math.nan
            metric_warnings.append(f"f1_failed:{exc}")
        try:
            metrics["balacc"] = float(balanced_accuracy_score(y_true, y_pred))
        except Exception as exc:
            metrics["balacc"] = math.nan
            metric_warnings.append(f"balacc_failed:{exc}")
        try:
            metrics["precision"] = float(
                precision_score(y_true, y_pred, zero_division=0)
            )
        except Exception as exc:
            metrics["precision"] = math.nan
            metric_warnings.append(f"precision_failed:{exc}")
        try:
            metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        except Exception as exc:
            metrics["recall"] = math.nan
            metric_warnings.append(f"recall_failed:{exc}")

    try:
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    except Exception as exc:
        metrics["brier"] = math.nan
        metric_warnings.append(f"brier_failed:{exc}")

    if config.enable_ece:
        try:
            metrics["ece"] = compute_ece(y_true, y_prob, n_bins=config.calibration_bins)
        except Exception as exc:
            metrics["ece"] = math.nan
            metric_warnings.append(f"ece_failed:{exc}")
    else:
        metrics["ece"] = math.nan

    return metrics, metric_warnings


def make_inner_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    config: ExperimentConfig,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, list[str]]:
    warnings_out: list[str] = []
    unique_groups = pd.Series(groups_train).dropna().astype(str).unique()
    if len(unique_groups) >= 2:
        n_splits = min(3, len(unique_groups))
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(X_train, y_train, groups=groups_train))
        return splits, "GroupKFold", warnings_out

    warnings_out.append(
        "inner_id_groupkfold_unavailable_single_source_domain_using_stratifiedkfold_fallback"
    )
    class_counts = y_train.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    if y_train.nunique() >= 2 and min_class_count >= 2:
        n_splits = min(3, min_class_count)
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=config.random_seed
        )
        splits = list(splitter.split(X_train, y_train))
        return splits, "StratifiedKFoldFallback", warnings_out

    warnings_out.append("inner_id_unavailable")
    return [], "Unavailable", warnings_out


def make_within_domain_cv_splits(
    domain_df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
    warnings_out: list[str] = []
    y_domain = domain_df[config.target_col].astype(int)

    if y_domain.nunique() < 2:
        warnings_out.append("within_domain_cv_unavailable_single_class")
        return [], warnings_out

    class_counts = y_domain.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    n_splits = min(config.within_domain_cv_splits, len(domain_df), min_class_count)
    if n_splits < 2:
        warnings_out.append("within_domain_cv_unavailable_too_few_minority_samples")
        return [], warnings_out

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=config.random_seed
    )
    splits = list(splitter.split(domain_df, y_domain))
    return splits, warnings_out


def run_pairwise_site_transfer(
    df: pd.DataFrame,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = build_models(config, logger)
    domains = sorted(df[config.domain_col].dropna().astype(str).unique().tolist())
    pairwise_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []

    if not config.enable_pairwise_site_transfer:
        return pd.DataFrame(), pd.DataFrame()

    logger.info(
        "Starting pairwise single-site transfer matrix with %s domains", len(domains)
    )

    for train_domain, test_domain in product(domains, domains):
        train_df = df.loc[
            df[config.domain_col].astype(str) == train_domain
        ].reset_index(drop=True)
        test_df = df.loc[df[config.domain_col].astype(str) == test_domain].reset_index(
            drop=True
        )
        is_diagonal = train_domain == test_domain

        for model_name, estimator in models.items():
            row: dict[str, Any] = {
                "dataset": config.dataset_name,
                "model": model_name,
                "train_domain": train_domain,
                "test_domain": test_domain,
                "is_diagonal": bool(is_diagonal),
                "evaluation_type": "within_domain_cv"
                if is_diagonal
                else "single_site_transfer",
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "positive_rate_train": float(train_df[config.target_col].mean()),
                "positive_rate_test": float(test_df[config.target_col].mean()),
            }
            pairwise_warnings: list[str] = []

            if train_df[config.target_col].nunique() < 2:
                pairwise_warnings.append("single_class_in_pairwise_training")
                for metric_name in [
                    "auc",
                    "accuracy",
                    "f1",
                    "balacc",
                    "precision",
                    "recall",
                    "brier",
                    "ece",
                ]:
                    row[metric_name] = math.nan
                row["n_features_raw"] = math.nan
                row["n_features_processed"] = math.nan
                row["warnings"] = aggregate_warning_list(pairwise_warnings)
                pairwise_rows.append(row)
                continue

            if is_diagonal:
                # Diagonal cells are estimated with within-site CV to avoid optimistic resubstitution.
                splits, split_warnings = make_within_domain_cv_splits(train_df, config)
                pairwise_warnings.extend(split_warnings)
                if not splits:
                    for metric_name in [
                        "auc",
                        "accuracy",
                        "f1",
                        "balacc",
                        "precision",
                        "recall",
                        "brier",
                        "ece",
                    ]:
                        row[metric_name] = math.nan
                    row["n_features_raw"] = math.nan
                    row["n_features_processed"] = math.nan
                    row["warnings"] = aggregate_warning_list(pairwise_warnings)
                    pairwise_rows.append(row)
                    continue

                fold_predictions: list[dict[str, Any]] = []
                raw_feature_counts: list[int] = []
                processed_feature_counts: list[float] = []
                for cv_fold_idx, (cv_train_idx, cv_val_idx) in enumerate(
                    splits, start=1
                ):
                    cv_train_df = train_df.iloc[cv_train_idx].reset_index(drop=True)
                    cv_val_df = train_df.iloc[cv_val_idx].reset_index(drop=True)
                    y_cv_train = cv_train_df[config.target_col].astype(int)
                    y_cv_val = cv_val_df[config.target_col].astype(int)

                    if y_cv_train.nunique() < 2:
                        pairwise_warnings.append(
                            f"within_domain_cv_fold_{cv_fold_idx}_single_class_in_training"
                        )
                        continue

                    spec = prepare_columns(cv_train_df, config)
                    pairwise_warnings.extend(
                        [
                            f"within_domain_cv_fold_{cv_fold_idx}_{warning}"
                            for warning in spec.warnings
                        ]
                    )
                    if not spec.feature_cols:
                        pairwise_warnings.append(
                            f"within_domain_cv_fold_{cv_fold_idx}_no_features"
                        )
                        continue

                    pipeline = build_pipeline(spec, estimator, config)
                    try:
                        pipeline, fit_warnings = fit_with_warnings(
                            pipeline, cv_train_df, y_cv_train
                        )
                        pairwise_warnings.extend(
                            [
                                f"within_domain_cv_fold_{cv_fold_idx}_{warning}"
                                for warning in fit_warnings
                            ]
                        )
                        y_prob = safe_predict_proba(pipeline, cv_val_df)
                    except Exception as exc:
                        pairwise_warnings.append(
                            f"within_domain_cv_fold_{cv_fold_idx}_fit_or_predict_failed:{exc}"
                        )
                        logger.warning(
                            "Pairwise diagonal evaluation failed for model=%s site=%s fold=%s: %s",
                            model_name,
                            train_domain,
                            cv_fold_idx,
                            exc,
                        )
                        continue

                    y_pred = (y_prob >= 0.5).astype(int)
                    raw_feature_counts.append(int(len(spec.feature_cols)))
                    processed_feature_counts.append(
                        get_transformed_feature_count(pipeline, cv_train_df)
                    )
                    for local_idx in range(len(cv_val_df)):
                        fold_predictions.append(
                            {
                                "dataset": config.dataset_name,
                                "model": model_name,
                                "split": "pairwise_within_domain_cv",
                                "train_domain": train_domain,
                                "test_domain": test_domain,
                                "subject_id": (
                                    cv_val_df.iloc[local_idx][config.id_cols[0]]
                                    if config.id_cols
                                    else pd.NA
                                ),
                                "y_true": int(y_cv_val.iloc[local_idx]),
                                "y_prob": float(y_prob[local_idx]),
                                "y_pred": int(y_pred[local_idx]),
                            }
                        )

                if not fold_predictions:
                    pairwise_warnings.append("within_domain_cv_no_valid_predictions")
                    for metric_name in [
                        "auc",
                        "accuracy",
                        "f1",
                        "balacc",
                        "precision",
                        "recall",
                        "brier",
                        "ece",
                    ]:
                        row[metric_name] = math.nan
                    row["n_features_raw"] = math.nan
                    row["n_features_processed"] = math.nan
                    row["warnings"] = aggregate_warning_list(pairwise_warnings)
                    pairwise_rows.append(row)
                    continue

                pred_df = pd.DataFrame(fold_predictions)
                metric_values, metric_warnings = compute_metrics(
                    pred_df["y_true"].to_numpy(dtype=int),
                    pred_df["y_prob"].to_numpy(dtype=float),
                    config=config,
                )
                pairwise_warnings.extend(
                    [f"within_domain_cv_{warning}" for warning in metric_warnings]
                )
                prediction_rows.append(pred_df)
                row.update(metric_values)
                row["n_features_raw"] = (
                    float(np.mean(raw_feature_counts))
                    if raw_feature_counts
                    else math.nan
                )
                row["n_features_processed"] = (
                    float(np.nanmean(processed_feature_counts))
                    if processed_feature_counts
                    else math.nan
                )
                row["warnings"] = aggregate_warning_list(pairwise_warnings)
                pairwise_rows.append(row)
                continue

            spec = prepare_columns(train_df, config)
            pairwise_warnings.extend(spec.warnings)
            if not spec.feature_cols:
                pairwise_warnings.append("no_usable_features_in_pairwise_training")
                for metric_name in [
                    "auc",
                    "accuracy",
                    "f1",
                    "balacc",
                    "precision",
                    "recall",
                    "brier",
                    "ece",
                ]:
                    row[metric_name] = math.nan
                row["n_features_raw"] = 0
                row["n_features_processed"] = math.nan
                row["warnings"] = aggregate_warning_list(pairwise_warnings)
                pairwise_rows.append(row)
                continue

            pipeline = build_pipeline(spec, estimator, config)
            try:
                pipeline, fit_warnings = fit_with_warnings(
                    pipeline, train_df, train_df[config.target_col].astype(int)
                )
                pairwise_warnings.extend(fit_warnings)
                y_prob = safe_predict_proba(pipeline, test_df)
                y_pred = (y_prob >= 0.5).astype(int)
                metric_values, metric_warnings = compute_metrics(
                    test_df[config.target_col].to_numpy(dtype=int),
                    y_prob,
                    config=config,
                )
                pairwise_warnings.extend(
                    [f"pairwise_{warning}" for warning in metric_warnings]
                )
            except Exception as exc:
                pairwise_warnings.append(f"pairwise_fit_or_predict_failed:{exc}")
                logger.warning(
                    "Pairwise transfer failed for model=%s train=%s test=%s: %s",
                    model_name,
                    train_domain,
                    test_domain,
                    exc,
                )
                metric_values = {
                    metric_name: math.nan
                    for metric_name in [
                        "auc",
                        "accuracy",
                        "f1",
                        "balacc",
                        "precision",
                        "recall",
                        "brier",
                        "ece",
                    ]
                }
                y_prob = np.array([])
                y_pred = np.array([])
                pipeline = None

            if pipeline is not None and len(y_prob) == len(test_df):
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "dataset": config.dataset_name,
                            "model": model_name,
                            "split": "pairwise_ood",
                            "train_domain": train_domain,
                            "test_domain": test_domain,
                            "subject_id": (
                                test_df[config.id_cols[0]].astype(str)
                                if config.id_cols
                                else pd.Series(pd.NA, index=test_df.index)
                            ),
                            "y_true": test_df[config.target_col].astype(int).to_numpy(),
                            "y_prob": y_prob,
                            "y_pred": y_pred,
                        }
                    )
                )

            row.update(metric_values)
            row["n_features_raw"] = int(len(spec.feature_cols))
            row["n_features_processed"] = (
                get_transformed_feature_count(pipeline, train_df)
                if pipeline is not None
                else math.nan
            )
            row["warnings"] = aggregate_warning_list(pairwise_warnings)
            pairwise_rows.append(row)

    pairwise_results = pd.DataFrame(pairwise_rows)
    pairwise_prediction_df = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame()
    )
    return pairwise_results, pairwise_prediction_df


def inner_id_evaluation(
    *,
    full_df: pd.DataFrame,
    train_idx: np.ndarray,
    estimator: Any,
    model_name: str,
    config: ExperimentConfig,
    logger: logging.Logger,
    outer_test_domain: str,
) -> tuple[dict[str, float], pd.DataFrame, list[str]]:
    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    X_train = train_df.copy()
    y_train = train_df[config.target_col].astype(int)
    groups_train = train_df[config.domain_col].astype("string")

    inner_splits, split_name, warnings_out = make_inner_splits(
        X_train, y_train, groups_train, config
    )
    prediction_rows: list[dict[str, Any]] = []

    if not inner_splits:
        id_metrics = {
            f"{metric}_id": math.nan
            for metric in [
                "auc",
                "accuracy",
                "f1",
                "balacc",
                "precision",
                "recall",
                "brier",
                "ece",
            ]
        }
        return id_metrics, pd.DataFrame(), warnings_out

    for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(
        inner_splits, start=1
    ):
        inner_train_df = train_df.iloc[inner_train_idx].reset_index(drop=True)
        inner_val_df = train_df.iloc[inner_val_idx].reset_index(drop=True)
        y_inner_train = inner_train_df[config.target_col].astype(int)
        y_inner_val = inner_val_df[config.target_col].astype(int)

        if y_inner_train.nunique() < 2:
            warnings_out.append(f"inner_fold_{inner_fold_idx}_single_class_in_training")
            continue
        if len(inner_val_df) == 0:
            warnings_out.append(f"inner_fold_{inner_fold_idx}_empty_validation")
            continue

        spec = prepare_columns(inner_train_df, config)
        warnings_out.extend(
            [f"inner_fold_{inner_fold_idx}_{warning}" for warning in spec.warnings]
        )
        if not spec.feature_cols:
            warnings_out.append(f"inner_fold_{inner_fold_idx}_no_features")
            continue

        pipeline = build_pipeline(spec, estimator, config)
        try:
            pipeline, fit_warnings = fit_with_warnings(
                pipeline, inner_train_df, y_inner_train
            )
            warnings_out.extend(
                [f"inner_fold_{inner_fold_idx}_{warning}" for warning in fit_warnings]
            )
            y_prob = safe_predict_proba(pipeline, inner_val_df)
        except Exception as exc:
            warnings_out.append(
                f"inner_fold_{inner_fold_idx}_fit_or_predict_failed:{exc}"
            )
            logger.warning(
                "Inner ID evaluation failed for %s outer=%s inner=%s: %s",
                model_name,
                outer_test_domain,
                inner_fold_idx,
                exc,
            )
            continue

        y_pred = (y_prob >= 0.5).astype(int)
        validation_domains = sorted(
            inner_val_df[config.domain_col].astype("string").unique().tolist()
        )
        validation_domain_label = "|".join(validation_domains)
        for local_idx, row_idx in enumerate(inner_val_df.index):
            prediction_rows.append(
                {
                    "dataset": config.dataset_name,
                    "model": model_name,
                    "split": "id",
                    "outer_test_domain": outer_test_domain,
                    "evaluation_domain": validation_domain_label,
                    "subject_id": inner_val_df.iloc[local_idx][config.id_cols[0]]
                    if config.id_cols
                    else pd.NA,
                    "y_true": int(y_inner_val.iloc[local_idx]),
                    "y_prob": float(y_prob[local_idx]),
                    "y_pred": int(y_pred[local_idx]),
                }
            )

    if not prediction_rows:
        id_metrics = {
            f"{metric}_id": math.nan
            for metric in [
                "auc",
                "accuracy",
                "f1",
                "balacc",
                "precision",
                "recall",
                "brier",
                "ece",
            ]
        }
        warnings_out.append("inner_id_no_valid_predictions")
        return id_metrics, pd.DataFrame(), warnings_out

    pred_df = pd.DataFrame(prediction_rows)
    metric_values, metric_warnings = compute_metrics(
        pred_df["y_true"].to_numpy(),
        pred_df["y_prob"].to_numpy(),
        config=config,
    )
    warnings_out.extend([f"id_{warning}" for warning in metric_warnings])
    id_metrics = {f"{metric}_id": value for metric, value in metric_values.items()}
    id_metrics["id_validation_scheme"] = split_name
    return id_metrics, pred_df, warnings_out


def aggregate_warning_list(items: list[str]) -> str:
    cleaned = [str(item) for item in items if str(item).strip()]
    if not cleaned:
        return ""
    deduplicated = list(dict.fromkeys(cleaned))
    return " | ".join(deduplicated)


def run_lodo_evaluation(
    df: pd.DataFrame,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = build_models(config, logger)
    outer_domains = sorted(df[config.domain_col].dropna().astype(str).unique().tolist())
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []

    logger.info(
        "Starting outer LODO loop with %s domains: %s",
        len(outer_domains),
        ", ".join(outer_domains),
    )

    for outer_fold_idx, test_domain in enumerate(outer_domains, start=1):
        test_mask = df[config.domain_col].astype(str) == test_domain
        train_mask = ~test_mask
        train_df = df.loc[train_mask].reset_index(drop=True)
        test_df = df.loc[test_mask].reset_index(drop=True)

        outer_warnings: list[str] = []
        if len(test_df) < config.min_domain_size:
            outer_warnings.append(
                f"test_domain_smaller_than_min_domain_size:{len(test_df)}<{config.min_domain_size}"
            )
        if train_df[config.target_col].nunique() < 2:
            outer_warnings.append("single_class_in_outer_training")
        if test_df[config.target_col].nunique() < 2:
            outer_warnings.append("single_class_in_outer_test")

        logger.info(
            "Outer fold %s/%s | test_domain=%s | n_train=%s | n_test=%s",
            outer_fold_idx,
            len(outer_domains),
            test_domain,
            len(train_df),
            len(test_df),
        )

        for model_name, estimator in models.items():
            model_warnings = list(outer_warnings)
            row: dict[str, Any] = {
                "dataset": config.dataset_name,
                "model": model_name,
                "outer_fold": outer_fold_idx,
                "test_domain": test_domain,
                "source_domains": "|".join(
                    sorted(train_df[config.domain_col].astype(str).unique().tolist())
                ),
                "n_source_domains": int(train_df[config.domain_col].nunique()),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "positive_rate_train": float(train_df[config.target_col].mean()),
                "positive_rate_test": float(test_df[config.target_col].mean()),
                "feature_selection_method": config.feature_selection_method,
            }

            if train_df[config.target_col].nunique() < 2:
                for metric_name in [
                    "auc",
                    "accuracy",
                    "f1",
                    "balacc",
                    "precision",
                    "recall",
                    "brier",
                    "ece",
                ]:
                    row[f"{metric_name}_id"] = math.nan
                    row[f"{metric_name}_ood"] = math.nan
                row["gap_auc"] = math.nan
                row["gap_accuracy"] = math.nan
                row["gap_f1"] = math.nan
                row["gap_balacc"] = math.nan
                row["n_features_raw"] = math.nan
                row["n_features_processed"] = math.nan
                row["warnings"] = aggregate_warning_list(model_warnings)
                fold_rows.append(row)
                continue

            id_metrics, id_predictions, id_warnings = inner_id_evaluation(
                full_df=train_df,
                train_idx=np.arange(len(train_df)),
                estimator=estimator,
                model_name=model_name,
                config=config,
                logger=logger,
                outer_test_domain=test_domain,
            )
            model_warnings.extend(id_warnings)
            if not id_predictions.empty:
                prediction_rows.append(id_predictions)

            spec = prepare_columns(train_df, config)
            model_warnings.extend(spec.warnings)
            if not spec.feature_cols:
                model_warnings.append("no_usable_features_in_outer_training")
                for metric_name in [
                    "auc",
                    "accuracy",
                    "f1",
                    "balacc",
                    "precision",
                    "recall",
                    "brier",
                    "ece",
                ]:
                    row[f"{metric_name}_id"] = id_metrics.get(
                        f"{metric_name}_id", math.nan
                    )
                    row[f"{metric_name}_ood"] = math.nan
                row["gap_auc"] = math.nan
                row["gap_accuracy"] = math.nan
                row["gap_f1"] = math.nan
                row["gap_balacc"] = math.nan
                row["n_features_raw"] = 0
                row["n_features_processed"] = math.nan
                row["warnings"] = aggregate_warning_list(model_warnings)
                fold_rows.append(row)
                continue

            pipeline = build_pipeline(spec, estimator, config)
            try:
                pipeline, fit_warnings = fit_with_warnings(
                    pipeline,
                    train_df,
                    train_df[config.target_col].astype(int),
                )
                model_warnings.extend(fit_warnings)
                y_prob_ood = safe_predict_proba(pipeline, test_df)
                y_pred_ood = (y_prob_ood >= 0.5).astype(int)
                ood_metric_values, ood_metric_warnings = compute_metrics(
                    test_df[config.target_col].to_numpy(dtype=int),
                    y_prob_ood,
                    config=config,
                )
                model_warnings.extend(
                    [f"ood_{warning}" for warning in ood_metric_warnings]
                )
            except Exception as exc:
                model_warnings.append(f"outer_fit_or_predict_failed:{exc}")
                logger.warning(
                    "Outer fold failed for model=%s domain=%s: %s",
                    model_name,
                    test_domain,
                    exc,
                )
                y_prob_ood = np.array([])
                y_pred_ood = np.array([])
                ood_metric_values = {
                    metric_name: math.nan
                    for metric_name in [
                        "auc",
                        "accuracy",
                        "f1",
                        "balacc",
                        "precision",
                        "recall",
                        "brier",
                        "ece",
                    ]
                }
                pipeline = None

            if pipeline is not None and len(y_prob_ood) == len(test_df):
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "dataset": config.dataset_name,
                            "model": model_name,
                            "split": "ood",
                            "outer_test_domain": test_domain,
                            "evaluation_domain": test_domain,
                            "subject_id": test_df[config.id_cols[0]].astype(str)
                            if config.id_cols
                            else pd.Series(pd.NA, index=test_df.index),
                            "y_true": test_df[config.target_col].astype(int).to_numpy(),
                            "y_prob": y_prob_ood,
                            "y_pred": y_pred_ood,
                        }
                    )
                )

            row.update(id_metrics)
            for metric_name, metric_value in ood_metric_values.items():
                row[f"{metric_name}_ood"] = metric_value
            row["gap_auc"] = (
                row.get("auc_id") - row.get("auc_ood")
                if pd.notna(row.get("auc_id")) and pd.notna(row.get("auc_ood"))
                else math.nan
            )
            row["gap_accuracy"] = (
                row.get("accuracy_id") - row.get("accuracy_ood")
                if pd.notna(row.get("accuracy_id"))
                and pd.notna(row.get("accuracy_ood"))
                else math.nan
            )
            row["gap_f1"] = (
                row.get("f1_id") - row.get("f1_ood")
                if pd.notna(row.get("f1_id")) and pd.notna(row.get("f1_ood"))
                else math.nan
            )
            row["gap_balacc"] = (
                row.get("balacc_id") - row.get("balacc_ood")
                if pd.notna(row.get("balacc_id")) and pd.notna(row.get("balacc_ood"))
                else math.nan
            )
            row["n_features_raw"] = int(len(spec.feature_cols))
            row["n_features_processed"] = (
                get_transformed_feature_count(pipeline, train_df)
                if pipeline is not None
                else math.nan
            )
            row["warnings"] = aggregate_warning_list(model_warnings)
            fold_rows.append(row)

    fold_results = pd.DataFrame(fold_rows)
    prediction_df = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame()
    )
    return fold_results, prediction_df


def aggregate_results(
    fold_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_roots = [
        "auc",
        "accuracy",
        "f1",
        "balacc",
        "brier",
        "ece",
        "precision",
        "recall",
    ]
    model_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []

    for model_name, model_df in fold_results.groupby("model", dropna=False):
        row: dict[str, Any] = {"model": model_name, "n_folds_total": int(len(model_df))}
        row["n_folds_valid_auc_ood"] = (
            int(model_df["auc_ood"].notna().sum()) if "auc_ood" in model_df else 0
        )
        row["mean_n_features_processed"] = (
            float(model_df["n_features_processed"].dropna().mean())
            if model_df["n_features_processed"].notna().any()
            else math.nan
        )
        for metric_root in metric_roots:
            for split in ["id", "ood"]:
                column = f"{metric_root}_{split}"
                if column in model_df.columns:
                    row[f"{column}_mean"] = (
                        float(model_df[column].dropna().mean())
                        if model_df[column].notna().any()
                        else math.nan
                    )
                    row[f"{column}_std"] = (
                        float(model_df[column].dropna().std(ddof=1))
                        if model_df[column].notna().sum() > 1
                        else math.nan
                    )
                    row[f"{column}_median"] = (
                        float(model_df[column].dropna().median())
                        if model_df[column].notna().any()
                        else math.nan
                    )
            gap_column = (
                f"gap_{metric_root}"
                if metric_root in {"auc", "accuracy", "f1", "balacc"}
                else None
            )
            if gap_column and gap_column in model_df.columns:
                row[f"{gap_column}_mean"] = (
                    float(model_df[gap_column].dropna().mean())
                    if model_df[gap_column].notna().any()
                    else math.nan
                )
                row[f"{gap_column}_std"] = (
                    float(model_df[gap_column].dropna().std(ddof=1))
                    if model_df[gap_column].notna().sum() > 1
                    else math.nan
                )
        model_rows.append(row)

    model_summary = pd.DataFrame(model_rows).sort_values("model").reset_index(drop=True)

    for (test_domain, model_name), group_df in fold_results.groupby(
        ["test_domain", "model"], dropna=False
    ):
        row = {
            "test_domain": test_domain,
            "model": model_name,
            "n_test": int(group_df["n_test"].mean()),
            "auc_ood_mean": float(group_df["auc_ood"].dropna().mean())
            if group_df["auc_ood"].notna().any()
            else math.nan,
            "accuracy_ood_mean": float(group_df["accuracy_ood"].dropna().mean())
            if group_df["accuracy_ood"].notna().any()
            else math.nan,
            "f1_ood_mean": float(group_df["f1_ood"].dropna().mean())
            if group_df["f1_ood"].notna().any()
            else math.nan,
            "balacc_ood_mean": float(group_df["balacc_ood"].dropna().mean())
            if group_df["balacc_ood"].notna().any()
            else math.nan,
            "brier_ood_mean": float(group_df["brier_ood"].dropna().mean())
            if group_df["brier_ood"].notna().any()
            else math.nan,
            "ece_ood_mean": float(group_df["ece_ood"].dropna().mean())
            if group_df["ece_ood"].notna().any()
            else math.nan,
            "warnings": aggregate_warning_list(
                group_df["warnings"].fillna("").tolist()
            ),
        }
        domain_rows.append(row)

    domain_by_model = (
        pd.DataFrame(domain_rows)
        .sort_values(["test_domain", "model"])
        .reset_index(drop=True)
    )
    overall_domain = (
        fold_results.groupby("test_domain", dropna=False)
        .agg(
            n_test=("n_test", "mean"),
            auc_ood_mean=("auc_ood", "mean"),
            accuracy_ood_mean=("accuracy_ood", "mean"),
            f1_ood_mean=("f1_ood", "mean"),
            balacc_ood_mean=("balacc_ood", "mean"),
        )
        .reset_index()
        .sort_values("auc_ood_mean", ascending=True)
        .reset_index(drop=True)
    )
    overall_domain["difficulty_rank_auc"] = np.arange(1, len(overall_domain) + 1)
    overall_domain = overall_domain.sort_values(
        "accuracy_ood_mean", ascending=True
    ).reset_index(drop=True)
    overall_domain["difficulty_rank_accuracy"] = np.arange(1, len(overall_domain) + 1)
    domain_summary = domain_by_model.merge(
        overall_domain, on="test_domain", suffixes=("", "_overall")
    )
    return model_summary, domain_summary, overall_domain


def aggregate_pairwise_results(
    pairwise_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pairwise_results.empty:
        return pd.DataFrame(), pd.DataFrame()

    offdiag = pairwise_results.loc[~pairwise_results["is_diagonal"]].copy()
    if offdiag.empty:
        return pd.DataFrame(), pd.DataFrame()

    pairwise_model_summary = (
        offdiag.groupby("model", dropna=False)
        .agg(
            n_pairs=("model", "size"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            balacc_mean=("balacc", "mean"),
            balacc_std=("balacc", "std"),
        )
        .reset_index()
        .sort_values("model")
        .reset_index(drop=True)
    )

    pairwise_test_summary = (
        offdiag.groupby(["test_domain", "model"], dropna=False)
        .agg(
            n_pairs=("model", "size"),
            auc_mean=("auc", "mean"),
            accuracy_mean=("accuracy", "mean"),
            f1_mean=("f1", "mean"),
            balacc_mean=("balacc", "mean"),
        )
        .reset_index()
        .sort_values(["test_domain", "model"])
        .reset_index(drop=True)
    )
    return pairwise_model_summary, pairwise_test_summary


def save_figure(fig: plt.Figure, path_no_suffix: Path, save_pdf: bool) -> None:
    ensure_directory(path_no_suffix.parent)
    fig.savefig(path_no_suffix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(path_no_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_grouped_metric_by_domain(
    fold_results: pd.DataFrame,
    *,
    metric_column: str,
    title: str,
    ylabel: str,
    output_path_no_suffix: Path,
    save_pdf: bool,
) -> None:
    if fold_results.empty or metric_column not in fold_results.columns:
        return
    pivot = fold_results.pivot_table(
        index="test_domain", columns="model", values=metric_column, aggfunc="mean"
    )
    if pivot.empty:
        return
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.index) * 1.2), 6))
    for model_idx, model_name in enumerate(pivot.columns):
        ax.bar(
            x + model_idx * width - (len(pivot.columns) - 1) * width / 2,
            pivot[model_name].to_numpy(),
            width,
            label=model_name,
        )
    ax.set_title(title)
    ax.set_xlabel("Test Domain")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=35, ha="right")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, output_path_no_suffix, save_pdf)


def plot_mean_performance_by_model(
    model_summary: pd.DataFrame, output_path_no_suffix: Path, save_pdf: bool
) -> None:
    metric_columns = {
        "AUC OOD": "auc_ood_mean",
        "Accuracy OOD": "accuracy_ood_mean",
        "F1 OOD": "f1_ood_mean",
        "BalAcc OOD": "balacc_ood_mean",
    }
    if model_summary.empty or any(
        column not in model_summary.columns for column in metric_columns.values()
    ):
        return
    x = np.arange(len(model_summary))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 6))
    center_offset = (len(metric_columns) - 1) / 2
    for idx, (label, column) in enumerate(metric_columns.items()):
        ax.bar(
            x + (idx - center_offset) * width,
            model_summary[column].to_numpy(),
            width,
            label=label,
        )
    ax.set_title("Mean OOD Performance by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(model_summary["model"].tolist(), rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, output_path_no_suffix, save_pdf)


def plot_domain_difficulty(
    overall_domain: pd.DataFrame,
    *,
    metric_column: str,
    title: str,
    ylabel: str,
    output_path_no_suffix: Path,
    save_pdf: bool,
) -> None:
    if overall_domain.empty or metric_column not in overall_domain.columns:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(overall_domain) * 1.2), 6))
    x = np.arange(len(overall_domain))
    ax.bar(x, overall_domain[metric_column].to_numpy())
    ax.set_title(title)
    ax.set_xlabel("Test Domain")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(overall_domain["test_domain"].tolist(), rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, output_path_no_suffix, save_pdf)


def plot_heatmap(
    fold_results: pd.DataFrame,
    *,
    metric_column: str,
    title: str,
    output_path_no_suffix: Path,
    save_pdf: bool,
) -> None:
    if fold_results.empty or metric_column not in fold_results.columns:
        return
    pivot = fold_results.pivot_table(
        index="model", columns="test_domain", values=metric_column, aggfunc="mean"
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot.columns) * 1.2), max(4, len(pivot.index) * 0.9))
    )
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Test Domain")
    ax.set_ylabel("Model")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    cbar = fig.colorbar(image, ax=ax)
    cbar.ax.set_ylabel(metric_column, rotation=90)
    save_figure(fig, output_path_no_suffix, save_pdf)


def plot_pairwise_site_heatmap(
    pairwise_results: pd.DataFrame,
    *,
    model_name: str,
    metric_column: str,
    title: str,
    output_path_no_suffix: Path,
    save_pdf: bool,
) -> None:
    if pairwise_results.empty or metric_column not in pairwise_results.columns:
        return

    model_df = pairwise_results.loc[pairwise_results["model"] == model_name].copy()
    if model_df.empty:
        return

    pivot = (
        model_df.pivot_table(
            index="train_domain",
            columns="test_domain",
            values=metric_column,
            aggfunc="mean",
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    if pivot.empty:
        return

    matrix = pivot.to_numpy(dtype=float)
    fig_w = max(7, len(pivot.columns) * 1.6)
    fig_h = max(6, len(pivot.index) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)

    ax.set_title(title)
    ax.set_xlabel("Test Site")
    ax.set_ylabel("Train Site")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            label = "NA" if np.isnan(value) else f"{value:.2f}"
            text_color = "black" if np.isnan(value) or value < 0.65 else "white"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.ax.set_ylabel(metric_column, rotation=90)
    save_figure(fig, output_path_no_suffix, save_pdf)


def build_calibration_dataframe(
    prediction_df: pd.DataFrame, config: ExperimentConfig
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if prediction_df.empty:
        return pd.DataFrame()
    for model_name, model_df in prediction_df[prediction_df["split"] == "ood"].groupby(
        "model"
    ):
        y_true = model_df["y_true"].to_numpy(dtype=int)
        y_prob = model_df["y_prob"].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            continue
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=config.calibration_bins, strategy="uniform"
        )
        for idx, (pred_bin, true_bin) in enumerate(zip(prob_pred, prob_true), start=1):
            rows.append(
                {
                    "model": model_name,
                    "bin_index": idx,
                    "predicted_probability": float(pred_bin),
                    "observed_frequency": float(true_bin),
                }
            )
    return pd.DataFrame(rows)


def plot_calibration_curves(
    calibration_df: pd.DataFrame,
    output_dir: Path,
    save_pdf: bool,
) -> None:
    if calibration_df.empty:
        return
    ensure_directory(output_dir)
    for model_name, model_df in calibration_df.groupby("model"):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration"
        )
        ax.plot(
            model_df["predicted_probability"],
            model_df["observed_frequency"],
            marker="o",
            linewidth=2,
            label=model_name,
        )
        ax.set_title(f"Reliability Curve: {model_name}")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()
        save_figure(
            fig, output_dir / f"calibration_{sanitize_name(model_name)}", save_pdf
        )


def make_plots(
    fold_results: pd.DataFrame,
    model_summary: pd.DataFrame,
    overall_domain: pd.DataFrame,
    calibration_df: pd.DataFrame,
    output_dir: Path,
    config: ExperimentConfig,
) -> None:
    plots_dir = ensure_directory(output_dir / "plots")
    plot_grouped_metric_by_domain(
        fold_results,
        metric_column="auc_ood",
        title="OOD ROC-AUC by Domain and Model",
        ylabel="ROC-AUC",
        output_path_no_suffix=plots_dir / "auc_ood_by_domain",
        save_pdf=config.save_pdf_plots,
    )
    plot_grouped_metric_by_domain(
        fold_results,
        metric_column="accuracy_ood",
        title="OOD Accuracy by Domain and Model",
        ylabel="Accuracy",
        output_path_no_suffix=plots_dir / "accuracy_ood_by_domain",
        save_pdf=config.save_pdf_plots,
    )
    plot_grouped_metric_by_domain(
        fold_results,
        metric_column="f1_ood",
        title="OOD F1 by Domain and Model",
        ylabel="F1-score",
        output_path_no_suffix=plots_dir / "f1_ood_by_domain",
        save_pdf=config.save_pdf_plots,
    )
    plot_grouped_metric_by_domain(
        fold_results,
        metric_column="gap_auc",
        title="ID-OOD Gap in ROC-AUC by Domain and Model",
        ylabel="Gap (ID - OOD)",
        output_path_no_suffix=plots_dir / "gap_auc_by_domain",
        save_pdf=config.save_pdf_plots,
    )
    plot_grouped_metric_by_domain(
        fold_results,
        metric_column="gap_accuracy",
        title="ID-OOD Gap in Accuracy by Domain and Model",
        ylabel="Gap (ID - OOD)",
        output_path_no_suffix=plots_dir / "gap_accuracy_by_domain",
        save_pdf=config.save_pdf_plots,
    )
    plot_grouped_metric_by_domain(
        fold_results,
        metric_column="gap_f1",
        title="ID-OOD Gap in F1 by Domain and Model",
        ylabel="Gap (ID - OOD)",
        output_path_no_suffix=plots_dir / "gap_f1_by_domain",
        save_pdf=config.save_pdf_plots,
    )
    plot_mean_performance_by_model(
        model_summary,
        output_path_no_suffix=plots_dir / "mean_performance_by_model",
        save_pdf=config.save_pdf_plots,
    )
    plot_domain_difficulty(
        overall_domain,
        metric_column="auc_ood_mean",
        title="Domain Difficulty Ranking by Mean OOD AUC",
        ylabel="Mean OOD AUC",
        output_path_no_suffix=plots_dir / "domain_difficulty_ranking",
        save_pdf=config.save_pdf_plots,
    )
    plot_domain_difficulty(
        overall_domain,
        metric_column="accuracy_ood_mean",
        title="Site Ranking by Mean OOD Accuracy",
        ylabel="Mean OOD Accuracy",
        output_path_no_suffix=plots_dir / "site_accuracy_ranking",
        save_pdf=config.save_pdf_plots,
    )
    plot_heatmap(
        fold_results,
        metric_column="auc_ood",
        title="Heatmap: OOD ROC-AUC",
        output_path_no_suffix=plots_dir / "heatmap_auc_ood",
        save_pdf=config.save_pdf_plots,
    )
    plot_heatmap(
        fold_results,
        metric_column="accuracy_ood",
        title="Heatmap: OOD Accuracy",
        output_path_no_suffix=plots_dir / "heatmap_accuracy_ood",
        save_pdf=config.save_pdf_plots,
    )
    plot_heatmap(
        fold_results,
        metric_column="f1_ood",
        title="Heatmap: OOD F1",
        output_path_no_suffix=plots_dir / "heatmap_f1_ood",
        save_pdf=config.save_pdf_plots,
    )
    if config.enable_calibration_plots:
        plot_calibration_curves(calibration_df, plots_dir, config.save_pdf_plots)


def make_heatmaps(
    pairwise_results: pd.DataFrame,
    output_dir: Path,
    config: ExperimentConfig,
) -> None:
    plots_dir = ensure_directory(output_dir / "plots")
    if pairwise_results.empty:
        return

    for model_name in sorted(
        pairwise_results["model"].dropna().astype(str).unique().tolist()
    ):
        safe_model_name = sanitize_name(model_name)
        plot_pairwise_site_heatmap(
            pairwise_results,
            model_name=model_name,
            metric_column="auc",
            title=f"Train Site -> Test Site Heatmap (AUC): {model_name}",
            output_path_no_suffix=plots_dir
            / f"train_test_heatmap_auc_{safe_model_name}",
            save_pdf=config.save_pdf_plots,
        )
        plot_pairwise_site_heatmap(
            pairwise_results,
            model_name=model_name,
            metric_column="f1",
            title=f"Train Site -> Test Site Heatmap (F1): {model_name}",
            output_path_no_suffix=plots_dir
            / f"train_test_heatmap_f1_{safe_model_name}",
            save_pdf=config.save_pdf_plots,
        )


def dataframe_to_block(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows available._"
    view = df.head(max_rows) if max_rows is not None else df
    return "```text\n" + view.to_string(index=False) + "\n```"


def write_report(
    *,
    config: ExperimentConfig,
    target_mapping: dict[str, Any],
    dataset_df: pd.DataFrame,
    fold_results: pd.DataFrame,
    pairwise_results: pd.DataFrame,
    model_summary: pd.DataFrame,
    domain_summary: pd.DataFrame,
    overall_domain: pd.DataFrame,
    pairwise_model_summary: pd.DataFrame,
    pairwise_test_summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    invalid_outer = (
        int(fold_results["auc_ood"].isna().sum()) if "auc_ood" in fold_results else 0
    )
    hardest_domains = overall_domain.head(3)[
        ["test_domain", "auc_ood_mean", "accuracy_ood_mean", "f1_ood_mean"]
    ]
    easiest_domains = overall_domain.tail(3).sort_values(
        "accuracy_ood_mean", ascending=False
    )[["test_domain", "auc_ood_mean", "accuracy_ood_mean", "f1_ood_mean"]]
    offdiag_pairwise = (
        pairwise_results.loc[~pairwise_results["is_diagonal"]].copy()
        if not pairwise_results.empty and "is_diagonal" in pairwise_results.columns
        else pd.DataFrame()
    )
    weakest_transfers = (
        offdiag_pairwise.sort_values(["auc", "f1"], ascending=[True, True]).head(6)[
            ["model", "train_domain", "test_domain", "auc", "accuracy", "f1"]
        ]
        if not offdiag_pairwise.empty
        else pd.DataFrame()
    )
    strongest_transfers = (
        offdiag_pairwise.sort_values(["auc", "f1"], ascending=[False, False]).head(6)[
            ["model", "train_domain", "test_domain", "auc", "accuracy", "f1"]
        ]
        if not offdiag_pairwise.empty
        else pd.DataFrame()
    )

    comments: list[str] = []
    if not model_summary.empty:
        mean_gap = (
            model_summary["gap_auc_mean"].dropna().mean()
            if "gap_auc_mean" in model_summary
            else math.nan
        )
        auc_std = (
            model_summary["auc_ood_std"].dropna().mean()
            if "auc_ood_std" in model_summary
            else math.nan
        )
        if pd.notna(mean_gap) and mean_gap > 0.10:
            comments.append(
                "The mean ID-OOD AUC gap is large, suggesting limited raw transportability across domains."
            )
        if pd.notna(auc_std) and auc_std > 0.10:
            comments.append(
                "OOD performance varies substantially across domains, indicating marked cross-site heterogeneity."
            )
        stable_candidates = model_summary.sort_values(
            ["auc_ood_mean", "gap_auc_mean"], ascending=[False, True]
        )
        if not stable_candidates.empty:
            best_model = stable_candidates.iloc[0]["model"]
            comments.append(
                f"{best_model} appears to be the strongest baseline under the current raw M0 setting."
            )
    if not offdiag_pairwise.empty:
        pairwise_auc_mean = offdiag_pairwise["auc"].dropna().mean()
        diagonal_auc_mean = (
            pairwise_results.loc[pairwise_results["is_diagonal"], "auc"].dropna().mean()
        )
        if (
            pd.notna(pairwise_auc_mean)
            and pd.notna(diagonal_auc_mean)
            and diagonal_auc_mean - pairwise_auc_mean > 0.10
        ):
            comments.append(
                "Single-site transfer heatmaps show a clear within-site versus cross-site drop, reinforcing the transportability challenge."
            )

    comment_lines = (
        [f"- {comment}" for comment in comments]
        if comments
        else ["- No major automatic warning comment was triggered."]
    )

    report_lines = [
        f"# Baseline M0 Report: {config.dataset_name}",
        "",
        "## Dataset",
        f"- Input path: `{config.input_path}`",
        f"- Shape: `{dataset_df.shape[0]} rows x {dataset_df.shape[1]} columns`",
        f"- Target column: `{config.target_col}`",
        f"- Domain column: `{config.domain_col}`",
        f"- Number of domains: `{dataset_df[config.domain_col].nunique(dropna=True)}`",
        f"- Target mapping: `{target_mapping}`",
        "",
        "## Experiment Configuration",
        f"- Outer validation: `Leave-One-Domain-Out (LODO)`",
        "- ID estimate: internal validation on source domains only, using `GroupKFold` when possible.",
        "- All preprocessing steps are fit on training data only within each fold.",
        f"- Models: `{', '.join(model_summary['model'].tolist())}`",
        f"- Feature selection method: `{config.feature_selection_method}`",
        f"- Random seed: `{config.random_seed}`",
        "",
        "## Model Summary",
        dataframe_to_block(model_summary.round(4)),
        "",
        "## Hardest Domains",
        dataframe_to_block(hardest_domains.round(4)),
        "",
        "## Easiest Domains",
        dataframe_to_block(easiest_domains.round(4)),
        "",
        "## Domain Summary",
        dataframe_to_block(
            domain_summary[
                [
                    "test_domain",
                    "model",
                    "auc_ood_mean",
                    "accuracy_ood_mean",
                    "f1_ood_mean",
                    "balacc_ood_mean",
                    "difficulty_rank_accuracy",
                    "difficulty_rank_auc",
                ]
            ].round(4)
        ),
        "",
        "## Pairwise Train->Test Site Summary",
        dataframe_to_block(pairwise_model_summary.round(4)),
        "",
        "## Weakest Pairwise Transfers",
        dataframe_to_block(weakest_transfers.round(4), max_rows=6),
        "",
        "## Strongest Pairwise Transfers",
        dataframe_to_block(strongest_transfers.round(4), max_rows=6),
        "",
        "## Pairwise Test-Site Summary",
        dataframe_to_block(pairwise_test_summary.round(4), max_rows=20),
        "",
        "## Warning Summary",
        f"- Outer model-folds with non-computable OOD AUC: `{invalid_outer}`",
        f"- Total warnings recorded: `{int(fold_results['warnings'].fillna('').ne('').sum())}`",
        f"- Pairwise experiments with recorded warnings: `{int(pairwise_results['warnings'].fillna('').ne('').sum()) if not pairwise_results.empty else 0}`",
        "",
        "## Automatic Comment",
        *comment_lines,
        "",
        "## Methodological Notes",
        "- No harmonization, domain adaptation, or reweighting was applied in this baseline M0 run.",
        "- Calibration is summarized with Brier score and, when enabled, Expected Calibration Error (ECE).",
        "- Pairwise heatmap diagonals are based on within-site cross-validation, not on resubstitution, to avoid optimistic self-test estimates.",
        "- Off-diagonal heatmap cells are obtained by fitting on one source site only and evaluating on a different held-out site.",
        "- Any metric that is undefined for a given fold is stored as `NaN` and logged without stopping the experiment.",
    ]
    report_path = output_dir / "baseline_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return report_path


def save_outputs(
    *,
    config: ExperimentConfig,
    target_mapping: dict[str, Any],
    dataset_df: pd.DataFrame,
    fold_results: pd.DataFrame,
    prediction_df: pd.DataFrame,
    pairwise_results: pd.DataFrame,
    pairwise_prediction_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    domain_summary: pd.DataFrame,
    overall_domain: pd.DataFrame,
    pairwise_model_summary: pd.DataFrame,
    pairwise_test_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    ensure_directory(output_dir)
    ensure_directory(output_dir / "plots")

    fold_results.to_csv(output_dir / "fold_results.csv", index=False)
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    domain_summary.to_csv(output_dir / "domain_summary.csv", index=False)
    overall_domain.to_csv(output_dir / "domain_difficulty_summary.csv", index=False)
    prediction_df.to_csv(output_dir / "prediction_records.csv", index=False)
    pairwise_results.to_csv(output_dir / "train_test_site_results.csv", index=False)
    pairwise_prediction_df.to_csv(
        output_dir / "train_test_site_prediction_records.csv", index=False
    )
    pairwise_model_summary.to_csv(
        output_dir / "train_test_site_model_summary.csv", index=False
    )
    pairwise_test_summary.to_csv(
        output_dir / "train_test_site_test_domain_summary.csv", index=False
    )

    calibration_df = build_calibration_dataframe(prediction_df, config)
    calibration_df.to_csv(output_dir / "calibration_curve_data.csv", index=False)
    make_plots(
        fold_results, model_summary, overall_domain, calibration_df, output_dir, config
    )
    make_heatmaps(pairwise_results, output_dir, config)
    write_report(
        config=config,
        target_mapping=target_mapping,
        dataset_df=dataset_df,
        fold_results=fold_results,
        pairwise_results=pairwise_results,
        model_summary=model_summary,
        domain_summary=domain_summary,
        overall_domain=overall_domain,
        pairwise_model_summary=pairwise_model_summary,
        pairwise_test_summary=pairwise_test_summary,
        output_dir=output_dir,
    )

    config_snapshot = asdict(config)
    config_snapshot["resolved_output_dir"] = str(output_dir)
    config_snapshot["target_mapping"] = target_mapping
    with (output_dir / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(config_snapshot, handle, indent=2, sort_keys=True, default=str)


def run_experiment(config: ExperimentConfig) -> dict[str, Path]:
    output_dir = ensure_directory(config.output_dir)
    logger = init_logger(output_dir / "logs.txt")
    dataset_df, target_mapping, load_warnings = load_dataset(config, logger)
    for warning_message in load_warnings:
        logger.warning("Load warning: %s", warning_message)

    fold_results, prediction_df = run_lodo_evaluation(dataset_df, config, logger)
    model_summary, domain_summary, overall_domain = aggregate_results(fold_results)
    pairwise_results, pairwise_prediction_df = run_pairwise_site_transfer(
        dataset_df, config, logger
    )
    pairwise_model_summary, pairwise_test_summary = aggregate_pairwise_results(
        pairwise_results
    )
    save_outputs(
        config=config,
        target_mapping=target_mapping,
        dataset_df=dataset_df,
        fold_results=fold_results,
        prediction_df=prediction_df,
        pairwise_results=pairwise_results,
        pairwise_prediction_df=pairwise_prediction_df,
        model_summary=model_summary,
        domain_summary=domain_summary,
        overall_domain=overall_domain,
        pairwise_model_summary=pairwise_model_summary,
        pairwise_test_summary=pairwise_test_summary,
        output_dir=output_dir,
    )
    logger.info("Finished baseline M0 experiment for %s", config.dataset_name)
    return {
        "output_dir": output_dir,
        "fold_results": output_dir / "fold_results.csv",
        "model_summary": output_dir / "model_summary.csv",
        "domain_summary": output_dir / "domain_summary.csv",
        "pairwise_results": output_dir / "train_test_site_results.csv",
        "report": output_dir / "baseline_report.md",
    }


DOMAIN_DISPLAY = {
    "cleveland": "Cleveland",
    "hungarian": "Hungarian",
    "long_beach_va": "VA",
    "switzerland": "Switzerland",
}

MODEL_DISPLAY = {
    "LogisticRegression": "logistic",
    "RandomForest": "random_forest",
    "HistGradientBoosting": "hist_gradient_boosting",
}

DISPLAY_DOMAIN_ORDER = ["Cleveland", "Hungarian", "Switzerland", "VA"]
DISPLAY_MODEL_ORDER = ["logistic", "random_forest", "hist_gradient_boosting"]


def pretty_domain_name(domain: str) -> str:
    return DOMAIN_DISPLAY.get(str(domain), str(domain))


def pretty_model_name(model: str) -> str:
    return MODEL_DISPLAY.get(str(model), str(model))


def ordered_domain_labels(values: list[str]) -> list[str]:
    if set(values).issubset(set(DISPLAY_DOMAIN_ORDER)):
        return [site for site in DISPLAY_DOMAIN_ORDER if site in values]
    return sorted(values)


def performance_cmap():
    return plt.get_cmap("RdYlGn")


def gap_cmap():
    return plt.get_cmap("RdYlGn_r")


def annotate_bars(ax: plt.Axes, bars) -> None:
    for bar in bars:
        value = bar.get_height()
        if not np.isfinite(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_grouped_bar_redgreen(
    plot_df: pd.DataFrame,
    *,
    value_col: str,
    x_col: str,
    group_col: str,
    title: str,
    ylabel: str,
    output_path_no_suffix: Path,
    save_pdf_flag: bool,
    good_high: bool = True,
    y_limits: tuple[float, float] | None = None,
) -> None:
    if plot_df.empty or value_col not in plot_df.columns:
        return
    pivot = plot_df.pivot_table(
        index=x_col, columns=group_col, values=value_col, aggfunc="mean"
    )
    if pivot.empty:
        return

    if x_col == "test_domain_display":
        pivot = pivot.reindex(ordered_domain_labels(pivot.index.tolist()))
    if group_col == "model_display":
        pivot = pivot.reindex(
            columns=[model for model in DISPLAY_MODEL_ORDER if model in pivot.columns]
        )

    values = pivot.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return

    cmap = performance_cmap() if good_high else gap_cmap()
    vmin = 0.0 if y_limits and y_limits[0] == 0.0 else float(finite_values.min())
    vmax = 1.0 if y_limits and y_limits[1] == 1.0 else float(finite_values.max())
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    x_positions = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.index) * 1.4), 6.5))
    for model_idx, model_name in enumerate(pivot.columns):
        series = pivot[model_name].to_numpy(dtype=float)
        colors_for_bars = cmap(norm(np.nan_to_num(series, nan=vmin)))
        bars = ax.bar(
            x_positions + model_idx * width - (len(pivot.columns) - 1) * width / 2,
            series,
            width,
            label=model_name,
            color=colors_for_bars,
            edgecolor="black",
            linewidth=0.7,
        )
        annotate_bars(ax, bars)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel(
        "Holdout Site"
        if x_col == "test_domain_display"
        else x_col.replace("_", " ").title(),
        fontsize=14,
    )
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(pivot.index.tolist(), rotation=25, ha="right")
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model", frameon=False)
    save_figure(fig, output_path_no_suffix, save_pdf_flag)


def heatmap_with_annotations(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    title: str,
    cmap,
    vmin: float,
    vmax: float,
    fmt: str = ".3f",
) -> any:
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = "NA" if np.isnan(value) else format(value, fmt)
            ax.text(j, i, label, ha="center", va="center", color="black", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#888888")
    return image


def run_training_domain_experiments(
    df: pd.DataFrame,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = build_models(config, logger)
    domains = sorted(df[config.domain_col].dropna().astype(str).unique().tolist())
    rows: list[dict[str, Any]] = []

    for test_domain in domains:
        source_domains = [domain for domain in domains if domain != test_domain]
        test_df = df.loc[df[config.domain_col].astype(str) == test_domain].reset_index(
            drop=True
        )
        if "TCGA" in config.dataset_name.upper() and len(source_domains) >= 4:
            candidate_train_sizes = sorted(set([1, 3, len(source_domains)]))
        else:
            candidate_train_sizes = list(range(1, len(source_domains) + 1))
        for n_train_sites in candidate_train_sizes:
            for combo in combinations(source_domains, n_train_sites):
                train_df = df.loc[
                    df[config.domain_col].astype(str).isin(combo)
                ].reset_index(drop=True)
                train_sites_label = " + ".join(
                    pretty_domain_name(domain) for domain in combo
                )
                for model_name, estimator in models.items():
                    warnings_out: list[str] = []
                    row = {
                        "test_site": pretty_domain_name(test_domain),
                        "test_site_raw": test_domain,
                        "train_sites": train_sites_label,
                        "train_sites_raw": "|".join(combo),
                        "n_train_sites": n_train_sites,
                        "model": pretty_model_name(model_name),
                        "n_train_samples": int(len(train_df)),
                        "n_test_samples": int(len(test_df)),
                        "positive_rate_train": float(
                            train_df[config.target_col].mean()
                        ),
                        "positive_rate_test": float(test_df[config.target_col].mean()),
                    }

                    if train_df[config.target_col].nunique() < 2:
                        warnings_out.append("single_class_in_training")
                        row.update(
                            {
                                "accuracy": math.nan,
                                "balanced_accuracy": math.nan,
                                "auc": math.nan,
                                "f1": math.nan,
                                "brier": math.nan,
                            }
                        )
                        row["issues"] = aggregate_warning_list(warnings_out)
                        rows.append(row)
                        continue

                    spec = prepare_columns(train_df, config)
                    warnings_out.extend(spec.warnings)
                    if not spec.feature_cols:
                        warnings_out.append("no_usable_features")
                        row.update(
                            {
                                "accuracy": math.nan,
                                "balanced_accuracy": math.nan,
                                "auc": math.nan,
                                "f1": math.nan,
                                "brier": math.nan,
                            }
                        )
                        row["issues"] = aggregate_warning_list(warnings_out)
                        rows.append(row)
                        continue

                    pipeline = build_pipeline(spec, estimator, config)
                    try:
                        pipeline, fit_warnings = fit_with_warnings(
                            pipeline, train_df, train_df[config.target_col].astype(int)
                        )
                        warnings_out.extend(fit_warnings)
                        y_prob = safe_predict_proba(pipeline, test_df)
                        metric_values, metric_warnings = compute_metrics(
                            test_df[config.target_col].to_numpy(dtype=int),
                            y_prob,
                            config=config,
                        )
                        warnings_out.extend(metric_warnings)
                    except Exception as exc:
                        warnings_out.append(f"fit_or_predict_failed:{exc}")
                        metric_values = {
                            "accuracy": math.nan,
                            "balacc": math.nan,
                            "auc": math.nan,
                            "f1": math.nan,
                            "brier": math.nan,
                        }

                    row["accuracy"] = metric_values.get("accuracy", math.nan)
                    row["balanced_accuracy"] = metric_values.get("balacc", math.nan)
                    row["auc"] = metric_values.get("auc", math.nan)
                    row["f1"] = metric_values.get("f1", math.nan)
                    row["brier"] = metric_values.get("brier", math.nan)
                    row["issues"] = aggregate_warning_list(warnings_out)
                    rows.append(row)

    combo_results = pd.DataFrame(rows)
    summary_by_model = (
        combo_results.groupby(["model", "n_train_sites"], dropna=False)
        .agg(
            n_experiments=("model", "size"),
            mean_accuracy=("accuracy", "mean"),
            sd_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            sd_balanced_accuracy=("balanced_accuracy", "std"),
            mean_auc=("auc", "mean"),
            sd_auc=("auc", "std"),
        )
        .reset_index()
    )
    summary_by_site = (
        combo_results.groupby(["test_site", "model", "n_train_sites"], dropna=False)
        .agg(
            n_experiments=("model", "size"),
            mean_accuracy=("accuracy", "mean"),
            sd_accuracy=("accuracy", "std"),
            mean_auc=("auc", "mean"),
            sd_auc=("auc", "std"),
        )
        .reset_index()
    )
    return combo_results, summary_by_model, summary_by_site


def plot_site_model_ood_heatmap(
    fold_results: pd.DataFrame,
    final_dir: Path,
    save_pdf_flag: bool,
) -> None:
    plot_df = fold_results.copy()
    plot_df["test_domain_display"] = plot_df["test_domain"].map(pretty_domain_name)
    plot_df["model_display"] = plot_df["model"].map(pretty_model_name)
    metrics = [
        ("auc_ood", "auc_ood"),
        ("balacc_ood", "balacc_ood"),
        ("f1_ood", "f1_ood"),
    ]
    row_order = list(
        reversed(
            ordered_domain_labels(
                plot_df["test_domain_display"].dropna().unique().tolist()
            )
        )
    )
    col_order = [
        model
        for model in DISPLAY_MODEL_ORDER
        if model in plot_df["model_display"].unique()
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 8), constrained_layout=True)
    fig.suptitle("Heatmap: site x model OOD performance", fontsize=22, y=1.02)
    all_values = []
    for metric_col, _ in metrics:
        all_values.extend(plot_df[metric_col].dropna().tolist())
    vmin, vmax = 0.0, 1.0 if all_values else 1.0

    image = None
    for ax, (metric_col, metric_title) in zip(axes, metrics):
        pivot = plot_df.pivot_table(
            index="test_domain_display",
            columns="model_display",
            values=metric_col,
            aggfunc="mean",
        ).reindex(index=row_order, columns=col_order)
        image = heatmap_with_annotations(
            ax,
            pivot.to_numpy(dtype=float),
            pivot.index.tolist(),
            pivot.columns.tolist(),
            title=metric_title,
            cmap=performance_cmap(),
            vmin=vmin,
            vmax=vmax,
        )
        if ax is axes[0]:
            ax.set_ylabel("Holdout site", fontsize=18)
    cbar = fig.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.6, location="bottom", pad=0.08
    )
    cbar.set_label("Value", fontsize=13)
    save_figure(fig, final_dir / "heatmap_site_model_ood", save_pdf_flag)


def plot_training_domain_all_combinations_heatmap(
    combo_results: pd.DataFrame,
    final_dir: Path,
) -> None:
    test_sites = ordered_domain_labels(
        combo_results["test_site"].dropna().unique().tolist()
    )
    model_order = [
        model
        for model in DISPLAY_MODEL_ORDER
        if model in combo_results["model"].unique()
    ]
    fig, axes = plt.subplots(
        len(test_sites), 1, figsize=(12, 3.3 * len(test_sites)), constrained_layout=True
    )
    if len(test_sites) == 1:
        axes = [axes]
    fig.suptitle(
        "Exhaustive training-site combinations for each held-out test site",
        fontsize=18,
        y=1.02,
    )
    image = None
    for ax, test_site in zip(axes, test_sites):
        subset = combo_results.loc[combo_results["test_site"] == test_site].copy()
        subset["row_label"] = subset.apply(
            lambda row: (
                f"Train {int(row['n_train_sites'])} site(s) | {row['train_sites']}"
            ),
            axis=1,
        )
        row_order = (
            subset[["row_label", "n_train_sites", "train_sites"]]
            .drop_duplicates()
            .sort_values(["n_train_sites", "train_sites"], ascending=[False, True])[
                "row_label"
            ]
            .tolist()
        )
        pivot = subset.pivot_table(
            index="row_label", columns="model", values="accuracy", aggfunc="mean"
        ).reindex(index=row_order, columns=model_order)
        image = heatmap_with_annotations(
            ax,
            pivot.to_numpy(dtype=float),
            pivot.index.tolist(),
            pivot.columns.tolist(),
            title=test_site,
            cmap=performance_cmap(),
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_ylabel("Training combination", fontsize=10)
    cbar = fig.colorbar(image, ax=axes, shrink=0.5, location="bottom", pad=0.04)
    cbar.set_label("Accuracy", fontsize=12)
    save_figure(
        fig,
        final_dir / "transportability_training_domains_all_combinations_heatmap",
        False,
    )


def plot_training_domains_by_site_heatmap(
    combo_results: pd.DataFrame,
    final_dir: Path,
) -> None:
    summary = (
        combo_results.groupby(["test_site", "model", "n_train_sites"], dropna=False)[
            "accuracy"
        ]
        .mean()
        .reset_index(name="mean_accuracy")
    )
    test_sites = list(
        reversed(ordered_domain_labels(summary["test_site"].dropna().unique().tolist()))
    )
    model_order = [
        model for model in DISPLAY_MODEL_ORDER if model in summary["model"].unique()
    ]
    n_train_values = sorted(summary["n_train_sites"].dropna().unique().tolist())

    fig, axes = plt.subplots(
        1,
        len(model_order),
        figsize=(5.5 * len(model_order), 8),
        constrained_layout=True,
    )
    if len(model_order) == 1:
        axes = [axes]
    fig.suptitle(
        "Transportability improvement by held-out site as training domains increase",
        fontsize=20,
        y=1.02,
    )
    image = None
    for ax, model in zip(axes, model_order):
        subset = summary.loc[summary["model"] == model]
        pivot = subset.pivot_table(
            index="test_site",
            columns="n_train_sites",
            values="mean_accuracy",
            aggfunc="mean",
        ).reindex(index=test_sites, columns=n_train_values)
        image = heatmap_with_annotations(
            ax,
            pivot.to_numpy(dtype=float),
            pivot.index.tolist(),
            [str(value) for value in pivot.columns.tolist()],
            title=model,
            cmap=performance_cmap(),
            vmin=0.0,
            vmax=1.0,
        )
        if ax is axes[0]:
            ax.set_ylabel("Held-out test site", fontsize=18)
        ax.set_xlabel("Number of training sites", fontsize=12)
    cbar = fig.colorbar(image, ax=axes, shrink=0.55, location="bottom", pad=0.05)
    cbar.set_label("Mean accuracy", fontsize=12)
    save_figure(
        fig, final_dir / "transportability_training_domains_by_site_heatmap", False
    )


def plot_training_domains_heatmap(
    summary_by_model: pd.DataFrame,
    final_dir: Path,
) -> None:
    plot_df = summary_by_model.copy()
    plot_df["model"] = plot_df["model"].map(pretty_model_name)
    model_order = [
        model
        for model in reversed(DISPLAY_MODEL_ORDER)
        if model in plot_df["model"].unique()
    ]
    n_train_values = sorted(plot_df["n_train_sites"].dropna().unique().tolist())
    pivot = plot_df.pivot_table(
        index="model", columns="n_train_sites", values="mean_accuracy", aggfunc="mean"
    ).reindex(index=model_order, columns=n_train_values)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    image = heatmap_with_annotations(
        ax,
        pivot.to_numpy(dtype=float),
        pivot.index.tolist(),
        [str(value) for value in pivot.columns.tolist()],
        title="Effect of training domain diversity on transportability",
        cmap=performance_cmap(),
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Number of training sites", fontsize=14)
    ax.set_ylabel("Model", fontsize=14)
    cbar = fig.colorbar(image, ax=ax, shrink=0.72, location="bottom", pad=0.12)
    cbar.set_label("Mean accuracy", fontsize=12)
    save_figure(fig, final_dir / "transportability_training_domains_heatmap", False)


def plot_pairwise_heatmaps_flat(
    pairwise_results: pd.DataFrame,
    final_dir: Path,
    save_pdf_flag: bool,
) -> None:
    plot_df = pairwise_results.copy()
    plot_df["train_domain_display"] = plot_df["train_domain"].map(pretty_domain_name)
    plot_df["test_domain_display"] = plot_df["test_domain"].map(pretty_domain_name)
    plot_df["model_display"] = plot_df["model"].map(pretty_model_name)

    row_order = ordered_domain_labels(
        plot_df["train_domain_display"].dropna().unique().tolist()
    )
    col_order = ordered_domain_labels(
        plot_df["test_domain_display"].dropna().unique().tolist()
    )
    model_order = [
        model
        for model in DISPLAY_MODEL_ORDER
        if model in plot_df["model_display"].unique()
    ]

    for metric_col, prefix in [("auc", "auc"), ("f1", "f1")]:
        for model in model_order:
            subset = plot_df.loc[plot_df["model_display"] == model]
            pivot = subset.pivot_table(
                index="train_domain_display",
                columns="test_domain_display",
                values=metric_col,
                aggfunc="mean",
            ).reindex(index=row_order, columns=col_order)
            fig, ax = plt.subplots(figsize=(8, 6.3))
            image = heatmap_with_annotations(
                ax,
                pivot.to_numpy(dtype=float),
                pivot.index.tolist(),
                pivot.columns.tolist(),
                title=f"Train site to test site ({prefix.upper()}): {model}",
                cmap=performance_cmap(),
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_xlabel("Test site", fontsize=13)
            ax.set_ylabel("Train site", fontsize=13)
            cbar = fig.colorbar(image, ax=ax, shrink=0.82)
            cbar.set_label(prefix.upper(), fontsize=11)
            save_figure(
                fig, final_dir / f"train_test_heatmap_{prefix}_{model}", save_pdf_flag
            )


def plot_simple_heatmap_flat(
    fold_results: pd.DataFrame,
    metric_column: str,
    title: str,
    output_path_no_suffix: Path,
    save_pdf_flag: bool,
) -> None:
    plot_df = fold_results.copy()
    plot_df["test_domain_display"] = plot_df["test_domain"].map(pretty_domain_name)
    plot_df["model_display"] = plot_df["model"].map(pretty_model_name)
    pivot = plot_df.pivot_table(
        index="model_display",
        columns="test_domain_display",
        values=metric_column,
        aggfunc="mean",
    ).reindex(
        index=[
            model
            for model in DISPLAY_MODEL_ORDER
            if model in plot_df["model_display"].unique()
        ],
        columns=ordered_domain_labels(
            plot_df["test_domain_display"].dropna().unique().tolist()
        ),
    )
    cmap = gap_cmap() if metric_column.startswith("gap_") else performance_cmap()
    upper = (
        1.0
        if not metric_column.startswith("gap_")
        else float(np.nanmax(pivot.to_numpy(dtype=float)))
    )
    lower = (
        0.0
        if not metric_column.startswith("gap_")
        else float(np.nanmin(pivot.to_numpy(dtype=float)))
    )
    fig, ax = plt.subplots(figsize=(8, 5.2))
    image = heatmap_with_annotations(
        ax,
        pivot.to_numpy(dtype=float),
        pivot.index.tolist(),
        pivot.columns.tolist(),
        title=title,
        cmap=cmap,
        vmin=lower,
        vmax=upper,
    )
    ax.set_xlabel("Holdout site", fontsize=13)
    ax.set_ylabel("Model", fontsize=13)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label(metric_column, fontsize=10)
    save_figure(fig, output_path_no_suffix, save_pdf_flag)


def write_html_report(markdown_path: Path, html_path: Path) -> None:
    markdown_text = markdown_path.read_text(encoding="utf-8")
    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>UCI Baseline M0 Report</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem auto; max-width: 1100px; line-height: 1.5; color: #222; }}
    pre {{ white-space: pre-wrap; background: #f7f7f7; padding: 1rem; border: 1px solid #ddd; }}
  </style>
</head>
<body>
<pre>{escape(markdown_text)}</pre>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")


def write_readme(
    final_dir: Path,
    *,
    config: ExperimentConfig,
    model_summary: pd.DataFrame,
    combo_summary: pd.DataFrame,
) -> None:
    best_auc_model = (
        model_summary.sort_values("auc_ood_mean", ascending=False).iloc[0]["model"]
        if not model_summary.empty
        else "NA"
    )
    best_accuracy_model = (
        model_summary.sort_values("accuracy_ood_mean", ascending=False).iloc[0]["model"]
        if not model_summary.empty
        else "NA"
    )
    is_tcga = "TCGA" in config.dataset_name.upper()
    title = "TCGA-BRCA Baseline M0" if is_tcga else "UCI Heart Disease Baseline M0"
    intro = (
        "This folder contains a full, single-script TCGA-BRCA baseline rebuild for cross-site transportability."
        if is_tcga
        else "This folder contains a full, single-script UCI baseline rebuild for cross-site transportability."
    )
    script_name = "tcga_m0_uniform.py" if is_tcga else "uci_m0_uniform.py"
    extra_notes = (
        [
            "- The raw TCGA input was restricted to `tumor_level in {1, 11}`.",
            "- `tumor_level = 6` was excluded because it contains only 7 samples overall.",
            "- Sites without both binary classes were excluded before modeling.",
            "- Feature selection is performed inside each training fold only.",
            "- Multisite transportability heatmaps summarize grouped training-domain sizes (1, 3, and all remaining source sites).",
        ]
        if is_tcga
        else [
            "- Leave-One-Domain-Out evaluation for in-domain vs out-of-domain performance.",
            "- Train-site to test-site transfer heatmaps based on single-site training.",
            "- Exhaustive training-domain combination analysis across 1, 2, and 3 source sites.",
        ]
    )
    lines = [
        f"# {title}",
        "",
        intro,
        "",
        "## Input",
        f"- Dataset: `{config.input_path}`",
        f"- Target: `{config.target_col}`",
        f"- Domain: `{config.domain_col}`",
        "",
        "## What Was Done",
        *extra_notes,
        "- Leakage-safe preprocessing inside the training partition only.",
        "- Red-to-green thesis-ready plots with metric axes fixed to 1.0 where appropriate.",
        "",
        "## Main Results",
        f"- Best mean OOD AUC model: `{best_auc_model}`",
        f"- Best mean OOD accuracy model: `{best_accuracy_model}`",
        f"- Number of model summaries: `{len(model_summary)}`",
        f"- Number of training-domain diversity summaries: `{len(combo_summary)}`",
        "",
        "## Main Files",
        f"- `{script_name}`: single reproducible Python script.",
        "- `baseline_report.md`: detailed results report.",
        "- `model_summary.csv`: aggregate model-level results.",
        "- `prediction_records.csv`: prediction-level outputs.",
        "- PNG/PDF heatmaps and barplots ready for thesis use.",
        "",
        "## Run",
        "```bash",
        f'cd "{final_dir}"',
        f"python3 {script_name}",
        "```",
    ]
    (final_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_tcga_binary_input(
    raw_input_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    df = pd.read_csv(raw_input_path)
    original_shape = df.shape
    df = df.loc[df["tumor_level"].isin([1, 11])].copy()
    df["target_binary"] = (df["tumor_level"] == 1).astype(int)

    valid_sites = (
        df.groupby("site")["target_binary"]
        .nunique()
        .reset_index(name="n_classes")
        .query("n_classes == 2")["site"]
        .tolist()
    )
    df = df.loc[df["site"].isin(valid_sites)].copy()
    metadata_cols = [
        "sample_barcode",
        "subject_id",
        "site",
        "tumor_level",
        "sample_type_label",
        "domain_primary",
        "domain_source",
        "target_binary",
    ]
    df.to_csv(output_path, index=False)
    return {
        "original_shape": original_shape,
        "filtered_shape": df.shape,
        "valid_sites": sorted(valid_sites),
        "target_counts": df["target_binary"].value_counts().to_dict(),
        "gene_count_retained_for_foldwise_selection": int(
            df.shape[1] - len(metadata_cols)
        ),
    }


def create_thesis_plots_and_tables(
    *,
    build_dir: Path,
    final_dir: Path,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> None:
    ensure_directory(final_dir)
    for file_name in [
        "baseline_report.md",
        "model_summary.csv",
        "prediction_records.csv",
        "calibration_curve_data.csv",
    ]:
        shutil.copy2(build_dir / file_name, final_dir / file_name)

    fold_results = pd.read_csv(build_dir / "fold_results.csv")
    model_summary = pd.read_csv(build_dir / "model_summary.csv")
    pairwise_results = pd.read_csv(build_dir / "train_test_site_results.csv")
    dataset_df, _, _ = load_dataset(config, logger)
    combo_results, combo_summary_by_model, combo_summary_by_site = (
        run_training_domain_experiments(dataset_df, config, logger)
    )
    combo_results.to_csv(
        final_dir / "training_domain_experiment_results.csv", index=False
    )
    combo_summary_by_model.to_csv(
        final_dir / "training_domain_experiment_summary_by_model.csv", index=False
    )
    combo_summary_by_site.to_csv(
        final_dir / "training_domain_experiment_summary_by_site.csv", index=False
    )

    plot_df = fold_results.copy()
    plot_df["test_domain_display"] = plot_df["test_domain"].map(pretty_domain_name)
    plot_df["model_display"] = plot_df["model"].map(pretty_model_name)
    model_summary_plot = model_summary.copy()
    model_summary_plot["model_display"] = model_summary_plot["model"].map(
        pretty_model_name
    )

    plot_grouped_bar_redgreen(
        plot_df,
        value_col="auc_ood",
        x_col="test_domain_display",
        group_col="model_display",
        title="Out-of-domain ROC-AUC by holdout site and model",
        ylabel="ROC-AUC",
        output_path_no_suffix=final_dir / "auc_ood_by_domain",
        save_pdf_flag=False,
        good_high=True,
        y_limits=(0.0, 1.0),
    )
    plot_grouped_bar_redgreen(
        plot_df,
        value_col="accuracy_ood",
        x_col="test_domain_display",
        group_col="model_display",
        title="Out-of-domain accuracy by holdout site and model",
        ylabel="Accuracy",
        output_path_no_suffix=final_dir / "accuracy_ood_by_domain",
        save_pdf_flag=False,
        good_high=True,
        y_limits=(0.0, 1.0),
    )
    plot_grouped_bar_redgreen(
        plot_df,
        value_col="f1_ood",
        x_col="test_domain_display",
        group_col="model_display",
        title="Out-of-domain F1-score by holdout site and model",
        ylabel="F1-score",
        output_path_no_suffix=final_dir / "f1_ood_by_domain",
        save_pdf_flag=True,
        good_high=True,
        y_limits=(0.0, 1.0),
    )
    plot_grouped_bar_redgreen(
        plot_df,
        value_col="gap_auc",
        x_col="test_domain_display",
        group_col="model_display",
        title="ID-OOD ROC-AUC gap by holdout site and model",
        ylabel="Gap (ID - OOD)",
        output_path_no_suffix=final_dir / "gap_auc_by_domain",
        save_pdf_flag=False,
        good_high=False,
        y_limits=None,
    )

    mean_perf_df = model_summary_plot.melt(
        id_vars=["model_display"],
        value_vars=[
            "auc_ood_mean",
            "accuracy_ood_mean",
            "f1_ood_mean",
            "balacc_ood_mean",
        ],
        var_name="metric_label",
        value_name="value",
    )
    mean_perf_df["metric_label"] = (
        mean_perf_df["metric_label"]
        .str.replace("_ood_mean", "", regex=False)
        .str.upper()
    )
    plot_grouped_bar_redgreen(
        mean_perf_df,
        value_col="value",
        x_col="metric_label",
        group_col="model_display",
        title="Mean out-of-domain performance by model",
        ylabel="Score",
        output_path_no_suffix=final_dir / "mean_performance_by_model",
        save_pdf_flag=True,
        good_high=True,
        y_limits=(0.0, 1.0),
    )

    plot_simple_heatmap_flat(
        fold_results,
        "auc_ood",
        "Heatmap: OOD ROC-AUC",
        final_dir / "heatmap_auc_ood",
        True,
    )
    plot_simple_heatmap_flat(
        fold_results,
        "accuracy_ood",
        "Heatmap: OOD accuracy",
        final_dir / "heatmap_accuracy_ood",
        True,
    )
    plot_simple_heatmap_flat(
        fold_results,
        "f1_ood",
        "Heatmap: OOD F1-score",
        final_dir / "heatmap_f1_ood",
        True,
    )
    plot_site_model_ood_heatmap(fold_results, final_dir, True)
    plot_pairwise_heatmaps_flat(pairwise_results, final_dir, True)
    plot_training_domain_all_combinations_heatmap(combo_results, final_dir)
    plot_training_domains_by_site_heatmap(combo_results, final_dir)
    plot_training_domains_heatmap(combo_summary_by_model, final_dir)

    calibration_df = pd.read_csv(build_dir / "calibration_curve_data.csv")
    for model_name in ["logisticregression", "randomforest", "histgradientboosting"]:
        source_png = build_dir / "plots" / f"calibration_{model_name}.png"
        if source_png.exists():
            shutil.copy2(source_png, final_dir / source_png.name)

    write_html_report(
        final_dir / "baseline_report.md", final_dir / "baseline_report.html"
    )
    write_readme(
        final_dir,
        config=config,
        model_summary=model_summary,
        combo_summary=combo_summary_by_model,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent
    build_dir = project_root / "_build_tcga_safe"
    final_dir = project_root / "output_flat_tcga_safe"
    prepared_input_path = build_dir / "tcga_modeling_input.csv"
    ensure_directory(build_dir)
    tcga_prep_info = prepare_tcga_binary_input(
        _configured_path("legacy_preprocessed_tcga_tcga_brca_preprossed.csv"),
        prepared_input_path,
    )
    config = ExperimentConfig(
        dataset_name="TCGA_BRCA_01_vs_11_SitesWithBothClasses",
        input_path=str(prepared_input_path),
        output_dir=str(build_dir),
        target_col="target_binary",
        domain_col="site",
        id_cols=["sample_barcode", "subject_id"],
        extra_exclude_cols=[
            "tumor_level",
            "sample_type_label",
            "domain_primary",
            "domain_source",
        ],
        force_categorical_cols=[],
        force_numeric_cols=[],
        random_seed=42,
        n_jobs=1,
        enable_xgboost=True,
        enable_ece=True,
        enable_calibration_plots=True,
        calibration_bins=10,
        min_domain_size=20,
        save_pdf_plots=True,
        low_cardinality_as_categorical=True,
        max_low_cardinality_unique=10,
        auto_exclude_id_like=True,
        feature_selection_method="selectkbest",
        feature_selection_top_k=200,
        variance_threshold=0.0,
        pca_n_components=None,
        enable_pairwise_site_transfer=True,
        within_domain_cv_splits=5,
    )
    run_experiment(config)
    logger = init_logger(build_dir / "packaging_logs.txt")
    create_thesis_plots_and_tables(
        build_dir=build_dir, final_dir=final_dir, config=config, logger=logger
    )
    pd.DataFrame([tcga_prep_info]).to_csv(
        final_dir / "tcga_filtering_summary.csv", index=False
    )
    shutil.copy2(prepared_input_path, final_dir / prepared_input_path.name)
    shutil.copy2(Path(__file__), final_dir / Path(__file__).name)


if __name__ == "__main__":
    main()
