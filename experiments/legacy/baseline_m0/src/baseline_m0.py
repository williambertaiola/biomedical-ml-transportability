"""Transportability experiment: baseline m0."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
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


def build_preprocessor(spec: ColumnSpec) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if spec.numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
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
            max_iter=5000,
            random_state=config.random_seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
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
            max_iter=300,
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
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(spec)),
            ("selector", build_selector(config)),
            ("model", clone(estimator)),
        ]
    )


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
        n_splits = min(5, len(unique_groups))
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(X_train, y_train, groups=groups_train))
        return splits, "GroupKFold", warnings_out

    warnings_out.append(
        "inner_id_groupkfold_unavailable_single_source_domain_using_stratifiedkfold_fallback"
    )
    class_counts = y_train.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    if y_train.nunique() >= 2 and min_class_count >= 2:
        n_splits = min(5, min_class_count)
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=config.random_seed
        )
        splits = list(splitter.split(X_train, y_train))
        return splits, "StratifiedKFoldFallback", warnings_out

    warnings_out.append("inner_id_unavailable")
    return [], "Unavailable", warnings_out


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
    model_summary: pd.DataFrame,
    domain_summary: pd.DataFrame,
    overall_domain: pd.DataFrame,
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
        "## Warning Summary",
        f"- Outer model-folds with non-computable OOD AUC: `{invalid_outer}`",
        f"- Total warnings recorded: `{int(fold_results['warnings'].fillna('').ne('').sum())}`",
        "",
        "## Automatic Comment",
        *comment_lines,
        "",
        "## Methodological Notes",
        "- No harmonization, domain adaptation, or reweighting was applied in this baseline M0 run.",
        "- Calibration is summarized with Brier score and, when enabled, Expected Calibration Error (ECE).",
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
    model_summary: pd.DataFrame,
    domain_summary: pd.DataFrame,
    overall_domain: pd.DataFrame,
    output_dir: Path,
) -> None:
    ensure_directory(output_dir)
    ensure_directory(output_dir / "plots")

    fold_results.to_csv(output_dir / "fold_results.csv", index=False)
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    domain_summary.to_csv(output_dir / "domain_summary.csv", index=False)
    overall_domain.to_csv(output_dir / "domain_difficulty_summary.csv", index=False)
    prediction_df.to_csv(output_dir / "prediction_records.csv", index=False)

    calibration_df = build_calibration_dataframe(prediction_df, config)
    calibration_df.to_csv(output_dir / "calibration_curve_data.csv", index=False)
    make_plots(
        fold_results, model_summary, overall_domain, calibration_df, output_dir, config
    )
    write_report(
        config=config,
        target_mapping=target_mapping,
        dataset_df=dataset_df,
        fold_results=fold_results,
        model_summary=model_summary,
        domain_summary=domain_summary,
        overall_domain=overall_domain,
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
    save_outputs(
        config=config,
        target_mapping=target_mapping,
        dataset_df=dataset_df,
        fold_results=fold_results,
        prediction_df=prediction_df,
        model_summary=model_summary,
        domain_summary=domain_summary,
        overall_domain=overall_domain,
        output_dir=output_dir,
    )
    logger.info("Finished baseline M0 experiment for %s", config.dataset_name)
    return {
        "output_dir": output_dir,
        "fold_results": output_dir / "fold_results.csv",
        "model_summary": output_dir / "model_summary.csv",
        "domain_summary": output_dir / "domain_summary.csv",
        "report": output_dir / "baseline_report.md",
    }
