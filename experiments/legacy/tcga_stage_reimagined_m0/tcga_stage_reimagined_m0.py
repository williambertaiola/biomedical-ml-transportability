"""Transportability experiment: tcga stage reimagined m0."""

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


import argparse
import json
import logging
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from itertools import combinations, product
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT_PATH = _configured_path(
    "legacy_preprocessed_tcga_tcga_brca_preprossed.csv"
)
DEFAULT_CLINICAL_PATH = _configured_path("tcga_clinical")
DEFAULT_OUTPUT_DIR = str(Path(_configured_path("legacy_output")))


@dataclass
class ExperimentConfig:
    input_path: str = DEFAULT_INPUT_PATH
    clinical_path: str = DEFAULT_CLINICAL_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    random_seed: int = 42
    min_per_class_per_site: int = 5
    top_k_genes: int = 200
    inner_cv_splits: int = 3
    within_site_cv_splits: int = 3
    rf_trees: int = 50
    hgb_max_iter: int = 50
    diversity_k_values: list[int] = field(default_factory=lambda: [1, 3, 5])
    diversity_sampled_combinations: int = 3
    save_pdf_plots: bool = True
    enable_pairwise_heatmaps: bool = True
    enable_shuffle_test: bool = True
    pairwise_models: list[str] = field(default_factory=lambda: ["LogisticRegression"])
    diversity_models: list[str] = field(default_factory=lambda: ["LogisticRegression"])
    shuffle_models: list[str] = field(default_factory=lambda: ["LogisticRegression"])


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def init_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("tcga_stage_reimagined_m0")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(
        output_dir / "logs.txt", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return cleaned.lower() or "value"


def aggregate_warning_list(items: list[str]) -> str:
    cleaned = [str(item) for item in items if str(item).strip()]
    if not cleaned:
        return ""
    return " | ".join(dict.fromkeys(cleaned))


def read_tcga_expression(
    config: ExperimentConfig, logger: logging.Logger
) -> tuple[pd.DataFrame, list[str]]:
    header = pd.read_csv(config.input_path, nrows=0)
    columns = header.columns.tolist()
    gene_cols = [column for column in columns if column.startswith("ENSG")]
    meta_cols = [
        column
        for column in ["sample_barcode", "subject_id", "site", "tumor_level"]
        if column in columns
    ]
    usecols = meta_cols + gene_cols
    dtype_map = {column: "float32" for column in gene_cols}
    df = pd.read_csv(config.input_path, usecols=usecols, dtype=dtype_map)
    logger.info(
        "Loaded expression matrix with shape %s and %s gene features",
        df.shape,
        len(gene_cols),
    )
    return df, gene_cols


def _first_non_missing(series: pd.Series) -> Any:
    non_missing = series.dropna()
    if non_missing.empty:
        return pd.NA
    return non_missing.iloc[0]


def read_clinical_stage(
    config: ExperimentConfig, logger: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_csv(
        config.clinical_path,
        sep="\t",
        usecols=["cases.submitter_id", "diagnoses.ajcc_pathologic_stage"],
        low_memory=False,
    )
    clinical = clinical.replace("'--", pd.NA)
    conflict_df = (
        clinical.dropna(subset=["diagnoses.ajcc_pathologic_stage"])
        .groupby("cases.submitter_id")["diagnoses.ajcc_pathologic_stage"]
        .nunique()
        .reset_index(name="n_unique_stage_labels")
    )
    conflict_df = conflict_df[conflict_df["n_unique_stage_labels"] > 1].sort_values(
        ["n_unique_stage_labels", "cases.submitter_id"], ascending=[False, True]
    )
    dedup = (
        clinical.groupby("cases.submitter_id", as_index=False)[
            "diagnoses.ajcc_pathologic_stage"
        ]
        .agg(_first_non_missing)
        .rename(
            columns={
                "cases.submitter_id": "subject_id",
                "diagnoses.ajcc_pathologic_stage": "ajcc_pathologic_stage_raw",
            }
        )
    )
    logger.info(
        "Loaded clinical stage metadata for %s subjects (%s subjects with conflicting repeated stage rows)",
        len(dedup),
        len(conflict_df),
    )
    return dedup, conflict_df


def collapse_pathologic_stage(raw_value: Any) -> str | None:
    if pd.isna(raw_value):
        return None
    value = str(raw_value).strip()
    if re.match(r"^Stage\s+IV(?:\b|[A-C])", value):
        return "Stage III-IV"
    if re.match(r"^Stage\s+III(?:\b|[A-C])", value):
        return "Stage III-IV"
    if re.match(r"^Stage\s+II(?:\b|[A-C])", value):
        return "Stage I-II"
    if re.match(r"^Stage\s+I(?:\b|[A-C])", value):
        return "Stage I-II"
    return None


def prepare_tcga_stage_dataset(
    config: ExperimentConfig,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[str], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    expression_df, gene_cols = read_tcga_expression(config, logger)
    clinical_df, conflict_df = read_clinical_stage(config, logger)

    audit: dict[str, Any] = {
        "n_expression_rows_raw": int(len(expression_df)),
        "n_gene_features_raw": int(len(gene_cols)),
    }

    expression_df = expression_df.loc[expression_df["tumor_level"] == 1].copy()
    audit["n_primary_tumor_rows_before_subject_dedup"] = int(len(expression_df))

    duplicate_subject_df = (
        expression_df.groupby("subject_id")
        .agg(n_primary_samples=("sample_barcode", "nunique"))
        .reset_index()
        .query("n_primary_samples > 1")
        .sort_values(["n_primary_samples", "subject_id"], ascending=[False, True])
    )
    audit["n_subjects_with_multiple_primary_samples"] = int(len(duplicate_subject_df))

    expression_df = (
        expression_df.sort_values(["subject_id", "sample_barcode"])
        .drop_duplicates("subject_id", keep="first")
        .reset_index(drop=True)
    )
    audit["n_primary_subject_level_rows"] = int(len(expression_df))

    merged = expression_df.merge(clinical_df, on="subject_id", how="left")
    merged["target_label"] = merged["ajcc_pathologic_stage_raw"].map(
        collapse_pathologic_stage
    )
    merged["target"] = merged["target_label"].map({"Stage I-II": 0, "Stage III-IV": 1})
    audit["n_rows_with_mapped_stage_target"] = int(merged["target"].notna().sum())

    pre_site_filter = merged.dropna(subset=["target"]).copy()
    pre_site_filter["target"] = pre_site_filter["target"].astype(int)
    site_target_counts = pd.crosstab(
        pre_site_filter["site"], pre_site_filter["target_label"]
    )
    site_target_counts = site_target_counts.sort_index()
    valid_sites = site_target_counts.index[
        site_target_counts.min(axis=1) >= config.min_per_class_per_site
    ].tolist()
    audit["n_sites_before_stage_filter"] = int(pre_site_filter["site"].nunique())
    audit["n_valid_sites_after_min_class_filter"] = int(len(valid_sites))
    audit["valid_sites"] = valid_sites

    modeling_df = (
        pre_site_filter.loc[pre_site_filter["site"].isin(valid_sites)]
        .copy()
        .reset_index(drop=True)
    )
    modeling_df["target"] = modeling_df["target"].astype(int)
    modeling_df["target_label"] = modeling_df["target_label"].astype("string")

    audit["n_modeling_rows"] = int(len(modeling_df))
    audit["class_counts"] = modeling_df["target_label"].value_counts().to_dict()
    audit["site_counts"] = modeling_df["site"].value_counts().sort_index().to_dict()
    audit["positive_rate_global"] = float(modeling_df["target"].mean())

    site_summary = (
        modeling_df.groupby("site")
        .agg(
            n_samples=("subject_id", "size"),
            n_subjects=("subject_id", "nunique"),
            n_early=("target", lambda s: int((s == 0).sum())),
            n_late=("target", lambda s: int((s == 1).sum())),
            positive_rate=("target", "mean"),
        )
        .reset_index()
        .sort_values(["n_samples", "site"], ascending=[False, True])
        .reset_index(drop=True)
    )

    logger.info(
        "Prepared stage dataset: %s subjects across %s sites (class counts: %s)",
        len(modeling_df),
        modeling_df["site"].nunique(),
        audit["class_counts"],
    )
    return (
        modeling_df,
        gene_cols,
        audit,
        site_summary,
        duplicate_subject_df,
        conflict_df,
    )


def run_leakage_checks(df: pd.DataFrame, gene_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    suspicious_patterns = [
        (r"sample", "sample_identifier_like"),
        (r"subject", "subject_identifier_like"),
        (r"patient", "patient_identifier_like"),
        (r"barcode", "barcode_like"),
        (r"site", "site_or_domain_like"),
        (r"domain", "site_or_domain_like"),
        (r"target", "target_like"),
        (r"stage", "target_like"),
        (r"tumor_level", "target_or_sample_type_like"),
    ]
    excluded_cols = [
        "sample_barcode",
        "subject_id",
        "site",
        "tumor_level",
        "sample_type_label",
        "domain_primary",
        "domain_source",
        "target",
        "target_label",
        "ajcc_pathologic_stage_raw",
    ]
    for column in excluded_cols:
        if column in df.columns:
            rows.append(
                {
                    "column_name": column,
                    "status": "excluded",
                    "reason": "identifier_or_target_metadata",
                }
            )

    suspicious_feature_cols: list[str] = []
    for column in gene_cols:
        for pattern, reason in suspicious_patterns:
            if re.search(pattern, column, flags=re.IGNORECASE):
                suspicious_feature_cols.append(column)
                rows.append(
                    {
                        "column_name": column,
                        "status": "flagged_feature",
                        "reason": reason,
                    }
                )
                break

    if not suspicious_feature_cols:
        rows.append(
            {
                "column_name": "*gene_feature_block*",
                "status": "pass",
                "reason": "no_suspicious_feature_names_detected",
            }
        )

    duplicates = pd.Series(gene_cols).duplicated().sum()
    rows.append(
        {
            "column_name": "*gene_feature_block*",
            "status": "pass" if duplicates == 0 else "flagged",
            "reason": f"duplicated_gene_columns={int(duplicates)}",
        }
    )
    return pd.DataFrame(rows)


class TopVarianceSelector(BaseEstimator, TransformerMixin):
    def __init__(self, top_k: int = 1000):
        self.top_k = int(top_k)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "TopVarianceSelector":
        array = np.asarray(X, dtype=float)
        variances = np.nanvar(array, axis=0)
        variances = np.where(np.isnan(variances), -np.inf, variances)
        k = min(self.top_k, array.shape[1])
        selected = np.argsort(variances)[::-1][:k]
        self.selected_indices_ = np.sort(selected)
        self.variances_ = variances
        self.n_selected_ = int(len(self.selected_indices_))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        return array[:, self.selected_indices_]


def build_models(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "LogisticRegression": LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=config.random_seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=config.rf_trees,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.random_seed,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            max_iter=config.hgb_max_iter,
            random_state=config.random_seed,
        ),
    }


def build_pipeline(estimator: Any, config: ExperimentConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", TopVarianceSelector(top_k=config.top_k_genes)),
            ("scaler", StandardScaler()),
            ("model", clone(estimator)),
        ]
    )


def safe_predict_proba(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = pipeline.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise AttributeError("Model does not provide predict_proba or decision_function")


def get_selected_feature_count(pipeline: Pipeline) -> int | float:
    try:
        selector = pipeline.named_steps.get("selector")
        return int(selector.n_selected_)
    except Exception:
        return math.nan


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return math.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(y_true[mask]))
        bin_conf = float(np.mean(y_prob[mask]))
        ece += float(np.mean(mask)) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[dict[str, float], list[str]]:
    warnings_out: list[str] = []
    metrics = {
        "auc": math.nan,
        "accuracy": math.nan,
        "balacc": math.nan,
        "f1": math.nan,
        "precision": math.nan,
        "recall": math.nan,
        "brier": math.nan,
        "ece": math.nan,
    }
    if len(y_true) == 0:
        warnings_out.append("empty_evaluation_set")
        return metrics, warnings_out

    y_pred = (y_prob >= 0.5).astype(int)
    try:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    except Exception as exc:
        warnings_out.append(f"accuracy_failed:{exc}")

    try:
        metrics["balacc"] = float(balanced_accuracy_score(y_true, y_pred))
    except Exception as exc:
        warnings_out.append(f"balacc_failed:{exc}")

    try:
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception as exc:
        warnings_out.append(f"f1_failed:{exc}")

    try:
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    except Exception as exc:
        warnings_out.append(f"precision_failed:{exc}")

    try:
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    except Exception as exc:
        warnings_out.append(f"recall_failed:{exc}")

    try:
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    except Exception as exc:
        warnings_out.append(f"brier_failed:{exc}")

    if len(np.unique(y_true)) < 2:
        warnings_out.append("single_class_in_evaluation")
        return metrics, warnings_out

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception as exc:
        warnings_out.append(f"auc_failed:{exc}")

    try:
        metrics["ece"] = float(compute_ece(y_true, y_prob, n_bins=10))
    except Exception as exc:
        warnings_out.append(f"ece_failed:{exc}")

    return metrics, warnings_out


def fit_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    gene_cols: list[str],
    estimator: Any,
    config: ExperimentConfig,
) -> tuple[Pipeline | None, np.ndarray, list[str]]:
    warnings_out: list[str] = []
    y_train = train_df["target"].astype(int)
    pipeline = build_pipeline(estimator, config)
    try:
        pipeline.fit(train_df[gene_cols], y_train)
        y_prob = safe_predict_proba(pipeline, test_df[gene_cols])
    except Exception as exc:
        warnings_out.append(f"fit_or_predict_failed:{exc}")
        return None, np.array([]), warnings_out
    return pipeline, y_prob, warnings_out


def build_prediction_records(
    df: pd.DataFrame,
    *,
    model_name: str,
    split: str,
    outer_test_site: str,
    eval_site: str,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    y_pred = (y_prob >= 0.5).astype(int)
    return pd.DataFrame(
        {
            "model": model_name,
            "split": split,
            "outer_test_site": outer_test_site,
            "evaluation_site": eval_site,
            "subject_id": df["subject_id"].astype(str).to_numpy(),
            "sample_barcode": df["sample_barcode"].astype(str).to_numpy(),
            "site": df["site"].astype(str).to_numpy(),
            "y_true": df["target"].astype(int).to_numpy(),
            "y_prob": y_prob,
            "y_pred": y_pred,
        }
    )


def inner_id_evaluation(
    train_df: pd.DataFrame,
    gene_cols: list[str],
    estimator: Any,
    model_name: str,
    outer_test_site: str,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> tuple[dict[str, float], pd.DataFrame, list[str]]:
    warnings_out: list[str] = []
    y = train_df["target"].astype(int)
    if y.nunique() < 2:
        warnings_out.append("inner_id_unavailable_single_class")
        return (
            {
                f"{metric}_id": math.nan
                for metric in [
                    "auc",
                    "accuracy",
                    "balacc",
                    "f1",
                    "precision",
                    "recall",
                    "brier",
                    "ece",
                ]
            },
            pd.DataFrame(),
            warnings_out,
        )
    min_class = int(y.value_counts().min())
    n_splits = min(config.inner_cv_splits, len(train_df), min_class)
    if n_splits < 2:
        warnings_out.append("inner_id_unavailable_too_few_minority_samples")
        return (
            {
                f"{metric}_id": math.nan
                for metric in [
                    "auc",
                    "accuracy",
                    "balacc",
                    "f1",
                    "precision",
                    "recall",
                    "brier",
                    "ece",
                ]
            },
            pd.DataFrame(),
            warnings_out,
        )

    # ID is defined as validation on samples drawn from the already observed source domains.
    # This estimates within-source generalization, while the outer LODO fold remains the true OOD test.
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=config.random_seed
    )
    prediction_parts: list[pd.DataFrame] = []

    for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(
        splitter.split(train_df, y),
        start=1,
    ):
        inner_train_df = train_df.iloc[inner_train_idx].reset_index(drop=True)
        inner_val_df = train_df.iloc[inner_val_idx].reset_index(drop=True)
        if inner_train_df["target"].nunique() < 2:
            warnings_out.append(f"inner_fold_{inner_idx}_single_class_training")
            continue
        pipeline, y_prob, fit_warnings = fit_and_predict(
            inner_train_df, inner_val_df, gene_cols, estimator, config
        )
        warnings_out.extend(
            [f"inner_fold_{inner_idx}_{warning}" for warning in fit_warnings]
        )
        if pipeline is None:
            logger.warning(
                "Inner fold failed for model=%s outer_site=%s inner_fold=%s",
                model_name,
                outer_test_site,
                inner_idx,
            )
            continue
        eval_sites = "|".join(
            sorted(inner_val_df["site"].astype(str).unique().tolist())
        )
        prediction_parts.append(
            build_prediction_records(
                inner_val_df,
                model_name=model_name,
                split="id",
                outer_test_site=outer_test_site,
                eval_site=eval_sites,
                y_prob=y_prob,
            )
        )

    if not prediction_parts:
        warnings_out.append("inner_id_no_valid_predictions")
        return (
            {
                f"{metric}_id": math.nan
                for metric in [
                    "auc",
                    "accuracy",
                    "balacc",
                    "f1",
                    "precision",
                    "recall",
                    "brier",
                    "ece",
                ]
            },
            pd.DataFrame(),
            warnings_out,
        )

    pred_df = pd.concat(prediction_parts, ignore_index=True)
    metrics, metric_warnings = compute_metrics(
        pred_df["y_true"].to_numpy(), pred_df["y_prob"].to_numpy()
    )
    warnings_out.extend([f"id_{warning}" for warning in metric_warnings])
    return (
        {f"{metric}_id": value for metric, value in metrics.items()},
        pred_df,
        warnings_out,
    )


def run_lodo_evaluation(
    df: pd.DataFrame,
    gene_cols: list[str],
    config: ExperimentConfig,
    logger: logging.Logger,
    model_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_models = build_models(config)
    if model_names is None:
        models = all_models
    else:
        models = {
            name: model
            for name, model in all_models.items()
            if name in set(model_names)
        }
    sites = sorted(df["site"].astype(str).unique().tolist())
    fold_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []

    logger.info(
        "Running LODO evaluation across %s sites: %s", len(sites), ", ".join(sites)
    )

    for fold_idx, test_site in enumerate(sites, start=1):
        test_df = df.loc[df["site"].astype(str) == test_site].reset_index(drop=True)
        train_df = df.loc[df["site"].astype(str) != test_site].reset_index(drop=True)
        logger.info(
            "LODO fold %s/%s | test_site=%s | n_train=%s | n_test=%s",
            fold_idx,
            len(sites),
            test_site,
            len(train_df),
            len(test_df),
        )

        outer_warnings: list[str] = []
        if train_df["target"].nunique() < 2:
            outer_warnings.append("outer_training_single_class")
        if test_df["target"].nunique() < 2:
            outer_warnings.append("outer_test_single_class")

        for model_name, estimator in models.items():
            row: dict[str, Any] = {
                "model": model_name,
                "fold": fold_idx,
                "test_site": test_site,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "positive_rate_train": float(train_df["target"].mean()),
                "positive_rate_test": float(test_df["target"].mean()),
                "top_k_genes": int(config.top_k_genes),
            }
            warnings_out = list(outer_warnings)
            id_metrics, id_predictions, id_warnings = inner_id_evaluation(
                train_df=train_df,
                gene_cols=gene_cols,
                estimator=estimator,
                model_name=model_name,
                outer_test_site=test_site,
                config=config,
                logger=logger,
            )
            warnings_out.extend(id_warnings)
            if not id_predictions.empty:
                prediction_parts.append(id_predictions)

            if train_df["target"].nunique() < 2:
                ood_metrics = {
                    metric: math.nan
                    for metric in [
                        "auc",
                        "accuracy",
                        "balacc",
                        "f1",
                        "precision",
                        "recall",
                        "brier",
                        "ece",
                    ]
                }
                selected_count = math.nan
            else:
                pipeline, y_prob, fit_warnings = fit_and_predict(
                    train_df, test_df, gene_cols, estimator, config
                )
                warnings_out.extend([f"outer_{warning}" for warning in fit_warnings])
                if pipeline is None:
                    ood_metrics = {
                        metric: math.nan
                        for metric in [
                            "auc",
                            "accuracy",
                            "balacc",
                            "f1",
                            "precision",
                            "recall",
                            "brier",
                            "ece",
                        ]
                    }
                    selected_count = math.nan
                else:
                    selected_count = get_selected_feature_count(pipeline)
                    ood_metrics, ood_metric_warnings = compute_metrics(
                        test_df["target"].to_numpy(), y_prob
                    )
                    warnings_out.extend(
                        [f"ood_{warning}" for warning in ood_metric_warnings]
                    )
                    prediction_parts.append(
                        build_prediction_records(
                            test_df,
                            model_name=model_name,
                            split="ood",
                            outer_test_site=test_site,
                            eval_site=test_site,
                            y_prob=y_prob,
                        )
                    )

            row.update(id_metrics)
            for metric_name, metric_value in ood_metrics.items():
                row[f"{metric_name}_ood"] = metric_value
            row["selected_gene_count"] = selected_count
            for metric_name in ["auc", "accuracy", "balacc", "f1"]:
                id_key = f"{metric_name}_id"
                ood_key = f"{metric_name}_ood"
                row[f"gap_{metric_name}"] = (
                    row[id_key] - row[ood_key]
                    if pd.notna(row.get(id_key)) and pd.notna(row.get(ood_key))
                    else math.nan
                )
            row["warnings"] = aggregate_warning_list(warnings_out)
            fold_rows.append(row)

    return pd.DataFrame(fold_rows), pd.concat(
        prediction_parts, ignore_index=True
    ) if prediction_parts else pd.DataFrame()


def _within_site_cv_splits(
    site_df: pd.DataFrame, config: ExperimentConfig
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = site_df["target"].astype(int)
    if y.nunique() < 2:
        return []
    min_class = int(y.value_counts().min())
    n_splits = min(config.within_site_cv_splits, len(site_df), min_class)
    if n_splits < 2:
        return []
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=config.random_seed
    )
    return list(splitter.split(site_df, y))


def run_pairwise_site_transfer(
    df: pd.DataFrame,
    gene_cols: list[str],
    config: ExperimentConfig,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.enable_pairwise_heatmaps:
        return pd.DataFrame(), pd.DataFrame()

    all_models = build_models(config)
    models = {
        name: model
        for name, model in all_models.items()
        if name in set(config.pairwise_models)
    }
    sites = sorted(df["site"].astype(str).unique().tolist())
    result_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    logger.info("Running pairwise train-site -> test-site transfer matrix")

    for train_site, test_site in product(sites, sites):
        train_site_df = df.loc[df["site"].astype(str) == train_site].reset_index(
            drop=True
        )
        test_site_df = df.loc[df["site"].astype(str) == test_site].reset_index(
            drop=True
        )
        diagonal = train_site == test_site
        for model_name, estimator in models.items():
            row = {
                "model": model_name,
                "train_site": train_site,
                "test_site": test_site,
                "is_diagonal": diagonal,
                "n_train": int(len(train_site_df)),
                "n_test": int(len(test_site_df)),
                "positive_rate_train": float(train_site_df["target"].mean()),
                "positive_rate_test": float(test_site_df["target"].mean()),
            }
            warnings_out: list[str] = []

            if diagonal:
                splits = _within_site_cv_splits(train_site_df, config)
                if not splits:
                    warnings_out.append("within_site_cv_unavailable")
                    metrics = {
                        metric: math.nan
                        for metric in [
                            "auc",
                            "accuracy",
                            "balacc",
                            "f1",
                            "precision",
                            "recall",
                            "brier",
                            "ece",
                        ]
                    }
                    row.update(metrics)
                    row["selected_gene_count"] = math.nan
                    row["warnings"] = aggregate_warning_list(warnings_out)
                    result_rows.append(row)
                    continue

                cv_predictions: list[pd.DataFrame] = []
                selected_counts: list[int] = []
                for split_idx, (inner_train_idx, inner_test_idx) in enumerate(
                    splits, start=1
                ):
                    inner_train_df = train_site_df.iloc[inner_train_idx].reset_index(
                        drop=True
                    )
                    inner_test_df = train_site_df.iloc[inner_test_idx].reset_index(
                        drop=True
                    )
                    pipeline, y_prob, fit_warnings = fit_and_predict(
                        inner_train_df, inner_test_df, gene_cols, estimator, config
                    )
                    warnings_out.extend(
                        [f"diag_fold_{split_idx}_{warning}" for warning in fit_warnings]
                    )
                    if pipeline is None:
                        continue
                    selected_counts.append(int(get_selected_feature_count(pipeline)))
                    cv_predictions.append(
                        build_prediction_records(
                            inner_test_df,
                            model_name=model_name,
                            split="pairwise_diagonal",
                            outer_test_site=test_site,
                            eval_site=test_site,
                            y_prob=y_prob,
                        )
                    )
                if not cv_predictions:
                    warnings_out.append("within_site_cv_failed_all_splits")
                    metrics = {
                        metric: math.nan
                        for metric in [
                            "auc",
                            "accuracy",
                            "balacc",
                            "f1",
                            "precision",
                            "recall",
                            "brier",
                            "ece",
                        ]
                    }
                    row.update(metrics)
                    row["selected_gene_count"] = math.nan
                    row["warnings"] = aggregate_warning_list(warnings_out)
                    result_rows.append(row)
                    continue

                pred_df = pd.concat(cv_predictions, ignore_index=True)
                metrics, metric_warnings = compute_metrics(
                    pred_df["y_true"].to_numpy(), pred_df["y_prob"].to_numpy()
                )
                warnings_out.extend([f"diag_{warning}" for warning in metric_warnings])
                prediction_parts.append(pred_df)
                row.update(metrics)
                row["selected_gene_count"] = (
                    float(np.mean(selected_counts)) if selected_counts else math.nan
                )
                row["warnings"] = aggregate_warning_list(warnings_out)
                result_rows.append(row)
                continue

            pipeline, y_prob, fit_warnings = fit_and_predict(
                train_site_df, test_site_df, gene_cols, estimator, config
            )
            warnings_out.extend(fit_warnings)
            if pipeline is None:
                metrics = {
                    metric: math.nan
                    for metric in [
                        "auc",
                        "accuracy",
                        "balacc",
                        "f1",
                        "precision",
                        "recall",
                        "brier",
                        "ece",
                    ]
                }
                row.update(metrics)
                row["selected_gene_count"] = math.nan
                row["warnings"] = aggregate_warning_list(warnings_out)
                result_rows.append(row)
                continue

            metrics, metric_warnings = compute_metrics(
                test_site_df["target"].to_numpy(), y_prob
            )
            warnings_out.extend([f"offdiag_{warning}" for warning in metric_warnings])
            prediction_parts.append(
                build_prediction_records(
                    test_site_df,
                    model_name=model_name,
                    split="pairwise_ood",
                    outer_test_site=test_site,
                    eval_site=test_site,
                    y_prob=y_prob,
                )
            )
            row.update(metrics)
            row["selected_gene_count"] = get_selected_feature_count(pipeline)
            row["warnings"] = aggregate_warning_list(warnings_out)
            result_rows.append(row)

    return pd.DataFrame(result_rows), pd.concat(
        prediction_parts, ignore_index=True
    ) if prediction_parts else pd.DataFrame()


def evaluate_specific_training_sites(
    df: pd.DataFrame,
    gene_cols: list[str],
    train_sites: tuple[str, ...],
    test_site: str,
    estimator: Any,
    config: ExperimentConfig,
) -> tuple[dict[str, float], int | float, str]:
    train_df = df.loc[df["site"].isin(train_sites)].reset_index(drop=True)
    test_df = df.loc[df["site"] == test_site].reset_index(drop=True)
    if train_df["target"].nunique() < 2:
        return (
            {
                metric: math.nan
                for metric in [
                    "auc",
                    "accuracy",
                    "balacc",
                    "f1",
                    "precision",
                    "recall",
                    "brier",
                    "ece",
                ]
            },
            math.nan,
            "training_single_class",
        )

    pipeline, y_prob, fit_warnings = fit_and_predict(
        train_df, test_df, gene_cols, estimator, config
    )
    if pipeline is None:
        return (
            {
                metric: math.nan
                for metric in [
                    "auc",
                    "accuracy",
                    "balacc",
                    "f1",
                    "precision",
                    "recall",
                    "brier",
                    "ece",
                ]
            },
            math.nan,
            aggregate_warning_list(fit_warnings),
        )
    metrics, metric_warnings = compute_metrics(test_df["target"].to_numpy(), y_prob)
    all_warnings = aggregate_warning_list(fit_warnings + metric_warnings)
    return metrics, get_selected_feature_count(pipeline), all_warnings


def run_training_diversity_experiment(
    df: pd.DataFrame,
    gene_cols: list[str],
    fold_results: pd.DataFrame,
    pairwise_results: pd.DataFrame,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> pd.DataFrame:
    all_models = build_models(config)
    models = {
        name: model
        for name, model in all_models.items()
        if name in set(config.diversity_models)
    }
    sites = sorted(df["site"].astype(str).unique().tolist())
    n_total_sites = len(sites)
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict[str, Any]] = []

    logger.info(
        "Running training-domain diversity experiment on grouped source-site counts"
    )

    for test_site in sites:
        candidate_train_sites = [site for site in sites if site != test_site]

        for model_name, estimator in models.items():
            # k = 1 comes from pairwise results
            pairwise_subset = pairwise_results.loc[
                (pairwise_results["model"] == model_name)
                & (pairwise_results["test_site"] == test_site)
                & (~pairwise_results["is_diagonal"])
            ].copy()
            for _, pair_row in pairwise_subset.iterrows():
                rows.append(
                    {
                        "model": model_name,
                        "test_site": test_site,
                        "train_site_count": 1,
                        "train_sites": pair_row["train_site"],
                        "sampling_source": "pairwise_exact",
                        "selected_gene_count": pair_row.get(
                            "selected_gene_count", math.nan
                        ),
                        "warnings": pair_row.get("warnings", ""),
                        **{
                            metric: pair_row.get(metric, math.nan)
                            for metric in [
                                "auc",
                                "accuracy",
                                "balacc",
                                "f1",
                                "precision",
                                "recall",
                                "brier",
                                "ece",
                            ]
                        },
                    }
                )

            for k in sorted(config.diversity_k_values):
                if k <= 1 or k >= n_total_sites:
                    continue
                all_combos = list(combinations(candidate_train_sites, k))
                if not all_combos:
                    continue
                sampled_combos = all_combos
                if len(all_combos) > config.diversity_sampled_combinations:
                    chosen_idx = rng.choice(
                        len(all_combos),
                        size=config.diversity_sampled_combinations,
                        replace=False,
                    )
                    sampled_combos = [all_combos[idx] for idx in sorted(chosen_idx)]
                for combo in sampled_combos:
                    metrics, selected_count, warning_text = (
                        evaluate_specific_training_sites(
                            df=df,
                            gene_cols=gene_cols,
                            train_sites=combo,
                            test_site=test_site,
                            estimator=estimator,
                            config=config,
                        )
                    )
                    rows.append(
                        {
                            "model": model_name,
                            "test_site": test_site,
                            "train_site_count": int(k),
                            "train_sites": "|".join(combo),
                            "sampling_source": "sampled_combinations",
                            "selected_gene_count": selected_count,
                            "warnings": warning_text,
                            **metrics,
                        }
                    )

            # k = all source sites comes from the already-computed LODO run.
            lodo_subset = fold_results.loc[
                (fold_results["model"] == model_name)
                & (fold_results["test_site"] == test_site)
            ]
            for _, lodo_row in lodo_subset.iterrows():
                rows.append(
                    {
                        "model": model_name,
                        "test_site": test_site,
                        "train_site_count": n_total_sites - 1,
                        "train_sites": "all_other_sites",
                        "sampling_source": "lodo_exact",
                        "selected_gene_count": lodo_row.get(
                            "selected_gene_count", math.nan
                        ),
                        "warnings": lodo_row.get("warnings", ""),
                        **{
                            metric: lodo_row.get(f"{metric}_ood", math.nan)
                            for metric in [
                                "auc",
                                "accuracy",
                                "balacc",
                                "f1",
                                "precision",
                                "recall",
                                "brier",
                                "ece",
                            ]
                        },
                    }
                )

    return pd.DataFrame(rows)


def run_shuffle_test(
    df: pd.DataFrame,
    gene_cols: list[str],
    config: ExperimentConfig,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not config.enable_shuffle_test:
        return pd.DataFrame()
    rng = np.random.default_rng(config.random_seed)
    shuffled_df = df.copy()
    shuffled_df["target"] = rng.permutation(shuffled_df["target"].to_numpy())
    shuffled_df["target_label"] = shuffled_df["target"].map(
        {0: "Stage I-II", 1: "Stage III-IV"}
    )
    logger.info("Running shuffle test with globally permuted labels")
    shuffle_results, _ = run_lodo_evaluation(
        shuffled_df,
        gene_cols,
        config,
        logger,
        model_names=config.shuffle_models,
    )
    if not shuffle_results.empty:
        shuffle_results["test_type"] = "shuffled_labels"
    return shuffle_results


def summarize_model_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if fold_results.empty:
        return pd.DataFrame()
    for model_name, model_df in fold_results.groupby("model"):
        row: dict[str, Any] = {"model": model_name, "n_folds": int(len(model_df))}
        for metric in [
            "auc",
            "accuracy",
            "balacc",
            "f1",
            "precision",
            "recall",
            "brier",
            "ece",
        ]:
            for split in ["id", "ood"]:
                column = f"{metric}_{split}"
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
        for metric in ["auc", "accuracy", "balacc", "f1"]:
            gap_col = f"gap_{metric}"
            row[f"{gap_col}_mean"] = (
                float(model_df[gap_col].dropna().mean())
                if model_df[gap_col].notna().any()
                else math.nan
            )
            row[f"{gap_col}_std"] = (
                float(model_df[gap_col].dropna().std(ddof=1))
                if model_df[gap_col].notna().sum() > 1
                else math.nan
            )
        row["selected_gene_count_mean"] = (
            float(model_df["selected_gene_count"].dropna().mean())
            if model_df["selected_gene_count"].notna().any()
            else math.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def summarize_site_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if fold_results.empty:
        return pd.DataFrame()
    for (test_site, model_name), site_df in fold_results.groupby(
        ["test_site", "model"]
    ):
        rows.append(
            {
                "test_site": test_site,
                "model": model_name,
                "n_test": int(site_df["n_test"].mean()),
                "accuracy_ood_mean": float(site_df["accuracy_ood"].mean()),
                "balacc_ood_mean": float(site_df["balacc_ood"].mean()),
                "f1_ood_mean": float(site_df["f1_ood"].mean()),
                "auc_ood_mean": float(site_df["auc_ood"].mean()),
                "gap_accuracy_mean": float(site_df["gap_accuracy"].mean()),
                "gap_balacc_mean": float(site_df["gap_balacc"].mean()),
                "gap_f1_mean": float(site_df["gap_f1"].mean()),
                "gap_auc_mean": float(site_df["gap_auc"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["test_site", "model"]).reset_index(drop=True)


def summarize_overall_site_difficulty(site_summary_df: pd.DataFrame) -> pd.DataFrame:
    if site_summary_df.empty:
        return pd.DataFrame()
    overall = (
        site_summary_df.groupby("test_site")
        .agg(
            mean_accuracy_ood=("accuracy_ood_mean", "mean"),
            mean_balacc_ood=("balacc_ood_mean", "mean"),
            mean_f1_ood=("f1_ood_mean", "mean"),
            mean_auc_ood=("auc_ood_mean", "mean"),
        )
        .reset_index()
        .sort_values(["mean_balacc_ood", "mean_auc_ood"], ascending=[True, True])
        .reset_index(drop=True)
    )
    overall["difficulty_rank"] = np.arange(1, len(overall) + 1)
    return overall


def summarize_pairwise_results(pairwise_results: pd.DataFrame) -> pd.DataFrame:
    if pairwise_results.empty:
        return pd.DataFrame()
    offdiag = pairwise_results.loc[~pairwise_results["is_diagonal"]].copy()
    if offdiag.empty:
        return pd.DataFrame()
    return (
        offdiag.groupby("model")
        .agg(
            n_pairs=("model", "size"),
            mean_accuracy=("accuracy", "mean"),
            mean_balacc=("balacc", "mean"),
            mean_f1=("f1", "mean"),
            mean_auc=("auc", "mean"),
        )
        .reset_index()
        .sort_values("model")
        .reset_index(drop=True)
    )


def summarize_training_diversity(diversity_results: pd.DataFrame) -> pd.DataFrame:
    if diversity_results.empty:
        return pd.DataFrame()
    return (
        diversity_results.groupby(["model", "test_site", "train_site_count"])
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_balacc=("balacc", "mean"),
            mean_f1=("f1", "mean"),
            mean_auc=("auc", "mean"),
            n_evaluations=("train_sites", "size"),
        )
        .reset_index()
        .sort_values(["model", "test_site", "train_site_count"])
        .reset_index(drop=True)
    )


def save_figure(fig: plt.Figure, path_no_suffix: Path, save_pdf: bool) -> None:
    fig.savefig(path_no_suffix.with_suffix(".png"), dpi=320, bbox_inches="tight")
    if save_pdf:
        fig.savefig(path_no_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _score_colors(
    values: np.ndarray,
    *,
    cmap_name: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> list[Any]:
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return [
        cmap(norm(float(value))) if pd.notna(value) else (0.75, 0.75, 0.75, 1.0)
        for value in values
    ]


def plot_metric_bars(
    fold_results: pd.DataFrame,
    metric_column: str,
    title: str,
    ylabel: str,
    output_path_no_suffix: Path,
    save_pdf: bool,
    ylim_01: bool = True,
) -> None:
    pivot = fold_results.pivot_table(
        index="test_site", columns="model", values=metric_column, aggfunc="mean"
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.index) * 1.2), 6.5))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))
    for idx, model_name in enumerate(pivot.columns):
        values = pivot[model_name].to_numpy(dtype=float)
        xpos = x + idx * width - (len(pivot.columns) - 1) * width / 2
        bars = ax.bar(
            xpos, values, width=width, label=model_name, color=_score_colors(values)
        )
        for bar, value in zip(bars, values):
            if pd.notna(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Held-out site", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=35, ha="right")
    if ylim_01:
        ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model", frameon=False)
    save_figure(fig, output_path_no_suffix, save_pdf)


def plot_gap_bars(
    fold_results: pd.DataFrame,
    metric_column: str,
    title: str,
    ylabel: str,
    output_path_no_suffix: Path,
    save_pdf: bool,
) -> None:
    pivot = fold_results.pivot_table(
        index="test_site", columns="model", values=metric_column, aggfunc="mean"
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.index) * 1.2), 6.5))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))
    vmax = max(
        0.05,
        float(np.nanmax(pivot.to_numpy(dtype=float)))
        if np.isfinite(np.nanmax(pivot.to_numpy(dtype=float)))
        else 0.05,
    )
    for idx, model_name in enumerate(pivot.columns):
        values = pivot[model_name].to_numpy(dtype=float)
        xpos = x + idx * width - (len(pivot.columns) - 1) * width / 2
        bars = ax.bar(
            xpos,
            values,
            width=width,
            label=model_name,
            color=_score_colors(vmax - values, cmap_name="RdYlGn", vmin=0.0, vmax=vmax),
        )
        for bar, value in zip(bars, values):
            if pd.notna(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.006,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Held-out site", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=35, ha="right")
    ax.set_ylim(0.0, max(0.25, vmax + 0.05))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model", frameon=False)
    save_figure(fig, output_path_no_suffix, save_pdf)


def heatmap_with_annotations(
    data: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path_no_suffix: Path,
    save_pdf: bool,
    cmap_name: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    if data.empty:
        return
    matrix = data.to_numpy(dtype=float)
    fig, ax = plt.subplots(
        figsize=(max(7, data.shape[1] * 1.3), max(5, data.shape[0] * 0.7))
    )
    image = ax.imshow(matrix, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(data.columns.tolist(), rotation=35, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index.tolist())
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            label = "NA" if np.isnan(value) else f"{value:.2f}"
            text_color = (
                "black" if np.isnan(value) or value < (vmin + vmax) / 2 else "white"
            )
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )
    cbar = fig.colorbar(image, ax=ax)
    cbar.ax.set_ylabel("Score", rotation=90)
    save_figure(fig, output_path_no_suffix, save_pdf)


def plot_site_model_ood_heatmap(
    fold_results: pd.DataFrame,
    output_path_no_suffix: Path,
    save_pdf: bool,
) -> None:
    metrics = [
        ("auc_ood", "ROC-AUC"),
        ("balacc_ood", "Balanced accuracy"),
        ("f1_ood", "F1-score"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    for ax, (metric_col, metric_label) in zip(axes, metrics):
        pivot = fold_results.pivot_table(
            index="test_site", columns="model", values=metric_col, aggfunc="mean"
        )
        matrix = pivot.to_numpy(dtype=float)
        image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
        ax.set_title(metric_label, fontsize=14, pad=10)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.tolist(), rotation=30, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist())
        if ax is axes[0]:
            ax.set_ylabel("Held-out site", fontsize=12)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                label = "NA" if np.isnan(value) else f"{value:.3f}"
                text_color = "black" if np.isnan(value) or value < 0.55 else "white"
                ax.text(
                    col_idx,
                    row_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                )
    cbar = fig.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.8, location="bottom", pad=0.12
    )
    cbar.ax.set_xlabel("Performance", fontsize=12)
    fig.suptitle(
        "TCGA-BRCA baseline M0: site × model OOD performance",
        fontsize=18,
        weight="bold",
        y=1.02,
    )
    save_figure(fig, output_path_no_suffix, save_pdf)


def plot_model_comparison(
    model_summary: pd.DataFrame, output_path_no_suffix: Path, save_pdf: bool
) -> None:
    if model_summary.empty:
        return
    plot_df = model_summary[
        [
            "model",
            "accuracy_id_mean",
            "accuracy_ood_mean",
            "balacc_id_mean",
            "balacc_ood_mean",
        ]
    ].copy()
    melted = plot_df.melt(id_vars="model", var_name="metric_split", value_name="score")
    labels = {
        "accuracy_id_mean": "Accuracy ID",
        "accuracy_ood_mean": "Accuracy OOD",
        "balacc_id_mean": "BalAcc ID",
        "balacc_ood_mean": "BalAcc OOD",
    }
    melted["metric_split"] = melted["metric_split"].map(labels)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_summary))
    unique_labels = list(labels.values())
    width = 0.18
    for idx, label in enumerate(unique_labels):
        values = melted.loc[melted["metric_split"] == label, "score"].to_numpy(
            dtype=float
        )
        xpos = x + idx * width - (len(unique_labels) - 1) * width / 2
        bars = ax.bar(
            xpos, values, width=width, label=label, color=_score_colors(values)
        )
        for bar, value in zip(bars, values):
            if pd.notna(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_title("Baseline M0 model comparison", fontsize=16, weight="bold")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_summary["model"].tolist(), rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, output_path_no_suffix, save_pdf)


def plot_pairwise_heatmaps(
    pairwise_results: pd.DataFrame, plots_dir: Path, save_pdf: bool
) -> None:
    if pairwise_results.empty:
        return
    for model_name in sorted(pairwise_results["model"].unique().tolist()):
        model_df = pairwise_results.loc[pairwise_results["model"] == model_name].copy()
        for metric_col, label in [
            ("accuracy", "Accuracy"),
            ("auc", "ROC-AUC"),
            ("f1", "F1-score"),
        ]:
            pivot = (
                model_df.pivot_table(
                    index="train_site",
                    columns="test_site",
                    values=metric_col,
                    aggfunc="mean",
                )
                .sort_index()
                .sort_index(axis=1)
            )
            heatmap_with_annotations(
                pivot,
                title=f"{model_name}: train site → test site ({label})",
                xlabel="Test site",
                ylabel="Train site",
                output_path_no_suffix=plots_dir
                / f"train_test_heatmap_{sanitize_name(metric_col)}_{sanitize_name(model_name)}",
                save_pdf=save_pdf,
            )


def plot_training_diversity_heatmaps(
    diversity_summary: pd.DataFrame,
    plots_dir: Path,
    save_pdf: bool,
) -> None:
    if diversity_summary.empty:
        return
    for model_name in sorted(diversity_summary["model"].unique().tolist()):
        model_df = diversity_summary.loc[
            diversity_summary["model"] == model_name
        ].copy()
        pivot = model_df.pivot_table(
            index="test_site",
            columns="train_site_count",
            values="mean_accuracy",
            aggfunc="mean",
        )
        pivot = pivot.sort_index(axis=1)
        heatmap_with_annotations(
            pivot,
            title=f"{model_name}: OOD accuracy as training-site diversity increases",
            xlabel="Number of training sites",
            ylabel="Held-out site",
            output_path_no_suffix=plots_dir
            / f"training_diversity_by_site_{sanitize_name(model_name)}",
            save_pdf=save_pdf,
        )

    overall = (
        diversity_summary.groupby(["model", "train_site_count"])
        .agg(mean_accuracy=("mean_accuracy", "mean"))
        .reset_index()
    )
    pivot_overall = overall.pivot(
        index="model", columns="train_site_count", values="mean_accuracy"
    ).sort_index(axis=1)
    heatmap_with_annotations(
        pivot_overall,
        title="Effect of training-site diversity on OOD accuracy",
        xlabel="Number of training sites",
        ylabel="Model",
        output_path_no_suffix=plots_dir / "training_diversity_overall",
        save_pdf=save_pdf,
    )


def build_calibration_dataframe(prediction_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if prediction_df.empty:
        return pd.DataFrame()
    ood_predictions = prediction_df.loc[prediction_df["split"] == "ood"].copy()
    for model_name, model_df in ood_predictions.groupby("model"):
        y_true = model_df["y_true"].to_numpy(dtype=int)
        y_prob = model_df["y_prob"].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            continue
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=10, strategy="uniform"
        )
        brier = float(brier_score_loss(y_true, y_prob))
        for bin_idx, (pred_bin, true_bin) in enumerate(
            zip(prob_pred, prob_true), start=1
        ):
            rows.append(
                {
                    "model": model_name,
                    "bin_idx": bin_idx,
                    "predicted_probability": float(pred_bin),
                    "observed_frequency": float(true_bin),
                    "brier": brier,
                }
            )
    return pd.DataFrame(rows)


def plot_calibration_curves(
    calibration_df: pd.DataFrame, plots_dir: Path, save_pdf: bool
) -> None:
    if calibration_df.empty:
        return
    for model_name, model_df in calibration_df.groupby("model"):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            linewidth=1.5,
            label="Perfect calibration",
        )
        ax.plot(
            model_df["predicted_probability"].to_numpy(),
            model_df["observed_frequency"].to_numpy(),
            marker="o",
            linewidth=2.0,
            color="#1b7837",
            label=f"{model_name} (Brier={model_df['brier'].iloc[0]:.3f})",
        )
        ax.set_title(f"Calibration curve: {model_name}", fontsize=15, weight="bold")
        ax.set_xlabel("Mean predicted probability", fontsize=12)
        ax.set_ylabel("Observed positive rate", fontsize=12)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, loc="upper left")
        save_figure(
            fig, plots_dir / f"calibration_{sanitize_name(model_name)}", save_pdf
        )


def make_plots(
    fold_results: pd.DataFrame,
    model_summary: pd.DataFrame,
    pairwise_results: pd.DataFrame,
    diversity_summary: pd.DataFrame,
    prediction_df: pd.DataFrame,
    output_dir: Path,
    save_pdf: bool,
) -> None:
    plots_dir = ensure_dir(output_dir / "plots")
    plot_metric_bars(
        fold_results,
        metric_column="accuracy_ood",
        title="TCGA-BRCA baseline M0: OOD accuracy by held-out site",
        ylabel="Accuracy",
        output_path_no_suffix=plots_dir / "accuracy_ood_by_site",
        save_pdf=save_pdf,
        ylim_01=True,
    )
    plot_metric_bars(
        fold_results,
        metric_column="balacc_ood",
        title="TCGA-BRCA baseline M0: OOD balanced accuracy by held-out site",
        ylabel="Balanced accuracy",
        output_path_no_suffix=plots_dir / "balacc_ood_by_site",
        save_pdf=save_pdf,
        ylim_01=True,
    )
    plot_metric_bars(
        fold_results,
        metric_column="f1_ood",
        title="TCGA-BRCA baseline M0: OOD F1-score by held-out site",
        ylabel="F1-score",
        output_path_no_suffix=plots_dir / "f1_ood_by_site",
        save_pdf=save_pdf,
        ylim_01=True,
    )
    plot_metric_bars(
        fold_results,
        metric_column="auc_ood",
        title="TCGA-BRCA baseline M0: OOD ROC-AUC by held-out site",
        ylabel="ROC-AUC",
        output_path_no_suffix=plots_dir / "auc_ood_by_site",
        save_pdf=save_pdf,
        ylim_01=True,
    )
    plot_gap_bars(
        fold_results,
        metric_column="gap_balacc",
        title="ID-OOD gap in balanced accuracy by held-out site",
        ylabel="Gap (ID - OOD)",
        output_path_no_suffix=plots_dir / "gap_balacc_by_site",
        save_pdf=save_pdf,
    )
    plot_gap_bars(
        fold_results,
        metric_column="gap_auc",
        title="ID-OOD gap in ROC-AUC by held-out site",
        ylabel="Gap (ID - OOD)",
        output_path_no_suffix=plots_dir / "gap_auc_by_site",
        save_pdf=save_pdf,
    )
    plot_model_comparison(model_summary, plots_dir / "model_comparison", save_pdf)
    plot_site_model_ood_heatmap(
        fold_results, plots_dir / "heatmap_site_model_ood", save_pdf
    )
    plot_pairwise_heatmaps(pairwise_results, plots_dir, save_pdf)
    plot_training_diversity_heatmaps(diversity_summary, plots_dir, save_pdf)
    calibration_df = build_calibration_dataframe(prediction_df)
    calibration_df.to_csv(output_dir / "calibration_curve_data.csv", index=False)
    plot_calibration_curves(calibration_df, plots_dir, save_pdf)


def dataframe_block(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows available._"
    view = df.head(max_rows) if max_rows is not None else df
    return "```text\n" + view.to_string(index=False) + "\n```"


def write_report(
    config: ExperimentConfig,
    audit: dict[str, Any],
    site_summary: pd.DataFrame,
    fold_results: pd.DataFrame,
    model_summary: pd.DataFrame,
    overall_site_difficulty: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    diversity_summary: pd.DataFrame,
    shuffle_summary: pd.DataFrame,
    leakage_checks: pd.DataFrame,
    output_dir: Path,
) -> None:
    hardest_sites = overall_site_difficulty.head(5)
    easiest_sites = overall_site_difficulty.tail(5).sort_values(
        "mean_balacc_ood", ascending=False
    )
    diversity_overall = (
        diversity_summary.groupby(["model", "train_site_count"])
        .agg(
            mean_accuracy=("mean_accuracy", "mean"),
            mean_balacc=("mean_balacc", "mean"),
            mean_auc=("mean_auc", "mean"),
        )
        .reset_index()
        .sort_values(["model", "train_site_count"])
    )
    shuffle_model_summary = (
        shuffle_summary.groupby("model")
        .agg(
            mean_accuracy_ood=("accuracy_ood", "mean"),
            mean_balacc_ood=("balacc_ood", "mean"),
            mean_auc_ood=("auc_ood", "mean"),
        )
        .reset_index()
        .sort_values("model")
        if not shuffle_summary.empty
        else pd.DataFrame()
    )

    mean_gap_balacc = (
        float(model_summary["gap_balacc_mean"].dropna().mean())
        if not model_summary.empty
        else math.nan
    )
    mean_gap_auc = (
        float(model_summary["gap_auc_mean"].dropna().mean())
        if not model_summary.empty
        else math.nan
    )
    mean_shuffle_balacc = (
        float(shuffle_model_summary["mean_balacc_ood"].dropna().mean())
        if not shuffle_model_summary.empty
        else math.nan
    )

    comments: list[str] = []
    if pd.notna(mean_gap_balacc) and mean_gap_balacc > 0.05:
        comments.append(
            "The average ID-to-OOD balanced-accuracy gap is non-negligible, so transportability is visibly weaker than source-domain validation would suggest."
        )
    if (
        not hardest_sites.empty
        and hardest_sites["mean_balacc_ood"].max()
        - hardest_sites["mean_balacc_ood"].min()
        > 0.10
    ):
        comments.append(
            "There is substantial cross-site heterogeneity: some hold-out sites are consistently harder than others."
        )
    if pd.notna(mean_shuffle_balacc) and mean_shuffle_balacc < 0.60:
        comments.append(
            "The shuffle test collapses performance toward chance, which argues against trivial leakage-driven discrimination."
        )
    if (
        not pairwise_summary.empty
        and pairwise_summary["mean_balacc"].max()
        - pairwise_summary["mean_balacc"].min()
        > 0.05
    ):
        comments.append(
            "Single-site train→test transfer is more unstable than multi-site LODO, reinforcing the value of source-domain diversity."
        )

    report_lines = [
        "# TCGA-BRCA Reimagined Baseline M0",
        "",
        "## Why the original task was too easy",
        "- The previous baseline used `01 vs 11` on BRCA RNA-seq, which is essentially primary tumor versus solid normal tissue.",
        "- On gene expression data this contrast is biologically very strong and often close to linearly separable, so near-perfect OOD scores are not surprising.",
        "- That setup hides the transportability problem because class signal dominates site effects, making harmonization or domain-aware methods hard to justify empirically.",
        "",
        "## New task design",
        "- Preferred option in principle: PAM50 molecular subtype prediction.",
        "- Practical limitation: a complete PAM50 label table is not available in the local files. Only a one-row clinical template exists, which is not sufficient for a defensible experiment.",
        "- Implemented task: **primary-tumor-only pathologic risk stratification**, defined as `Stage I-II` versus `Stage III-IV` from `diagnoses.ajcc_pathologic_stage`.",
        "- This task is clinically meaningful, label-available, substantially less trivial than tumor-vs-normal, and it preserves 14 sites after site-level class filtering.",
        "",
        "## Anti-leakage design",
        "- Expression features only: identifier columns, site/domain columns, tumor-level metadata, and the target-stage columns are excluded from X.",
        "- Outer validation is leave-one-site-out (LODO).",
        "- ID performance is estimated only on the pooled source data via inner stratified cross-validation, so the validation samples remain in-domain while the outer hold-out site remains truly OOD.",
        "- Feature reduction is train-only: the top-variance genes are selected **inside each fit**, never globally.",
        "- The final pipeline for every model is `median imputation -> top-K variance genes -> z-score scaling -> classifier`.",
        f"- Full train-site -> test-site transfer matrices are generated for: `{', '.join(config.pairwise_models)}`.",
        "",
        "## Dataset audit",
        f"- Raw preprocessed expression rows: `{audit['n_expression_rows_raw']}`",
        f"- Raw gene features: `{audit['n_gene_features_raw']}`",
        f"- Primary-tumor rows before subject deduplication: `{audit['n_primary_tumor_rows_before_subject_dedup']}`",
        f"- Subjects with multiple primary-tumor samples: `{audit['n_subjects_with_multiple_primary_samples']}`",
        f"- Subject-level primary-tumor rows after deduplication: `{audit['n_primary_subject_level_rows']}`",
        f"- Rows with usable mapped stage target: `{audit['n_rows_with_mapped_stage_target']}`",
        f"- Sites retained after requiring at least `{config.min_per_class_per_site}` samples per class: `{audit['n_valid_sites_after_min_class_filter']}`",
        f"- Retained sites: `{', '.join(audit['valid_sites'])}`",
        f"- Final modeling rows: `{audit['n_modeling_rows']}`",
        f"- Class counts: `{audit['class_counts']}`",
        "",
        "## Site composition",
        dataframe_block(site_summary.round(4)),
        "",
        "## Model summary",
        dataframe_block(model_summary.round(4)),
        "",
        "## Hardest hold-out sites",
        dataframe_block(hardest_sites.round(4), max_rows=5),
        "",
        "## Easiest hold-out sites",
        dataframe_block(easiest_sites.round(4), max_rows=5),
        "",
        "## Pairwise single-site transfer summary",
        dataframe_block(pairwise_summary.round(4)),
        "",
        "## Training-site diversity summary",
        dataframe_block(diversity_overall.round(4), max_rows=20),
        "",
        "## Shuffle test summary",
        dataframe_block(shuffle_model_summary.round(4), max_rows=20),
        "",
        "## Leakage checks",
        dataframe_block(leakage_checks, max_rows=20),
        "",
        "## Automatic interpretation",
    ]
    if comments:
        report_lines.extend([f"- {comment}" for comment in comments])
    else:
        report_lines.append("- No automatic comment was triggered.")
    report_lines.extend(
        [
            "",
            "## Methodological notes",
            "- This baseline is intentionally raw M0: no ComBat, no reweighting, no DASHI guidance, and no domain adaptation.",
            "- Performance degradation from ID to OOD is therefore attributable to the raw cross-site generalization problem plus ordinary model misspecification.",
            "- The retained task is still binary, but it is clinically grounded and materially harder than tumor-vs-normal.",
            "- Pairwise train->test heatmaps are focused on Logistic Regression to keep the transfer matrix interpretable and computationally tractable, while the main LODO benchmark still compares all three baseline models.",
        ]
    )
    (output_dir / "baseline_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


def write_readme(config: ExperimentConfig, output_dir: Path) -> None:
    readme = f"""# TCGA-BRCA Reimagined Baseline M0

This folder contains a redesigned TCGA-BRCA baseline meant for transportability analysis.

## Task

- Cohort: primary tumor samples only (`tumor_level == 1`)
- Subject handling: one primary-tumor sample per subject, keeping the lexicographically first barcode
- Target: `Stage I-II` vs `Stage III-IV` from `diagnoses.ajcc_pathologic_stage`
- Domain: `site`

## Why this task

The previous `01 vs 11` experiment was too easy because tumor-vs-normal gene-expression separation is near-trivial. This redesign uses a clinically meaningful endpoint that is locally available and much less separable.

## How to run

```bash
python3 tcga_stage_reimagined_m0.py \\
  --input "{config.input_path}" \\
  --clinical "{config.clinical_path}" \\
  --output "{config.output_dir}"
```

## Main outputs

- `prepared_tcga_stage_dataset.csv`
- `fold_results.csv`
- `model_summary.csv`
- `site_summary.csv`
- `train_test_site_results.csv`
- `training_diversity_results.csv`
- `shuffle_test_results.csv`
- `baseline_report.md`
- `plots/`

## Leakage safeguards

- Train-only feature reduction with top-variance genes inside each fit
- LODO outer split by site
- Inner stratified validation on the pooled source data for ID estimates
- Explicit exclusion of identifiers and site/target columns from X
- Shuffle-label sanity check
- Detailed train-site -> test-site transfer heatmaps are produced for Logistic Regression, while the main LODO benchmark still covers all three models
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def save_outputs(
    config: ExperimentConfig,
    prepared_df: pd.DataFrame,
    duplicate_subject_df: pd.DataFrame,
    conflict_df: pd.DataFrame,
    leakage_checks: pd.DataFrame,
    site_summary: pd.DataFrame,
    fold_results: pd.DataFrame,
    prediction_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    site_model_summary: pd.DataFrame,
    overall_site_difficulty: pd.DataFrame,
    pairwise_results: pd.DataFrame,
    pairwise_predictions: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    diversity_results: pd.DataFrame,
    diversity_summary: pd.DataFrame,
    shuffle_results: pd.DataFrame,
    shuffle_summary: pd.DataFrame,
    audit: dict[str, Any],
    output_dir: Path,
) -> None:
    prepared_df.to_csv(output_dir / "prepared_tcga_stage_dataset.csv", index=False)
    site_summary.to_csv(output_dir / "site_summary.csv", index=False)
    duplicate_subject_df.to_csv(
        output_dir / "duplicate_primary_subjects.csv", index=False
    )
    conflict_df.to_csv(output_dir / "clinical_stage_conflicts.csv", index=False)
    leakage_checks.to_csv(output_dir / "leakage_checks.csv", index=False)
    fold_results.to_csv(output_dir / "fold_results.csv", index=False)
    prediction_df.to_csv(output_dir / "prediction_records.csv", index=False)
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    site_model_summary.to_csv(output_dir / "site_model_summary.csv", index=False)
    overall_site_difficulty.to_csv(
        output_dir / "site_difficulty_summary.csv", index=False
    )
    pairwise_results.to_csv(output_dir / "train_test_site_results.csv", index=False)
    pairwise_predictions.to_csv(
        output_dir / "train_test_site_prediction_records.csv", index=False
    )
    pairwise_summary.to_csv(
        output_dir / "train_test_site_model_summary.csv", index=False
    )
    diversity_results.to_csv(output_dir / "training_diversity_results.csv", index=False)
    diversity_summary.to_csv(output_dir / "training_diversity_summary.csv", index=False)
    shuffle_results.to_csv(output_dir / "shuffle_test_results.csv", index=False)
    shuffle_summary.to_csv(output_dir / "shuffle_test_model_summary.csv", index=False)
    with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True, default=str)
    with (output_dir / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, sort_keys=True, default=str)

    make_plots(
        fold_results=fold_results,
        model_summary=model_summary,
        pairwise_results=pairwise_results,
        diversity_summary=diversity_summary,
        prediction_df=prediction_df,
        output_dir=output_dir,
        save_pdf=config.save_pdf_plots,
    )
    write_report(
        config=config,
        audit=audit,
        site_summary=site_summary,
        fold_results=fold_results,
        model_summary=model_summary,
        overall_site_difficulty=overall_site_difficulty,
        pairwise_summary=pairwise_summary,
        diversity_summary=diversity_summary,
        shuffle_summary=shuffle_results,
        leakage_checks=leakage_checks,
        output_dir=output_dir,
    )
    write_readme(config, output_dir)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Reimagined TCGA-BRCA baseline M0 for transportability"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Path to the preprocessed TCGA expression table",
    )
    parser.add_argument(
        "--clinical",
        default=DEFAULT_CLINICAL_PATH,
        help="Path to the TCGA clinical TSV",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Top-K variance genes kept per training fit",
    )
    parser.add_argument(
        "--min-per-class-per-site",
        type=int,
        default=5,
        help="Minimum per-class count required to keep a site",
    )
    parser.add_argument(
        "--diversity-combos",
        type=int,
        default=3,
        help="Maximum sampled combinations for grouped training-site diversity",
    )
    args = parser.parse_args()
    return ExperimentConfig(
        input_path=args.input,
        clinical_path=args.clinical,
        output_dir=args.output,
        random_seed=args.seed,
        top_k_genes=args.top_k,
        min_per_class_per_site=args.min_per_class_per_site,
        diversity_sampled_combinations=args.diversity_combos,
    )


def main() -> None:
    config = parse_args()
    output_dir = ensure_dir(config.output_dir)
    logger = init_logger(output_dir)

    logger.info("Starting redesigned TCGA baseline M0")
    prepared_df, gene_cols, audit, site_summary, duplicate_subject_df, conflict_df = (
        prepare_tcga_stage_dataset(config, logger)
    )
    leakage_checks = run_leakage_checks(prepared_df, gene_cols)
    fold_results, prediction_df = run_lodo_evaluation(
        prepared_df, gene_cols, config, logger
    )
    model_summary = summarize_model_results(fold_results)
    site_model_summary = summarize_site_results(fold_results)
    overall_site_difficulty = summarize_overall_site_difficulty(site_model_summary)
    pairwise_results, pairwise_predictions = run_pairwise_site_transfer(
        prepared_df, gene_cols, config, logger
    )
    pairwise_summary = summarize_pairwise_results(pairwise_results)
    diversity_results = run_training_diversity_experiment(
        prepared_df, gene_cols, fold_results, pairwise_results, config, logger
    )
    diversity_summary = summarize_training_diversity(diversity_results)
    shuffle_results = run_shuffle_test(prepared_df, gene_cols, config, logger)
    shuffle_summary = (
        summarize_model_results(shuffle_results)
        if not shuffle_results.empty
        else pd.DataFrame()
    )

    save_outputs(
        config=config,
        prepared_df=prepared_df,
        duplicate_subject_df=duplicate_subject_df,
        conflict_df=conflict_df,
        leakage_checks=leakage_checks,
        site_summary=site_summary,
        fold_results=fold_results,
        prediction_df=prediction_df,
        model_summary=model_summary,
        site_model_summary=site_model_summary,
        overall_site_difficulty=overall_site_difficulty,
        pairwise_results=pairwise_results,
        pairwise_predictions=pairwise_predictions,
        pairwise_summary=pairwise_summary,
        diversity_results=diversity_results,
        diversity_summary=diversity_summary,
        shuffle_results=shuffle_results,
        shuffle_summary=shuffle_summary,
        audit=audit,
        output_dir=output_dir,
    )
    logger.info("Finished redesigned TCGA baseline M0. Outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
