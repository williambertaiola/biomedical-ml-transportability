"""Transportability experiment: tcga m0 from scratch."""

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


import itertools
import json
import logging
import math
import os
import re
import shutil
import tempfile
import warnings
from dataclasses import asdict, dataclass
from html import escape
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
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
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
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")


@dataclass
class Config:
    dataset_name: str = "TCGA_BRCA_01_vs_11_M0_FromScratch"
    input_path: str = _configured_path(
        "legacy_preprocessed_tcga_tcga_brca_preprossed.csv"
    )
    output_dir: str = _configured_path("legacy_output")
    random_seed: int = 42
    variance_prefilter_k: int = 5000
    top_k_genes: int = 250
    min_site_size: int = 20
    save_pdf_plots: bool = True
    calibration_bins: int = 10


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def init_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger(f"tcga_m0::{log_path}")
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


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def pretty_model_name(value: str) -> str:
    mapping = {
        "logisticregression": "Logistic Regression",
        "randomforest": "Random Forest",
        "histgradientboosting": "HistGradientBoosting",
    }
    return mapping.get(sanitize_name(value), value)


def aggregate_warning_list(items: list[str]) -> str:
    cleaned = [item for item in items if item]
    seen: list[str] = []
    for item in cleaned:
        if item not in seen:
            seen.append(item)
    return " | ".join(seen)


def good_cmap():
    return plt.get_cmap("RdYlGn")


def bad_cmap():
    return plt.get_cmap("RdYlGn_r")


def save_figure(fig: plt.Figure, path_no_suffix: Path, save_pdf: bool = True) -> None:
    fig.tight_layout()
    fig.savefig(path_no_suffix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(path_no_suffix.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return math.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    score = 0.0
    total = len(y_true)
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        accuracy = float(np.mean(y_true[mask]))
        confidence = float(np.mean(y_prob[mask]))
        score += (np.sum(mask) / total) * abs(accuracy - confidence)
    return float(score)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[dict[str, float], list[str]]:
    metrics: dict[str, float] = {}
    warnings_out: list[str] = []
    if len(y_true) == 0:
        for key in [
            "auc",
            "accuracy",
            "f1",
            "balacc",
            "precision",
            "recall",
            "brier",
            "ece",
        ]:
            metrics[key] = math.nan
        return metrics, ["empty_labels"]

    y_pred = (y_prob >= 0.5).astype(int)
    unique_classes = np.unique(y_true)

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    metrics["ece"] = compute_ece(y_true, y_prob, n_bins=n_bins)

    if len(unique_classes) < 2:
        warnings_out.append("single_class_in_evaluation")
        metrics["auc"] = math.nan
        metrics["f1"] = math.nan
        metrics["balacc"] = math.nan
        metrics["precision"] = math.nan
        metrics["recall"] = math.nan
        return metrics, warnings_out

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception as exc:
        metrics["auc"] = math.nan
        warnings_out.append(f"auc_failed:{exc}")
    try:
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception as exc:
        metrics["f1"] = math.nan
        warnings_out.append(f"f1_failed:{exc}")
    try:
        metrics["balacc"] = float(balanced_accuracy_score(y_true, y_pred))
    except Exception as exc:
        metrics["balacc"] = math.nan
        warnings_out.append(f"balacc_failed:{exc}")
    try:
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    except Exception as exc:
        metrics["precision"] = math.nan
        warnings_out.append(f"precision_failed:{exc}")
    try:
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    except Exception as exc:
        metrics["recall"] = math.nan
        warnings_out.append(f"recall_failed:{exc}")
    return metrics, warnings_out


def safe_predict_proba(model: Any, x_data: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_data)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_data)
        return 1.0 / (1.0 + np.exp(-scores))
    raise AttributeError("Model does not expose predict_proba or decision_function.")


def build_models(seed: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            solver="liblinear",
            C=1.0,
            max_iter=2000,
            random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=1,
        ),
    }
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
    except Exception:
        models["HistGradientBoosting"] = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            max_iter=150,
            random_state=seed,
        )
    return models


class FoldwiseFeatureSelector:
    """
    Train-only selector for TCGA:
    1. remove constant genes on the training samples
    2. keep top variance genes on the training samples only
    3. keep top ANOVA F-score genes on the training samples only
    """

    def __init__(
        self,
        x_matrix: np.ndarray,
        y: np.ndarray,
        variance_prefilter_k: int,
        top_k_genes: int,
    ):
        self.x_matrix = x_matrix
        self.y = y
        self.variance_prefilter_k = variance_prefilter_k
        self.top_k_genes = top_k_genes
        self.cache: dict[tuple[int, ...], np.ndarray] = {}

    def select(self, train_idx: np.ndarray) -> np.ndarray:
        key = tuple(train_idx.tolist())
        if key in self.cache:
            return self.cache[key]

        x_train = self.x_matrix[train_idx]
        y_train = self.y[train_idx]
        variances = np.var(x_train, axis=0)
        non_constant = np.flatnonzero(variances > 0.0)
        if len(non_constant) == 0:
            raise ValueError("No non-constant genes in this training fold.")

        if len(non_constant) > self.variance_prefilter_k:
            var_order = np.argsort(variances[non_constant])[
                -self.variance_prefilter_k :
            ]
            variance_kept = non_constant[var_order]
        else:
            variance_kept = non_constant

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores, _ = f_classif(x_train[:, variance_kept], y_train)
        scores = np.asarray(scores, dtype=float)
        scores = np.nan_to_num(
            scores, nan=-np.inf, neginf=-np.inf, posinf=np.finfo(np.float32).max
        )

        k = min(self.top_k_genes, len(variance_kept))
        top_order = np.argsort(scores)[-k:]
        selected = np.sort(variance_kept[top_order])
        self.cache[key] = selected
        return selected


class NumericScaler:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x_train: np.ndarray) -> "NumericScaler":
        self.mean_ = x_train.mean(axis=0)
        scale = x_train.std(axis=0)
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, x_data: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler must be fit before transform.")
        return (x_data - self.mean_) / self.scale_


def load_tcga_dataset(
    config: Config, logger: logging.Logger
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    logger.info("Loading TCGA input from %s", config.input_path)
    df = pd.read_csv(config.input_path)
    raw_shape = df.shape

    keep_levels = [1, 11]
    df = df.loc[df["tumor_level"].isin(keep_levels)].copy()

    site_class_counts = df.groupby("site")["tumor_level"].nunique()
    valid_sites = sorted(site_class_counts[site_class_counts == 2].index.tolist())
    df = df.loc[df["site"].isin(valid_sites)].copy()

    rows_before_dedup = len(df)
    df = (
        df.sort_values(["subject_id", "tumor_level", "sample_barcode"])
        .drop_duplicates(["subject_id", "tumor_level"], keep="first")
        .reset_index(drop=True)
    )

    site_sizes = df["site"].value_counts()
    valid_sites_after_size = sorted(
        site_sizes[site_sizes >= config.min_site_size].index.tolist()
    )
    df = df.loc[df["site"].isin(valid_sites_after_size)].copy().reset_index(drop=True)

    df["target_binary"] = (df["tumor_level"] == 1).astype(int)
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
    gene_cols = [column for column in df.columns if column not in metadata_cols]
    x_matrix = df[gene_cols].to_numpy(dtype=np.float32, copy=True)
    y = df["target_binary"].to_numpy(dtype=np.int32)

    summary = {
        "raw_shape": raw_shape,
        "filtered_shape": df.shape,
        "n_subjects": int(df["subject_id"].nunique()),
        "valid_sites": valid_sites_after_size,
        "rows_before_dedup": rows_before_dedup,
        "rows_removed_by_dedup": int(rows_before_dedup - len(df)),
        "target_counts": df["target_binary"].value_counts().sort_index().to_dict(),
        "site_counts": df["site"].value_counts().sort_index().to_dict(),
    }
    logger.info(
        "TCGA modeling dataset ready: %s rows, %s genes, %s sites",
        len(df),
        len(gene_cols),
        df["site"].nunique(),
    )
    return df, x_matrix, y, gene_cols, summary


def fit_and_predict(
    *,
    model_name: str,
    model: Any,
    selector: FoldwiseFeatureSelector,
    x_matrix: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> tuple[np.ndarray, int]:
    selected_idx = selector.select(train_idx)
    x_train = x_matrix[train_idx][:, selected_idx]
    x_eval = x_matrix[eval_idx][:, selected_idx]

    train_input = x_train
    eval_input = x_eval
    if model_name == "LogisticRegression":
        scaler = NumericScaler().fit(x_train)
        train_input = scaler.transform(x_train)
        eval_input = scaler.transform(x_eval)

    fitted_model = model.__class__(**model.get_params())
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Could not find the number of physical cores.*",
        )
        fitted_model.fit(train_input, y[train_idx])
    y_prob = safe_predict_proba(fitted_model, eval_input)
    return y_prob, int(len(selected_idx))


def make_source_site_splits(
    sites: np.ndarray, train_idx: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    train_sites = sites[train_idx]
    unique_sites = sorted(np.unique(train_sites).tolist())
    splits: list[tuple[np.ndarray, np.ndarray, str]] = []
    for heldout_site in unique_sites:
        mask_val = train_sites == heldout_site
        mask_train = ~mask_val
        inner_train_idx = train_idx[mask_train]
        inner_val_idx = train_idx[mask_val]
        if len(inner_train_idx) == 0 or len(inner_val_idx) == 0:
            continue
        splits.append((inner_train_idx, inner_val_idx, str(heldout_site)))
    return splits


def make_within_site_group_splits(
    site_indices: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    requested_splits: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(site_indices) < 4:
        return []
    groups = subject_ids[site_indices]
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return []
    n_splits = min(requested_splits, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    dummy_x = np.zeros((len(site_indices), 1), dtype=float)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_local, test_local in splitter.split(
        dummy_x, y[site_indices], groups=groups
    ):
        splits.append((site_indices[train_local], site_indices[test_local]))
    return splits


def run_outer_lodo(
    *,
    df: pd.DataFrame,
    x_matrix: np.ndarray,
    y: np.ndarray,
    gene_cols: list[str],
    config: Config,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selector = FoldwiseFeatureSelector(
        x_matrix, y, config.variance_prefilter_k, config.top_k_genes
    )
    models = build_models(config.random_seed)
    sites = df["site"].astype(str).to_numpy()
    sample_barcodes = df["sample_barcode"].astype(str).to_numpy()
    subject_ids = df["subject_id"].astype(str).to_numpy()

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    unique_sites = sorted(np.unique(sites).tolist())

    logger.info(
        "Starting TCGA outer LODO with %s sites: %s",
        len(unique_sites),
        ", ".join(unique_sites),
    )
    for fold_number, test_site in enumerate(unique_sites, start=1):
        test_idx = np.flatnonzero(sites == test_site)
        train_idx = np.flatnonzero(sites != test_site)
        logger.info(
            "Outer fold %s/%s | holdout site=%s | n_train=%s | n_test=%s",
            fold_number,
            len(unique_sites),
            test_site,
            len(train_idx),
            len(test_idx),
        )

        inner_splits = make_source_site_splits(sites, train_idx)

        for model_name, model in models.items():
            warnings_out: list[str] = []
            id_prob_parts: list[np.ndarray] = []
            id_true_parts: list[np.ndarray] = []
            id_meta_rows: list[dict[str, Any]] = []

            for inner_fold_number, (
                inner_train_idx,
                inner_val_idx,
                inner_val_site,
            ) in enumerate(inner_splits, start=1):
                try:
                    id_prob, id_feature_count = fit_and_predict(
                        model_name=model_name,
                        model=model,
                        selector=selector,
                        x_matrix=x_matrix,
                        y=y,
                        train_idx=inner_train_idx,
                        eval_idx=inner_val_idx,
                    )
                except Exception as exc:
                    warnings_out.append(f"inner_fold_{inner_fold_number}_failed:{exc}")
                    continue

                id_true = y[inner_val_idx]
                id_prob_parts.append(id_prob)
                id_true_parts.append(id_true)
                for local_idx, row_idx in enumerate(inner_val_idx):
                    id_meta_rows.append(
                        {
                            "split": "id_inner_source_cv",
                            "model": model_name,
                            "fold": fold_number,
                            "domain": inner_val_site,
                            "sample_barcode": sample_barcodes[row_idx],
                            "subject_id": subject_ids[row_idx],
                            "y_true": int(id_true[local_idx]),
                            "y_prob": float(id_prob[local_idx]),
                            "n_features": id_feature_count,
                        }
                    )

            if id_prob_parts:
                y_id = np.concatenate(id_true_parts)
                prob_id = np.concatenate(id_prob_parts)
                metrics_id, id_metric_warnings = compute_metrics(
                    y_id, prob_id, n_bins=config.calibration_bins
                )
                warnings_out.extend([f"id_{item}" for item in id_metric_warnings])
            else:
                metrics_id = {
                    key: math.nan
                    for key in [
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
                warnings_out.append("id_predictions_unavailable")

            try:
                prob_ood, feature_count = fit_and_predict(
                    model_name=model_name,
                    model=model,
                    selector=selector,
                    x_matrix=x_matrix,
                    y=y,
                    train_idx=train_idx,
                    eval_idx=test_idx,
                )
                y_ood = y[test_idx]
                metrics_ood, ood_metric_warnings = compute_metrics(
                    y_ood, prob_ood, n_bins=config.calibration_bins
                )
                warnings_out.extend([f"ood_{item}" for item in ood_metric_warnings])
            except Exception as exc:
                feature_count = math.nan
                y_ood = y[test_idx]
                prob_ood = np.full(shape=len(test_idx), fill_value=np.nan)
                metrics_ood = {
                    key: math.nan
                    for key in [
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
                warnings_out.append(f"outer_fit_failed:{exc}")

            for row in id_meta_rows:
                prediction_rows.append(row)
            for local_idx, row_idx in enumerate(test_idx):
                prediction_rows.append(
                    {
                        "split": "outer_ood",
                        "model": model_name,
                        "fold": fold_number,
                        "domain": test_site,
                        "sample_barcode": sample_barcodes[row_idx],
                        "subject_id": subject_ids[row_idx],
                        "y_true": int(y_ood[local_idx]),
                        "y_prob": float(prob_ood[local_idx])
                        if not np.isnan(prob_ood[local_idx])
                        else math.nan,
                        "n_features": feature_count,
                    }
                )

            fold_rows.append(
                {
                    "dataset": config.dataset_name,
                    "model": model_name,
                    "test_domain": test_site,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_train_subjects": int(df.iloc[train_idx]["subject_id"].nunique()),
                    "n_test_subjects": int(df.iloc[test_idx]["subject_id"].nunique()),
                    "positive_rate_train": float(np.mean(y[train_idx])),
                    "positive_rate_test": float(np.mean(y[test_idx])),
                    "n_features_selected": feature_count,
                    "auc_id": metrics_id["auc"],
                    "auc_ood": metrics_ood["auc"],
                    "accuracy_id": metrics_id["accuracy"],
                    "accuracy_ood": metrics_ood["accuracy"],
                    "f1_id": metrics_id["f1"],
                    "f1_ood": metrics_ood["f1"],
                    "balacc_id": metrics_id["balacc"],
                    "balacc_ood": metrics_ood["balacc"],
                    "brier_id": metrics_id["brier"],
                    "brier_ood": metrics_ood["brier"],
                    "ece_id": metrics_id["ece"],
                    "ece_ood": metrics_ood["ece"],
                    "precision_id": metrics_id["precision"],
                    "precision_ood": metrics_ood["precision"],
                    "recall_id": metrics_id["recall"],
                    "recall_ood": metrics_ood["recall"],
                    "gap_auc": metrics_id["auc"] - metrics_ood["auc"]
                    if pd.notna(metrics_id["auc"]) and pd.notna(metrics_ood["auc"])
                    else math.nan,
                    "gap_accuracy": metrics_id["accuracy"] - metrics_ood["accuracy"]
                    if pd.notna(metrics_id["accuracy"])
                    and pd.notna(metrics_ood["accuracy"])
                    else math.nan,
                    "gap_f1": metrics_id["f1"] - metrics_ood["f1"]
                    if pd.notna(metrics_id["f1"]) and pd.notna(metrics_ood["f1"])
                    else math.nan,
                    "gap_balacc": metrics_id["balacc"] - metrics_ood["balacc"]
                    if pd.notna(metrics_id["balacc"])
                    and pd.notna(metrics_ood["balacc"])
                    else math.nan,
                    "warnings": aggregate_warning_list(warnings_out),
                }
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)


def run_pairwise_site_transfer(
    *,
    df: pd.DataFrame,
    x_matrix: np.ndarray,
    y: np.ndarray,
    config: Config,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selector = FoldwiseFeatureSelector(
        x_matrix, y, config.variance_prefilter_k, config.top_k_genes
    )
    models = build_models(config.random_seed)
    sites = df["site"].astype(str).to_numpy()
    sample_barcodes = df["sample_barcode"].astype(str).to_numpy()
    subject_ids = df["subject_id"].astype(str).to_numpy()
    unique_sites = sorted(np.unique(sites).tolist())

    logger.info(
        "Starting TCGA train-site -> test-site matrix with %s sites", len(unique_sites)
    )
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for model_name, model in models.items():
        logger.info("Pairwise matrix for model=%s", model_name)
        for train_site in unique_sites:
            train_idx = np.flatnonzero(sites == train_site)

            # Diagonal: within-site grouped CV to avoid within-subject leakage.
            diagonal_splits = make_within_site_group_splits(
                train_idx,
                y=y,
                subject_ids=subject_ids,
                requested_splits=3,
            )
            diag_prob_parts: list[np.ndarray] = []
            diag_true_parts: list[np.ndarray] = []
            diag_feature_counts: list[int] = []
            diag_warnings: list[str] = []
            if diagonal_splits:
                for fold_idx, (diag_train_idx, diag_val_idx) in enumerate(
                    diagonal_splits, start=1
                ):
                    try:
                        diag_prob, diag_feature_count = fit_and_predict(
                            model_name=model_name,
                            model=model,
                            selector=selector,
                            x_matrix=x_matrix,
                            y=y,
                            train_idx=diag_train_idx,
                            eval_idx=diag_val_idx,
                        )
                        diag_true = y[diag_val_idx]
                        diag_prob_parts.append(diag_prob)
                        diag_true_parts.append(diag_true)
                        diag_feature_counts.append(diag_feature_count)
                        for local_idx, row_idx in enumerate(diag_val_idx):
                            prediction_rows.append(
                                {
                                    "split": "pairwise_within_site_cv",
                                    "model": model_name,
                                    "train_domain": train_site,
                                    "test_domain": train_site,
                                    "sample_barcode": sample_barcodes[row_idx],
                                    "subject_id": subject_ids[row_idx],
                                    "y_true": int(diag_true[local_idx]),
                                    "y_prob": float(diag_prob[local_idx]),
                                    "n_features": diag_feature_count,
                                }
                            )
                    except Exception as exc:
                        diag_warnings.append(f"diagonal_fold_{fold_idx}_failed:{exc}")
                if diag_prob_parts:
                    y_diag = np.concatenate(diag_true_parts)
                    prob_diag = np.concatenate(diag_prob_parts)
                    diag_metrics, diag_metric_warnings = compute_metrics(
                        y_diag, prob_diag, n_bins=config.calibration_bins
                    )
                    diag_warnings.extend(diag_metric_warnings)
                    rows.append(
                        {
                            "model": model_name,
                            "train_domain": train_site,
                            "test_domain": train_site,
                            "is_diagonal": True,
                            "n_train": int(len(train_idx)),
                            "n_test": int(len(train_idx)),
                            "positive_rate_train": float(np.mean(y[train_idx])),
                            "positive_rate_test": float(np.mean(y[train_idx])),
                            "n_features_selected": float(np.mean(diag_feature_counts))
                            if diag_feature_counts
                            else math.nan,
                            "auc": diag_metrics["auc"],
                            "accuracy": diag_metrics["accuracy"],
                            "f1": diag_metrics["f1"],
                            "balacc": diag_metrics["balacc"],
                            "precision": diag_metrics["precision"],
                            "recall": diag_metrics["recall"],
                            "brier": diag_metrics["brier"],
                            "ece": diag_metrics["ece"],
                            "warnings": aggregate_warning_list(diag_warnings),
                        }
                    )
            else:
                rows.append(
                    {
                        "model": model_name,
                        "train_domain": train_site,
                        "test_domain": train_site,
                        "is_diagonal": True,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(train_idx)),
                        "positive_rate_train": float(np.mean(y[train_idx])),
                        "positive_rate_test": float(np.mean(y[train_idx])),
                        "n_features_selected": math.nan,
                        "auc": math.nan,
                        "accuracy": math.nan,
                        "f1": math.nan,
                        "balacc": math.nan,
                        "precision": math.nan,
                        "recall": math.nan,
                        "brier": math.nan,
                        "ece": math.nan,
                        "warnings": "no_valid_diagonal_grouped_cv",
                    }
                )

            for test_site in unique_sites:
                if test_site == train_site:
                    continue
                test_idx = np.flatnonzero(sites == test_site)
                warnings_out: list[str] = []
                try:
                    prob_ood, feature_count = fit_and_predict(
                        model_name=model_name,
                        model=model,
                        selector=selector,
                        x_matrix=x_matrix,
                        y=y,
                        train_idx=train_idx,
                        eval_idx=test_idx,
                    )
                    metrics, metric_warnings = compute_metrics(
                        y[test_idx], prob_ood, n_bins=config.calibration_bins
                    )
                    warnings_out.extend(metric_warnings)
                except Exception as exc:
                    prob_ood = np.full(shape=len(test_idx), fill_value=np.nan)
                    feature_count = math.nan
                    metrics = {
                        key: math.nan
                        for key in [
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
                    warnings_out.append(f"pairwise_fit_failed:{exc}")

                for local_idx, row_idx in enumerate(test_idx):
                    prediction_rows.append(
                        {
                            "split": "pairwise_ood",
                            "model": model_name,
                            "train_domain": train_site,
                            "test_domain": test_site,
                            "sample_barcode": sample_barcodes[row_idx],
                            "subject_id": subject_ids[row_idx],
                            "y_true": int(y[row_idx]),
                            "y_prob": float(prob_ood[local_idx])
                            if not np.isnan(prob_ood[local_idx])
                            else math.nan,
                            "n_features": feature_count,
                        }
                    )

                rows.append(
                    {
                        "model": model_name,
                        "train_domain": train_site,
                        "test_domain": test_site,
                        "is_diagonal": False,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "positive_rate_train": float(np.mean(y[train_idx])),
                        "positive_rate_test": float(np.mean(y[test_idx])),
                        "n_features_selected": feature_count,
                        "auc": metrics["auc"],
                        "accuracy": metrics["accuracy"],
                        "f1": metrics["f1"],
                        "balacc": metrics["balacc"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "brier": metrics["brier"],
                        "ece": metrics["ece"],
                        "warnings": aggregate_warning_list(warnings_out),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def run_training_domain_diversity(
    *,
    df: pd.DataFrame,
    x_matrix: np.ndarray,
    y: np.ndarray,
    config: Config,
    logger: logging.Logger,
) -> pd.DataFrame:
    selector = FoldwiseFeatureSelector(
        x_matrix, y, config.variance_prefilter_k, config.top_k_genes
    )
    models = build_models(config.random_seed)
    sites = df["site"].astype(str).to_numpy()
    unique_sites = sorted(np.unique(sites).tolist())
    rows: list[dict[str, Any]] = []

    logger.info(
        "Starting grouped training-domain diversity analysis for %s sites",
        len(unique_sites),
    )
    for holdout_site in unique_sites:
        source_sites = [site for site in unique_sites if site != holdout_site]
        train_sizes = sorted({1, min(3, len(source_sites)), len(source_sites)})
        test_idx = np.flatnonzero(sites == holdout_site)

        for train_size in train_sizes:
            for source_combo in itertools.combinations(source_sites, train_size):
                combo_label = " + ".join(source_combo)
                train_mask = np.isin(sites, list(source_combo))
                train_idx = np.flatnonzero(train_mask)

                for model_name, model in models.items():
                    warnings_out: list[str] = []
                    try:
                        prob_ood, feature_count = fit_and_predict(
                            model_name=model_name,
                            model=model,
                            selector=selector,
                            x_matrix=x_matrix,
                            y=y,
                            train_idx=train_idx,
                            eval_idx=test_idx,
                        )
                        metrics, metric_warnings = compute_metrics(
                            y[test_idx], prob_ood, n_bins=config.calibration_bins
                        )
                        warnings_out.extend(metric_warnings)
                    except Exception as exc:
                        prob_ood = np.full(shape=len(test_idx), fill_value=np.nan)
                        feature_count = math.nan
                        metrics = {
                            key: math.nan
                            for key in [
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
                        warnings_out.append(f"combo_fit_failed:{exc}")

                    rows.append(
                        {
                            "model": model_name,
                            "holdout_site": holdout_site,
                            "n_train_sites": train_size,
                            "training_sites": combo_label,
                            "n_train": int(len(train_idx)),
                            "n_test": int(len(test_idx)),
                            "positive_rate_train": float(np.mean(y[train_idx])),
                            "positive_rate_test": float(np.mean(y[test_idx])),
                            "n_features_selected": feature_count,
                            "auc": metrics["auc"],
                            "accuracy": metrics["accuracy"],
                            "f1": metrics["f1"],
                            "balacc": metrics["balacc"],
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "brier": metrics["brier"],
                            "ece": metrics["ece"],
                            "warnings": aggregate_warning_list(warnings_out),
                        }
                    )
    return pd.DataFrame(rows)


def summarise_outer_results(
    fold_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_summary = (
        fold_results.groupby("model", dropna=False)
        .agg(
            n_folds_total=("test_domain", "size"),
            n_folds_valid_auc_ood=("auc_ood", lambda s: int(s.notna().sum())),
            n_features_selected_mean=("n_features_selected", "mean"),
            auc_id_mean=("auc_id", "mean"),
            auc_id_std=("auc_id", "std"),
            auc_ood_mean=("auc_ood", "mean"),
            auc_ood_std=("auc_ood", "std"),
            accuracy_id_mean=("accuracy_id", "mean"),
            accuracy_id_std=("accuracy_id", "std"),
            accuracy_ood_mean=("accuracy_ood", "mean"),
            accuracy_ood_std=("accuracy_ood", "std"),
            f1_id_mean=("f1_id", "mean"),
            f1_id_std=("f1_id", "std"),
            f1_ood_mean=("f1_ood", "mean"),
            f1_ood_std=("f1_ood", "std"),
            balacc_ood_mean=("balacc_ood", "mean"),
            balacc_ood_std=("balacc_ood", "std"),
            brier_ood_mean=("brier_ood", "mean"),
            brier_ood_std=("brier_ood", "std"),
            ece_ood_mean=("ece_ood", "mean"),
            ece_ood_std=("ece_ood", "std"),
            gap_auc_mean=("gap_auc", "mean"),
            gap_accuracy_mean=("gap_accuracy", "mean"),
            gap_f1_mean=("gap_f1", "mean"),
        )
        .reset_index()
        .sort_values("auc_ood_mean", ascending=False)
    )

    domain_summary = fold_results.sort_values(["test_domain", "model"]).reset_index(
        drop=True
    )

    domain_difficulty = (
        fold_results.groupby("test_domain", dropna=False)
        .agg(
            auc_ood_mean=("auc_ood", "mean"),
            accuracy_ood_mean=("accuracy_ood", "mean"),
            f1_ood_mean=("f1_ood", "mean"),
            balacc_ood_mean=("balacc_ood", "mean"),
            n_models=("model", "size"),
        )
        .reset_index()
        .sort_values(["accuracy_ood_mean", "auc_ood_mean"], ascending=[True, True])
        .reset_index(drop=True)
    )
    domain_difficulty["difficulty_rank_accuracy"] = np.arange(
        1, len(domain_difficulty) + 1
    )
    return model_summary, domain_summary, domain_difficulty


def summarise_pairwise_results(
    pairwise_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_summary = (
        pairwise_results.loc[~pairwise_results["is_diagonal"]]
        .groupby("model", dropna=False)
        .agg(
            n_pairs=("test_domain", "size"),
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
        .sort_values("auc_mean", ascending=False)
    )
    test_summary = (
        pairwise_results.loc[~pairwise_results["is_diagonal"]]
        .groupby(["test_domain", "model"], dropna=False)
        .agg(
            n_pairs=("train_domain", "size"),
            auc_mean=("auc", "mean"),
            accuracy_mean=("accuracy", "mean"),
            f1_mean=("f1", "mean"),
            balacc_mean=("balacc", "mean"),
        )
        .reset_index()
        .sort_values(["test_domain", "model"])
    )
    return model_summary, test_summary


def summarise_combo_results(
    combo_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_model = (
        combo_results.groupby(["model", "n_train_sites"], dropna=False)
        .agg(
            n_combos=("training_sites", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            balacc_mean=("balacc", "mean"),
            balacc_std=("balacc", "std"),
        )
        .reset_index()
        .sort_values(["model", "n_train_sites"])
    )
    by_site = (
        combo_results.groupby(["holdout_site", "model", "n_train_sites"], dropna=False)
        .agg(
            n_combos=("training_sites", "size"),
            accuracy_mean=("accuracy", "mean"),
            auc_mean=("auc", "mean"),
            f1_mean=("f1", "mean"),
            balacc_mean=("balacc", "mean"),
        )
        .reset_index()
        .sort_values(["holdout_site", "model", "n_train_sites"])
    )
    return by_model, by_site


def annotate_heatmap(ax: plt.Axes, matrix: np.ndarray, fmt: str = ".3f") -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                text = "NA"
            else:
                text = format(float(value), fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color="black")


def plot_grouped_bar(
    df: pd.DataFrame,
    *,
    x_col: str,
    group_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    good_high: bool = True,
    save_pdf: bool = True,
) -> None:
    if df.empty:
        return
    x_levels = list(df[x_col].dropna().astype(str).unique())
    group_levels = list(df[group_col].dropna().astype(str).unique())
    pivot = df.pivot(index=x_col, columns=group_col, values=value_col).reindex(
        index=x_levels, columns=group_levels
    )
    x_positions = np.arange(len(x_levels))
    width = 0.8 / max(len(group_levels), 1)
    cmap = good_cmap() if good_high else bad_cmap()

    fig, ax = plt.subplots(figsize=(max(10, len(x_levels) * 1.35), 6.8))
    for group_index, group_name in enumerate(group_levels):
        values = pivot[group_name].to_numpy(dtype=float)
        offsets = x_positions - 0.4 + width / 2 + group_index * width
        colors = [
            cmap(float(value) if pd.notna(value) else 0.0)
            for value in np.nan_to_num(values, nan=0.0)
        ]
        bars = ax.bar(
            offsets,
            values,
            width=width,
            color=colors,
            edgecolor="#333333",
            linewidth=0.8,
            label=group_name,
        )
        for bar, value in zip(bars, values):
            if pd.notna(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_levels, rotation=0, fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlabel("Holdout site", fontsize=13)
    ax.set_title(title, fontsize=18, pad=14)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(title="Model", frameon=False, fontsize=10, title_fontsize=11)
    save_figure(fig, output_path, save_pdf)


def plot_simple_heatmap(
    df: pd.DataFrame,
    *,
    row_col: str,
    col_col: str,
    value_col: str,
    title: str,
    cbar_label: str,
    output_path: Path,
    save_pdf: bool = True,
) -> None:
    if df.empty:
        return
    row_levels = list(df[row_col].dropna().astype(str).unique())
    col_levels = list(df[col_col].dropna().astype(str).unique())
    matrix = (
        df.pivot(index=row_col, columns=col_col, values=value_col)
        .reindex(index=row_levels, columns=col_levels)
        .to_numpy(dtype=float)
    )
    fig, ax = plt.subplots(
        figsize=(max(6, len(col_levels) * 1.3), max(4, len(row_levels) * 0.75))
    )
    image = ax.imshow(matrix, aspect="auto", cmap=good_cmap(), vmin=0.0, vmax=1.0)
    annotate_heatmap(ax, matrix)
    ax.set_xticks(np.arange(len(col_levels)))
    ax.set_xticklabels(col_levels, rotation=0, fontsize=11)
    ax.set_yticks(np.arange(len(row_levels)))
    ax.set_yticklabels(row_levels, fontsize=11)
    ax.set_xlabel(col_col.replace("_", " ").title(), fontsize=13)
    ax.set_ylabel(row_col.replace("_", " ").title(), fontsize=13)
    ax.set_title(title, fontsize=18, pad=14)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label(cbar_label, fontsize=11)
    save_figure(fig, output_path, save_pdf)


def plot_site_model_ood_heatmap(
    fold_results: pd.DataFrame, output_path: Path, save_pdf: bool = True
) -> None:
    if fold_results.empty:
        return
    metrics = [
        ("accuracy_ood", "Accuracy"),
        ("balacc_ood", "Balanced accuracy"),
        ("f1_ood", "F1-score"),
    ]
    sites = sorted(fold_results["test_domain"].dropna().astype(str).unique().tolist())
    models = [
        pretty_model_name(value)
        for value in fold_results["model"].dropna().astype(str).unique().tolist()
    ]
    model_map = {
        value: pretty_model_name(value)
        for value in fold_results["model"].dropna().astype(str).unique().tolist()
    }
    plot_df = fold_results.copy()
    plot_df["model_display"] = plot_df["model"].map(model_map)

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(5.8 * len(metrics), max(5.8, len(sites) * 1.0)),
        sharey=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_col, metric_label) in zip(axes, metrics):
        matrix = (
            plot_df.pivot(
                index="test_domain", columns="model_display", values=metric_col
            )
            .reindex(index=sites, columns=models)
            .to_numpy(dtype=float)
        )
        image = ax.imshow(matrix, aspect="auto", cmap=good_cmap(), vmin=0.0, vmax=1.0)
        annotate_heatmap(ax, matrix)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=11)
        ax.set_yticks(np.arange(len(sites)))
        ax.set_yticklabels(sites, fontsize=11)
        ax.set_title(metric_col, fontsize=13, pad=10)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Holdout site", fontsize=12)

    cbar = fig.colorbar(image, ax=axes, shrink=0.82)
    cbar.set_label("Performance", fontsize=11)
    fig.suptitle("Heatmap: site x model OOD performance", fontsize=19, y=1.02)
    save_figure(fig, output_path, save_pdf)


def plot_pairwise_heatmaps(
    pairwise_results: pd.DataFrame, output_dir: Path, save_pdf: bool = True
) -> None:
    if pairwise_results.empty:
        return
    models = sorted(pairwise_results["model"].dropna().astype(str).unique().tolist())
    train_sites = sorted(
        pairwise_results["train_domain"].dropna().astype(str).unique().tolist()
    )
    test_sites = sorted(
        pairwise_results["test_domain"].dropna().astype(str).unique().tolist()
    )
    metrics = [("accuracy", "Accuracy"), ("auc", "ROC-AUC"), ("f1", "F1-score")]

    for model_name in models:
        model_df = pairwise_results.loc[pairwise_results["model"] == model_name].copy()
        for metric_col, metric_label in metrics:
            matrix = (
                model_df.pivot(
                    index="train_domain", columns="test_domain", values=metric_col
                )
                .reindex(index=train_sites, columns=test_sites)
                .to_numpy(dtype=float)
            )
            fig, ax = plt.subplots(
                figsize=(max(7, len(test_sites) * 1.5), max(6, len(train_sites) * 1.3))
            )
            image = ax.imshow(
                matrix, aspect="auto", cmap=good_cmap(), vmin=0.0, vmax=1.0
            )
            annotate_heatmap(ax, matrix)
            ax.set_xticks(np.arange(len(test_sites)))
            ax.set_xticklabels(test_sites, rotation=25, ha="right", fontsize=11)
            ax.set_yticks(np.arange(len(train_sites)))
            ax.set_yticklabels(train_sites, fontsize=11)
            ax.set_xlabel("Test site", fontsize=13)
            ax.set_ylabel("Train site", fontsize=13)
            ax.set_title(
                f"{metric_label}: train site -> test site ({pretty_model_name(model_name)})",
                fontsize=17,
                pad=14,
            )
            cbar = fig.colorbar(image, ax=ax, shrink=0.82)
            cbar.set_label(metric_label, fontsize=11)
            save_figure(
                fig,
                output_dir
                / f"train_test_heatmap_{metric_col}_{sanitize_name(model_name)}",
                save_pdf,
            )


def plot_calibration_curves(
    prediction_df: pd.DataFrame, output_dir: Path, save_pdf: bool = True
) -> pd.DataFrame:
    outer_df = prediction_df.loc[prediction_df["split"] == "outer_ood"].copy()
    curve_rows: list[dict[str, Any]] = []
    for model_name in sorted(outer_df["model"].dropna().astype(str).unique().tolist()):
        model_df = (
            outer_df.loc[outer_df["model"] == model_name]
            .dropna(subset=["y_prob"])
            .copy()
        )
        if model_df.empty:
            continue
        frac_pos, mean_pred = calibration_curve(
            model_df["y_true"].to_numpy(dtype=int),
            model_df["y_prob"].to_numpy(dtype=float),
            n_bins=10,
            strategy="uniform",
        )
        fig, ax = plt.subplots(figsize=(6.8, 6.2))
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="#666666",
            linewidth=1.2,
            label="Perfect calibration",
        )
        ax.plot(
            mean_pred,
            frac_pos,
            marker="o",
            linewidth=2,
            color=good_cmap()(0.78),
            label=pretty_model_name(model_name),
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Mean predicted probability", fontsize=13)
        ax.set_ylabel("Observed positive fraction", fontsize=13)
        ax.set_title(
            f"Calibration curve: {pretty_model_name(model_name)}", fontsize=17, pad=14
        )
        ax.grid(linestyle="--", alpha=0.25)
        ax.legend(frameon=False, fontsize=10)
        save_figure(
            fig, output_dir / f"calibration_{sanitize_name(model_name)}", save_pdf
        )
        for bin_idx, (x_value, y_value) in enumerate(zip(mean_pred, frac_pos), start=1):
            curve_rows.append(
                {
                    "model": model_name,
                    "bin_index": bin_idx,
                    "mean_predicted_probability": float(x_value),
                    "observed_positive_fraction": float(y_value),
                }
            )
    return pd.DataFrame(curve_rows)


def plot_domain_difficulty(
    domain_difficulty: pd.DataFrame, output_path: Path, save_pdf: bool = True
) -> None:
    if domain_difficulty.empty:
        return
    plot_df = domain_difficulty.sort_values("accuracy_ood_mean", ascending=True).copy()
    x = np.arange(len(plot_df))
    values = plot_df["accuracy_ood_mean"].to_numpy(dtype=float)
    colors = [good_cmap()(value) for value in values]
    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 1.3), 6.2))
    bars = ax.bar(x, values, color=colors, edgecolor="#333333", linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["test_domain"].tolist(), fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Holdout site", fontsize=13)
    ax.set_ylabel("Mean OOD accuracy", fontsize=13)
    ax.set_title(
        "Ranking of holdout-site difficulty by mean OOD accuracy", fontsize=18, pad=14
    )
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    save_figure(fig, output_path, save_pdf)


def plot_training_domain_by_site_heatmap(
    combo_summary_by_site: pd.DataFrame, output_path: Path, save_pdf: bool = True
) -> None:
    if combo_summary_by_site.empty:
        return
    models = sorted(
        combo_summary_by_site["model"].dropna().astype(str).unique().tolist()
    )
    holdout_sites = sorted(
        combo_summary_by_site["holdout_site"].dropna().astype(str).unique().tolist()
    )
    train_sizes = sorted(
        combo_summary_by_site["n_train_sites"].dropna().astype(int).unique().tolist()
    )

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(5.5 * len(models), max(5.5, len(holdout_sites) * 1.0)),
        sharey=True,
    )
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        model_df = combo_summary_by_site.loc[
            combo_summary_by_site["model"] == model_name
        ].copy()
        matrix = (
            model_df.pivot(
                index="holdout_site", columns="n_train_sites", values="accuracy_mean"
            )
            .reindex(index=holdout_sites, columns=train_sizes)
            .to_numpy(dtype=float)
        )
        image = ax.imshow(matrix, aspect="auto", cmap=good_cmap(), vmin=0.0, vmax=1.0)
        annotate_heatmap(ax, matrix)
        ax.set_xticks(np.arange(len(train_sizes)))
        ax.set_xticklabels(train_sizes, fontsize=11)
        ax.set_yticks(np.arange(len(holdout_sites)))
        ax.set_yticklabels(holdout_sites, fontsize=11)
        ax.set_xlabel("Number of training sites", fontsize=12)
        ax.set_ylabel("Holdout test site", fontsize=12)
        ax.set_title(sanitize_name(model_name), fontsize=13, pad=10)

    cbar = fig.colorbar(image, ax=axes, shrink=0.82)
    cbar.set_label("Mean accuracy", fontsize=11)
    fig.suptitle(
        "Transportability improvement by holdout site as training domains increase",
        fontsize=19,
        y=1.02,
    )
    save_figure(fig, output_path, save_pdf)


def plot_training_domain_summary_heatmap(
    combo_summary_by_model: pd.DataFrame, output_path: Path, save_pdf: bool = True
) -> None:
    if combo_summary_by_model.empty:
        return
    models = sorted(
        combo_summary_by_model["model"].dropna().astype(str).unique().tolist()
    )
    train_sizes = sorted(
        combo_summary_by_model["n_train_sites"].dropna().astype(int).unique().tolist()
    )
    matrix = (
        combo_summary_by_model.pivot(
            index="model", columns="n_train_sites", values="accuracy_mean"
        )
        .reindex(index=models, columns=train_sizes)
        .to_numpy(dtype=float)
    )
    fig, ax = plt.subplots(
        figsize=(max(8, len(train_sizes) * 2.2), max(5, len(models) * 1.2))
    )
    image = ax.imshow(matrix, aspect="auto", cmap=good_cmap(), vmin=0.0, vmax=1.0)
    annotate_heatmap(ax, matrix)
    ax.set_xticks(np.arange(len(train_sizes)))
    ax.set_xticklabels(train_sizes, fontsize=11)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([sanitize_name(model) for model in models], fontsize=11)
    ax.set_xlabel("Number of training sites", fontsize=13)
    ax.set_ylabel("Model", fontsize=13)
    ax.set_title(
        "Effect of training domain diversity on transportability", fontsize=19, pad=14
    )
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label("Mean accuracy", fontsize=11)
    save_figure(fig, output_path, save_pdf)


def plot_training_domain_all_combinations(
    combo_results: pd.DataFrame, output_path: Path, save_pdf: bool = True
) -> None:
    if combo_results.empty:
        return
    holdout_sites = sorted(
        combo_results["holdout_site"].dropna().astype(str).unique().tolist()
    )
    models = sorted(combo_results["model"].dropna().astype(str).unique().tolist())
    n_sites = len(holdout_sites)
    fig, axes = plt.subplots(
        n_sites, 1, figsize=(16, max(10, n_sites * 3.5)), sharex=False
    )
    if n_sites == 1:
        axes = [axes]

    for ax, holdout_site in zip(axes, holdout_sites):
        site_df = combo_results.loc[
            combo_results["holdout_site"] == holdout_site
        ].copy()
        site_df = site_df.sort_values(["n_train_sites", "training_sites"])
        row_labels = [
            f"Train {int(row.n_train_sites)} site(s) | {row.training_sites}"
            for row in site_df.itertuples(index=False)
        ]
        matrix = (
            site_df.pivot_table(
                index=["n_train_sites", "training_sites"],
                columns="model",
                values="accuracy",
                aggfunc="mean",
            )
            .reset_index()
            .sort_values(["n_train_sites", "training_sites"])
            .set_index(["n_train_sites", "training_sites"])
            .reindex(columns=models)
            .to_numpy(dtype=float)
        )
        image = ax.imshow(matrix, aspect="auto", cmap=good_cmap(), vmin=0.0, vmax=1.0)
        annotate_heatmap(ax, matrix)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels([sanitize_name(model) for model in models], fontsize=11)
        ax.set_ylabel(holdout_site, fontsize=12, rotation=270, labelpad=18)

    axes[0].set_title(
        "Grouped training-site combinations for each held-out TCGA site",
        fontsize=19,
        pad=14,
    )
    axes[-1].set_xlabel("Model", fontsize=13)
    cbar = fig.colorbar(image, ax=axes, shrink=0.82)
    cbar.set_label("Accuracy", fontsize=11)
    save_figure(fig, output_path, save_pdf)


def write_report(
    *,
    output_dir: Path,
    config: Config,
    dataset_summary: dict[str, Any],
    fold_results: pd.DataFrame,
    model_summary: pd.DataFrame,
    domain_difficulty: pd.DataFrame,
    pairwise_model_summary: pd.DataFrame,
    pairwise_test_summary: pd.DataFrame,
    combo_summary_by_model: pd.DataFrame,
) -> None:
    hardest_domains = domain_difficulty.head(5)
    easiest_domains = domain_difficulty.sort_values(
        "accuracy_ood_mean", ascending=False
    ).head(5)
    warning_count = (
        int(fold_results["warnings"].fillna("").ne("").sum())
        if not fold_results.empty
        else 0
    )
    best_model = model_summary.iloc[0]["model"] if not model_summary.empty else "NA"
    best_auc = (
        model_summary.iloc[0]["auc_ood_mean"] if not model_summary.empty else math.nan
    )

    if pd.notna(best_auc) and best_auc >= 0.98:
        automatic_comment = (
            "The baseline appears extremely strong. For TCGA-BRCA this likely reflects the biological ease of separating "
            "primary tumour (01) from solid tissue normal (11), even across held-out sites."
        )
    elif pd.notna(best_auc) and best_auc < 0.8:
        automatic_comment = "The baseline shows limited raw transportability, with a clear performance drop on held-out sites."
    else:
        automatic_comment = "The baseline shows intermediate cross-site transportability with non-trivial domain sensitivity."

    lines = [
        f"# Baseline M0 Report: {config.dataset_name}",
        "",
        "## Dataset",
        f"- Input path: `{config.input_path}`",
        f"- Raw shape: `{dataset_summary['raw_shape'][0]} rows x {dataset_summary['raw_shape'][1]} columns`",
        f"- Modeling shape: `{dataset_summary['filtered_shape'][0]} rows x {dataset_summary['filtered_shape'][1]} columns`",
        f"- Modeling subjects: `{dataset_summary['n_subjects']}`",
        f"- Valid sites: `{', '.join(dataset_summary['valid_sites'])}`",
        f"- Rows removed by subject_id+tumor_level deduplication: `{dataset_summary['rows_removed_by_dedup']}`",
        f"- Binary target: `01 -> 1`, `11 -> 0`",
        "",
        "## Methodological Choices",
        "- Rebuilt from scratch for TCGA only.",
        "- Restricted to tumour levels `01` and `11`; excluded `06` due to very low frequency.",
        "- Kept only sites containing both binary classes to avoid undefined test metrics.",
        "- Deduplicated repeated samples within the same `subject_id + tumor_level` combination by lexicographic sample barcode.",
        "- All feature filtering is training-only inside each fit.",
        f"- Fold-wise variance prefilter: top `{config.variance_prefilter_k}` genes by training variance.",
        f"- Fold-wise ANOVA filter: top `{config.top_k_genes}` genes by training F-score.",
        "- No ComBat, no harmonization, no domain adaptation, no reweighting.",
        "",
        "## Outer LODO Model Summary",
        "```text",
        model_summary.round(4).to_string(index=False),
        "```",
        "",
        "## Hardest Holdout Sites",
        "```text",
        hardest_domains.round(4).to_string(index=False),
        "```",
        "",
        "## Easiest Holdout Sites",
        "```text",
        easiest_domains.round(4).to_string(index=False),
        "```",
        "",
        "## Pairwise Train->Test Site Summary",
        "```text",
        pairwise_model_summary.round(4).to_string(index=False),
        "```",
        "",
        "## Pairwise Test-Site Summary",
        "```text",
        pairwise_test_summary.round(4).to_string(index=False),
        "```",
        "",
        "## Training-Domain Diversity Summary",
        "```text",
        combo_summary_by_model.round(4).to_string(index=False),
        "```",
        "",
        "## Warning Summary",
        f"- Outer folds with recorded warnings: `{warning_count}`",
        "",
        "## Automatic Comment",
        f"- {automatic_comment}",
        "",
        "## Main Conclusion",
        f"- `{best_model}` is the strongest baseline under the raw M0 TCGA setting.",
        "- If performance remains near-perfect, the main interpretation should emphasise task ease rather than transportability robustness.",
        "",
    ]
    (output_dir / "baseline_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_html_report(markdown_path: Path, html_path: Path) -> None:
    markdown_text = markdown_path.read_text(encoding="utf-8")
    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TCGA Baseline M0 Report</title>
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
    output_dir: Path,
    config: Config,
    model_summary: pd.DataFrame,
    dataset_summary: dict[str, Any],
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
    lines = [
        "# TCGA-BRCA Baseline M0",
        "",
        "This folder contains a fresh TCGA baseline rebuild for cross-site transportability.",
        "",
        "## Dataset",
        f"- Input: `{config.input_path}`",
        f"- Modeling rows: `{dataset_summary['filtered_shape'][0]}`",
        f"- Valid sites: `{', '.join(dataset_summary['valid_sites'])}`",
        "",
        "## What Was Done",
        "- Binary task rebuilt from scratch as `01 vs 11`.",
        "- Sites without both classes were removed before evaluation.",
        "- Repeated samples within the same `subject_id + tumor_level` were deduplicated.",
        "- Feature filtering is fit on training data only.",
        "- Pairwise train-site -> test-site heatmaps and grouped training-domain transportability plots are included.",
        "",
        "## Best Models",
        f"- Best mean OOD AUC: `{best_auc_model}`",
        f"- Best mean OOD accuracy: `{best_accuracy_model}`",
        "",
        "## Main Files",
        "- `tcga_m0_from_scratch.py`: single runnable script.",
        "- `baseline_report.md`: markdown report.",
        "- `model_summary.csv`: aggregate LODO results.",
        "- `train_test_site_results.csv`: train-site -> test-site transfer matrix.",
        "- `training_domain_experiment_results.csv`: grouped training-domain diversity experiments.",
        "",
        "## Run",
        "```bash",
        f'cd "{output_dir}"',
        "python3 tcga_m0_from_scratch.py",
        "```",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_csvs(
    *,
    output_dir: Path,
    config: Config,
    dataset_summary: dict[str, Any],
    fold_results: pd.DataFrame,
    prediction_records: pd.DataFrame,
    model_summary: pd.DataFrame,
    domain_summary: pd.DataFrame,
    domain_difficulty: pd.DataFrame,
    pairwise_results: pd.DataFrame,
    pairwise_predictions: pd.DataFrame,
    pairwise_model_summary: pd.DataFrame,
    pairwise_test_summary: pd.DataFrame,
    combo_results: pd.DataFrame,
    combo_summary_by_model: pd.DataFrame,
    combo_summary_by_site: pd.DataFrame,
    calibration_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
) -> None:
    fold_results.to_csv(output_dir / "fold_results.csv", index=False)
    prediction_records.to_csv(output_dir / "prediction_records.csv", index=False)
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    domain_summary.to_csv(output_dir / "domain_summary.csv", index=False)
    domain_difficulty.to_csv(output_dir / "domain_difficulty_summary.csv", index=False)
    pairwise_results.to_csv(output_dir / "train_test_site_results.csv", index=False)
    pairwise_predictions.to_csv(
        output_dir / "train_test_site_prediction_records.csv", index=False
    )
    pairwise_model_summary.to_csv(
        output_dir / "train_test_site_model_summary.csv", index=False
    )
    pairwise_test_summary.to_csv(
        output_dir / "train_test_site_test_domain_summary.csv", index=False
    )
    combo_results.to_csv(
        output_dir / "training_domain_experiment_results.csv", index=False
    )
    combo_summary_by_model.to_csv(
        output_dir / "training_domain_experiment_summary_by_model.csv", index=False
    )
    combo_summary_by_site.to_csv(
        output_dir / "training_domain_experiment_summary_by_site.csv", index=False
    )
    calibration_df.to_csv(output_dir / "calibration_curve_data.csv", index=False)
    dataset_df.to_csv(output_dir / "tcga_modeling_input.csv", index=False)
    pd.DataFrame([dataset_summary]).to_csv(
        output_dir / "tcga_filtering_summary.csv", index=False
    )
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )


def main() -> None:
    config = Config()
    output_dir = ensure_dir(config.output_dir)
    logger = init_logger(output_dir / "logs.txt")

    logger.info("Starting fresh TCGA baseline rebuild from scratch.")
    dataset_df, x_matrix, y, gene_cols, dataset_summary = load_tcga_dataset(
        config, logger
    )

    fold_results, prediction_records = run_outer_lodo(
        df=dataset_df,
        x_matrix=x_matrix,
        y=y,
        gene_cols=gene_cols,
        config=config,
        logger=logger,
    )
    model_summary, domain_summary, domain_difficulty = summarise_outer_results(
        fold_results
    )

    pairwise_results, pairwise_predictions = run_pairwise_site_transfer(
        df=dataset_df,
        x_matrix=x_matrix,
        y=y,
        config=config,
        logger=logger,
    )
    pairwise_model_summary, pairwise_test_summary = summarise_pairwise_results(
        pairwise_results
    )

    combo_results = run_training_domain_diversity(
        df=dataset_df,
        x_matrix=x_matrix,
        y=y,
        config=config,
        logger=logger,
    )
    combo_summary_by_model, combo_summary_by_site = summarise_combo_results(
        combo_results
    )

    plot_df = fold_results.copy()
    plot_df["model_display"] = plot_df["model"].map(pretty_model_name)

    plot_grouped_bar(
        plot_df,
        x_col="test_domain",
        group_col="model_display",
        value_col="accuracy_ood",
        title="Out-of-domain accuracy by holdout site and model",
        ylabel="Accuracy",
        output_path=output_dir / "accuracy_ood_by_domain",
        good_high=True,
        save_pdf=config.save_pdf_plots,
    )
    plot_grouped_bar(
        plot_df,
        x_col="test_domain",
        group_col="model_display",
        value_col="auc_ood",
        title="Out-of-domain ROC-AUC by holdout site and model",
        ylabel="ROC-AUC",
        output_path=output_dir / "auc_ood_by_domain",
        good_high=True,
        save_pdf=config.save_pdf_plots,
    )
    plot_grouped_bar(
        plot_df,
        x_col="test_domain",
        group_col="model_display",
        value_col="f1_ood",
        title="Out-of-domain F1-score by holdout site and model",
        ylabel="F1-score",
        output_path=output_dir / "f1_ood_by_domain",
        good_high=True,
        save_pdf=config.save_pdf_plots,
    )
    plot_grouped_bar(
        plot_df,
        x_col="test_domain",
        group_col="model_display",
        value_col="gap_accuracy",
        title="ID-OOD accuracy gap by holdout site and model",
        ylabel="Gap (ID - OOD)",
        output_path=output_dir / "gap_accuracy_by_domain",
        good_high=False,
        save_pdf=config.save_pdf_plots,
    )
    plot_grouped_bar(
        plot_df,
        x_col="test_domain",
        group_col="model_display",
        value_col="gap_auc",
        title="ID-OOD ROC-AUC gap by holdout site and model",
        ylabel="Gap (ID - OOD)",
        output_path=output_dir / "gap_auc_by_domain",
        good_high=False,
        save_pdf=config.save_pdf_plots,
    )
    plot_grouped_bar(
        plot_df,
        x_col="test_domain",
        group_col="model_display",
        value_col="gap_f1",
        title="ID-OOD F1 gap by holdout site and model",
        ylabel="Gap (ID - OOD)",
        output_path=output_dir / "gap_f1_by_domain",
        good_high=False,
        save_pdf=config.save_pdf_plots,
    )

    mean_perf_df = model_summary.melt(
        id_vars=["model"],
        value_vars=[
            "auc_ood_mean",
            "accuracy_ood_mean",
            "f1_ood_mean",
            "balacc_ood_mean",
        ],
        var_name="metric",
        value_name="value",
    )
    mean_perf_df["metric"] = (
        mean_perf_df["metric"].str.replace("_ood_mean", "", regex=False).str.upper()
    )
    mean_perf_df["model_display"] = mean_perf_df["model"].map(pretty_model_name)
    plot_grouped_bar(
        mean_perf_df,
        x_col="metric",
        group_col="model_display",
        value_col="value",
        title="Mean out-of-domain performance by model",
        ylabel="Score",
        output_path=output_dir / "mean_performance_by_model",
        good_high=True,
        save_pdf=config.save_pdf_plots,
    )

    plot_simple_heatmap(
        plot_df,
        row_col="test_domain",
        col_col="model_display",
        value_col="accuracy_ood",
        title="Heatmap: OOD accuracy",
        cbar_label="Accuracy",
        output_path=output_dir / "heatmap_accuracy_ood",
        save_pdf=config.save_pdf_plots,
    )
    plot_simple_heatmap(
        plot_df,
        row_col="test_domain",
        col_col="model_display",
        value_col="auc_ood",
        title="Heatmap: OOD ROC-AUC",
        cbar_label="ROC-AUC",
        output_path=output_dir / "heatmap_auc_ood",
        save_pdf=config.save_pdf_plots,
    )
    plot_simple_heatmap(
        plot_df,
        row_col="test_domain",
        col_col="model_display",
        value_col="f1_ood",
        title="Heatmap: OOD F1-score",
        cbar_label="F1-score",
        output_path=output_dir / "heatmap_f1_ood",
        save_pdf=config.save_pdf_plots,
    )
    plot_site_model_ood_heatmap(
        fold_results,
        output_dir / "heatmap_site_model_ood",
        save_pdf=config.save_pdf_plots,
    )
    plot_pairwise_heatmaps(pairwise_results, output_dir, save_pdf=config.save_pdf_plots)
    plot_training_domain_by_site_heatmap(
        combo_summary_by_site,
        output_dir / "transportability_training_domains_by_site_heatmap",
        save_pdf=config.save_pdf_plots,
    )
    plot_training_domain_summary_heatmap(
        combo_summary_by_model,
        output_dir / "transportability_training_domains_heatmap",
        save_pdf=config.save_pdf_plots,
    )
    plot_training_domain_all_combinations(
        combo_results,
        output_dir / "transportability_training_domains_all_combinations_heatmap",
        save_pdf=config.save_pdf_plots,
    )
    plot_domain_difficulty(
        domain_difficulty,
        output_dir / "site_accuracy_ranking",
        save_pdf=config.save_pdf_plots,
    )

    calibration_df = plot_calibration_curves(
        prediction_records, output_dir, save_pdf=config.save_pdf_plots
    )

    save_csvs(
        output_dir=output_dir,
        config=config,
        dataset_summary=dataset_summary,
        fold_results=fold_results,
        prediction_records=prediction_records,
        model_summary=model_summary,
        domain_summary=domain_summary,
        domain_difficulty=domain_difficulty,
        pairwise_results=pairwise_results,
        pairwise_predictions=pairwise_predictions,
        pairwise_model_summary=pairwise_model_summary,
        pairwise_test_summary=pairwise_test_summary,
        combo_results=combo_results,
        combo_summary_by_model=combo_summary_by_model,
        combo_summary_by_site=combo_summary_by_site,
        calibration_df=calibration_df,
        dataset_df=dataset_df,
    )
    write_report(
        output_dir=output_dir,
        config=config,
        dataset_summary=dataset_summary,
        fold_results=fold_results,
        model_summary=model_summary,
        domain_difficulty=domain_difficulty,
        pairwise_model_summary=pairwise_model_summary,
        pairwise_test_summary=pairwise_test_summary,
        combo_summary_by_model=combo_summary_by_model,
    )
    write_html_report(
        output_dir / "baseline_report.md", output_dir / "baseline_report.html"
    )
    write_readme(output_dir, config, model_summary, dataset_summary)

    shutil.copy2(Path(__file__), output_dir / Path(__file__).name)
    logger.info("Fresh TCGA baseline bundle written to %s", output_dir)


if __name__ == "__main__":
    main()
