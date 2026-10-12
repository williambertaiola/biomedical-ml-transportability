"""Transportability experiment: analysis."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .adaptation import PreparedFoldData
from .config import METRIC_COLUMNS
from .data_utils import InferredColumns


def summarize_site_distances(
    prepared: PreparedFoldData,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    inferred: InferredColumns,
) -> tuple[dict[str, object], pd.DataFrame]:
    pre_metrics, _ = _distribution_distance(
        prepared.X_source_pre, prepared.X_target_pre
    )
    post_metrics, _ = _distribution_distance(
        prepared.X_source_aligned, prepared.X_target_eval
    )
    feature_shift_df = compute_feature_shift_table(source_df, target_df, inferred)
    feature_shift_df["target_site"] = prepared.target_site

    summary_row: dict[str, object] = {
        "target_site": prepared.target_site,
        "source_n": int(len(source_df)),
        "target_n": int(len(target_df)),
        "pre_mean_shift_l2": pre_metrics["mean_shift_l2"],
        "pre_mean_abs_mean_diff": pre_metrics["mean_abs_mean_diff"],
        "pre_mean_abs_smd": pre_metrics["mean_abs_smd"],
        "pre_max_abs_smd": pre_metrics["max_abs_smd"],
        "pre_covariance_fro": pre_metrics["covariance_fro"],
        "post_mean_shift_l2": post_metrics["mean_shift_l2"],
        "post_mean_abs_mean_diff": post_metrics["mean_abs_mean_diff"],
        "post_mean_abs_smd": post_metrics["mean_abs_smd"],
        "post_max_abs_smd": post_metrics["max_abs_smd"],
        "post_covariance_fro": post_metrics["covariance_fro"],
        "delta_mean_abs_smd": post_metrics["mean_abs_smd"]
        - pre_metrics["mean_abs_smd"],
        "delta_covariance_fro": post_metrics["covariance_fro"]
        - pre_metrics["covariance_fro"],
    }
    return summary_row, feature_shift_df


def compute_feature_shift_table(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    inferred: InferredColumns,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for column in inferred.feature_columns:
        source_values, target_values = _prepare_feature_for_shift(
            source_df[column],
            target_df[column],
            is_categorical=column in inferred.categorical_features,
        )
        pooled_sd = np.sqrt(
            (np.var(source_values) + np.var(target_values)) / 2.0 + 1e-8
        )
        mean_diff = float(np.mean(source_values) - np.mean(target_values))
        abs_smd = float(abs(mean_diff) / pooled_sd) if pooled_sd > 0 else 0.0
        rows.append(
            {
                "feature": column,
                "source_mean": float(np.mean(source_values)),
                "target_mean": float(np.mean(target_values)),
                "mean_diff": mean_diff,
                "abs_smd": abs_smd,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("abs_smd", ascending=False)
        .reset_index(drop=True)
    )


def extract_feature_importance(
    model,
    feature_names: list[str],
    model_name: str,
    variant: str,
    target_site: str,
) -> pd.DataFrame:
    values: np.ndarray | None = None
    importance_type = ""

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.abs(coef.ravel())
        importance_type = "abs_coefficient"
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        importance_type = "feature_importance"

    if values is None or values.size == 0:
        return pd.DataFrame()

    if values.shape[0] != len(feature_names):
        feature_names = feature_names[: values.shape[0]]

    total = float(np.sum(values))
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values[: len(feature_names)],
            "importance_share": values[: len(feature_names)] / total
            if total > 0
            else 0.0,
            "importance_type": importance_type,
            "model_name": model_name,
            "variant": variant,
            "target_site": target_site,
        }
    )
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def build_variant_comparison_table(
    fold_metrics_df: pd.DataFrame,
    variant_a: str = "coral_only",
    variant_b: str = "coral_plus_pseudolabel",
    metrics: Iterable[str] = METRIC_COLUMNS,
) -> pd.DataFrame:
    if fold_metrics_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_cols = ["target_site", "model_name"]

    for keys, group in fold_metrics_df.groupby(group_cols, dropna=False):
        site, model_name = keys
        left = group.loc[group["variant"] == variant_a]
        right = group.loc[group["variant"] == variant_b]
        if left.empty or right.empty:
            continue

        row: dict[str, object] = {
            "target_site": site,
            "model_name": model_name,
            "variant_reference": variant_a,
            "variant_compare": variant_b,
        }
        for metric in metrics:
            left_value = (
                float(left.iloc[0][metric])
                if pd.notna(left.iloc[0][metric])
                else np.nan
            )
            right_value = (
                float(right.iloc[0][metric])
                if pd.notna(right.iloc[0][metric])
                else np.nan
            )
            row[f"{metric}_{variant_a}"] = left_value
            row[f"{metric}_{variant_b}"] = right_value
            row[f"delta_{metric}"] = (
                right_value - left_value
                if pd.notna(left_value) and pd.notna(right_value)
                else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


def build_site_difficulty_ranking(
    fold_metrics_df: pd.DataFrame,
    primary_metric: str,
) -> pd.DataFrame:
    if fold_metrics_df.empty:
        return pd.DataFrame()

    grouped = (
        fold_metrics_df.groupby("target_site", dropna=False)[primary_metric]
        .agg(
            mean_score="mean",
            std_score="std",
            min_score="min",
            max_score="max",
            n_runs="size",
        )
        .reset_index()
        .sort_values("mean_score", ascending=False)
    )
    grouped["difficulty_rank"] = np.arange(1, len(grouped) + 1)
    grouped["difficulty_label"] = np.where(
        grouped["difficulty_rank"] == 1,
        "easiest",
        np.where(grouped["difficulty_rank"] == len(grouped), "hardest", "intermediate"),
    )
    return grouped


def build_error_analysis_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_cols = ["target_site", "model_name", "variant"]

    for (site, model_name, variant), group in predictions_df.groupby(
        group_cols, dropna=False
    ):
        y_true = group["y_true"].to_numpy(dtype=int)
        y_pred = group["y_pred"].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        errors = group.loc[group["is_correct"] == 0]
        correct = group.loc[group["is_correct"] == 1]
        rows.append(
            {
                "target_site": site,
                "model_name": model_name,
                "variant": variant,
                "error_rate": float(np.mean(y_true != y_pred)),
                "false_positive_rate": float(fp / max(fp + tn, 1)),
                "false_negative_rate": float(fn / max(fn + tp, 1)),
                "n_errors": int(len(errors)),
                "mean_confidence_correct": float(correct["confidence"].mean())
                if not correct.empty
                else np.nan,
                "mean_confidence_error": float(errors["confidence"].mean())
                if not errors.empty
                else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["error_rate", "false_negative_rate"], ascending=[False, False]
    )


def _distribution_distance(
    X_source: np.ndarray, X_target: np.ndarray
) -> tuple[dict[str, float], np.ndarray]:
    X_source = np.asarray(X_source, dtype=float)
    X_target = np.asarray(X_target, dtype=float)
    source_mean = X_source.mean(axis=0)
    target_mean = X_target.mean(axis=0)
    mean_diff = source_mean - target_mean
    pooled_sd = np.sqrt((X_source.var(axis=0) + X_target.var(axis=0)) / 2.0 + 1e-8)
    smd = mean_diff / pooled_sd

    source_cov = np.cov(X_source, rowvar=False)
    target_cov = np.cov(X_target, rowvar=False)
    if np.ndim(source_cov) == 0:
        source_cov = np.array([[float(source_cov)]], dtype=float)
        target_cov = np.array([[float(target_cov)]], dtype=float)

    metrics = {
        "mean_shift_l2": float(np.linalg.norm(mean_diff)),
        "mean_abs_mean_diff": float(np.mean(np.abs(mean_diff))),
        "mean_abs_smd": float(np.mean(np.abs(smd))),
        "max_abs_smd": float(np.max(np.abs(smd))),
        "covariance_fro": float(np.linalg.norm(source_cov - target_cov, ord="fro")),
    }
    return metrics, smd


def _prepare_feature_for_shift(
    source_series: pd.Series,
    target_series: pd.Series,
    is_categorical: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if is_categorical:
        fill_value = source_series.mode(dropna=True)
        fill_value = fill_value.iloc[0] if not fill_value.empty else "__missing__"
        source_filled = source_series.fillna(fill_value)
        target_filled = target_series.fillna(fill_value)
        categories = pd.Index(
            pd.concat([source_filled, target_filled], axis=0).astype(str).unique()
        )
        source_codes = pd.Categorical(
            source_filled.astype(str), categories=categories
        ).codes.astype(float)
        target_codes = pd.Categorical(
            target_filled.astype(str), categories=categories
        ).codes.astype(float)
        return source_codes, target_codes

    fill_value = pd.to_numeric(source_series, errors="coerce").median()
    if pd.isna(fill_value):
        fill_value = 0.0
    source_numeric = (
        pd.to_numeric(source_series, errors="coerce")
        .fillna(fill_value)
        .to_numpy(dtype=float)
    )
    target_numeric = (
        pd.to_numeric(target_series, errors="coerce")
        .fillna(fill_value)
        .to_numpy(dtype=float)
    )
    return source_numeric, target_numeric
