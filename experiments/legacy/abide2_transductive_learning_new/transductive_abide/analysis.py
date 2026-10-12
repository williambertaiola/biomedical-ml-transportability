"""Transportability experiment: analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import METRIC_COLUMNS
from .preprocessing import PreparedFoldData


@dataclass
class DistanceArtifacts:
    summary_row: dict[str, object]
    feature_shift_df: pd.DataFrame
    alignment_plot_df: pd.DataFrame


def summarize_site_distances(
    prepared: PreparedFoldData,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    max_top_features: int,
    max_points_plot: int,
    random_state: int,
) -> DistanceArtifacts:
    pre_metrics = distribution_distance(prepared.X_source_pre, prepared.X_target_pre)
    post_metrics = distribution_distance(
        prepared.X_source_aligned, prepared.X_target_eval
    )
    feature_shift_df = compute_feature_shift_table(
        source_df=source_df,
        target_df=target_df,
        selected_features=prepared.selected_feature_names,
        target_domain=prepared.target_domain,
        top_n=max_top_features,
    )
    alignment_plot_df = build_alignment_plot_records(
        X_source_pre=prepared.X_source_pre,
        X_target_pre=prepared.X_target_pre,
        X_source_post=prepared.X_source_aligned,
        X_target_post=prepared.X_target_eval,
        target_domain=prepared.target_domain,
        max_points=max_points_plot,
        random_state=random_state,
    )
    summary_row = {
        "target_domain": prepared.target_domain,
        "source_n": int(prepared.X_source_pre.shape[0]),
        "target_n": int(prepared.X_target_pre.shape[0]),
        "pre_mean_shift_l2": pre_metrics["mean_shift_l2"],
        "pre_mean_abs_smd": pre_metrics["mean_abs_smd"],
        "pre_covariance_fro": pre_metrics["covariance_fro"],
        "post_mean_shift_l2": post_metrics["mean_shift_l2"],
        "post_mean_abs_smd": post_metrics["mean_abs_smd"],
        "post_covariance_fro": post_metrics["covariance_fro"],
        "delta_mean_abs_smd": post_metrics["mean_abs_smd"]
        - pre_metrics["mean_abs_smd"],
        "delta_covariance_fro": post_metrics["covariance_fro"]
        - pre_metrics["covariance_fro"],
    }
    return DistanceArtifacts(
        summary_row=summary_row,
        feature_shift_df=feature_shift_df,
        alignment_plot_df=alignment_plot_df,
    )


def distribution_distance(
    X_source: np.ndarray, X_target: np.ndarray
) -> dict[str, float]:
    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)
    source_mean = X_source.mean(axis=0)
    target_mean = X_target.mean(axis=0)
    mean_diff = source_mean - target_mean
    pooled_sd = np.sqrt((X_source.var(axis=0) + X_target.var(axis=0)) / 2.0 + 1e-8)
    smd = mean_diff / pooled_sd

    source_cov = np.cov(X_source, rowvar=False)
    target_cov = np.cov(X_target, rowvar=False)
    if np.ndim(source_cov) == 0:
        source_cov = np.array([[float(source_cov)]], dtype=np.float64)
        target_cov = np.array([[float(target_cov)]], dtype=np.float64)

    return {
        "mean_shift_l2": float(np.linalg.norm(mean_diff)),
        "mean_abs_smd": float(np.mean(np.abs(smd))),
        "covariance_fro": float(np.linalg.norm(source_cov - target_cov, ord="fro")),
    }


def compute_feature_shift_table(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    selected_features: list[str],
    target_domain: str,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in selected_features:
        source_values = pd.to_numeric(source_df[feature], errors="coerce").to_numpy(
            dtype=np.float64
        )
        target_values = pd.to_numeric(target_df[feature], errors="coerce").to_numpy(
            dtype=np.float64
        )
        source_fill = safe_nanmedian(source_values)
        target_fill = safe_nanmedian(target_values)
        if source_fill is not None:
            source_values = np.nan_to_num(source_values, nan=source_fill)
        if target_fill is not None:
            target_values = np.nan_to_num(target_values, nan=target_fill)
        mean_diff = float(source_values.mean() - target_values.mean())
        pooled_sd = float(
            np.sqrt((source_values.var() + target_values.var()) / 2.0 + 1e-8)
        )
        abs_smd = abs(mean_diff) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append(
            {
                "target_domain": target_domain,
                "feature": feature,
                "source_mean": float(source_values.mean()),
                "target_mean": float(target_values.mean()),
                "mean_diff": mean_diff,
                "abs_smd": float(abs_smd),
            }
        )
    shift_df = (
        pd.DataFrame(rows)
        .sort_values("abs_smd", ascending=False)
        .reset_index(drop=True)
    )
    return shift_df.head(top_n).copy()


def safe_nanmedian(values: np.ndarray) -> float | None:
    if not np.isnan(values).any():
        return None
    finite_values = values[~np.isnan(values)]
    if finite_values.size == 0:
        return 0.0
    return float(np.median(finite_values))


def extract_feature_importance(
    model,
    model_name: str,
    variant: str,
    target_domain: str,
    selected_feature_names: list[str],
    pca_feature_names: list[str],
    pca,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=np.float64)
        if coef.ndim == 1:
            coef = coef[None, :]
        if pca is not None and hasattr(pca, "components_"):
            gene_scores = np.abs(pca.components_.T @ coef.T).mean(axis=1)
            for feature, value in zip(selected_feature_names, gene_scores):
                rows.append(
                    {
                        "model_name": model_name,
                        "variant": variant,
                        "target_domain": target_domain,
                        "feature": feature,
                        "importance": float(value),
                        "importance_type": "backprojected_abs_coefficient",
                    }
                )
        else:
            coef_scores = np.abs(coef).mean(axis=0)
            for feature, value in zip(pca_feature_names, coef_scores):
                rows.append(
                    {
                        "model_name": model_name,
                        "variant": variant,
                        "target_domain": target_domain,
                        "feature": feature,
                        "importance": float(value),
                        "importance_type": "abs_coefficient",
                    }
                )

    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=np.float64)
        for feature, value in zip(pca_feature_names, values):
            rows.append(
                {
                    "model_name": model_name,
                    "variant": variant,
                    "target_domain": target_domain,
                    "feature": feature,
                    "importance": float(value),
                    "importance_type": "feature_importance",
                }
            )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
        .head(100)
    )


def build_variant_comparison_table(
    fold_metrics_df: pd.DataFrame,
    variant_a: str = "coral_only",
    variant_b: str = "coral_plus_pseudolabel",
) -> pd.DataFrame:
    if fold_metrics_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    metrics = METRIC_COLUMNS
    for (target_domain, model_name), group in fold_metrics_df.groupby(
        ["target_domain", "model_name"], dropna=False
    ):
        left = group.loc[group["variant"] == variant_a]
        right = group.loc[group["variant"] == variant_b]
        if left.empty or right.empty:
            continue
        left_row = left.iloc[0]
        right_row = right.iloc[0]
        row: dict[str, object] = {
            "target_domain": target_domain,
            "model_name": model_name,
            "variant_reference": variant_a,
            "variant_compare": variant_b,
        }
        for metric in metrics:
            left_value = (
                float(left_row[metric]) if pd.notna(left_row[metric]) else np.nan
            )
            right_value = (
                float(right_row[metric]) if pd.notna(right_row[metric]) else np.nan
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
    fold_metrics_df: pd.DataFrame, primary_metric: str
) -> pd.DataFrame:
    if fold_metrics_df.empty:
        return pd.DataFrame()

    grouped = (
        fold_metrics_df.groupby("target_domain", dropna=False)[primary_metric]
        .agg(
            mean_score="mean",
            std_score="std",
            min_score="min",
            max_score="max",
            n_runs="size",
            n_valid="count",
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
    for (domain, model_name, variant), group in predictions_df.groupby(
        ["target_domain", "model_name", "variant"], dropna=False
    ):
        errors = group.loc[group["is_correct"] == 0]
        correct = group.loc[group["is_correct"] == 1]
        rows.append(
            {
                "target_domain": domain,
                "model_name": model_name,
                "variant": variant,
                "error_rate": float((group["is_correct"] == 0).mean()),
                "mean_confidence_error": float(errors["confidence"].mean())
                if not errors.empty
                else np.nan,
                "mean_confidence_correct": float(correct["confidence"].mean())
                if not correct.empty
                else np.nan,
                "n_errors": int(len(errors)),
            }
        )
    return pd.DataFrame(rows).sort_values("error_rate", ascending=False)


def build_alignment_plot_records(
    X_source_pre: np.ndarray,
    X_target_pre: np.ndarray,
    X_source_post: np.ndarray,
    X_target_post: np.ndarray,
    target_domain: str,
    max_points: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    def sample_rows(X: np.ndarray, max_n: int) -> tuple[np.ndarray, np.ndarray]:
        if X.shape[0] <= max_n:
            indices = np.arange(X.shape[0], dtype=int)
            return X, indices
        indices = np.sort(rng.choice(X.shape[0], size=max_n, replace=False))
        return X[indices], indices

    X_source_pre_s, source_idx = sample_rows(X_source_pre, max_points)
    X_target_pre_s, target_idx = sample_rows(X_target_pre, max_points)
    X_source_post_s = X_source_post[source_idx]
    X_target_post_s = X_target_post[target_idx]

    pca = PCA(n_components=2, random_state=random_state)
    combined_pre = np.vstack([X_source_pre_s, X_target_pre_s])
    pca.fit(combined_pre)
    combined_post = np.vstack([X_source_post_s, X_target_post_s])

    pre_coords = pca.transform(combined_pre)
    post_coords = pca.transform(combined_post)

    rows: list[dict[str, object]] = []
    for phase, coords in [
        ("pre_alignment", pre_coords),
        ("post_alignment", post_coords),
    ]:
        split_index = X_source_pre_s.shape[0]
        source_coords = coords[:split_index]
        target_coords = coords[split_index:]
        for subset_name, subset_coords in [
            ("source", source_coords),
            ("target", target_coords),
        ]:
            for pc1, pc2 in subset_coords:
                rows.append(
                    {
                        "target_domain": target_domain,
                        "phase": phase,
                        "subset": subset_name,
                        "pc1": float(pc1),
                        "pc2": float(pc2),
                    }
                )
    return pd.DataFrame(rows)
