"""Transportability experiment: plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import colors
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PRIMARY_PLOT_METRICS
from .data_utils import InferredColumns
from .metrics import calibration_curve_data

PLOT_DPI = 360
SEQUENTIAL_RED_GREEN = colors.LinearSegmentedColormap.from_list(
    "sequential_red_green",
    ["#b2182b", "#ef8a62", "#fddbc7", "#d9f0d3", "#66bd63", "#1a9850"],
)
DIVERGING_RED_GREEN = colors.LinearSegmentedColormap.from_list(
    "diverging_red_green",
    ["#b2182b", "#f7f7f7", "#1a9850"],
)
MODEL_COLORS = {
    "logistic_regression": "#b2182b",
    "random_forest": "#ef8a62",
    "xgboost": "#4daf4a",
    "lightgbm": "#4daf4a",
    "hist_gradient_boosting": "#1a9850",
}
VARIANT_LINESTYLES = {
    "coral_only": "-",
    "coral_plus_pseudolabel": "--",
}
VARIANT_LABELS = {
    "coral_only": "CORAL alignment only",
    "coral_plus_pseudolabel": "CORAL + pseudo-labeling",
}

plt.rcParams.update(
    {
        "figure.dpi": PLOT_DPI,
        "savefig.dpi": PLOT_DPI,
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
    }
)


def plot_confusion_matrix_png(
    matrix: np.ndarray, title: str, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(matrix, cmap=SEQUENTIAL_RED_GREEN)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{int(matrix[i, j])}",
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_heatmaps(
    fold_metrics_df: pd.DataFrame,
    plots_dir: Path,
    metrics: list[str] | None = None,
) -> None:
    if fold_metrics_df.empty:
        return

    metrics = metrics or list(PRIMARY_PLOT_METRICS)
    for variant in sorted(fold_metrics_df["variant"].unique().tolist()):
        variant_df = fold_metrics_df.loc[fold_metrics_df["variant"] == variant].copy()
        if variant_df.empty:
            continue

        fig, axes = plt.subplots(
            1, len(metrics), figsize=(6 * len(metrics), 6.5), squeeze=False
        )
        for ax, metric in zip(axes.flatten(), metrics):
            pivot = (
                variant_df.pivot(
                    index="target_site", columns="model_name", values=metric
                )
                .sort_index()
                .sort_index(axis=1)
            )
            _draw_heatmap(
                ax=ax,
                data=pivot.to_numpy(dtype=float),
                row_labels=pivot.index.tolist(),
                col_labels=[_display_name(name) for name in pivot.columns.tolist()],
                title=f"{VARIANT_LABELS.get(variant, variant)}\n{metric.replace('_', ' ').title()}",
                cmap=SEQUENTIAL_RED_GREEN,
                vmin=0.0,
                vmax=1.0,
            )
        fig.tight_layout()
        fig.savefig(
            plots_dir / f"heatmap_main_metrics_{variant}.png", bbox_inches="tight"
        )
        plt.close(fig)


def create_variant_heatmap(
    variant_comparison_df: pd.DataFrame,
    plots_dir: Path,
    metrics: list[str] | None = None,
) -> None:
    if variant_comparison_df.empty:
        return

    metrics = metrics or list(PRIMARY_PLOT_METRICS)
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(6 * len(metrics), 6.5), squeeze=False
    )
    for ax, metric in zip(axes.flatten(), metrics):
        pivot = (
            variant_comparison_df.pivot(
                index="target_site", columns="model_name", values=f"delta_{metric}"
            )
            .sort_index()
            .sort_index(axis=1)
        )
        values = pivot.to_numpy(dtype=float)
        bound = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
        bound = max(bound, 1e-6)
        _draw_heatmap(
            ax=ax,
            data=values,
            row_labels=pivot.index.tolist(),
            col_labels=[_display_name(name) for name in pivot.columns.tolist()],
            title=f"Delta pseudo-labeling - alignment only\n{metric.replace('_', ' ').title()}",
            cmap=DIVERGING_RED_GREEN,
            vmin=-bound,
            vmax=bound,
            center=0.0,
        )
    fig.tight_layout()
    fig.savefig(plots_dir / "heatmap_variant_delta.png", bbox_inches="tight")
    plt.close(fig)


def create_site_barplots(
    fold_metrics_df: pd.DataFrame,
    plots_dir: Path,
    metrics: list[str] | None = None,
) -> None:
    if fold_metrics_df.empty:
        return

    metrics = metrics or list(PRIMARY_PLOT_METRICS)
    variants = sorted(fold_metrics_df["variant"].unique().tolist())
    fig, axes = plt.subplots(
        len(variants),
        len(metrics),
        figsize=(7 * len(metrics), 5.6 * len(variants)),
        squeeze=False,
    )

    for row_idx, variant in enumerate(variants):
        variant_df = fold_metrics_df.loc[fold_metrics_df["variant"] == variant]
        sites = sorted(variant_df["target_site"].unique().tolist())
        models = sorted(variant_df["model_name"].unique().tolist())
        x = np.arange(len(sites), dtype=float)
        width = min(0.8 / max(len(models), 1), 0.22)

        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for model_idx, model_name in enumerate(models):
                sub = variant_df.loc[variant_df["model_name"] == model_name]
                values = []
                for site in sites:
                    cell = sub.loc[sub["target_site"] == site, metric]
                    values.append(float(cell.iloc[0]) if not cell.empty else np.nan)
                offset = (model_idx - (len(models) - 1) / 2.0) * width
                ax.bar(
                    x + offset,
                    values,
                    width=width,
                    label=_display_name(model_name),
                    color=MODEL_COLORS.get(model_name, "#666666"),
                )
            ax.set_xticks(x, labels=sites, rotation=25, ha="right")
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(VARIANT_LABELS.get(variant, variant))
            ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(handles), 4),
            frameon=False,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(plots_dir / "barplot_per_site_primary_metrics.png", bbox_inches="tight")
    plt.close(fig)


def create_boxplots(
    fold_metrics_df: pd.DataFrame,
    plots_dir: Path,
    metrics: list[str] | None = None,
) -> None:
    if fold_metrics_df.empty:
        return

    metrics = metrics or list(PRIMARY_PLOT_METRICS)
    fold_metrics_df = fold_metrics_df.copy()
    fold_metrics_df["model_variant"] = [
        f"{_display_name(model)}\n{VARIANT_LABELS.get(variant, variant)}"
        for model, variant in zip(
            fold_metrics_df["model_name"], fold_metrics_df["variant"]
        )
    ]
    labels = fold_metrics_df["model_variant"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(6.2 * len(metrics), 7), squeeze=False
    )

    for ax, metric in zip(axes.flatten(), metrics):
        data = [
            fold_metrics_df.loc[
                fold_metrics_df["model_variant"] == label, metric
            ].to_numpy(dtype=float)
            for label in labels
        ]
        box = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.65)
        for patch, label in zip(box["boxes"], labels):
            patch.set_facecolor(
                "#f4c7cf" if "pseudo-labeling" not in label else "#ccebc5"
            )
            patch.set_edgecolor("#444444")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        plots_dir / "boxplot_primary_metrics_across_sites.png", bbox_inches="tight"
    )
    plt.close(fig)


def create_reliability_plot(
    predictions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
    ece_bins: int,
) -> None:
    if predictions_df.empty or summary_df.empty:
        return

    best_model_name, best_variant = _select_best_combo(summary_df)
    best_df = predictions_df.loc[
        (predictions_df["model_name"] == best_model_name)
        & (predictions_df["variant"] == best_variant)
    ].copy()
    if best_df.empty:
        return

    curve_df = calibration_curve_data(
        best_df["y_true"].to_numpy(dtype=int),
        best_df["y_prob"].to_numpy(dtype=float),
        n_bins=ece_bins,
        model_name=best_model_name,
        variant=best_variant,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#444444", linewidth=1.1)
    axes[0].plot(
        curve_df["mean_predicted_probability"],
        curve_df["empirical_positive_rate"],
        marker="o",
        linewidth=2.0,
        color=MODEL_COLORS.get(best_model_name, "#1a9850"),
    )
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed positive rate")
    axes[0].set_title(
        f"Reliability plot\n{_display_name(best_model_name)} | {VARIANT_LABELS.get(best_variant, best_variant)}"
    )
    axes[0].grid(alpha=0.25)

    axes[1].hist(
        best_df["y_prob"],
        bins=20,
        color=MODEL_COLORS.get(best_model_name, "#1a9850"),
        alpha=0.8,
    )
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Prediction probability distribution")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_aggregate_curve_plots(predictions_df: pd.DataFrame, plots_dir: Path) -> None:
    if predictions_df.empty:
        return

    combos = (
        predictions_df[["model_name", "variant"]]
        .drop_duplicates()
        .sort_values(["model_name", "variant"])
        .itertuples(index=False, name=None)
    )

    roc_fig, roc_ax = plt.subplots(figsize=(8.5, 7.2))
    pr_fig, pr_ax = plt.subplots(figsize=(8.5, 7.2))
    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="#444444", linewidth=1.0)

    for model_name, variant in combos:
        subset = predictions_df.loc[
            (predictions_df["model_name"] == model_name)
            & (predictions_df["variant"] == variant)
        ].copy()
        if subset["y_true"].nunique() < 2:
            continue
        y_true = subset["y_true"].to_numpy(dtype=int)
        y_prob = subset["y_prob"].to_numpy(dtype=float)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        roc_value = auc(fpr, tpr)
        pr_value = auc(recall[::-1], precision[::-1])

        label = f"{_display_name(model_name)} | {VARIANT_LABELS.get(variant, variant)}"
        color = MODEL_COLORS.get(model_name, "#666666")
        linestyle = VARIANT_LINESTYLES.get(variant, "-")

        roc_ax.plot(
            fpr,
            tpr,
            label=f"{label} (AUC={roc_value:.3f})",
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
        )
        pr_ax.plot(
            recall,
            precision,
            label=f"{label} (AUC={pr_value:.3f})",
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
        )

    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate")
    roc_ax.set_title("Aggregate ROC curves across hold-out sites")
    roc_ax.grid(alpha=0.25)
    roc_ax.legend(frameon=False, loc="lower right")

    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title("Aggregate precision-recall curves across hold-out sites")
    pr_ax.grid(alpha=0.25)
    pr_ax.legend(frameon=False, loc="lower left")

    roc_fig.tight_layout()
    roc_fig.savefig(plots_dir / "aggregate_roc_curves.png", bbox_inches="tight")
    plt.close(roc_fig)

    pr_fig.tight_layout()
    pr_fig.savefig(plots_dir / "aggregate_pr_curves.png", bbox_inches="tight")
    plt.close(pr_fig)


def create_confidence_distribution_plot(
    confidence_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
    threshold: float,
) -> None:
    if confidence_df.empty or summary_df.empty:
        return

    best_model_name, best_variant = _select_best_combo(summary_df)
    del best_variant
    best_df = confidence_df.loc[confidence_df["model_name"] == best_model_name].copy()
    if best_df.empty:
        return

    sites = sorted(best_df["target_site"].unique().tolist())
    n_cols = 2
    n_rows = int(np.ceil(len(sites) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 4.5 * n_rows), squeeze=False)

    for ax, site in zip(axes.flatten(), sites):
        subset = best_df.loc[best_df["target_site"] == site]
        not_selected = subset.loc[
            subset["selected_for_pseudolabeling"] == 0, "confidence"
        ]
        selected = subset.loc[subset["selected_for_pseudolabeling"] == 1, "confidence"]
        ax.hist(
            not_selected,
            bins=20,
            alpha=0.65,
            density=True,
            label="not selected",
            color="#d6604d",
        )
        if not selected.empty:
            ax.hist(
                selected,
                bins=20,
                alpha=0.65,
                density=True,
                label="selected",
                color="#1a9850",
            )
        ax.axvline(threshold, linestyle="--", color="#333333", linewidth=1.1)
        ax.set_title(site)
        ax.set_xlabel("Max predicted probability")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)

    for ax in axes.flatten()[len(sites) :]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        f"Confidence distribution on hold-out target\nBest model: {_display_name(best_model_name)}"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_pseudolabel_count_plot(threshold_df: pd.DataFrame, plots_dir: Path) -> None:
    if threshold_df.empty:
        return

    model_names = sorted(threshold_df["model_name"].unique().tolist())
    fig, axes = plt.subplots(
        1, len(model_names), figsize=(6 * len(model_names), 6), squeeze=False
    )

    for ax, model_name in zip(axes.flatten(), model_names):
        subset = threshold_df.loc[threshold_df["model_name"] == model_name].copy()
        pivot = (
            subset.pivot(
                index="target_site", columns="threshold", values="n_selected_after_caps"
            )
            .sort_index()
            .sort_index(axis=1)
        )
        _draw_heatmap(
            ax=ax,
            data=pivot.to_numpy(dtype=float),
            row_labels=pivot.index.tolist(),
            col_labels=[f"{float(col):.2f}" for col in pivot.columns.tolist()],
            title=f"Accepted pseudo-labels\n{_display_name(model_name)}",
            cmap=SEQUENTIAL_RED_GREEN,
            vmin=0.0,
            vmax=float(np.nanmax(pivot.to_numpy(dtype=float)))
            if np.isfinite(pivot.to_numpy(dtype=float)).any()
            else 1.0,
            integer_annotations=True,
        )
    fig.tight_layout()
    fig.savefig(
        plots_dir / "pseudolabel_counts_by_site_threshold.png", bbox_inches="tight"
    )
    plt.close(fig)


def create_summary_plot(
    summary_df: pd.DataFrame,
    output_path: Path,
    metrics: list[str] | None = None,
) -> None:
    if summary_df.empty:
        return

    metrics = metrics or list(PRIMARY_PLOT_METRICS)
    labels = [
        f"{_display_name(model_name)}\n{VARIANT_LABELS.get(variant, variant)}"
        for model_name, variant in zip(summary_df["model_name"], summary_df["variant"])
    ]
    x = np.arange(len(labels), dtype=float)

    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(12, 4.2 * len(metrics)), squeeze=False
    )
    for ax, metric in zip(axes.flatten(), metrics):
        means = summary_df[f"{metric}_mean"].to_numpy(dtype=float)
        stds = summary_df[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
        point_colors = [
            SEQUENTIAL_RED_GREEN(value) for value in np.clip(means, 0.0, 1.0)
        ]
        ax.errorbar(
            x, means, yerr=stds, fmt="none", capsize=5, linewidth=1.5, color="#555555"
        )
        ax.scatter(
            x, means, c=point_colors, s=75, edgecolor="#333333", linewidth=0.8, zorder=3
        )
        ax.set_xticks(x, labels=labels, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_site_distance_plots(
    distance_df: pd.DataFrame,
    feature_shift_df: pd.DataFrame,
    plots_dir: Path,
    top_n: int,
) -> None:
    if not distance_df.empty:
        sites = distance_df["target_site"].astype(str).tolist()
        x = np.arange(len(sites), dtype=float)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
        metrics = [
            ("mean_abs_smd", "Mean absolute SMD"),
            ("covariance_fro", "Covariance Frobenius distance"),
            ("mean_shift_l2", "Mean shift L2 distance"),
        ]
        for ax, (metric_key, title) in zip(axes, metrics):
            ax.bar(
                x - 0.18,
                distance_df[f"pre_{metric_key}"],
                width=0.36,
                label="pre-alignment",
                color="#d6604d",
            )
            ax.bar(
                x + 0.18,
                distance_df[f"post_{metric_key}"],
                width=0.36,
                label="post-alignment",
                color="#1a9850",
            )
            ax.set_xticks(x, labels=sites, rotation=25, ha="right")
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        fig.savefig(
            plots_dir / "site_distance_pre_post_alignment.png", bbox_inches="tight"
        )
        plt.close(fig)

    if feature_shift_df.empty:
        return

    top_features = (
        feature_shift_df.groupby("feature", dropna=False)["abs_smd"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    plot_df = feature_shift_df.loc[
        feature_shift_df["feature"].isin(top_features)
    ].copy()
    pivot = (
        plot_df.pivot(index="target_site", columns="feature", values="abs_smd")
        .sort_index()
        .reindex(columns=top_features)
    )
    fig, ax = plt.subplots(figsize=(2.2 * max(len(top_features), 4), 6.2))
    _draw_heatmap(
        ax=ax,
        data=pivot.to_numpy(dtype=float),
        row_labels=pivot.index.tolist(),
        col_labels=pivot.columns.tolist(),
        title="Most shifted original features by hold-out site",
        cmap=SEQUENTIAL_RED_GREEN,
        vmin=0.0,
        vmax=float(np.nanmax(pivot.to_numpy(dtype=float)))
        if np.isfinite(pivot.to_numpy(dtype=float)).any()
        else 1.0,
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "feature_shift_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def create_site_pca_plot(
    df: pd.DataFrame,
    inferred: InferredColumns,
    output_path: Path,
    max_points_per_site: int,
) -> None:
    if df.empty:
        return

    sampled_frames = []
    for site, sub in df.groupby(inferred.domain_column, dropna=False):
        sub = sub.copy()
        if len(sub) > max_points_per_site:
            sub = sub.sample(max_points_per_site, random_state=42)
        sampled_frames.append(sub)
    plot_df = pd.concat(sampled_frames, axis=0, ignore_index=True)

    matrix = _prepare_pca_matrix(plot_df, inferred)
    matrix = StandardScaler().fit_transform(matrix)
    embedding = PCA(n_components=2, random_state=42).fit_transform(matrix)

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for site, sub_idx in plot_df.groupby(
        inferred.domain_column, dropna=False
    ).groups.items():
        indices = np.asarray(list(sub_idx), dtype=int)
        ax.scatter(
            embedding[indices, 0],
            embedding[indices, 1],
            s=26,
            alpha=0.75,
            label=str(site),
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Global PCA view of sites (unlabeled diagnostic)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_summary_plots(
    fold_metrics_df: pd.DataFrame,
    variant_comparison_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    confidence_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    feature_shift_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    inferred: InferredColumns,
    plots_dir: Path,
    ece_bins: int,
    top_shift_features: int,
    pseudo_threshold: float,
) -> None:
    create_heatmaps(fold_metrics_df, plots_dir)
    create_variant_heatmap(variant_comparison_df, plots_dir)
    create_site_barplots(fold_metrics_df, plots_dir)
    create_boxplots(fold_metrics_df, plots_dir)
    create_reliability_plot(
        predictions_df,
        summary_df,
        plots_dir / "calibration_reliability_best_model.png",
        ece_bins,
    )
    create_aggregate_curve_plots(predictions_df, plots_dir)
    create_confidence_distribution_plot(
        confidence_df,
        summary_df,
        plots_dir / "confidence_distribution_best_model.png",
        pseudo_threshold,
    )
    create_pseudolabel_count_plot(threshold_df, plots_dir)
    create_summary_plot(summary_df, plots_dir / "summary_primary_metrics.png")
    create_site_distance_plots(
        distance_df, feature_shift_df, plots_dir, top_n=top_shift_features
    )
    create_site_pca_plot(
        raw_df, inferred, plots_dir / "site_pca_overview.png", max_points_per_site=350
    )


def create_legacy_report_compatibility_plots(
    fold_metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_dir: Path,
    preferred_model_name: str = "logistic_regression",
) -> None:
    if fold_metrics_df.empty or predictions_df.empty:
        return

    compat_dir = output_dir / "plots" / "legacy_report_compatibility"
    compat_dir.mkdir(parents=True, exist_ok=True)

    model_name = preferred_model_name
    if model_name not in fold_metrics_df["model_name"].astype(str).unique().tolist():
        model_name = str(fold_metrics_df["model_name"].mode(dropna=True).iloc[0])

    sub = fold_metrics_df.loc[fold_metrics_df["model_name"] == model_name].copy()
    if sub.empty:
        return

    variant_map = {
        "coral_only": "baseline",
        "coral_plus_pseudolabel": "transductive",
    }
    sub = sub.loc[sub["variant"].isin(variant_map)].copy()
    sub["method"] = sub["variant"].map(variant_map)

    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    mean_rows: list[dict[str, object]] = []
    for method_name in ["baseline", "transductive"]:
        method_sub = sub.loc[sub["method"] == method_name]
        if method_sub.empty:
            continue
        for metric in metrics:
            values = pd.to_numeric(method_sub[metric], errors="coerce")
            mean_rows.append(
                {
                    "method": method_name,
                    "metric": metric,
                    "mean": float(values.mean(skipna=True)),
                    "std": float(values.std(skipna=True, ddof=1))
                    if values.notna().sum() > 1
                    else 0.0,
                    "n_valid_folds": int(values.notna().sum()),
                }
            )
    mean_df = pd.DataFrame(mean_rows)

    precision_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    sites = sorted(sub["target_site"].astype(str).unique().tolist())
    for site in sites:
        base = sub.loc[(sub["target_site"] == site) & (sub["method"] == "baseline")]
        trans = sub.loc[
            (sub["target_site"] == site) & (sub["method"] == "transductive")
        ]
        if base.empty or trans.empty:
            continue
        base_row = base.iloc[0]
        trans_row = trans.iloc[0]
        precision_rows.append(
            {
                "target_site": site,
                "precision_baseline": float(base_row["precision"]),
                "precision_transductive": float(trans_row["precision"]),
                "delta_precision": float(
                    trans_row["precision"] - base_row["precision"]
                ),
            }
        )
        for metric in metrics:
            comparison_rows.append(
                {
                    "target_site": site,
                    "metric": metric,
                    "baseline": float(base_row[metric])
                    if pd.notna(base_row[metric])
                    else np.nan,
                    "transductive": float(trans_row[metric])
                    if pd.notna(trans_row[metric])
                    else np.nan,
                    "delta_transductive_minus_baseline": (
                        float(trans_row[metric] - base_row[metric])
                        if pd.notna(base_row[metric]) and pd.notna(trans_row[metric])
                        else np.nan
                    ),
                }
            )

    precision_df = pd.DataFrame(precision_rows)
    comparison_df = pd.DataFrame(comparison_rows)

    _plot_legacy_baseline_vs_transductive_by_site(
        sub, metrics, compat_dir / "barplot_baseline_vs_transductive_per_site.png"
    )
    _plot_legacy_mean_metrics(
        mean_df, metrics, compat_dir / "plot_metriche_medie_finali.png"
    )
    _plot_legacy_precision_by_site(
        precision_df, compat_dir / "precision_barplot_per_site.png"
    )
    _plot_legacy_precision_mean_aggregate(
        precision_df, compat_dir / "precision_media_aggregata.png"
    )
    _plot_legacy_precision_delta_by_site(
        precision_df, compat_dir / "precision_delta_per_site.png"
    )

    pred_sub = predictions_df.loc[predictions_df["model_name"] == model_name].copy()
    pred_sub = pred_sub.loc[pred_sub["variant"].isin(variant_map)].copy()
    pred_sub["method"] = pred_sub["variant"].map(variant_map)
    for site in sorted(pred_sub["target_site"].astype(str).unique().tolist()):
        for method_name in ["baseline", "transductive"]:
            site_sub = pred_sub.loc[
                (pred_sub["target_site"] == site) & (pred_sub["method"] == method_name)
            ]
            if site_sub.empty:
                continue
            cm = confusion_matrix(
                site_sub["y_true"].to_numpy(dtype=int),
                site_sub["y_pred"].to_numpy(dtype=int),
                labels=[0, 1],
            )
            plot_confusion_matrix_png(
                cm,
                title=f"{_display_name(model_name)} | {method_name} | target={site}",
                output_path=compat_dir / f"confusion_matrix_{method_name}_{site}.png",
            )

    mean_df.to_csv(compat_dir / "metriche_medie_finali.csv", index=False)
    precision_df.to_csv(compat_dir / "precision_confronto_per_site.csv", index=False)
    comparison_df.to_csv(
        compat_dir / "confronto_baseline_vs_transductive.csv", index=False
    )


def _draw_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    center: float | None = None,
    integer_annotations: bool = False,
) -> None:
    data = np.asarray(data, dtype=float)
    norm = None
    if center is not None:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
        norm=norm,
    )
    ax.set_xticks(
        np.arange(len(col_labels)), labels=col_labels, rotation=25, ha="right"
    )
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_title(title)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isnan(value):
                text = "NA"
            elif integer_annotations:
                text = f"{int(round(value))}"
            else:
                text = f"{value:.3f}"
            color = (
                "white"
                if np.isfinite(value) and _value_is_dark(value, vmin, vmax, center)
                else "black"
            )
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _value_is_dark(
    value: float, vmin: float, vmax: float, center: float | None
) -> bool:
    if not np.isfinite(value):
        return False
    if center is not None:
        span = max(vmax - center, center - vmin, 1e-8)
        relative = abs(value - center) / span
        return relative > 0.45
    relative = (value - vmin) / max(vmax - vmin, 1e-8)
    return relative > 0.55


def _display_name(model_name: str) -> str:
    return (
        model_name.replace("_", " ")
        .replace("regression", "Regression")
        .replace("forest", "Forest")
        .replace("boosting", "Boosting")
        .title()
    )


def _select_best_combo(summary_df: pd.DataFrame) -> tuple[str, str]:
    sorted_df = summary_df.sort_values(
        ["roc_auc_mean", "balanced_accuracy_mean", "f1_mean"], ascending=False
    )
    best_row = sorted_df.iloc[0]
    return str(best_row["model_name"]), str(best_row["variant"])


def _prepare_pca_matrix(df: pd.DataFrame, inferred: InferredColumns) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for column in inferred.feature_columns:
        series = df[column]
        if column in inferred.categorical_features:
            fill_value = series.mode(dropna=True)
            fill_value = fill_value.iloc[0] if not fill_value.empty else "__missing__"
            series = series.fillna(fill_value).astype(str)
            categories = pd.Index(series.unique())
            blocks.append(
                pd.Categorical(series, categories=categories).codes.astype(float)[
                    :, None
                ]
            )
        else:
            fill_value = pd.to_numeric(series, errors="coerce").median()
            if pd.isna(fill_value):
                fill_value = 0.0
            blocks.append(
                pd.to_numeric(series, errors="coerce")
                .fillna(fill_value)
                .to_numpy(dtype=float)[:, None]
            )
    return np.hstack(blocks)


def _plot_legacy_baseline_vs_transductive_by_site(
    fold_results_df: pd.DataFrame,
    metrics: list[str],
    output_path: Path,
) -> None:
    if fold_results_df.empty:
        return

    sites = sorted(fold_results_df["target_site"].astype(str).unique().tolist())
    methods = ["baseline", "transductive"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    method_colors = {"baseline": "#d6604d", "transductive": "#1a9850"}

    for ax, metric in zip(axes, metrics):
        x = np.arange(len(sites), dtype=float)
        width = 0.36
        for method_idx, method_name in enumerate(methods):
            values = []
            for site in sites:
                match = fold_results_df.loc[
                    (fold_results_df["target_site"] == site)
                    & (fold_results_df["method"] == method_name)
                ]
                values.append(
                    float(match.iloc[0][metric]) if not match.empty else np.nan
                )
            offset = (-0.5 * width) if method_idx == 0 else (0.5 * width)
            label = method_name if metric == metrics[0] else None
            ax.bar(
                x + offset,
                values,
                width=width,
                label=label,
                alpha=0.95,
                color=method_colors[method_name],
            )
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(sites, rotation=20)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Baseline vs Transductive per site target", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_legacy_mean_metrics(
    mean_df: pd.DataFrame, metrics: list[str], output_path: Path
) -> None:
    if mean_df.empty:
        return

    pivot_df = mean_df.pivot(index="metric", columns="method", values="mean").reindex(
        metrics
    )
    x = np.arange(len(metrics), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(
        x - width / 2,
        pivot_df.get("baseline", np.nan),
        width=width,
        label="baseline",
        color="#d6604d",
    )
    ax.bar(
        x + width / 2,
        pivot_df.get("transductive", np.nan),
        width=width,
        label="transductive",
        color="#1a9850",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mean score")
    ax.set_title("Metriche medie finali")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_legacy_precision_by_site(
    precision_df: pd.DataFrame, output_path: Path
) -> None:
    if precision_df.empty:
        return

    sites = precision_df["target_site"].astype(str).tolist()
    x = np.arange(len(sites), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))
    baseline_vals = precision_df["precision_baseline"].to_numpy(dtype=float)
    trans_vals = precision_df["precision_transductive"].to_numpy(dtype=float)
    bars_baseline = ax.bar(
        x - width / 2, baseline_vals, width=width, label="baseline", color="#d6604d"
    )
    bars_trans = ax.bar(
        x + width / 2, trans_vals, width=width, label="transductive", color="#1a9850"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("precision")
    ax.set_title("Precision per site target: baseline vs transductive")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)

    for bar in list(bars_baseline) + list(bars_trans):
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_legacy_precision_mean_aggregate(
    precision_df: pd.DataFrame, output_path: Path
) -> None:
    if precision_df.empty:
        return

    mean_baseline = float(precision_df["precision_baseline"].mean())
    mean_transductive = float(precision_df["precision_transductive"].mean())

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["baseline_mean", "transductive_mean"]
    values = [mean_baseline, mean_transductive]
    bars = ax.bar(labels, values, width=0.55, color=["#d6604d", "#1a9850"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("precision")
    ax.set_title("Precision media: baseline vs transductive")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_legacy_precision_delta_by_site(
    precision_df: pd.DataFrame, output_path: Path
) -> None:
    if precision_df.empty:
        return

    sites = precision_df["target_site"].astype(str).tolist()
    delta = precision_df["delta_precision"].to_numpy(dtype=float)
    x = np.arange(len(sites), dtype=float)
    bar_colors = ["#1a9850" if value >= 0 else "#d6604d" for value in delta]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(x, delta, color=bar_colors, width=0.55)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=20)
    ax.set_ylabel("delta precision (transductive - baseline)")
    ax.set_title("Delta precision per site target")
    ax.grid(axis="y", alpha=0.3)
    max_abs = max(0.02, float(np.max(np.abs(delta))) if len(delta) else 0.02)
    ax.set_ylim(-max_abs * 1.35, max_abs * 1.35)

    for bar, value in zip(bars, delta):
        offset = 0.005 if value >= 0 else -0.005
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.3f}",
            ha="center",
            va=va,
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
