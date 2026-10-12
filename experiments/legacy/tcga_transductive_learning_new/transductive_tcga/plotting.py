"""Transportability experiment: plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import colors
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PRIMARY_PLOT_METRICS

PLOT_DPI = 360
FEATURE_SHIFT_HEATMAP_TOP_N = 30
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
    }
)


def plot_confusion_matrix_png(
    matrix: np.ndarray, class_labels: list[str], title: str, output_path: Path
) -> None:
    fig, ax = plt.subplots(
        figsize=(0.9 * len(class_labels) + 3.5, 0.9 * len(class_labels) + 3.0)
    )
    im = ax.imshow(matrix, cmap=SEQUENTIAL_RED_GREEN)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(
        np.arange(len(class_labels)), labels=class_labels, rotation=35, ha="right"
    )
    ax.set_yticks(np.arange(len(class_labels)), labels=class_labels)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{int(matrix[row_idx, col_idx])}",
                ha="center",
                va="center",
                fontsize=9,
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_heatmaps(
    fold_metrics_df: pd.DataFrame, plots_dir: Path, metrics: list[str] | None = None
) -> None:
    if fold_metrics_df.empty:
        return

    metrics = metrics or list(PRIMARY_PLOT_METRICS)
    for variant in sorted(fold_metrics_df["variant"].astype(str).unique().tolist()):
        subset = fold_metrics_df.loc[fold_metrics_df["variant"] == variant].copy()
        if subset.empty:
            continue
        fig, axes = plt.subplots(
            1, len(metrics), figsize=(6 * len(metrics), 7), squeeze=False
        )
        for ax, metric in zip(axes.flatten(), metrics):
            pivot = (
                subset.pivot(index="target_domain", columns="model_name", values=metric)
                .sort_index()
                .sort_index(axis=1)
            )
            draw_heatmap(
                ax=ax,
                data=pivot.to_numpy(dtype=float),
                row_labels=pivot.index.astype(str).tolist(),
                col_labels=[display_name(name) for name in pivot.columns.tolist()],
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
        1, len(metrics), figsize=(6 * len(metrics), 7), squeeze=False
    )
    for ax, metric in zip(axes.flatten(), metrics):
        pivot = (
            variant_comparison_df.pivot(
                index="target_domain", columns="model_name", values=f"delta_{metric}"
            )
            .sort_index()
            .sort_index(axis=1)
        )
        values = pivot.to_numpy(dtype=float)
        bound = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
        bound = max(bound, 1e-6)
        draw_heatmap(
            ax=ax,
            data=values,
            row_labels=pivot.index.astype(str).tolist(),
            col_labels=[display_name(name) for name in pivot.columns.tolist()],
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
    fold_metrics_df: pd.DataFrame, plots_dir: Path, metrics: list[str] | None = None
) -> None:
    if fold_metrics_df.empty:
        return

    metrics = metrics or list(PRIMARY_PLOT_METRICS)
    variants = sorted(fold_metrics_df["variant"].astype(str).unique().tolist())
    fig, axes = plt.subplots(
        len(variants),
        len(metrics),
        figsize=(7 * len(metrics), 5.2 * len(variants)),
        squeeze=False,
    )

    for row_idx, variant in enumerate(variants):
        subset = fold_metrics_df.loc[fold_metrics_df["variant"] == variant]
        domains = sorted(subset["target_domain"].astype(str).unique().tolist())
        models = sorted(subset["model_name"].astype(str).unique().tolist())
        x = np.arange(len(domains), dtype=float)
        width = min(0.84 / max(len(models), 1), 0.22)

        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for model_idx, model_name in enumerate(models):
                model_sub = subset.loc[subset["model_name"] == model_name]
                values = []
                for domain in domains:
                    cell = model_sub.loc[model_sub["target_domain"] == domain, metric]
                    values.append(float(cell.iloc[0]) if not cell.empty else np.nan)
                offset = (model_idx - (len(models) - 1) / 2.0) * width
                ax.bar(
                    x + offset,
                    values,
                    width=width,
                    label=display_name(model_name),
                    color=MODEL_COLORS.get(model_name, "#666666"),
                )
            ax.set_xticks(x)
            ax.set_xticklabels(domains, rotation=30, ha="right")
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
    fig.savefig(
        plots_dir / "barplot_per_domain_primary_metrics.png", bbox_inches="tight"
    )
    plt.close(fig)


def create_boxplots(
    fold_metrics_df: pd.DataFrame, plots_dir: Path, metrics: list[str] | None = None
) -> None:
    if fold_metrics_df.empty:
        return

    metrics = metrics or list(PRIMARY_PLOT_METRICS)
    plot_df = fold_metrics_df.copy()
    plot_df["model_variant"] = [
        f"{display_name(model)}\n{VARIANT_LABELS.get(variant, variant)}"
        for model, variant in zip(plot_df["model_name"], plot_df["variant"])
    ]
    labels = plot_df["model_variant"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(6.2 * len(metrics), 7.2), squeeze=False
    )

    for ax, metric in zip(axes.flatten(), metrics):
        data = [
            plot_df.loc[plot_df["model_variant"] == label, metric].to_numpy(dtype=float)
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
        ax.tick_params(axis="x", rotation=28)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        plots_dir / "boxplot_primary_metrics_across_domains.png", bbox_inches="tight"
    )
    plt.close(fig)


def create_aggregate_curve_plots(predictions_df: pd.DataFrame, plots_dir: Path) -> None:
    if predictions_df.empty:
        return

    prob_columns = [
        column for column in predictions_df.columns if column.startswith("prob_")
    ]
    if not prob_columns:
        return

    combos = (
        predictions_df[["model_name", "variant"]]
        .drop_duplicates()
        .sort_values(["model_name", "variant"])
        .itertuples(index=False, name=None)
    )

    roc_fig, roc_ax = plt.subplots(figsize=(8.8, 7.5))
    pr_fig, pr_ax = plt.subplots(figsize=(8.8, 7.5))
    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="#444444", linewidth=1.0)

    for model_name, variant in combos:
        subset = predictions_df.loc[
            (predictions_df["model_name"] == model_name)
            & (predictions_df["variant"] == variant)
        ].copy()
        if subset.empty:
            continue
        y_true = subset["y_true"].to_numpy(dtype=int)
        y_prob = subset.loc[:, prob_columns].to_numpy(dtype=float)
        classes = np.arange(y_prob.shape[1], dtype=int)
        y_true_bin = label_binarize(y_true, classes=classes)
        if y_true_bin.shape[1] != y_prob.shape[1]:
            continue
        if y_true_bin.sum() == 0 or y_true_bin.sum() == y_true_bin.size:
            continue

        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        precision, recall, _ = precision_recall_curve(
            y_true_bin.ravel(), y_prob.ravel()
        )
        roc_value = auc(fpr, tpr)
        pr_value = auc(recall[::-1], precision[::-1])

        label = f"{display_name(model_name)} | {VARIANT_LABELS.get(variant, variant)}"
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
    roc_ax.set_title("Aggregate micro-averaged ROC curves across hold-out domains")
    roc_ax.grid(alpha=0.25)
    roc_ax.legend(frameon=False, loc="lower right")

    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title("Aggregate micro-averaged PR curves across hold-out domains")
    pr_ax.grid(alpha=0.25)
    pr_ax.legend(frameon=False, loc="lower left")

    roc_fig.tight_layout()
    roc_fig.savefig(plots_dir / "aggregate_roc_curves.png", bbox_inches="tight")
    plt.close(roc_fig)

    pr_fig.tight_layout()
    pr_fig.savefig(plots_dir / "aggregate_pr_curves.png", bbox_inches="tight")
    plt.close(pr_fig)


def create_reliability_plot(
    predictions_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    if predictions_df.empty or calibration_df.empty or summary_df.empty:
        return

    best_model_name, best_variant = select_best_combo(summary_df)
    cal_sub = calibration_df.loc[
        (calibration_df["model_name"] == best_model_name)
        & (calibration_df["variant"] == best_variant)
    ].copy()
    pred_sub = predictions_df.loc[
        (predictions_df["model_name"] == best_model_name)
        & (predictions_df["variant"] == best_variant)
    ].copy()
    if cal_sub.empty or pred_sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#444444", linewidth=1.0)
    axes[0].plot(
        cal_sub["mean_confidence"],
        cal_sub["empirical_accuracy"],
        marker="o",
        linewidth=2.0,
        color=MODEL_COLORS.get(best_model_name, "#1a9850"),
    )
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel("Mean top-label confidence")
    axes[0].set_ylabel("Observed top-label accuracy")
    axes[0].set_title(
        f"Top-label reliability\n{display_name(best_model_name)} | {VARIANT_LABELS.get(best_variant, best_variant)}"
    )
    axes[0].grid(alpha=0.25)

    axes[1].hist(
        pred_sub["confidence"],
        bins=20,
        color=MODEL_COLORS.get(best_model_name, "#1a9850"),
        alpha=0.8,
    )
    axes[1].set_xlabel("Top-label confidence")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Confidence distribution")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_confidence_distribution_plot(
    confidence_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
    threshold: float,
) -> None:
    if confidence_df.empty or summary_df.empty:
        return

    best_model_name, _ = select_best_combo(summary_df)
    subset = confidence_df.loc[confidence_df["model_name"] == best_model_name].copy()
    if subset.empty:
        return

    domains = sorted(subset["target_domain"].astype(str).unique().tolist())
    n_cols = 2
    n_rows = int(np.ceil(len(domains) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(11.5, 4.2 * n_rows), squeeze=False
    )

    for ax, domain in zip(axes.flatten(), domains):
        domain_sub = subset.loc[subset["target_domain"] == domain]
        not_selected = domain_sub.loc[
            domain_sub["selected_for_pseudolabeling"] == 0, "confidence"
        ]
        selected = domain_sub.loc[
            domain_sub["selected_for_pseudolabeling"] == 1, "confidence"
        ]
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
        ax.axvline(threshold, linestyle="--", color="#333333", linewidth=1.0)
        ax.set_title(domain)
        ax.set_xlabel("Top-label confidence")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)

    for ax in axes.flatten()[len(domains) :]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        f"Confidence distribution on hold-out target domains\nBest model: {display_name(best_model_name)}"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_pseudolabel_count_plot(threshold_df: pd.DataFrame, plots_dir: Path) -> None:
    if threshold_df.empty:
        return

    model_names = sorted(threshold_df["model_name"].astype(str).unique().tolist())
    fig, axes = plt.subplots(
        1, len(model_names), figsize=(6 * len(model_names), 7), squeeze=False
    )
    for ax, model_name in zip(axes.flatten(), model_names):
        subset = threshold_df.loc[threshold_df["model_name"] == model_name]
        pivot = (
            subset.pivot(
                index="target_domain",
                columns="threshold",
                values="n_selected_after_caps",
            )
            .sort_index()
            .sort_index(axis=1)
        )
        vmax = (
            float(np.nanmax(pivot.to_numpy(dtype=float)))
            if np.isfinite(pivot.to_numpy(dtype=float)).any()
            else 1.0
        )
        draw_heatmap(
            ax=ax,
            data=pivot.to_numpy(dtype=float),
            row_labels=pivot.index.astype(str).tolist(),
            col_labels=[f"{float(col):.2f}" for col in pivot.columns.tolist()],
            title=f"Accepted pseudo-labels\n{display_name(model_name)}",
            cmap=SEQUENTIAL_RED_GREEN,
            vmin=0.0,
            vmax=vmax,
            integer_annotations=True,
        )
    fig.tight_layout()
    fig.savefig(
        plots_dir / "pseudolabel_counts_by_domain_threshold.png", bbox_inches="tight"
    )
    plt.close(fig)


def create_summary_plot(
    summary_df: pd.DataFrame, output_path: Path, metrics: list[str] | None = None
) -> None:
    if summary_df.empty:
        return

    metrics = metrics or list(PRIMARY_PLOT_METRICS)
    labels = [
        f"{display_name(model)}\n{VARIANT_LABELS.get(variant, variant)}"
        for model, variant in zip(summary_df["model_name"], summary_df["variant"])
    ]
    x = np.arange(len(labels), dtype=float)
    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(12.2, 4.1 * len(metrics)), squeeze=False
    )

    for ax, metric in zip(axes.flatten(), metrics):
        means = summary_df[f"{metric}_mean"].to_numpy(dtype=float)
        stds = summary_df[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
        point_colors = [
            SEQUENTIAL_RED_GREEN(np.clip(value, 0.0, 1.0)) for value in means
        ]
        ax.errorbar(
            x, means, yerr=stds, fmt="none", color="#555555", capsize=5, linewidth=1.4
        )
        ax.scatter(
            x, means, s=75, c=point_colors, edgecolor="#333333", linewidth=0.8, zorder=3
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_site_distance_plots(
    distance_df: pd.DataFrame, feature_shift_df: pd.DataFrame, plots_dir: Path
) -> None:
    if not distance_df.empty:
        domains = distance_df["target_domain"].astype(str).tolist()
        x = np.arange(len(domains), dtype=float)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.7))
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
            ax.set_xticks(x)
            ax.set_xticklabels(domains, rotation=30, ha="right")
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
        .head(FEATURE_SHIFT_HEATMAP_TOP_N)
        .index.tolist()
    )
    heatmap_df = feature_shift_df.loc[
        feature_shift_df["feature"].isin(top_features)
    ].copy()
    pivot = (
        heatmap_df.pivot(index="target_domain", columns="feature", values="abs_smd")
        .sort_index()
        .sort_index(axis=1)
    )
    vmax = (
        float(np.nanmax(pivot.to_numpy(dtype=float)))
        if np.isfinite(pivot.to_numpy(dtype=float)).any()
        else 1.0
    )
    fig_width = max(12.0, 0.55 * max(pivot.shape[1], 4))
    fig, ax = plt.subplots(figsize=(fig_width, 6.8))
    draw_heatmap(
        ax=ax,
        data=pivot.to_numpy(dtype=float),
        row_labels=pivot.index.astype(str).tolist(),
        col_labels=pivot.columns.astype(str).tolist(),
        title=f"Top {len(top_features)} shifted selected genes by hold-out domain",
        cmap=SEQUENTIAL_RED_GREEN,
        vmin=0.0,
        vmax=vmax,
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "feature_shift_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def create_alignment_pca_plot(
    alignment_plot_df: pd.DataFrame, output_path: Path
) -> None:
    if alignment_plot_df.empty:
        return

    domains = sorted(alignment_plot_df["target_domain"].astype(str).unique().tolist())
    fig, axes = plt.subplots(
        len(domains), 2, figsize=(12, 3.2 * len(domains)), squeeze=False
    )
    subset_colors = {"source": "#d6604d", "target": "#1a9850"}
    phase_titles = {"pre_alignment": "Before CORAL", "post_alignment": "After CORAL"}

    for row_idx, domain in enumerate(domains):
        domain_sub = alignment_plot_df.loc[alignment_plot_df["target_domain"] == domain]
        for col_idx, phase in enumerate(["pre_alignment", "post_alignment"]):
            ax = axes[row_idx, col_idx]
            phase_sub = domain_sub.loc[domain_sub["phase"] == phase]
            for subset_name in ["source", "target"]:
                points = phase_sub.loc[phase_sub["subset"] == subset_name]
                ax.scatter(
                    points["pc1"],
                    points["pc2"],
                    s=18,
                    alpha=0.55,
                    label=subset_name if row_idx == 0 and col_idx == 0 else None,
                    color=subset_colors[subset_name],
                )
            ax.set_title(f"{domain} | {phase_titles[phase]}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.grid(alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_all_plots(
    fold_metrics_df: pd.DataFrame,
    variant_comparison_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    confidence_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    feature_shift_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    alignment_plot_df: pd.DataFrame,
    plots_dir: Path,
    pseudo_threshold: float,
) -> None:
    create_heatmaps(fold_metrics_df, plots_dir)
    create_variant_heatmap(variant_comparison_df, plots_dir)
    create_site_barplots(fold_metrics_df, plots_dir)
    create_boxplots(fold_metrics_df, plots_dir)
    create_aggregate_curve_plots(predictions_df, plots_dir)
    create_reliability_plot(
        predictions_df,
        calibration_df,
        summary_df,
        plots_dir / "calibration_reliability_best_model.png",
    )
    create_confidence_distribution_plot(
        confidence_df,
        summary_df,
        plots_dir / "confidence_distribution_best_model.png",
        pseudo_threshold,
    )
    create_pseudolabel_count_plot(threshold_df, plots_dir)
    create_summary_plot(summary_df, plots_dir / "summary_primary_metrics.png")
    create_site_distance_plots(distance_df, feature_shift_df, plots_dir)
    create_alignment_pca_plot(
        alignment_plot_df, plots_dir / "alignment_pca_before_after.png"
    )


def draw_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cmap,
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
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            if np.isnan(value):
                text = "NA"
            elif integer_annotations:
                text = f"{int(round(value))}"
            else:
                text = f"{value:.3f}"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8.5)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def display_name(model_name: str) -> str:
    return (
        model_name.replace("_", " ")
        .replace("regression", "Regression")
        .replace("forest", "Forest")
        .replace("boosting", "Boosting")
        .title()
    )


def select_best_combo(summary_df: pd.DataFrame) -> tuple[str, str]:
    sort_cols = ["balanced_accuracy_mean", "f1_macro_mean", "roc_auc_macro_mean"]
    best_row = summary_df.sort_values(sort_cols, ascending=False).iloc[0]
    return str(best_row["model_name"]), str(best_row["variant"])
