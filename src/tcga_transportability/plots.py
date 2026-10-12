"""Transportability pipeline module: plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from tcga_transportability.config import (
    METHOD_ORDER,
    ExperimentConfig,
    pretty_model_name,
    pretty_target_site,
)

matplotlib.use("Agg")


@dataclass(frozen=True)
class PlotMetric:
    column: str
    title: str
    ylabel: str
    higher_is_better: bool
    fixed_bounds: tuple[float, float] | None
    fmt: str = ".3f"


PLOT_METRICS = (
    PlotMetric("roc_auc_ood", "OOD ROC-AUC", "ROC-AUC", True, (0.0, 1.0)),
    PlotMetric("f1_ood", "OOD F1-score", "F1-score", True, (0.0, 1.0)),
    PlotMetric(
        "balanced_accuracy_ood",
        "OOD Balanced Accuracy",
        "Balanced Accuracy",
        True,
        (0.0, 1.0),
    ),
    PlotMetric("ece_ood", "OOD Expected Calibration Error", "ECE", False, None),
    PlotMetric("gap_roc_auc", "ID - OOD ROC-AUC Gap", "Gap", False, None),
    PlotMetric("gap_f1", "ID - OOD F1-score Gap", "Gap", False, None),
    PlotMetric(
        "gap_balanced_accuracy", "ID - OOD Balanced Accuracy Gap", "Gap", False, None
    ),
)


def apply_base_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#555555",
            "axes.labelcolor": "#222222",
            "axes.grid": False,
            "font.size": 11,
            "savefig.bbox": "tight",
        }
    )


def thesis_cmap(*, higher_is_better: bool):
    return plt.get_cmap("RdYlGn" if higher_is_better else "RdYlGn_r")


def _site_order(results_df: pd.DataFrame) -> list[str]:
    return sorted(results_df["target_site"].dropna().astype(str).unique().tolist())


def _method_order(results_df: pd.DataFrame) -> list[str]:
    observed = results_df["method"].dropna().astype(str).unique().tolist()
    ordered = [method for method in METHOD_ORDER if method in observed]
    extras = [method for method in observed if method not in ordered]
    return ordered + extras


def _value_bounds(values: np.ndarray, metric: PlotMetric) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return (0.0, 1.0)
    if metric.fixed_bounds is not None:
        return metric.fixed_bounds
    vmin = float(finite_values.min())
    vmax = float(finite_values.max())
    if np.isclose(vmin, vmax):
        delta = 0.05 if np.isclose(vmin, 0.0) else abs(vmin) * 0.1
        return (vmin - delta, vmax + delta)
    return (vmin, vmax)


def _chunked_site_order(site_order: list[str], chunk_size: int = 7) -> list[list[str]]:
    if not site_order:
        return []
    return [
        site_order[idx : idx + chunk_size]
        for idx in range(0, len(site_order), chunk_size)
    ]


def save_figure_bundle(fig: plt.Figure, output_path_no_suffix: Path) -> None:
    output_path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path_no_suffix.with_suffix(".png"), dpi=200)
    fig.savefig(output_path_no_suffix.with_suffix(".pdf"))
    plt.close(fig)


def plot_metric_barplot(
    *,
    results_df: pd.DataFrame,
    model_name: str,
    metric: PlotMetric,
    output_dir: Path,
) -> None:
    subset = results_df.loc[results_df["model"] == model_name].copy()
    if subset.empty or metric.column not in subset.columns:
        return
    model_display = pretty_model_name(model_name)

    pivot = subset.pivot_table(
        index="target_site", columns="method", values=metric.column, aggfunc="mean"
    )
    site_order = _site_order(subset)
    method_order = _method_order(subset)
    pivot = pivot.reindex(index=site_order, columns=method_order)
    if pivot.empty:
        return

    apply_base_style()
    values = pivot.to_numpy(dtype=float)
    vmin, vmax = _value_bounds(values, metric)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = thesis_cmap(higher_is_better=metric.higher_is_better)
    y_values = values[np.isfinite(values)]
    if metric.fixed_bounds is not None:
        y_min, y_max = metric.fixed_bounds
    elif y_values.size == 0:
        y_min, y_max = (0.0, 1.0)
    else:
        padding = max(0.02, (float(y_values.max()) - float(y_values.min())) * 0.15)
        y_min = float(y_values.min()) - padding
        y_max = float(y_values.max()) + padding
        if metric.column == "ece_ood":
            y_min = min(0.0, y_min)
        if metric.column.startswith("gap_"):
            y_min = min(0.0, y_min)

    site_chunks = _chunked_site_order(site_order, chunk_size=7)
    n_rows = max(1, len(site_chunks))
    max_sites_in_row = max(len(chunk_sites) for chunk_sites in site_chunks)
    fig_width = max(12.0, max_sites_in_row * max(1.45, len(pivot.columns) * 0.72))
    fig_height = 5.2 * n_rows + 0.8
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes], dtype=object)

    fig.suptitle(f"{metric.title} by target site | {model_display}", y=0.985)

    for row_idx, (ax, chunk_sites) in enumerate(zip(axes, site_chunks, strict=False)):
        chunk_pivot = pivot.loc[chunk_sites]
        x_positions = np.arange(len(chunk_pivot.index))
        width = 0.8 / max(1, len(chunk_pivot.columns))

        bar_tick_positions: list[float] = []
        bar_tick_labels: list[str] = []
        for method_idx, method_name in enumerate(chunk_pivot.columns):
            series = chunk_pivot[method_name].to_numpy(dtype=float)
            bar_positions = (
                x_positions
                + method_idx * width
                - (len(chunk_pivot.columns) - 1) * width / 2
            )
            bar_tick_positions.extend(bar_positions.tolist())
            bar_tick_labels.extend([str(method_name)] * len(bar_positions))
            bar_colors = [
                cmap(norm(value)) if np.isfinite(value) else (0.82, 0.82, 0.82, 1.0)
                for value in series
            ]
            bars = ax.bar(
                bar_positions,
                series,
                width=width,
                color=bar_colors,
                edgecolor="#404040",
                linewidth=0.6,
            )
            for bar, value in zip(bars, series):
                if not np.isfinite(value):
                    continue
                y_offset = 0.015 if value >= 0 else -0.03
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + y_offset,
                    format(float(value), metric.fmt),
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=7,
                )

        if row_idx == 0:
            ax.set_ylabel(metric.ylabel)
        else:
            ax.set_ylabel(metric.ylabel)
        ax.set_xticks(bar_tick_positions)
        ax.set_xticklabels(bar_tick_labels, rotation=0, ha="center", fontsize=9)
        ax.tick_params(axis="x", which="major", pad=3, length=0)
        for site_position, site in zip(x_positions, chunk_pivot.index, strict=False):
            ax.text(
                site_position,
                -0.105,
                pretty_target_site(site),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
                color="#222222",
            )
        ax.set_ylim(y_min, y_max)
        ax.grid(False)
        ax.margins(x=0.02)
        if row_idx == n_rows - 1:
            ax.set_xlabel("Target site", labelpad=44)

    for stale_path in output_dir.glob(f"{metric.column}_{model_name.lower()}_part*.*"):
        stale_path.unlink()
    save_figure_bundle(fig, output_dir / f"{metric.column}_{model_name.lower()}")


def plot_metric_heatmap(
    *,
    results_df: pd.DataFrame,
    model_name: str,
    metric: PlotMetric,
    output_dir: Path,
) -> None:
    subset = results_df.loc[results_df["model"] == model_name].copy()
    if subset.empty or metric.column not in subset.columns:
        return
    model_display = pretty_model_name(model_name)

    pivot = subset.pivot_table(
        index="method", columns="target_site", values=metric.column, aggfunc="mean"
    )
    site_order = _site_order(subset)
    method_order = _method_order(subset)
    pivot = pivot.reindex(index=method_order, columns=site_order)
    if pivot.empty:
        return

    values = pivot.to_numpy(dtype=float)
    vmin, vmax = _value_bounds(values, metric)
    cmap = thesis_cmap(higher_is_better=metric.higher_is_better)

    apply_base_style()
    fig, ax = plt.subplots(
        figsize=(max(10.0, 0.9 * len(site_order)), max(3.0, 1.1 * len(method_order)))
    )
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{metric.title} heatmap | {model_display}")
    ax.set_xlabel("Target site")
    ax.set_ylabel("Method")
    ax.set_xticks(np.arange(len(site_order)))
    ax.set_xticklabels(
        [pretty_target_site(site) for site in site_order], rotation=35, ha="right"
    )
    ax.set_yticks(np.arange(len(method_order)))
    ax.set_yticklabels(method_order)
    ax.grid(False)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            label = "NA" if pd.isna(value) else format(float(value), metric.fmt)
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                color="#111111",
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.set_ylabel(metric.ylabel, rotation=90)
    save_figure_bundle(fig, output_dir / f"{metric.column}_{model_name.lower()}")


def generate_all_plots(
    *,
    results_df: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: Path,
) -> None:
    del config
    for model_name in results_df["model"].dropna().astype(str).unique():
        for metric in PLOT_METRICS:
            plot_metric_barplot(
                results_df=results_df,
                model_name=model_name,
                metric=metric,
                output_dir=output_dir / "barplots",
            )
            plot_metric_heatmap(
                results_df=results_df,
                model_name=model_name,
                metric=metric,
                output_dir=output_dir / "heatmaps",
            )
