"""Transportability experiment: bootstrap reporting."""

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


import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcga_transportability.config import (
    ExperimentConfig,
    pretty_model_name,
    pretty_target_site,
)


TARGET_RESULT_METRICS = [
    "roc_auc_id",
    "roc_auc_ood",
    "gap_roc_auc",
    "f1_id",
    "f1_ood",
    "gap_f1",
    "balanced_accuracy_id",
    "balanced_accuracy_ood",
    "gap_balanced_accuracy",
    "ece_id",
    "ece_ood",
    "decision_threshold",
]

MODEL_SUMMARY_GROUP_COLUMNS = ["method", "model", "model_display"]
TARGET_RESULT_GROUP_COLUMNS = [
    "method",
    "model",
    "model_display",
    "target_site",
    "target_site_display",
]

ERRORBAR_STYLE = {
    "ecolor": "black",
    "elinewidth": 1.45,
    "capsize": 4.2,
    "capthick": 1.45,
}


def _ensure_dirs(output_dir: Path) -> dict[str, Path]:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    reports_dir = output_dir / "reports"
    logs_dir = output_dir / "logs"
    for directory in [
        tables_dir,
        figures_dir / "barplots",
        figures_dir / "heatmaps",
        reports_dir,
        logs_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "tables": tables_dir,
        "figures": figures_dir,
        "reports": reports_dir,
        "logs": logs_dir,
    }


def _collapse_static(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return ""
    unique_values = non_null.astype(str).unique().tolist()
    if len(unique_values) == 1:
        value = non_null.iloc[0]
        if isinstance(value, np.generic):
            return value.item()
        return value
    return "<varies_across_bootstrap>"


def _aggregate_frame(
    *,
    raw_df: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    excluded = set(
        group_columns + metric_columns + ["bootstrap_index", "bootstrap_seed"]
    )
    static_columns = [column for column in raw_df.columns if column not in excluded]

    static_df = (
        raw_df.groupby(group_columns, dropna=False)[static_columns]
        .agg(_collapse_static)
        .reset_index()
    )
    mean_df = (
        raw_df.groupby(group_columns, dropna=False)[metric_columns].mean().reset_index()
    )
    std_df = (
        raw_df.groupby(group_columns, dropna=False)[metric_columns]
        .std(ddof=0)
        .fillna(0.0)
        .rename(columns={metric: f"{metric}_std" for metric in metric_columns})
        .reset_index()
    )
    count_df = (
        raw_df.groupby(group_columns, dropna=False)
        .size()
        .reset_index(name="n_bootstrap")
    )

    merged = static_df.merge(mean_df, on=group_columns, how="left")
    merged = merged.merge(std_df, on=group_columns, how="left")
    merged = merged.merge(count_df, on=group_columns, how="left")
    return merged


def aggregate_target_site_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = _aggregate_frame(
        raw_df=raw_df,
        group_columns=TARGET_RESULT_GROUP_COLUMNS,
        metric_columns=TARGET_RESULT_METRICS,
    )
    if summary_df.empty:
        return summary_df
    return summary_df.sort_values(["method", "model", "target_site"]).reset_index(
        drop=True
    )


def aggregate_model_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    metric_columns = [
        column
        for column in raw_df.columns
        if column.endswith("_mean") and column not in {"n_target_sites_mean"}
    ]
    summary_df = _aggregate_frame(
        raw_df=raw_df,
        group_columns=MODEL_SUMMARY_GROUP_COLUMNS,
        metric_columns=metric_columns,
    )
    if summary_df.empty:
        return summary_df
    return summary_df.sort_values(["method", "model"]).reset_index(drop=True)


def plot_metric_barplot_with_errorbars(
    *,
    results_df: pd.DataFrame,
    model_name: str,
    metric,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors

    from tcga_transportability.plots import (
        _chunked_site_order,
        _method_order,
        _site_order,
        _value_bounds,
        apply_base_style,
        save_figure_bundle,
        thesis_cmap,
    )

    subset = results_df.loc[results_df["model"] == model_name].copy()
    if subset.empty or metric.column not in subset.columns:
        return

    error_column = f"{metric.column}_std"
    if error_column not in subset.columns:
        subset[error_column] = 0.0

    model_display = pretty_model_name(model_name)
    pivot_mean = subset.pivot_table(
        index="target_site", columns="method", values=metric.column, aggfunc="mean"
    )
    pivot_std = subset.pivot_table(
        index="target_site", columns="method", values=error_column, aggfunc="mean"
    )

    site_order = _site_order(subset)
    method_order = _method_order(subset)
    pivot_mean = pivot_mean.reindex(index=site_order, columns=method_order)
    pivot_std = pivot_std.reindex(index=site_order, columns=method_order).fillna(0.0)
    if pivot_mean.empty:
        return

    apply_base_style()
    values = pivot_mean.to_numpy(dtype=float)
    errors = pivot_std.to_numpy(dtype=float)
    vmin, vmax = _value_bounds(values, metric)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = thesis_cmap(higher_is_better=metric.higher_is_better)

    y_values = values[np.isfinite(values)]
    y_errors = errors[np.isfinite(errors)]
    max_error = float(y_errors.max()) if y_errors.size else 0.0
    if metric.fixed_bounds is not None:
        y_min, y_max = metric.fixed_bounds
    elif y_values.size == 0:
        y_min, y_max = (0.0, 1.0)
    else:
        padding = max(0.02, (float(y_values.max()) - float(y_values.min())) * 0.15)
        padding = max(padding, max_error * 1.35)
        y_min = float(y_values.min()) - padding
        y_max = float(y_values.max()) + padding
        if metric.column == "ece_ood":
            y_min = min(0.0, y_min)
        if metric.column.startswith("gap_"):
            y_min = min(0.0, y_min)

    site_chunks = _chunked_site_order(site_order, chunk_size=7)
    n_rows = max(1, len(site_chunks))
    max_sites_in_row = max(len(chunk_sites) for chunk_sites in site_chunks)
    fig_width = max(12.0, max_sites_in_row * max(1.45, len(pivot_mean.columns) * 0.72))
    fig_height = 5.2 * n_rows + 0.8
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes], dtype=object)

    fig.suptitle(f"{metric.title} by target site | {model_display}", y=0.985)

    for row_idx, (ax, chunk_sites) in enumerate(zip(axes, site_chunks, strict=False)):
        chunk_pivot = pivot_mean.loc[chunk_sites]
        chunk_errors = pivot_std.loc[chunk_sites]
        x_positions = np.arange(len(chunk_pivot.index))
        width = 0.8 / max(1, len(chunk_pivot.columns))

        bar_tick_positions: list[float] = []
        bar_tick_labels: list[str] = []
        for method_idx, method_name in enumerate(chunk_pivot.columns):
            series = chunk_pivot[method_name].to_numpy(dtype=float)
            series_std = chunk_errors[method_name].to_numpy(dtype=float)
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
                yerr=series_std,
                error_kw=ERRORBAR_STYLE,
                zorder=3,
            )
            for bar, value, std_value in zip(bars, series, series_std, strict=False):
                if not np.isfinite(value):
                    continue
                visible_error = abs(float(std_value)) if np.isfinite(std_value) else 0.0
                y_offset = max(0.015, visible_error + 0.01)
                text_y = value + y_offset if value >= 0 else value - y_offset
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    text_y,
                    format(float(value), metric.fmt),
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=7,
                )

        ax.set_ylabel(metric.ylabel)
        ax.set_xticks(bar_tick_positions)
        ax.set_xticklabels(bar_tick_labels, rotation=0, ha="center", fontsize=9)
        ax.tick_params(axis="x", which="major", pad=3, length=0)
        for site_position, site in zip(x_positions, chunk_pivot.index, strict=False):
            display_name = pretty_target_site(site)
            if "target_site_display" in subset.columns:
                display_matches = subset.loc[
                    subset["target_site"] == site, "target_site_display"
                ].dropna()
                if not display_matches.empty:
                    display_name = str(display_matches.iloc[0])
            ax.text(
                site_position,
                -0.105,
                display_name,
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


def generate_bootstrap_plots(
    *,
    results_summary_df: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: Path,
) -> None:
    from tcga_transportability.plots import PLOT_METRICS, plot_metric_heatmap

    del config
    for model_name in results_summary_df["model"].dropna().astype(str).unique():
        for metric in PLOT_METRICS:
            plot_metric_barplot_with_errorbars(
                results_df=results_summary_df,
                model_name=model_name,
                metric=metric,
                output_dir=output_dir / "barplots",
            )
            plot_metric_heatmap(
                results_df=results_summary_df,
                model_name=model_name,
                metric=metric,
                output_dir=output_dir / "heatmaps",
            )


def save_bootstrap_tables(
    *,
    results_summary_df: pd.DataFrame,
    results_raw_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    model_summary_raw_df: pd.DataFrame,
    id_validation_fold_metrics_df: pd.DataFrame,
    prediction_records_df: pd.DataFrame,
    split_manifest_df: pd.DataFrame,
    split_assignments_df: pd.DataFrame,
    m2_pairwise_weighting_df: pd.DataFrame,
    bootstrap_iterations_df: pd.DataFrame,
    config: ExperimentConfig,
    bootstrap_metadata: dict[str, Any],
    logger: logging.Logger | None = None,
) -> None:
    directories = _ensure_dirs(config.output_dir)
    tables_dir = directories["tables"]

    results_summary_df.to_csv(tables_dir / "target_site_results.csv", index=False)
    results_raw_df.to_csv(
        tables_dir / "target_site_results_bootstrap_raw.csv", index=False
    )
    model_summary_df.to_csv(tables_dir / "model_summary.csv", index=False)
    model_summary_raw_df.to_csv(
        tables_dir / "model_summary_bootstrap_raw.csv", index=False
    )
    id_validation_fold_metrics_df.to_csv(
        tables_dir / "id_validation_fold_metrics.csv", index=False
    )
    prediction_records_df.to_csv(tables_dir / "prediction_records.csv", index=False)
    split_manifest_df.to_csv(tables_dir / "target_site_split_manifest.csv", index=False)
    split_assignments_df.to_csv(
        tables_dir / "target_site_split_assignments.csv", index=False
    )
    m2_pairwise_weighting_df.to_csv(
        tables_dir / "m2_pairwise_source_weighting.csv", index=False
    )
    bootstrap_iterations_df.to_csv(tables_dir / "bootstrap_iterations.csv", index=False)
    (tables_dir / "bootstrap_metadata.json").write_text(
        json.dumps(bootstrap_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if logger is not None:
        logger.info("Saved bootstrap tables under %s", config.output_dir)


def save_bootstrap_outputs(
    *,
    results_summary_df: pd.DataFrame,
    results_raw_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    model_summary_raw_df: pd.DataFrame,
    id_validation_fold_metrics_df: pd.DataFrame,
    prediction_records_df: pd.DataFrame,
    split_manifest_df: pd.DataFrame,
    split_assignments_df: pd.DataFrame,
    m2_pairwise_weighting_df: pd.DataFrame,
    bootstrap_iterations_df: pd.DataFrame,
    config: ExperimentConfig,
    bootstrap_metadata: dict[str, Any],
    logger: logging.Logger,
) -> None:
    save_bootstrap_tables(
        results_summary_df=results_summary_df,
        results_raw_df=results_raw_df,
        model_summary_df=model_summary_df,
        model_summary_raw_df=model_summary_raw_df,
        id_validation_fold_metrics_df=id_validation_fold_metrics_df,
        prediction_records_df=prediction_records_df,
        split_manifest_df=split_manifest_df,
        split_assignments_df=split_assignments_df,
        m2_pairwise_weighting_df=m2_pairwise_weighting_df,
        bootstrap_iterations_df=bootstrap_iterations_df,
        config=config,
        bootstrap_metadata=bootstrap_metadata,
        logger=None,
    )

    generate_bootstrap_plots(
        results_summary_df=results_summary_df,
        config=config,
        output_dir=_ensure_dirs(config.output_dir)["figures"],
    )
    logger.info("Saved bootstrap tables and figures under %s", config.output_dir)


def _markdown_table(df: pd.DataFrame, *, max_rows: int = 200) -> list[str]:
    if df.empty:
        return ["No rows available."]
    preview = df.head(max_rows).copy().fillna("NA")
    headers = [str(column) for column in preview.columns]
    rows = [[str(value) for value in row] for row in preview.to_numpy()]
    widths = [
        max(len(headers[col_idx]), *(len(row[col_idx]) for row in rows))
        for col_idx in range(len(headers))
    ]

    def format_row(values: list[str]) -> str:
        return (
            "| "
            + " | ".join(value.ljust(width) for value, width in zip(values, widths))
            + " |"
        )

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return lines


def write_bootstrap_markdown_report(
    *,
    config: ExperimentConfig,
    output_path: Path,
    bootstrap_metadata: dict[str, Any],
    results_summary_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    split_manifest_df: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines: list[str] = [
        "# TCGA-BRCA bootstrap transportability report",
        "",
        "## Setup",
        f"- Dataset: `{config.dataset_name}`",
        f"- Enabled methods: `{', '.join(config.enabled_methods)}`",
        f"- `n_bootstrap`: `{bootstrap_metadata['n_bootstrap']}`",
        f"- Base random seed: `{bootstrap_metadata['base_seed']}`",
        f"- Bootstrap seeds used: `{', '.join(str(seed) for seed in bootstrap_metadata['bootstrap_seeds'])}`",
        "- Target-site protocol: for each bootstrap iteration, each target site is re-split into `train_1` and `test_1` with a shuffled 50/50 split.",
        "- Source-site protocol: all non-target sites are used as source domains for initial training.",
        "- ID metric definition: grouped validation on source domains using up to 3 source-domain folds.",
        f"- Operating threshold: selected separately inside each bootstrap on out-of-fold source-domain ID predictions by maximizing balanced accuracy (tie-break: F1), starting from reference threshold `{config.threshold:.3f}`, and then applied unchanged to `test_1`.",
        f"- TCGA train-only preprocessing: near-zero variance gene removal, top-variance gene prefiltering (`{config.top_k_variance_genes}` genes), ANOVA feature selection (`{config.top_k_model_genes}` genes), and source-only ComBat gene pools (`{config.combat_top_k_variance_genes}` genes max) are re-fit inside each bootstrap iteration.",
        "- Summary protocol: means and standard deviations are computed across bootstrap iterations; standard deviations are used as black error bars in the final grouped barplots.",
        "",
        "## Bootstrap iterations",
        "",
    ]
    report_lines.extend(_markdown_table(pd.DataFrame(bootstrap_metadata["iterations"])))
    report_lines.extend(["", "## Target split manifest", ""])
    report_lines.extend(_markdown_table(split_manifest_df))
    report_lines.extend(["", "## Target-site result summary", ""])

    results_report_df = results_summary_df.copy()
    for column in [
        "method",
        "model_display",
        "target_site_display",
        "decision_threshold",
        "decision_threshold_std",
        "roc_auc_ood",
        "roc_auc_ood_std",
        "f1_ood",
        "f1_ood_std",
        "balanced_accuracy_ood",
        "balanced_accuracy_ood_std",
        "ece_ood",
        "ece_ood_std",
    ]:
        if column not in results_report_df.columns:
            results_report_df[column] = pd.NA
    report_lines.extend(
        _markdown_table(
            results_report_df[
                [
                    "method",
                    "model_display",
                    "target_site_display",
                    "decision_threshold",
                    "decision_threshold_std",
                    "roc_auc_ood",
                    "roc_auc_ood_std",
                    "f1_ood",
                    "f1_ood_std",
                    "balanced_accuracy_ood",
                    "balanced_accuracy_ood_std",
                    "ece_ood",
                    "ece_ood_std",
                ]
            ]
        )
    )
    report_lines.extend(["", "## Model summary", ""])
    summary_report_df = model_summary_df.copy()
    for column in [
        "method",
        "model_display",
        "decision_threshold_mean",
        "decision_threshold_mean_std",
        "roc_auc_ood_mean",
        "roc_auc_ood_mean_std",
        "f1_ood_mean",
        "f1_ood_mean_std",
        "balanced_accuracy_ood_mean",
        "balanced_accuracy_ood_mean_std",
        "ece_ood_mean",
        "ece_ood_mean_std",
    ]:
        if column not in summary_report_df.columns:
            summary_report_df[column] = pd.NA
    report_lines.extend(
        _markdown_table(
            summary_report_df[
                [
                    "method",
                    "model_display",
                    "decision_threshold_mean",
                    "decision_threshold_mean_std",
                    "roc_auc_ood_mean",
                    "roc_auc_ood_mean_std",
                    "f1_ood_mean",
                    "f1_ood_mean_std",
                    "balanced_accuracy_ood_mean",
                    "balanced_accuracy_ood_mean_std",
                    "ece_ood_mean",
                    "ece_ood_mean_std",
                ]
            ]
        )
    )
    report_lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `tables/target_site_results.csv`: bootstrap summary table with mean and std for each target site, method, and model.",
            "- `tables/target_site_results_bootstrap_raw.csv`: one row per bootstrap, target site, method, and model.",
            "- `tables/model_summary.csv`: summary of model-level means and standard deviations across bootstrap iterations.",
            "- `tables/model_summary_bootstrap_raw.csv`: raw model summaries for each bootstrap iteration.",
            "- `tables/bootstrap_iterations.csv`: bootstrap index and seed mapping.",
            "- `tables/bootstrap_metadata.json`: bootstrap configuration metadata.",
            "- `figures/barplots/`: grouped TCGA barplots with black error bars and the same two-row site layout as the original figures.",
            "- `figures/heatmaps/`: mean heatmaps per metric and model.",
        ]
    )
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
