"""Transportability experiment: reporting."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from uci_transportability.config import ExperimentConfig
from uci_transportability.plots import generate_all_plots


def _ensure_dirs(output_dir: Path) -> dict[str, Path]:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    reports_dir = output_dir / "reports"
    for directory in [tables_dir, figures_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "tables": tables_dir,
        "figures": figures_dir,
        "reports": reports_dir,
    }


def save_outputs(*, outputs, config: ExperimentConfig, logger: logging.Logger) -> None:
    directories = _ensure_dirs(config.output_dir)
    outputs.target_site_results.to_csv(
        directories["tables"] / "target_site_results.csv", index=False
    )
    outputs.id_validation_fold_metrics.to_csv(
        directories["tables"] / "id_validation_fold_metrics.csv", index=False
    )
    outputs.prediction_records.to_csv(
        directories["tables"] / "prediction_records.csv", index=False
    )
    outputs.split_manifest.to_csv(
        directories["tables"] / "target_site_split_manifest.csv", index=False
    )
    outputs.split_assignments.to_csv(
        directories["tables"] / "target_site_split_assignments.csv", index=False
    )
    outputs.model_summary.to_csv(
        directories["tables"] / "model_summary.csv", index=False
    )

    generate_all_plots(
        results_df=outputs.target_site_results,
        config=config,
        output_dir=directories["figures"],
    )
    logger.info("Saved tables and figures under %s", config.output_dir)


def _markdown_table(df: pd.DataFrame, *, max_rows: int = 20) -> list[str]:
    if df.empty:
        return ["No rows available."]
    preview = df.head(max_rows).copy()
    preview = preview.fillna("NA")
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


def write_markdown_report(
    *, outputs, config: ExperimentConfig, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_df = outputs.dataset_df
    results_df = outputs.target_site_results
    summary_df = outputs.model_summary
    split_df = outputs.split_manifest

    report_lines: list[str] = [
        "# UCI Heart Disease M0 baseline report",
        "",
        "## Setup",
        f"- Dataset: `{config.dataset_name}`",
        f"- Method: `{config.method_name}`",
        "- Target-site protocol: split target into `train_1` and `test_1` using a stratified 50/50 split.",
        "- Source-site protocol: all non-target sites are used as source domains for initial training.",
        "- ID metric definition: out-of-fold validation on source domains using `LeaveOneGroupOut`.",
        "- OOD metric definition: evaluation on `test_1` from the held-out target site.",
        "",
        "## Dataset overview",
        f"- Total rows: `{len(dataset_df)}`",
        f"- Total target sites: `{dataset_df[config.domain_column].nunique()}`",
        f"- Positive prevalence: `{dataset_df[config.target_column].mean():.4f}`",
        "",
        "## Target split manifest",
        "",
    ]
    report_lines.extend(_markdown_table(split_df))
    report_lines.extend(
        [
            "",
            "## Target-site results",
            "",
        ]
    )
    report_lines.extend(
        _markdown_table(
            results_df[
                [
                    "method",
                    "model_display",
                    "target_site_display",
                    "roc_auc_id",
                    "roc_auc_ood",
                    "gap_roc_auc",
                    "f1_ood",
                    "balanced_accuracy_ood",
                    "ece_ood",
                ]
            ]
        )
    )
    report_lines.extend(
        [
            "",
            "## Model summary",
            "",
        ]
    )
    report_lines.extend(
        _markdown_table(
            summary_df[
                [
                    "method",
                    "model_display",
                    "roc_auc_ood_mean",
                    "f1_ood_mean",
                    "balanced_accuracy_ood_mean",
                    "ece_ood_mean",
                    "gap_roc_auc_mean",
                ]
            ]
        )
    )
    report_lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `tables/target_site_results.csv`: one row per target site, method, and model.",
            "- `tables/id_validation_fold_metrics.csv`: fold-level source validation metrics.",
            "- `tables/prediction_records.csv`: prediction-level outputs for ID and OOD evaluations.",
            "- `tables/target_site_split_manifest.csv`: split sizes and class balance for each target-site setup.",
            "- `tables/target_site_split_assignments.csv`: row-level partition labels for each target-site evaluation.",
            "- `tables/model_summary.csv`: mean performance per model.",
            "- `figures/barplots/`: grouped barplots per metric and model.",
            "- `figures/heatmaps/`: method-by-target heatmaps per metric and model.",
        ]
    )
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
