"""Transportability pipeline module: reporting."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from tcga_transportability.config import ExperimentConfig
from tcga_transportability.plots import generate_all_plots


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
    outputs.m2_pairwise_weighting.to_csv(
        directories["tables"] / "m2_pairwise_source_weighting.csv", index=False
    )

    generate_all_plots(
        results_df=outputs.target_site_results,
        config=config,
        output_dir=directories["figures"],
    )
    logger.info("Saved tables and figures under %s", config.output_dir)


def _markdown_table(df: pd.DataFrame, *, max_rows: int = 200) -> list[str]:
    if df.empty:
        return ["No rows available."]
    preview = df.head(max_rows).copy().fillna("NA")
    headers = [str(column) for column in preview.columns]
    rows = [[str(value) for value in row] for row in preview.to_numpy()]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(headers))
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
        "# TCGA-BRCA cross-site transportability report",
        "",
        "## Setup",
        f"- Dataset: `{config.dataset_name}`",
        f"- Enabled methods: `{', '.join(config.enabled_methods)}`",
        "- Target-site protocol: split target into `train_1` and `test_1` using a stratified 50/50 split.",
        "- Source-site protocol: all non-target sites are used as source domains for initial training.",
        "- ID metric definition: grouped validation on source domains using up to 3 source-domain folds.",
        f"- Operating threshold: selected on out-of-fold source-domain ID predictions by maximizing balanced accuracy (tie-break: F1), starting from reference threshold `{config.threshold:.3f}`, and then applied unchanged to `test_1`.",
        "- OOD metric definition: evaluation on `test_1` from the held-out target site.",
        "- TCGA train-only preprocessing: near-zero variance gene removal, top-variance gene prefiltering, and ANOVA gene selection are fit only on source training partitions.",
        "- Domain definition: `site` (TCGA TSS code).",
        "- Target definition: binary `target` where 0 = `Stage I-II` and 1 = `Stage III-IV`.",
        "",
        "## Dataset overview",
        f"- Total rows after site filtering: `{len(dataset_df)}`",
        f"- Total target sites: `{dataset_df[config.domain_column].nunique()}`",
        f"- Positive prevalence: `{dataset_df[config.target_column].mean():.4f}`",
        f"- Gene features available: `{len(outputs.schema['gene_features'])}`",
        "",
        "## Target split manifest",
        "",
    ]
    if "M1" in config.enabled_methods:
        report_lines.insert(
            10,
            f"- M1 ComBat protocol: train-only harmonization of a source-only top-variance gene pool (`{config.combat_top_k_variance_genes}` genes max) with `neuroCombat`, using the largest source batch as reference and the source `target` label as a biological covariate; target `test_1` remains untouched.",
        )
    if "M2" in config.enabled_methods:
        report_lines.insert(
            11,
            "- M2 weighting protocol: source-only DASHI-like pairwise transferability weights estimated among source domains; neither `train_1` nor `test_1` from the target site is used to compute weights.",
        )
    if "M3" in config.enabled_methods:
        report_lines.insert(
            12,
            f"- M3 transductive protocol: source-trained model adapted with `train_1` from the target site through a same-model probability blend, using target blend weight `{config.m3_target_blend_weight:.3f}`, then evaluated on `test_1`.",
        )
    if "M4" in config.enabled_methods:
        report_lines.insert(
            13,
            f"- M4 ComBat + transductive protocol: source domains are first harmonized with train-only `neuroCombat`, then the source-trained model is adapted with target `train_1` through the same probability blend (weight `{config.m3_target_blend_weight:.3f}`), and finally evaluated on raw target `test_1`.",
        )
    report_lines.extend(_markdown_table(split_df))
    report_lines.extend(["", "## Target-site results", ""])
    results_report_df = results_df.copy()
    for column in [
        "method",
        "model_display",
        "target_site_display",
        "decision_threshold",
        "roc_auc_id",
        "roc_auc_ood",
        "gap_roc_auc",
        "f1_ood",
        "balanced_accuracy_ood",
        "ece_ood",
        "n_selected_genes",
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
                    "roc_auc_id",
                    "roc_auc_ood",
                    "gap_roc_auc",
                    "f1_ood",
                    "balanced_accuracy_ood",
                    "ece_ood",
                    "n_selected_genes",
                ]
            ]
        )
    )
    report_lines.extend(["", "## Model summary", ""])
    summary_report_df = summary_df.copy()
    for column in [
        "method",
        "model_display",
        "decision_threshold_mean",
        "roc_auc_ood_mean",
        "f1_ood_mean",
        "balanced_accuracy_ood_mean",
        "ece_ood_mean",
        "gap_roc_auc_mean",
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
            "- `tables/m2_pairwise_source_weighting.csv`: source-only pairwise transfer scores used to derive M2 domain weights.",
            "- `figures/barplots/`: grouped barplots per metric and model.",
            "- `figures/heatmaps/`: method-by-target heatmaps per metric and model.",
        ]
    )
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
