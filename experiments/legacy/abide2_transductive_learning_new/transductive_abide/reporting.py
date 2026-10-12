"""Transportability experiment: reporting."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import METRIC_COLUMNS
from .data_utils import InferredColumns


@dataclass
class OutputPaths:
    root: Path
    code: Path
    tables: Path
    plots: Path
    models: Path
    report: Path
    logs: Path
    confusion_matrices: Path


class RunLogger:
    """Minimal logger that prints and keeps a file-ready history."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: str) -> None:
        print(message)
        self.lines.append(message)

    def save(self, output_path: Path) -> None:
        output_path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def ensure_output_structure(output_dir: Path) -> OutputPaths:
    code_dir = output_dir / "code"
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    models_dir = output_dir / "models"
    report_dir = output_dir / "report"
    logs_dir = output_dir / "logs"
    confusion_dir = plots_dir / "confusion_matrices"

    for path in [
        output_dir,
        code_dir,
        tables_dir,
        plots_dir,
        models_dir,
        report_dir,
        logs_dir,
        confusion_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return OutputPaths(
        root=output_dir,
        code=code_dir,
        tables=tables_dir,
        plots=plots_dir,
        models=models_dir,
        report=report_dir,
        logs=logs_dir,
        confusion_matrices=confusion_dir,
    )


def save_config(config_dict: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")


def build_summary_table(
    fold_metrics_df: pd.DataFrame, primary_metric: str
) -> pd.DataFrame:
    if fold_metrics_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (model_name, variant), group in fold_metrics_df.groupby(
        ["model_name", "variant"], dropna=False
    ):
        row: dict[str, Any] = {
            "model_name": model_name,
            "variant": variant,
            "n_domains_total": int(group["target_domain"].nunique()),
        }
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = (
                float(values.mean(skipna=True)) if values.notna().any() else np.nan
            )
            row[f"{metric}_std"] = (
                float(values.std(skipna=True, ddof=1))
                if values.notna().sum() > 1
                else 0.0
            )
            row[f"{metric}_valid_domains"] = int(values.notna().sum())
        row["pseudo_selected_mean"] = float(
            pd.to_numeric(group["pseudo_selected_count"], errors="coerce").mean(
                skipna=True
            )
        )
        row["pseudo_selected_fraction_mean"] = float(
            pd.to_numeric(group["pseudo_selected_fraction"], errors="coerce").mean(
                skipna=True
            )
        )
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    sort_cols = [f"{primary_metric}_mean", "f1_macro_mean", "roc_auc_macro_mean"]
    summary_df = summary_df.sort_values(sort_cols, ascending=False).reset_index(
        drop=True
    )
    return summary_df


def build_thesis_ready_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        out = {
            "model_name": row["model_name"],
            "variant": row["variant"],
        }
        for metric in METRIC_COLUMNS:
            out[metric] = format_mean_std(
                row.get(f"{metric}_mean", np.nan), row.get(f"{metric}_std", np.nan)
            )
        rows.append(out)
    return pd.DataFrame(rows)


def snapshot_code(output_paths: OutputPaths, main_script_path: Path) -> None:
    code_root = output_paths.code
    package_dir = main_script_path.parent / "transductive_abide"
    shutil.copy2(main_script_path, code_root / main_script_path.name)

    destination_package = code_root / "transductive_abide"
    if destination_package.exists():
        shutil.rmtree(destination_package)
    shutil.copytree(package_dir, destination_package)


def save_reports(
    output_paths: OutputPaths,
    dataset_file: Path,
    inferred: InferredColumns,
    dataset_tables: dict[str, pd.DataFrame],
    model_names: list[str],
    summary_df: pd.DataFrame,
    site_ranking_df: pd.DataFrame,
    variant_comparison_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    error_analysis_df: pd.DataFrame,
) -> None:
    md_path = output_paths.report / "abide2_transductive_learning_report.md"
    txt_path = output_paths.report / "abide2_transductive_learning_report.txt"

    overview_df = dataset_tables["dataset_overview"]
    domain_distribution = dataset_tables.get("domain_distribution", pd.DataFrame())
    best_row = summary_df.iloc[0] if not summary_df.empty else None
    easiest_row = site_ranking_df.iloc[0] if not site_ranking_df.empty else None
    hardest_row = site_ranking_df.iloc[-1] if not site_ranking_df.empty else None

    lines = [
        "# ABIDE II Cross-Site Transductive Learning Report",
        "",
        "## Dataset",
        f"- Input dataset file: `{dataset_file}`",
        f"- Number of samples: {int(overview_df.loc[overview_df['field'] == 'n_rows', 'value'].iloc[0])}",
        f"- Number of columns: {int(overview_df.loc[overview_df['field'] == 'n_columns', 'value'].iloc[0])}",
        f"- Number of retained predictive features before fold-wise reduction: {int(overview_df.loc[overview_df['field'] == 'n_features', 'value'].iloc[0])}",
        f"- Inferred target column: `{inferred.target_column}`",
        f"- Inferred domain column: `{inferred.domain_column}`",
        f"- Inferred subject identifier: `{inferred.subject_id_column or 'synthetic id'}`",
        f"- Used artificial domains: `{bool(inferred.used_artificial_domains)}`",
        "",
        "## Domains",
    ]

    if not domain_distribution.empty:
        for _, row in domain_distribution.iterrows():
            lines.append(f"- {row['domain']}: n={int(row['domain_n'])}")

    lines.extend(
        [
            "",
            "## Models Tested",
            *[f"- {model_name}" for model_name in model_names],
            "",
            "## Transductive Pipeline",
            "- Leave-one-domain-out evaluation: one ABIDE II acquisition site is held out at a time.",
            "- Source-only preprocessing: median imputation, variance filtering, scaling and optional PCA are fit only on source sites.",
            "- Unsupervised adaptation: CORAL aligns source covariance to the unlabeled hold-out target domain.",
            "- Optional conservative pseudo-labeling: only very high-confidence target predictions are added with caps and reduced weights.",
            "- Target labels are never used during fitting or adaptation and are used only for final OOD evaluation.",
            "",
            "## Main Results",
        ]
    )

    if best_row is not None:
        lines.extend(
            [
                f"- Best configuration by mean balanced accuracy: `{best_row['model_name']}` with `{best_row['variant']}`.",
                f"- Mean balanced accuracy: {best_row['balanced_accuracy_mean']:.3f} ± {best_row['balanced_accuracy_std']:.3f}",
                f"- Mean macro F1: {best_row['f1_macro_mean']:.3f} ± {best_row['f1_macro_std']:.3f}",
                f"- Mean macro ROC AUC: {best_row['roc_auc_macro_mean']:.3f} ± {best_row['roc_auc_macro_std']:.3f}",
                f"- Mean macro PR AUC: {best_row['pr_auc_macro_mean']:.3f} ± {best_row['pr_auc_macro_std']:.3f}",
                f"- Mean ECE: {best_row['ece_mean']:.3f} ± {best_row['ece_std']:.3f}",
            ]
        )

    if not variant_comparison_df.empty:
        delta_bal = pd.to_numeric(
            variant_comparison_df["delta_balanced_accuracy"], errors="coerce"
        )
        delta_f1 = pd.to_numeric(
            variant_comparison_df["delta_f1_macro"], errors="coerce"
        )
        lines.extend(
            [
                f"- Mean pseudo-labeling delta on balanced accuracy: {delta_bal.mean(skipna=True):+.3f}",
                f"- Mean pseudo-labeling delta on macro F1: {delta_f1.mean(skipna=True):+.3f}",
            ]
        )

    lines.extend(["", "## Domain-Level Interpretation"])
    if easiest_row is not None:
        lines.append(
            f"- Easiest domain on the ranking metric: `{easiest_row['target_domain']}` ({easiest_row['mean_score']:.3f})."
        )
    if hardest_row is not None:
        lines.append(
            f"- Hardest domain on the ranking metric: `{hardest_row['target_domain']}` ({hardest_row['mean_score']:.3f})."
        )

    if not distance_df.empty:
        largest_shift = distance_df.sort_values(
            "pre_mean_abs_smd", ascending=False
        ).iloc[0]
        best_shift_reduction = distance_df.sort_values(
            "delta_mean_abs_smd", ascending=True
        ).iloc[0]
        lines.extend(
            [
                f"- Strongest pre-alignment shift: `{largest_shift['target_domain']}` (mean abs SMD={largest_shift['pre_mean_abs_smd']:.3f}).",
                f"- Largest CORAL reduction in mean abs SMD: `{best_shift_reduction['target_domain']}` ({best_shift_reduction['delta_mean_abs_smd']:.3f}).",
            ]
        )

    if not error_analysis_df.empty:
        worst_error = error_analysis_df.iloc[0]
        lines.append(
            f"- Highest error rate among evaluated configurations: `{worst_error['target_domain']}` with `{worst_error['model_name']}` / `{worst_error['variant']}` (error_rate={worst_error['error_rate']:.3f})."
        )

    lines.extend(
        [
            "",
            "## Interpretation for Transportability",
            "- Domain-level variability indicates that ABIDE II site effects remain relevant even after standard preprocessing.",
            "- CORAL addresses unsupervised marginal domain shift in the reduced feature space without using target labels.",
            "- Conservative pseudo-labeling helps only when the target manifold is sufficiently confident; otherwise the method correctly stays close to alignment-only.",
            "- Persistent performance gaps across sites support a thesis discussion centered on limited transportability under multisite clinical and acquisition heterogeneity.",
        ]
    )

    text = "\n".join(lines) + "\n"
    md_path.write_text(text, encoding="utf-8")
    txt_path.write_text(text, encoding="utf-8")


def format_mean_std(mean_value: float, std_value: float) -> str:
    if pd.isna(mean_value):
        return "NA"
    std_value = 0.0 if pd.isna(std_value) else float(std_value)
    return f"{float(mean_value):.3f} ± {std_value:.3f}"
