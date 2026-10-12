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
    fold_metrics_df: pd.DataFrame,
    primary_metric: str,
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
            "n_sites": int(group["target_site"].nunique()),
        }
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = (
                float(values.mean(skipna=True)) if not values.dropna().empty else np.nan
            )
            row[f"{metric}_std"] = (
                float(values.std(skipna=True, ddof=1))
                if values.dropna().shape[0] > 1
                else 0.0
            )
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
    sort_columns = [f"{primary_metric}_mean", "balanced_accuracy_mean", "f1_mean"]
    summary_df = summary_df.sort_values(sort_columns, ascending=False).reset_index(
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
            mean_value = row.get(f"{metric}_mean", np.nan)
            std_value = row.get(f"{metric}_std", np.nan)
            out[metric] = _format_mean_std(mean_value, std_value)
        rows.append(out)
    return pd.DataFrame(rows)


def snapshot_code(output_paths: OutputPaths, main_script_path: Path) -> None:
    code_root = output_paths.code
    package_dir = main_script_path.parent / "transductive_uci"
    shutil.copy2(main_script_path, code_root / main_script_path.name)

    destination_package = code_root / "transductive_uci"
    if destination_package.exists():
        shutil.rmtree(destination_package)
    shutil.copytree(package_dir, destination_package)


def save_reports(
    output_paths: OutputPaths,
    input_csv: Path,
    inferred: InferredColumns,
    dataset_tables: dict[str, pd.DataFrame],
    model_names: list[str],
    summary_df: pd.DataFrame,
    site_ranking_df: pd.DataFrame,
    variant_comparison_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    error_analysis_df: pd.DataFrame,
) -> None:
    md_path = output_paths.report / "transductive_learning_report.md"
    txt_path = output_paths.report / "transductive_learning_report.txt"

    site_distribution = dataset_tables.get("site_distribution", pd.DataFrame())
    best_row = summary_df.iloc[0] if not summary_df.empty else None
    best_site = site_ranking_df.iloc[0] if not site_ranking_df.empty else None
    hardest_site = site_ranking_df.iloc[-1] if not site_ranking_df.empty else None

    lines = [
        "# UCI Heart Disease Cross-Site Transductive Learning Report",
        "",
        "## Dataset",
        f"- Input file: `{input_csv}`",
        f"- Number of samples: {int(dataset_tables['dataset_overview'].loc[dataset_tables['dataset_overview']['field'] == 'n_rows', 'value'].iloc[0])}",
        f"- Number of sites: {int(dataset_tables['dataset_overview'].loc[dataset_tables['dataset_overview']['field'] == 'n_sites', 'value'].iloc[0])}",
        f"- Inferred target column: `{inferred.target_column}`",
        f"- Inferred site/domain column: `{inferred.domain_column}`",
        f"- Inferred subject identifier: `{inferred.subject_id_column or 'synthetic row id'}`",
        f"- Predictive features retained: {len(inferred.feature_columns)}",
        "",
        "## Site Distribution",
    ]

    if not site_distribution.empty:
        for _, row in site_distribution.iterrows():
            lines.append(
                f"- {row['site']}: n={int(row['site_n'])}, positives={int(row['target_positive'])}, positive_rate={float(row['target_rate']):.3f}"
            )

    lines.extend(
        [
            "",
            "## Models Tested",
            *[f"- {model_name}" for model_name in model_names],
            "",
            "## Transductive Method",
            "- Leave-one-site-out evaluation: each site is held out once as the target domain.",
            "- Source-only preprocessing: all imputers, encoders and scalers are fitted only on labeled source sites.",
            "- Unsupervised CORAL alignment: source features are covariance-aligned toward the unlabeled hold-out target site.",
            "- Optional conservative pseudo-labeling: only high-confidence target samples are added, with caps and reduced sample weights.",
            "- Target labels are used only after training/adaptation for final OOD evaluation.",
            "",
            "## Main Results",
        ]
    )

    if best_row is not None:
        lines.extend(
            [
                f"- Best transductive configuration by mean ROC AUC: `{best_row['model_name']}` with `{best_row['variant']}`.",
                f"- Mean ROC AUC: {best_row['roc_auc_mean']:.3f} ± {best_row['roc_auc_std']:.3f}",
                f"- Mean F1: {best_row['f1_mean']:.3f} ± {best_row['f1_std']:.3f}",
                f"- Mean balanced accuracy: {best_row['balanced_accuracy_mean']:.3f} ± {best_row['balanced_accuracy_std']:.3f}",
                f"- Mean ECE: {best_row['ece_mean']:.3f} ± {best_row['ece_std']:.3f}",
            ]
        )

    if not variant_comparison_df.empty:
        delta_auc = pd.to_numeric(
            variant_comparison_df["delta_roc_auc"], errors="coerce"
        )
        delta_f1 = pd.to_numeric(variant_comparison_df["delta_f1"], errors="coerce")
        lines.extend(
            [
                f"- Pseudo-labeling vs alignment-only mean delta ROC AUC: {delta_auc.mean(skipna=True):+.3f}",
                f"- Pseudo-labeling vs alignment-only mean delta F1: {delta_f1.mean(skipna=True):+.3f}",
            ]
        )

    lines.extend(["", "## Site-Level Interpretation"])
    if best_site is not None:
        lines.append(
            f"- Easiest site on average: `{best_site['target_site']}` with mean {best_site['mean_score']:.3f} on the ranking metric."
        )
    if hardest_site is not None:
        lines.append(
            f"- Hardest site on average: `{hardest_site['target_site']}` with mean {hardest_site['mean_score']:.3f} on the ranking metric."
        )

    if not distance_df.empty:
        strongest_shift = distance_df.sort_values(
            "pre_mean_abs_smd", ascending=False
        ).iloc[0]
        best_alignment = distance_df.sort_values(
            "delta_mean_abs_smd", ascending=True
        ).iloc[0]
        lines.extend(
            [
                f"- Largest pre-alignment distribution shift: `{strongest_shift['target_site']}` (mean abs SMD={strongest_shift['pre_mean_abs_smd']:.3f}).",
                f"- Strongest alignment-induced reduction in mean abs SMD: `{best_alignment['target_site']}` ({best_alignment['delta_mean_abs_smd']:.3f}).",
            ]
        )

    if not error_analysis_df.empty:
        worst_errors = error_analysis_df.sort_values(
            "error_rate", ascending=False
        ).iloc[0]
        lines.append(
            f"- Highest error rate among evaluated configurations: `{worst_errors['target_site']}` with `{worst_errors['model_name']}` / `{worst_errors['variant']}` (error_rate={worst_errors['error_rate']:.3f})."
        )

    lines.extend(
        [
            "",
            "## Interpretation for Transportability",
            "- Strong heterogeneity across sites indicates that the learned decision rule is not equally transportable everywhere.",
            "- CORAL helps by reducing marginal feature mismatch between source and target without touching target labels.",
            "- Pseudo-labeling is deliberately conservative; when it helps, it suggests the target manifold contains exploitable structure even without labels.",
            "- When a site remains difficult after adaptation, the result supports a discussion of residual lack of transportability driven by covariate shift, calibration drift or site-specific case-mix.",
        ]
    )

    text = "\n".join(lines) + "\n"
    md_path.write_text(text, encoding="utf-8")
    txt_path.write_text(text, encoding="utf-8")


def _format_mean_std(mean_value: float, std_value: float) -> str:
    if pd.isna(mean_value):
        return "NA"
    std_value = 0.0 if pd.isna(std_value) else float(std_value)
    return f"{float(mean_value):.3f} ± {std_value:.3f}"
