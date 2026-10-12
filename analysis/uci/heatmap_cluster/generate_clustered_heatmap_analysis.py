"""Post-experiment analysis: generate clustered heatmap analysis."""

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
from thesis_tools.paths import configured_path as _configured_path


import argparse
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import zscore

import matplotlib

os.environ["MPLCONFIGDIR"] = str(
    Path(tempfile.gettempdir()) / "codex_cluster_heatmap_mpl"
)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_ORDER = ["M0", "M1", "M2", "M3", "M4"]
METHOD_FULL_LABELS = {
    "M0": "M0: Baseline",
    "M1": "M1: ComBat",
    "M2": "M2: Weighting",
    "M3": "M3: Transductive",
    "M4": "M4: ComBat + Transductive",
}
METHOD_COLORS = {
    "M0": "#4C78A8",
    "M1": "#F58518",
    "M2": "#54A24B",
    "M3": "#E45756",
    "M4": "#72B7B2",
}
METHOD_LEGEND_TEXT = "M0: Baseline | M1: ComBat | M2: Weighting | M3: Transductive | M4: ComBat + Transductive"
PERFORMANCE_METRICS = {
    "roc_auc": {
        "ood": "roc_auc_ood",
        "id": "roc_auc_id",
        "gap": "gap_roc_auc",
        "label": "ROC-AUC",
    },
    "f1": {
        "ood": "f1_ood",
        "id": "f1_id",
        "gap": "gap_f1",
        "label": "F1-score",
    },
    "balanced_accuracy": {
        "ood": "balanced_accuracy_ood",
        "id": "balanced_accuracy_id",
        "gap": "gap_balanced_accuracy",
        "label": "Balanced Accuracy",
    },
}
DISTANCE_COLUMNS = [
    "euclidean_centroid_distance",
    "mahalanobis_distance",
    "wasserstein_mean_distance",
    "jensen_shannon_mean_divergence",
    "mmd_rbf",
]
BUNDLED_EXCEL_PYTHON = Path(_configured_path("python_executable"))


@dataclass(frozen=True)
class OutputCandidate:
    output_dir: Path
    config_snapshot_path: Path
    config_snapshot: dict[str, Any]
    results_summary_path: Path
    results_bootstrap_raw_path: Path | None
    is_bootstrap: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clustered heatmap analysis from existing cross-site outputs."
    )
    parser.add_argument(
        "--input-root", type=Path, required=True, help="Dataset root, e.g. .../UCI"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where clustered heatmap outputs will be written",
    )
    return parser.parse_args()


def apply_plot_style() -> None:
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


def slugify(text: str) -> str:
    output = []
    for char in text.lower():
        if char.isalnum():
            output.append(char)
        else:
            if not output or output[-1] != "_":
                output.append("_")
    return "".join(output).strip("_")


def write_single_sheet_excel(path: Path, sheet_name: str, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_excel_with_fallback(
        df,
        path,
        sheet_name=sheet_name[:31],
        index=True if df.index.name is not None else False,
    )


def write_excel_with_fallback(
    df: pd.DataFrame, path: Path, *, sheet_name: str, index: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(path) as writer:
            df.to_excel(writer, sheet_name=sheet_name[:31], index=index)
        return
    except ModuleNotFoundError:
        if not BUNDLED_EXCEL_PYTHON.exists():
            raise RuntimeError(
                "Could not write Excel output because openpyxl is unavailable and bundled runtime was not found."
            )

    temp_dir = Path(tempfile.mkdtemp(prefix="cluster_heatmap_xlsx_"))
    temp_csv = temp_dir / "data.csv"
    df.to_csv(temp_csv, index=index)
    helper_code = """
import sys
import pandas as pd

csv_path = sys.argv[1]
xlsx_path = sys.argv[2]
sheet_name = sys.argv[3]
use_index = sys.argv[4] == '1'

if use_index:
    df = pd.read_csv(csv_path, index_col=0)
else:
    df = pd.read_csv(csv_path)

with pd.ExcelWriter(xlsx_path) as writer:
    df.to_excel(writer, sheet_name=sheet_name[:31], index=use_index)
"""
    subprocess.run(
        [
            str(BUNDLED_EXCEL_PYTHON),
            "-c",
            helper_code,
            str(temp_csv),
            str(path),
            sheet_name[:31],
            "1" if index else "0",
        ],
        check=True,
    )


def discover_output_candidates(input_root: Path) -> list[OutputCandidate]:
    methods_root = input_root / "Methods"
    candidates: list[OutputCandidate] = []
    for config_snapshot_path in sorted(methods_root.glob("*/config_snapshot.json")):
        output_dir = config_snapshot_path.parent
        results_summary_path = output_dir / "tables" / "target_site_results.csv"
        if not results_summary_path.exists():
            continue
        config_snapshot = json.loads(config_snapshot_path.read_text(encoding="utf-8"))
        results_bootstrap_raw_path = (
            output_dir / "tables" / "target_site_results_bootstrap_raw.csv"
        )
        candidates.append(
            OutputCandidate(
                output_dir=output_dir,
                config_snapshot_path=config_snapshot_path,
                config_snapshot=config_snapshot,
                results_summary_path=results_summary_path,
                results_bootstrap_raw_path=results_bootstrap_raw_path
                if results_bootstrap_raw_path.exists()
                else None,
                is_bootstrap=results_bootstrap_raw_path.exists()
                or "bootstrap" in output_dir.name.lower(),
            )
        )
    return candidates


def candidate_score(candidate: OutputCandidate) -> int:
    preview_df = pd.read_csv(candidate.results_summary_path, low_memory=False)
    method_count = (
        preview_df["method"].nunique() if "method" in preview_df.columns else 0
    )
    model_count = preview_df["model"].nunique() if "model" in preview_df.columns else 0
    target_count = (
        preview_df["target_site"].nunique()
        if "target_site" in preview_df.columns
        else 0
    )
    std_bonus = sum(1 for column in preview_df.columns if column.endswith("_std"))
    return method_count * 10_000 + model_count * 100 + target_count * 10 + std_bonus


def choose_candidate(
    candidates: list[OutputCandidate], *, want_bootstrap: bool
) -> OutputCandidate | None:
    filtered = [
        candidate
        for candidate in candidates
        if candidate.is_bootstrap == want_bootstrap
    ]
    if not filtered:
        return None
    return max(filtered, key=candidate_score)


def load_performance_table(
    input_root: Path, warnings: list[str]
) -> tuple[pd.DataFrame, str, list[str], list[str]]:
    joined_path = input_root / "Site_distance" / "site_distance_vs_performance.csv"
    found_inputs: list[str] = []
    missing_inputs: list[str] = []

    if joined_path.exists():
        found_inputs.append(str(joined_path))
        df = pd.read_csv(joined_path, low_memory=False)
        if "dataset" not in df.columns:
            df["dataset"] = input_root.name
        required_columns = [
            "dataset",
            "target_site",
            "model",
            "metric",
            "method",
            "mean_ood_performance",
            "std_ood_performance",
            "mean_id_performance",
            "id_ood_gap",
            "mean_ece",
        ]
        optional = [
            column
            for column in ["target_site_display", "model_display"]
            if column in df.columns
        ]
        missing_required = [
            column for column in required_columns if column not in df.columns
        ]
        if missing_required:
            warnings.append(
                "site_distance_vs_performance.csv found but missing required columns: "
                + ", ".join(missing_required)
                + ". Falling back to transportability outputs."
            )
        else:
            return (
                df[required_columns + optional].copy(),
                df["dataset"].iloc[0],
                found_inputs,
                missing_inputs,
            )

    candidates = discover_output_candidates(input_root)
    bootstrap_candidate = choose_candidate(candidates, want_bootstrap=True)
    standard_candidate = choose_candidate(candidates, want_bootstrap=False)

    if bootstrap_candidate is not None:
        found_inputs.append(str(bootstrap_candidate.results_summary_path))
    else:
        missing_inputs.append("bootstrap target_site_results.csv")
    if standard_candidate is not None:
        found_inputs.append(str(standard_candidate.results_summary_path))
    else:
        missing_inputs.append("standard target_site_results.csv")

    base_candidate = bootstrap_candidate or standard_candidate
    if base_candidate is None:
        raise RuntimeError(
            "Could not find any transportability output table to build clustered heatmaps."
        )

    result_df = pd.read_csv(base_candidate.results_summary_path, low_memory=False)
    dataset_name = str(
        base_candidate.config_snapshot.get("dataset_name", input_root.name)
    )

    long_rows: list[dict[str, Any]] = []
    for _, row in result_df.iterrows():
        for metric_key, metric_config in PERFORMANCE_METRICS.items():
            ood_column = metric_config["ood"]
            id_column = metric_config["id"]
            gap_column = metric_config["gap"]
            if ood_column not in result_df.columns:
                continue
            mean_ood = float(row[ood_column])
            mean_id = (
                float(row[id_column]) if id_column in result_df.columns else math.nan
            )
            gap_value = (
                float(row[gap_column])
                if gap_column in result_df.columns
                else mean_id - mean_ood
            )
            std_column = f"{ood_column}_std"
            std_ood = (
                float(row[std_column]) if std_column in result_df.columns else math.nan
            )
            long_rows.append(
                {
                    "dataset": dataset_name,
                    "target_site": row["target_site"],
                    "target_site_display": row.get(
                        "target_site_display", row["target_site"]
                    ),
                    "model": row["model"],
                    "model_display": row.get("model_display", row["model"]),
                    "metric": metric_key,
                    "method": row["method"],
                    "mean_ood_performance": mean_ood,
                    "std_ood_performance": std_ood,
                    "mean_id_performance": mean_id,
                    "id_ood_gap": gap_value,
                    "mean_ece": float(row["ece_ood"])
                    if "ece_ood" in result_df.columns
                    else math.nan,
                }
            )
    return pd.DataFrame(long_rows), dataset_name, found_inputs, missing_inputs


def load_optional_table(
    path: Path, *, required_columns: list[str], warnings: list[str]
) -> pd.DataFrame | None:
    if not path.exists():
        warnings.append(f"Missing optional input: {path}")
        return None
    df = pd.read_csv(path, low_memory=False)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        warnings.append(
            f"Optional input {path.name} is missing columns: {', '.join(missing)}"
        )
        return None
    return df


def build_meta_feature_table(
    performance_df: pd.DataFrame,
    sample_size_df: pd.DataFrame | None,
    entropy_df: pd.DataFrame | None,
    site_distance_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[str]]:
    missing_notes: list[str] = []

    site_labels = (
        performance_df[["target_site", "target_site_display"]]
        .drop_duplicates()
        .assign(
            target_site_display=lambda df: df["target_site_display"].fillna(
                df["target_site"]
            )
        )
    )

    meta_df = site_labels.copy()

    if sample_size_df is not None:
        sample_features = sample_size_df.rename(
            columns={
                "Target site": "target_site",
                "N total target": "n_total_target",
                "N source": "n_source",
                "N train_1": "n_train_1",
                "N test_1": "n_test_1",
            }
        )
        sample_columns = [
            column
            for column in [
                "target_site",
                "n_total_target",
                "n_source",
                "n_train_1",
                "n_test_1",
            ]
            if column in sample_features.columns
        ]
        meta_df = meta_df.merge(
            sample_features[sample_columns], on="target_site", how="left"
        )
    else:
        missing_notes.append("sample size tables not available")

    if entropy_df is not None:
        entropy_columns = [
            column
            for column in [
                "target_site",
                "full_target_entropy",
                "full_target_normalized_entropy",
                "mean_train_1_entropy",
                "std_train_1_entropy",
                "mean_test_1_entropy",
                "std_test_1_entropy",
                "mean_source_entropy",
                "std_source_entropy",
            ]
            if column in entropy_df.columns
        ]
        meta_df = meta_df.merge(
            entropy_df[entropy_columns], on="target_site", how="left"
        )
    else:
        missing_notes.append("entropy summary not available")

    if site_distance_df is not None:
        distance_columns = [
            column
            for column in [
                "target_site",
                "euclidean_centroid_distance",
                "mahalanobis_distance",
                "wasserstein_mean_distance",
                "jensen_shannon_mean_divergence",
                "mmd_rbf",
            ]
            if column in site_distance_df.columns
        ]
        meta_df = meta_df.merge(
            site_distance_df[distance_columns], on="target_site", how="left"
        )
    else:
        missing_notes.append("site distance table not available")

    performance_agg = performance_df.groupby(
        ["target_site", "target_site_display"], as_index=False
    )[["mean_ood_performance", "std_ood_performance", "id_ood_gap", "mean_ece"]].mean(
        numeric_only=True
    )
    meta_df = meta_df.merge(
        performance_agg,
        on=["target_site", "target_site_display"],
        how="left",
    )
    return meta_df, missing_notes


def prepare_meta_feature_matrices(
    meta_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, float]]:
    original_df = meta_df.copy()
    numeric_columns = [
        column
        for column in original_df.columns
        if column not in {"target_site", "target_site_display"}
    ]
    numeric_df = original_df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    removed_all_nan = [
        column for column in numeric_columns if numeric_df[column].isna().all()
    ]
    numeric_df = numeric_df.drop(columns=removed_all_nan)
    numeric_columns = numeric_df.columns.tolist()

    imputed_columns: dict[str, float] = {}
    for column in numeric_columns:
        if numeric_df[column].isna().any():
            median_value = float(numeric_df[column].median())
            imputed_columns[column] = median_value
            numeric_df[column] = numeric_df[column].fillna(median_value)

    zscore_df = numeric_df.apply(
        lambda column: (
            pd.Series(zscore(column, nan_policy="omit"), index=column.index)
            if column.nunique(dropna=True) > 1
            else pd.Series(np.zeros(len(column)), index=column.index)
        ),
        axis=0,
    )
    zscore_df.index = original_df["target_site_display"]
    original_numeric_df = numeric_df.copy()
    original_numeric_df.index = original_df["target_site_display"]
    return original_numeric_df, zscore_df, removed_all_nan, imputed_columns


def ordered_methods(columns: list[str]) -> list[str]:
    return [method for method in METHOD_ORDER if method in columns] + [
        column for column in columns if column not in METHOD_ORDER
    ]


def matrix_dimensions(data: pd.DataFrame) -> tuple[float, float]:
    width = max(8.0, 1.5 * data.shape[1] + 4.0)
    height = max(6.0, 0.65 * data.shape[0] + 4.0)
    return width, height


def save_dataframe(
    data: pd.DataFrame,
    csv_path: Path,
    xlsx_path: Path,
    *,
    index: bool = True,
    sheet_name: str = "data",
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=index)
    write_excel_with_fallback(data, xlsx_path, sheet_name=sheet_name[:31], index=index)


def save_simple_heatmap(
    data: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    cmap: str,
    center: float | None = None,
    value_label: str,
    x_label: str = "Method",
    y_label: str = "Target site",
    xtick_rotation: float = 0.0,
    bottom_legend_text: str | None = None,
) -> None:
    width, height = matrix_dimensions(data)
    fig, ax = plt.subplots(figsize=(width, height))
    values = data.to_numpy(dtype=float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if center is not None:
        span = max(abs(vmin - center), abs(vmax - center))
        vmin = center - span
        vmax = center + span
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(
        data.columns.tolist(),
        rotation=xtick_rotation,
        ha="right" if xtick_rotation else "center",
    )
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index.tolist(), rotation=0)
    for row_index in range(data.shape[0]):
        for col_index in range(data.shape[1]):
            value = values[row_index, col_index]
            if np.isnan(value):
                continue
            ax.text(
                col_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#111111",
            )
    cbar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(value_label)
    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if bottom_legend_text:
        fig.text(0.5, 0.02, bottom_legend_text, ha="center", va="bottom", fontsize=10)
        fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    else:
        fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def clustered_order(data: pd.DataFrame, axis: int) -> list[int]:
    values = data.to_numpy(dtype=float)
    matrix = values if axis == 0 else values.T
    if matrix.shape[0] < 2:
        return list(range(matrix.shape[0]))
    linkage_matrix = linkage(matrix, method="average", metric="euclidean")
    dendro = dendrogram(linkage_matrix, no_plot=True)
    return list(dendro["leaves"])


def plot_top_dendrogram(
    ax: plt.Axes, linkage_matrix: np.ndarray, *, n_leaves: int
) -> None:
    dendro = dendrogram(linkage_matrix, no_plot=True)
    max_distance = (
        max(max(coords) for coords in dendro["dcoord"]) if dendro["dcoord"] else 1.0
    )
    for xs, ys in zip(dendro["icoord"], dendro["dcoord"], strict=False):
        mapped_x = [((value - 5.0) / 10.0) for value in xs]
        ax.plot(mapped_x, ys, color="black", linewidth=1.35, solid_capstyle="round")
    ax.set_xlim(-0.5, n_leaves - 0.5)
    ax.set_ylim(0.0, max_distance * 1.08)
    ax.axis("off")


def plot_left_dendrogram(
    ax: plt.Axes, linkage_matrix: np.ndarray, *, n_leaves: int
) -> None:
    dendro = dendrogram(linkage_matrix, no_plot=True)
    max_distance = (
        max(max(coords) for coords in dendro["dcoord"]) if dendro["dcoord"] else 1.0
    )
    for ys, xs in zip(dendro["icoord"], dendro["dcoord"], strict=False):
        mapped_y = [((value - 5.0) / 10.0) for value in ys]
        mapped_x = [max_distance - value for value in xs]
        ax.plot(
            mapped_x, mapped_y, color="black", linewidth=1.35, solid_capstyle="round"
        )
    ax.set_xlim(-max_distance * 0.02, max_distance * 1.02)
    ax.set_ylim(n_leaves - 0.5, -0.5)
    ax.axis("off")


def save_cluster_heatmap(
    data: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    cmap: str,
    center: float | None,
    value_label: str,
    row_cluster: bool,
    col_cluster: bool,
    x_label: str = "Method",
    y_label: str = "Target site",
    xtick_rotation: float = 0.0,
    bottom_legend_text: str | None = None,
) -> None:
    if data.shape[0] < 2 or data.shape[1] < 2:
        save_simple_heatmap(
            data,
            output_path,
            title=title,
            cmap=cmap,
            center=center,
            value_label=value_label,
            x_label=x_label,
            y_label=y_label,
            xtick_rotation=xtick_rotation,
            bottom_legend_text=bottom_legend_text,
        )
        return

    row_order = (
        clustered_order(data, axis=0) if row_cluster else list(range(data.shape[0]))
    )
    col_order = (
        clustered_order(data, axis=1) if col_cluster else list(range(data.shape[1]))
    )
    ordered = data.iloc[row_order, col_order]

    width, height = matrix_dimensions(ordered)
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.2, 0.85, 5.6, 0.32],
        height_ratios=[1.05, 5.0],
        wspace=0.02,
        hspace=0.06,
    )
    ax_empty_left = fig.add_subplot(grid[0, 0])
    ax_empty_mid = fig.add_subplot(grid[0, 1])
    ax_empty_cbar = fig.add_subplot(grid[0, 3])
    ax_spacer = fig.add_subplot(grid[1, 1])
    ax_heatmap = fig.add_subplot(grid[1, 2])
    ax_col_dendrogram = fig.add_subplot(grid[0, 2], sharex=ax_heatmap)
    ax_row_dendrogram = fig.add_subplot(grid[1, 0], sharey=ax_heatmap)
    ax_cbar = fig.add_subplot(grid[1, 3])
    ax_empty_left.axis("off")
    ax_empty_mid.axis("off")
    ax_empty_cbar.axis("off")
    ax_spacer.axis("off")

    if col_cluster and ordered.shape[1] >= 2:
        linkage_matrix = linkage(
            data.to_numpy(dtype=float).T, method="average", metric="euclidean"
        )
        plot_top_dendrogram(
            ax_col_dendrogram, linkage_matrix, n_leaves=ordered.shape[1]
        )
    else:
        ax_col_dendrogram.axis("off")

    if row_cluster and ordered.shape[0] >= 2:
        linkage_matrix = linkage(
            data.to_numpy(dtype=float), method="average", metric="euclidean"
        )
        plot_left_dendrogram(
            ax_row_dendrogram, linkage_matrix, n_leaves=ordered.shape[0]
        )
    else:
        ax_row_dendrogram.axis("off")

    values = ordered.to_numpy(dtype=float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if center is not None:
        span = max(abs(vmin - center), abs(vmax - center))
        vmin = center - span
        vmax = center + span

    image = ax_heatmap.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_heatmap.set_xticks(np.arange(ordered.shape[1]))
    ax_heatmap.set_xticklabels(
        ordered.columns.tolist(),
        rotation=xtick_rotation,
        ha="right" if xtick_rotation else "center",
    )
    ax_heatmap.set_yticks(np.arange(ordered.shape[0]))
    ax_heatmap.set_yticklabels(ordered.index.tolist(), rotation=0)
    ax_heatmap.tick_params(axis="y", pad=2)
    ax_heatmap.set_xlabel(x_label)
    ax_heatmap.set_ylabel(y_label)
    for row_index in range(ordered.shape[0]):
        for col_index in range(ordered.shape[1]):
            value = values[row_index, col_index]
            if np.isnan(value):
                continue
            ax_heatmap.text(
                col_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#111111",
            )
    ax_col_dendrogram.set_xlim(ax_heatmap.get_xlim())
    ax_row_dendrogram.set_ylim(ax_heatmap.get_ylim())
    cbar = fig.colorbar(image, cax=ax_cbar)
    cbar.set_label(value_label)
    fig.suptitle(title, fontsize=16, y=0.98)
    bottom_margin = 0.12 if bottom_legend_text else 0.08
    fig.subplots_adjust(
        left=0.08, right=0.94, top=0.90, bottom=bottom_margin, wspace=0.10, hspace=0.06
    )
    if bottom_legend_text:
        fig.text(0.5, 0.02, bottom_legend_text, ha="center", va="bottom", fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_matrix(
    performance_df: pd.DataFrame,
    *,
    model: str,
    metric: str,
    value_column: str,
) -> pd.DataFrame | None:
    subset = performance_df[
        (performance_df["model"] == model) & (performance_df["metric"] == metric)
    ].copy()
    if subset.empty or value_column not in subset.columns:
        return None
    subset["target_site_display"] = subset.get(
        "target_site_display", subset["target_site"]
    ).fillna(subset["target_site"])
    matrix = subset.pivot_table(
        index="target_site_display",
        columns="method",
        values=value_column,
        aggfunc="mean",
    )
    matrix = matrix.reindex(columns=ordered_methods(matrix.columns.tolist()))
    return matrix.sort_index()


def generate_performance_heatmaps(
    performance_df: pd.DataFrame,
    output_dir: Path,
    warnings: list[str],
) -> None:
    figures_dir = output_dir / "figures"
    matrices_dir = output_dir / "matrices"
    figures_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)

    available_models = sorted(performance_df["model"].dropna().unique().tolist())
    for model in available_models:
        model_display = (
            performance_df.loc[performance_df["model"] == model, "model_display"]
            .dropna()
            .iloc[0]
            if "model_display" in performance_df.columns
            and not performance_df.loc[
                performance_df["model"] == model, "model_display"
            ]
            .dropna()
            .empty
            else model
        )
        model_slug = slugify(model)
        for metric_key, metric_config in PERFORMANCE_METRICS.items():
            metric_label = metric_config["label"]
            metric_slug = slugify(metric_key)

            ood_matrix = build_matrix(
                performance_df,
                model=model,
                metric=metric_key,
                value_column="mean_ood_performance",
            )
            if ood_matrix is not None:
                save_dataframe(
                    ood_matrix,
                    matrices_dir
                    / f"ood_performance_matrix_{model_slug}_{metric_slug}.csv",
                    matrices_dir
                    / f"ood_performance_matrix_{model_slug}_{metric_slug}.xlsx",
                    sheet_name="ood_performance",
                )
                save_cluster_heatmap(
                    ood_matrix,
                    figures_dir
                    / f"ood_performance_clustermap_{model_slug}_{metric_slug}.png",
                    title=f"OOD performance clustered heatmap | {model_display} | {metric_label}",
                    cmap="RdYlGn",
                    center=None,
                    value_label="Mean OOD performance",
                    row_cluster=True,
                    col_cluster=True,
                    bottom_legend_text=METHOD_LEGEND_TEXT,
                )
                save_cluster_heatmap(
                    ood_matrix.reindex(
                        columns=ordered_methods(ood_matrix.columns.tolist())
                    ),
                    figures_dir
                    / f"ood_performance_clustermap_{model_slug}_{metric_slug}_methods_ordered.png",
                    title=f"OOD performance clustered heatmap (methods ordered) | {model_display} | {metric_label}",
                    cmap="RdYlGn",
                    center=None,
                    value_label="Mean OOD performance",
                    row_cluster=True,
                    col_cluster=False,
                    bottom_legend_text=METHOD_LEGEND_TEXT,
                )
            else:
                warnings.append(
                    f"Could not create OOD performance matrix for {model} / {metric_key}"
                )

            gap_matrix = build_matrix(
                performance_df,
                model=model,
                metric=metric_key,
                value_column="id_ood_gap",
            )
            if gap_matrix is not None:
                save_dataframe(
                    gap_matrix,
                    matrices_dir / f"id_ood_gap_matrix_{model_slug}_{metric_slug}.csv",
                    matrices_dir / f"id_ood_gap_matrix_{model_slug}_{metric_slug}.xlsx",
                    sheet_name="id_ood_gap",
                )
                save_cluster_heatmap(
                    gap_matrix,
                    figures_dir
                    / f"id_ood_gap_clustermap_{model_slug}_{metric_slug}.png",
                    title=f"ID-OOD gap clustered heatmap | {model_display} | {metric_label}",
                    cmap="RdYlGn_r",
                    center=0.0,
                    value_label="ID-OOD gap",
                    row_cluster=True,
                    col_cluster=True,
                    bottom_legend_text=METHOD_LEGEND_TEXT,
                )
                save_cluster_heatmap(
                    gap_matrix.reindex(
                        columns=ordered_methods(gap_matrix.columns.tolist())
                    ),
                    figures_dir
                    / f"id_ood_gap_clustermap_{model_slug}_{metric_slug}_methods_ordered.png",
                    title=f"ID-OOD gap clustered heatmap (methods ordered) | {model_display} | {metric_label}",
                    cmap="RdYlGn_r",
                    center=0.0,
                    value_label="ID-OOD gap",
                    row_cluster=True,
                    col_cluster=False,
                    bottom_legend_text=METHOD_LEGEND_TEXT,
                )
            else:
                warnings.append(
                    f"Could not create ID-OOD gap matrix for {model} / {metric_key}"
                )

            variability_matrix = build_matrix(
                performance_df,
                model=model,
                metric=metric_key,
                value_column="std_ood_performance",
            )
            if (
                variability_matrix is not None
                and not variability_matrix.isna().all().all()
            ):
                save_dataframe(
                    variability_matrix,
                    matrices_dir
                    / f"ood_variability_matrix_{model_slug}_{metric_slug}.csv",
                    matrices_dir
                    / f"ood_variability_matrix_{model_slug}_{metric_slug}.xlsx",
                    sheet_name="ood_variability",
                )
                save_cluster_heatmap(
                    variability_matrix,
                    figures_dir
                    / f"ood_variability_clustermap_{model_slug}_{metric_slug}.png",
                    title=f"OOD variability clustered heatmap | {model_display} | {metric_label}",
                    cmap="RdYlGn_r",
                    center=None,
                    value_label="STD OOD performance",
                    row_cluster=True,
                    col_cluster=True,
                    bottom_legend_text=METHOD_LEGEND_TEXT,
                )
            else:
                warnings.append(
                    f"OOD variability not available for {model} / {metric_key}"
                )


def generate_meta_feature_heatmap(
    meta_original_df: pd.DataFrame,
    meta_zscore_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    matrices_dir = output_dir / "matrices"
    figures_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)

    save_dataframe(
        meta_original_df,
        matrices_dir / "meta_features_original_values.csv",
        matrices_dir / "meta_features_original_values.xlsx",
        sheet_name="meta_original",
    )
    save_dataframe(
        meta_zscore_df,
        matrices_dir / "meta_features_zscore_values.csv",
        matrices_dir / "meta_features_zscore_values.xlsx",
        sheet_name="meta_zscore",
    )

    save_cluster_heatmap(
        meta_zscore_df,
        figures_dir / "meta_features_clustermap.png",
        title="Meta-feature clustered heatmap (z-scored by column)",
        cmap="RdYlGn",
        center=0.0,
        value_label="Z-score",
        row_cluster=True,
        col_cluster=True,
        x_label="Meta-feature",
        y_label="Target site",
        xtick_rotation=45.0,
    )


def generate_correlation_heatmap(
    meta_original_df: pd.DataFrame, output_dir: Path, warnings: list[str]
) -> None:
    candidate_columns = [
        column
        for column in DISTANCE_COLUMNS
        + ["mean_ood_performance", "std_ood_performance", "id_ood_gap", "mean_ece"]
        if column in meta_original_df.columns
    ]
    if len(candidate_columns) < 2:
        warnings.append(
            "Not enough columns available to generate site-distance correlation heatmap."
        )
        return

    correlation_matrix = meta_original_df[candidate_columns].corr(method="pearson")
    values = correlation_matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(
        figsize=(
            max(8, 1.1 * len(candidate_columns)),
            max(6, 0.8 * len(candidate_columns)),
        )
    )
    image = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(correlation_matrix.shape[1]))
    ax.set_xticklabels(correlation_matrix.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(correlation_matrix.shape[0]))
    ax.set_yticklabels(correlation_matrix.index.tolist(), rotation=0)
    for row_index in range(correlation_matrix.shape[0]):
        for col_index in range(correlation_matrix.shape[1]):
            value = values[row_index, col_index]
            if np.isnan(value):
                continue
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#111111",
            )
    cbar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Pearson correlation")
    ax.set_title("Meta-feature correlation heatmap", fontsize=16, pad=14)
    fig.tight_layout()
    figure_path = output_dir / "figures" / "meta_features_correlation_heatmap.png"
    legacy_alias_path = output_dir / "figures" / "site_distance_correlation_heatmap.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220)
    fig.savefig(legacy_alias_path, dpi=220)
    plt.close(fig)


def write_readme(
    output_dir: Path,
    *,
    dataset_name: str,
    found_inputs: list[str],
    missing_inputs: list[str],
    warnings: list[str],
    removed_meta_columns: list[str],
    imputed_meta_columns: dict[str, float],
) -> None:
    warning_lines = (
        "\n".join(f"- {warning}" for warning in warnings) if warnings else "- none"
    )
    found_lines = (
        "\n".join(f"- {item}" for item in found_inputs) if found_inputs else "- none"
    )
    missing_lines = (
        "\n".join(f"- {item}" for item in missing_inputs)
        if missing_inputs
        else "- none"
    )
    removed_lines = (
        "\n".join(f"- {column}" for column in removed_meta_columns)
        if removed_meta_columns
        else "- none"
    )
    imputed_lines = (
        "\n".join(
            f"- {column}: median={value:.6f}"
            for column, value in imputed_meta_columns.items()
        )
        if imputed_meta_columns
        else "- none"
    )

    readme_text = f"""Clustered heatmap analysis outputs for {dataset_name}

This folder contains dedicated clustered heatmaps built from the existing cross-site and bootstrap outputs.

What the heatmaps represent:
- OOD performance heatmaps: rows are target sites, columns are methods M0-M4, values are mean OOD performance.
- ID-OOD gap heatmaps: rows are target sites, columns are methods, values are ID minus OOD performance.
- OOD variability heatmaps: rows are target sites, columns are methods, values are bootstrap standard deviation of OOD performance.
- Meta-feature heatmap: rows are target sites, columns are standardized meta-features such as sample size, entropy, site distance and aggregate performance summaries.

How to interpret clustering:
- target sites that cluster together show similar profiles across methods;
- methods that cluster together behave similarly across target sites;
- in the meta-feature heatmap, sites that cluster together share similar structural properties (distance, entropy, size) and outcome profiles.

Clustering settings:
- hierarchical clustering with linkage = average
- distance metric = euclidean
- meta-feature columns standardized with z-score before clustering

Found inputs:
{found_lines}

Missing inputs:
{missing_lines}

Meta-feature preprocessing notes:
- columns removed because completely empty:
{removed_lines}

- columns imputed with the median before z-scoring:
{imputed_lines}

Warnings:
{warning_lines}

How to read the figures:
- higher values in OOD performance heatmaps are better.
- larger positive values in ID-OOD gap heatmaps indicate a bigger loss from ID to OOD.
- larger values in OOD variability heatmaps indicate less stable results across bootstrap resampling.
- in the meta-feature heatmap, red/blue only indicate higher/lower z-scored values within each column, not absolute performance.

No original pipeline files or existing figures were modified.
"""
    (output_dir / "README.txt").write_text(readme_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_plot_style()

    warnings: list[str] = []
    performance_df, dataset_name, found_inputs, missing_inputs = load_performance_table(
        input_root, warnings
    )

    sample_size_path = input_root / "Sample_size" / "sample_size_by_site.csv"
    entropy_path = input_root / "Entropy" / "entropy_summary_by_site.csv"
    site_distance_path = input_root / "Site_distance" / "site_distance_by_target.csv"

    sample_size_df = load_optional_table(
        sample_size_path, required_columns=["Target site"], warnings=warnings
    )
    entropy_df = load_optional_table(
        entropy_path, required_columns=["target_site"], warnings=warnings
    )
    site_distance_df = load_optional_table(
        site_distance_path, required_columns=["target_site"], warnings=warnings
    )

    for optional_path, optional_df in [
        (sample_size_path, sample_size_df),
        (entropy_path, entropy_df),
        (site_distance_path, site_distance_df),
    ]:
        if optional_df is not None:
            found_inputs.append(str(optional_path))
        else:
            missing_inputs.append(str(optional_path))

    generate_performance_heatmaps(performance_df, output_dir, warnings)

    meta_df, meta_missing_notes = build_meta_feature_table(
        performance_df=performance_df,
        sample_size_df=sample_size_df,
        entropy_df=entropy_df,
        site_distance_df=site_distance_df,
    )
    warnings.extend(meta_missing_notes)
    meta_original_df, meta_zscore_df, removed_meta_columns, imputed_meta_columns = (
        prepare_meta_feature_matrices(meta_df)
    )

    generate_meta_feature_heatmap(meta_original_df, meta_zscore_df, output_dir)
    generate_correlation_heatmap(meta_original_df, output_dir, warnings)

    write_readme(
        output_dir,
        dataset_name=dataset_name,
        found_inputs=found_inputs,
        missing_inputs=missing_inputs,
        warnings=warnings,
        removed_meta_columns=removed_meta_columns,
        imputed_meta_columns=imputed_meta_columns,
    )

    print(f"Wrote clustered heatmap analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
