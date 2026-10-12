"""Post-experiment analysis: generate method robustness condition analysis."""

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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib

os.environ["MPLCONFIGDIR"] = str(
    Path(tempfile.gettempdir()) / "codex_method_robustness_mpl"
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


METHOD_ORDER = ["M0", "M1", "M2", "M3", "M4"]
METHOD_COLORS = {
    "M0": "#4C78A8",
    "M1": "#F58518",
    "M2": "#54A24B",
    "M3": "#E45756",
    "M4": "#72B7B2",
}
METHOD_LABELS = {
    "M0": "M0: Baseline",
    "M1": "M1: ComBat",
    "M2": "M2: Weighting",
    "M3": "M3: Transductive",
    "M4": "M4: ComBat + Transductive",
}
METHOD_LEGEND_TEXT = " | ".join(METHOD_LABELS[method] for method in METHOD_ORDER)
DATASET_ROOTS = {
    "UCI": Path(_configured_path("results_uci")),
    "TCGA": Path(_configured_path("results_tcga")),
    "ABIDE": Path(_configured_path("results_abide")),
}
PATTERN_MASTER_RELATIVE = Path("Pattern_discovery/pattern_discovery_master_table.csv")
METADATA_SHIFT_RELATIVE = Path("Metadata_discussion/metadata_shift_by_target_site.csv")
HEATMAP_MATRIX_RELATIVE = Path("Heatmap_cluster/matrices")
RANDOM_STATE = 42
BUNDLED_EXCEL_PYTHON = Path(_configured_path("python_executable"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final method robustness and condition analysis."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def save_dataframe(
    df: pd.DataFrame,
    csv_path: Path,
    xlsx_path: Path | None = None,
    *,
    sheet_name: str = "Sheet1",
    index: bool = False,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=index)
    if xlsx_path is not None:
        write_excel_with_fallback(df, xlsx_path, sheet_name=sheet_name, index=index)


def write_excel_with_fallback(
    df: pd.DataFrame, path: Path, *, sheet_name: str, index: bool
) -> None:
    try:
        with pd.ExcelWriter(path) as writer:
            df.to_excel(writer, sheet_name=sheet_name[:31], index=index)
        return
    except ModuleNotFoundError:
        if not BUNDLED_EXCEL_PYTHON.exists():
            raise RuntimeError(
                "openpyxl is unavailable and the bundled runtime was not found."
            )

    temp_dir = Path(tempfile.mkdtemp(prefix="method_robustness_xlsx_"))
    temp_csv = temp_dir / "data.csv"
    df.to_csv(temp_csv, index=index)
    helper_code = """
import sys
import pandas as pd

csv_path, xlsx_path, sheet_name, use_index = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == '1'
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


def safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float, int]:
    valid = pd.concat(
        [pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1
    ).dropna()
    n_obs = len(valid)
    if n_obs < 3:
        return math.nan, math.nan, math.nan, math.nan, n_obs
    x_values = valid.iloc[:, 0]
    y_values = valid.iloc[:, 1]
    if x_values.nunique() < 2 or y_values.nunique() < 2:
        return math.nan, math.nan, math.nan, math.nan, n_obs
    pearson_r, pearson_p = pearsonr(x_values, y_values)
    spearman_r, spearman_p = spearmanr(x_values, y_values)
    return (
        float(pearson_r),
        float(pearson_p),
        float(spearman_r),
        float(spearman_p),
        n_obs,
    )


def safe_zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mean = numeric.mean()
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (numeric - mean) / std


def minmax_scale(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series(0.5, index=series.index)
    minimum = numeric.min()
    maximum = numeric.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(0.5, index=series.index)
    return (numeric - minimum) / (maximum - minimum)


def qcut_tertiles(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series("MEDIUM", index=series.index, dtype="object")
    valid = numeric.dropna()
    if valid.nunique() < 2:
        result.loc[valid.index] = "MEDIUM"
        result.loc[numeric.isna()] = "UNKNOWN"
        return result
    ranked = valid.rank(method="first")
    try:
        bins = pd.qcut(ranked, q=3, labels=["LOW", "MEDIUM", "HIGH"])
        result.loc[bins.index] = bins.astype(str)
    except ValueError:
        median = valid.median()
        result.loc[valid.index] = np.where(valid <= median, "LOW", "HIGH")
    result.loc[numeric.isna()] = "UNKNOWN"
    return result


def add_method_footer(fig: plt.Figure, text: str = METHOD_LEGEND_TEXT) -> None:
    fig.text(0.5, 0.012, text, ha="center", va="bottom", fontsize=10, color="#333333")


def pretty_method(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def prettify_condition_name(variable: str, group: str) -> str:
    mapping = {
        "site_distance_group": "site distance",
        "entropy_group": "entropy",
        "sample_size_group": "sample size",
        "metadata_shift_group": "metadata shift",
        "calibration_group": "calibration instability",
    }
    label = mapping.get(variable, variable.replace("_", " "))
    return f"{group.lower()} {label}"


def load_optional_csv(
    path: Path, warnings: list[str], found_inputs: list[str]
) -> pd.DataFrame | None:
    if not path.exists():
        warnings.append(f"Missing optional input: {path}")
        return None
    found_inputs.append(str(path))
    return pd.read_csv(path, low_memory=False)


def load_dataset_outputs(
    dataset_short: str, dataset_root: Path, warnings: list[str], found_inputs: list[str]
) -> pd.DataFrame:
    pattern_path = dataset_root / PATTERN_MASTER_RELATIVE
    metadata_path = dataset_root / METADATA_SHIFT_RELATIVE
    heatmap_dir = dataset_root / HEATMAP_MATRIX_RELATIVE

    if not pattern_path.exists():
        raise RuntimeError(f"Required file missing: {pattern_path}")
    master_df = pd.read_csv(pattern_path, low_memory=False)
    found_inputs.append(str(pattern_path))

    if "ece" in master_df.columns and "mean_ece" not in master_df.columns:
        master_df = master_df.rename(columns={"ece": "mean_ece"})

    metadata_df = load_optional_csv(metadata_path, warnings, found_inputs)
    if metadata_df is not None and not metadata_df.empty:
        merge_columns = [
            column
            for column in metadata_df.columns
            if column != "dataset" and column not in master_df.columns
        ]
        if merge_columns:
            master_df = master_df.merge(
                metadata_df[["target_site", *merge_columns]],
                on="target_site",
                how="left",
            )

    if heatmap_dir.exists():
        found_inputs.append(str(heatmap_dir))
    else:
        warnings.append(f"Missing optional clustered heatmap matrices: {heatmap_dir}")

    master_df["dataset_short"] = dataset_short
    if "target_site_display" not in master_df.columns:
        master_df["target_site_display"] = master_df["target_site"]
    if "model_display" not in master_df.columns:
        master_df["model_display"] = master_df["model"]
    return master_df


def build_master_table(input_root: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    warnings: list[str] = []
    found_inputs: list[str] = []
    frames: list[pd.DataFrame] = []
    for dataset_short, dataset_root in DATASET_ROOTS.items():
        frames.append(
            load_dataset_outputs(dataset_short, dataset_root, warnings, found_inputs)
        )

    master_df = pd.concat(frames, ignore_index=True)
    if "mean_ece" not in master_df.columns:
        master_df["mean_ece"] = np.nan

    desired_order = [
        "dataset_short",
        "dataset",
        "target_site",
        "target_site_display",
        "model",
        "model_display",
        "metric",
        "method",
        "mean_ood_performance",
        "std_ood_performance",
        "mean_id_performance",
        "id_ood_gap",
        "mean_ece",
        "n_total_target",
        "n_source",
        "n_train_1",
        "n_test_1",
        "entropy",
        "normalized_entropy",
        "euclidean_centroid_distance",
        "mahalanobis_distance",
        "wasserstein_mean_distance",
        "jensen_shannon_mean_divergence",
        "mmd_rbf",
        "overall_metadata_shift_score",
        "mean_continuous_metadata_shift",
        "mean_categorical_metadata_shift",
    ]
    existing = [column for column in desired_order if column in master_df.columns]
    remaining = [column for column in master_df.columns if column not in existing]
    master_df = master_df[existing + remaining].copy()
    return master_df, warnings, found_inputs


def add_composite_columns(master_df: pd.DataFrame) -> pd.DataFrame:
    distance_columns = [
        "euclidean_centroid_distance",
        "mahalanobis_distance",
        "wasserstein_mean_distance",
        "jensen_shannon_mean_divergence",
        "mmd_rbf",
    ]
    site_key = ["dataset_short", "dataset", "target_site", "target_site_display"]
    site_meta = master_df[
        site_key
        + distance_columns
        + ["normalized_entropy", "n_total_target", "overall_metadata_shift_score"]
    ].drop_duplicates()

    site_meta["site_distance_composite"] = np.nan
    for dataset_short, subset in site_meta.groupby("dataset_short"):
        index = subset.index
        z_frames = []
        for column in distance_columns:
            if column not in subset.columns:
                continue
            z_frames.append(safe_zscore(subset[column]))
        if z_frames:
            z_matrix = pd.concat(z_frames, axis=1)
            site_meta.loc[index, "site_distance_composite"] = z_matrix.mean(axis=1)

    master_df = master_df.merge(
        site_meta[site_key + ["site_distance_composite"]], on=site_key, how="left"
    )
    return master_df


def build_site_summary(master_df: pd.DataFrame) -> pd.DataFrame:
    site_key = ["dataset_short", "dataset", "target_site", "target_site_display"]
    first_columns = [
        "n_total_target",
        "n_source",
        "n_train_1",
        "n_test_1",
        "entropy",
        "normalized_entropy",
        "euclidean_centroid_distance",
        "mahalanobis_distance",
        "wasserstein_mean_distance",
        "jensen_shannon_mean_divergence",
        "mmd_rbf",
        "overall_metadata_shift_score",
        "mean_continuous_metadata_shift",
        "mean_categorical_metadata_shift",
        "site_distance_composite",
    ]
    agg_map = {
        column: "first" for column in first_columns if column in master_df.columns
    }
    agg_map.update(
        {
            "mean_ood_performance": "mean",
            "std_ood_performance": "mean",
            "id_ood_gap": "mean",
            "mean_ece": "mean",
        }
    )
    site_df = master_df.groupby(site_key, as_index=False).agg(agg_map)
    site_df = site_df.rename(
        columns={
            "mean_ood_performance": "site_mean_ood_performance",
            "std_ood_performance": "site_mean_ood_variability",
            "id_ood_gap": "site_mean_id_ood_gap",
            "mean_ece": "site_mean_ece",
        }
    )

    for dataset_short, subset in site_df.groupby("dataset_short"):
        idx = subset.index
        site_df.loc[idx, "site_distance_group"] = qcut_tertiles(
            subset["site_distance_composite"]
        ).to_numpy()
        site_df.loc[idx, "entropy_group"] = qcut_tertiles(
            subset["normalized_entropy"]
        ).to_numpy()
        site_df.loc[idx, "sample_size_group"] = qcut_tertiles(
            subset["n_total_target"]
        ).to_numpy()
        site_df.loc[idx, "metadata_shift_group"] = qcut_tertiles(
            subset["overall_metadata_shift_score"]
        ).to_numpy()
        site_df.loc[idx, "calibration_group"] = qcut_tertiles(
            subset["site_mean_ece"]
        ).to_numpy()
    return site_df


def build_method_site_table(
    master_df: pd.DataFrame, site_df: pd.DataFrame
) -> pd.DataFrame:
    group_cols = [
        "dataset_short",
        "dataset",
        "target_site",
        "target_site_display",
        "method",
    ]
    first_columns = [
        "n_total_target",
        "n_source",
        "n_train_1",
        "n_test_1",
        "entropy",
        "normalized_entropy",
        "euclidean_centroid_distance",
        "mahalanobis_distance",
        "wasserstein_mean_distance",
        "jensen_shannon_mean_divergence",
        "mmd_rbf",
        "overall_metadata_shift_score",
        "mean_continuous_metadata_shift",
        "mean_categorical_metadata_shift",
        "site_distance_composite",
    ]
    agg_map = {
        column: "first" for column in first_columns if column in master_df.columns
    }
    agg_map.update(
        {
            "mean_ood_performance": "mean",
            "std_ood_performance": "mean",
            "mean_id_performance": "mean",
            "id_ood_gap": "mean",
            "mean_ece": "mean",
        }
    )
    method_df = master_df.groupby(group_cols, as_index=False).agg(agg_map)
    method_df = method_df.merge(
        site_df[
            [
                "dataset_short",
                "target_site",
                "site_distance_group",
                "entropy_group",
                "sample_size_group",
                "metadata_shift_group",
                "calibration_group",
            ]
        ],
        on=["dataset_short", "target_site"],
        how="left",
    )
    return method_df


def choose_best_method(
    subset: pd.DataFrame,
    value_column: str,
    *,
    higher_is_better: bool,
    absolute: bool = False,
) -> tuple[str, float]:
    working = subset.copy()
    values = pd.to_numeric(working[value_column], errors="coerce")
    if absolute:
        values = values.abs()
    working["_criterion"] = values
    ascending = not higher_is_better
    working["_method_rank"] = working["method"].map(
        {method: idx for idx, method in enumerate(METHOD_ORDER)}
    )
    ordered = working.sort_values(
        ["_criterion", "_method_rank"], ascending=[ascending, True]
    )
    row = ordered.iloc[0]
    return str(row["method"]), float(row[value_column])


def build_best_method_table(master_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "dataset_short",
        "dataset",
        "target_site",
        "target_site_display",
        "model",
        "model_display",
        "metric",
    ]
    for group_key, subset in master_df.groupby(group_cols):
        (
            dataset_short,
            dataset,
            target_site,
            target_display,
            model,
            model_display,
            metric,
        ) = group_key
        best_ood, best_ood_value = choose_best_method(
            subset, "mean_ood_performance", higher_is_better=True
        )
        best_gap, best_gap_value = choose_best_method(
            subset, "id_ood_gap", higher_is_better=False, absolute=True
        )
        best_ece, best_ece_value = choose_best_method(
            subset, "mean_ece", higher_is_better=False
        )
        best_var, best_var_value = choose_best_method(
            subset, "std_ood_performance", higher_is_better=False
        )
        rows.append(
            {
                "dataset_short": dataset_short,
                "dataset": dataset,
                "target_site": target_site,
                "target_site_display": target_display,
                "model": model,
                "model_display": model_display,
                "metric": metric,
                "best_method_ood": best_ood,
                "best_method_ood_value": best_ood_value,
                "best_method_gap": best_gap,
                "best_method_gap_value": best_gap_value,
                "best_method_ece": best_ece,
                "best_method_ece_value": best_ece_value,
                "best_method_variability": best_var,
                "best_method_variability_value": best_var_value,
            }
        )
    return pd.DataFrame(rows)


def build_winner_frequency_table(best_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        rows.append(
            {
                "method": method,
                "method_label": pretty_method(method),
                "n_best_ood": int((best_df["best_method_ood"] == method).sum()),
                "n_best_gap": int((best_df["best_method_gap"] == method).sum()),
                "n_best_ece": int((best_df["best_method_ece"] == method).sum()),
                "n_best_variability": int(
                    (best_df["best_method_variability"] == method).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_winner_frequency_bars(winner_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("n_best_ood", "winner_frequency_ood.png", "Winner frequency: OOD performance"),
        ("n_best_gap", "winner_frequency_gap.png", "Winner frequency: ID-OOD gap"),
        ("n_best_ece", "winner_frequency_ece.png", "Winner frequency: calibration"),
        (
            "n_best_variability",
            "winner_frequency_variability.png",
            "Winner frequency: OOD variability",
        ),
    ]
    x = np.arange(len(METHOD_ORDER))
    for column, filename, title in metrics:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        values = [
            float(winner_df.loc[winner_df["method"] == method, column].iloc[0])
            for method in METHOD_ORDER
        ]
        colors = [METHOD_COLORS[method] for method in METHOD_ORDER]
        bars = ax.bar(x, values, color=colors, edgecolor="#333333", linewidth=0.8)
        for bar, value in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{int(value)}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.set_ylim(0, max(values) + 2 if values else 1)
        add_method_footer(fig)
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)


def summarize_by_condition(method_df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    group_name = group_column.replace("_group", "")
    summary = (
        method_df.groupby(
            ["dataset_short", "dataset", group_column, "method"], as_index=False
        )
        .agg(
            n_sites=("target_site", "nunique"),
            mean_ood_performance=("mean_ood_performance", "mean"),
            mean_id_ood_gap=("id_ood_gap", "mean"),
            mean_ece=("mean_ece", "mean"),
            mean_std_ood_performance=("std_ood_performance", "mean"),
        )
        .rename(columns={group_column: "condition_level"})
    )
    summary.insert(2, "condition_variable", group_name)
    return summary


def build_method_sensitivity_table(method_df: pd.DataFrame) -> pd.DataFrame:
    relationships = [
        ("site_distance_composite", "mean_ood_performance", "site_distance_vs_ood"),
        ("site_distance_composite", "id_ood_gap", "site_distance_vs_gap"),
        ("normalized_entropy", "mean_ood_performance", "entropy_vs_ood"),
        ("n_total_target", "std_ood_performance", "sample_size_vs_ood_std"),
        ("overall_metadata_shift_score", "id_ood_gap", "metadata_shift_vs_gap"),
    ]
    rows: list[dict[str, Any]] = []
    for dataset_scope, scoped_df in [
        ("ALL", method_df),
        *[(name, subset) for name, subset in method_df.groupby("dataset_short")],
    ]:
        for method, subset in scoped_df.groupby("method"):
            for predictor, outcome, relationship in relationships:
                pearson_r, pearson_p, spearman_r, spearman_p, n_obs = safe_corr(
                    subset[predictor], subset[outcome]
                )
                slope = math.nan
                valid = subset[[predictor, outcome]].dropna()
                if len(valid) >= 2 and valid[predictor].nunique() > 1:
                    slope = float(np.polyfit(valid[predictor], valid[outcome], 1)[0])
                rows.append(
                    {
                        "dataset_scope": dataset_scope,
                        "method": method,
                        "predictor": predictor,
                        "outcome": outcome,
                        "relationship": relationship,
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                        "spearman_r": spearman_r,
                        "spearman_p": spearman_p,
                        "slope": slope,
                        "n_sites": n_obs,
                    }
                )
    return pd.DataFrame(rows)


def build_pattern_correlations(master_df: pd.DataFrame) -> pd.DataFrame:
    explanatory = [
        "n_total_target",
        "n_source",
        "entropy",
        "normalized_entropy",
        "site_distance_composite",
        "euclidean_centroid_distance",
        "mahalanobis_distance",
        "wasserstein_mean_distance",
        "jensen_shannon_mean_divergence",
        "mmd_rbf",
        "overall_metadata_shift_score",
        "mean_continuous_metadata_shift",
        "mean_categorical_metadata_shift",
        "mean_ece",
    ]
    outcomes = ["mean_ood_performance", "std_ood_performance", "id_ood_gap"]
    rows: list[dict[str, Any]] = []
    for dataset_scope, scoped_df in [
        ("ALL", master_df),
        *[(name, subset) for name, subset in master_df.groupby("dataset_short")],
    ]:
        for variable_x in explanatory:
            if variable_x not in scoped_df.columns:
                continue
            for variable_y in outcomes:
                if variable_y not in scoped_df.columns:
                    continue
                pearson_r, pearson_p, spearman_r, spearman_p, n_obs = safe_corr(
                    scoped_df[variable_x], scoped_df[variable_y]
                )
                rows.append(
                    {
                        "dataset_scope": dataset_scope,
                        "variable_x": variable_x,
                        "variable_y": variable_y,
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                        "spearman_r": spearman_r,
                        "spearman_p": spearman_p,
                        "n_obs": n_obs,
                    }
                )
    return pd.DataFrame(rows)


def plot_pattern_correlation_heatmap(
    pattern_corr_df: pd.DataFrame, output_path: Path
) -> None:
    subset = pattern_corr_df[pattern_corr_df["dataset_scope"] == "ALL"].copy()
    if subset.empty:
        return
    matrix = subset.pivot(index="variable_x", columns="variable_y", values="spearman_r")
    fig, ax = plt.subplots(figsize=(6.8, max(6.2, 0.45 * len(matrix.index) + 1.5)))
    image = ax.imshow(
        matrix.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1
    )
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        [column.replace("_", " ") for column in matrix.columns], rotation=28, ha="right"
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([row.replace("_", " ") for row in matrix.index])
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix.iloc[row_idx, col_idx]
            if pd.isna(value):
                continue
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="#111111",
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Spearman r")
    ax.set_title("Pattern correlation heatmap")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_method_robustness_scores(
    method_df: pd.DataFrame, sensitivity_df: pd.DataFrame
) -> pd.DataFrame:
    scope_filters = {
        "global": method_df.index,
        "high_shift": method_df.index[method_df["metadata_shift_group"] == "HIGH"],
        "low_entropy": method_df.index[method_df["entropy_group"] == "LOW"],
        "small_sample": method_df.index[method_df["sample_size_group"] == "LOW"],
    }

    sensitivity_lookup = {}
    for _, row in sensitivity_df[sensitivity_df["dataset_scope"] == "ALL"].iterrows():
        sensitivity_lookup[(row["method"], row["relationship"])] = row

    score_rows: list[dict[str, Any]] = []
    for scope_name, indexer in scope_filters.items():
        subset = method_df.loc[indexer].copy()
        if subset.empty:
            continue
        method_summary = subset.groupby("method", as_index=False).agg(
            mean_ood_performance=("mean_ood_performance", "mean"),
            mean_abs_gap=(
                "id_ood_gap",
                lambda s: np.nanmean(np.abs(pd.to_numeric(s, errors="coerce"))),
            ),
            mean_std_ood_performance=("std_ood_performance", "mean"),
            mean_ece=("mean_ece", "mean"),
            n_sites=("target_site", "nunique"),
        )
        if scope_name == "global":
            distance_sensitivity = []
            metadata_sensitivity = []
            for method in method_summary["method"]:
                site_row = sensitivity_lookup.get((method, "site_distance_vs_ood"))
                meta_row = sensitivity_lookup.get((method, "metadata_shift_vs_gap"))
                distance_sensitivity.append(
                    abs(site_row["spearman_r"])
                    if site_row is not None and pd.notna(site_row["spearman_r"])
                    else math.nan
                )
                metadata_sensitivity.append(
                    abs(meta_row["spearman_r"])
                    if meta_row is not None and pd.notna(meta_row["spearman_r"])
                    else math.nan
                )
            method_summary["distance_sensitivity_abs_spearman"] = distance_sensitivity
            method_summary["metadata_shift_sensitivity_abs_spearman"] = (
                metadata_sensitivity
            )
        else:
            distance_sens_rows = []
            metadata_sens_rows = []
            for method, method_subset in subset.groupby("method"):
                _, _, spearman_r_dist, _, _ = safe_corr(
                    method_subset["site_distance_composite"],
                    method_subset["mean_ood_performance"],
                )
                _, _, spearman_r_meta, _, _ = safe_corr(
                    method_subset["overall_metadata_shift_score"],
                    method_subset["id_ood_gap"],
                )
                distance_sens_rows.append(
                    (
                        method,
                        abs(spearman_r_dist) if pd.notna(spearman_r_dist) else math.nan,
                    )
                )
                metadata_sens_rows.append(
                    (
                        method,
                        abs(spearman_r_meta) if pd.notna(spearman_r_meta) else math.nan,
                    )
                )
            dist_map = dict(distance_sens_rows)
            meta_map = dict(metadata_sens_rows)
            method_summary["distance_sensitivity_abs_spearman"] = method_summary[
                "method"
            ].map(dist_map)
            method_summary["metadata_shift_sensitivity_abs_spearman"] = method_summary[
                "method"
            ].map(meta_map)

        component_columns = {
            "z_ood": method_summary["mean_ood_performance"],
            "z_gap": -safe_zscore(method_summary["mean_abs_gap"]),
            "z_var": -safe_zscore(method_summary["mean_std_ood_performance"]),
            "z_ece": -safe_zscore(method_summary["mean_ece"]),
            "z_distance_sensitivity": -safe_zscore(
                method_summary["distance_sensitivity_abs_spearman"]
            ),
            "z_metadata_sensitivity": -safe_zscore(
                method_summary["metadata_shift_sensitivity_abs_spearman"]
            ),
        }
        for name, values in component_columns.items():
            if name == "z_ood":
                method_summary[name] = safe_zscore(values)
            else:
                method_summary[name] = values
        method_summary["robustness_score"] = method_summary[
            [
                "z_ood",
                "z_gap",
                "z_var",
                "z_ece",
                "z_distance_sensitivity",
                "z_metadata_sensitivity",
            ]
        ].mean(axis=1, skipna=True)
        method_summary["scope"] = scope_name
        score_rows.append(method_summary)

    scores_df = (
        pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    )
    if not scores_df.empty:
        scope_to_column = {
            "global": "robustness_score_global",
            "high_shift": "robustness_score_high_shift",
            "low_entropy": "robustness_score_low_entropy",
            "small_sample": "robustness_score_small_sample",
        }
        scores_df["score_label"] = scores_df["scope"].map(scope_to_column)
    return scores_df


def plot_robustness_score_barplot(scores_df: pd.DataFrame, output_path: Path) -> None:
    if scores_df.empty:
        return
    scope_order = ["global", "high_shift", "low_entropy", "small_sample"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), sharey=True)
    for ax, scope in zip(axes.flatten(), scope_order, strict=False):
        subset = scores_df[scores_df["scope"] == scope].copy()
        subset["method_rank"] = subset["method"].map(
            {method: idx for idx, method in enumerate(METHOD_ORDER)}
        )
        subset = subset.sort_values("method_rank")
        colors = [METHOD_COLORS[method] for method in subset["method"]]
        ax.bar(
            subset["method"],
            subset["robustness_score"],
            color=colors,
            edgecolor="#333333",
            linewidth=0.8,
        )
        for _, row in subset.iterrows():
            ax.text(
                row["method"],
                row["robustness_score"] + 0.05,
                f"{row['robustness_score']:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(scope.replace("_", " ").title())
        ax.axhline(0, color="#666666", linewidth=0.8)
        ax.set_xlabel("Method")
        ax.set_ylabel("Robustness score")
    fig.suptitle("Method robustness scores", fontsize=16, y=0.98)
    add_method_footer(fig)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_robustness_radar(scores_df: pd.DataFrame, output_path: Path) -> None:
    subset = scores_df[scores_df["scope"] == "global"].copy()
    if subset.empty:
        return
    component_columns = [
        "z_ood",
        "z_gap",
        "z_var",
        "z_ece",
        "z_distance_sensitivity",
        "z_metadata_sensitivity",
    ]
    labels = [
        "OOD",
        "Gap",
        "Variability",
        "ECE",
        "Distance sensitivity",
        "Metadata sensitivity",
    ]
    scaled = subset.copy()
    for column in component_columns:
        scaled[column] = minmax_scale(scaled[column])
    angles = np.linspace(0, 2 * np.pi, len(component_columns), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(10.8, 8.8))
    ax = fig.add_subplot(111, polar=True)
    for method in METHOD_ORDER:
        row = scaled[scaled["method"] == method]
        if row.empty:
            continue
        values = row.iloc[0][component_columns].tolist()
        values += values[:1]
        ax.plot(
            angles,
            values,
            color=METHOD_COLORS[method],
            linewidth=2.2,
            label=pretty_method(method),
        )
        ax.fill(angles, values, color=METHOD_COLORS[method], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels([])
    ax.set_title("Global robustness profile by method", y=1.12, fontsize=16)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def fit_explanatory_models(
    site_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    predictors = [
        "normalized_entropy",
        "n_total_target",
        "site_distance_composite",
        "overall_metadata_shift_score",
        "site_mean_ece",
    ]
    outcomes = {
        "site_mean_ood_performance": "mean_ood_performance",
        "site_mean_id_ood_gap": "id_ood_gap",
        "site_mean_ood_variability": "std_ood_performance",
    }
    model_specs = {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    }
    result_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}

    design = site_df[predictors].copy()
    for outcome_column, outcome_label in outcomes.items():
        y = pd.to_numeric(site_df[outcome_column], errors="coerce")
        valid_mask = y.notna()
        X = design.loc[valid_mask].copy()
        y_valid = y.loc[valid_mask].copy()
        if len(X) < 5:
            continue
        X = X.apply(pd.to_numeric, errors="coerce")
        cv = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=RANDOM_STATE)
        best_prediction_df: pd.DataFrame | None = None
        best_r2 = -np.inf

        for model_name, estimator in model_specs.items():
            pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", estimator),
                ]
            )
            predictions = cross_val_predict(pipeline, X, y_valid, cv=cv)
            rmse = float(math.sqrt(mean_squared_error(y_valid, predictions)))
            result_rows.append(
                {
                    "dataset_scope": "ALL",
                    "outcome": outcome_label,
                    "model_type": model_name,
                    "n_sites": len(X),
                    "r2": float(r2_score(y_valid, predictions)),
                    "mae": float(mean_absolute_error(y_valid, predictions)),
                    "rmse": rmse,
                }
            )

            pipeline.fit(X, y_valid)
            final_model = pipeline.named_steps["model"]
            for feature_name, coefficient in zip(
                predictors,
                getattr(final_model, "coef_", np.full(len(predictors), np.nan)),
                strict=False,
            ):
                coefficient_rows.append(
                    {
                        "dataset_scope": "ALL",
                        "outcome": outcome_label,
                        "model_type": model_name,
                        "feature": feature_name,
                        "coefficient": float(coefficient)
                        if coefficient is not None
                        else math.nan,
                        "intercept": float(
                            getattr(final_model, "intercept_", math.nan)
                        ),
                    }
                )

            current_r2 = r2_score(y_valid, predictions)
            if current_r2 > best_r2:
                best_r2 = current_r2
                best_prediction_df = pd.DataFrame(
                    {
                        "dataset_short": site_df.loc[
                            valid_mask, "dataset_short"
                        ].to_numpy(),
                        "target_site": site_df.loc[
                            valid_mask, "target_site"
                        ].to_numpy(),
                        "target_site_display": site_df.loc[
                            valid_mask, "target_site_display"
                        ].to_numpy(),
                        "observed": y_valid.to_numpy(),
                        "predicted": predictions,
                        "outcome": outcome_label,
                        "model_type": model_name,
                    }
                )
        if best_prediction_df is not None:
            prediction_frames[outcome_label] = best_prediction_df

    return pd.DataFrame(result_rows), pd.DataFrame(coefficient_rows), prediction_frames


def plot_predicted_vs_observed(
    predictions_df: pd.DataFrame, output_path: Path, title: str
) -> None:
    if predictions_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    colors = (
        predictions_df["dataset_short"]
        .map({"UCI": "#4C78A8", "TCGA": "#F58518", "ABIDE": "#54A24B"})
        .fillna("#555555")
    )
    ax.scatter(
        predictions_df["observed"],
        predictions_df["predicted"],
        c=colors,
        s=65,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    min_val = float(
        min(predictions_df["observed"].min(), predictions_df["predicted"].min())
    )
    max_val = float(
        max(predictions_df["observed"].max(), predictions_df["predicted"].max())
    )
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="#444444",
        linewidth=1.1,
    )
    for _, row in predictions_df.iterrows():
        ax.text(
            row["observed"] + 0.002,
            row["predicted"] + 0.002,
            str(row["target_site_display"]),
            fontsize=8.7,
            color="#333333",
        )
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_hard_easy_table(site_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    median_ood = site_df["site_mean_ood_performance"].median()
    median_gap = site_df["site_mean_id_ood_gap"].median()
    site_df = site_df.copy()
    site_df["hard_by_ood"] = (
        site_df["site_mean_ood_performance"] <= median_ood
    ).astype(int)
    site_df["hard_by_gap"] = (site_df["site_mean_id_ood_gap"] >= median_gap).astype(int)
    compare_columns = {
        "normalized_entropy": "entropy",
        "n_total_target": "sample_size",
        "site_distance_composite": "site_distance",
        "overall_metadata_shift_score": "metadata_shift",
        "site_mean_ece": "ece",
    }
    for definition_column, definition_label in [
        ("hard_by_ood", "hard_by_ood"),
        ("hard_by_gap", "hard_by_gap"),
    ]:
        for variable, variable_label in compare_columns.items():
            hard = pd.to_numeric(
                site_df.loc[site_df[definition_column] == 1, variable], errors="coerce"
            )
            easy = pd.to_numeric(
                site_df.loc[site_df[definition_column] == 0, variable], errors="coerce"
            )
            rows.append(
                {
                    "definition": definition_label,
                    "variable": variable_label,
                    "hard_mean": float(hard.mean()),
                    "easy_mean": float(easy.mean()),
                    "difference_hard_minus_easy": float(hard.mean() - easy.mean()),
                    "n_hard": int(site_df[definition_column].sum()),
                    "n_easy": int((site_df[definition_column] == 0).sum()),
                }
            )
    return site_df, pd.DataFrame(rows)


def plot_hard_easy_boxplots(site_df: pd.DataFrame, output_dir: Path) -> None:
    plot_specs = [
        ("normalized_entropy", "hard_vs_easy_entropy.png", "Normalized entropy"),
        (
            "site_distance_composite",
            "hard_vs_easy_site_distance.png",
            "Site distance composite",
        ),
        ("n_total_target", "hard_vs_easy_sample_size.png", "Target sample size"),
        (
            "overall_metadata_shift_score",
            "hard_vs_easy_metadata_shift.png",
            "Metadata shift score",
        ),
    ]
    site_df = site_df.copy()
    site_df["hard_by_ood_label"] = np.where(site_df["hard_by_ood"] == 1, "Hard", "Easy")
    for variable, filename, ylabel in plot_specs:
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        hard_values = (
            pd.to_numeric(
                site_df.loc[site_df["hard_by_ood"] == 1, variable], errors="coerce"
            )
            .dropna()
            .to_numpy()
        )
        easy_values = (
            pd.to_numeric(
                site_df.loc[site_df["hard_by_ood"] == 0, variable], errors="coerce"
            )
            .dropna()
            .to_numpy()
        )
        bp = ax.boxplot(
            [easy_values, hard_values],
            tick_labels=["Easy", "Hard"],
            patch_artist=True,
            widths=0.6,
        )
        colors = ["#9ecae1", "#fdae6b"]
        for patch, color in zip(bp["boxes"], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_edgecolor("#444444")
        for median in bp["medians"]:
            median.set_color("#111111")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Hard vs easy domains: {ylabel.lower()}")
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)


def plot_method_relationship(
    method_df: pd.DataFrame,
    predictor: str,
    outcome: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    text_lines = []
    for method in METHOD_ORDER:
        subset = method_df[method_df["method"] == method].copy()
        if subset.empty:
            continue
        x = pd.to_numeric(subset[predictor], errors="coerce")
        y = pd.to_numeric(subset[outcome], errors="coerce")
        valid = pd.concat([x, y], axis=1).dropna()
        if valid.empty:
            continue
        ax.scatter(
            valid.iloc[:, 0],
            valid.iloc[:, 1],
            color=METHOD_COLORS[method],
            alpha=0.75,
            s=55,
            edgecolor="white",
            linewidth=0.5,
        )
        if len(valid) >= 2 and valid.iloc[:, 0].nunique() > 1:
            slope, intercept = np.polyfit(valid.iloc[:, 0], valid.iloc[:, 1], 1)
            xs = np.linspace(valid.iloc[:, 0].min(), valid.iloc[:, 0].max(), 100)
            ax.plot(
                xs, slope * xs + intercept, color=METHOD_COLORS[method], linewidth=2
            )
        _, _, spearman_r, _, n_obs = safe_corr(valid.iloc[:, 0], valid.iloc[:, 1])
        if pd.notna(spearman_r):
            text_lines.append(f"{method}: rho={spearman_r:.2f} (n={n_obs})")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if text_lines:
        ax.text(
            1.02,
            0.98,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            bbox={
                "facecolor": "white",
                "edgecolor": "#CCCCCC",
                "boxstyle": "round,pad=0.35",
            },
        )
    add_method_footer(fig)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_condition_best_method_summary(method_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_column in [
        "site_distance_group",
        "entropy_group",
        "sample_size_group",
        "metadata_shift_group",
        "calibration_group",
    ]:
        for group_value, subset in method_df.groupby(group_column):
            if group_value == "UNKNOWN":
                continue
            summary = (
                subset.groupby("method", as_index=False)
                .agg(
                    mean_ood_performance=("mean_ood_performance", "mean"),
                    mean_id_ood_gap=("id_ood_gap", "mean"),
                    mean_ece=("mean_ece", "mean"),
                    mean_std_ood=("std_ood_performance", "mean"),
                    n_sites=("target_site", "nunique"),
                )
                .sort_values(
                    ["mean_ood_performance", "mean_id_ood_gap"], ascending=[False, True]
                )
            )
            best_row = summary.iloc[0]
            interpretation = (
                f"{pretty_method(best_row['method'])} has the highest mean OOD performance "
                f"({best_row['mean_ood_performance']:.3f}) in this condition, with mean gap "
                f"{best_row['mean_id_ood_gap']:.3f} and mean ECE {best_row['mean_ece']:.3f} "
                f"across {int(best_row['n_sites'])} target sites."
            )
            rows.append(
                {
                    "condition": prettify_condition_name(
                        group_column, str(group_value)
                    ),
                    "condition_variable": group_column,
                    "condition_level": group_value,
                    "best_method": best_row["method"],
                    "best_method_label": pretty_method(best_row["method"]),
                    "interpretation": interpretation,
                }
            )
    summary_df = pd.DataFrame(rows)
    order_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    summary_df["sort_key"] = summary_df["condition_level"].map(order_map).fillna(9)
    return summary_df.sort_values(["condition_variable", "sort_key"]).drop(
        columns="sort_key"
    )


def build_outlier_cases(method_df: pd.DataFrame, site_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = method_df.groupby(["dataset_short", "target_site"])
    low_threshold = site_df["site_mean_ood_performance"].quantile(0.25)
    for (dataset_short, target_site), subset in grouped:
        dataset = subset["dataset"].iloc[0]
        target_display = subset["target_site_display"].iloc[0]
        site_distance_group = subset["site_distance_group"].iloc[0]
        metadata_shift_group = subset["metadata_shift_group"].iloc[0]
        entropy_group = subset["entropy_group"].iloc[0]
        baseline = subset[subset["method"] == "M0"]
        if not baseline.empty:
            baseline_ood = float(baseline["mean_ood_performance"].iloc[0])
            best_method = subset.sort_values(
                "mean_ood_performance", ascending=False
            ).iloc[0]["method"]
            if best_method == "M0" and (
                site_distance_group == "HIGH" or metadata_shift_group == "HIGH"
            ):
                rows.append(
                    {
                        "case_type": "baseline_surprisingly_good",
                        "dataset_short": dataset_short,
                        "dataset": dataset,
                        "target_site": target_site,
                        "target_site_display": target_display,
                        "method": "M0",
                        "mean_ood_performance": baseline_ood,
                        "delta_vs_m0": 0.0,
                        "id_ood_gap": float(baseline["id_ood_gap"].iloc[0]),
                        "site_distance_group": site_distance_group,
                        "metadata_shift_group": metadata_shift_group,
                        "entropy_group": entropy_group,
                    }
                )
            for method in ["M3", "M4"]:
                adaptive = subset[subset["method"] == method]
                if adaptive.empty:
                    continue
                delta = float(adaptive["mean_ood_performance"].iloc[0] - baseline_ood)
                if delta <= -0.05:
                    rows.append(
                        {
                            "case_type": "adaptation_fails",
                            "dataset_short": dataset_short,
                            "dataset": dataset,
                            "target_site": target_site,
                            "target_site_display": target_display,
                            "method": method,
                            "mean_ood_performance": float(
                                adaptive["mean_ood_performance"].iloc[0]
                            ),
                            "delta_vs_m0": delta,
                            "id_ood_gap": float(adaptive["id_ood_gap"].iloc[0]),
                            "site_distance_group": site_distance_group,
                            "metadata_shift_group": metadata_shift_group,
                            "entropy_group": entropy_group,
                        }
                    )
            m4 = subset[subset["method"] == "M4"]
            if not m4.empty:
                delta = float(m4["mean_ood_performance"].iloc[0] - baseline_ood)
                if delta >= 0.05:
                    rows.append(
                        {
                            "case_type": "m4_improves_drastically",
                            "dataset_short": dataset_short,
                            "dataset": dataset,
                            "target_site": target_site,
                            "target_site_display": target_display,
                            "method": "M4",
                            "mean_ood_performance": float(
                                m4["mean_ood_performance"].iloc[0]
                            ),
                            "delta_vs_m0": delta,
                            "id_ood_gap": float(m4["id_ood_gap"].iloc[0]),
                            "site_distance_group": site_distance_group,
                            "metadata_shift_group": metadata_shift_group,
                            "entropy_group": entropy_group,
                        }
                    )

        site_mean_ood = float(subset["mean_ood_performance"].mean())
        if site_mean_ood <= low_threshold:
            rows.append(
                {
                    "case_type": "all_methods_fail",
                    "dataset_short": dataset_short,
                    "dataset": dataset,
                    "target_site": target_site,
                    "target_site_display": target_display,
                    "method": "ALL",
                    "mean_ood_performance": site_mean_ood,
                    "delta_vs_m0": math.nan,
                    "id_ood_gap": float(subset["id_ood_gap"].mean()),
                    "site_distance_group": site_distance_group,
                    "metadata_shift_group": metadata_shift_group,
                    "entropy_group": entropy_group,
                }
            )
    outlier_df = pd.DataFrame(rows)
    if not outlier_df.empty:
        outlier_df = outlier_df.merge(
            site_df[
                [
                    "dataset_short",
                    "target_site",
                    "normalized_entropy",
                    "n_total_target",
                    "site_distance_composite",
                    "overall_metadata_shift_score",
                    "site_mean_ece",
                ]
            ],
            on=["dataset_short", "target_site"],
            how="left",
        )
    return outlier_df


def plot_outlier_scatterplots(
    method_df: pd.DataFrame, outlier_df: pd.DataFrame, output_path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.0))
    panels = [
        ("site_distance_composite", "mean_ood_performance", "Distance vs OOD"),
        ("normalized_entropy", "mean_ood_performance", "Entropy vs OOD"),
        ("overall_metadata_shift_score", "id_ood_gap", "Metadata shift vs gap"),
        ("n_total_target", "std_ood_performance", "Sample size vs variability"),
    ]
    summary = method_df.groupby(
        ["dataset_short", "target_site", "target_site_display"], as_index=False
    ).agg(
        mean_ood_performance=("mean_ood_performance", "mean"),
        std_ood_performance=("std_ood_performance", "mean"),
        id_ood_gap=("id_ood_gap", "mean"),
        normalized_entropy=("normalized_entropy", "first"),
        site_distance_composite=("site_distance_composite", "first"),
        overall_metadata_shift_score=("overall_metadata_shift_score", "first"),
        n_total_target=("n_total_target", "first"),
    )
    highlight_sites = (
        set(outlier_df["target_site"].tolist()) if not outlier_df.empty else set()
    )
    for ax, (x_col, y_col, title) in zip(axes.flatten(), panels, strict=False):
        ax.scatter(
            summary[x_col],
            summary[y_col],
            color="#9ecae1",
            s=58,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )
        for _, row in summary.iterrows():
            if row["target_site"] in highlight_sites:
                ax.scatter(
                    row[x_col],
                    row[y_col],
                    color="#d62728",
                    s=72,
                    edgecolor="#7f0000",
                    linewidth=0.6,
                )
                ax.text(
                    row[x_col] + 0.01,
                    row[y_col] + 0.01,
                    row["target_site_display"],
                    fontsize=8.5,
                    color="#333333",
                )
        ax.set_title(title)
        ax.set_xlabel(x_col.replace("_", " "))
        ax.set_ylabel(y_col.replace("_", " "))
    fig.suptitle("Method outlier cases", fontsize=16, y=0.98)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_summary_text(
    winner_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    site_df: pd.DataFrame,
    method_sensitivity_df: pd.DataFrame,
    regression_results_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    warnings: list[str],
) -> str:
    lines = ["Method robustness final summary", ""]

    if not winner_df.empty:
        top_ood = winner_df.sort_values("n_best_ood", ascending=False).iloc[0]
        top_gap = winner_df.sort_values("n_best_gap", ascending=False).iloc[0]
        lines.append(
            f"- Most frequent OOD winner: {pretty_method(top_ood['method'])} ({int(top_ood['n_best_ood'])} wins)."
        )
        lines.append(
            f"- Most frequent low-gap winner: {pretty_method(top_gap['method'])} ({int(top_gap['n_best_gap'])} wins)."
        )

    if not scores_df.empty:
        global_scores = scores_df[scores_df["scope"] == "global"].sort_values(
            "robustness_score", ascending=False
        )
        best_global = global_scores.iloc[0]
        lines.append(
            f"- Highest global robustness score: {pretty_method(best_global['method'])} ({best_global['robustness_score']:.2f})."
        )
        for scope in ["high_shift", "low_entropy", "small_sample"]:
            scope_df = scores_df[scores_df["scope"] == scope].sort_values(
                "robustness_score", ascending=False
            )
            if not scope_df.empty:
                best_scope = scope_df.iloc[0]
                lines.append(
                    f"- Best method in {scope.replace('_', ' ')} conditions: {pretty_method(best_scope['method'])} "
                    f"({best_scope['robustness_score']:.2f})."
                )

    if not condition_df.empty:
        high_shift = condition_df[condition_df["condition"] == "high metadata shift"]
        if not high_shift.empty:
            row = high_shift.iloc[0]
            lines.append(
                f"- Under high metadata shift, {row['best_method_label']} gives the best average OOD performance."
            )

    if not site_df.empty:
        hardest = site_df.sort_values(
            ["site_mean_ood_performance", "site_mean_id_ood_gap"],
            ascending=[True, False],
        ).head(5)
        lines.append(
            f"- Hardest target domains overall: {', '.join(hardest['target_site_display'].tolist())}."
        )

    if not method_sensitivity_df.empty:
        subset = method_sensitivity_df[
            (method_sensitivity_df["dataset_scope"] == "ALL")
            & (method_sensitivity_df["relationship"] == "site_distance_vs_ood")
        ].dropna(subset=["spearman_r"])
        if not subset.empty:
            subset = subset.assign(
                abs_r=lambda frame: frame["spearman_r"].abs()
            ).sort_values("abs_r")
            most_robust = subset.iloc[0]
            lines.append(
                f"- Lowest sensitivity of OOD performance to site distance: {pretty_method(most_robust['method'])} "
                f"(Spearman {most_robust['spearman_r']:.2f})."
            )

    if not regression_results_df.empty:
        best_model = (
            regression_results_df[
                regression_results_df["outcome"] == "mean_ood_performance"
            ]
            .sort_values("r2", ascending=False)
            .iloc[0]
        )
        lines.append(
            f"- Best explanatory model for mean OOD performance: {best_model['model_type']} "
            f"(R²={best_model['r2']:.2f}, MAE={best_model['mae']:.3f})."
        )

    if not outlier_df.empty:
        counts = outlier_df["case_type"].value_counts()
        top_case = counts.index[0]
        lines.append(
            f"- Most frequent outlier pattern: {top_case.replace('_', ' ')} ({int(counts.iloc[0])} cases)."
        )

    lines.extend(
        [
            "",
            "Interpretation focus:",
            "- robustness is conditional, not absolute;",
            "- the same method can excel in low-shift domains and lose its advantage in hard target sites;",
            "- trade-offs between raw OOD performance, calibration, gap and variability should be discussed explicitly.",
            "",
            "Limits:",
            "- results inherit the finite number of target sites available in each dataset;",
            "- sensitivity and regression analyses remain descriptive and should be interpreted as exploratory rather than causal.",
        ]
    )
    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"


def build_readme(found_inputs: list[str], warnings: list[str]) -> str:
    lines = [
        "Method robustness and condition analysis",
        "",
        "This folder integrates the existing UCI, TCGA and ABIDE analyses to study conditional cross-site robustness.",
        "",
        "Main blocks:",
        "- final master table across dataset x target site x model x metric x method;",
        "- best-method and winner-frequency analyses;",
        "- condition-based summaries for site distance, entropy, sample size, metadata shift and calibration;",
        "- method sensitivity, robustness scores and explanatory models;",
        "- hard-vs-easy target domain analysis and outlier method cases.",
        "",
        "How to interpret robustness:",
        "- high robustness means good OOD performance combined with low gap, low variability and low calibration error;",
        "- the condition tables show when a method becomes advantageous rather than who wins on average;",
        "- the sensitivity plots quantify whether a method degrades sharply as target domains become harder.",
        "",
        "Inputs found:",
        *[f"- {path}" for path in found_inputs],
    ]
    if warnings:
        lines.extend(["", "Warnings:", *[f"- {warning}" for warning in warnings]])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    apply_plot_style()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    master_df, warnings, found_inputs = build_master_table(args.input_root)
    master_df = add_composite_columns(master_df)

    site_df = build_site_summary(master_df)
    method_df = build_method_site_table(master_df, site_df)

    save_dataframe(
        master_df,
        output_dir / "method_robustness_master_table.csv",
        output_dir / "method_robustness_master_table.xlsx",
        sheet_name="master",
        index=False,
    )

    best_df = build_best_method_table(master_df)
    save_dataframe(
        best_df,
        output_dir / "best_method_by_site.csv",
        output_dir / "best_method_by_site.xlsx",
        sheet_name="best_method",
        index=False,
    )

    winner_df = build_winner_frequency_table(best_df)
    save_dataframe(
        winner_df,
        output_dir / "winner_frequency_table.csv",
        output_dir / "winner_frequency_table.xlsx",
        sheet_name="winner_frequency",
        index=False,
    )
    plot_winner_frequency_bars(winner_df, output_dir)

    distance_group_df = summarize_by_condition(method_df, "site_distance_group")
    entropy_group_df = summarize_by_condition(method_df, "entropy_group")
    sample_size_group_df = summarize_by_condition(method_df, "sample_size_group")
    metadata_shift_group_df = summarize_by_condition(method_df, "metadata_shift_group")
    save_dataframe(
        distance_group_df,
        output_dir / "performance_by_distance_group.csv",
        output_dir / "performance_by_distance_group.xlsx",
        sheet_name="distance_group",
        index=False,
    )
    save_dataframe(
        entropy_group_df,
        output_dir / "performance_by_entropy_group.csv",
        output_dir / "performance_by_entropy_group.xlsx",
        sheet_name="entropy_group",
        index=False,
    )
    save_dataframe(
        sample_size_group_df,
        output_dir / "performance_by_sample_size_group.csv",
        output_dir / "performance_by_sample_size_group.xlsx",
        sheet_name="sample_group",
        index=False,
    )
    save_dataframe(
        metadata_shift_group_df,
        output_dir / "performance_by_metadata_shift_group.csv",
        output_dir / "performance_by_metadata_shift_group.xlsx",
        sheet_name="shift_group",
        index=False,
    )

    sensitivity_df = build_method_sensitivity_table(method_df)
    save_dataframe(
        sensitivity_df,
        output_dir / "method_sensitivity_analysis.csv",
        output_dir / "method_sensitivity_analysis.xlsx",
        sheet_name="sensitivity",
        index=False,
    )

    scores_df = build_method_robustness_scores(method_df, sensitivity_df)
    save_dataframe(
        scores_df,
        output_dir / "method_robustness_scores.csv",
        output_dir / "method_robustness_scores.xlsx",
        sheet_name="robustness_scores",
        index=False,
    )
    plot_robustness_score_barplot(
        scores_df, output_dir / "robustness_score_barplot.png"
    )
    plot_robustness_radar(scores_df, output_dir / "robustness_score_radarplot.png")

    regression_results_df, regression_coefficients_df, prediction_frames = (
        fit_explanatory_models(site_df)
    )
    save_dataframe(
        regression_results_df,
        output_dir / "explanatory_models_results.csv",
        output_dir / "explanatory_models_results.xlsx",
        sheet_name="model_results",
        index=False,
    )
    save_dataframe(
        regression_coefficients_df,
        output_dir / "explanatory_models_coefficients.csv",
        output_dir / "explanatory_models_coefficients.xlsx",
        sheet_name="coefficients",
        index=False,
    )
    if "mean_ood_performance" in prediction_frames:
        plot_predicted_vs_observed(
            prediction_frames["mean_ood_performance"],
            output_dir / "predicted_vs_observed_ood.png",
            "Predicted vs observed mean OOD performance",
        )
    if "id_ood_gap" in prediction_frames:
        plot_predicted_vs_observed(
            prediction_frames["id_ood_gap"],
            output_dir / "predicted_vs_observed_gap.png",
            "Predicted vs observed ID-OOD gap",
        )

    site_df, hard_easy_df = build_hard_easy_table(site_df)
    save_dataframe(
        hard_easy_df,
        output_dir / "hard_vs_easy_domains.csv",
        output_dir / "hard_vs_easy_domains.xlsx",
        sheet_name="hard_easy",
        index=False,
    )
    plot_hard_easy_boxplots(site_df, output_dir)

    plot_method_relationship(
        method_df,
        "site_distance_composite",
        "mean_ood_performance",
        "Method sensitivity to site distance",
        "Site distance composite",
        "Mean OOD performance",
        output_dir / "method_distance_sensitivity.png",
    )
    plot_method_relationship(
        method_df,
        "normalized_entropy",
        "mean_ood_performance",
        "Method sensitivity to entropy",
        "Normalized entropy",
        "Mean OOD performance",
        output_dir / "method_entropy_sensitivity.png",
    )
    plot_method_relationship(
        method_df,
        "overall_metadata_shift_score",
        "id_ood_gap",
        "Method sensitivity to metadata shift",
        "Metadata shift score",
        "ID-OOD gap",
        output_dir / "method_metadata_shift_sensitivity.png",
    )
    plot_method_relationship(
        method_df,
        "n_total_target",
        "std_ood_performance",
        "Method variability vs sample size",
        "Target sample size",
        "OOD variability (std)",
        output_dir / "method_sample_size_variability.png",
    )

    condition_df = build_condition_best_method_summary(method_df)
    save_dataframe(
        condition_df,
        output_dir / "condition_best_method_summary.csv",
        output_dir / "condition_best_method_summary.xlsx",
        sheet_name="condition_best",
        index=False,
    )
    (output_dir / "condition_best_method_summary.txt").write_text(
        "\n".join(
            f"- {row['condition']}: {row['best_method_label']} -> {row['interpretation']}"
            for _, row in condition_df.iterrows()
        )
        + "\n",
        encoding="utf-8",
    )

    outlier_df = build_outlier_cases(method_df, site_df)
    save_dataframe(
        outlier_df,
        output_dir / "method_outlier_cases.csv",
        output_dir / "method_outlier_cases.xlsx",
        sheet_name="outliers",
        index=False,
    )
    plot_outlier_scatterplots(
        method_df, outlier_df, output_dir / "method_outlier_scatterplots.png"
    )

    pattern_corr_df = build_pattern_correlations(master_df)
    save_dataframe(
        pattern_corr_df,
        output_dir / "pattern_correlations.csv",
        output_dir / "pattern_correlations.xlsx",
        sheet_name="pattern_corr",
        index=False,
    )
    plot_pattern_correlation_heatmap(
        pattern_corr_df, output_dir / "pattern_correlation_heatmap.png"
    )

    final_summary = build_summary_text(
        winner_df,
        scores_df,
        condition_df,
        site_df,
        sensitivity_df,
        regression_results_df,
        outlier_df,
        warnings,
    )
    (output_dir / "method_robustness_final_summary.txt").write_text(
        final_summary, encoding="utf-8"
    )
    (output_dir / "README.txt").write_text(
        build_readme(found_inputs, warnings), encoding="utf-8"
    )

    metadata = {
        "n_master_rows": int(len(master_df)),
        "n_site_rows": int(len(site_df)),
        "n_method_site_rows": int(len(method_df)),
        "datasets": sorted(master_df["dataset_short"].dropna().unique().tolist()),
        "warnings": warnings,
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote method robustness analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
