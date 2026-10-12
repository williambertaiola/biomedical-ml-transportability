"""Post-experiment analysis: generate pattern discovery analysis."""

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
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.stats import pearsonr, spearmanr, zscore

import matplotlib

os.environ["MPLCONFIGDIR"] = str(
    Path(tempfile.gettempdir()) / "codex_pattern_discovery_mpl"
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:  # pragma: no cover - optional dependency
    import umap  # type: ignore

    HAS_UMAP = True
except Exception:  # pragma: no cover - optional dependency
    HAS_UMAP = False


METHOD_ORDER = ["M0", "M1", "M2", "M3", "M4"]
METHOD_FULL_LABELS = {
    "M0": "M0: Baseline",
    "M1": "M1: ComBat",
    "M2": "M2: Weighting",
    "M3": "M3: Transductive",
    "M4": "M4: ComBat + Transductive",
}
METHOD_LEGEND_TEXT = "M0: Baseline | M1: ComBat | M2: Weighting | M3: Transductive | M4: ComBat + Transductive"
METHOD_COLORS = {
    "M0": "#4C78A8",
    "M1": "#F58518",
    "M2": "#54A24B",
    "M3": "#E45756",
    "M4": "#72B7B2",
}
DISTANCE_COLUMNS = [
    "euclidean_centroid_distance",
    "mahalanobis_distance",
    "wasserstein_mean_distance",
    "jensen_shannon_mean_divergence",
    "mmd_rbf",
]
EXPLANATORY_COLUMNS = [
    "n_total_target",
    "n_source",
    "entropy",
    "normalized_entropy",
    *DISTANCE_COLUMNS,
    "ece",
]
OUTCOME_COLUMNS = ["mean_ood_performance", "std_ood_performance", "id_ood_gap"]
BUNDLED_EXCEL_PYTHON = Path(_configured_path("python_executable"))
RANDOM_STATE = 42
PRETTY_LABELS = {
    "mean_ood_performance": "OOD mean",
    "std_ood_performance": "OOD std",
    "id_ood_gap": "ID-OOD gap",
    "mean_id_ood_gap": "ID-OOD gap",
    "ece": "ECE",
    "entropy": "Entropy",
    "normalized_entropy": "Norm. entropy",
    "n_total_target": "N target",
    "n_source": "N source",
    "euclidean_centroid_distance": "Euclidean",
    "mahalanobis_distance": "Mahalanobis",
    "wasserstein_mean_distance": "Wasserstein",
    "jensen_shannon_mean_divergence": "Jensen-Shannon",
    "mmd_rbf": "MMD-RBF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pattern discovery analysis from existing transportability outputs."
    )
    parser.add_argument(
        "--input-root", type=Path, required=True, help="Dataset root, e.g. .../UCI"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where pattern discovery outputs will be written",
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


def pretty_label(text: str) -> str:
    return PRETTY_LABELS.get(text, text.replace("_", " "))


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

    temp_dir = Path(tempfile.mkdtemp(prefix="pattern_discovery_xlsx_"))
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


def save_dataframe(
    df: pd.DataFrame,
    csv_path: Path,
    xlsx_path: Path,
    *,
    sheet_name: str,
    index: bool = False,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=index)
    write_excel_with_fallback(df, xlsx_path, sheet_name=sheet_name, index=index)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def available_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in df.columns]


def safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float, int]:
    valid = pd.concat([safe_numeric(x), safe_numeric(y)], axis=1).dropna()
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


def ensure_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric_df = df.copy()
    for column in columns:
        if column in numeric_df.columns:
            numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce")
    return numeric_df


def load_optional_csv(
    path: Path,
    warnings: list[str],
    found_inputs: list[str],
    *,
    required_columns: list[str] | None = None,
) -> pd.DataFrame | None:
    if not path.exists():
        warnings.append(f"Missing optional input: {path}")
        return None
    df = pd.read_csv(path, low_memory=False)
    if required_columns:
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            warnings.append(
                f"Optional input {path.name} is missing columns: {', '.join(missing)}"
            )
            return None
    found_inputs.append(str(path))
    return df


def load_performance_table(
    input_root: Path, warnings: list[str], found_inputs: list[str]
) -> tuple[pd.DataFrame, str]:
    joined_path = input_root / "Site_distance" / "site_distance_vs_performance.csv"
    if joined_path.exists():
        found_inputs.append(str(joined_path))
        df = pd.read_csv(joined_path, low_memory=False)
        if "dataset" not in df.columns:
            df["dataset"] = input_root.name
        dataset_name = str(df["dataset"].iloc[0])
        base_columns = [
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
        ]
        for column in base_columns:
            if column not in df.columns:
                df[column] = math.nan
        df = df[
            base_columns
            + available_columns(df, DISTANCE_COLUMNS + ["n_source", "n_target"])
        ]
        df = df.rename(columns={"mean_ece": "ece"})
        return ensure_numeric_columns(
            df,
            [
                "mean_ood_performance",
                "std_ood_performance",
                "mean_id_performance",
                "id_ood_gap",
                "ece",
                "n_source",
                "n_target",
                *DISTANCE_COLUMNS,
            ],
        ), dataset_name
    raise RuntimeError(
        "Could not find site_distance_vs_performance.csv required to seed pattern discovery."
    )


def build_master_table(
    input_root: Path, warnings: list[str], found_inputs: list[str]
) -> tuple[pd.DataFrame, str]:
    performance_df, dataset_name = load_performance_table(
        input_root, warnings, found_inputs
    )

    sample_size_df = load_optional_csv(
        input_root / "Sample_size" / "sample_size_by_site.csv",
        warnings,
        found_inputs,
        required_columns=["Target site"],
    )
    entropy_df = load_optional_csv(
        input_root / "Entropy" / "entropy_summary_by_site.csv",
        warnings,
        found_inputs,
        required_columns=["target_site"],
    )
    site_distance_df = load_optional_csv(
        input_root / "Site_distance" / "site_distance_by_target.csv",
        warnings,
        found_inputs,
        required_columns=["target_site"],
    )

    heatmap_matrices_dir = input_root / "Heatmap_cluster" / "matrices"
    if heatmap_matrices_dir.exists():
        found_inputs.append(str(heatmap_matrices_dir))
    else:
        warnings.append(f"Missing optional input directory: {heatmap_matrices_dir}")

    master_df = performance_df.copy()

    if sample_size_df is not None:
        sample_df = sample_size_df.rename(
            columns={
                "Target site": "target_site",
                "N total target": "n_total_target",
                "N source": "n_source",
                "N train_1": "n_train_1",
                "N test_1": "n_test_1",
                "Target prevalence": "class_1_ratio",
            }
        ).copy()
        sample_df["class_1_ratio"] = safe_numeric(sample_df["class_1_ratio"])
        sample_df["class_0_ratio"] = 1.0 - sample_df["class_1_ratio"]
        sample_columns = [
            "target_site",
            "n_total_target",
            "n_source",
            "n_train_1",
            "n_test_1",
            "class_0_ratio",
            "class_1_ratio",
        ]
        sample_df = ensure_numeric_columns(
            sample_df, [column for column in sample_columns if column != "target_site"]
        )
        master_df = master_df.merge(
            sample_df[sample_columns],
            on="target_site",
            how="left",
            suffixes=("", "_sample"),
        )

    if entropy_df is not None:
        entropy_use = entropy_df.rename(
            columns={
                "full_target_entropy": "entropy",
                "full_target_normalized_entropy": "normalized_entropy",
            }
        )
        entropy_columns = [
            "target_site",
            "entropy",
            "normalized_entropy",
            "mean_train_1_entropy",
            "std_train_1_entropy",
            "mean_test_1_entropy",
            "std_test_1_entropy",
            "mean_source_entropy",
            "std_source_entropy",
        ]
        entropy_use = ensure_numeric_columns(
            entropy_use,
            [column for column in entropy_columns if column != "target_site"],
        )
        master_df = master_df.merge(
            entropy_use[available_columns(entropy_use, entropy_columns)],
            on="target_site",
            how="left",
        )

    if site_distance_df is not None:
        distance_columns = [
            "target_site",
            "n_source",
            "n_target",
            *DISTANCE_COLUMNS,
        ]
        distance_use = ensure_numeric_columns(
            site_distance_df,
            [column for column in distance_columns if column != "target_site"],
        )
        master_df = master_df.merge(
            distance_use[available_columns(distance_use, distance_columns)],
            on="target_site",
            how="left",
            suffixes=("", "_distance"),
        )

    for column in ["n_source_distance", "n_target_distance"]:
        if column in master_df.columns:
            base_column = column.replace("_distance", "")
            if (
                base_column not in master_df.columns
                or master_df[base_column].isna().all()
            ):
                master_df[base_column] = master_df[column]
            master_df = master_df.drop(columns=[column])

    preferred_columns = [
        "dataset",
        "target_site",
        "target_site_display",
        "model",
        "model_display",
        "method",
        "metric",
        "mean_ood_performance",
        "std_ood_performance",
        "mean_id_performance",
        "id_ood_gap",
        "ece",
        "n_total_target",
        "n_source",
        "n_train_1",
        "n_test_1",
        "class_0_ratio",
        "class_1_ratio",
        "entropy",
        "normalized_entropy",
        *DISTANCE_COLUMNS,
    ]
    for column in preferred_columns:
        if column not in master_df.columns:
            master_df[column] = math.nan

    numeric_columns = [
        column
        for column in preferred_columns
        if column
        not in {
            "dataset",
            "target_site",
            "target_site_display",
            "model",
            "model_display",
            "method",
            "metric",
        }
    ]
    master_df = ensure_numeric_columns(master_df, numeric_columns)
    master_df = master_df[preferred_columns]
    return master_df.sort_values(
        ["target_site", "model", "method", "metric"]
    ).reset_index(drop=True), dataset_name


def correlation_analysis(master_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subgroup_specs = [("overall", "all", master_df)]
    for metric in sorted(master_df["metric"].dropna().unique()):
        subgroup_specs.append(
            ("metric", str(metric), master_df[master_df["metric"] == metric])
        )
    for method in METHOD_ORDER:
        subset = master_df[master_df["method"] == method]
        if not subset.empty:
            subgroup_specs.append(("method", method, subset))
    for model in sorted(master_df["model"].dropna().unique()):
        subgroup_specs.append(
            ("model", str(model), master_df[master_df["model"] == model])
        )

    explanatory = [
        column for column in EXPLANATORY_COLUMNS if column in master_df.columns
    ]
    outcomes = [column for column in OUTCOME_COLUMNS if column in master_df.columns]

    for subgroup_type, subgroup_value, subset_df in subgroup_specs:
        for variable_x in explanatory:
            for variable_y in outcomes:
                pearson_r, pearson_p, spearman_r, spearman_p, n_obs = safe_corr(
                    subset_df[variable_x], subset_df[variable_y]
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "subgroup_type": subgroup_type,
                        "subgroup_value": subgroup_value,
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


def correlation_heatmap(correlation_df: pd.DataFrame, output_path: Path) -> None:
    subset = correlation_df[correlation_df["subgroup_type"] == "overall"].copy()
    matrix = subset.pivot(index="variable_x", columns="variable_y", values="spearman_r")
    if matrix.empty:
        return
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(
        figsize=(max(7, 1.2 * matrix.shape[1] + 3), max(6, 0.65 * matrix.shape[0] + 2))
    )
    image = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        [pretty_label(column) for column in matrix.columns.tolist()],
        rotation=15,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(
        [pretty_label(index) for index in matrix.index.tolist()], rotation=0
    )
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
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
    cbar.set_label("Spearman correlation")
    ax.set_title("Pattern discovery correlation heatmap", fontsize=16, pad=12)
    fig.tight_layout(rect=(0.02, 0.04, 1.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_site_level_table(master_df: pd.DataFrame) -> pd.DataFrame:
    site_level = (
        master_df.groupby(["dataset", "target_site"], as_index=False)
        .agg(
            mean_ood_performance=("mean_ood_performance", "mean"),
            std_ood_performance=("std_ood_performance", "mean"),
            mean_id_ood_gap=("id_ood_gap", "mean"),
            mean_ece=("ece", "mean"),
            n_total_target=("n_total_target", "first"),
            n_source=("n_source", "first"),
            n_train_1=("n_train_1", "first"),
            n_test_1=("n_test_1", "first"),
            class_0_ratio=("class_0_ratio", "first"),
            class_1_ratio=("class_1_ratio", "first"),
            entropy=("entropy", "first"),
            normalized_entropy=("normalized_entropy", "first"),
            euclidean_centroid_distance=("euclidean_centroid_distance", "first"),
            mahalanobis_distance=("mahalanobis_distance", "first"),
            wasserstein_mean_distance=("wasserstein_mean_distance", "first"),
            jensen_shannon_mean_divergence=("jensen_shannon_mean_divergence", "first"),
            mmd_rbf=("mmd_rbf", "first"),
        )
        .reset_index(drop=True)
    )

    distance_z = []
    for column in DISTANCE_COLUMNS:
        if column in site_level.columns:
            z = pd.Series(
                zscore(site_level[column], nan_policy="omit"), index=site_level.index
            )
            z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            distance_z.append(z)
    if distance_z:
        site_level["mean_site_distance"] = pd.concat(distance_z, axis=1).mean(axis=1)
    else:
        site_level["mean_site_distance"] = math.nan
    return site_level


def prepare_feature_matrix(
    site_level_df: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, float]]:
    numeric_df = site_level_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    medians: dict[str, float] = {}
    for column in numeric_df.columns:
        if numeric_df[column].isna().any():
            median_value = float(numeric_df[column].median())
            medians[column] = median_value
            numeric_df[column] = numeric_df[column].fillna(median_value)
    zscore_df = numeric_df.apply(
        lambda col: (
            pd.Series(zscore(col, nan_policy="omit"), index=col.index)
            if col.nunique(dropna=True) > 1
            else pd.Series(np.zeros(len(col)), index=col.index)
        ),
        axis=0,
    )
    zscore_df = zscore_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    zscore_df.index = site_level_df["target_site"]
    return zscore_df, medians


def plot_site_embedding(
    coords_df: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    x_column: str,
    y_column: str,
    color_column: str,
    size_column: str | None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = coords_df[color_column].astype(float)
    sizes = (
        90
        + 210
        * (
            (coords_df[size_column] - coords_df[size_column].min())
            / (coords_df[size_column].max() - coords_df[size_column].min() + 1e-9)
        )
        if size_column and size_column in coords_df.columns
        else pd.Series(140.0, index=coords_df.index)
    )
    scatter = ax.scatter(
        coords_df[x_column],
        coords_df[y_column],
        c=colors,
        s=sizes,
        cmap="RdYlGn_r",
        edgecolor="#333333",
        linewidth=0.8,
        alpha=0.92,
    )
    x_values = coords_df[x_column].astype(float)
    y_values = coords_df[y_column].astype(float)
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    x_pad = 0.08 * x_span
    y_pad = 0.08 * y_span
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    x_mid = float(np.median(x_values))
    y_mid = float(np.median(y_values))
    for _, row in coords_df.iterrows():
        dx = -0.035 * x_span if row[x_column] > x_mid else 0.035 * x_span
        dy = -0.03 * y_span if row[y_column] > y_mid else 0.03 * y_span
        ha = "right" if row[x_column] > x_mid else "left"
        va = "top" if row[y_column] > y_mid else "bottom"
        ax.text(
            row[x_column] + dx,
            row[y_column] + dy,
            row["target_site"],
            fontsize=9,
            ha=ha,
            va=va,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.68,
                "pad": 0.15,
            },
            clip_on=False,
        )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(pretty_label(color_column))
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(title, fontsize=15, pad=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_pca_umap(
    site_level_df: pd.DataFrame, output_dir: Path, warnings: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    feature_columns = [
        column
        for column in [
            "n_total_target",
            "normalized_entropy",
            *DISTANCE_COLUMNS,
            "mean_ood_performance",
            "std_ood_performance",
            "mean_id_ood_gap",
            "mean_ece",
        ]
        if column in site_level_df.columns and not site_level_df[column].isna().all()
    ]
    zscore_df, _ = prepare_feature_matrix(site_level_df, feature_columns)

    if zscore_df.shape[0] < 2 or zscore_df.shape[1] < 2:
        warnings.append("Not enough site-level dimensions to run PCA/UMAP.")
        empty_pca = site_level_df[["target_site"]].copy()
        return empty_pca, None, pd.DataFrame()

    pca = PCA(n_components=min(2, zscore_df.shape[1]), random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(zscore_df.to_numpy(dtype=float))
    pca_df = site_level_df[
        ["dataset", "target_site", "n_total_target", "mean_id_ood_gap"]
    ].copy()
    pca_df["PC1"] = pca_coords[:, 0]
    pca_df["PC2"] = pca_coords[:, 1] if pca_coords.shape[1] > 1 else 0.0

    explained_df = pd.DataFrame(
        {
            "component": [
                f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))
            ],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )

    plot_site_embedding(
        pca_df,
        output_path=output_dir / "site_pattern_pca.png",
        title="Site pattern PCA",
        x_column="PC1",
        y_column="PC2",
        color_column="mean_id_ood_gap",
        size_column="n_total_target",
    )

    umap_df: pd.DataFrame | None = None
    if HAS_UMAP and zscore_df.shape[0] >= 3:
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)  # type: ignore[attr-defined]
        umap_coords = reducer.fit_transform(zscore_df.to_numpy(dtype=float))
        umap_df = site_level_df[
            ["dataset", "target_site", "n_total_target", "mean_id_ood_gap"]
        ].copy()
        umap_df["UMAP1"] = umap_coords[:, 0]
        umap_df["UMAP2"] = umap_coords[:, 1]
        plot_site_embedding(
            umap_df,
            output_path=output_dir / "site_pattern_umap.png",
            title="Site pattern UMAP",
            x_column="UMAP1",
            y_column="UMAP2",
            color_column="mean_id_ood_gap",
            size_column="n_total_target",
        )
    else:
        warnings.append(
            "UMAP not available or not enough sites; only PCA was generated."
        )

    return pca_df, umap_df, explained_df


def plot_dendrogram(
    site_level_df: pd.DataFrame, feature_matrix: pd.DataFrame, output_path: Path
) -> tuple[np.ndarray, pd.DataFrame]:
    linkage_matrix = linkage(
        feature_matrix.to_numpy(dtype=float), method="average", metric="euclidean"
    )
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(site_level_df) + 5), 6))
    dendrogram(
        linkage_matrix,
        labels=site_level_df["target_site"].tolist(),
        leaf_rotation=0,
        leaf_font_size=10,
        ax=ax,
        color_threshold=None,
    )
    for line in ax.get_lines():
        line.set_color("black")
        line.set_linewidth(1.35)
    ax.set_title("Target site dendrogram", fontsize=15, pad=12)
    ax.set_ylabel("Hierarchical distance")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return linkage_matrix, feature_matrix


def run_site_clustering(
    site_level_df: pd.DataFrame, output_dir: Path, warnings: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_features = [
        column
        for column in [
            "n_total_target",
            "normalized_entropy",
            "mean_site_distance",
            "mean_ood_performance",
            "mean_id_ood_gap",
            "mean_ece",
        ]
        if column in site_level_df.columns and not site_level_df[column].isna().all()
    ]
    zscore_df, _ = prepare_feature_matrix(site_level_df, cluster_features)
    linkage_matrix, _ = plot_dendrogram(
        site_level_df, zscore_df, output_dir / "dendrogram_target_sites.png"
    )

    n_sites = len(site_level_df)
    n_clusters = 2 if n_sites >= 4 else 1
    cluster_labels = (
        fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
        if n_clusters > 1
        else np.ones(n_sites, dtype=int)
    )

    cluster_df = site_level_df[
        [
            "dataset",
            "target_site",
            "n_total_target",
            "normalized_entropy",
            "mean_site_distance",
            "mean_ood_performance",
            "mean_id_ood_gap",
            "mean_ece",
        ]
    ].copy()
    cluster_df["cluster_id"] = cluster_labels

    if n_sites >= 4:
        for k in [2, 3, 4]:
            if k < n_sites:
                km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
                cluster_df[f"kmeans_k{k}_cluster_id"] = (
                    km.fit_predict(zscore_df.to_numpy(dtype=float)) + 1
                )
    else:
        warnings.append("Too few sites for optional KMeans clustering.")

    cluster_summary = cluster_df.groupby("cluster_id", as_index=False).agg(
        n_sites=("target_site", "count"),
        mean_n_total_target=("n_total_target", "mean"),
        mean_normalized_entropy=("normalized_entropy", "mean"),
        mean_site_distance=("mean_site_distance", "mean"),
        mean_ood_performance=("mean_ood_performance", "mean"),
        mean_id_ood_gap=("mean_id_ood_gap", "mean"),
        mean_ece=("mean_ece", "mean"),
    )
    return cluster_df, cluster_summary


def boxplot_comparison(
    site_level_df: pd.DataFrame,
    *,
    category_column: str,
    value_column: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    valid_df = site_level_df[[category_column, value_column]].dropna()
    groups = [
        valid_df.loc[valid_df[category_column] == label, value_column].tolist()
        for label in ["easy", "hard"]
    ]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    bp = ax.boxplot(groups, tick_labels=["Easy", "Hard"], patch_artist=True)
    box_colors = ["#74C476", "#E45756"]
    for patch, color in zip(bp["boxes"], box_colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel(y_label)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def hard_site_analysis(site_level_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    work_df = site_level_df.copy()
    ood_median = float(work_df["mean_ood_performance"].median())
    gap_median = float(work_df["mean_id_ood_gap"].median())
    work_df["hard_by_ood"] = np.where(
        work_df["mean_ood_performance"] < ood_median, "hard", "easy"
    )
    work_df["hard_by_gap"] = np.where(
        work_df["mean_id_ood_gap"] > gap_median, "hard", "easy"
    )

    summary_rows: list[dict[str, Any]] = []
    for definition, column in [
        ("hard_by_ood", "hard_by_ood"),
        ("hard_by_gap", "hard_by_gap"),
    ]:
        grouped = work_df.groupby(column)
        for group_name, group_df in grouped:
            summary_rows.append(
                {
                    "definition": definition,
                    "group": group_name,
                    "n_sites": len(group_df),
                    "mean_normalized_entropy": group_df["normalized_entropy"].mean(),
                    "mean_mean_site_distance": group_df["mean_site_distance"].mean(),
                    "mean_n_total_target": group_df["n_total_target"].mean(),
                    "mean_mean_ece": group_df["mean_ece"].mean(),
                    "sites": "|".join(group_df["target_site"].tolist()),
                }
            )

    boxplot_comparison(
        work_df.rename(columns={"hard_by_ood": "difficulty_group"}),
        category_column="difficulty_group",
        value_column="normalized_entropy",
        output_path=output_dir / "boxplot_hard_vs_easy_entropy.png",
        title="Hard vs easy sites by OOD performance | Entropy",
        y_label="Normalized entropy",
    )
    boxplot_comparison(
        work_df.rename(columns={"hard_by_ood": "difficulty_group"}),
        category_column="difficulty_group",
        value_column="mean_site_distance",
        output_path=output_dir / "boxplot_hard_vs_easy_site_distance.png",
        title="Hard vs easy sites by OOD performance | Composite site distance",
        y_label="Composite site distance",
    )
    boxplot_comparison(
        work_df.rename(columns={"hard_by_ood": "difficulty_group"}),
        category_column="difficulty_group",
        value_column="n_total_target",
        output_path=output_dir / "boxplot_hard_vs_easy_sample_size.png",
        title="Hard vs easy sites by OOD performance | Sample size",
        y_label="N total target",
    )
    if "mean_ece" in work_df.columns and not work_df["mean_ece"].isna().all():
        boxplot_comparison(
            work_df.rename(columns={"hard_by_ood": "difficulty_group"}),
            category_column="difficulty_group",
            value_column="mean_ece",
            output_path=output_dir / "boxplot_hard_vs_easy_ece.png",
            title="Hard vs easy sites by OOD performance | ECE",
            y_label="Mean ECE",
        )

    return pd.DataFrame(summary_rows)


def run_regression_analysis(
    site_level_df: pd.DataFrame, output_dir: Path, warnings: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictor_columns = [
        column
        for column in [
            "normalized_entropy",
            "n_total_target",
            *DISTANCE_COLUMNS,
            "mean_ece",
        ]
        if column in site_level_df.columns and not site_level_df[column].isna().all()
    ]
    zscore_df, _ = prepare_feature_matrix(site_level_df, predictor_columns)
    n_obs = len(site_level_df)
    if n_obs < 6:
        warnings.append(
            "Regression analysis is exploratory only because the number of target sites is very small."
        )

    results_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    model_specs = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
    }
    target_map = {
        "mean_ood_performance": "predicted_vs_observed_ood_performance.png",
        "mean_id_ood_gap": "predicted_vs_observed_id_ood_gap.png",
        "std_ood_performance": "predicted_vs_observed_std_ood_performance.png",
    }

    for target_column, figure_name in target_map.items():
        if (
            target_column not in site_level_df.columns
            or site_level_df[target_column].isna().all()
        ):
            continue
        y = safe_numeric(site_level_df[target_column]).to_numpy(dtype=float)
        valid_mask = ~np.isnan(y)
        X = zscore_df.loc[valid_mask, :]
        y_valid = y[valid_mask]
        if len(y_valid) < 3:
            warnings.append(
                f"Skipping regression for {target_column}: fewer than 3 observations."
            )
            continue

        pred_plot_df = pd.DataFrame(
            {
                "target_site": site_level_df.loc[valid_mask, "target_site"].tolist(),
                "observed": y_valid,
            }
        )
        fig, ax = plt.subplots(figsize=(6.8, 5.8))
        color_map = {"LinearRegression": "#4C78A8", "Ridge": "#F58518"}

        for model_name, estimator in model_specs.items():
            estimator.fit(X, y_valid)
            y_pred = estimator.predict(X)
            mae = mean_absolute_error(y_valid, y_pred)
            rmse = math.sqrt(mean_squared_error(y_valid, y_pred))
            r2 = r2_score(y_valid, y_pred) if len(y_valid) > 1 else math.nan
            intercept = float(estimator.intercept_)
            results_rows.append(
                {
                    "dataset": site_level_df["dataset"].iloc[0],
                    "target_variable": target_column,
                    "model_type": model_name,
                    "n_observations": len(y_valid),
                    "intercept": intercept,
                    "r2": r2,
                    "mae": mae,
                    "rmse": rmse,
                }
            )
            coefficients = (
                estimator.coef_.ravel().tolist()
                if np.ndim(estimator.coef_) > 0
                else [float(estimator.coef_)]
            )
            for predictor_name, coefficient in zip(
                X.columns.tolist(), coefficients, strict=False
            ):
                coef_rows.append(
                    {
                        "dataset": site_level_df["dataset"].iloc[0],
                        "target_variable": target_column,
                        "model_type": model_name,
                        "predictor": predictor_name,
                        "coefficient": float(coefficient),
                    }
                )
            ax.scatter(
                y_valid,
                y_pred,
                s=80,
                alpha=0.85,
                label=model_name,
                color=color_map[model_name],
                edgecolor="#333333",
            )

        min_value = float(min(y_valid.min(), min(pred_plot_df["observed"])))
        max_value = float(max(y_valid.max(), max(pred_plot_df["observed"])))
        ax.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--",
            color="black",
            linewidth=1.2,
        )
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title(target_column.replace("_", " ").title(), fontsize=14, pad=12)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / figure_name, dpi=220)
        plt.close(fig)

    return pd.DataFrame(results_rows), pd.DataFrame(coef_rows)


def barplot_metric(
    df: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    ordered_df = df.set_index("method").reindex(METHOD_ORDER).reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(ordered_df))
    values = ordered_df[value_column].to_numpy(dtype=float)
    colors = [METHOD_COLORS.get(method, "#999999") for method in ordered_df["method"]]
    ax.bar(x, values, color=colors, edgecolor="#333333", linewidth=0.8)
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    finite_values = values[np.isfinite(values)]
    if finite_values.size:
        y_min = float(np.min(finite_values))
        y_max = float(np.max(finite_values))
        y_span = max(y_max - y_min, 0.1)
        ax.set_ylim(y_min - 0.22 * y_span - 0.02, y_max + 0.18 * y_span + 0.02)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_df["method"].tolist())
    ax.tick_params(axis="x", pad=6)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=14, pad=12)
    y0, y1 = ax.get_ylim()
    label_offset = 0.045 * (y1 - y0)
    for idx, value in enumerate(values):
        if np.isnan(value):
            continue
        offset = label_offset if value >= 0 else -label_offset
        ax.text(
            idx,
            value + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
            clip_on=False,
        )
    fig.text(0.5, 0.02, METHOD_LEGEND_TEXT, ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_method_robustness(master_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        subset = master_df[master_df["method"] == method].copy()
        if subset.empty:
            continue
        if "mean_site_distance" not in subset.columns:
            subset["mean_site_distance"] = (
                subset[DISTANCE_COLUMNS]
                .apply(pd.to_numeric, errors="coerce")
                .apply(lambda row: np.nanmean(row.to_numpy(dtype=float)), axis=1)
            )
        distance_ood = safe_corr(
            subset["mean_site_distance"], subset["mean_ood_performance"]
        )
        distance_gap = safe_corr(subset["mean_site_distance"], subset["id_ood_gap"])
        entropy_ood = safe_corr(
            subset["normalized_entropy"], subset["mean_ood_performance"]
        )
        size_std = safe_corr(subset["n_total_target"], subset["std_ood_performance"])

        rows.append(
            {
                "dataset": subset["dataset"].iloc[0],
                "method": method,
                "distance_vs_ood_pearson_r": distance_ood[0],
                "distance_vs_ood_spearman_r": distance_ood[2],
                "distance_vs_gap_pearson_r": distance_gap[0],
                "distance_vs_gap_spearman_r": distance_gap[2],
                "entropy_vs_ood_pearson_r": entropy_ood[0],
                "entropy_vs_ood_spearman_r": entropy_ood[2],
                "sample_size_vs_std_pearson_r": size_std[0],
                "sample_size_vs_std_spearman_r": size_std[2],
                "n_obs_distance_ood": distance_ood[4],
                "n_obs_distance_gap": distance_gap[4],
                "n_obs_entropy_ood": entropy_ood[4],
                "n_obs_size_std": size_std[4],
            }
        )

    robustness_df = pd.DataFrame(rows)
    if robustness_df.empty:
        return robustness_df

    barplot_metric(
        robustness_df,
        value_column="distance_vs_ood_spearman_r",
        title="Method robustness | Distance sensitivity vs OOD performance",
        y_label="Spearman(distance, OOD performance)",
        output_path=output_dir / "method_robustness_distance_sensitivity.png",
    )
    barplot_metric(
        robustness_df,
        value_column="entropy_vs_ood_spearman_r",
        title="Method robustness | Entropy sensitivity vs OOD performance",
        y_label="Spearman(entropy, OOD performance)",
        output_path=output_dir / "method_robustness_entropy_sensitivity.png",
    )
    barplot_metric(
        robustness_df,
        value_column="distance_vs_gap_spearman_r",
        title="Method robustness | Distance sensitivity vs ID-OOD gap",
        y_label="Spearman(distance, ID-OOD gap)",
        output_path=output_dir / "method_robustness_gap_sensitivity.png",
    )
    return robustness_df


def annotate_points(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    highlighted_sites: set[str] | None = None,
) -> None:
    if df.empty:
        return
    highlighted_sites = highlighted_sites or set()
    x_values = df[x_column].astype(float)
    y_values = df[y_column].astype(float)
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    x_mid = float(np.median(x_values))
    y_mid = float(np.median(y_values))
    for _, row in df.iterrows():
        dx = -0.03 * x_span if row[x_column] > x_mid else 0.03 * x_span
        dy = -0.025 * y_span if row[y_column] > y_mid else 0.025 * y_span
        ha = "right" if row[x_column] > x_mid else "left"
        va = "top" if row[y_column] > y_mid else "bottom"
        is_highlighted = str(row["target_site"]) in highlighted_sites
        ax.text(
            row[x_column] + dx,
            row[y_column] + dy,
            row["target_site"],
            fontsize=9.5 if is_highlighted else 9,
            ha=ha,
            va=va,
            color="#111111" if is_highlighted else "#333333",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78 if is_highlighted else 0.68,
                "pad": 0.15,
            },
            clip_on=False,
        )


def scatter_outliers(
    df: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    outlier_mask: pd.Series,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(
        df[x_column],
        df[y_column],
        s=80,
        color="#9ECAE1",
        edgecolor="#333333",
        alpha=0.8,
    )
    highlighted = df[outlier_mask]
    ax.scatter(
        highlighted[x_column],
        highlighted[y_column],
        s=110,
        color="#E45756",
        edgecolor="#333333",
        alpha=0.95,
    )
    x_values = df[x_column].astype(float)
    y_values = df[y_column].astype(float)
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    ax.set_xlim(x_min - 0.05 * x_span, x_max + 0.05 * x_span)
    ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.05 * y_span)
    annotate_points(
        ax,
        df,
        x_column,
        y_column,
        highlighted_sites=set(highlighted["target_site"].astype(str).tolist()),
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=14, pad=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_outlier_analysis(site_level_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    work_df = site_level_df.copy()
    for column in [
        "mean_site_distance",
        "mean_ood_performance",
        "mean_id_ood_gap",
        "normalized_entropy",
        "std_ood_performance",
        "mean_ece",
    ]:
        if column in work_df.columns:
            values = safe_numeric(work_df[column])
            z_values = (
                pd.Series(zscore(values, nan_policy="omit"), index=work_df.index)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            work_df[f"{column}_z"] = z_values

    work_df["flag_high_distance_good_ood"] = (
        work_df["mean_site_distance_z"] >= 1.0
    ) & (work_df["mean_ood_performance_z"] >= 0.5)
    work_df["flag_low_distance_bad_ood"] = (work_df["mean_site_distance_z"] <= -1.0) & (
        work_df["mean_ood_performance_z"] <= -0.5
    )
    work_df["flag_high_entropy_unstable"] = (work_df["normalized_entropy_z"] >= 0.5) & (
        work_df["std_ood_performance_z"] >= 1.0
    )
    work_df["flag_low_entropy_good_ood"] = (work_df["normalized_entropy_z"] <= -1.0) & (
        work_df["mean_ood_performance_z"] >= 0.5
    )
    work_df["flag_large_gap"] = work_df["mean_id_ood_gap_z"] >= 1.0
    if "mean_ece_z" in work_df.columns:
        work_df["flag_high_ece"] = work_df["mean_ece_z"] >= 1.0
    else:
        work_df["flag_high_ece"] = False

    flag_columns = [column for column in work_df.columns if column.startswith("flag_")]
    work_df["outlier_score"] = work_df[flag_columns].sum(axis=1)
    outlier_df = work_df[(work_df["outlier_score"] > 0)].copy()

    outlier_mask = work_df["outlier_score"] > 0
    scatter_outliers(
        work_df,
        x_column="mean_site_distance",
        y_column="mean_ood_performance",
        outlier_mask=outlier_mask,
        output_path=output_dir / "outlier_scatter_distance_vs_ood.png",
        title="Outlier analysis | Distance vs OOD performance",
        x_label="Composite site distance",
        y_label="Mean OOD performance",
    )
    scatter_outliers(
        work_df,
        x_column="normalized_entropy",
        y_column="mean_ood_performance",
        outlier_mask=outlier_mask,
        output_path=output_dir / "outlier_scatter_entropy_vs_ood.png",
        title="Outlier analysis | Entropy vs OOD performance",
        x_label="Normalized entropy",
        y_label="Mean OOD performance",
    )
    scatter_outliers(
        work_df,
        x_column="mean_site_distance",
        y_column="mean_id_ood_gap",
        outlier_mask=outlier_mask,
        output_path=output_dir / "outlier_scatter_distance_vs_gap.png",
        title="Outlier analysis | Distance vs ID-OOD gap",
        x_label="Composite site distance",
        y_label="Mean ID-OOD gap",
    )
    return outlier_df


def summary_text(
    *,
    correlation_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    hard_summary_df: pd.DataFrame,
    regression_results_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    warnings: list[str],
) -> str:
    lines = ["Pattern discovery summary", ""]

    overall_corr = correlation_df[correlation_df["subgroup_type"] == "overall"].copy()
    for outcome in ["mean_ood_performance", "id_ood_gap", "std_ood_performance"]:
        outcome_corr = overall_corr[overall_corr["variable_y"] == outcome].copy()
        outcome_corr = outcome_corr.dropna(subset=["spearman_r"])
        if not outcome_corr.empty:
            top_row = outcome_corr.iloc[outcome_corr["spearman_r"].abs().argmax()]
            lines.append(
                f"- Strongest overall association with {outcome}: {top_row['variable_x']} "
                f"(Spearman {top_row['spearman_r']:.3f}, n={int(top_row['n_obs'])})."
            )

    lines.append("")
    lines.append("Cluster signals:")
    if not cluster_summary_df.empty:
        for _, row in cluster_summary_df.iterrows():
            lines.append(
                f"- Cluster {int(row['cluster_id'])}: {int(row['n_sites'])} sites, "
                f"mean OOD={row['mean_ood_performance']:.3f}, "
                f"mean gap={row['mean_id_ood_gap']:.3f}, "
                f"mean entropy={row['mean_normalized_entropy']:.3f}."
            )

    lines.append("")
    lines.append("Hard-site signals:")
    if not hard_summary_df.empty:
        for _, row in hard_summary_df.iterrows():
            lines.append(
                f"- {row['definition']} / {row['group']}: n={int(row['n_sites'])}, "
                f"mean entropy={row['mean_normalized_entropy']:.3f}, "
                f"mean distance={row['mean_mean_site_distance']:.3f}, "
                f"sites={row['sites']}."
            )

    lines.append("")
    lines.append("Method robustness:")
    if not robustness_df.empty:
        ranked = robustness_df.copy()
        ranked["robustness_score"] = (
            ranked["distance_vs_ood_spearman_r"].abs().fillna(9.0)
            + ranked["distance_vs_gap_spearman_r"].abs().fillna(9.0)
            + ranked["sample_size_vs_std_spearman_r"].abs().fillna(9.0)
        )
        ranked = ranked.sort_values("robustness_score")
        best_row = ranked.iloc[0]
        lines.append(
            f"- Most stable exploratory method signal: {best_row['method']} "
            f"(robustness score {best_row['robustness_score']:.3f})."
        )

    lines.append("")
    lines.append("Outliers:")
    if outlier_df.empty:
        lines.append(
            "- No strong site-level outliers were flagged by the current z-score rules."
        )
    else:
        for _, row in outlier_df.iterrows():
            flag_names = [
                column.replace("flag_", "")
                for column in outlier_df.columns
                if column.startswith("flag_") and bool(row[column])
            ]
            lines.append(f"- {row['target_site']}: {', '.join(flag_names)}.")

    lines.append("")
    lines.append("Regression notes:")
    if regression_results_df.empty:
        lines.append("- Explanatory regressions were not stable enough to estimate.")
    else:
        for _, row in regression_results_df.iterrows():
            lines.append(
                f"- {row['model_type']} on {row['target_variable']}: "
                f"R²={row['r2']:.3f}, MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}, n={int(row['n_observations'])}."
            )

    if warnings:
        lines.append("")
        lines.append("Warnings / limitations:")
        for warning in warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def write_readme(
    output_dir: Path,
    *,
    dataset_name: str,
    found_inputs: list[str],
    warnings: list[str],
    has_umap: bool,
) -> None:
    found_text = (
        "\n".join(f"- {item}" for item in found_inputs) if found_inputs else "- none"
    )
    warning_text = (
        "\n".join(f"- {warning}" for warning in warnings) if warnings else "- none"
    )
    readme = f"""Pattern discovery analysis outputs for {dataset_name}

This folder integrates the already generated cross-site results into a single descriptive pattern-discovery layer.

What was done:
- built a master table at the level dataset | target_site | model | method | metric;
- joined OOD performance, bootstrap variability, ID-OOD gap, sample size, entropy, site distance and ECE;
- computed global and subgroup correlation analyses;
- created a site-level PCA embedding{" and UMAP embedding" if has_umap else ""};
- assigned hierarchical clusters to target sites and summarized each cluster;
- compared hard vs easy sites;
- fit simple exploratory regressions for OOD performance, gap and variability;
- summarized method robustness;
- flagged site-level outliers.

Found inputs:
{found_text}

Warnings:
{warning_text}

Interpretation notes:
- the master table is the central output and should be used as the starting point for further analyses.
- site-level analyses are exploratory when the number of target sites is small.
- composite site distance is the mean z-scored value across the available distance metrics.
- method robustness figures should be read as sensitivity diagnostics, not as formal causal evidence.

No original pipeline files or existing figures were modified.
"""
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_plot_style()

    warnings: list[str] = []
    found_inputs: list[str] = []

    master_df, dataset_name = build_master_table(input_root, warnings, found_inputs)
    save_dataframe(
        master_df,
        output_dir / "pattern_discovery_master_table.csv",
        output_dir / "pattern_discovery_master_table.xlsx",
        sheet_name="master_table",
    )

    correlation_df = correlation_analysis(master_df, dataset_name)
    save_dataframe(
        correlation_df,
        output_dir / "pattern_correlations.csv",
        output_dir / "pattern_correlations.xlsx",
        sheet_name="correlations",
    )
    correlation_heatmap(correlation_df, output_dir / "pattern_correlation_heatmap.png")

    site_level_df = build_site_level_table(master_df)

    pca_df, umap_df, explained_df = run_pca_umap(site_level_df, output_dir, warnings)
    if not pca_df.empty:
        pca_df.to_csv(output_dir / "site_pattern_pca_coordinates.csv", index=False)
        explained_df.to_csv(output_dir / "pca_explained_variance.csv", index=False)
    if umap_df is not None:
        umap_df.to_csv(output_dir / "site_pattern_umap_coordinates.csv", index=False)

    cluster_df, cluster_summary_df = run_site_clustering(
        site_level_df, output_dir, warnings
    )
    save_dataframe(
        cluster_df,
        output_dir / "site_clusters.csv",
        output_dir / "site_clusters.xlsx",
        sheet_name="site_clusters",
    )
    save_dataframe(
        cluster_summary_df,
        output_dir / "cluster_summary.csv",
        output_dir / "cluster_summary.xlsx",
        sheet_name="cluster_summary",
    )

    hard_summary_df = hard_site_analysis(site_level_df, output_dir)
    save_dataframe(
        hard_summary_df,
        output_dir / "hard_site_summary.csv",
        output_dir / "hard_site_summary.xlsx",
        sheet_name="hard_sites",
    )

    regression_results_df, regression_coef_df = run_regression_analysis(
        site_level_df, output_dir, warnings
    )
    save_dataframe(
        regression_results_df,
        output_dir / "explanatory_regression_results.csv",
        output_dir / "explanatory_regression_results.xlsx",
        sheet_name="regression_results",
    )
    save_dataframe(
        regression_coef_df,
        output_dir / "explanatory_regression_coefficients.csv",
        output_dir / "explanatory_regression_coefficients.xlsx",
        sheet_name="regression_coef",
    )

    robustness_df = run_method_robustness(master_df, output_dir)
    save_dataframe(
        robustness_df,
        output_dir / "method_robustness_summary.csv",
        output_dir / "method_robustness_summary.xlsx",
        sheet_name="method_robustness",
    )

    outlier_df = run_outlier_analysis(site_level_df, output_dir)
    save_dataframe(
        outlier_df,
        output_dir / "outlier_sites.csv",
        output_dir / "outlier_sites.xlsx",
        sheet_name="outliers",
    )

    summary = summary_text(
        correlation_df=correlation_df,
        cluster_df=cluster_df,
        cluster_summary_df=cluster_summary_df,
        hard_summary_df=hard_summary_df,
        regression_results_df=regression_results_df,
        robustness_df=robustness_df,
        outlier_df=outlier_df,
        warnings=warnings,
    )
    (output_dir / "pattern_discovery_summary.txt").write_text(summary, encoding="utf-8")

    write_readme(
        output_dir,
        dataset_name=dataset_name,
        found_inputs=found_inputs,
        warnings=warnings,
        has_umap=umap_df is not None,
    )

    print(f"Wrote pattern discovery analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
