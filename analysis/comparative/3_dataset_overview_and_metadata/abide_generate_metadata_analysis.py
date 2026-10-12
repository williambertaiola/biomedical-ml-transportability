"""Post-experiment analysis: abide generate metadata analysis."""

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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import jensenshannon
from scipy.stats import (
    chi2_contingency,
    f_oneway,
    kruskal,
    ks_2samp,
    mannwhitneyu,
    pearsonr,
    spearmanr,
    ttest_ind,
    wasserstein_distance,
    zscore,
)

import matplotlib

os.environ["MPLCONFIGDIR"] = str(
    Path(tempfile.gettempdir()) / "codex_metadata_analysis_mpl"
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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
METHOD_LEGEND_TEXT = "M0: Baseline | M1: ComBat | M2: Weighting | M3: Transductive | M4: ComBat + Transductive"
RANDOM_STATE = 42
MAX_METADATA_MISSING_RATE = 0.40
DATASET_SHORT_LABEL = "ABIDE"
DATASET_FULL_LABEL = "ABIDE II Core Multisite (strict phenotype)"
GENE_PREFIX = "__NO_GENE_PREFIX__"
TOP_GENES_FOR_SITE_PREDICTABILITY = 0
DIRECT_SITE_METADATA_COLUMNS = {"site"}
BUNDLED_EXCEL_PYTHON = Path(_configured_path("python_executable"))
SITE_COLUMN_CANDIDATES = [
    "domain_id",
    "site",
    "site_id",
    "domain",
    "center",
    "tissue_source_site",
]
TARGET_COLUMN_CANDIDATES = [
    "target",
    "label",
    "class",
    "disease_level_code",
    "stage_binary",
]
ID_COLUMN_KEYWORDS = ("subject", "patient", "sample", "id")
PRETTY_LABELS = {
    "target_site_display": "Target site",
    "age_at_scan": "Age at scan",
    "sex_code": "Sex",
    "handedness_category_code": "Handedness",
    "eye_status_at_scan_code": "Eye status",
    "fiq": "Full-scale IQ",
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain",
    "trestbps": "Resting BP",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate",
    "exang": "Exercise angina",
    "oldpeak": "ST depression",
    "slope": "ST slope",
    "ca": "Major vessels",
    "thal": "Thalassemia",
    "mean_ood_performance": "OOD mean",
    "std_ood_performance": "OOD std",
    "id_ood_gap": "ID-OOD gap",
    "mean_ece": "ECE",
    "ece": "ECE",
    "normalized_entropy": "Norm. entropy",
    "n_total_target": "N target",
    "n_source": "N source",
    "overall_metadata_shift_score": "Metadata shift score",
    "mean_continuous_metadata_shift": "Continuous shift",
    "mean_categorical_metadata_shift": "Categorical shift",
    "euclidean_centroid_distance": "Euclidean",
    "mahalanobis_distance": "Mahalanobis",
    "wasserstein_mean_distance": "Wasserstein",
    "jensen_shannon_mean_divergence": "Jensen-Shannon",
    "mmd_rbf": "MMD-RBF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate metadata analysis from existing transportability outputs."
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


def slugify(text: str) -> str:
    pieces = []
    for char in text.lower():
        if char.isalnum():
            pieces.append(char)
        else:
            if not pieces or pieces[-1] != "_":
                pieces.append("_")
    return "".join(pieces).strip("_")


def pretty_label(text: str) -> str:
    return PRETTY_LABELS.get(text, text.replace("_", " "))


def normalize_dataset_key(value: Any) -> str:
    text = str(value or "").lower()
    return "".join(char for char in text if char.isalnum())


def sort_category_labels(categories: list[str]) -> list[str]:
    def _key(value: str) -> tuple[int, float | str]:
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, str(value))

    return sorted(categories, key=_key)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_mode(series: pd.Series) -> Any:
    mode = series.mode(dropna=True)
    if mode.empty:
        return np.nan
    return mode.iloc[0]


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
                "Excel writing failed because openpyxl is unavailable and bundled runtime was not found."
            )

    temp_dir = Path(tempfile.mkdtemp(prefix="metadata_analysis_xlsx_"))
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


def minmax_scale(series: pd.Series) -> pd.Series:
    numeric = safe_numeric(series)
    if numeric.dropna().empty:
        return pd.Series(np.nan, index=series.index)
    minimum = numeric.min()
    maximum = numeric.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(0.5, index=series.index)
    return (numeric - minimum) / (maximum - minimum)


def cramers_v_from_table(table: pd.DataFrame | np.ndarray) -> float:
    array = np.asarray(table, dtype=float)
    n = array.sum()
    if n <= 0:
        return math.nan
    chi2, _, _, _ = chi2_contingency(array)
    r, k = array.shape
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return math.nan
    return float(math.sqrt(phi2corr / denom))


def cohen_d(source: pd.Series, target: pd.Series) -> float:
    source_values = safe_numeric(source).dropna().to_numpy(dtype=float)
    target_values = safe_numeric(target).dropna().to_numpy(dtype=float)
    if len(source_values) < 2 or len(target_values) < 2:
        return math.nan
    source_std = source_values.std(ddof=1)
    target_std = target_values.std(ddof=1)
    pooled_var = (
        (len(source_values) - 1) * source_std**2
        + (len(target_values) - 1) * target_std**2
    ) / (len(source_values) + len(target_values) - 2)
    if pooled_var <= 0:
        return math.nan
    return float((target_values.mean() - source_values.mean()) / math.sqrt(pooled_var))


def categorical_entropy(series: pd.Series) -> float:
    counts = series.dropna().astype(str).value_counts(normalize=True)
    if counts.empty:
        return math.nan
    values = counts.to_numpy(dtype=float)
    return float(-(values * np.log2(values + 1e-12)).sum())


def js_divergence_from_series(source: pd.Series, target: pd.Series) -> float:
    source_counts = source.dropna().astype(str).value_counts(normalize=True)
    target_counts = target.dropna().astype(str).value_counts(normalize=True)
    categories = sorted(set(source_counts.index).union(target_counts.index))
    if not categories:
        return math.nan
    source_probs = np.array(
        [source_counts.get(category, 0.0) for category in categories], dtype=float
    )
    target_probs = np.array(
        [target_counts.get(category, 0.0) for category in categories], dtype=float
    )
    return float(jensenshannon(source_probs, target_probs, base=2.0) ** 2)


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


def load_optional_csv(
    path: Path, warnings: list[str], found_inputs: list[str]
) -> pd.DataFrame | None:
    if not path.exists():
        warnings.append(f"Missing optional input: {path}")
        return None
    found_inputs.append(str(path))
    return pd.read_csv(path, low_memory=False)


def discover_dataset_path(input_root: Path) -> Path:
    dataset_name = input_root.name.lower()
    root = input_root.parent / "Data"
    if dataset_name == "abide":
        candidates = sorted(
            root.glob(
                "ABIDE2_preprocessing/data/processed/abide2_core_multisite_transportability_dataset_strict.csv"
            )
        )
        if candidates:
            return candidates[0]
    raise RuntimeError(f"Could not discover processed dataset CSV for {input_root}.")


def load_supporting_inputs(
    input_root: Path, warnings: list[str], found_inputs: list[str]
) -> dict[str, Any]:
    dataset_path = discover_dataset_path(input_root)
    dataset_df = pd.read_csv(dataset_path, low_memory=False)
    found_inputs.append(str(dataset_path))

    schema_path = (
        dataset_path.parent / "abide2_core_multisite_preprocessing_blueprint.json"
    )
    schema = {}
    if schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        found_inputs.append(str(schema_path))
    else:
        warnings.append(f"Missing optional schema: {schema_path}")

    variable_inventory_path = (
        input_root / "Quality_report" / "abide_variable_inventory.csv"
    )
    variable_inventory_df = load_optional_csv(
        variable_inventory_path, warnings, found_inputs
    )

    performance_path = input_root / "Site_distance" / "site_distance_vs_performance.csv"
    performance_df = load_optional_csv(performance_path, warnings, found_inputs)
    if performance_df is None:
        raise RuntimeError("Required file missing: site_distance_vs_performance.csv")

    bootstrap_results_path = (
        input_root
        / "Methods"
        / "ABIDE2_M4_ComBat_transductive_bootstrap10_outputs"
        / "tables"
        / "target_site_results.csv"
    )
    bootstrap_raw_path = (
        input_root
        / "Methods"
        / "ABIDE2_M4_ComBat_transductive_bootstrap10_outputs"
        / "tables"
        / "target_site_results_bootstrap_raw.csv"
    )
    bootstrap_summary_df = load_optional_csv(
        bootstrap_results_path, warnings, found_inputs
    )
    bootstrap_raw_df = load_optional_csv(bootstrap_raw_path, warnings, found_inputs)

    sample_size_df = load_optional_csv(
        input_root / "Sample_size" / "sample_size_by_site.csv", warnings, found_inputs
    )
    entropy_df = load_optional_csv(
        input_root / "Entropy" / "entropy_summary_by_site.csv", warnings, found_inputs
    )
    site_distance_df = load_optional_csv(
        input_root / "Site_distance" / "site_distance_by_target.csv",
        warnings,
        found_inputs,
    )

    return {
        "dataset_df": dataset_df,
        "dataset_path": dataset_path,
        "schema": schema,
        "variable_inventory_df": variable_inventory_df,
        "performance_df": performance_df,
        "bootstrap_summary_df": bootstrap_summary_df,
        "bootstrap_raw_df": bootstrap_raw_df,
        "sample_size_df": sample_size_df,
        "entropy_df": entropy_df,
        "site_distance_df": site_distance_df,
    }


def detect_site_target_id_columns(
    dataset_df: pd.DataFrame, schema: dict[str, Any]
) -> tuple[str, str, list[str]]:
    site_column = next(
        (
            column
            for column in [schema.get("domain_column"), *SITE_COLUMN_CANDIDATES]
            if column in dataset_df.columns
        ),
        None,
    )
    target_column = next(
        (
            column
            for column in [schema.get("target_column"), *TARGET_COLUMN_CANDIDATES]
            if column in dataset_df.columns
        ),
        None,
    )
    if site_column is None or target_column is None:
        raise RuntimeError("Could not detect site or target column.")
    id_columns = [
        column
        for column in schema.get("id_columns", [])
        if column in dataset_df.columns
    ]
    if not id_columns:
        id_columns = [
            column
            for column in dataset_df.columns
            if any(keyword in column.lower() for keyword in ID_COLUMN_KEYWORDS)
        ]
    return site_column, target_column, id_columns


def infer_metadata_candidates(
    dataset_df: pd.DataFrame,
    schema: dict[str, Any],
    variable_inventory_df: pd.DataFrame | None,
    site_column: str,
    target_column: str,
    id_columns: list[str],
) -> pd.DataFrame:
    inventory_lookup: dict[str, dict[str, Any]] = {}
    metadata_candidates: list[str] = []
    numeric_from_schema = set(schema.get("numeric_features", []))
    categorical_from_schema = set(schema.get("categorical_features", []))
    if (
        variable_inventory_df is not None
        and not variable_inventory_df.empty
        and {"column_name", "role"} <= set(variable_inventory_df.columns)
    ):
        inventory_lookup = variable_inventory_df.set_index("column_name").to_dict(
            orient="index"
        )
        role_mask = variable_inventory_df["role"].isin(
            ["predictive_feature", "metadata", "qc_feature"]
        )
        metadata_candidates = [
            column
            for column in variable_inventory_df.loc[role_mask, "column_name"]
            .astype(str)
            .tolist()
            if column in dataset_df.columns
        ]
    if not metadata_candidates:
        schema_candidates = [
            column
            for column in schema.get("predictive_feature_columns", [])
            if column in dataset_df.columns
        ]
        if schema_candidates:
            metadata_candidates = schema_candidates
        excluded = set(
            id_columns
            + [site_column, target_column, schema.get("target_original_column", "")]
        )
    if not metadata_candidates:
        metadata_candidates = [
            column
            for column in dataset_df.columns
            if column not in excluded and not str(column).startswith(GENE_PREFIX)
        ]

    rows = []
    for column in metadata_candidates:
        series = dataset_df[column]
        missing_rate = float(series.isna().mean())
        n_unique = int(series.dropna().nunique())
        conceptual_type = str(
            inventory_lookup.get(column, {}).get("conceptual_type", "")
        ).lower()
        variable_group = str(
            inventory_lookup.get(column, {}).get("variable_group", "")
        ).lower()
        modeling_group = str(
            inventory_lookup.get(column, {}).get("modeling_group", "")
        ).lower()
        exclusion_reason = ""

        if (
            column in numeric_from_schema
            or "numeric" in conceptual_type
            or "continuous" in conceptual_type
            or "count" in conceptual_type
            or "proportion" in conceptual_type
        ):
            inferred_type = "continuous"
        elif column in categorical_from_schema:
            inferred_type = "binary" if n_unique <= 2 else "categorical"
        elif n_unique <= 2:
            inferred_type = "binary"
        elif pd.api.types.is_numeric_dtype(series) and n_unique > 10:
            inferred_type = "continuous"
        else:
            inferred_type = "categorical"

        used_in_analysis = True
        if (
            column in DIRECT_SITE_METADATA_COLUMNS
            or variable_group == "domain_metadata"
            or modeling_group == "domain_metadata"
        ):
            used_in_analysis = False
            exclusion_reason = "direct_site_or_domain_encoding"
        elif missing_rate > MAX_METADATA_MISSING_RATE:
            used_in_analysis = False
            exclusion_reason = f"missing_rate>{MAX_METADATA_MISSING_RATE:.2f}"
        elif n_unique <= 1:
            used_in_analysis = False
            exclusion_reason = "single_unique_value"
        elif str(column).startswith(GENE_PREFIX):
            used_in_analysis = False
            exclusion_reason = "gene_expression_feature"
        elif column in id_columns or column in {
            site_column,
            target_column,
            "target_label",
            "ajcc_pathologic_stage_raw",
        }:
            used_in_analysis = False
            exclusion_reason = "identifier_or_label"

        if not used_in_analysis and not exclusion_reason:
            exclusion_reason = f"missing_rate>{MAX_METADATA_MISSING_RATE:.2f}"

        rows.append(
            {
                "column_name": column,
                "inferred_type": inferred_type,
                "missing_rate": missing_rate,
                "n_unique": n_unique,
                "used_in_analysis": used_in_analysis,
                "exclusion_reason": exclusion_reason,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["used_in_analysis", "column_name"], ascending=[False, True])
        .reset_index(drop=True)
    )


def metadata_summary_by_site(
    dataset_df: pd.DataFrame, site_column: str, metadata_info_df: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for site_name, site_df in dataset_df.groupby(site_column):
        for _, info in metadata_info_df.iterrows():
            column = info["column_name"]
            metadata_type = info["inferred_type"]
            series = site_df[column]
            n_non_missing = int(series.notna().sum())
            missing_rate = float(series.isna().mean())

            if metadata_type == "continuous":
                numeric = safe_numeric(series).dropna()
                q1 = float(numeric.quantile(0.25)) if not numeric.empty else math.nan
                q3 = float(numeric.quantile(0.75)) if not numeric.empty else math.nan
                rows.append(
                    {
                        "dataset": DATASET_FULL_LABEL,
                        "site": site_name,
                        "metadata_column": column,
                        "metadata_type": metadata_type,
                        "summary_row_type": "continuous_summary",
                        "category": "",
                        "category_count": math.nan,
                        "category_ratio": math.nan,
                        "dominant_category": "",
                        "categorical_entropy": math.nan,
                        "n_non_missing": n_non_missing,
                        "missing_rate": missing_rate,
                        "mean": float(numeric.mean())
                        if not numeric.empty
                        else math.nan,
                        "std": float(numeric.std(ddof=1))
                        if len(numeric) > 1
                        else math.nan,
                        "median": float(numeric.median())
                        if not numeric.empty
                        else math.nan,
                        "iqr": float(q3 - q1) if not numeric.empty else math.nan,
                        "min": float(numeric.min()) if not numeric.empty else math.nan,
                        "max": float(numeric.max()) if not numeric.empty else math.nan,
                    }
                )
            else:
                counts = series.dropna().astype(str).value_counts()
                total = counts.sum()
                dominant = counts.index[0] if not counts.empty else ""
                entropy_value = categorical_entropy(series)
                if counts.empty:
                    rows.append(
                        {
                            "dataset": DATASET_FULL_LABEL,
                            "site": site_name,
                            "metadata_column": column,
                            "metadata_type": metadata_type,
                            "summary_row_type": "categorical_category",
                            "category": "",
                            "category_count": 0,
                            "category_ratio": math.nan,
                            "dominant_category": dominant,
                            "categorical_entropy": entropy_value,
                            "n_non_missing": n_non_missing,
                            "missing_rate": missing_rate,
                            "mean": math.nan,
                            "std": math.nan,
                            "median": math.nan,
                            "iqr": math.nan,
                            "min": math.nan,
                            "max": math.nan,
                        }
                    )
                else:
                    for category, count in counts.items():
                        rows.append(
                            {
                                "dataset": DATASET_FULL_LABEL,
                                "site": site_name,
                                "metadata_column": column,
                                "metadata_type": metadata_type,
                                "summary_row_type": "categorical_category",
                                "category": category,
                                "category_count": int(count),
                                "category_ratio": float(count / total),
                                "dominant_category": dominant,
                                "categorical_entropy": entropy_value,
                                "n_non_missing": n_non_missing,
                                "missing_rate": missing_rate,
                                "mean": math.nan,
                                "std": math.nan,
                                "median": math.nan,
                                "iqr": math.nan,
                                "min": math.nan,
                                "max": math.nan,
                            }
                        )
    return pd.DataFrame(rows)


def plot_continuous_metadata_boxplots(
    dataset_df: pd.DataFrame,
    site_column: str,
    metadata_info_df: pd.DataFrame,
    output_path: Path,
) -> None:
    continuous_columns = metadata_info_df.loc[
        metadata_info_df["inferred_type"] == "continuous", "column_name"
    ].tolist()
    if not continuous_columns:
        return
    n_cols = 2
    n_rows = math.ceil(len(continuous_columns) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7.8 * n_cols, 4.7 * n_rows), squeeze=False
    )
    sites = sorted(dataset_df[site_column].astype(str).unique())
    colors = plt.cm.Set3(np.linspace(0.15, 0.85, len(sites)))

    for ax, column in zip(axes.flatten(), continuous_columns, strict=False):
        grouped = [
            safe_numeric(dataset_df.loc[dataset_df[site_column] == site, column])
            .dropna()
            .to_numpy(dtype=float)
            for site in sites
        ]
        bp = ax.boxplot(grouped, tick_labels=sites, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_edgecolor("#444444")
        for median in bp["medians"]:
            median.set_color("#111111")
            median.set_linewidth(1.2)
        ax.set_title(pretty_label(column), fontsize=13, pad=10)
        ax.set_ylabel(pretty_label(column))
        ax.tick_params(axis="x", rotation=20)
    for ax in axes.flatten()[len(continuous_columns) :]:
        ax.axis("off")
    fig.suptitle("Continuous metadata by site", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_categorical_metadata_stacked_bars(
    dataset_df: pd.DataFrame,
    site_column: str,
    metadata_info_df: pd.DataFrame,
    output_path: Path,
) -> None:
    categorical_columns = metadata_info_df.loc[
        metadata_info_df["used_in_analysis"]
        & metadata_info_df["inferred_type"].isin(["categorical", "binary"]),
        "column_name",
    ].tolist()
    if not categorical_columns:
        return
    n_cols = 2
    n_rows = math.ceil(len(categorical_columns) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(8.0 * n_cols, 4.5 * n_rows), squeeze=False
    )
    sites = sorted(dataset_df[site_column].astype(str).unique())

    for ax, column in zip(axes.flatten(), categorical_columns, strict=False):
        summary = (
            dataset_df[[site_column, column]]
            .assign(
                category=lambda frame: (
                    frame[column].where(frame[column].notna(), "Missing").astype(str)
                )
            )
            .groupby([site_column, "category"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=sites, fill_value=0)
        )
        if summary.shape[1] > 8:
            top_categories = (
                summary.sum(axis=0).sort_values(ascending=False).head(7).index.tolist()
            )
            remainder = [
                category
                for category in summary.columns
                if category not in top_categories
            ]
            summary = summary[top_categories].copy()
            if remainder:
                summary["Other"] = summary.reindex(columns=remainder, fill_value=0).sum(
                    axis=1
                )
        summary = summary.reindex(
            columns=sort_category_labels(summary.columns.tolist()), fill_value=0
        )
        proportions = summary.div(
            summary.sum(axis=1).replace(0, np.nan), axis=0
        ).fillna(0.0)
        bottom = np.zeros(len(proportions))
        categories = proportions.columns.tolist()
        color_palette = [
            "#4C78A8",
            "#F58518",
            "#54A24B",
            "#E45756",
            "#72B7B2",
            "#EECA3B",
            "#B279A2",
            "#FF9DA6",
            "#9D755D",
            "#BAB0AC",
        ]
        for index, category in enumerate(categories):
            values = proportions[category].to_numpy(dtype=float)
            color = color_palette[index % len(color_palette)]
            ax.bar(
                np.arange(len(sites)),
                values,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                label=str(category),
            )
            bottom += values
        ax.set_title(pretty_label(column), fontsize=13, pad=10)
        ax.set_xticks(np.arange(len(sites)))
        ax.set_xticklabels(sites, rotation=20)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Proportion")
        ax.legend(
            frameon=False,
            fontsize=8,
            ncol=min(4, len(categories)),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
        )
    for ax in axes.flatten()[len(categorical_columns) :]:
        ax.axis("off")
    fig.suptitle("Categorical metadata by site", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.97))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def calculate_metadata_shift(
    dataset_df: pd.DataFrame, site_column: str, metadata_info_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []

    active_metadata = metadata_info_df[metadata_info_df["used_in_analysis"]].copy()
    for target_site in sorted(dataset_df[site_column].astype(str).unique()):
        target_df = dataset_df[
            dataset_df[site_column].astype(str) == target_site
        ].copy()
        source_df = dataset_df[
            dataset_df[site_column].astype(str) != target_site
        ].copy()
        for _, info in active_metadata.iterrows():
            column = info["column_name"]
            metadata_type = info["inferred_type"]
            n_source = int(source_df[column].notna().sum())
            n_target = int(target_df[column].notna().sum())

            if metadata_type == "continuous":
                source_values = safe_numeric(source_df[column]).dropna()
                target_values = safe_numeric(target_df[column]).dropna()
                mean_diff = (
                    float(target_values.mean() - source_values.mean())
                    if not source_values.empty and not target_values.empty
                    else math.nan
                )
                effect = cohen_d(source_values, target_values)
                wasserstein = (
                    float(wasserstein_distance(source_values, target_values))
                    if not source_values.empty and not target_values.empty
                    else math.nan
                )
                if len(source_values) >= 2 and len(target_values) >= 2:
                    ks_stat, ks_p = ks_2samp(source_values, target_values)
                else:
                    ks_stat, ks_p = math.nan, math.nan
                metrics = [
                    ("difference_in_means", mean_diff, math.nan),
                    ("cohens_d", effect, math.nan),
                    ("wasserstein_distance", wasserstein, math.nan),
                    (
                        "ks_statistic",
                        float(ks_stat) if not pd.isna(ks_stat) else math.nan,
                        float(ks_p) if not pd.isna(ks_p) else math.nan,
                    ),
                ]
                representative_value = (
                    abs(effect) if not pd.isna(effect) else wasserstein
                )
            else:
                source_series = source_df[column].dropna().astype(str)
                target_series = target_df[column].dropna().astype(str)
                categories = sorted(
                    set(source_series.unique()).union(target_series.unique())
                )
                source_probs = np.array(
                    [
                        (source_series == category).mean()
                        if len(source_series)
                        else 0.0
                        for category in categories
                    ],
                    dtype=float,
                )
                target_probs = np.array(
                    [
                        (target_series == category).mean()
                        if len(target_series)
                        else 0.0
                        for category in categories
                    ],
                    dtype=float,
                )
                mean_abs_diff = (
                    float(np.mean(np.abs(target_probs - source_probs)))
                    if categories
                    else math.nan
                )
                js_div = js_divergence_from_series(source_series, target_series)
                entropy_diff = categorical_entropy(target_series) - categorical_entropy(
                    source_series
                )
                if categories and len(source_series) > 0 and len(target_series) > 0:
                    contingency = pd.DataFrame(
                        {
                            "source": source_series.value_counts(),
                            "target": target_series.value_counts(),
                        }
                    ).fillna(0.0)
                    chi2, chi2_p, _, _ = chi2_contingency(
                        contingency.to_numpy(dtype=float)
                    )
                else:
                    chi2, chi2_p = math.nan, math.nan
                metrics = [
                    (
                        "mean_abs_category_proportion_difference",
                        mean_abs_diff,
                        math.nan,
                    ),
                    ("jensen_shannon_divergence", js_div, math.nan),
                    (
                        "chi_square_statistic",
                        float(chi2) if not pd.isna(chi2) else math.nan,
                        float(chi2_p) if not pd.isna(chi2_p) else math.nan,
                    ),
                    ("categorical_entropy_difference", entropy_diff, math.nan),
                ]
                representative_value = js_div

            for metric_name, metric_value, p_value in metrics:
                long_rows.append(
                    {
                        "dataset": DATASET_FULL_LABEL,
                        "target_site": target_site,
                        "metadata_column": column,
                        "metadata_type": metadata_type,
                        "n_source": n_source,
                        "n_target": n_target,
                        "shift_metric_name": metric_name,
                        "shift_metric_value": metric_value,
                        "p_value": p_value,
                    }
                )
            representative_rows.append(
                {
                    "dataset": DATASET_FULL_LABEL,
                    "target_site": target_site,
                    "metadata_column": column,
                    "metadata_type": metadata_type,
                    "representative_shift": representative_value,
                }
            )

    long_df = pd.DataFrame(long_rows)
    representative_df = pd.DataFrame(representative_rows)

    summary_rows = []
    for target_site, group_df in representative_df.groupby("target_site"):
        continuous_values = (
            group_df.loc[
                group_df["metadata_type"] == "continuous", "representative_shift"
            ]
            .dropna()
            .abs()
        )
        categorical_values = (
            group_df.loc[
                group_df["metadata_type"].isin(["categorical", "binary"]),
                "representative_shift",
            ]
            .dropna()
            .abs()
        )
        summary_rows.append(
            {
                "dataset": DATASET_FULL_LABEL,
                "target_site": target_site,
                "mean_continuous_metadata_shift_raw": float(continuous_values.mean())
                if not continuous_values.empty
                else math.nan,
                "max_continuous_metadata_shift": float(continuous_values.max())
                if not continuous_values.empty
                else math.nan,
                "mean_categorical_metadata_shift_raw": float(categorical_values.mean())
                if not categorical_values.empty
                else math.nan,
                "max_categorical_metadata_shift": float(categorical_values.max())
                if not categorical_values.empty
                else math.nan,
                "n_metadata_used": int(group_df["representative_shift"].notna().sum()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df["mean_continuous_metadata_shift"] = summary_df[
        "mean_continuous_metadata_shift_raw"
    ]
    summary_df["mean_categorical_metadata_shift"] = summary_df[
        "mean_categorical_metadata_shift_raw"
    ]
    cont_scaled = minmax_scale(summary_df["mean_continuous_metadata_shift"])
    cat_scaled = minmax_scale(summary_df["mean_categorical_metadata_shift"])
    summary_df["overall_metadata_shift_score"] = pd.concat(
        [cont_scaled, cat_scaled], axis=1
    ).mean(axis=1, skipna=True)
    summary_df = summary_df.drop(
        columns=[
            "mean_continuous_metadata_shift_raw",
            "mean_categorical_metadata_shift_raw",
        ]
    )

    representative_wide = (
        representative_df.pivot(
            index="target_site",
            columns="metadata_column",
            values="representative_shift",
        )
        .add_prefix("metadata_shift__")
        .reset_index()
    )
    summary_df = summary_df.merge(representative_wide, on="target_site", how="left")
    return long_df, summary_df, representative_df


def build_metadata_shift_vs_performance(
    performance_df: pd.DataFrame, shift_summary_df: pd.DataFrame
) -> pd.DataFrame:
    merged = performance_df.copy()
    shift_frame = shift_summary_df.copy()
    if "mean_ece" in merged.columns and "ece" not in merged.columns:
        merged = merged.rename(columns={"mean_ece": "ece"})

    if {"dataset", "target_site"} <= set(merged.columns) and {
        "dataset",
        "target_site",
    } <= set(shift_frame.columns):
        merged["_dataset_key"] = merged["dataset"].map(normalize_dataset_key)
        shift_frame["_dataset_key"] = shift_frame["dataset"].map(normalize_dataset_key)
        merged = merged.merge(
            shift_frame.drop(columns=["dataset"]),
            on=["target_site", "_dataset_key"],
            how="left",
        )
        merged = merged.drop(columns=["_dataset_key"])
        if (
            "overall_metadata_shift_score" in merged.columns
            and merged["overall_metadata_shift_score"].isna().all()
        ):
            merged = performance_df.copy().merge(
                shift_summary_df.drop(columns=["dataset"]),
                on=["target_site"],
                how="left",
            )
    else:
        merged = merged.merge(shift_frame, on=["target_site"], how="left")

    if "id_ood_gap" not in merged.columns and {
        "mean_id_performance",
        "mean_ood_performance",
    } <= set(merged.columns):
        merged["id_ood_gap"] = (
            merged["mean_id_performance"] - merged["mean_ood_performance"]
        )
    return merged


def regression_line(ax: plt.Axes, x: pd.Series, y: pd.Series) -> None:
    valid = pd.concat([safe_numeric(x), safe_numeric(y)], axis=1).dropna()
    if len(valid) < 2:
        return
    slope, intercept = np.polyfit(valid.iloc[:, 0], valid.iloc[:, 1], deg=1)
    x_line = np.linspace(valid.iloc[:, 0].min(), valid.iloc[:, 0].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="#333333", linestyle="--", linewidth=1.2)


def scatter_metadata_shift_vs_outcome(
    subset: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    valid_for_range = pd.to_numeric(subset[x_column], errors="coerce").dropna()
    x_range = (
        float(valid_for_range.max() - valid_for_range.min())
        if len(valid_for_range) >= 2
        else 0.0
    )
    jitter_scale = max(x_range * 0.03, 0.006)
    method_offsets = {
        method: (index - (len(METHOD_ORDER) - 1) / 2) * jitter_scale
        for index, method in enumerate(METHOD_ORDER)
    }
    label_offsets = {
        "M0": (5, 6),
        "M1": (5, -10),
        "M2": (5, 11),
        "M3": (5, -14),
        "M4": (5, 1),
    }
    label_column = (
        "target_site_display"
        if "target_site_display" in subset.columns
        and subset["target_site_display"].notna().any()
        else "target_site"
    )
    for method in METHOD_ORDER:
        method_df = subset[subset["method"] == method]
        if method_df.empty:
            continue
        x_values = pd.to_numeric(
            method_df[x_column], errors="coerce"
        ) + method_offsets.get(method, 0.0)
        y_values = pd.to_numeric(method_df[y_column], errors="coerce")
        ax.scatter(
            x_values,
            y_values,
            color=METHOD_COLORS.get(method, "#999999"),
            edgecolor="#333333",
            linewidth=0.7,
            s=82,
            alpha=0.92,
            zorder=3,
        )
        x_text_offset, y_text_offset = label_offsets.get(method, (5, 5))
        for row_idx, (_, row) in enumerate(method_df.reset_index(drop=True).iterrows()):
            x_value = (
                x_values.iloc[row_idx]
                if hasattr(x_values, "iloc")
                else x_values[row_idx]
            )
            y_value = (
                y_values.iloc[row_idx]
                if hasattr(y_values, "iloc")
                else y_values[row_idx]
            )
            if pd.isna(x_value) or pd.isna(y_value):
                continue
            label = str(row.get(label_column, row.get("target_site", "")))
            ax.annotate(
                label,
                xy=(x_value, y_value),
                xytext=(x_text_offset, y_text_offset),
                textcoords="offset points",
                fontsize=8.2,
                color="#333333",
                ha="left",
                va="center",
                fontweight="normal",
                zorder=4,
            )

    regression_line(ax, subset[x_column], subset[y_column])
    pearson_r, _, spearman_r, _, n_obs = safe_corr(subset[x_column], subset[y_column])
    annotation = (
        f"Pearson r = {pearson_r:.2f}\nSpearman rho = {spearman_r:.2f}\nn = {n_obs}"
    )
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#DDDDDD"},
    )
    ax.set_xlabel(pretty_label(x_column))
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=14, pad=12)
    fig.text(0.5, 0.02, METHOD_LEGEND_TEXT, ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def generate_shift_performance_figures(
    performance_shift_df: pd.DataFrame, output_dir: Path
) -> None:
    combos = (
        performance_shift_df[["model", "metric"]]
        .drop_duplicates()
        .sort_values(["model", "metric"])
    )
    for _, combo in combos.iterrows():
        subset = performance_shift_df[
            (performance_shift_df["model"] == combo["model"])
            & (performance_shift_df["metric"] == combo["metric"])
        ].copy()
        if subset.empty:
            continue
        model_slug = slugify(str(combo["model"]))
        metric_slug = slugify(str(combo["metric"]))
        scatter_metadata_shift_vs_outcome(
            subset,
            x_column="overall_metadata_shift_score",
            y_column="mean_ood_performance",
            output_path=output_dir
            / f"metadata_shift_vs_ood_performance_{model_slug}_{metric_slug}.png",
            title=f"Metadata shift vs OOD performance | {combo['model']} | {combo['metric']}",
            y_label="Mean OOD performance",
        )
        scatter_metadata_shift_vs_outcome(
            subset,
            x_column="overall_metadata_shift_score",
            y_column="id_ood_gap",
            output_path=output_dir
            / f"metadata_shift_vs_id_ood_gap_{model_slug}_{metric_slug}.png",
            title=f"Metadata shift vs ID-OOD gap | {combo['model']} | {combo['metric']}",
            y_label="ID-OOD gap",
        )
        if (
            "std_ood_performance" in subset.columns
            and not subset["std_ood_performance"].isna().all()
        ):
            scatter_metadata_shift_vs_outcome(
                subset,
                x_column="overall_metadata_shift_score",
                y_column="std_ood_performance",
                output_path=output_dir
                / f"metadata_shift_vs_ood_std_{model_slug}_{metric_slug}.png",
                title=f"Metadata shift vs OOD variability | {combo['model']} | {combo['metric']}",
                y_label="OOD performance std",
            )


def build_metadata_label_association(
    dataset_df: pd.DataFrame, target_column: str, metadata_info_df: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    active_metadata = metadata_info_df[metadata_info_df["used_in_analysis"]].copy()
    for _, info in active_metadata.iterrows():
        column = info["column_name"]
        metadata_type = info["inferred_type"]
        subset = dataset_df[[column, target_column]].dropna()
        if subset.empty:
            continue
        if metadata_type == "continuous":
            group0 = safe_numeric(
                subset.loc[subset[target_column] == 0, column]
            ).dropna()
            group1 = safe_numeric(
                subset.loc[subset[target_column] == 1, column]
            ).dropna()
            if len(group0) >= 2 and len(group1) >= 2:
                t_stat, t_p = ttest_ind(group0, group1, equal_var=False)
                mw_stat, mw_p = mannwhitneyu(group0, group1, alternative="two-sided")
                effect = cohen_d(group0, group1)
            else:
                t_stat, t_p, mw_stat, mw_p, effect = (
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                )
            rows.extend(
                [
                    {
                        "dataset": DATASET_FULL_LABEL,
                        "metadata_column": column,
                        "metadata_type": metadata_type,
                        "test_name": "welch_t_test",
                        "statistic": float(t_stat) if not pd.isna(t_stat) else math.nan,
                        "p_value": float(t_p) if not pd.isna(t_p) else math.nan,
                        "effect_size": effect,
                        "group_0_n": int(len(group0)),
                        "group_1_n": int(len(group1)),
                        "group_0_mean": float(group0.mean())
                        if not group0.empty
                        else math.nan,
                        "group_1_mean": float(group1.mean())
                        if not group1.empty
                        else math.nan,
                    },
                    {
                        "dataset": DATASET_FULL_LABEL,
                        "metadata_column": column,
                        "metadata_type": metadata_type,
                        "test_name": "mann_whitney_u",
                        "statistic": float(mw_stat)
                        if not pd.isna(mw_stat)
                        else math.nan,
                        "p_value": float(mw_p) if not pd.isna(mw_p) else math.nan,
                        "effect_size": effect,
                        "group_0_n": int(len(group0)),
                        "group_1_n": int(len(group1)),
                        "group_0_mean": float(group0.mean())
                        if not group0.empty
                        else math.nan,
                        "group_1_mean": float(group1.mean())
                        if not group1.empty
                        else math.nan,
                    },
                ]
            )
        else:
            contingency = pd.crosstab(subset[column].astype(str), subset[target_column])
            if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                chi2, p_value, _, _ = chi2_contingency(contingency)
                cramer_v = cramers_v_from_table(contingency)
            else:
                chi2, p_value, cramer_v = math.nan, math.nan, math.nan
            rows.append(
                {
                    "dataset": DATASET_FULL_LABEL,
                    "metadata_column": column,
                    "metadata_type": metadata_type,
                    "test_name": "chi_square",
                    "statistic": float(chi2) if not pd.isna(chi2) else math.nan,
                    "p_value": float(p_value) if not pd.isna(p_value) else math.nan,
                    "effect_size": cramer_v,
                    "group_0_n": int((subset[target_column] == 0).sum()),
                    "group_1_n": int((subset[target_column] == 1).sum()),
                    "group_0_mean": math.nan,
                    "group_1_mean": math.nan,
                    "contingency_json": json.dumps(contingency.to_dict()),
                }
            )
    return pd.DataFrame(rows)


def build_site_label_association(
    dataset_df: pd.DataFrame, site_column: str, target_column: str
) -> pd.DataFrame:
    contingency = pd.crosstab(
        dataset_df[site_column].astype(str), dataset_df[target_column]
    )
    chi2, p_value, _, _ = chi2_contingency(contingency)
    return pd.DataFrame(
        [
            {
                "dataset": DATASET_FULL_LABEL,
                "site_column": site_column,
                "target_column": target_column,
                "chi_square_statistic": float(chi2),
                "p_value": float(p_value),
                "cramers_v": cramers_v_from_table(contingency),
                "contingency_json": json.dumps(contingency.to_dict()),
            }
        ]
    )


def build_metadata_site_association(
    dataset_df: pd.DataFrame, site_column: str, metadata_info_df: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    active_metadata = metadata_info_df[metadata_info_df["used_in_analysis"]].copy()
    sites = sorted(dataset_df[site_column].astype(str).unique())
    for _, info in active_metadata.iterrows():
        column = info["column_name"]
        metadata_type = info["inferred_type"]
        if metadata_type == "continuous":
            grouped = [
                safe_numeric(dataset_df.loc[dataset_df[site_column] == site, column])
                .dropna()
                .to_numpy(dtype=float)
                for site in sites
            ]
            grouped = [values for values in grouped if len(values) > 0]
            if len(grouped) >= 2 and all(len(values) >= 2 for values in grouped):
                anova_stat, anova_p = f_oneway(*grouped)
                kruskal_stat, kruskal_p = kruskal(*grouped)
            else:
                anova_stat, anova_p, kruskal_stat, kruskal_p = (
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                )
            rows.extend(
                [
                    {
                        "dataset": DATASET_FULL_LABEL,
                        "metadata_column": column,
                        "metadata_type": metadata_type,
                        "test_name": "anova",
                        "statistic": float(anova_stat)
                        if not pd.isna(anova_stat)
                        else math.nan,
                        "p_value": float(anova_p) if not pd.isna(anova_p) else math.nan,
                        "effect_size": math.nan,
                    },
                    {
                        "dataset": DATASET_FULL_LABEL,
                        "metadata_column": column,
                        "metadata_type": metadata_type,
                        "test_name": "kruskal_wallis",
                        "statistic": float(kruskal_stat)
                        if not pd.isna(kruskal_stat)
                        else math.nan,
                        "p_value": float(kruskal_p)
                        if not pd.isna(kruskal_p)
                        else math.nan,
                        "effect_size": math.nan,
                    },
                ]
            )
        else:
            contingency = pd.crosstab(
                dataset_df[site_column].astype(str), dataset_df[column].astype(str)
            )
            if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                chi2, p_value, _, _ = chi2_contingency(contingency)
                cramer_v = cramers_v_from_table(contingency)
            else:
                chi2, p_value, cramer_v = math.nan, math.nan, math.nan
            rows.append(
                {
                    "dataset": DATASET_FULL_LABEL,
                    "metadata_column": column,
                    "metadata_type": metadata_type,
                    "test_name": "chi_square",
                    "statistic": float(chi2) if not pd.isna(chi2) else math.nan,
                    "p_value": float(p_value) if not pd.isna(p_value) else math.nan,
                    "effect_size": cramer_v,
                }
            )
    return pd.DataFrame(rows)


def prepare_model_matrix(
    dataset_df: pd.DataFrame, columns: list[str], metadata_info_df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], list[str]]:
    matrix_df = dataset_df[columns].copy()
    metadata_type_lookup = (
        metadata_info_df.set_index("column_name")["inferred_type"].to_dict()
        if not metadata_info_df.empty
        else {}
    )
    continuous_columns = [
        column
        for column in columns
        if metadata_type_lookup.get(column) == "continuous"
        or str(column).startswith(GENE_PREFIX)
    ]
    categorical_columns = [
        column for column in columns if column not in continuous_columns
    ]

    continuous_frame = pd.DataFrame(index=matrix_df.index)
    for column in continuous_columns:
        numeric = safe_numeric(matrix_df[column])
        continuous_frame[column] = numeric.fillna(numeric.median())

    categorical_frames: list[pd.DataFrame] = []
    for column in categorical_columns:
        fill_value = safe_mode(matrix_df[column])
        filled = (
            matrix_df[column]
            .fillna(fill_value if pd.notna(fill_value) else "Missing")
            .astype(str)
        )
        dummies = pd.get_dummies(
            filled, prefix=f"{column}__cat__", prefix_sep="", dtype=float
        )
        categorical_frames.append(dummies)
    encoded_df = pd.concat([continuous_frame, *categorical_frames], axis=1)
    return encoded_df.astype(float), continuous_columns, categorical_columns


def summarize_importance_column(encoded_column: str) -> str:
    if "__cat__" in encoded_column:
        return encoded_column.split("__cat__", 1)[0]
    return encoded_column


def select_top_variable_gene_columns(
    dataset_df: pd.DataFrame, top_k: int = TOP_GENES_FOR_SITE_PREDICTABILITY
) -> list[str]:
    gene_columns = [
        column for column in dataset_df.columns if str(column).startswith(GENE_PREFIX)
    ]
    if not gene_columns:
        return []
    variances = (
        dataset_df[gene_columns]
        .apply(pd.to_numeric, errors="coerce")
        .var(axis=0, skipna=True)
    )
    return variances.sort_values(ascending=False).head(top_k).index.tolist()


def site_predictability(
    dataset_df: pd.DataFrame,
    site_column: str,
    metadata_info_df: pd.DataFrame,
    output_dir: Path,
    warnings: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_columns = metadata_info_df.loc[
        metadata_info_df["used_in_analysis"], "column_name"
    ].tolist()
    if not metadata_columns:
        warnings.append(
            "No metadata columns available for site predictability analysis."
        )
        return pd.DataFrame(), pd.DataFrame()

    feature_sets = {
        "biomedical_features_only": metadata_columns,
        "metadata_only": metadata_columns,
        "features_plus_metadata": metadata_columns,
    }
    y = dataset_df[site_column].astype(str).to_numpy()
    min_class_count = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(5, min_class_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    model_specs: dict[str, Any] = {
        "logisticregression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=4000, random_state=RANDOM_STATE)),
            ]
        ),
        "randomforest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "histgradientboosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
    }

    results_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    labels = sorted(pd.Series(y).unique().tolist())

    for feature_set_name, columns in feature_sets.items():
        if not columns:
            warnings.append(
                f"Skipped empty feature set for site predictability: {feature_set_name}."
            )
            continue
        X_df, _, _ = prepare_model_matrix(dataset_df, columns, metadata_info_df)
        for model_name, estimator in model_specs.items():
            y_pred = cross_val_predict(
                clone(estimator), X_df, y, cv=cv, method="predict"
            )
            results_rows.append(
                {
                    "dataset": DATASET_FULL_LABEL,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "n_features": len(columns),
                    "n_encoded_features": int(X_df.shape[1]),
                    "cv_splits": n_splits,
                    "accuracy": float(accuracy_score(y, y_pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
                    "macro_f1": float(f1_score(y, y_pred, average="macro")),
                }
            )

            matrix = confusion_matrix(y, y_pred, labels=labels)
            fig, ax = plt.subplots(figsize=(6.4, 5.6))
            image = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels)
            for row_index in range(matrix.shape[0]):
                for col_index in range(matrix.shape[1]):
                    ax.text(
                        col_index,
                        row_index,
                        str(int(matrix[row_index, col_index])),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
            ax.set_xlabel("Predicted site")
            ax.set_ylabel("True site")
            ax.set_title(
                f"Site predictability | {model_name} | {feature_set_name}",
                fontsize=13,
                pad=10,
            )
            cbar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
            cbar.set_label("Count (samples)")
            fig.tight_layout()
            fig.savefig(
                output_dir
                / f"site_predictability_confusion_matrix_{model_name}_{feature_set_name}.png",
                dpi=220,
            )
            plt.close(fig)

            if feature_set_name == "metadata_only" and model_name in {
                "logisticregression",
                "randomforest",
            }:
                fitted = clone(estimator).fit(X_df, y)
                if model_name == "randomforest":
                    importances = fitted.named_steps["model"].feature_importances_
                    importance_type = "random_forest_feature_importance"
                else:
                    coef = fitted.named_steps["model"].coef_
                    importances = np.abs(coef).mean(axis=0)
                    importance_type = "logistic_abs_coef"
                importance_df_encoded = pd.DataFrame(
                    {
                        "encoded_column": X_df.columns.tolist(),
                        "metadata_column": [
                            summarize_importance_column(column)
                            for column in X_df.columns.tolist()
                        ],
                        "importance": importances,
                    }
                )
                collapsed_importance = importance_df_encoded.groupby(
                    "metadata_column", as_index=False
                )["importance"].sum()
                for _, row in collapsed_importance.iterrows():
                    importance_rows.append(
                        {
                            "dataset": DATASET_FULL_LABEL,
                            "feature_set": feature_set_name,
                            "model": model_name,
                            "metadata_column": row["metadata_column"],
                            "importance_type": importance_type,
                            "importance": float(row["importance"]),
                        }
                    )

    importance_df = pd.DataFrame(importance_rows)
    return pd.DataFrame(results_rows), importance_df


def plot_metadata_importance(importance_df: pd.DataFrame, output_path: Path) -> None:
    if importance_df.empty:
        return
    model_order = [
        model
        for model in ["randomforest", "logisticregression"]
        if model in importance_df["model"].unique()
    ]
    fig, axes = plt.subplots(
        1, len(model_order), figsize=(7.2 * len(model_order), 5.5), squeeze=False
    )
    for ax, model_name in zip(axes.flatten(), model_order, strict=False):
        subset = importance_df[importance_df["model"] == model_name].sort_values(
            "importance", ascending=True
        )
        subset = subset.tail(12)
        ax.barh(
            np.arange(len(subset)),
            subset["importance"],
            color="#4C78A8" if model_name == "randomforest" else "#E45756",
            edgecolor="#333333",
            linewidth=0.7,
        )
        ax.set_yticks(np.arange(len(subset)))
        ax.set_yticklabels(
            [pretty_label(column) for column in subset["metadata_column"]]
        )
        ax.set_title(model_name, fontsize=13, pad=10)
        ax.set_xlabel("Importance")
    fig.suptitle("Variable importance for site prediction", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.apply(
            lambda col: (
                pd.Series(zscore(col, nan_policy="omit"), index=col.index)
                if col.nunique(dropna=True) > 1
                else pd.Series(np.zeros(len(col)), index=col.index)
            ),
            axis=0,
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def plot_clustered_heatmap(
    matrix_df: pd.DataFrame, output_path: Path, *, title: str, standardize_columns: bool
) -> None:
    if matrix_df.empty:
        return
    work_df = matrix_df.copy()
    work_df = work_df.dropna(axis=1, how="all")
    if work_df.empty:
        return
    work_df = work_df.apply(pd.to_numeric, errors="coerce")
    for column in work_df.columns:
        if work_df[column].isna().any():
            work_df[column] = work_df[column].fillna(work_df[column].median())
    if standardize_columns:
        plot_df = zscore_frame(work_df)
        vlim = (
            float(np.nanmax(np.abs(plot_df.to_numpy(dtype=float))))
            if not plot_df.empty
            else 1.0
        )
        vmin, vmax = -max(vlim, 1.0), max(vlim, 1.0)
    else:
        plot_df = work_df
        vmin = (
            float(np.nanmin(plot_df.to_numpy(dtype=float)))
            if not plot_df.empty
            else 0.0
        )
        vmax = (
            float(np.nanmax(plot_df.to_numpy(dtype=float)))
            if not plot_df.empty
            else 1.0
        )
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0

    row_order = list(range(plot_df.shape[0]))
    col_order = list(range(plot_df.shape[1]))
    row_linkage = (
        linkage(plot_df.to_numpy(dtype=float), method="average", metric="euclidean")
        if plot_df.shape[0] >= 2
        else None
    )
    col_linkage = (
        linkage(plot_df.to_numpy(dtype=float).T, method="average", metric="euclidean")
        if plot_df.shape[1] >= 2
        else None
    )
    if row_linkage is not None:
        row_order = dendrogram(row_linkage, no_plot=True)["leaves"]
    if col_linkage is not None:
        col_order = dendrogram(col_linkage, no_plot=True)["leaves"]
    ordered = plot_df.iloc[row_order, col_order]

    fig = plt.figure(
        figsize=(
            max(8.5, 0.65 * ordered.shape[1] + 5),
            max(7.0, 0.50 * ordered.shape[0] + 4),
        )
    )
    gs = gridspec.GridSpec(
        2,
        3,
        width_ratios=[0.80, 7.90, 0.60],
        height_ratios=[1.25, 6.9],
        wspace=0.34,
        hspace=0.04,
    )
    ax_col = fig.add_subplot(gs[0, 1])
    ax_row = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    if col_linkage is not None:
        dendrogram(col_linkage, ax=ax_col, no_labels=True, color_threshold=None)
        for collection in ax_col.collections:
            collection.set_color("black")
            collection.set_linewidth(1.2)
    ax_col.set_xticks([])
    ax_col.set_yticks([])
    for spine in ax_col.spines.values():
        spine.set_visible(False)

    if row_linkage is not None:
        dendrogram(
            row_linkage,
            ax=ax_row,
            orientation="left",
            no_labels=True,
            color_threshold=None,
        )
        for collection in ax_row.collections:
            collection.set_color("black")
            collection.set_linewidth(1.2)
    ax_row.set_xticks([])
    ax_row.set_yticks([])
    for spine in ax_row.spines.values():
        spine.set_visible(False)

    image = ax_heat.imshow(
        ordered.to_numpy(dtype=float),
        aspect="auto",
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
    )
    ax_heat.set_xticks(np.arange(ordered.shape[1]))
    ax_heat.set_xticklabels(
        [pretty_label(column) for column in ordered.columns],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax_heat.set_yticks(np.arange(ordered.shape[0]))
    ax_heat.set_yticklabels(ordered.index.tolist())
    ax_heat.tick_params(axis="y", pad=12, labelsize=10.0)
    ax_heat.tick_params(axis="x", labelsize=11)
    n_cells = ordered.shape[0] * ordered.shape[1]
    if n_cells <= 320:
        if n_cells <= 120:
            annotation_fontsize = 7.5
        elif n_cells <= 180:
            annotation_fontsize = 6.6
        elif n_cells <= 260:
            annotation_fontsize = 5.8
        else:
            annotation_fontsize = 5.1
        for row_idx in range(ordered.shape[0]):
            for col_idx in range(ordered.shape[1]):
                value = ordered.iloc[row_idx, col_idx]
                if pd.isna(value):
                    continue
                ax_heat.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=annotation_fontsize,
                    color="#111111",
                )

    cbar = fig.colorbar(image, cax=ax_cbar)
    cbar.set_label("Z-score" if standardize_columns else "Value")
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.subplots_adjust(
        left=0.05, right=0.96, bottom=0.12, top=0.92, wspace=0.34, hspace=0.04
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_metadata_performance_correlation_table(
    performance_shift_df: pd.DataFrame,
) -> pd.DataFrame:
    shift_columns = [
        column
        for column in performance_shift_df.columns
        if column
        in {
            "overall_metadata_shift_score",
            "mean_continuous_metadata_shift",
            "mean_categorical_metadata_shift",
        }
        or column.startswith("metadata_shift__")
    ]
    outcome_columns = [
        column
        for column in [
            "mean_ood_performance",
            "id_ood_gap",
            "std_ood_performance",
            "ece",
        ]
        if column in performance_shift_df.columns
    ]

    rows = []
    combos = (
        performance_shift_df[["model", "metric", "method"]]
        .drop_duplicates()
        .sort_values(["model", "metric", "method"])
    )
    for _, combo in combos.iterrows():
        subset = performance_shift_df[
            (performance_shift_df["model"] == combo["model"])
            & (performance_shift_df["metric"] == combo["metric"])
            & (performance_shift_df["method"] == combo["method"])
        ].copy()
        for variable_x in shift_columns:
            for variable_y in outcome_columns:
                pearson_r, pearson_p, spearman_r, spearman_p, n_obs = safe_corr(
                    subset[variable_x], subset[variable_y]
                )
                rows.append(
                    {
                        "dataset": DATASET_FULL_LABEL,
                        "model": combo["model"],
                        "metric": combo["metric"],
                        "method": combo["method"],
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


def build_metadata_analysis_summary(
    *,
    shift_long_df: pd.DataFrame,
    shift_summary_df: pd.DataFrame,
    metadata_label_df: pd.DataFrame,
    site_label_df: pd.DataFrame,
    metadata_site_df: pd.DataFrame,
    site_predictability_df: pd.DataFrame,
    metadata_perf_corr_df: pd.DataFrame,
    metadata_info_df: pd.DataFrame,
    warnings: list[str],
) -> str:
    lines = ["Metadata analysis summary", ""]

    active_metadata = metadata_info_df[metadata_info_df["used_in_analysis"]][
        "column_name"
    ].tolist()
    excluded_metadata = metadata_info_df[~metadata_info_df["used_in_analysis"]][
        "column_name"
    ].tolist()
    lines.append(
        f"- Active metadata used in shift/confounding analyses: {', '.join(active_metadata)}."
    )
    if excluded_metadata:
        lines.append(
            f"- Metadata excluded from advanced analyses: {', '.join(excluded_metadata)}."
        )

    if not shift_long_df.empty:
        representative = shift_long_df[
            shift_long_df["shift_metric_name"].isin(
                ["cohens_d", "jensen_shannon_divergence"]
            )
        ].copy()
        if not representative.empty:
            top_metadata = (
                representative.assign(
                    abs_shift=lambda frame: frame["shift_metric_value"].abs()
                )
                .groupby("metadata_column")["abs_shift"]
                .mean()
                .sort_values(ascending=False)
                .head(5)
            )
            lines.append(
                f"- Metadata with largest average source-vs-target shift: {', '.join(top_metadata.index.tolist())}."
            )

    if not shift_summary_df.empty:
        hardest = shift_summary_df.sort_values(
            "overall_metadata_shift_score", ascending=False
        ).head(3)
        lines.append(
            f"- Target sites with highest metadata shift: {', '.join(hardest['target_site'].tolist())}."
        )

    if not metadata_label_df.empty:
        best_label_assoc = (
            metadata_label_df.dropna(subset=["p_value"])
            .sort_values(["p_value", "effect_size"], ascending=[True, False])
            .head(3)
        )
        if not best_label_assoc.empty:
            lines.append(
                "- Metadata most associated with the target label: "
                + ", ".join(
                    f"{row['metadata_column']} ({row['test_name']}, p={row['p_value']:.3g})"
                    for _, row in best_label_assoc.iterrows()
                )
                + "."
            )

    if not metadata_site_df.empty:
        best_site_assoc = (
            metadata_site_df.dropna(subset=["p_value"])
            .sort_values(["p_value", "effect_size"], ascending=[True, False])
            .head(3)
        )
        if not best_site_assoc.empty:
            lines.append(
                "- Metadata most associated with site membership: "
                + ", ".join(
                    f"{row['metadata_column']} ({row['test_name']}, p={row['p_value']:.3g})"
                    for _, row in best_site_assoc.iterrows()
                )
                + "."
            )

    if not site_label_df.empty:
        row = site_label_df.iloc[0]
        lines.append(
            f"- Site-label association: chi-square={row['chi_square_statistic']:.2f}, p={row['p_value']:.3g}, Cramer's V={row['cramers_v']:.3f}."
        )

    if not site_predictability_df.empty:
        best_predictability = site_predictability_df.sort_values(
            ["balanced_accuracy", "macro_f1"], ascending=False
        ).iloc[0]
        lines.append(
            f"- Best site predictability: {best_predictability['model']} on {best_predictability['feature_set']} "
            f"(balanced_accuracy={best_predictability['balanced_accuracy']:.3f}, macro_f1={best_predictability['macro_f1']:.3f})."
        )

    if not metadata_perf_corr_df.empty:
        subset = (
            metadata_perf_corr_df[
                metadata_perf_corr_df["variable_y"] == "mean_ood_performance"
            ]
            .dropna(subset=["spearman_r"])
            .copy()
        )
        if not subset.empty:
            top_corr = subset.iloc[subset["spearman_r"].abs().argmax()]
            lines.append(
                f"- Strongest metadata-shift relation with OOD performance: {top_corr['variable_x']} "
                f"for {top_corr['model']} / {top_corr['metric']} / {top_corr['method']} (Spearman {top_corr['spearman_r']:.3f})."
            )

    if warnings:
        lines.append("")
        lines.append("Warnings / limitations:")
        for warning in warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def build_readme(
    found_inputs: list[str], warnings: list[str], metadata_info_df: pd.DataFrame
) -> str:
    active = metadata_info_df[metadata_info_df["used_in_analysis"]][
        "column_name"
    ].tolist()
    excluded_df = metadata_info_df[~metadata_info_df["used_in_analysis"]][
        ["column_name", "missing_rate", "exclusion_reason"]
    ]
    excluded_lines = [
        f"{row['column_name']} ({row['missing_rate']:.1%} missing; {row['exclusion_reason']})"
        for _, row in excluded_df.iterrows()
    ]
    readme = [
        "Metadata discussion outputs",
        "",
        "This folder contains a metadata-focused analysis of ABIDE II strict cross-site transportability.",
        "",
        "What was done:",
        "- detected metadata columns from the analysis-ready dataset and the quality inventory;",
        "- generated descriptive summaries by site for continuous and categorical metadata;",
        "- quantified metadata shift between each target site and the pooled source domains;",
        "- linked metadata shift with OOD performance, ID-OOD gap, variability and ECE;",
        "- tested metadata-label, site-label and metadata-site associations;",
        "- evaluated how easily the site can be predicted from the available variables;",
        "- estimated metadata importance for site prediction;",
        "- generated clustered heatmaps combining metadata shift and transportability outcomes.",
        "",
        f"Active metadata used in advanced analyses: {', '.join(active)}.",
        f"Excluded from advanced analyses because missing rate exceeded {MAX_METADATA_MISSING_RATE:.0%}: {', '.join(excluded_lines) if excluded_lines else 'none'}.",
        "",
        "Important note for ABIDE:",
        "- the strict ABIDE transportability dataset is composed almost entirely of clinically meaningful metadata-like variables;",
        "- the `biomedical_features_only`, `metadata_only` and `features_plus_metadata` site-prediction sets therefore overlap strongly and can yield identical results;",
        "- the direct site label itself was excluded from metadata modeling to avoid trivial site-label leakage.",
        "",
        "Interpretation notes:",
        "- high metadata shift suggests the target site differs clinically from the pooled source domains;",
        "- high site predictability indicates a strong site/domain signature in the available variables;",
        "- if metadata shift correlates with lower OOD performance or larger ID-OOD gap, metadata may partly explain transportability loss.",
        "",
        "Inputs found:",
        *[f"- {path}" for path in found_inputs],
    ]
    if warnings:
        readme.extend(["", "Warnings:", *[f"- {warning}" for warning in warnings]])
    return "\n".join(readme) + "\n"


def main() -> None:
    args = parse_args()
    apply_plot_style()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    found_inputs: list[str] = []
    loaded = load_supporting_inputs(args.input_root, warnings, found_inputs)
    dataset_df: pd.DataFrame = loaded["dataset_df"]
    dataset_path: Path = loaded["dataset_path"]
    schema: dict[str, Any] = loaded["schema"]
    variable_inventory_df: pd.DataFrame | None = loaded["variable_inventory_df"]
    performance_df: pd.DataFrame = loaded["performance_df"]

    site_column, target_column, id_columns = detect_site_target_id_columns(
        dataset_df, schema
    )
    metadata_info_df = infer_metadata_candidates(
        dataset_df,
        schema,
        variable_inventory_df,
        site_column,
        target_column,
        id_columns,
    )

    save_dataframe(
        metadata_info_df,
        output_dir / "detected_metadata_columns.csv",
        None,
        sheet_name="metadata_columns",
    )

    summary_df = metadata_summary_by_site(dataset_df, site_column, metadata_info_df)
    save_dataframe(
        summary_df,
        output_dir / "metadata_summary_by_site.csv",
        output_dir / "metadata_summary_by_site.xlsx",
        sheet_name="metadata_summary",
    )

    plot_continuous_metadata_boxplots(
        dataset_df,
        site_column,
        metadata_info_df,
        output_dir / "continuous_metadata_by_site_boxplots.png",
    )
    plot_categorical_metadata_stacked_bars(
        dataset_df,
        site_column,
        metadata_info_df,
        output_dir / "categorical_metadata_by_site_stacked_bars.png",
    )

    shift_long_df, shift_summary_df, representative_shift_df = calculate_metadata_shift(
        dataset_df, site_column, metadata_info_df
    )
    save_dataframe(
        shift_long_df,
        output_dir / "metadata_shift_source_vs_target_long.csv",
        output_dir / "metadata_shift_source_vs_target_long.xlsx",
        sheet_name="metadata_shift_long",
    )
    save_dataframe(
        shift_summary_df,
        output_dir / "metadata_shift_by_target_site.csv",
        output_dir / "metadata_shift_by_target_site.xlsx",
        sheet_name="metadata_shift_site",
    )

    performance_shift_df = build_metadata_shift_vs_performance(
        performance_df, shift_summary_df
    )
    save_dataframe(
        performance_shift_df,
        output_dir / "metadata_shift_vs_performance.csv",
        output_dir / "metadata_shift_vs_performance.xlsx",
        sheet_name="shift_vs_perf",
    )
    generate_shift_performance_figures(performance_shift_df, output_dir)

    metadata_label_df = build_metadata_label_association(
        dataset_df, target_column, metadata_info_df
    )
    save_dataframe(
        metadata_label_df,
        output_dir / "metadata_label_association.csv",
        output_dir / "metadata_label_association.xlsx",
        sheet_name="metadata_label",
    )

    site_label_df = build_site_label_association(dataset_df, site_column, target_column)
    save_dataframe(
        site_label_df,
        output_dir / "site_label_association.csv",
        output_dir / "site_label_association.xlsx",
        sheet_name="site_label",
    )

    metadata_site_df = build_metadata_site_association(
        dataset_df, site_column, metadata_info_df
    )
    save_dataframe(
        metadata_site_df,
        output_dir / "metadata_site_association.csv",
        output_dir / "metadata_site_association.xlsx",
        sheet_name="metadata_site",
    )

    site_predictability_df, importance_df = site_predictability(
        dataset_df, site_column, metadata_info_df, output_dir, warnings
    )
    save_dataframe(
        site_predictability_df,
        output_dir / "site_predictability_results.csv",
        output_dir / "site_predictability_results.xlsx",
        sheet_name="site_predict",
    )
    save_dataframe(
        importance_df,
        output_dir / "metadata_importance_site_prediction.csv",
        output_dir / "metadata_importance_site_prediction.xlsx",
        sheet_name="metadata_importance",
    )
    plot_metadata_importance(
        importance_df, output_dir / "metadata_importance_site_prediction.png"
    )

    shift_matrix = representative_shift_df.pivot(
        index="target_site", columns="metadata_column", values="representative_shift"
    )
    plot_clustered_heatmap(
        shift_matrix,
        output_dir / "metadata_shift_clustermap.png",
        title="Metadata shift clustermap",
        standardize_columns=True,
    )

    meta_heatmap_df = (
        shift_summary_df[
            [
                "target_site",
                "overall_metadata_shift_score",
                "n_metadata_used",
                "mean_continuous_metadata_shift",
                "mean_categorical_metadata_shift",
            ]
        ]
        .drop_duplicates(subset=["target_site"])
        .set_index("target_site")
    )

    sample_size_df = loaded["sample_size_df"]
    entropy_df = loaded["entropy_df"]
    site_distance_df = loaded["site_distance_df"]
    if sample_size_df is not None:
        meta_heatmap_df = meta_heatmap_df.merge(
            sample_size_df.rename(
                columns={
                    "Target site": "target_site",
                    "N total target": "n_total_target",
                    "N source": "n_source",
                }
            )[["target_site", "n_total_target", "n_source"]]
            .drop_duplicates("target_site")
            .set_index("target_site"),
            left_index=True,
            right_index=True,
            how="left",
        )
    if entropy_df is not None:
        meta_heatmap_df = meta_heatmap_df.merge(
            entropy_df[["target_site", "full_target_normalized_entropy"]]
            .rename(columns={"full_target_normalized_entropy": "normalized_entropy"})
            .set_index("target_site"),
            left_index=True,
            right_index=True,
            how="left",
        )
    if site_distance_df is not None:
        meta_heatmap_df = meta_heatmap_df.merge(
            site_distance_df[
                [
                    "target_site",
                    "euclidean_centroid_distance",
                    "mahalanobis_distance",
                    "wasserstein_mean_distance",
                    "jensen_shannon_mean_divergence",
                    "mmd_rbf",
                ]
            ]
            .drop_duplicates("target_site")
            .set_index("target_site"),
            left_index=True,
            right_index=True,
            how="left",
            suffixes=("", "_site"),
        )

    performance_agg = {
        "mean_ood_performance": ("mean_ood_performance", "mean"),
        "id_ood_gap": ("id_ood_gap", "mean"),
    }
    if "ece" in performance_shift_df.columns:
        performance_agg["ece"] = ("ece", "mean")
    elif "mean_ece" in performance_shift_df.columns:
        performance_agg["ece"] = ("mean_ece", "mean")
    performance_site_means = (
        performance_shift_df.groupby("target_site", as_index=False)
        .agg(**performance_agg)
        .set_index("target_site")
    )
    meta_heatmap_df = meta_heatmap_df.merge(
        performance_site_means, left_index=True, right_index=True, how="left"
    )
    plot_clustered_heatmap(
        meta_heatmap_df,
        output_dir / "metadata_performance_meta_clustermap.png",
        title="Metadata + performance meta-clustermap",
        standardize_columns=True,
    )

    metadata_perf_corr_df = build_metadata_performance_correlation_table(
        performance_shift_df
    )
    save_dataframe(
        metadata_perf_corr_df,
        output_dir / "metadata_performance_correlations.csv",
        output_dir / "metadata_performance_correlations.xlsx",
        sheet_name="metadata_corr",
    )

    summary_text = build_metadata_analysis_summary(
        shift_long_df=shift_long_df,
        shift_summary_df=shift_summary_df,
        metadata_label_df=metadata_label_df,
        site_label_df=site_label_df,
        metadata_site_df=metadata_site_df,
        site_predictability_df=site_predictability_df,
        metadata_perf_corr_df=metadata_perf_corr_df,
        metadata_info_df=metadata_info_df,
        warnings=warnings,
    )
    (output_dir / "metadata_analysis_summary.txt").write_text(
        summary_text, encoding="utf-8"
    )

    readme = build_readme(found_inputs, warnings, metadata_info_df)
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")

    print(f"Wrote metadata analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
