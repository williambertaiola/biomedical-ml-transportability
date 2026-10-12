"""Post-experiment analysis: run meta learning analysis."""

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


import json
import math
import os
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXTRA_LIB_PATH = Path(_configured_path("optional_site_packages"))
if EXTRA_LIB_PATH.exists():
    sys.path.insert(0, str(EXTRA_LIB_PATH))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
INPUT_ROOT = Path(_configured_path("results_tcga"))
OUTPUT_ROOT = INPUT_ROOT / "SHAP"
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
SCRIPTS_DIR = OUTPUT_ROOT / "scripts"
SCRIPT_OUTPUT_PATH = SCRIPTS_DIR / "run_meta_learning_analysis.py"
RUN_LOG_PATH = OUTPUT_ROOT / "run_log.txt"
README_PATH = OUTPUT_ROOT / "README.txt"

PATTERN_TABLE = INPUT_ROOT / "Pattern_discovery" / "pattern_discovery_master_table.xlsx"
METADATA_SHIFT_BY_SITE = (
    INPUT_ROOT / "Metadata_discussion" / "metadata_shift_by_target_site.xlsx"
)
METADATA_SHIFT_VS_PERFORMANCE = (
    INPUT_ROOT / "Metadata_discussion" / "metadata_shift_vs_performance.xlsx"
)
SUPERVISED_INPUT = INPUT_ROOT / "Dashi" / "tables" / "supervised_input.csv"
QUALITY_DOMAIN_SUMMARY = INPUT_ROOT / "Quality_report" / "tcga_site_quality_summary.csv"
VARIABLE_INVENTORY = INPUT_ROOT / "Quality_report" / "tcga_metadata_inventory.csv"
BOOTSTRAP_RESULTS = (
    INPUT_ROOT
    / "Methods"
    / "TCGA_BRCA_M4_bootstrap"
    / "tables"
    / "target_site_results.csv"
)


OOD_METRIC_MAP = {
    "roc_auc": "OOD_ROC_AUC",
    "f1": "OOD_F1",
    "balanced_accuracy": "OOD_BALANCED_ACCURACY",
}
DISTANCE_COLUMNS = [
    "euclidean_centroid_distance",
    "mahalanobis_distance",
    "wasserstein_mean_distance",
    "jensen_shannon_mean_divergence",
    "mmd_rbf",
]
NUMERIC_META_CANDIDATES = [
    "sample_size",
    "n_total_target",
    "n_source",
    "n_train_1",
    "n_test_1",
    "n_features",
    "class_entropy",
    "normalized_entropy",
    "class_balance",
    "minority_class_ratio",
    "missing_percentage",
    "site_distance_composite",
    "euclidean_centroid_distance",
    "mahalanobis_distance",
    "wasserstein_mean_distance",
    "jensen_shannon_mean_divergence",
    "mmd_rbf",
    "overall_metadata_shift_score",
    "mean_continuous_metadata_shift",
    "mean_categorical_metadata_shift",
    "mean_total_signal",
    "mean_nonzero_gene_count",
    "mean_zero_fraction",
    "low_total_signal_outliers",
]
CATEGORICAL_CANDIDATES = ["method", "model", "dataset"]
DISPLAY_NAME_MAP = {
    "OOD_ROC_AUC": "OOD ROC-AUC",
    "OOD_F1": "OOD F1-score",
    "OOD_BALANCED_ACCURACY": "OOD Balanced Accuracy",
    "ECE": "OOD Expected Calibration Error",
    "ID_OOD_GAP": "Mean ID-OOD Gap",
}
FILE_STEM_MAP = {
    "OOD_ROC_AUC": "OOD_ROC_AUC",
    "OOD_F1": "OOD_F1",
    "OOD_BALANCED_ACCURACY": "balanced_accuracy",
    "ECE": "ECE",
    "ID_OOD_GAP": "ID_OOD_gap",
}


@dataclass
class AnalysisResult:
    metric: str
    n_rows: int
    n_sites: int
    cv_scheme: str
    numeric_features: list[str]
    categorical_features: list[str]


LOG_MESSAGES: list[str] = []


def log(message: str) -> None:
    print(message)
    LOG_MESSAGES.append(message)


def ensure_output_dirs() -> None:
    for directory in [OUTPUT_ROOT, TABLES_DIR, FIGURES_DIR, SCRIPTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_dataframe(
    df: pd.DataFrame, csv_path: Path, xlsx_path: Path | None = None
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if xlsx_path is not None:
        xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(xlsx_path) as writer:
            df.to_excel(writer, index=False, sheet_name="data")


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def detect_site_column(df: pd.DataFrame) -> str:
    for candidate in ["domain_id", "target_site", "site", "site_id"]:
        if candidate in df.columns:
            return candidate
    raise KeyError("Could not detect site/domain column.")


def detect_target_column(df: pd.DataFrame) -> str:
    for candidate in ["target", "label", "outcome", "class", "y"]:
        if candidate in df.columns:
            return candidate
    raise KeyError("Could not detect target/label column.")


def build_input_inventory() -> pd.DataFrame:
    rows = []
    for label, path in [
        ("pattern_table", PATTERN_TABLE),
        ("metadata_shift_by_site", METADATA_SHIFT_BY_SITE),
        ("metadata_shift_vs_performance", METADATA_SHIFT_VS_PERFORMANCE),
        ("supervised_input", SUPERVISED_INPUT),
        ("quality_domain_summary", QUALITY_DOMAIN_SUMMARY),
        ("variable_inventory", VARIABLE_INVENTORY),
        ("bootstrap_results", BOOTSTRAP_RESULTS),
    ]:
        rows.append(
            {
                "input_name": label,
                "path": str(path),
                "exists": path.exists(),
            }
        )
    inventory_df = pd.DataFrame(rows)
    save_dataframe(
        inventory_df,
        TABLES_DIR / "input_files_used.csv",
        TABLES_DIR / "input_files_used.xlsx",
    )
    return inventory_df


def compute_site_missingness() -> pd.DataFrame:
    df = pd.read_csv(SUPERVISED_INPUT, low_memory=False)
    site_col = detect_site_column(df)
    target_col = detect_target_column(df)
    excluded = {site_col, target_col}
    feature_cols = [column for column in df.columns if column not in excluded]
    rows: list[dict[str, Any]] = []
    for site, group in df.groupby(site_col, dropna=False):
        rows.append(
            {
                "target_site": str(site),
                "n_supervised_features": len(feature_cols),
                "missing_percentage_input": float(
                    group[feature_cols].isna().mean().mean() * 100.0
                ),
                "missing_fraction_input": float(
                    group[feature_cols].isna().mean().mean()
                ),
                "computed_class_1_ratio": float(group[target_col].mean()),
            }
        )
    result = pd.DataFrame(rows)
    result["computed_class_0_ratio"] = 1.0 - result["computed_class_1_ratio"]

    if QUALITY_DOMAIN_SUMMARY.exists():
        quality_df = pd.read_csv(QUALITY_DOMAIN_SUMMARY, low_memory=False)
        quality_site_col = detect_site_column(quality_df)
        missing_pct_columns = [
            column for column in quality_df.columns if column.startswith("missing_pct_")
        ]
        if missing_pct_columns:
            quality_missing = quality_df[
                [quality_site_col, *missing_pct_columns]
            ].copy()
            quality_missing["target_site"] = quality_missing[quality_site_col].astype(
                str
            )
            quality_missing["missing_percentage_quality"] = quality_missing[
                missing_pct_columns
            ].mean(axis=1)
            result = result.merge(
                quality_missing[
                    ["target_site", "missing_percentage_quality"]
                ].drop_duplicates(subset=["target_site"]),
                on="target_site",
                how="left",
            )
        else:
            result["missing_percentage_quality"] = np.nan
    else:
        result["missing_percentage_quality"] = np.nan

    result["missing_percentage"] = result["missing_percentage_quality"].fillna(
        result["missing_percentage_input"]
    )
    result["missing_fraction"] = result["missing_percentage"] / 100.0

    if QUALITY_DOMAIN_SUMMARY.exists():
        quality_df = pd.read_csv(QUALITY_DOMAIN_SUMMARY, low_memory=False)
        quality_site_col = detect_site_column(quality_df)
        keep_columns = [
            quality_site_col,
            "mean_total_signal",
            "mean_nonzero_gene_count",
            "mean_zero_fraction",
            "low_total_signal_outliers",
        ]
        available = [column for column in keep_columns if column in quality_df.columns]
        if len(available) > 1:
            quality_keep = quality_df[available].copy()
            quality_keep["target_site"] = quality_keep[quality_site_col].astype(str)
            result = result.merge(
                quality_keep.drop(
                    columns=[quality_site_col], errors="ignore"
                ).drop_duplicates(subset=["target_site"]),
                on="target_site",
                how="left",
            )
    return result


def build_site_feature_table() -> pd.DataFrame:
    pattern_df = read_table(PATTERN_TABLE)
    base_columns = [
        "dataset",
        "target_site",
        "target_site_display",
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
    site_df = (
        pattern_df[base_columns]
        .drop_duplicates(subset=["dataset", "target_site", "target_site_display"])
        .copy()
    )
    site_df["sample_size"] = site_df["n_total_target"]
    site_df["class_entropy"] = site_df["entropy"]
    site_df["class_balance"] = (
        1.0 - (site_df["class_0_ratio"] - site_df["class_1_ratio"]).abs()
    )
    site_df["minority_class_ratio"] = site_df[["class_0_ratio", "class_1_ratio"]].min(
        axis=1
    )

    missingness_df = compute_site_missingness()
    site_df = site_df.merge(missingness_df, on="target_site", how="left")

    if METADATA_SHIFT_BY_SITE.exists():
        metadata_shift_df = read_table(METADATA_SHIFT_BY_SITE)
        metadata_keep = [
            "target_site",
            "overall_metadata_shift_score",
            "mean_continuous_metadata_shift",
            "mean_categorical_metadata_shift",
            "n_metadata_used",
        ]
        available = [
            column for column in metadata_keep if column in metadata_shift_df.columns
        ]
        site_df = site_df.merge(
            metadata_shift_df[available].drop_duplicates(subset=["target_site"]),
            on="target_site",
            how="left",
        )

    if METADATA_SHIFT_VS_PERFORMANCE.exists():
        metadata_perf_df = read_table(METADATA_SHIFT_VS_PERFORMANCE)
        if "n_features_used" in metadata_perf_df.columns:
            features_df = metadata_perf_df[
                ["target_site", "n_features_used"]
            ].drop_duplicates(subset=["target_site"])
            site_df = site_df.merge(features_df, on="target_site", how="left")
            site_df["n_features"] = site_df["n_features_used"]
        else:
            site_df["n_features_used"] = np.nan
    else:
        site_df["n_features_used"] = np.nan

    if "n_features" not in site_df.columns:
        site_df["n_features"] = np.nan
    site_df["n_features"] = site_df["n_features"].fillna(site_df["n_features_used"])
    if "n_supervised_features" in site_df.columns:
        site_df["n_features"] = site_df["n_features"].fillna(
            site_df["n_supervised_features"]
        )

    zscore_parts = []
    for column in DISTANCE_COLUMNS:
        if column in site_df.columns:
            std = site_df[column].std(ddof=0)
            if pd.notna(std) and std > 0:
                zscore_parts.append((site_df[column] - site_df[column].mean()) / std)
    if zscore_parts:
        site_df["site_distance_composite"] = pd.concat(zscore_parts, axis=1).mean(
            axis=1
        )
    else:
        site_df["site_distance_composite"] = np.nan

    save_dataframe(
        site_df,
        TABLES_DIR / "site_level_metafeatures.csv",
        TABLES_DIR / "site_level_metafeatures.xlsx",
    )
    return site_df


def build_meta_learning_table(site_feature_df: pd.DataFrame) -> pd.DataFrame:
    pattern_df = read_table(PATTERN_TABLE).copy()

    ood_rows = pattern_df.copy()
    ood_rows["metric"] = ood_rows["metric"].map(OOD_METRIC_MAP)
    ood_rows["performance"] = ood_rows["mean_ood_performance"]
    ood_rows["performance_source"] = "mean_ood_performance"
    ood_rows["metric_definition"] = "Observed OOD performance for the original metric."

    group_keys = [
        "dataset",
        "target_site",
        "target_site_display",
        "model",
        "model_display",
        "method",
    ]
    gap_rows = (
        pattern_df.groupby(group_keys, as_index=False)
        .agg(
            performance=("id_ood_gap", "mean"),
            mean_ood_performance=("mean_ood_performance", "mean"),
            std_ood_performance=("std_ood_performance", "mean"),
            mean_id_performance=("mean_id_performance", "mean"),
            ece=("ece", "mean"),
        )
        .copy()
    )
    gap_rows["metric"] = "ID_OOD_GAP"
    gap_rows["performance_source"] = (
        "mean(id_ood_gap across roc_auc, f1, balanced_accuracy)"
    )
    gap_rows["metric_definition"] = (
        "Average of the metric-specific ID-OOD gaps across ROC-AUC, F1, and balanced accuracy."
    )

    ece_rows = (
        pattern_df.groupby(group_keys, as_index=False)
        .agg(
            performance=("ece", "mean"),
            mean_ood_performance=("mean_ood_performance", "mean"),
            std_ood_performance=("std_ood_performance", "mean"),
            mean_id_performance=("mean_id_performance", "mean"),
            id_ood_gap=("id_ood_gap", "mean"),
        )
        .copy()
    )
    ece_rows["metric"] = "ECE"
    ece_rows["ece"] = ece_rows["performance"]
    ece_rows["performance_source"] = "ece"
    ece_rows["metric_definition"] = "Observed OOD expected calibration error."

    long_df = pd.concat([ood_rows, gap_rows, ece_rows], ignore_index=True, sort=False)
    site_merge_df = site_feature_df.drop(
        columns=["dataset", "target_site_display"], errors="ignore"
    )
    long_df = long_df.merge(
        site_merge_df,
        on="target_site",
        how="left",
        suffixes=("", "_site"),
    )

    for column in site_merge_df.columns:
        site_column = f"{column}_site"
        if site_column not in long_df.columns:
            continue
        if column in long_df.columns:
            long_df[column] = long_df[column].fillna(long_df[site_column])
            long_df = long_df.drop(columns=[site_column])
        else:
            long_df = long_df.rename(columns={site_column: column})

    ordered_columns = [
        "dataset",
        "target_site",
        "target_site_display",
        "method",
        "model",
        "model_display",
        "metric",
        "performance",
        "performance_source",
        "metric_definition",
        "mean_ood_performance",
        "std_ood_performance",
        "mean_id_performance",
        "id_ood_gap",
        "ece",
        "sample_size",
        "n_total_target",
        "n_source",
        "n_train_1",
        "n_test_1",
        "n_features",
        "class_0_ratio",
        "class_1_ratio",
        "class_balance",
        "minority_class_ratio",
        "class_entropy",
        "normalized_entropy",
        "missing_percentage",
        "mean_total_signal",
        "mean_nonzero_gene_count",
        "mean_zero_fraction",
        "low_total_signal_outliers",
        "site_distance_composite",
        *DISTANCE_COLUMNS,
        "overall_metadata_shift_score",
        "mean_continuous_metadata_shift",
        "mean_categorical_metadata_shift",
    ]
    ordered_columns = [
        column for column in ordered_columns if column in long_df.columns
    ]
    long_df = long_df[ordered_columns].copy()
    return long_df


def choose_splits(
    df: pd.DataFrame,
) -> tuple[list[tuple[np.ndarray, np.ndarray]] | None, str]:
    unique_sites = df["target_site"].nunique()
    if unique_sites >= 3:
        n_splits = min(4, unique_sites)
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(df, df["performance"], groups=df["target_site"]))
        return splits, f"GroupKFold(n_splits={n_splits}, groups=target_site)"
    if len(df) >= 5:
        n_splits = min(5, len(df))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = list(splitter.split(df))
        return splits, f"KFold(n_splits={n_splits}, shuffle=True)"
    return None, "In-sample fit only (too few observations for CV)"


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def cross_validated_predictions(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]] | None,
) -> np.ndarray:
    if not splits:
        fitted = clone(estimator).fit(X, y)
        return np.asarray(fitted.predict(X), dtype=float)

    predictions = np.full(len(X), np.nan, dtype=float)
    for train_idx, test_idx in splits:
        fitted = clone(estimator).fit(X.iloc[train_idx], y[train_idx])
        predictions[test_idx] = np.asarray(
            fitted.predict(X.iloc[test_idx]), dtype=float
        )
    return predictions


def compute_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }


def sanitize_metric_for_sheet(metric: str) -> str:
    return metric.replace("OOD_", "").replace("_", " ")[:31]


def fit_and_explain_metric(
    metric_name: str, df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, AnalysisResult]:
    numeric_features = [
        column
        for column in NUMERIC_META_CANDIDATES
        if column in df.columns and df[column].notna().any()
    ]
    categorical_features = [
        column
        for column in CATEGORICAL_CANDIDATES
        if column in df.columns and df[column].nunique(dropna=True) > 1
    ]

    X = df[numeric_features + categorical_features].copy()
    y = df["performance"].to_numpy(dtype=float)
    splits, cv_scheme = choose_splits(df)

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    ridge_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=500,
                    random_state=SEED,
                    min_samples_leaf=2,
                    n_jobs=1,
                ),
            ),
        ]
    )

    ridge_cv_pred = cross_validated_predictions(ridge_pipeline, X, y, splits)
    rf_cv_pred = cross_validated_predictions(rf_pipeline, X, y, splits)

    ridge_metrics = compute_model_metrics(y, ridge_cv_pred)
    rf_metrics = compute_model_metrics(y, rf_cv_pred)

    ridge_fitted = clone(ridge_pipeline).fit(X, y)
    rf_fitted = clone(rf_pipeline).fit(X, y)

    ridge_preprocessor = ridge_fitted.named_steps["preprocessor"]
    rf_preprocessor = rf_fitted.named_steps["preprocessor"]
    ridge_regressor = ridge_fitted.named_steps["regressor"]
    rf_regressor = rf_fitted.named_steps["regressor"]

    feature_names = ridge_preprocessor.get_feature_names_out().tolist()
    coefficients_df = pd.DataFrame(
        {
            "metric": metric_name,
            "feature": feature_names,
            "ridge_coefficient": ridge_regressor.coef_,
            "ridge_abs_coefficient": np.abs(ridge_regressor.coef_),
            "n_rows": len(df),
            "n_sites": df["target_site"].nunique(),
            "cv_scheme": cv_scheme,
        }
    ).sort_values("ridge_abs_coefficient", ascending=False)

    transformed = rf_preprocessor.transform(X)
    transformed_df = pd.DataFrame(
        transformed, columns=rf_preprocessor.get_feature_names_out(), index=df.index
    )
    explainer = shap.TreeExplainer(rf_regressor)
    shap_values = explainer.shap_values(transformed_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values, dtype=float)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importances_df = pd.DataFrame(
        {
            "metric": metric_name,
            "feature": transformed_df.columns,
            "rf_importance": rf_regressor.feature_importances_,
            "mean_abs_shap": mean_abs_shap,
            "n_rows": len(df),
            "n_sites": df["target_site"].nunique(),
            "cv_scheme": cv_scheme,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    metric_stem = FILE_STEM_MAP.get(metric_name, metric_name)
    display_name = DISPLAY_NAME_MAP.get(metric_name, metric_name)

    shap.summary_plot(
        shap_values,
        transformed_df,
        show=False,
        max_display=min(20, transformed_df.shape[1]),
    )
    plt.title(f"SHAP | {display_name}")
    plt.xlabel("SHAP value")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"shap_beeswarm_{metric_stem}.png", dpi=220)
    plt.close()

    shap.summary_plot(
        shap_values,
        transformed_df,
        plot_type="bar",
        show=False,
        max_display=min(20, transformed_df.shape[1]),
    )
    plt.title(f"SHAP barplot | {display_name}")
    plt.xlabel("mean |SHAP|")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"shap_bar_{metric_stem}.png", dpi=220)
    plt.close()

    alias_map = {
        "OOD_ROC_AUC": FIGURES_DIR / "shap_beeswarm_OOD_ROC_AUC.png",
        "OOD_F1": FIGURES_DIR / "shap_beeswarm_OOD_F1.png",
        "OOD_BALANCED_ACCURACY": FIGURES_DIR / "shap_beeswarm_balanced_accuracy.png",
        "ECE": FIGURES_DIR / "shap_beeswarm_ECE.png",
        "ID_OOD_GAP": FIGURES_DIR / "shap_beeswarm_ID_OOD_gap.png",
    }
    alias_path = alias_map.get(metric_name)
    source_path = FIGURES_DIR / f"shap_beeswarm_{metric_stem}.png"
    if alias_path is not None and alias_path.resolve() != source_path.resolve():
        shutil.copy2(source_path, alias_path)

    performance_rows = pd.DataFrame(
        [
            {
                "metric": metric_name,
                "model_type": "Ridge",
                "r2": ridge_metrics["r2"],
                "mae": ridge_metrics["mae"],
                "rmse": ridge_metrics["rmse"],
                "n_rows": len(df),
                "n_sites": df["target_site"].nunique(),
                "cv_scheme": cv_scheme,
            },
            {
                "metric": metric_name,
                "model_type": "RandomForestRegressor",
                "r2": rf_metrics["r2"],
                "mae": rf_metrics["mae"],
                "rmse": rf_metrics["rmse"],
                "n_rows": len(df),
                "n_sites": df["target_site"].nunique(),
                "cv_scheme": cv_scheme,
            },
        ]
    )

    analysis_result = AnalysisResult(
        metric=metric_name,
        n_rows=len(df),
        n_sites=df["target_site"].nunique(),
        cv_scheme=cv_scheme,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    return (
        coefficients_df,
        pd.concat([performance_rows, importances_df], ignore_index=False),
        analysis_result,
    )


def split_performance_and_importances(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    performance_mask = (
        df["model_type"].notna()
        if "model_type" in df.columns
        else pd.Series(False, index=df.index)
    )
    performance_df = df.loc[performance_mask].copy()
    importance_df = df.loc[~performance_mask].copy()
    return performance_df, importance_df


def write_summary(
    analysis_results: list[AnalysisResult], master_table: pd.DataFrame
) -> None:
    available_metrics = ", ".join(
        master_table["metric"].dropna().astype(str).unique().tolist()
    )
    summary = f"""
    Meta-learning analysis built from existing cross-site transportability outputs.

    Input root:
    {INPUT_ROOT}

    Output root:
    {OUTPUT_ROOT}

    Main source tables used:
    - {PATTERN_TABLE}
    - {METADATA_SHIFT_BY_SITE}
    - {METADATA_SHIFT_VS_PERFORMANCE}
    - {SUPERVISED_INPUT}

    Long-table metrics:
    {available_metrics}

    Important definition:
    - ID_OOD_GAP is defined as the mean of the metric-specific gaps across ROC-AUC, F1, and balanced accuracy.
    - ECE is taken from the already aggregated OOD calibration results.

    Modeled metrics:
    {os.linesep.join(f"- {result.metric}: n_rows={result.n_rows}, n_sites={result.n_sites}, cv={result.cv_scheme}" for result in analysis_results)}

    Numeric meta-features considered:
    {", ".join(NUMERIC_META_CANDIDATES)}

    Categorical variables considered when informative:
    {", ".join(CATEGORICAL_CANDIDATES)}

    Notes:
    - No original model training on raw biomedical data was rerun.
    - The analysis operates purely at meta-learning level using existing site-level experiment outputs.
    - The regression/SHAP outputs should still be interpreted as meta-analytic and descriptive rather than causal.
    """
    save_text(README_PATH, textwrap.dedent(summary).strip() + "\n")


def main() -> None:
    ensure_output_dirs()
    inventory_df = build_input_inventory()
    missing_inputs = inventory_df.loc[~inventory_df["exists"], "input_name"].tolist()
    if missing_inputs:
        log(f"Missing optional inputs: {', '.join(missing_inputs)}")
    else:
        log("All expected input files were found.")

    site_feature_df = build_site_feature_table()
    master_long_df = build_meta_learning_table(site_feature_df)
    save_dataframe(
        master_long_df,
        TABLES_DIR / "meta_learning_table_long.csv",
        TABLES_DIR / "meta_learning_table_long.xlsx",
    )

    coefficients_frames: list[pd.DataFrame] = []
    performance_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    analysis_results: list[AnalysisResult] = []

    for metric_name in [
        "OOD_ROC_AUC",
        "OOD_F1",
        "OOD_BALANCED_ACCURACY",
        "ECE",
        "ID_OOD_GAP",
    ]:
        metric_df = master_long_df.loc[master_long_df["metric"] == metric_name].copy()
        metric_df = metric_df.loc[metric_df["performance"].notna()].reset_index(
            drop=True
        )
        if metric_df.empty:
            log(f"Skipping {metric_name}: no rows available.")
            continue

        log(f"Running meta-learning models for {metric_name} on {len(metric_df)} rows.")
        coefficients_df, perf_and_imp_df, analysis_result = fit_and_explain_metric(
            metric_name, metric_df
        )
        performance_df, importance_df = split_performance_and_importances(
            perf_and_imp_df
        )

        coefficients_frames.append(coefficients_df)
        performance_frames.append(performance_df)
        importance_frames.append(importance_df)
        analysis_results.append(analysis_result)

    coefficients_all = (
        pd.concat(coefficients_frames, ignore_index=True)
        if coefficients_frames
        else pd.DataFrame()
    )
    performance_all = (
        pd.concat(performance_frames, ignore_index=True)
        if performance_frames
        else pd.DataFrame()
    )
    importance_all = (
        pd.concat(importance_frames, ignore_index=True)
        if importance_frames
        else pd.DataFrame()
    )

    if not performance_all.empty:
        performance_all = performance_all[
            [
                "metric",
                "model_type",
                "r2",
                "mae",
                "rmse",
                "n_rows",
                "n_sites",
                "cv_scheme",
            ]
        ].copy()
    if not importance_all.empty:
        importance_all = importance_all[
            [
                "metric",
                "feature",
                "rf_importance",
                "mean_abs_shap",
                "n_rows",
                "n_sites",
                "cv_scheme",
            ]
        ].copy()

    save_dataframe(
        coefficients_all,
        TABLES_DIR / "regression_coefficients_by_metric.csv",
        TABLES_DIR / "regression_coefficients_by_metric.xlsx",
    )
    save_dataframe(
        performance_all,
        TABLES_DIR / "model_performance_by_metric.csv",
        TABLES_DIR / "model_performance_by_metric.xlsx",
    )
    save_dataframe(
        importance_all,
        TABLES_DIR / "feature_importance_by_metric.csv",
        TABLES_DIR / "feature_importance_by_metric.xlsx",
    )

    summary_payload = {
        "input_root": str(INPUT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "n_master_rows": int(len(master_long_df)),
        "metrics_modeled": [result.metric for result in analysis_results],
    }
    save_text(OUTPUT_ROOT / "summary.json", json.dumps(summary_payload, indent=2))
    write_summary(analysis_results, master_long_df)

    script_source = Path(__file__).resolve()
    shutil.copy2(script_source, SCRIPT_OUTPUT_PATH)
    save_text(RUN_LOG_PATH, "\n".join(LOG_MESSAGES) + "\n")
    log(f"Meta-learning analysis completed. Results saved to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
