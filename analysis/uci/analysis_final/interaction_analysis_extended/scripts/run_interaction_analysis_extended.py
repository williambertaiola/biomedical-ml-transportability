"""Post-experiment analysis: run interaction analysis extended."""

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
from pathlib import Path

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
import statsmodels.formula.api as smf
from matplotlib.colors import ListedColormap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
INPUT_ROOT = Path(_configured_path("results_uci"))
SOURCE_META_TABLE = INPUT_ROOT / "SHAP" / "tables" / "meta_learning_table_long.csv"
SOURCE_SHAP_SCRIPT = INPUT_ROOT / "SHAP" / "scripts" / "run_meta_learning_analysis.py"

OUTPUT_ROOT = INPUT_ROOT / "analysis_final" / "Interaction_analysis_extended"
TABLES_DIR = OUTPUT_ROOT / "tables"
SHAP_PLOTS_DIR = OUTPUT_ROOT / "shap_dependence_plots"
BARPLOTS_DIR = OUTPUT_ROOT / "shap_interaction_barplots"
INTERACTION_PLOTS_DIR = OUTPUT_ROOT / "interaction_plots"
PDP_DIR = OUTPUT_ROOT / "partial_dependence_plots"
MAPS_DIR = OUTPUT_ROOT / "method_recommendation_maps"
SCRIPTS_DIR = OUTPUT_ROOT / "scripts"
README_PATH = OUTPUT_ROOT / "README.txt"
REPORT_PATH = OUTPUT_ROOT / "interaction_analysis_report.txt"
RUN_LOG_PATH = OUTPUT_ROOT / "run_log.txt"
SCRIPT_OUTPUT_PATH = SCRIPTS_DIR / "run_interaction_analysis_extended.py"
SUMMARY_JSON_PATH = OUTPUT_ROOT / "summary.json"

TARGET_METRIC = "OOD_ROC_AUC"
DISPLAY_METRIC = "OOD ROC-AUC"
MAX_MISSING_RATE = 0.40
TOP_INTERACTIONS_PER_METHOD = 5
DEPENDENCE_PLOTS_PER_METHOD = 3
METHOD_ORDER = ["M0", "M1", "M2", "M3", "M4"]
METHOD_LABELS = {
    "M0": "Baseline",
    "M1": "ComBat",
    "M2": "Domain weighting",
    "M3": "Transductive",
    "M4": "ComBat + Transductive",
}
MODEL_MARKERS = {
    "HistGradientBoosting": "o",
    "LogisticRegression": "s",
    "RandomForest": "^",
}
METHOD_COLORS = {
    "M0": "#4E79A7",
    "M1": "#F28E2B",
    "M2": "#59A14F",
    "M3": "#E15759",
    "M4": "#B07AA1",
}

NUMERIC_PREDICTOR_CANDIDATES = [
    "sample_size",
    "n_total_target",
    "n_source",
    "n_train_1",
    "n_test_1",
    "n_features",
    "class_balance",
    "minority_class_ratio",
    "class_entropy",
    "normalized_entropy",
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
]
CATEGORICAL_PREDICTOR_CANDIDATES = ["method", "model", "dataset"]
DIRECT_LEAKAGE_COLUMNS = {
    "performance",
    "performance_source",
    "metric_definition",
    "mean_ood_performance",
    "std_ood_performance",
    "mean_id_performance",
    "id_ood_gap",
    "ece",
}
NON_PREDICTOR_COLUMNS = {
    "target_site",
    "target_site_display",
    "model_display",
    "metric",
    "class_0_ratio",
    "class_1_ratio",
}
PREFERRED_INTERACTION_FEATURES = [
    "overall_metadata_shift_score",
    "site_distance_composite",
    "sample_size",
    "class_entropy",
    "class_balance",
    "mean_continuous_metadata_shift",
]
RECOMMENDATION_FEATURES = [
    "overall_metadata_shift_score",
    "sample_size",
    "class_entropy",
    "site_distance_composite",
]
LOW_HIGH_GRID_FEATURES = RECOMMENDATION_FEATURES.copy()

LOG_MESSAGES: list[str] = []


def log(message: str) -> None:
    print(message)
    LOG_MESSAGES.append(message)


def ensure_output_dirs() -> None:
    for directory in [
        OUTPUT_ROOT,
        TABLES_DIR,
        SHAP_PLOTS_DIR,
        BARPLOTS_DIR,
        INTERACTION_PLOTS_DIR,
        PDP_DIR,
        MAPS_DIR,
        SCRIPTS_DIR,
    ]:
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
        df.to_excel(xlsx_path, index=False)


def humanize_feature_name(name: str) -> str:
    clean = (
        name.replace("num__", "")
        .replace("cat__", "")
        .replace("method_", "")
        .replace("_", " ")
    )
    return clean.strip().title()


def load_source_table() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_META_TABLE)
    log(f"Loaded source meta-learning table: {SOURCE_META_TABLE}")
    log(f"Shape: {df.shape}")
    return df


def deduplicate_numeric_columns(
    df: pd.DataFrame, candidate_columns: list[str]
) -> tuple[list[str], list[dict[str, object]]]:
    kept: list[str] = []
    seen_vectors: dict[tuple[object, ...], str] = {}
    notes: list[dict[str, object]] = []
    for column in candidate_columns:
        vector = tuple(df[column].fillna("__NA__").tolist())
        if vector in seen_vectors:
            notes.append(
                {
                    "column_name": column,
                    "status": "excluded",
                    "reason": f"exact_duplicate_of:{seen_vectors[vector]}",
                }
            )
            continue
        seen_vectors[vector] = column
        kept.append(column)
    return kept, notes


def build_clean_analysis_dataframe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    working_df = df.loc[df["metric"] == TARGET_METRIC].copy()
    working_df = working_df.loc[working_df["performance"].notna()].reset_index(
        drop=True
    )

    predictor_notes: list[dict[str, object]] = []

    numeric_candidates = []
    for column in NUMERIC_PREDICTOR_CANDIDATES:
        if column not in working_df.columns:
            predictor_notes.append(
                {
                    "column_name": column,
                    "status": "excluded",
                    "reason": "missing_from_table",
                }
            )
            continue
        values = pd.to_numeric(working_df[column], errors="coerce")
        missing_rate = float(values.isna().mean())
        if missing_rate > MAX_MISSING_RATE:
            predictor_notes.append(
                {
                    "column_name": column,
                    "status": "excluded",
                    "reason": f"missing_rate>{MAX_MISSING_RATE:.2f}",
                }
            )
            continue
        if values.nunique(dropna=True) <= 1:
            predictor_notes.append(
                {
                    "column_name": column,
                    "status": "excluded",
                    "reason": "constant_column",
                }
            )
            continue
        numeric_candidates.append(column)

    numeric_features, duplicate_notes = deduplicate_numeric_columns(
        working_df, numeric_candidates
    )
    predictor_notes.extend(duplicate_notes)
    for column in numeric_features:
        predictor_notes.append(
            {"column_name": column, "status": "kept", "reason": "numeric_predictor"}
        )

    categorical_features: list[str] = []
    for column in CATEGORICAL_PREDICTOR_CANDIDATES:
        if column not in working_df.columns:
            predictor_notes.append(
                {
                    "column_name": column,
                    "status": "excluded",
                    "reason": "missing_from_table",
                }
            )
            continue
        if working_df[column].nunique(dropna=True) <= 1:
            predictor_notes.append(
                {
                    "column_name": column,
                    "status": "excluded",
                    "reason": "single_unique_value",
                }
            )
            continue
        categorical_features.append(column)
        predictor_notes.append(
            {"column_name": column, "status": "kept", "reason": "categorical_predictor"}
        )

    explicitly_excluded = sorted(
        DIRECT_LEAKAGE_COLUMNS.union(NON_PREDICTOR_COLUMNS).intersection(
            working_df.columns
        )
    )
    for column in explicitly_excluded:
        predictor_notes.append(
            {
                "column_name": column,
                "status": "excluded",
                "reason": "not_used_as_predictor",
            }
        )

    ordered_columns = [
        "dataset",
        "target_site",
        "target_site_display",
        "method",
        "model",
        "model_display",
        "metric",
        "performance",
    ] + numeric_features
    clean_df = working_df[ordered_columns].copy()

    inventory_df = (
        pd.DataFrame(predictor_notes)
        .drop_duplicates()
        .sort_values(["status", "column_name"])
        .reset_index(drop=True)
    )
    return clean_df, inventory_df, numeric_features, categorical_features


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


def fit_meta_model(
    clean_df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]
) -> tuple[Pipeline, pd.DataFrame, np.ndarray, np.ndarray | None]:
    X = clean_df[numeric_features + categorical_features].copy()
    y = clean_df["performance"].to_numpy(dtype=float)
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features),
            ),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=700,
                    random_state=SEED,
                    min_samples_leaf=2,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(X, y)

    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]
    transformed = preprocessor.transform(X)
    transformed_df = pd.DataFrame(
        transformed, columns=preprocessor.get_feature_names_out(), index=clean_df.index
    )

    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(transformed_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values, dtype=float)

    interaction_values = None
    try:
        interaction_values = explainer.shap_interaction_values(transformed_df)
        if isinstance(interaction_values, list):
            interaction_values = interaction_values[0]
        interaction_values = np.asarray(interaction_values, dtype=float)
    except Exception as exc:  # pragma: no cover - fallback
        log(f"SHAP interaction values were not available: {exc}")

    return pipeline, transformed_df, shap_values, interaction_values


def method_feature_columns(transformed_df: pd.DataFrame) -> list[str]:
    return [
        column for column in transformed_df.columns if column.startswith("cat__method_")
    ]


def numeric_feature_columns(transformed_df: pd.DataFrame) -> list[str]:
    return [column for column in transformed_df.columns if column.startswith("num__")]


def build_shap_interaction_tables(
    transformed_df: pd.DataFrame, interaction_values: np.ndarray | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if interaction_values is None:
        empty = pd.DataFrame(
            columns=[
                "method",
                "method_feature",
                "metafeature",
                "mean_abs_shap_interaction",
                "rank_within_method",
                "global_rank",
            ]
        )
        return empty, empty

    rows: list[dict[str, object]] = []
    method_features = method_feature_columns(transformed_df)
    numeric_features = numeric_feature_columns(transformed_df)

    for method_feature in method_features:
        method_idx = transformed_df.columns.get_loc(method_feature)
        method_name = method_feature.replace("cat__method_", "")
        for feature in numeric_features:
            feature_idx = transformed_df.columns.get_loc(feature)
            mean_abs = float(
                np.mean(np.abs(interaction_values[:, method_idx, feature_idx]))
            )
            rows.append(
                {
                    "method": method_name,
                    "method_feature": method_feature,
                    "metafeature": feature.replace("num__", ""),
                    "mean_abs_shap_interaction": mean_abs,
                }
            )

    rankings_df = (
        pd.DataFrame(rows)
        .sort_values(["method", "mean_abs_shap_interaction"], ascending=[True, False])
        .reset_index(drop=True)
    )
    rankings_df["rank_within_method"] = rankings_df.groupby("method").cumcount() + 1
    rankings_df = rankings_df.sort_values(
        "mean_abs_shap_interaction", ascending=False
    ).reset_index(drop=True)
    rankings_df["global_rank"] = np.arange(1, len(rankings_df) + 1)

    summary_df = (
        rankings_df.groupby("method", as_index=False)
        .head(TOP_INTERACTIONS_PER_METHOD)
        .reset_index(drop=True)
    )
    return rankings_df, summary_df


def plot_shap_interaction_bars(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    for method in METHOD_ORDER:
        method_df = summary_df.loc[summary_df["method"] == method].copy()
        if method_df.empty:
            continue
        method_df = method_df.sort_values("mean_abs_shap_interaction", ascending=True)
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.barh(
            method_df["metafeature"].str.replace("_", " "),
            method_df["mean_abs_shap_interaction"],
            color=METHOD_COLORS.get(method, "#4E79A7"),
        )
        ax.set_title(
            f"Top SHAP interactions | {method}: {METHOD_LABELS.get(method, method)}"
        )
        ax.set_xlabel("mean |SHAP interaction|")
        ax.set_ylabel("Metafeature")
        fig.tight_layout()
        fig.savefig(BARPLOTS_DIR / f"shap_interaction_barplot_{method}.png", dpi=220)
        plt.close(fig)


def plot_shap_dependence(
    transformed_df: pd.DataFrame,
    shap_values: np.ndarray,
    rankings_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        top_df = rankings_df.loc[rankings_df["method"] == method].head(
            DEPENDENCE_PLOTS_PER_METHOD
        )
        method_feature = f"cat__method_{method}"
        if method_feature not in transformed_df.columns:
            continue
        for _, row in top_df.iterrows():
            feature = f"num__{row['metafeature']}"
            if feature not in transformed_df.columns:
                continue
            output_path = (
                SHAP_PLOTS_DIR / f"shap_dependence_{method}_{row['metafeature']}.png"
            )
            plt.figure(figsize=(7.8, 5.8))
            shap.dependence_plot(
                feature,
                shap_values,
                transformed_df,
                interaction_index=method_feature,
                show=False,
                alpha=0.8,
            )
            plt.title(
                f"SHAP dependence | {DISPLAY_METRIC} | {method} x {humanize_feature_name(feature)}"
            )
            plt.xlabel(humanize_feature_name(feature))
            plt.tight_layout()
            plt.savefig(output_path, dpi=220)
            plt.close()
            plt.close("all")
            rows.append(
                {
                    "method": method,
                    "metafeature": row["metafeature"],
                    "plot_path": str(output_path),
                }
            )
    return pd.DataFrame(rows)


def selected_regression_features(
    rankings_df: pd.DataFrame, numeric_features: list[str]
) -> list[str]:
    available = [
        feature
        for feature in PREFERRED_INTERACTION_FEATURES
        if feature in numeric_features
    ]
    if available:
        return available
    top_ranked = rankings_df["metafeature"].dropna().astype(str).unique().tolist()
    return [feature for feature in top_ranked if feature in numeric_features][:6]


def extract_ols_terms(
    fitted_model,
    *,
    feature_name: str,
    model_spec: str,
) -> list[dict[str, object]]:
    interaction_terms = [
        term
        for term in fitted_model.params.index
        if "C(method)" in term and f":{feature_name}" in term
    ]
    rows: list[dict[str, object]] = []
    omnibus_p = math.nan
    if interaction_terms:
        try:
            constraint = ", ".join(f"{term} = 0" for term in interaction_terms)
            omnibus_p = float(fitted_model.f_test(constraint).pvalue)
        except Exception:
            omnibus_p = math.nan
    for term in interaction_terms:
        coefficient = float(fitted_model.params[term])
        ci_low, ci_high = [
            float(value) for value in fitted_model.conf_int().loc[term].tolist()
        ]
        p_value = float(fitted_model.pvalues[term])
        method_name = (
            term.split("[T.", 1)[1].split("]", 1)[0] if "[T." in term else "unknown"
        )
        direction = "positive" if coefficient > 0 else "negative"
        rows.append(
            {
                "meta_feature": feature_name,
                "model_spec": model_spec,
                "interaction_term": term,
                "method": method_name,
                "coefficient": coefficient,
                "p_value": p_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "direction": direction,
                "r2": float(fitted_model.rsquared),
                "adjusted_r2": float(fitted_model.rsquared_adj),
                "aic": float(fitted_model.aic),
                "bic": float(fitted_model.bic),
                "n_obs": int(fitted_model.nobs),
                "omnibus_interaction_p_value": omnibus_p,
            }
        )
    return rows


def run_regression_interactions(
    clean_df: pd.DataFrame, interaction_features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict[str, object]] = []
    feature_summary_rows: list[dict[str, object]] = []

    has_dataset_adjustment = (
        "dataset" in clean_df.columns and clean_df["dataset"].nunique(dropna=True) > 1
    )
    has_model_adjustment = (
        "model" in clean_df.columns and clean_df["model"].nunique(dropna=True) > 1
    )

    for feature in interaction_features:
        reg_df = (
            clean_df[["performance", "method", "model", "dataset", feature]]
            .copy()
            .dropna()
            .reset_index(drop=True)
        )
        if reg_df.empty or reg_df[feature].nunique(dropna=True) <= 1:
            continue

        simple_formula = f"performance ~ C(method) * {feature}"
        adjusted_formula = f"performance ~ C(method) * {feature}"
        if has_model_adjustment:
            adjusted_formula += " + C(model)"
        if has_dataset_adjustment:
            adjusted_formula += " + C(dataset)"

        try:
            simple_fit = smf.ols(simple_formula, data=reg_df).fit(cov_type="HC3")
            all_rows.extend(
                extract_ols_terms(
                    simple_fit, feature_name=feature, model_spec="unadjusted"
                )
            )
        except Exception as exc:  # pragma: no cover
            log(f"Unadjusted interaction regression failed for {feature}: {exc}")
            continue

        try:
            adjusted_fit = smf.ols(adjusted_formula, data=reg_df).fit(cov_type="HC3")
            adjusted_rows = extract_ols_terms(
                adjusted_fit, feature_name=feature, model_spec="adjusted"
            )
            all_rows.extend(adjusted_rows)
        except Exception as exc:  # pragma: no cover
            log(f"Adjusted interaction regression failed for {feature}: {exc}")
            adjusted_rows = []

        feature_specific_rows = [
            row
            for row in all_rows
            if row["meta_feature"] == feature and row["model_spec"] == "adjusted"
        ]
        if not feature_specific_rows:
            feature_specific_rows = [
                row for row in all_rows if row["meta_feature"] == feature
            ]
        if feature_specific_rows:
            ranked = sorted(
                feature_specific_rows,
                key=lambda row: (math.isnan(row["p_value"]), abs(row["coefficient"])),
                reverse=False,
            )
            strongest = ranked[0]
            feature_summary_rows.append(
                {
                    "meta_feature": feature,
                    "strongest_interaction_term": strongest["interaction_term"],
                    "method": strongest["method"],
                    "coefficient": strongest["coefficient"],
                    "p_value": strongest["p_value"],
                    "direction": strongest["direction"],
                    "model_spec": strongest["model_spec"],
                    "omnibus_interaction_p_value": strongest[
                        "omnibus_interaction_p_value"
                    ],
                }
            )

    results_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(feature_summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["p_value", "omnibus_interaction_p_value"], na_position="last"
        ).reset_index(drop=True)
    return results_df, summary_df


def plot_interaction_regressions(
    clean_df: pd.DataFrame, interaction_features: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in interaction_features:
        plot_df = (
            clean_df[["performance", "method", "model", feature]]
            .copy()
            .dropna()
            .reset_index(drop=True)
        )
        if plot_df.empty or plot_df[feature].nunique(dropna=True) <= 1:
            continue

        formula = f"performance ~ C(method) * {feature}"
        if plot_df["model"].nunique(dropna=True) > 1:
            formula += " + C(model)"
        try:
            fitted = smf.ols(formula, data=plot_df).fit(cov_type="HC3")
        except Exception as exc:  # pragma: no cover
            log(f"Interaction plot model failed for {feature}: {exc}")
            continue

        feature_grid = np.linspace(plot_df[feature].min(), plot_df[feature].max(), 60)
        fig, ax = plt.subplots(figsize=(8.0, 5.8))

        for model_name, marker in MODEL_MARKERS.items():
            model_subset = plot_df.loc[plot_df["model"] == model_name]
            if model_subset.empty:
                continue
            for method in METHOD_ORDER:
                subset = model_subset.loc[model_subset["method"] == method]
                if subset.empty:
                    continue
                ax.scatter(
                    subset[feature],
                    subset["performance"],
                    color=METHOD_COLORS.get(method, "#4E79A7"),
                    marker=marker,
                    s=58,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.5,
                )

        for method in METHOD_ORDER:
            pred_rows = []
            for model_name in sorted(plot_df["model"].dropna().unique().tolist()):
                pred_rows.extend(
                    [
                        {feature: grid_value, "method": method, "model": model_name}
                        for grid_value in feature_grid
                    ]
                )
            pred_df = pd.DataFrame(pred_rows)
            mean_predictions = (
                pred_df.assign(pred=fitted.predict(pred_df))
                .groupby(feature, as_index=False)["pred"]
                .mean()
            )
            ax.plot(
                mean_predictions[feature],
                mean_predictions["pred"],
                color=METHOD_COLORS.get(method, "#4E79A7"),
                linewidth=2.2,
                label=f"{method}: {METHOD_LABELS.get(method, method)}",
            )

        ax.set_title(
            f"Interaction plot | {DISPLAY_METRIC} | {feature.replace('_', ' ')}"
        )
        ax.set_xlabel(feature.replace("_", " ").title())
        ax.set_ylabel(DISPLAY_METRIC)
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
        fig.tight_layout()
        output_path = INTERACTION_PLOTS_DIR / f"interaction_plot_{feature}.png"
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        rows.append({"meta_feature": feature, "plot_path": str(output_path)})

    return pd.DataFrame(rows)


def fit_method_specific_rf(
    method_df: pd.DataFrame, numeric_features: list[str]
) -> Pipeline | None:
    subset_numeric = [
        feature
        for feature in numeric_features
        if feature in method_df.columns and method_df[feature].nunique(dropna=True) > 1
    ]
    subset_categorical = [
        feature
        for feature in ["model"]
        if feature in method_df.columns and method_df[feature].nunique(dropna=True) > 1
    ]
    if len(method_df) < 6 or not subset_numeric:
        return None
    X = method_df[subset_numeric + subset_categorical].copy()
    y = method_df["performance"].to_numpy(dtype=float)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(subset_numeric, subset_categorical)),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=400,
                    random_state=SEED,
                    min_samples_leaf=2,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(X, y)
    pipeline._subset_numeric = subset_numeric  # type: ignore[attr-defined]
    pipeline._subset_categorical = subset_categorical  # type: ignore[attr-defined]
    return pipeline


def generate_partial_dependence(
    clean_df: pd.DataFrame, numeric_features: list[str], interaction_features: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in interaction_features:
        fig, ax = plt.subplots(figsize=(8.0, 5.8))
        plotted = False
        for method in METHOD_ORDER:
            method_df = clean_df.loc[clean_df["method"] == method].copy()
            pipeline = fit_method_specific_rf(method_df, numeric_features)
            if pipeline is None:
                continue
            if feature not in getattr(pipeline, "_subset_numeric", []):
                continue
            X_base = method_df[
                getattr(pipeline, "_subset_numeric")
                + getattr(pipeline, "_subset_categorical")
            ].copy()
            grid = np.linspace(method_df[feature].min(), method_df[feature].max(), 40)
            pd_values = []
            for grid_value in grid:
                X_eval = X_base.copy()
                X_eval[feature] = grid_value
                predictions = pipeline.predict(X_eval)
                pd_values.append(float(np.mean(predictions)))
            ax.plot(
                grid,
                pd_values,
                color=METHOD_COLORS.get(method, "#4E79A7"),
                linewidth=2.2,
                label=f"{method}: {METHOD_LABELS.get(method, method)}",
            )
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(
            f"Partial dependence by method | {DISPLAY_METRIC} | {feature.replace('_', ' ')}"
        )
        ax.set_xlabel(feature.replace("_", " ").title())
        ax.set_ylabel(f"Predicted {DISPLAY_METRIC}")
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        output_path = PDP_DIR / f"partial_dependence_{feature}.png"
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        rows.append({"meta_feature": feature, "plot_path": str(output_path)})
    return pd.DataFrame(rows)


def fit_full_recommendation_model(
    clean_df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    X = clean_df[numeric_features + categorical_features].copy()
    y = clean_df["performance"].to_numpy(dtype=float)
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features),
            ),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=800,
                    random_state=SEED,
                    min_samples_leaf=2,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def build_typical_feature_row(
    clean_df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]
) -> dict[str, object]:
    row: dict[str, object] = {}
    for feature in numeric_features:
        row[feature] = float(pd.to_numeric(clean_df[feature], errors="coerce").median())
    for feature in categorical_features:
        row[feature] = clean_df[feature].mode(dropna=True).iloc[0]
    return row


def quantile_value(series: pd.Series, quantile: float) -> float:
    return float(pd.to_numeric(series, errors="coerce").quantile(quantile))


def generate_recommendation_maps(
    clean_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pipeline = fit_full_recommendation_model(
        clean_df, numeric_features, categorical_features
    )
    typical_row = build_typical_feature_row(
        clean_df, numeric_features, categorical_features
    )
    model_values = (
        sorted(clean_df["model"].dropna().unique().tolist())
        if "model" in clean_df.columns
        else []
    )

    summary_rows: list[dict[str, object]] = []
    for metadata_level, metadata_value in [
        ("low", quantile_value(clean_df["overall_metadata_shift_score"], 0.25)),
        ("high", quantile_value(clean_df["overall_metadata_shift_score"], 0.75)),
    ]:
        for sample_level, sample_value in [
            ("low", quantile_value(clean_df["sample_size"], 0.25)),
            ("high", quantile_value(clean_df["sample_size"], 0.75)),
        ]:
            for entropy_level, entropy_value in [
                ("low", quantile_value(clean_df["class_entropy"], 0.25)),
                ("high", quantile_value(clean_df["class_entropy"], 0.75)),
            ]:
                for distance_level, distance_value in [
                    ("low", quantile_value(clean_df["site_distance_composite"], 0.25)),
                    ("high", quantile_value(clean_df["site_distance_composite"], 0.75)),
                ]:
                    method_predictions = {}
                    for method in METHOD_ORDER:
                        pred_rows = []
                        for model_name in model_values:
                            row = typical_row.copy()
                            row["method"] = method
                            row["model"] = model_name
                            row["sample_size"] = sample_value
                            if "n_total_target" in row:
                                row["n_total_target"] = sample_value
                            row["class_entropy"] = entropy_value
                            if "normalized_entropy" in row:
                                row["normalized_entropy"] = entropy_value
                            row["overall_metadata_shift_score"] = metadata_value
                            row["site_distance_composite"] = distance_value
                            pred_rows.append(row)
                        pred_df = pd.DataFrame(pred_rows)
                        mean_pred = float(
                            np.mean(
                                pipeline.predict(
                                    pred_df[numeric_features + categorical_features]
                                )
                            )
                        )
                        method_predictions[method] = mean_pred
                    best_method = max(
                        method_predictions.items(), key=lambda item: item[1]
                    )[0]
                    summary_rows.append(
                        {
                            "metadata_shift_level": metadata_level,
                            "sample_size_level": sample_level,
                            "entropy_level": entropy_level,
                            "domain_distance_level": distance_level,
                            "best_method": best_method,
                            "predicted_best_ood_roc_auc": method_predictions[
                                best_method
                            ],
                            **{
                                f"pred_{method}": value
                                for method, value in method_predictions.items()
                            },
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)

    heatmap_rows: list[dict[str, object]] = []
    metadata_grid = np.linspace(
        clean_df["overall_metadata_shift_score"].min(),
        clean_df["overall_metadata_shift_score"].max(),
        25,
    )
    sample_grid = np.linspace(
        clean_df["sample_size"].min(), clean_df["sample_size"].max(), 25
    )
    entropy_median = float(clean_df["class_entropy"].median())
    distance_median = float(clean_df["site_distance_composite"].median())

    for metadata_value in metadata_grid:
        for sample_value in sample_grid:
            method_predictions = {}
            for method in METHOD_ORDER:
                pred_rows = []
                for model_name in model_values:
                    row = typical_row.copy()
                    row["method"] = method
                    row["model"] = model_name
                    row["sample_size"] = sample_value
                    if "n_total_target" in row:
                        row["n_total_target"] = sample_value
                    row["class_entropy"] = entropy_median
                    if "normalized_entropy" in row:
                        row["normalized_entropy"] = entropy_median
                    row["overall_metadata_shift_score"] = metadata_value
                    row["site_distance_composite"] = distance_median
                    pred_rows.append(row)
                pred_df = pd.DataFrame(pred_rows)
                method_predictions[method] = float(
                    np.mean(
                        pipeline.predict(
                            pred_df[numeric_features + categorical_features]
                        )
                    )
                )
            best_method = max(method_predictions.items(), key=lambda item: item[1])[0]
            heatmap_rows.append(
                {
                    "overall_metadata_shift_score": metadata_value,
                    "sample_size": sample_value,
                    "best_method": best_method,
                }
            )

    heatmap_df = pd.DataFrame(heatmap_rows)

    method_codes = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    pivot = heatmap_df.assign(
        best_code=lambda frame: frame["best_method"].map(method_codes)
    ).pivot(
        index="sample_size",
        columns="overall_metadata_shift_score",
        values="best_code",
    )
    fig, ax = plt.subplots(figsize=(9.0, 6.8))
    cmap = ListedColormap([METHOD_COLORS[method] for method in METHOD_ORDER])
    im = ax.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(METHOD_ORDER) - 0.5,
    )
    ax.set_title("Best method map | metadata shift vs sample size")
    ax.set_xlabel("Overall metadata shift score")
    ax.set_ylabel("Sample size")
    ax.set_xticks(np.linspace(0, pivot.shape[1] - 1, 5))
    ax.set_xticklabels(
        [
            f"{value:.2f}"
            for value in np.linspace(pivot.columns.min(), pivot.columns.max(), 5)
        ]
    )
    ax.set_yticks(np.linspace(0, pivot.shape[0] - 1, 5))
    ax.set_yticklabels(
        [
            f"{value:.0f}"
            for value in np.linspace(pivot.index.min(), pivot.index.max(), 5)
        ]
    )
    cbar = fig.colorbar(im, ax=ax, ticks=list(method_codes.values()))
    cbar.ax.set_yticklabels(METHOD_ORDER)
    fig.tight_layout()
    fig.savefig(MAPS_DIR / "best_method_map_metadata_shift_vs_sample_size.png", dpi=220)
    plt.close(fig)

    return summary_df, heatmap_df


def build_report(
    clean_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    shap_summary_df: pd.DataFrame,
    regression_summary_df: pd.DataFrame,
    recommendation_summary_df: pd.DataFrame,
    interaction_features: list[str],
) -> str:
    lines = [
        "Extended interaction analysis for cross-site transportability",
        "",
        f"Target metric: {DISPLAY_METRIC}",
        f"Rows analyzed: {len(clean_df)}",
        f"Target sites: {clean_df['target_site'].nunique()}",
        "",
        "Clean predictors retained:",
        ", ".join(interaction_features),
        "",
    ]

    if not shap_summary_df.empty:
        lines.append("Top SHAP interactions by method:")
        for method in METHOD_ORDER:
            method_df = shap_summary_df.loc[shap_summary_df["method"] == method].head(3)
            if method_df.empty:
                continue
            formatted = "; ".join(
                f"{row['metafeature']} (mean|interaction|={row['mean_abs_shap_interaction']:.4f})"
                for _, row in method_df.iterrows()
            )
            lines.append(f"- {method}: {formatted}")
        lines.append("")

    if not regression_summary_df.empty:
        lines.append("Strongest regression-based interactions:")
        for _, row in regression_summary_df.head(6).iterrows():
            p_value = row["p_value"]
            p_text = "NA" if pd.isna(p_value) else f"{p_value:.3g}"
            lines.append(
                f"- {row['meta_feature']}: {row['strongest_interaction_term']} | coef={row['coefficient']:.4f} | p={p_text} | {row['direction']}"
            )
        lines.append("")

    if not recommendation_summary_df.empty:
        high_shift = recommendation_summary_df.loc[
            recommendation_summary_df["metadata_shift_level"] == "high", "best_method"
        ].value_counts()
        low_shift = recommendation_summary_df.loc[
            recommendation_summary_df["metadata_shift_level"] == "low", "best_method"
        ].value_counts()
        lines.append("Recommendation-pattern summary:")
        lines.append(
            f"- Under high metadata shift, most frequent recommended methods: {high_shift.to_dict()}"
        )
        lines.append(
            f"- Under low metadata shift, most frequent recommended methods: {low_shift.to_dict()}"
        )
        lines.append("")

    lines.extend(
        [
            "Interpretation notes:",
            "- The analysis is conditional and exploratory, not causal.",
            "- The SHAP interaction analysis suggests which method effects vary with site/domain characteristics.",
            "- The regression interaction analysis provides a simpler significance-oriented cross-check for one metafeature at a time.",
            "- Because the UCI meta-learning table contains only 60 OOD ROC-AUC rows, all interaction patterns should be interpreted cautiously.",
        ]
    )

    if any(
        feature in interaction_features
        for feature in [
            "overall_metadata_shift_score",
            "mean_continuous_metadata_shift",
        ]
    ):
        lines.append(
            "- Metadata shift was explicitly included in both SHAP and regression interaction analyses."
        )
    if "site_distance_composite" in interaction_features:
        lines.append(
            "- Domain distance was summarized primarily through the site_distance_composite feature in the explicit interaction models."
        )
    if "class_entropy" in interaction_features:
        lines.append(
            "- Entropy-related effects were tested through class_entropy; normalized_entropy was removed when it duplicated class_entropy exactly."
        )
    if "sample_size" in interaction_features:
        lines.append(
            "- Sample-size effects were tested through sample_size; n_total_target was removed when it duplicated sample_size exactly."
        )

    return "\n".join(lines) + "\n"


def write_readme(
    clean_df: pd.DataFrame, inventory_df: pd.DataFrame, interaction_features: list[str]
) -> None:
    kept_predictors = inventory_df.loc[
        inventory_df["status"] == "kept", "column_name"
    ].tolist()
    excluded_predictors = inventory_df.loc[
        inventory_df["status"] == "excluded", ["column_name", "reason"]
    ]
    text = f"""
    Interaction_analysis_extended

    This folder contains a fresh interaction analysis built from the existing UCI meta-learning table.

    Main target:
    - OOD ROC-AUC only

    Main source table:
    - {SOURCE_META_TABLE}

    Clean analysis rows:
    - {len(clean_df)} rows
    - {clean_df["target_site"].nunique()} target sites

    Predictors retained:
    - {", ".join(kept_predictors)}

    Key meta-features used in explicit interaction models:
    - {", ".join(interaction_features)}

    Predictors excluded:
    {excluded_predictors.to_string(index=False) if not excluded_predictors.empty else "None"}

    Outputs:
    - tables/: cleaned data, SHAP interaction rankings, regression interaction results
    - shap_dependence_plots/: method-specific SHAP dependence plots
    - shap_interaction_barplots/: strongest SHAP interactions per method
    - interaction_plots/: regression-based visualization with one line per method
    - partial_dependence_plots/: method-specific partial dependence curves
    - method_recommendation_maps/: recommendation tables and maps
    - interaction_analysis_report.txt: automatic narrative summary
    """
    save_text(README_PATH, textwrap.dedent(text).strip() + "\n")


def main() -> None:
    ensure_output_dirs()
    source_df = load_source_table()
    clean_df, inventory_df, numeric_features, categorical_features = (
        build_clean_analysis_dataframe(source_df)
    )

    save_dataframe(
        clean_df,
        TABLES_DIR / "clean_analysis_dataframe.csv",
        TABLES_DIR / "clean_analysis_dataframe.xlsx",
    )
    save_dataframe(
        inventory_df,
        TABLES_DIR / "predictor_inventory.csv",
        TABLES_DIR / "predictor_inventory.xlsx",
    )

    model_pipeline, transformed_df, shap_values, interaction_values = fit_meta_model(
        clean_df, numeric_features, categorical_features
    )
    rankings_df, shap_summary_df = build_shap_interaction_tables(
        transformed_df, interaction_values
    )
    save_dataframe(
        rankings_df,
        TABLES_DIR / "shap_interaction_rankings.csv",
        TABLES_DIR / "shap_interaction_rankings.xlsx",
    )
    save_dataframe(
        shap_summary_df,
        TABLES_DIR / "shap_interaction_summary.csv",
        TABLES_DIR / "shap_interaction_summary.xlsx",
    )

    plot_shap_interaction_bars(shap_summary_df)
    dependence_index_df = plot_shap_dependence(transformed_df, shap_values, rankings_df)
    save_dataframe(
        dependence_index_df,
        TABLES_DIR / "shap_dependence_plot_index.csv",
        TABLES_DIR / "shap_dependence_plot_index.xlsx",
    )

    interaction_features = selected_regression_features(rankings_df, numeric_features)
    regression_results_df, regression_summary_df = run_regression_interactions(
        clean_df, interaction_features
    )
    save_dataframe(
        regression_results_df,
        TABLES_DIR / "regression_interaction_results.csv",
        TABLES_DIR / "regression_interaction_results.xlsx",
    )
    save_dataframe(
        regression_summary_df,
        TABLES_DIR / "interaction_summary_ranked.csv",
        TABLES_DIR / "interaction_summary_ranked.xlsx",
    )

    interaction_plot_index_df = plot_interaction_regressions(
        clean_df, interaction_features
    )
    save_dataframe(
        interaction_plot_index_df,
        TABLES_DIR / "interaction_plot_index.csv",
        TABLES_DIR / "interaction_plot_index.xlsx",
    )

    pdp_index_df = generate_partial_dependence(
        clean_df, numeric_features, interaction_features
    )
    save_dataframe(
        pdp_index_df,
        TABLES_DIR / "partial_dependence_plot_index.csv",
        TABLES_DIR / "partial_dependence_plot_index.xlsx",
    )

    recommendation_summary_df, recommendation_heatmap_df = generate_recommendation_maps(
        clean_df, numeric_features, categorical_features
    )
    save_dataframe(
        recommendation_summary_df,
        MAPS_DIR / "method_recommendation_summary.csv",
        MAPS_DIR / "method_recommendation_summary.xlsx",
    )
    save_dataframe(
        recommendation_heatmap_df,
        MAPS_DIR / "method_recommendation_grid_predictions.csv",
        MAPS_DIR / "method_recommendation_grid_predictions.xlsx",
    )

    report_text = build_report(
        clean_df,
        rankings_df,
        shap_summary_df,
        regression_summary_df,
        recommendation_summary_df,
        interaction_features,
    )
    save_text(REPORT_PATH, report_text)
    write_readme(clean_df, inventory_df, interaction_features)

    summary_payload = {
        "input_root": str(INPUT_ROOT),
        "source_table": str(SOURCE_META_TABLE),
        "output_root": str(OUTPUT_ROOT),
        "target_metric": TARGET_METRIC,
        "n_rows": int(len(clean_df)),
        "n_sites": int(clean_df["target_site"].nunique()),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "interaction_features": interaction_features,
    }
    save_text(SUMMARY_JSON_PATH, json.dumps(summary_payload, indent=2))

    shutil.copy2(Path(__file__).resolve(), SCRIPT_OUTPUT_PATH)
    save_text(RUN_LOG_PATH, "\n".join(LOG_MESSAGES) + "\n")
    log(f"Extended interaction analysis completed. Results saved to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
