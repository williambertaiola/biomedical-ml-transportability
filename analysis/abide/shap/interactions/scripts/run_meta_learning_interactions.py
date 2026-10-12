"""Post-experiment analysis: run meta learning interactions."""

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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
INPUT_ROOT = Path(_configured_path("results_abide"))
SHAP_ROOT = INPUT_ROOT / "SHAP"
META_TABLE_PATH = SHAP_ROOT / "tables" / "meta_learning_table_long.csv"
ORIGINAL_SCRIPT = SHAP_ROOT / "scripts" / "run_meta_learning_analysis.py"

OUTPUT_ROOT = SHAP_ROOT / "interactions"
TABLES_DIR = OUTPUT_ROOT / "tables"
DEPENDENCE_DIR = OUTPUT_ROOT / "dependence_plots"
HEATMAP_DIR = OUTPUT_ROOT / "interaction_heatmaps"
SUMMARY_DIR = OUTPUT_ROOT / "summary"
SCRIPTS_DIR = OUTPUT_ROOT / "scripts"
README_PATH = OUTPUT_ROOT / "README.txt"
RUN_LOG_PATH = OUTPUT_ROOT / "run_log.txt"
SCRIPT_OUTPUT_PATH = SCRIPTS_DIR / "run_meta_learning_interactions.py"

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
    "fiq_missing_pct",
    "handedness_missing_pct",
    "eye_status_missing_pct",
]
CATEGORICAL_CANDIDATES = ["method", "model", "dataset"]
HEATMAP_METAFEATURES = [
    "sample_size",
    "n_total_target",
    "class_entropy",
    "normalized_entropy",
    "class_balance",
    "minority_class_ratio",
    "overall_metadata_shift_score",
    "mean_continuous_metadata_shift",
    "wasserstein_mean_distance",
    "missing_percentage",
    "fiq_missing_pct",
]
REGRESSION_METAFEATURES = [
    "sample_size",
    "class_entropy",
    "class_balance",
    "overall_metadata_shift_score",
    "mean_continuous_metadata_shift",
    "missing_percentage",
]
REQUESTED_INTERACTIONS = {
    "M0": ["sample_size", "overall_metadata_shift_score"],
    "M1": ["class_entropy", "normalized_entropy", "overall_metadata_shift_score"],
    "M3": [
        "overall_metadata_shift_score",
        "mean_continuous_metadata_shift",
        "class_balance",
    ],
    "M4": ["overall_metadata_shift_score", "sample_size", "class_entropy"],
}
METRICS_TO_MODEL = [
    "OOD_ROC_AUC",
    "OOD_F1",
    "OOD_BALANCED_ACCURACY",
    "ECE",
    "ID_OOD_GAP",
]

LOG_MESSAGES: list[str] = []


def log(message: str) -> None:
    print(message)
    LOG_MESSAGES.append(message)


def ensure_output_dirs() -> None:
    for directory in [
        OUTPUT_ROOT,
        TABLES_DIR,
        DEPENDENCE_DIR,
        HEATMAP_DIR,
        SUMMARY_DIR,
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
    clean = name.replace("num__", "").replace("cat__method_", "").replace("_", " ")
    return clean.title()


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


def load_meta_table() -> pd.DataFrame:
    df = pd.read_csv(META_TABLE_PATH)
    log(f"Loaded meta-learning table: {META_TABLE_PATH}")
    log(f"Shape: {df.shape}")
    log("Available columns:")
    for column in df.columns:
        log(f" - {column}")
    return df


def select_available_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = []
    for feature in NUMERIC_META_CANDIDATES:
        if feature not in df.columns:
            continue
        values = pd.to_numeric(df[feature], errors="coerce")
        if values.notna().sum() == 0:
            continue
        if values.nunique(dropna=True) <= 1:
            continue
        numeric_features.append(feature)

    categorical_features = []
    for feature in CATEGORICAL_CANDIDATES:
        if feature not in df.columns:
            continue
        if df[feature].nunique(dropna=True) <= 1:
            continue
        categorical_features.append(feature)
    return numeric_features, categorical_features


def fit_rf_meta_model(metric_df: pd.DataFrame) -> dict[str, object]:
    numeric_features, categorical_features = select_available_features(metric_df)
    feature_frame = metric_df[numeric_features + categorical_features].copy()
    y = metric_df["performance"].to_numpy(dtype=float)

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
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
    pipeline.fit(feature_frame, y)

    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]
    transformed = fitted_preprocessor.transform(feature_frame)
    transformed_df = pd.DataFrame(
        transformed,
        columns=fitted_preprocessor.get_feature_names_out(),
        index=metric_df.index,
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
    except Exception as exc:  # pragma: no cover - exploratory fallback
        log(
            f"SHAP interaction values not available for {metric_df['metric'].iloc[0]}: {exc}"
        )

    return {
        "metric_df": metric_df,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "transformed_df": transformed_df,
        "shap_values": shap_values,
        "interaction_values": interaction_values,
    }


def save_dependence_plots(
    metric_name: str, transformed_df: pd.DataFrame, shap_values: np.ndarray
) -> pd.DataFrame:
    plot_rows: list[dict[str, object]] = []
    metric_stem = FILE_STEM_MAP[metric_name]
    display_name = DISPLAY_NAME_MAP[metric_name]

    for method, metafeatures in REQUESTED_INTERACTIONS.items():
        method_feature = f"cat__method_{method}"
        if method_feature not in transformed_df.columns:
            log(
                f"Skipping dependence plots for {metric_name} | {method}: feature {method_feature} not found."
            )
            continue

        for metafeature in metafeatures:
            numeric_feature = f"num__{metafeature}"
            if numeric_feature not in transformed_df.columns:
                log(
                    f"Skipping dependence plots for {metric_name} | {method} x {metafeature}: feature missing."
                )
                continue

            forward_path = (
                DEPENDENCE_DIR / f"dependence_{metric_stem}_{method}_{metafeature}.png"
            )
            reverse_path = (
                DEPENDENCE_DIR
                / f"dependence_{metric_stem}_{method}_{metafeature}_reverse.png"
            )

            plt.figure(figsize=(7.8, 5.8))
            shap.dependence_plot(
                numeric_feature,
                shap_values,
                transformed_df,
                interaction_index=method_feature,
                show=False,
                alpha=0.8,
            )
            plt.title(
                f"SHAP dependence | {display_name} | {method} x {humanize_feature_name(numeric_feature)}"
            )
            plt.xlabel(humanize_feature_name(numeric_feature))
            plt.tight_layout()
            plt.savefig(forward_path, dpi=220)
            plt.close()
            plt.close("all")

            plt.figure(figsize=(7.2, 5.4))
            shap.dependence_plot(
                method_feature,
                shap_values,
                transformed_df,
                interaction_index=numeric_feature,
                show=False,
                alpha=0.8,
            )
            plt.title(
                f"SHAP dependence | {display_name} | {method} x {humanize_feature_name(numeric_feature)}"
            )
            plt.xlabel(method)
            plt.tight_layout()
            plt.savefig(reverse_path, dpi=220)
            plt.close()
            plt.close("all")

            plot_rows.append(
                {
                    "metric": metric_name,
                    "method_feature": method_feature,
                    "metafeature": numeric_feature,
                    "forward_plot": str(forward_path),
                    "reverse_plot": str(reverse_path),
                }
            )
    return pd.DataFrame(plot_rows)


def compute_direction_hint(
    transformed_df: pd.DataFrame,
    shap_values: np.ndarray,
    method_feature: str,
    numeric_feature: str,
) -> str:
    if (
        method_feature not in transformed_df.columns
        or numeric_feature not in transformed_df.columns
    ):
        return "unclear"

    method_mask = transformed_df[method_feature].to_numpy(dtype=float) > 0.5
    numeric_values = transformed_df[numeric_feature].to_numpy(dtype=float)
    feature_index = transformed_df.columns.get_loc(numeric_feature)
    shap_component = np.asarray(shap_values[:, feature_index], dtype=float)

    if method_mask.sum() >= 3:
        x = numeric_values[method_mask]
        y = shap_component[method_mask]
    else:
        x = numeric_values
        y = shap_component

    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return "unclear"

    correlation = float(np.corrcoef(x, y)[0, 1])
    if math.isnan(correlation):
        return "unclear"
    if abs(correlation) < 0.10:
        return "weak / mixed"
    if correlation > 0:
        return "more positive at higher values"
    return "more negative at higher values"


def build_interaction_table(
    metric_name: str,
    transformed_df: pd.DataFrame,
    interaction_values: np.ndarray | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = [f"cat__method_M{i}" for i in range(5)]
    metafeatures = [f"num__{feature}" for feature in HEATMAP_METAFEATURES]

    available_methods = [
        feature for feature in methods if feature in transformed_df.columns
    ]
    available_metafeatures = [
        feature for feature in metafeatures if feature in transformed_df.columns
    ]

    if (
        interaction_values is None
        or not available_methods
        or not available_metafeatures
    ):
        return pd.DataFrame(
            columns=[
                "metric",
                "method_feature",
                "metafeature",
                "mean_abs_shap_interaction",
                "rank",
            ]
        ), pd.DataFrame()

    rows: list[dict[str, object]] = []
    for method_feature in available_methods:
        method_index = transformed_df.columns.get_loc(method_feature)
        for metafeature in available_metafeatures:
            feature_index = transformed_df.columns.get_loc(metafeature)
            mean_abs = float(
                np.mean(np.abs(interaction_values[:, method_index, feature_index]))
            )
            rows.append(
                {
                    "metric": metric_name,
                    "method_feature": method_feature,
                    "metafeature": metafeature,
                    "mean_abs_shap_interaction": mean_abs,
                }
            )

    interaction_df = (
        pd.DataFrame(rows)
        .sort_values("mean_abs_shap_interaction", ascending=False)
        .reset_index(drop=True)
    )
    interaction_df["rank"] = np.arange(1, len(interaction_df) + 1)

    heatmap_df = (
        interaction_df.assign(
            method_label=lambda frame: frame["method_feature"].str.replace(
                "cat__method_", "", regex=False
            ),
            metafeature_label=lambda frame: frame["metafeature"].str.replace(
                "num__", "", regex=False
            ),
        )
        .pivot(
            index="method_label",
            columns="metafeature_label",
            values="mean_abs_shap_interaction",
        )
        .reindex(index=[f"M{i}" for i in range(5)])
    )
    return interaction_df, heatmap_df


def save_interaction_heatmap(metric_name: str, heatmap_df: pd.DataFrame) -> Path | None:
    if heatmap_df.empty:
        return None

    metric_stem = FILE_STEM_MAP[metric_name]
    output_path = HEATMAP_DIR / f"shap_interaction_heatmap_{metric_stem}.png"

    fig_width = max(9.0, 1.2 * max(1, heatmap_df.shape[1]))
    fig_height = max(4.2, 0.9 * max(1, heatmap_df.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    values = heatmap_df.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(np.arange(heatmap_df.shape[1]))
    ax.set_xticklabels(
        [label.replace("_", " ") for label in heatmap_df.columns],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(np.arange(heatmap_df.shape[0]))
    ax.set_yticklabels(heatmap_df.index.tolist())
    ax.set_title(f"SHAP interaction heatmap | {DISPLAY_NAME_MAP[metric_name]}")

    for row_idx in range(heatmap_df.shape[0]):
        for col_idx in range(heatmap_df.shape[1]):
            value = values[row_idx, col_idx]
            if np.isnan(value):
                continue
            text_color = "black" if value < np.nanmax(values) * 0.7 else "white"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("mean |SHAP interaction|")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    plt.close("all")
    return output_path


def fit_regression_interactions(
    metric_name: str, metric_df: pd.DataFrame
) -> pd.DataFrame:
    available_features = []
    for feature in REGRESSION_METAFEATURES:
        if feature not in metric_df.columns:
            continue
        values = pd.to_numeric(metric_df[feature], errors="coerce")
        if values.notna().sum() == 0 or values.nunique(dropna=True) <= 1:
            continue
        available_features.append(feature)

    base_columns = ["performance", "method", "model"] + available_features
    if "dataset" in metric_df.columns and metric_df["dataset"].nunique(dropna=True) > 1:
        base_columns.append("dataset")

    reg_df = metric_df[base_columns].copy().dropna().reset_index(drop=True)
    if reg_df.empty or reg_df["method"].nunique(dropna=True) <= 1:
        return pd.DataFrame(
            columns=[
                "metric",
                "interaction_term",
                "coefficient",
                "p_value",
                "direction",
                "interpretation_hint",
            ]
        )

    main_terms = ["C(method)"]
    if reg_df["model"].nunique(dropna=True) > 1:
        main_terms.append("C(model)")
    if "dataset" in reg_df.columns and reg_df["dataset"].nunique(dropna=True) > 1:
        main_terms.append("C(dataset)")
    main_terms.extend(available_features)
    interaction_terms = [f"C(method):{feature}" for feature in available_features]
    formula = "performance ~ " + " + ".join(main_terms + interaction_terms)

    try:
        fitted = smf.ols(formula=formula, data=reg_df).fit(cov_type="HC3")
    except Exception as exc:  # pragma: no cover - exploratory fallback
        log(f"Regression interaction model failed for {metric_name}: {exc}")
        return pd.DataFrame(
            columns=[
                "metric",
                "interaction_term",
                "coefficient",
                "p_value",
                "direction",
                "interpretation_hint",
            ]
        )

    rows: list[dict[str, object]] = []
    for term, coefficient in fitted.params.items():
        if "C(method)" not in term or ":" not in term:
            continue
        p_value = float(fitted.pvalues.get(term, np.nan))
        direction = "positive" if coefficient > 0 else "negative"

        method_name = "method"
        feature_name = "feature"
        if "[T." in term:
            try:
                method_name = term.split("[T.", 1)[1].split("]", 1)[0]
                feature_name = term.rsplit(":", 1)[1]
            except Exception:  # pragma: no cover - defensive parsing
                pass

        interpretation = (
            f"As {feature_name} increases, the meta-model suggests a more {direction} deviation "
            f"for {method_name} relative to the reference method."
        )
        rows.append(
            {
                "metric": metric_name,
                "interaction_term": term,
                "coefficient": float(coefficient),
                "p_value": p_value,
                "direction": direction,
                "interpretation_hint": interpretation,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("p_value", ascending=True, na_position="last")
        .reset_index(drop=True)
    )


def build_metric_summary(
    metric_name: str,
    interaction_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    if interaction_df.empty:
        return pd.DataFrame(
            [
                {
                    "metric": metric_name,
                    "strongest_method_metafeature_interaction": "not available",
                    "direction_hint": "not available",
                    "regression_interaction_term": "not available",
                    "regression_coefficient": np.nan,
                    "cautious_interpretation": "SHAP interaction values were not available for this metric.",
                }
            ]
        )

    top_row = interaction_df.iloc[0]
    method_feature = str(top_row["method_feature"])
    metafeature = str(top_row["metafeature"])
    method_label = method_feature.replace("cat__method_", "")
    metafeature_label = metafeature.replace("num__", "")
    direction_hint = compute_direction_hint(
        transformed_df, shap_values, method_feature, metafeature
    )

    regression_match = pd.DataFrame()
    if not regression_df.empty:
        mask = regression_df["interaction_term"].str.contains(
            method_label, regex=False, na=False
        ) & regression_df["interaction_term"].str.contains(
            metafeature_label, regex=False, na=False
        )
        regression_match = regression_df.loc[mask].copy()

    if regression_match.empty:
        reg_term = "not available"
        reg_coef = np.nan
    else:
        best_reg = regression_match.iloc[0]
        reg_term = str(best_reg["interaction_term"])
        reg_coef = float(best_reg["coefficient"])

    cautious_text = (
        f"The meta-model identified {method_label} × {metafeature_label} as the strongest interaction for "
        f"{DISPLAY_NAME_MAP[metric_name]}. The dependence pattern appears {direction_hint}, which suggests that "
        f"{method_label} may behave differently as {metafeature_label} changes. This should be interpreted as "
        f"an exploratory association rather than a causal effect."
    )
    return pd.DataFrame(
        [
            {
                "metric": metric_name,
                "strongest_method_metafeature_interaction": f"{method_label} × {metafeature_label}",
                "direction_hint": direction_hint,
                "regression_interaction_term": reg_term,
                "regression_coefficient": reg_coef,
                "cautious_interpretation": cautious_text,
            }
        ]
    )


def write_readme() -> None:
    content = """
    Meta-learning interaction analysis built on top of the existing SHAP meta-model outputs.

    What this block adds:
    - SHAP dependence plots focused on method × metafeature interactions.
    - SHAP interaction-value tables and heatmaps.
    - A complementary regression-based interaction check using a limited set of site-level metafeatures.

    Main idea:
    The goal is not only to identify which metafeatures globally predict performance, but to inspect
    whether the apparent usefulness of M0-M4 depends on target-domain properties such as sample size,
    class entropy, class balance, and metadata shift.

    Important caution:
    - This analysis is descriptive and exploratory.
    - Interaction estimates should be interpreted cautiously, especially when the number of target sites is modest relative to the number of candidate metafeatures.
    - SHAP interaction values are model-dependent and summarize the fitted random-forest meta-model.

    Inputs:
    - Existing meta-learning table: {meta_table}
    - Existing global SHAP pipeline script: {original_script}

    Output folders:
    - dependence_plots/: method × metafeature SHAP dependence plots
    - interaction_heatmaps/: one heatmap per metric
    - tables/: SHAP interaction tables and regression interaction tables
    - summary/: cross-metric summary tables
    """.format(meta_table=META_TABLE_PATH, original_script=ORIGINAL_SCRIPT)
    save_text(README_PATH, textwrap.dedent(content).strip() + "\n")


def main() -> None:
    ensure_output_dirs()
    master_df = load_meta_table()

    interaction_tables: list[pd.DataFrame] = []
    regression_tables: list[pd.DataFrame] = []
    summary_tables: list[pd.DataFrame] = []
    dependence_index_frames: list[pd.DataFrame] = []
    model_inventory_rows: list[dict[str, object]] = []

    for metric_name in METRICS_TO_MODEL:
        metric_df = master_df.loc[
            (master_df["metric"] == metric_name) & (master_df["performance"].notna())
        ].copy()
        metric_df = metric_df.reset_index(drop=True)
        if metric_df.empty:
            log(f"Skipping {metric_name}: no rows available.")
            continue

        log(f"Running interaction analysis for {metric_name} on {len(metric_df)} rows.")
        metric_stem = FILE_STEM_MAP[metric_name]
        model_state = fit_rf_meta_model(metric_df)
        transformed_df = model_state["transformed_df"]
        shap_values = model_state["shap_values"]
        interaction_values = model_state["interaction_values"]

        model_inventory_rows.append(
            {
                "metric": metric_name,
                "n_rows": len(metric_df),
                "n_sites": metric_df["target_site"].nunique(),
                "numeric_features_used": ", ".join(model_state["numeric_features"]),
                "categorical_features_used": ", ".join(
                    model_state["categorical_features"]
                ),
                "transformed_feature_count": transformed_df.shape[1],
                "shap_interactions_available": interaction_values is not None,
            }
        )

        dependence_df = save_dependence_plots(metric_name, transformed_df, shap_values)
        if not dependence_df.empty:
            dependence_index_frames.append(dependence_df)

        interaction_df, heatmap_df = build_interaction_table(
            metric_name, transformed_df, interaction_values
        )
        regression_df = fit_regression_interactions(metric_name, metric_df)
        summary_df = build_metric_summary(
            metric_name, interaction_df, regression_df, transformed_df, shap_values
        )

        if not interaction_df.empty:
            interaction_tables.append(interaction_df)
            save_dataframe(
                interaction_df,
                TABLES_DIR / f"method_metafeature_interactions_{metric_stem}.csv",
                TABLES_DIR / f"method_metafeature_interactions_{metric_stem}.xlsx",
            )
        if heatmap_df is not None and not heatmap_df.empty:
            save_interaction_heatmap(metric_name, heatmap_df)
        if not regression_df.empty:
            regression_tables.append(regression_df)
        save_dataframe(
            regression_df,
            TABLES_DIR / f"regression_interaction_terms_{metric_stem}.csv",
            TABLES_DIR / f"regression_interaction_terms_{metric_stem}.xlsx",
        )
        summary_tables.append(summary_df)

    interaction_all = (
        pd.concat(interaction_tables, ignore_index=True)
        if interaction_tables
        else pd.DataFrame()
    )
    regression_all = (
        pd.concat(regression_tables, ignore_index=True)
        if regression_tables
        else pd.DataFrame()
    )
    summary_all = (
        pd.concat(summary_tables, ignore_index=True)
        if summary_tables
        else pd.DataFrame()
    )
    dependence_index = (
        pd.concat(dependence_index_frames, ignore_index=True)
        if dependence_index_frames
        else pd.DataFrame()
    )
    model_inventory = pd.DataFrame(model_inventory_rows)

    save_dataframe(
        interaction_all,
        TABLES_DIR / "method_metafeature_interactions_all_metrics.csv",
        TABLES_DIR / "method_metafeature_interactions_all_metrics.xlsx",
    )
    save_dataframe(
        regression_all,
        TABLES_DIR / "regression_interaction_terms_all_metrics.csv",
        TABLES_DIR / "regression_interaction_terms_all_metrics.xlsx",
    )
    save_dataframe(
        dependence_index,
        TABLES_DIR / "dependence_plot_index.csv",
        TABLES_DIR / "dependence_plot_index.xlsx",
    )
    save_dataframe(
        model_inventory,
        TABLES_DIR / "interaction_model_inventory.csv",
        TABLES_DIR / "interaction_model_inventory.xlsx",
    )
    save_dataframe(
        summary_all,
        SUMMARY_DIR / "interaction_summary_by_metric.csv",
        SUMMARY_DIR / "interaction_summary_by_metric.xlsx",
    )

    summary_payload = {
        "input_root": str(INPUT_ROOT),
        "meta_learning_table": str(META_TABLE_PATH),
        "n_master_rows": int(len(master_df)),
        "metrics_modeled": sorted(summary_all["metric"].dropna().unique().tolist())
        if not summary_all.empty
        else [],
    }
    save_text(OUTPUT_ROOT / "summary.json", json.dumps(summary_payload, indent=2))
    write_readme()

    shutil.copy2(Path(__file__).resolve(), SCRIPT_OUTPUT_PATH)
    save_text(RUN_LOG_PATH, "\n".join(LOG_MESSAGES) + "\n")
    log(f"Meta-learning interaction analysis completed. Results saved to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
