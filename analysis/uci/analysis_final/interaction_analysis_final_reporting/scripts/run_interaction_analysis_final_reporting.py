"""Post-experiment analysis: run interaction analysis final reporting."""

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
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.formula.api as smf
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_ROOT = Path(_configured_path("results_uci"))
SOURCE_META_TABLE = INPUT_ROOT / "SHAP" / "tables" / "meta_learning_table_long.csv"
PREVIOUS_ROOT = INPUT_ROOT / "analysis_final" / "Interaction_analysis_extended"
OUTPUT_ROOT = INPUT_ROOT / "analysis_final" / "Interaction_analysis_final_reporting"
TARGET_METRIC = "OOD_ROC_AUC"
RANDOM_STATE = 42
MAX_MISSING_RATE = 0.40
METHOD_ORDER = ["M0", "M1", "M2", "M3", "M4"]
MODEL_ORDER = ["LogisticRegression", "RandomForest", "HistGradientBoosting"]
METHOD_LABELS = {
    "M0": "M0: Baseline",
    "M1": "M1: ComBat",
    "M2": "M2: Weighting",
    "M3": "M3: Transductive",
    "M4": "M4: ComBat + Transductive",
}
METHOD_COLORS = {
    "M0": "#6c757d",
    "M1": "#1f77b4",
    "M2": "#2ca02c",
    "M3": "#d62728",
    "M4": "#ff7f0e",
}
FEATURE_LABELS = {
    "overall_metadata_shift_score": "Metadata shift score",
    "mean_continuous_metadata_shift": "Continuous metadata shift",
    "site_distance_composite": "Domain distance composite",
    "sample_size": "Target sample size",
    "class_entropy": "Class entropy",
    "class_balance": "Class balance",
    "missing_percentage": "Missing percentage",
    "n_source": "Source sample size",
    "n_train_1": "Target train size",
    "n_test_1": "Target test size",
    "euclidean_centroid_distance": "Euclidean centroid distance",
    "mahalanobis_distance": "Mahalanobis distance",
    "wasserstein_mean_distance": "Wasserstein mean distance",
    "jensen_shannon_mean_divergence": "Jensen-Shannon divergence",
    "mmd_rbf": "MMD-RBF",
    "minority_class_ratio": "Minority class ratio",
}
MAIN_FEATURE_PRIORITY = [
    "overall_metadata_shift_score",
    "site_distance_composite",
    "sample_size",
    "class_entropy",
    "class_balance",
    "mean_continuous_metadata_shift",
]
DECISION_FEATURE_PRIORITY = [
    "overall_metadata_shift_score",
    "mean_continuous_metadata_shift",
    "site_distance_composite",
    "sample_size",
    "class_entropy",
    "class_balance",
]
REQUIRED_DEPENDENCE_PAIRS = [
    ("M0", "class_balance"),
    ("M1", "overall_metadata_shift_score"),
    ("M1", "class_entropy"),
    ("M3", "sample_size"),
    ("M3", "overall_metadata_shift_score"),
    ("M4", "overall_metadata_shift_score"),
    ("M4", "class_entropy"),
]
RECOMMENDATION_FEATURE_PAIRS = [
    ("overall_metadata_shift_score", "sample_size"),
    ("overall_metadata_shift_score", "class_entropy"),
    ("site_distance_composite", "sample_size"),
]
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


def friendly_name(column_name: str) -> str:
    if column_name.startswith("num__"):
        column_name = column_name.replace("num__", "", 1)
    if column_name.startswith("cat__method_"):
        method = column_name.replace("cat__method_", "", 1)
        return METHOD_LABELS.get(method, method)
    if column_name.startswith("cat__model_"):
        return column_name.replace("cat__model_", "", 1)
    return FEATURE_LABELS.get(column_name, column_name.replace("_", " ").title())


def make_dirs() -> dict[str, Path]:
    subdirs = {
        "main_figures": OUTPUT_ROOT / "main_figures",
        "delta_vs_m0_figures": OUTPUT_ROOT / "main_figures" / "delta_vs_M0",
        "binned_condition_figures": OUTPUT_ROOT / "main_figures" / "binned_conditions",
        "win_rate_figures": OUTPUT_ROOT / "main_figures" / "win_rate_by_condition",
        "recommendation_heatmaps": OUTPUT_ROOT
        / "main_figures"
        / "recommendation_heatmaps",
        "shap_interactions": OUTPUT_ROOT / "shap_interactions",
        "shap_dependence_plots": OUTPUT_ROOT / "shap_dependence_plots",
        "statistical_tables": OUTPUT_ROOT / "statistical_tables",
        "appendix_tables": OUTPUT_ROOT / "appendix_tables",
        "reports": OUTPUT_ROOT / "reports",
        "scripts": OUTPUT_ROOT / "scripts",
    }
    for path in [OUTPUT_ROOT, *subdirs.values()]:
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def deduplicate_numeric_columns(
    df: pd.DataFrame, candidates: list[str]
) -> tuple[list[str], list[dict[str, str]]]:
    kept: list[str] = []
    notes: list[dict[str, str]] = []
    seen_vectors: dict[tuple[float | str, ...], str] = {}
    for column in candidates:
        series = pd.to_numeric(df[column], errors="coerce")
        vector = tuple("__nan__" if pd.isna(v) else float(v) for v in series.to_list())
        duplicate_of = seen_vectors.get(vector)
        if duplicate_of is not None:
            notes.append(
                {
                    "column_name": column,
                    "status": "excluded",
                    "reason": f"duplicate_of:{duplicate_of}",
                }
            )
            continue
        seen_vectors[vector] = column
        kept.append(column)
    return kept, notes


def build_clean_analysis_dataframe(
    source_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    df = source_df.loc[source_df["metric"] == TARGET_METRIC].copy()
    df = df.loc[df["performance"].notna()].reset_index(drop=True)

    inventory_rows: list[dict[str, str]] = []

    numeric_candidates: list[str] = []
    for column in NUMERIC_PREDICTOR_CANDIDATES:
        if column not in df.columns:
            inventory_rows.append(
                {
                    "column_name": column,
                    "status": "excluded",
                    "reason": "missing_from_table",
                }
            )
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().mean() > MAX_MISSING_RATE:
            inventory_rows.append(
                {
                    "column_name": column,
                    "status": "excluded",
                    "reason": f"missing_rate>{MAX_MISSING_RATE:.2f}",
                }
            )
            continue
        if values.nunique(dropna=True) <= 1:
            inventory_rows.append(
                {
                    "column_name": column,
                    "status": "excluded",
                    "reason": "constant_column",
                }
            )
            continue
        numeric_candidates.append(column)

    numeric_features, duplicate_notes = deduplicate_numeric_columns(
        df, numeric_candidates
    )
    inventory_rows.extend(duplicate_notes)
    for column in numeric_features:
        inventory_rows.append(
            {"column_name": column, "status": "kept", "reason": "numeric_predictor"}
        )

    categorical_features: list[str] = []
    for column in CATEGORICAL_PREDICTOR_CANDIDATES:
        if column not in df.columns:
            inventory_rows.append(
                {
                    "column_name": column,
                    "status": "excluded",
                    "reason": "missing_from_table",
                }
            )
            continue
        if df[column].nunique(dropna=True) <= 1:
            inventory_rows.append(
                {
                    "column_name": column,
                    "status": "excluded",
                    "reason": "single_unique_value",
                }
            )
            continue
        categorical_features.append(column)
        inventory_rows.append(
            {"column_name": column, "status": "kept", "reason": "categorical_predictor"}
        )

    for column in sorted(DIRECT_LEAKAGE_COLUMNS.intersection(df.columns)):
        inventory_rows.append(
            {
                "column_name": column,
                "status": "excluded",
                "reason": "direct_performance_or_leakage",
            }
        )
    for column in sorted(NON_PREDICTOR_COLUMNS.intersection(df.columns)):
        inventory_rows.append(
            {
                "column_name": column,
                "status": "excluded",
                "reason": "identifier_or_non_predictor",
            }
        )

    keep_columns = [
        "dataset",
        "target_site",
        "target_site_display",
        "method",
        "model",
        "model_display",
        "metric",
        "performance",
    ]
    keep_columns.extend([c for c in numeric_features if c not in keep_columns])
    clean_df = df[keep_columns].copy()
    inventory = (
        pd.DataFrame(inventory_rows)
        .drop_duplicates()
        .sort_values(["status", "column_name"])
        .reset_index(drop=True)
    )
    return clean_df, inventory, numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                categorical_features,
            ),
        ],
    )


def fit_meta_model(
    clean_df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]
):
    X = clean_df[numeric_features + categorical_features].copy()
    y = clean_df["performance"].astype(float).to_numpy()

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_transformed = preprocessor.fit_transform(X)
    feature_names = list(preprocessor.get_feature_names_out())

    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
    )
    model.fit(X_transformed, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    shap_interactions = explainer.shap_interaction_values(X_transformed)
    return (
        X,
        y,
        preprocessor,
        X_transformed,
        feature_names,
        model,
        shap_values,
        shap_interactions,
    )


def compute_shap_interaction_tables(
    feature_names: list[str],
    shap_interactions: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_features = [
        name for name in feature_names if name.startswith("cat__method_")
    ]
    numeric_features = [name for name in feature_names if name.startswith("num__")]
    index_map = {name: idx for idx, name in enumerate(feature_names)}

    ranking_rows: list[dict[str, object]] = []
    for method_feature in method_features:
        method = method_feature.replace("cat__method_", "", 1)
        for num_feature in numeric_features:
            idx_i = index_map[method_feature]
            idx_j = index_map[num_feature]
            interaction_strength = float(
                np.mean(np.abs(shap_interactions[:, idx_i, idx_j]))
            )
            ranking_rows.append(
                {
                    "method": method,
                    "method_feature": method_feature,
                    "metafeature": num_feature.replace("num__", "", 1),
                    "mean_abs_shap_interaction": interaction_strength,
                }
            )

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df = ranking_df.sort_values(
        ["method", "mean_abs_shap_interaction"], ascending=[True, False]
    ).reset_index(drop=True)
    ranking_df["rank_within_method"] = ranking_df.groupby("method").cumcount() + 1
    ranking_df = ranking_df.sort_values(
        "mean_abs_shap_interaction", ascending=False
    ).reset_index(drop=True)
    ranking_df["global_rank"] = np.arange(1, len(ranking_df) + 1)

    summary_df = (
        ranking_df.groupby("method", as_index=False)
        .head(5)
        .sort_values(["method", "rank_within_method"])
        .reset_index(drop=True)
    )
    return ranking_df, summary_df


def create_shap_interaction_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    methods_present = [
        method for method in METHOD_ORDER if method in summary_df["method"].unique()
    ]
    n_methods = len(methods_present)
    ncols = 2
    nrows = math.ceil(n_methods / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    max_value = (
        float(summary_df["mean_abs_shap_interaction"].max())
        if not summary_df.empty
        else 1.0
    )
    for ax in axes:
        ax.set_visible(False)

    for ax, method in zip(axes, methods_present):
        ax.set_visible(True)
        subset = summary_df.loc[summary_df["method"] == method].sort_values(
            "mean_abs_shap_interaction"
        )
        ax.barh(
            y=np.arange(len(subset)),
            width=subset["mean_abs_shap_interaction"],
            color=METHOD_COLORS.get(method, "#4c78a8"),
            alpha=0.9,
        )
        ax.set_yticks(np.arange(len(subset)))
        ax.set_yticklabels(
            [friendly_name(name) for name in subset["metafeature"]], fontsize=9
        )
        ax.set_xlim(0, max_value * 1.15)
        ax.set_xlabel("mean |SHAP interaction|")
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=11)
        ax.grid(axis="x", alpha=0.2)

    fig.suptitle("SHAP interaction ranking by method | OOD ROC-AUC", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_dependence_pairs(ranking_df: pd.DataFrame) -> list[tuple[str, str]]:
    preferred = set(MAIN_FEATURE_PRIORITY + ["n_source", "n_test_1"])
    chosen: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for pair in REQUIRED_DEPENDENCE_PAIRS:
        if pair not in seen:
            chosen.append(pair)
            seen.add(pair)

    for method in METHOD_ORDER:
        subset = ranking_df.loc[
            (ranking_df["method"] == method)
            & (ranking_df["metafeature"].isin(preferred))
        ]
        subset = subset.sort_values("mean_abs_shap_interaction", ascending=False).head(
            2
        )
        for _, row in subset.iterrows():
            pair = (method, str(row["metafeature"]))
            if pair not in seen:
                chosen.append(pair)
                seen.add(pair)
    return chosen


def create_shap_dependence_plots(
    feature_names: list[str],
    X_transformed: np.ndarray,
    shap_values: np.ndarray,
    ranking_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    output_rows: list[dict[str, str]] = []
    available_features = set(feature_names)
    for method, feature in select_dependence_pairs(ranking_df):
        feature_name = f"num__{feature}"
        method_feature = f"cat__method_{method}"
        if (
            feature_name not in available_features
            or method_feature not in available_features
        ):
            continue
        plt.figure(figsize=(7.2, 5.2))
        shap.dependence_plot(
            feature_name,
            shap_values,
            X_transformed,
            feature_names=feature_names,
            interaction_index=method_feature,
            show=False,
            alpha=0.85,
            dot_size=28,
        )
        ax = plt.gca()
        ax.set_title(
            f"SHAP dependence | {METHOD_LABELS.get(method, method)} × {friendly_name(feature)}",
            fontsize=12,
        )
        ax.set_xlabel(friendly_name(feature))
        ax.set_ylabel("SHAP value")
        output_path = output_dir / f"dependence_{method}_{feature}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        output_rows.append(
            {"method": method, "metafeature": feature, "figure_path": str(output_path)}
        )
    return pd.DataFrame(output_rows)


def build_formula(feature: str, include_model: bool) -> str:
    formula = f"performance ~ C(method) * {feature}"
    if include_model:
        formula += " + C(model)"
    return formula


def compute_omnibus_pvalue(result, feature: str) -> float | None:
    interaction_terms = [
        name
        for name in result.params.index
        if name.startswith("C(method)[T.") and name.endswith(f":{feature}")
    ]
    if not interaction_terms:
        return None
    hypothesis = ", ".join(f"{term} = 0" for term in interaction_terms)
    try:
        pvalue = float(result.f_test(hypothesis).pvalue)
    except Exception:
        return None
    return pvalue


def fit_interaction_models(
    clean_df: pd.DataFrame, selected_features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interaction_rows: list[dict[str, object]] = []
    appendix_rows: list[dict[str, object]] = []
    include_model = (
        "model" in clean_df.columns and clean_df["model"].nunique(dropna=True) > 1
    )

    for feature in selected_features:
        for model_spec in ["unadjusted", "adjusted"]:
            formula = build_formula(feature, include_model and model_spec == "adjusted")
            result = smf.ols(formula, data=clean_df).fit(cov_type="HC3")
            omnibus_pvalue = compute_omnibus_pvalue(result, feature)
            conf = result.conf_int()

            for term_name in result.params.index:
                term_type = (
                    "interaction"
                    if term_name.startswith("C(method)[T.")
                    and term_name.endswith(f":{feature}")
                    else "other"
                )
                ci_low, ci_high = conf.loc[term_name].tolist()
                row = {
                    "meta_feature": feature,
                    "model_spec": model_spec,
                    "model_formula": formula,
                    "term_name": term_name,
                    "term_type": term_type,
                    "coefficient": float(result.params[term_name]),
                    "std_error": float(result.bse[term_name]),
                    "p_value": float(result.pvalues[term_name]),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "r2": float(result.rsquared),
                    "adjusted_r2": float(result.rsquared_adj),
                    "aic": float(result.aic),
                    "bic": float(result.bic),
                    "n_obs": int(result.nobs),
                    "cov_type": str(result.cov_type),
                    "omnibus_interaction_p_value": omnibus_pvalue,
                }
                appendix_rows.append(row)

                if term_type == "interaction":
                    method = term_name.split("[T.", 1)[1].split("]", 1)[0]
                    interaction_rows.append(
                        {
                            "meta_feature": feature,
                            "model_spec": model_spec,
                            "model_formula": formula,
                            "interaction_term": term_name,
                            "method": method,
                            "coefficient": float(result.params[term_name]),
                            "std_error": float(result.bse[term_name]),
                            "p_value": float(result.pvalues[term_name]),
                            "ci_low": float(ci_low),
                            "ci_high": float(ci_high),
                            "direction": "positive"
                            if result.params[term_name] >= 0
                            else "negative",
                            "r2": float(result.rsquared),
                            "adjusted_r2": float(result.rsquared_adj),
                            "aic": float(result.aic),
                            "bic": float(result.bic),
                            "n_obs": int(result.nobs),
                            "cov_type": str(result.cov_type),
                            "omnibus_interaction_p_value": omnibus_pvalue,
                        }
                    )

    interaction_df = pd.DataFrame(interaction_rows)
    appendix_df = pd.DataFrame(appendix_rows)
    return interaction_df, appendix_df


def create_ranked_interaction_summary(
    interaction_df: pd.DataFrame, shap_rankings: pd.DataFrame
) -> pd.DataFrame:
    adjusted = interaction_df.loc[interaction_df["model_spec"] == "adjusted"].copy()
    if adjusted.empty:
        return pd.DataFrame()
    adjusted["abs_coefficient"] = adjusted["coefficient"].abs()
    strongest = (
        adjusted.sort_values(
            ["meta_feature", "abs_coefficient"], ascending=[True, False]
        )
        .groupby("meta_feature", as_index=False)
        .head(1)
    )
    strongest = strongest.rename(
        columns={"interaction_term": "strongest_interaction_term"}
    )

    shap_lookup = shap_rankings.copy()
    shap_lookup["method"] = shap_lookup["method"].astype(str)
    merged = strongest.merge(
        shap_lookup[
            ["method", "metafeature", "mean_abs_shap_interaction", "rank_within_method"]
        ],
        left_on=["method", "meta_feature"],
        right_on=["method", "metafeature"],
        how="left",
    )
    merged = merged.drop(columns=["metafeature"])
    return merged.sort_values(
        ["abs_coefficient", "mean_abs_shap_interaction"], ascending=[False, False]
    ).reset_index(drop=True)


def create_main_interaction_plots(
    clean_df: pd.DataFrame, selected_features: list[str], output_dir: Path
) -> pd.DataFrame:
    include_model = (
        "model" in clean_df.columns and clean_df["model"].nunique(dropna=True) > 1
    )
    model_levels = [
        m for m in MODEL_ORDER if m in clean_df["model"].unique()
    ] or sorted(clean_df["model"].dropna().unique().tolist())
    output_rows: list[dict[str, str]] = []

    for feature in selected_features:
        formula = build_formula(feature, include_model)
        result = smf.ols(formula, data=clean_df).fit(cov_type="HC3")
        feature_min, feature_max = clean_df[feature].min(), clean_df[feature].max()
        grid = np.linspace(feature_min, feature_max, 100)

        fig, ax = plt.subplots(figsize=(7.4, 5.6))
        for method in METHOD_ORDER:
            if method not in clean_df["method"].unique():
                continue
            pred_rows = []
            for x_value in grid:
                if include_model:
                    for model_name in model_levels:
                        pred_rows.append(
                            {"method": method, feature: x_value, "model": model_name}
                        )
                else:
                    pred_rows.append({"method": method, feature: x_value})
            pred_df = pd.DataFrame(pred_rows)
            prediction = result.get_prediction(pred_df).summary_frame()
            pred_df["mean"] = prediction["mean"].to_numpy()
            pred_df["mean_ci_lower"] = prediction["mean_ci_lower"].to_numpy()
            pred_df["mean_ci_upper"] = prediction["mean_ci_upper"].to_numpy()
            line_df = pred_df.groupby(feature, as_index=False)[
                ["mean", "mean_ci_lower", "mean_ci_upper"]
            ].mean()

            ax.plot(
                line_df[feature],
                line_df["mean"],
                color=METHOD_COLORS.get(method, "#333333"),
                linewidth=2.2,
                label=METHOD_LABELS.get(method, method),
            )
            ax.fill_between(
                line_df[feature],
                line_df["mean_ci_lower"],
                line_df["mean_ci_upper"],
                color=METHOD_COLORS.get(method, "#333333"),
                alpha=0.12,
            )

            scatter_df = clean_df.loc[clean_df["method"] == method]
            ax.scatter(
                scatter_df[feature],
                scatter_df["performance"],
                s=42,
                color=METHOD_COLORS.get(method, "#333333"),
                alpha=0.55,
                edgecolor="white",
                linewidth=0.4,
            )

        ax.set_title(f"Predicted OOD ROC-AUC by {friendly_name(feature)}", fontsize=13)
        ax.set_xlabel(friendly_name(feature))
        ax.set_ylabel("Predicted OOD ROC-AUC")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, ncol=1, fontsize=9, loc="best")
        fig.tight_layout()

        output_path = output_dir / f"main_interaction_{feature}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        output_rows.append({"meta_feature": feature, "figure_path": str(output_path)})

    return pd.DataFrame(output_rows)


def build_delta_vs_m0_table(
    clean_df: pd.DataFrame, decision_features: list[str]
) -> pd.DataFrame:
    base_cols = [
        "dataset",
        "target_site",
        "target_site_display",
        "model",
        "performance",
    ]
    base_cols.extend(
        [feature for feature in decision_features if feature in clean_df.columns]
    )
    baseline = (
        clean_df.loc[clean_df["method"] == "M0", base_cols]
        .rename(columns={"performance": "m0_ood_roc_auc"})
        .copy()
    )
    delta_df = clean_df.merge(
        baseline,
        on=[
            "dataset",
            "target_site",
            "target_site_display",
            "model",
            *[f for f in decision_features if f in clean_df.columns],
        ],
        how="left",
    )
    delta_df["delta_vs_M0"] = delta_df["performance"] - delta_df["m0_ood_roc_auc"]
    return delta_df


def create_delta_vs_m0_plots(
    delta_df: pd.DataFrame, decision_features: list[str], output_dir: Path
) -> pd.DataFrame:
    include_model = (
        "model" in delta_df.columns and delta_df["model"].nunique(dropna=True) > 1
    )
    model_levels = [
        m for m in MODEL_ORDER if m in delta_df["model"].unique()
    ] or sorted(delta_df["model"].dropna().unique().tolist())
    plotting_df = delta_df.loc[delta_df["method"] != "M0"].copy()
    output_rows: list[dict[str, str]] = []

    for feature in decision_features:
        if feature not in plotting_df.columns:
            continue
        formula = f"delta_vs_M0 ~ C(method) * {feature}"
        if include_model:
            formula += " + C(model)"
        result = smf.ols(formula, data=plotting_df).fit(cov_type="HC3")
        grid = np.linspace(plotting_df[feature].min(), plotting_df[feature].max(), 100)

        fig, ax = plt.subplots(figsize=(7.4, 5.6))
        for method in [m for m in METHOD_ORDER if m != "M0"]:
            if method not in plotting_df["method"].unique():
                continue
            pred_rows = []
            for x_value in grid:
                if include_model:
                    for model_name in model_levels:
                        pred_rows.append(
                            {"method": method, feature: x_value, "model": model_name}
                        )
                else:
                    pred_rows.append({"method": method, feature: x_value})
            pred_df = pd.DataFrame(pred_rows)
            prediction = result.get_prediction(pred_df).summary_frame()
            pred_df["mean"] = prediction["mean"].to_numpy()
            pred_df["mean_ci_lower"] = prediction["mean_ci_lower"].to_numpy()
            pred_df["mean_ci_upper"] = prediction["mean_ci_upper"].to_numpy()
            line_df = pred_df.groupby(feature, as_index=False)[
                ["mean", "mean_ci_lower", "mean_ci_upper"]
            ].mean()

            ax.plot(
                line_df[feature],
                line_df["mean"],
                color=METHOD_COLORS.get(method, "#333333"),
                linewidth=2.2,
                label=METHOD_LABELS.get(method, method),
            )
            ax.fill_between(
                line_df[feature],
                line_df["mean_ci_lower"],
                line_df["mean_ci_upper"],
                color=METHOD_COLORS.get(method, "#333333"),
                alpha=0.12,
            )

            scatter_df = plotting_df.loc[plotting_df["method"] == method]
            ax.scatter(
                scatter_df[feature],
                scatter_df["delta_vs_M0"],
                s=42,
                color=METHOD_COLORS.get(method, "#333333"),
                alpha=0.58,
                edgecolor="white",
                linewidth=0.4,
            )

        ax.axhline(0, color="black", linewidth=1.1, linestyle="--", alpha=0.8)
        ax.set_title(
            f"Delta OOD ROC-AUC vs M0 by {friendly_name(feature)}", fontsize=13
        )
        ax.set_xlabel(friendly_name(feature))
        ax.set_ylabel("Delta OOD ROC-AUC vs M0")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()

        output_path = output_dir / f"delta_vs_M0_{feature}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        output_rows.append({"meta_feature": feature, "figure_path": str(output_path)})

    return pd.DataFrame(output_rows)


def build_bin_map(values: pd.Series) -> tuple[dict[float, str], list[str]]:
    unique_values = pd.Series(sorted(pd.Series(values).dropna().unique()))
    if unique_values.empty:
        return {}, []
    n_bins = min(3, len(unique_values))
    if n_bins <= 1:
        label = "single"
        return {float(unique_values.iloc[0]): label}, [label]
    labels = ["low", "medium", "high"] if n_bins == 3 else ["low", "high"]
    ranked = pd.Series(np.arange(len(unique_values)))
    bins = pd.qcut(ranked, q=n_bins, labels=labels)
    mapping = {float(value): str(label) for value, label in zip(unique_values, bins)}
    return mapping, labels


def add_condition_bins(
    df: pd.DataFrame, feature: str
) -> tuple[pd.DataFrame, list[str]]:
    mapping, label_order = build_bin_map(df[feature])
    out = df.copy()
    out["condition_bin"] = out[feature].map(mapping)
    return out, label_order


def summarize_grouped_metric(
    df: pd.DataFrame, feature: str, value_column: str
) -> tuple[pd.DataFrame, list[str]]:
    working_df, label_order = add_condition_bins(df, feature)
    summary = (
        working_df.dropna(subset=["condition_bin"])
        .groupby(["condition_bin", "method"], as_index=False)
        .agg(
            mean_value=(value_column, "mean"),
            std_value=(value_column, "std"),
            n_obs=(value_column, "count"),
        )
    )
    summary["std_value"] = summary["std_value"].fillna(0.0)
    summary["standard_error"] = summary["std_value"] / np.sqrt(
        summary["n_obs"].clip(lower=1)
    )
    summary["meta_feature"] = feature
    return summary, label_order


def create_grouped_barplot(
    summary_df: pd.DataFrame,
    feature: str,
    value_label: str,
    title: str,
    output_path: Path,
    include_m0: bool,
) -> None:
    methods = METHOD_ORDER if include_m0 else [m for m in METHOD_ORDER if m != "M0"]
    bin_order = [
        label
        for label in ["low", "medium", "high", "single"]
        if label in summary_df["condition_bin"].unique()
    ]
    if not bin_order:
        return
    x = np.arange(len(bin_order))
    width = 0.14 if include_m0 else 0.18

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    for idx, method in enumerate(methods):
        subset = summary_df.loc[summary_df["method"] == method].set_index(
            "condition_bin"
        )
        means = [
            float(subset.loc[label, "mean_value"]) if label in subset.index else np.nan
            for label in bin_order
        ]
        errors = [
            float(subset.loc[label, "standard_error"]) if label in subset.index else 0.0
            for label in bin_order
        ]
        offset = (idx - (len(methods) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=errors,
            color=METHOD_COLORS.get(method, "#333333"),
            alpha=0.9,
            capsize=3,
            label=METHOD_LABELS.get(method, method),
        )

    if not include_m0:
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([label.title() for label in bin_order])
    ax.set_xlabel(f"{friendly_name(feature)} condition bin")
    ax.set_ylabel(value_label)
    ax.set_title(title, fontsize=13)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_binned_condition_outputs(
    clean_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    decision_features: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[pd.DataFrame] = []
    figure_rows: list[dict[str, str]] = []

    absolute_df = clean_df.copy()
    delta_nonbaseline = delta_df.loc[delta_df["method"] != "M0"].copy()

    for feature in decision_features:
        abs_summary, _ = summarize_grouped_metric(absolute_df, feature, "performance")
        abs_summary["summary_type"] = "absolute_ood_roc_auc"
        summary_rows.append(abs_summary)
        abs_path = output_dir / f"absolute_ood_roc_auc_by_{feature}.png"
        create_grouped_barplot(
            abs_summary,
            feature,
            "Mean OOD ROC-AUC",
            f"OOD ROC-AUC by method across {friendly_name(feature)} bins",
            abs_path,
            include_m0=True,
        )
        figure_rows.append(
            {
                "meta_feature": feature,
                "summary_type": "absolute_ood_roc_auc",
                "figure_path": str(abs_path),
            }
        )

        delta_summary, _ = summarize_grouped_metric(
            delta_nonbaseline, feature, "delta_vs_M0"
        )
        delta_summary["summary_type"] = "delta_vs_M0"
        summary_rows.append(delta_summary)
        delta_path = output_dir / f"delta_vs_M0_by_{feature}.png"
        create_grouped_barplot(
            delta_summary,
            feature,
            "Mean delta OOD ROC-AUC vs M0",
            f"Delta vs M0 across {friendly_name(feature)} bins",
            delta_path,
            include_m0=False,
        )
        figure_rows.append(
            {
                "meta_feature": feature,
                "summary_type": "delta_vs_M0",
                "figure_path": str(delta_path),
            }
        )

    summary_df = (
        pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    )
    figure_index = pd.DataFrame(figure_rows)
    return summary_df, figure_index


def compute_best_method_by_comparison(
    clean_df: pd.DataFrame, decision_features: list[str]
) -> pd.DataFrame:
    working_df = clean_df.copy()
    working_df["method_rank"] = (
        working_df["method"]
        .map({method: idx for idx, method in enumerate(METHOD_ORDER)})
        .fillna(999)
    )
    sort_cols = ["dataset", "target_site", "model", "performance", "method_rank"]
    winners = (
        working_df.sort_values(sort_cols, ascending=[True, True, True, False, True])
        .groupby(["dataset", "target_site", "model"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    keep_cols = [
        "dataset",
        "target_site",
        "target_site_display",
        "model",
        "method",
        "performance",
    ]
    keep_cols.extend(
        [feature for feature in decision_features if feature in winners.columns]
    )
    winners = winners[keep_cols].rename(
        columns={"method": "best_method", "performance": "best_ood_roc_auc"}
    )
    return winners


def create_win_rate_by_condition_outputs(
    best_df: pd.DataFrame, decision_features: list[str], output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    figure_rows: list[dict[str, str]] = []
    methods = METHOD_ORDER

    for feature in decision_features:
        binned_df, label_order = add_condition_bins(best_df, feature)
        for condition_label in label_order:
            subset = binned_df.loc[binned_df["condition_bin"] == condition_label]
            total = int(len(subset))
            for method in methods:
                wins = int((subset["best_method"] == method).sum())
                win_rate_pct = 100.0 * wins / total if total > 0 else np.nan
                rows.append(
                    {
                        "meta_feature": feature,
                        "condition_bin": condition_label,
                        "method": method,
                        "win_count": wins,
                        "total_comparisons": total,
                        "win_rate_pct": win_rate_pct,
                    }
                )

        summary_df = pd.DataFrame(
            [row for row in rows if row["meta_feature"] == feature]
        )
        if summary_df.empty:
            continue
        plot_df = summary_df.copy()
        bin_order = [
            label
            for label in ["low", "medium", "high", "single"]
            if label in plot_df["condition_bin"].unique()
        ]
        x = np.arange(len(bin_order))
        bottoms = np.zeros(len(bin_order))
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        for method in methods:
            subset = plot_df.loc[plot_df["method"] == method].set_index("condition_bin")
            values = np.array(
                [
                    float(subset.loc[label, "win_rate_pct"])
                    if label in subset.index
                    else 0.0
                    for label in bin_order
                ]
            )
            ax.bar(
                x,
                values,
                bottom=bottoms,
                color=METHOD_COLORS.get(method, "#333333"),
                alpha=0.92,
                label=METHOD_LABELS.get(method, method),
            )
            bottoms += values

        ax.set_xticks(x)
        ax.set_xticklabels([label.title() for label in bin_order])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Win rate (%)")
        ax.set_xlabel(f"{friendly_name(feature)} condition bin")
        ax.set_title(
            f"Method win rate across {friendly_name(feature)} bins", fontsize=13
        )
        ax.legend(
            frameon=False,
            fontsize=8,
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
        )
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        output_path = output_dir / f"win_rate_by_{feature}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figure_rows.append({"meta_feature": feature, "figure_path": str(output_path)})

    return pd.DataFrame(rows), pd.DataFrame(figure_rows)


def create_recommendation_heatmaps(
    clean_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    preprocessor: ColumnTransformer,
    model: RandomForestRegressor,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = [
        method for method in METHOD_ORDER if method in clean_df["method"].unique()
    ]
    model_levels = [
        m for m in MODEL_ORDER if m in clean_df["model"].unique()
    ] or sorted(clean_df["model"].dropna().unique().tolist())
    dataset_value = (
        clean_df["dataset"].mode().iloc[0]
        if "dataset" in categorical_features
        else None
    )
    numeric_medians = {
        feature: float(clean_df[feature].median()) for feature in numeric_features
    }

    grid_rows: list[dict[str, object]] = []
    figure_rows: list[dict[str, str]] = []
    method_to_code = {method: idx for idx, method in enumerate(methods)}
    cmap = ListedColormap([METHOD_COLORS[m] for m in methods])
    norm = BoundaryNorm(np.arange(-0.5, len(methods) + 0.5), cmap.N)

    for x_feature, y_feature in RECOMMENDATION_FEATURE_PAIRS:
        if x_feature not in numeric_features or y_feature not in numeric_features:
            continue
        x_values = np.linspace(clean_df[x_feature].min(), clean_df[x_feature].max(), 25)
        y_values = np.linspace(clean_df[y_feature].min(), clean_df[y_feature].max(), 25)

        pred_rows = []
        for y_value in y_values:
            for x_value in x_values:
                for method in methods:
                    for model_name in model_levels:
                        row = {
                            feature: numeric_medians[feature]
                            for feature in numeric_features
                        }
                        row[x_feature] = float(x_value)
                        row[y_feature] = float(y_value)
                        if "method" in categorical_features:
                            row["method"] = method
                        if "model" in categorical_features:
                            row["model"] = model_name
                        if "dataset" in categorical_features:
                            row["dataset"] = dataset_value
                        row["_x_value"] = float(x_value)
                        row["_y_value"] = float(y_value)
                        pred_rows.append(row)

        pred_df = pd.DataFrame(pred_rows)
        X_pred = pred_df[numeric_features + categorical_features].copy()
        pred_df["predicted_ood_roc_auc"] = model.predict(preprocessor.transform(X_pred))
        agg_df = (
            pred_df.groupby(["_x_value", "_y_value", "method"], as_index=False)[
                "predicted_ood_roc_auc"
            ]
            .mean()
            .rename(columns={"_x_value": "x_value", "_y_value": "y_value"})
        )
        wide_df = agg_df.pivot_table(
            index=["x_value", "y_value"],
            columns="method",
            values="predicted_ood_roc_auc",
        ).reset_index()
        wide_df.columns.name = None

        for _, row in wide_df.iterrows():
            preds = {
                method: float(row[method])
                for method in methods
                if method in wide_df.columns
            }
            best_method = max(preds, key=preds.get)
            gain_over_m0 = preds[best_method] - preds.get("M0", np.nan)
            output_row = {
                "x_feature": x_feature,
                "y_feature": y_feature,
                "x_value": float(row["x_value"]),
                "y_value": float(row["y_value"]),
                "best_method": best_method,
                "best_predicted_ood_roc_auc": preds[best_method],
                "expected_gain_over_M0": float(gain_over_m0),
            }
            for method, value in preds.items():
                output_row[f"pred_{method}"] = float(value)
            grid_rows.append(output_row)

        pair_df = pd.DataFrame(
            [
                row
                for row in grid_rows
                if row["x_feature"] == x_feature and row["y_feature"] == y_feature
            ]
        )
        if pair_df.empty:
            continue

        method_grid = pair_df.pivot(
            index="y_value", columns="x_value", values="best_method"
        )
        code_grid = method_grid.replace(method_to_code).astype(float)
        gain_grid = pair_df.pivot(
            index="y_value", columns="x_value", values="expected_gain_over_M0"
        )

        fig, ax = plt.subplots(figsize=(7.0, 5.8))
        im = ax.imshow(
            code_grid.to_numpy(),
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()],
        )
        cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(methods)))
        cbar.ax.set_yticklabels([METHOD_LABELS[m] for m in methods])
        ax.set_xlabel(friendly_name(x_feature))
        ax.set_ylabel(friendly_name(y_feature))
        ax.set_title(
            f"Recommended method | {friendly_name(x_feature)} × {friendly_name(y_feature)}",
            fontsize=12,
        )
        fig.tight_layout()
        best_path = output_dir / f"best_method_heatmap_{x_feature}_vs_{y_feature}.png"
        fig.savefig(best_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figure_rows.append(
            {
                "plot_type": "best_method",
                "x_feature": x_feature,
                "y_feature": y_feature,
                "figure_path": str(best_path),
            }
        )

        fig, ax = plt.subplots(figsize=(7.0, 5.8))
        im = ax.imshow(
            gain_grid.to_numpy(),
            origin="lower",
            aspect="auto",
            cmap="RdYlGn",
            extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()],
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Expected gain over M0")
        ax.set_xlabel(friendly_name(x_feature))
        ax.set_ylabel(friendly_name(y_feature))
        ax.set_title(
            f"Expected gain over M0 | {friendly_name(x_feature)} × {friendly_name(y_feature)}",
            fontsize=12,
        )
        fig.tight_layout()
        gain_path = output_dir / f"gain_over_M0_heatmap_{x_feature}_vs_{y_feature}.png"
        fig.savefig(gain_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figure_rows.append(
            {
                "plot_type": "gain_over_M0",
                "x_feature": x_feature,
                "y_feature": y_feature,
                "figure_path": str(gain_path),
            }
        )

    return pd.DataFrame(grid_rows), pd.DataFrame(figure_rows)


def create_thesis_method_recommendation_summary(
    absolute_binned: pd.DataFrame,
    delta_binned: pd.DataFrame,
    win_rate_df: pd.DataFrame,
) -> pd.DataFrame:
    features_for_summary = [
        feature
        for feature in [
            "overall_metadata_shift_score",
            "site_distance_composite",
            "sample_size",
            "class_entropy",
        ]
        if feature in absolute_binned["meta_feature"].unique()
    ]
    rows: list[dict[str, object]] = []
    for feature in features_for_summary:
        for condition in ["low", "high"]:
            abs_subset = absolute_binned.loc[
                (absolute_binned["meta_feature"] == feature)
                & (absolute_binned["condition_bin"] == condition)
            ]
            delta_subset = delta_binned.loc[
                (delta_binned["meta_feature"] == feature)
                & (delta_binned["condition_bin"] == condition)
            ]
            win_subset = win_rate_df.loc[
                (win_rate_df["meta_feature"] == feature)
                & (win_rate_df["condition_bin"] == condition)
            ]
            if abs_subset.empty or win_subset.empty:
                continue
            best_abs_row = abs_subset.sort_values("mean_value", ascending=False).iloc[0]
            best_method = str(best_abs_row["method"])
            delta_method_row = (
                delta_subset.sort_values("mean_value", ascending=False).iloc[0]
                if not delta_subset.empty
                else None
            )
            win_row = win_subset.sort_values("win_rate_pct", ascending=False).iloc[0]

            best_delta_method = (
                str(delta_method_row["method"])
                if delta_method_row is not None
                else "M0"
            )
            best_delta_value = (
                float(delta_method_row["mean_value"])
                if delta_method_row is not None
                else 0.0
            )
            win_method = str(win_row["method"])
            win_rate = float(win_row["win_rate_pct"])

            preferred_method = best_method
            if best_method != "M0" and best_delta_value <= 0:
                preferred_method = "M0"

            if preferred_method == "M0":
                interpretation = "baseline remained competitive in this condition, suggesting that more complex adaptation was not consistently necessary."
            elif (
                preferred_method in {"M3", "M4"}
                and feature == "sample_size"
                and condition == "high"
            ):
                interpretation = "transductive-style adaptation appears more useful when more target data are available."
            elif (
                preferred_method in {"M1", "M3", "M4"}
                and feature
                in {"overall_metadata_shift_score", "site_distance_composite"}
                and condition == "high"
            ):
                interpretation = "the preferred method under stronger shift was not the baseline, suggesting condition-dependent value of adaptation or harmonization."
            else:
                interpretation = "this method showed the highest average OOD ROC-AUC in the condition-specific summary, but the edge should be interpreted as exploratory."

            evidence = (
                f"Best mean OOD ROC-AUC={best_abs_row['mean_value']:.3f}; "
                f"top delta vs M0={best_delta_method} ({best_delta_value:+.3f}); "
                f"highest win-rate={win_method} ({win_rate:.1f}%)"
            )

            rows.append(
                {
                    "Condition": f"{friendly_name(feature)} | {condition}",
                    "Best / preferred method": preferred_method,
                    "Evidence": evidence,
                    "Interpretation": interpretation,
                }
            )
    return pd.DataFrame(rows)


def create_thesis_summary_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return
    display_df = summary_df.copy()
    max_rows = min(8, len(display_df))
    display_df = display_df.head(max_rows)
    wrapped = display_df.copy()
    wrapped["Evidence"] = wrapped["Evidence"].str.wrap(40)
    wrapped["Interpretation"] = wrapped["Interpretation"].str.wrap(48)

    fig_height = 0.8 + max_rows * 0.75
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=wrapped.values,
        colLabels=wrapped.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e9ecef")
    ax.set_title(
        "Thesis-oriented method recommendation summary | OOD ROC-AUC",
        fontsize=13,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_direction(coefficient: float) -> str:
    if coefficient > 0:
        return "more favorable as the metafeature increases"
    if coefficient < 0:
        return "less favorable as the metafeature increases"
    return "little directional change"


def interpretation_text(
    method: str, feature: str, coefficient: float, p_value: float | None
) -> str:
    method_label = METHOD_LABELS.get(method, method).split(": ", 1)[-1]
    feature_label = friendly_name(feature).lower()
    if coefficient > 0:
        base = f"{method_label} appears to gain relative advantage as {feature_label} increases"
    elif coefficient < 0:
        base = f"{method_label} appears to lose relative advantage as {feature_label} increases"
    else:
        base = f"{method_label} shows little relative change across {feature_label}"
    if p_value is None or pd.isna(p_value):
        return base + "; formal p-value was not available for this contrast."
    if p_value < 0.05:
        return (
            base
            + "; this interaction was among the clearer statistical signals in the exploratory models."
        )
    return (
        base
        + "; however, the associated p-value remained weak, so this should be treated as an exploratory pattern."
    )


def build_main_text_summary(ranked_df: pd.DataFrame) -> pd.DataFrame:
    if ranked_df.empty:
        return pd.DataFrame(
            columns=[
                "interaction",
                "direction",
                "strength/effect_size",
                "p_value",
                "interpretation",
            ]
        )
    rows: list[dict[str, object]] = []
    for _, row in ranked_df.iterrows():
        interaction_label = (
            f"{row['method']} × {friendly_name(str(row['meta_feature']))}"
        )
        shap_strength = row.get("mean_abs_shap_interaction", np.nan)
        effect_size = (
            f"coef={row['coefficient']:.4f}; mean|SHAP interaction|={shap_strength:.6f}"
            if pd.notna(shap_strength)
            else f"coef={row['coefficient']:.4f}"
        )
        rows.append(
            {
                "interaction": interaction_label,
                "direction": summarize_direction(float(row["coefficient"])),
                "strength/effect_size": effect_size,
                "p_value": float(row["p_value"]),
                "interpretation": interpretation_text(
                    str(row["method"]),
                    str(row["meta_feature"]),
                    float(row["coefficient"]),
                    float(row["p_value"]),
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    clean_df: pd.DataFrame,
    predictors: list[str],
    shap_summary: pd.DataFrame,
    ranked_df: pd.DataFrame,
    main_summary: pd.DataFrame,
    thesis_summary: pd.DataFrame,
    checklist: dict[str, bool],
    report_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("Interaction analysis final reporting")
    lines.append("")
    lines.append("A. Main findings suitable for the thesis text")
    lines.append(f"- Target analyzed: {TARGET_METRIC}")
    lines.append(
        f"- Meta-samples analyzed: {len(clean_df)} across {clean_df['target_site'].nunique()} target sites and {clean_df['model'].nunique()} classifiers."
    )
    lines.append(
        "- The main interaction effects are interpreted primarily through predicted-effect plots with one meta-feature at a time and one fitted line per method."
    )
    lines.append(
        "- For decision-oriented reporting, the most important additions are the delta-vs-M0 plots, binned-condition plots, method win-rate plots, and best-method heatmaps."
    )
    lines.append(
        "- These figures are intended for the main Results/Discussion section, while the numerical tables are supporting material."
    )
    lines.append("")
    lines.append("Highlighted condition-dependent patterns:")
    if thesis_summary.empty:
        lines.append("- No decision-oriented summary could be generated.")
    else:
        for _, row in thesis_summary.head(8).iterrows():
            lines.append(
                f"- {row['Condition']}: prefer {row['Best / preferred method']}. {row['Interpretation']}"
            )

    lines.append("")
    lines.append("B. Supporting statistical evidence")
    lines.append(
        f"- Clean predictors retained after leakage and duplicate checks: {', '.join(predictors)}"
    )
    lines.append(
        "- SHAP interaction values were computed from a RandomForestRegressor meta-model fitted only on OOD ROC-AUC rows."
    )
    lines.append(
        "- Explicit interaction regressions were fitted one meta-feature at a time using formulas of the form OOD_ROC_AUC ~ C(method) * meta_feature (+ C(model) when available)."
    )
    lines.append(
        "- Robust HC3 standard errors were used for the reported coefficients and p-values."
    )
    lines.append(
        "- Full supporting statistics, including confidence intervals, R², adjusted R², AIC, BIC, model formula, and number of observations, are saved in appendix tables."
    )
    lines.append(
        "- Delta-vs-M0 summaries were computed at the target-site/model level, so values above zero indicate improvement over the baseline and values below zero indicate deterioration."
    )
    lines.append(
        "- Win-rate summaries count how often each method achieved the highest observed OOD ROC-AUC within each low/medium/high condition bin."
    )
    lines.append("")
    lines.append("Strongest SHAP interactions by method:")
    for method in METHOD_ORDER:
        subset = shap_summary.loc[shap_summary["method"] == method].head(3)
        if subset.empty:
            continue
        snippets = [
            f"{friendly_name(str(mf))} (mean|interaction|={val:.6f})"
            for mf, val in zip(
                subset["metafeature"], subset["mean_abs_shap_interaction"]
            )
        ]
        lines.append(f"- {METHOD_LABELS.get(method, method)}: " + "; ".join(snippets))

    lines.append("")
    lines.append("C. Cautions / limitations")
    lines.append(
        "- The UCI meta-learning table contains only 60 OOD ROC-AUC observations, so interaction estimates are necessarily unstable and should be interpreted cautiously."
    )
    lines.append(
        "- Numerical significance is secondary here: the primary goal is to identify condition-dependent method behavior, not to claim a universally best method."
    )
    lines.append(
        "- The delta, binning, win-rate, and recommendation heatmap outputs are descriptive and exploratory. They are meant to clarify practical patterns, not to provide causal or fully confirmatory decision rules."
    )
    lines.append(
        "- Lack of strong p-values should not be overinterpreted as absence of any pattern; it more likely reflects limited meta-sample size and correlated meta-features."
    )
    lines.append(
        "- Conversely, visually strong patterns in predicted-effect plots should still be described as associations suggested by the meta-model, not causal effects."
    )
    lines.append("")
    lines.append("Checklist")
    for key, value in checklist.items():
        lines.append(f"- {'[x]' if value else '[ ]'} {key}")

    report_path.write_text("\n".join(lines))


def save_tables(
    df: pd.DataFrame, csv_path: Path, xlsx_path: Path | None = None
) -> None:
    df.to_csv(csv_path, index=False)
    if xlsx_path is not None:
        df.to_excel(xlsx_path, index=False)


def main() -> None:
    subdirs = make_dirs()

    source_df = pd.read_csv(SOURCE_META_TABLE)
    clean_df, predictor_inventory, numeric_features, categorical_features = (
        build_clean_analysis_dataframe(source_df)
    )
    selected_features = [
        feature for feature in MAIN_FEATURE_PRIORITY if feature in numeric_features
    ]
    decision_features = [
        feature for feature in DECISION_FEATURE_PRIORITY if feature in numeric_features
    ]

    (
        X,
        y,
        preprocessor,
        X_transformed,
        feature_names,
        model,
        shap_values,
        shap_interactions,
    ) = fit_meta_model(clean_df, numeric_features, categorical_features)

    shap_rankings, shap_summary = compute_shap_interaction_tables(
        feature_names, shap_interactions
    )
    dependence_index = create_shap_dependence_plots(
        feature_names=feature_names,
        X_transformed=X_transformed,
        shap_values=shap_values,
        ranking_df=shap_rankings,
        output_dir=subdirs["shap_dependence_plots"],
    )
    create_shap_interaction_figure(
        shap_summary,
        subdirs["shap_interactions"] / "shap_interaction_ranking_by_method.png",
    )

    interaction_df, appendix_df = fit_interaction_models(clean_df, selected_features)
    ranked_summary = create_ranked_interaction_summary(interaction_df, shap_rankings)
    main_summary = build_main_text_summary(ranked_summary)
    main_plot_index = create_main_interaction_plots(
        clean_df, selected_features, subdirs["main_figures"]
    )
    delta_df = build_delta_vs_m0_table(clean_df, decision_features)
    delta_plot_index = create_delta_vs_m0_plots(
        delta_df, decision_features, subdirs["delta_vs_m0_figures"]
    )
    binned_condition_summary, binned_figure_index = create_binned_condition_outputs(
        clean_df=clean_df,
        delta_df=delta_df,
        decision_features=decision_features,
        output_dir=subdirs["binned_condition_figures"],
    )
    best_method_df = compute_best_method_by_comparison(clean_df, decision_features)
    win_rate_summary, win_rate_figure_index = create_win_rate_by_condition_outputs(
        best_df=best_method_df,
        decision_features=decision_features,
        output_dir=subdirs["win_rate_figures"],
    )
    recommendation_grid_table, recommendation_figure_index = (
        create_recommendation_heatmaps(
            clean_df=clean_df,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            preprocessor=preprocessor,
            model=model,
            output_dir=subdirs["recommendation_heatmaps"],
        )
    )
    thesis_summary = create_thesis_method_recommendation_summary(
        absolute_binned=binned_condition_summary.loc[
            binned_condition_summary["summary_type"] == "absolute_ood_roc_auc"
        ].copy(),
        delta_binned=binned_condition_summary.loc[
            binned_condition_summary["summary_type"] == "delta_vs_M0"
        ].copy(),
        win_rate_df=win_rate_summary,
    )
    thesis_summary_figure = (
        subdirs["main_figures"] / "thesis_method_recommendation_summary.png"
    )
    create_thesis_summary_figure(thesis_summary, thesis_summary_figure)

    checklist = {
        "main interaction plots": not main_plot_index.empty,
        "delta vs M0 plots": not delta_plot_index.empty,
        "binned condition plots": not binned_figure_index.empty,
        "win-rate plots": not win_rate_figure_index.empty,
        "recommendation heatmaps": not recommendation_figure_index.empty,
        "SHAP interaction ranking": not shap_summary.empty
        and (
            subdirs["shap_interactions"] / "shap_interaction_ranking_by_method.png"
        ).exists(),
        "SHAP dependence plots": not dependence_index.empty,
        "main text summary table": not main_summary.empty,
        "thesis recommendation summary": not thesis_summary.empty,
        "appendix statistical table": not appendix_df.empty,
        "final interpretation report": False,
    }

    save_tables(
        clean_df,
        subdirs["statistical_tables"] / "clean_analysis_dataframe.csv",
        subdirs["statistical_tables"] / "clean_analysis_dataframe.xlsx",
    )
    save_tables(
        predictor_inventory,
        subdirs["statistical_tables"] / "predictor_inventory.csv",
        subdirs["statistical_tables"] / "predictor_inventory.xlsx",
    )
    save_tables(
        shap_rankings,
        subdirs["shap_interactions"] / "shap_interaction_rankings.csv",
        subdirs["shap_interactions"] / "shap_interaction_rankings.xlsx",
    )
    save_tables(
        shap_summary,
        subdirs["shap_interactions"] / "shap_interaction_summary.csv",
        subdirs["shap_interactions"] / "shap_interaction_summary.xlsx",
    )
    save_tables(
        dependence_index,
        subdirs["shap_dependence_plots"] / "shap_dependence_plot_index.csv",
        subdirs["shap_dependence_plots"] / "shap_dependence_plot_index.xlsx",
    )
    save_tables(
        interaction_df,
        subdirs["statistical_tables"] / "regression_interaction_results.csv",
        subdirs["statistical_tables"] / "regression_interaction_results.xlsx",
    )
    save_tables(
        delta_df,
        subdirs["statistical_tables"] / "delta_vs_M0_table.csv",
        subdirs["statistical_tables"] / "delta_vs_M0_table.xlsx",
    )
    save_tables(
        binned_condition_summary,
        subdirs["statistical_tables"] / "binned_condition_summary.csv",
        subdirs["statistical_tables"] / "binned_condition_summary.xlsx",
    )
    save_tables(
        win_rate_summary,
        subdirs["statistical_tables"] / "method_win_rate_by_condition.csv",
        subdirs["statistical_tables"] / "method_win_rate_by_condition.xlsx",
    )
    save_tables(
        recommendation_grid_table,
        subdirs["statistical_tables"] / "recommendation_grid_table.csv",
        subdirs["statistical_tables"] / "recommendation_grid_table.xlsx",
    )
    save_tables(
        ranked_summary,
        subdirs["statistical_tables"] / "interaction_summary_ranked.csv",
        subdirs["statistical_tables"] / "interaction_summary_ranked.xlsx",
    )
    save_tables(
        main_summary,
        subdirs["statistical_tables"] / "main_text_interaction_summary.csv",
        subdirs["statistical_tables"] / "main_text_interaction_summary.xlsx",
    )
    save_tables(
        thesis_summary,
        subdirs["statistical_tables"] / "thesis_method_recommendation_summary.csv",
        subdirs["statistical_tables"] / "thesis_method_recommendation_summary.xlsx",
    )
    save_tables(
        appendix_df,
        subdirs["appendix_tables"] / "appendix_full_interaction_models.csv",
        subdirs["appendix_tables"] / "appendix_full_interaction_models.xlsx",
    )
    save_tables(
        main_plot_index,
        subdirs["main_figures"] / "main_figure_index.csv",
        subdirs["main_figures"] / "main_figure_index.xlsx",
    )
    save_tables(
        delta_plot_index,
        subdirs["delta_vs_m0_figures"] / "delta_vs_M0_figure_index.csv",
        subdirs["delta_vs_m0_figures"] / "delta_vs_M0_figure_index.xlsx",
    )
    save_tables(
        binned_figure_index,
        subdirs["binned_condition_figures"] / "binned_condition_figure_index.csv",
        subdirs["binned_condition_figures"] / "binned_condition_figure_index.xlsx",
    )
    save_tables(
        win_rate_figure_index,
        subdirs["win_rate_figures"] / "win_rate_figure_index.csv",
        subdirs["win_rate_figures"] / "win_rate_figure_index.xlsx",
    )
    save_tables(
        recommendation_figure_index,
        subdirs["recommendation_heatmaps"] / "recommendation_heatmap_index.csv",
        subdirs["recommendation_heatmaps"] / "recommendation_heatmap_index.xlsx",
    )

    checklist["final interpretation report"] = True
    report_path = subdirs["reports"] / "interaction_analysis_report.txt"
    write_report(
        clean_df=clean_df,
        predictors=selected_features,
        shap_summary=shap_summary,
        ranked_df=ranked_summary,
        main_summary=main_summary,
        thesis_summary=thesis_summary,
        checklist=checklist,
        report_path=report_path,
    )
    checklist["final interpretation report"] = report_path.exists()

    checklist_lines = ["Interaction analysis final reporting checklist", ""]
    for key, value in checklist.items():
        checklist_lines.append(f"- {'[x]' if value else '[ ]'} {key}")
    checklist_text = "\n".join(checklist_lines)
    (subdirs["reports"] / "generation_checklist.txt").write_text(checklist_text)

    summary_payload = {
        "target_metric": TARGET_METRIC,
        "n_rows": int(len(clean_df)),
        "n_target_sites": int(clean_df["target_site"].nunique()),
        "selected_features": selected_features,
        "decision_features": decision_features,
        "checklist": checklist,
    }
    (subdirs["reports"] / "summary.json").write_text(
        json.dumps(summary_payload, indent=2)
    )

    shutil.copy2(
        Path(__file__),
        subdirs["scripts"] / "run_interaction_analysis_final_reporting.py",
    )

    print(checklist_text)


if __name__ == "__main__":
    main()
