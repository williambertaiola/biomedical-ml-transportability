"""Post-experiment analysis: run shap interaction values analysis."""

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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_ROOT = Path(
    os.environ.get("INTERACTION_INPUT_ROOT", _configured_path("results_uci"))
)
REPORTING_ROOT = Path(
    os.environ.get(
        "INTERACTION_REPORTING_ROOT",
        str(INPUT_ROOT / "analysis_final" / "Interaction_analysis_final_reporting"),
    )
)
SOURCE_TABLE = Path(
    os.environ.get(
        "INTERACTION_CLEAN_TABLE",
        str(REPORTING_ROOT / "statistical_tables" / "clean_analysis_dataframe.csv"),
    )
)
OUTPUT_ROOT = REPORTING_ROOT / "shap_interactions"
RANDOM_STATE = 42
METHOD_ORDER = ["M0", "M1", "M2", "M3", "M4"]
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
    "mean_categorical_metadata_shift": "Categorical metadata shift",
    "site_distance_composite": "Domain distance composite",
    "sample_size": "Target sample size",
    "n_source": "Source sample size",
    "n_train_1": "Target train size",
    "n_test_1": "Target test size",
    "class_entropy": "Class entropy",
    "class_balance": "Class balance",
    "minority_class_ratio": "Minority class ratio",
    "missing_percentage": "Missing percentage",
    "euclidean_centroid_distance": "Euclidean centroid distance",
    "mahalanobis_distance": "Mahalanobis distance",
    "wasserstein_mean_distance": "Wasserstein mean distance",
    "jensen_shannon_mean_divergence": "Jensen-Shannon divergence",
    "mmd_rbf": "MMD-RBF",
}
REQUIRED_DEPENDENCE_PAIRS = [
    ("M0", "class_entropy"),
    ("M1", "overall_metadata_shift_score"),
    ("M3", "sample_size"),
    ("M4", "overall_metadata_shift_score"),
]
SHIFT_FEATURES = {
    "overall_metadata_shift_score",
    "mean_continuous_metadata_shift",
    "mean_categorical_metadata_shift",
    "site_distance_composite",
    "euclidean_centroid_distance",
    "mahalanobis_distance",
    "wasserstein_mean_distance",
    "jensen_shannon_mean_divergence",
    "mmd_rbf",
}
SIZE_FEATURES = {"sample_size", "n_source", "n_train_1", "n_test_1"}


def friendly_name(raw_name: str) -> str:
    if raw_name.startswith("num__"):
        raw_name = raw_name.replace("num__", "", 1)
    if raw_name.startswith("cat__method_"):
        method = raw_name.replace("cat__method_", "", 1)
        return METHOD_LABELS.get(method, method)
    if raw_name.startswith("cat__model_"):
        return raw_name.replace("cat__model_", "", 1)
    return FEATURE_LABELS.get(raw_name, raw_name.replace("_", " ").title())


def make_dirs() -> dict[str, Path]:
    subdirs = {
        "barplots": OUTPUT_ROOT / "barplots",
        "dependence_plots": OUTPUT_ROOT / "dependence_plots",
        "reports": OUTPUT_ROOT / "reports",
    }
    for path in [OUTPUT_ROOT, *subdirs.values()]:
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def load_clean_df() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_TABLE)
    if "metric" in df.columns:
        df = df.loc[df["metric"] == "OOD_ROC_AUC"].copy()
    return df.reset_index(drop=True)


def infer_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_features = []
    for column in ["method", "model", "dataset"]:
        if column in df.columns and df[column].nunique(dropna=True) > 1:
            categorical_features.append(column)

    excluded = {
        "dataset",
        "target_site",
        "target_site_display",
        "method",
        "model",
        "model_display",
        "metric",
        "performance",
    }
    numeric_features = []
    for column in df.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        if values.nunique(dropna=True) <= 1:
            continue
        numeric_features.append(column)
    return numeric_features, categorical_features


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
        ]
    )


def fit_meta_model(
    df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]
):
    X = df[numeric_features + categorical_features].copy()
    y = df["performance"].astype(float).to_numpy()
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
    shap_interaction_values = explainer.shap_interaction_values(X_transformed)
    return (
        X,
        y,
        preprocessor,
        X_transformed,
        feature_names,
        model,
        shap_values,
        shap_interaction_values,
    )


def compute_method_interactions(
    feature_names: list[str], shap_interaction_values: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method_features = [
        name for name in feature_names if name.startswith("cat__method_")
    ]
    numeric_features = [name for name in feature_names if name.startswith("num__")]
    index_map = {name: idx for idx, name in enumerate(feature_names)}

    rows: list[dict[str, object]] = []
    for method_feature in method_features:
        method = method_feature.replace("cat__method_", "", 1)
        i = index_map[method_feature]
        for num_feature in numeric_features:
            j = index_map[num_feature]
            value = float(np.mean(np.abs(shap_interaction_values[:, i, j])))
            rows.append(
                {
                    "method": method,
                    "method_feature": method_feature,
                    "metafeature": num_feature.replace("num__", "", 1),
                    "mean_abs_shap_interaction": value,
                }
            )

    long_df = pd.DataFrame(rows)
    long_df = long_df.sort_values(
        ["method", "mean_abs_shap_interaction"], ascending=[True, False]
    ).reset_index(drop=True)
    long_df["rank_within_method"] = long_df.groupby("method").cumcount() + 1
    long_df = long_df.sort_values(
        "mean_abs_shap_interaction", ascending=False
    ).reset_index(drop=True)
    long_df["global_rank"] = np.arange(1, len(long_df) + 1)

    wide_df = long_df.pivot_table(
        index="metafeature",
        columns="method",
        values="mean_abs_shap_interaction",
        aggfunc="first",
    ).reset_index()
    wide_df.columns.name = None
    for method in METHOD_ORDER:
        if method not in wide_df.columns:
            wide_df[method] = np.nan
        wide_df = wide_df.rename(columns={method: f"interaction_{method}"})
    interaction_cols = [f"interaction_{method}" for method in METHOD_ORDER]
    wide_df["strongest_overall_interaction"] = wide_df[interaction_cols].max(axis=1)
    wide_df = wide_df.sort_values(
        "strongest_overall_interaction", ascending=False
    ).reset_index(drop=True)
    wide_df["overall_rank"] = np.arange(1, len(wide_df) + 1)

    top_df = (
        long_df.groupby("method", as_index=False)
        .head(10)
        .sort_values(["method", "rank_within_method"])
        .reset_index(drop=True)
    )
    return wide_df, long_df, top_df


def create_overall_interaction_figure(top_df: pd.DataFrame, output_path: Path) -> None:
    methods_present = [m for m in METHOD_ORDER if m in top_df["method"].unique()]
    ncols = 2
    nrows = math.ceil(len(methods_present) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_visible(False)

    max_val = (
        float(top_df["mean_abs_shap_interaction"].max()) if not top_df.empty else 1.0
    )
    for ax, method in zip(axes, methods_present):
        ax.set_visible(True)
        subset = top_df.loc[top_df["method"] == method].sort_values(
            "mean_abs_shap_interaction"
        )
        ax.barh(
            np.arange(len(subset)),
            subset["mean_abs_shap_interaction"],
            color=METHOD_COLORS.get(method, "#4c78a8"),
            alpha=0.9,
        )
        ax.set_yticks(np.arange(len(subset)))
        ax.set_yticklabels(
            [friendly_name(name) for name in subset["metafeature"]], fontsize=9
        )
        ax.set_xlabel("mean |SHAP interaction|")
        ax.set_xlim(0, max_val * 1.15)
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=11)
        ax.grid(axis="x", alpha=0.2)

    fig.suptitle("SHAP interaction ranking by method | OOD ROC-AUC", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_method_barplots(top_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for method in METHOD_ORDER:
        subset = top_df.loc[top_df["method"] == method].sort_values(
            "mean_abs_shap_interaction"
        )
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        ax.barh(
            np.arange(len(subset)),
            subset["mean_abs_shap_interaction"],
            color=METHOD_COLORS.get(method, "#4c78a8"),
            alpha=0.92,
        )
        ax.set_yticks(np.arange(len(subset)))
        ax.set_yticklabels(
            [friendly_name(name) for name in subset["metafeature"]], fontsize=10
        )
        ax.set_xlabel("mean |SHAP interaction|")
        ax.set_title(
            f"Top SHAP interactions | {METHOD_LABELS.get(method, method)}", fontsize=12
        )
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        output_path = output_dir / f"shap_interaction_barplot_{method}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        rows.append({"method": method, "figure_path": str(output_path)})
    return pd.DataFrame(rows)


def choose_dependence_pairs(rankings_df: pd.DataFrame) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in REQUIRED_DEPENDENCE_PAIRS:
        if pair not in seen:
            pairs.append(pair)
            seen.add(pair)
    for method in METHOD_ORDER:
        subset = rankings_df.loc[rankings_df["method"] == method].head(3)
        for _, row in subset.iterrows():
            pair = (method, str(row["metafeature"]))
            if pair not in seen:
                pairs.append(pair)
                seen.add(pair)
    return pairs


def create_dependence_plots(
    feature_names: list[str],
    X_transformed: np.ndarray,
    shap_values: np.ndarray,
    rankings_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    available = set(feature_names)
    for method, metafeature in choose_dependence_pairs(rankings_df):
        feature_name = f"num__{metafeature}"
        method_feature = f"cat__method_{method}"
        if feature_name not in available or method_feature not in available:
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
            f"SHAP dependence | {METHOD_LABELS.get(method, method)} × {friendly_name(metafeature)}",
            fontsize=12,
        )
        ax.set_xlabel(friendly_name(metafeature))
        ax.set_ylabel("SHAP value")
        plt.tight_layout()
        output_path = output_dir / f"dependence_{method}_{metafeature}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        rows.append(
            {
                "method": method,
                "metafeature": metafeature,
                "figure_path": str(output_path),
            }
        )
    return pd.DataFrame(rows)


def method_specific_summary(rankings_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        subset = rankings_df.loc[rankings_df["method"] == method].copy()
        if subset.empty:
            continue
        top3 = subset.head(3)
        rows.append(
            {
                "method": method,
                "top_metafeatures": "; ".join(
                    friendly_name(x) for x in top3["metafeature"]
                ),
                "mean_top3_interaction": float(
                    top3["mean_abs_shap_interaction"].mean()
                ),
                "mean_all_interactions": float(
                    subset["mean_abs_shap_interaction"].mean()
                ),
                "shift_interaction_mean": float(
                    subset.loc[
                        subset["metafeature"].isin(SHIFT_FEATURES),
                        "mean_abs_shap_interaction",
                    ].mean()
                ),
                "size_interaction_mean": float(
                    subset.loc[
                        subset["metafeature"].isin(SIZE_FEATURES),
                        "mean_abs_shap_interaction",
                    ].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    method_summary_df: pd.DataFrame,
    dependence_index: pd.DataFrame,
    report_path: Path,
) -> None:
    dataset_label = (
        str(df["dataset"].mode().iloc[0])
        if "dataset" in df.columns and not df["dataset"].dropna().empty
        else "the current dataset"
    )
    lines: list[str] = []
    lines.append("Standard SHAP interaction values analysis")
    lines.append("")
    lines.append("Scope")
    lines.append("- Target: OOD ROC-AUC only")
    lines.append(f"- Dataset block: {dataset_label}")
    lines.append(f"- Meta-samples analyzed: {len(df)}")
    lines.append(
        "- The interaction workflow follows the standard SHAP interaction pattern: a tree-based meta-model was fitted, SHAP interaction values were computed, then method × metafeature interactions were ranked by mean absolute SHAP interaction."
    )
    lines.append("")
    lines.append("Main interaction signals")
    for method in METHOD_ORDER:
        subset = rankings_df.loc[rankings_df["method"] == method].head(3)
        if subset.empty:
            continue
        snippets = [
            f"{friendly_name(mf)} ({val:.6f})"
            for mf, val in zip(
                subset["metafeature"], subset["mean_abs_shap_interaction"]
            )
        ]
        lines.append(f"- {METHOD_LABELS.get(method, method)}: " + "; ".join(snippets))

    lines.append("")
    lines.append("Method-level interpretation")
    m0 = method_summary_df.loc[method_summary_df["method"] == "M0"]
    m1 = method_summary_df.loc[method_summary_df["method"] == "M1"]
    m3 = method_summary_df.loc[method_summary_df["method"] == "M3"]
    m4 = method_summary_df.loc[method_summary_df["method"] == "M4"]
    if not m3.empty and not m0.empty:
        m3_shift = float(m3["shift_interaction_mean"].iloc[0])
        m0_shift = float(m0["shift_interaction_mean"].iloc[0])
        if m3_shift > m0_shift:
            lines.append(
                "- M3 appears more condition-dependent than M0 with respect to shift-related metafeatures, because its average SHAP interaction strength with shift descriptors is larger."
            )
        else:
            lines.append(
                "- M3 does not appear more shift-dependent than M0 in this block; the relative interaction magnitudes are comparable or smaller."
            )
    if not m4.empty and not m0.empty:
        m4_shift = float(m4["shift_interaction_mean"].iloc[0])
        m0_shift = float(m0["shift_interaction_mean"].iloc[0])
        if m4_shift > m0_shift:
            lines.append(
                "- M4 also shows stronger interaction with shift-related descriptors than M0, which suggests greater condition sensitivity rather than uniform benefit."
            )
        else:
            lines.append(
                "- M4 does not clearly exceed M0 in shift-related interaction strength in this block."
            )
    if not m1.empty and not m0.empty:
        m1_all = float(m1["mean_all_interactions"].iloc[0])
        m0_all = float(m0["mean_all_interactions"].iloc[0])
        if m1_all <= m0_all:
            lines.append(
                "- M1 appears at least as stable as M0 in overall interaction magnitude, suggesting less condition dependence."
            )
        else:
            lines.append(
                "- M1 does not appear globally more stable than M0; several metafeatures still modify its predicted effect."
            )

    lines.append("")
    lines.append("Cautions")
    lines.append(
        "- SHAP interaction values are descriptive and model-based. They indicate which metafeatures most strongly modify the contribution associated with each method, not causal treatment effects."
    )
    lines.append(
        "- Dependence plots should be used to inspect direction, threshold behavior, and nonlinearity, but they remain exploratory."
    )
    lines.append(
        "- The interaction analysis reuses the same leakage-safe meta-learning setup as the previous OOD ROC-AUC SHAP analysis."
    )
    lines.append(
        "- Numerical differences in interaction magnitude can be small, so interpretation should focus on relative ranking rather than absolute scale."
    )
    report_path.write_text("\n".join(lines))


def save_table(df: pd.DataFrame, csv_path: Path, xlsx_path: Path | None = None) -> None:
    df.to_csv(csv_path, index=False)
    if xlsx_path is not None:
        df.to_excel(xlsx_path, index=False)


def main() -> None:
    subdirs = make_dirs()
    df = load_clean_df()
    numeric_features, categorical_features = infer_features(df)
    (
        X,
        y,
        preprocessor,
        X_transformed,
        feature_names,
        model,
        shap_values,
        shap_interaction_values,
    ) = fit_meta_model(df, numeric_features, categorical_features)

    wide_df, rankings_df, top_df = compute_method_interactions(
        feature_names, shap_interaction_values
    )
    create_overall_interaction_figure(
        top_df, OUTPUT_ROOT / "shap_interaction_ranking_by_method.png"
    )
    barplot_index = create_method_barplots(top_df, subdirs["barplots"])
    dependence_index = create_dependence_plots(
        feature_names,
        X_transformed,
        shap_values,
        rankings_df,
        subdirs["dependence_plots"],
    )
    method_summary_df = method_specific_summary(rankings_df)

    save_table(
        wide_df,
        OUTPUT_ROOT / "shap_interaction_values_table.csv",
        OUTPUT_ROOT / "shap_interaction_values_table.xlsx",
    )
    save_table(
        rankings_df,
        OUTPUT_ROOT / "shap_interaction_rankings.csv",
        OUTPUT_ROOT / "shap_interaction_rankings.xlsx",
    )
    save_table(
        top_df,
        OUTPUT_ROOT / "shap_interaction_summary.csv",
        OUTPUT_ROOT / "shap_interaction_summary.xlsx",
    )
    save_table(
        method_summary_df,
        OUTPUT_ROOT / "method_interaction_strength_summary.csv",
        OUTPUT_ROOT / "method_interaction_strength_summary.xlsx",
    )
    save_table(
        barplot_index,
        subdirs["barplots"] / "barplot_index.csv",
        subdirs["barplots"] / "barplot_index.xlsx",
    )
    save_table(
        dependence_index,
        subdirs["dependence_plots"] / "dependence_plot_index.csv",
        subdirs["dependence_plots"] / "dependence_plot_index.xlsx",
    )

    report_path = subdirs["reports"] / "shap_interaction_report.txt"
    write_report(
        df, rankings_df, wide_df, method_summary_df, dependence_index, report_path
    )

    summary_payload = {
        "dataset": str(df["dataset"].mode().iloc[0])
        if "dataset" in df.columns and not df["dataset"].dropna().empty
        else None,
        "target_metric": "OOD_ROC_AUC",
        "n_rows": int(len(df)),
        "n_numeric_features": len(numeric_features),
        "n_categorical_features": len(categorical_features),
        "n_ranked_rows": int(len(rankings_df)),
        "n_dependence_plots": int(len(dependence_index)),
        "n_barplots": int(len(barplot_index)),
    }
    (subdirs["reports"] / "summary.json").write_text(
        json.dumps(summary_payload, indent=2)
    )
    shutil.copy2(
        Path(__file__), subdirs["reports"] / "run_shap_interaction_values_analysis.py"
    )

    print("SHAP interaction workflow complete")
    print(f"- interaction table: {OUTPUT_ROOT / 'shap_interaction_values_table.csv'}")
    print(f"- rankings: {OUTPUT_ROOT / 'shap_interaction_rankings.csv'}")
    print(f"- barplots: {subdirs['barplots']}")
    print(f"- dependence plots: {subdirs['dependence_plots']}")
    print(f"- report: {report_path}")


if __name__ == "__main__":
    main()
