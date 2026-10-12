"""Post-experiment analysis: run shap by method tcga."""

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
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:
    sns = None

import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_ROOT = Path(os.environ.get("SHAP2_INPUT_ROOT", _configured_path("results_uci")))
OUTPUT_ROOT = Path(os.environ.get("SHAP2_OUTPUT_ROOT", str(INPUT_ROOT / "SHAP2")))
TARGET_METRIC = "OOD_ROC_AUC"
METHOD_ORDER = ["M0", "M1", "M2", "M3", "M4"]
RANDOM_STATE = 42
MIN_ROWS_FOR_BEESWARM = 10
MIN_ROWS_FOR_DEPENDENCE = 8
TOP_N_BAR = 8
TOP_N_HEATMAP = 12


LEAKAGE_COLUMNS = {
    "performance",
    "mean_ood_performance",
    "std_ood_performance",
    "mean_id_performance",
    "id_ood_gap",
    "ece",
    "metric",
    "metric_definition",
    "performance_source",
    "target_site",
    "target_site_display",
    "method",
}

REDUNDANT_COLUMNS = {
    "class_0_ratio",
    "class_1_ratio",
}


MODEL_COLUMNS = {"model"}


DISPLAY_NAME_RULES = [
    ("sample_size", "Target sample size"),
    ("n_total_target", "Target sample size"),
    ("n_source", "Source sample size"),
    ("n_train_1", "Target train size"),
    ("n_test_1", "Target test size"),
    ("n_features", "Number of features"),
    ("class_balance", "Target class balance"),
    ("minority_class_ratio", "Target minority class ratio"),
    ("class_entropy", "Target class entropy"),
    ("normalized_entropy", "Target class entropy"),
    ("missing_percentage", "Missing percentage"),
    ("overall_metadata_shift_score", "Metadata shift"),
    ("metadata_shift_score", "Metadata shift"),
    ("mean_continuous_metadata_shift", "Mean continuous metadata shift"),
    ("mean_continuous_shift", "Mean continuous metadata shift"),
    ("mean_categorical_metadata_shift", "Mean categorical metadata shift"),
    ("mean_categorical_shift", "Mean categorical metadata shift"),
    ("site_distance_composite", "Composite domain distance"),
    ("euclidean_centroid_distance", "Centroid distance"),
    ("mahalanobis_distance", "Mahalanobis distance"),
    ("wasserstein_mean_distance", "Wasserstein distance"),
    ("wasserstein_distance", "Wasserstein distance"),
    ("jensen_shannon_mean_divergence", "Mean JS divergence"),
    ("mean_js_divergence", "Mean JS divergence"),
    ("mmd_rbf", "MMD-RBF distance"),
    ("mean_total_signal", "Mean total signal"),
    ("mean_nonzero_gene_count", "Mean nonzero gene count"),
    ("mean_zero_fraction", "Mean zero fraction"),
    ("low_total_signal_outliers", "Low total signal outliers"),
]


@dataclass
class MethodResult:
    method: str
    n_rows: int
    numeric_features: list[str]
    categorical_features: list[str]
    included_display_features: list[str]
    excluded_columns: list[str]
    shap_importance: pd.DataFrame
    generated_beeswarm: bool
    generated_dependence: bool
    top_metafeature: str | None


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_existing_file(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def auto_find_meta_table(root: Path) -> tuple[Path, str]:
    candidates = [
        root / "SHAP" / "tables" / "meta_learning_table_long.csv",
        root / "SHAP" / "tables" / "meta_learning_table_long.xlsx",
        root / "Pattern_discovery" / "pattern_discovery_master_table.csv",
        root / "Pattern_discovery" / "pattern_discovery_master_table.xlsx",
    ]
    found = find_existing_file(candidates)
    if found is None:
        raise FileNotFoundError(f"No suitable meta-learning table was found in {root}.")
    return found, "meta_learning_table"


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {path}")


def display_name_for_column(column_name: str) -> str:
    normalized = normalize_name(column_name)
    for rule_raw, display in DISPLAY_NAME_RULES:
        if normalize_name(rule_raw) == normalized:
            return display
    if column_name.startswith("model__"):
        return column_name.replace("model__", "Model: ")
    return str(column_name).replace("_", " ").title()


def deduplicate_identical_numeric_columns(
    df: pd.DataFrame, columns: list[str]
) -> tuple[list[str], dict[str, str]]:
    kept: list[str] = []
    duplicates: dict[str, str] = {}
    signatures: dict[tuple, str] = {}
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        signature = tuple(
            "__nan__" if pd.isna(v) else round(float(v), 12) for v in values.tolist()
        )
        previous = signatures.get(signature)
        if previous is not None:
            duplicates[column] = previous
            continue
        signatures[signature] = column
        kept.append(column)
    return kept, duplicates


def infer_method_predictors(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], dict[str, str], list[str]]:
    excluded_reasons: list[str] = []
    numeric_candidates: list[str] = []
    display_map: dict[str, str] = {}

    for column in df.columns:
        if column in LEAKAGE_COLUMNS:
            excluded_reasons.append(f"{column}: leakage_or_identifier")
            continue
        if column in REDUNDANT_COLUMNS:
            excluded_reasons.append(f"{column}: intentionally_removed_for_readability")
            continue
        if column in MODEL_COLUMNS:
            continue
        if column == "dataset":
            if df[column].nunique(dropna=True) <= 1:
                excluded_reasons.append(f"{column}: constant_dataset")
                continue
            excluded_reasons.append(f"{column}: dataset_not_used_for_UCI_per_method")
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() == 0:
            excluded_reasons.append(f"{column}: non_numeric_or_missing")
            continue
        if values.nunique(dropna=True) <= 1:
            excluded_reasons.append(f"{column}: constant")
            continue
        numeric_candidates.append(column)
        display_map[column] = display_name_for_column(column)

    numeric_features, duplicates = deduplicate_identical_numeric_columns(
        df, numeric_candidates
    )
    for dup, original in duplicates.items():
        excluded_reasons.append(f"{dup}: duplicate_of_{original}")

    categorical_features: list[str] = []
    if "model" in df.columns and df["model"].nunique(dropna=True) > 1:
        categorical_features.append("model")
    else:
        excluded_reasons.append("model: constant_or_missing")

    return numeric_features, categorical_features, display_map, excluded_reasons


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
        remainder="drop",
    )


def transformed_feature_display_names(
    feature_names: list[str], display_map: dict[str, str]
) -> list[str]:
    readable: list[str] = []
    for feature in feature_names:
        if feature.startswith("num__"):
            raw = feature.replace("num__", "", 1)
            readable.append(display_map.get(raw, display_name_for_column(raw)))
        elif feature.startswith("cat__model_"):
            model_name = feature.replace("cat__model_", "", 1)
            readable.append(f"Model: {model_name}")
        else:
            readable.append(feature)
    return readable


def metafeature_mask(feature_names: list[str]) -> np.ndarray:
    return np.array([name.startswith("num__") for name in feature_names], dtype=bool)


def fit_method_model(
    method_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
):
    X = method_df[numeric_features + categorical_features].copy()
    y = method_df["performance"].astype(float).to_numpy()

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_transformed = preprocessor.fit_transform(X)
    feature_names = list(preprocessor.get_feature_names_out())
    readable_names = transformed_feature_display_names(
        feature_names, {col: display_name_for_column(col) for col in numeric_features}
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=4,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_transformed, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    return (
        X,
        y,
        preprocessor,
        X_transformed,
        feature_names,
        readable_names,
        model,
        shap_values,
    )


def build_importance_table(
    feature_names: list[str],
    readable_names: list[str],
    shap_values: np.ndarray,
) -> pd.DataFrame:
    importances = np.mean(np.abs(shap_values), axis=0)
    rows = []
    for raw_name, display_name, value in zip(
        feature_names, readable_names, importances
    ):
        predictor_type = (
            "model" if raw_name.startswith("cat__model_") else "metafeature"
        )
        original_column = (
            raw_name.replace("num__", "", 1)
            if raw_name.startswith("num__")
            else raw_name.replace("cat__model_", "model__", 1)
        )
        rows.append(
            {
                "feature_raw": raw_name,
                "feature_display": display_name,
                "original_feature": original_column,
                "predictor_type": predictor_type,
                "mean_abs_shap": float(value),
            }
        )
    importance_df = (
        pd.DataFrame(rows)
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)
    return importance_df


def save_barplot(importance_df: pd.DataFrame, output_path: Path, method: str) -> None:
    plot_df = (
        importance_df.loc[importance_df["predictor_type"] == "metafeature"]
        .head(TOP_N_BAR)
        .iloc[::-1]
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.barh(
        np.arange(len(plot_df)), plot_df["mean_abs_shap"], color="#2a6fbb", alpha=0.9
    )
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["feature_display"], fontsize=10)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP barplot | {method} | OOD ROC-AUC", fontsize=12)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_beeswarm(
    method_df: pd.DataFrame,
    feature_names: list[str],
    readable_names: list[str],
    X_transformed: np.ndarray,
    shap_values: np.ndarray,
    output_path: Path,
    method: str,
) -> bool:
    if len(method_df) < MIN_ROWS_FOR_BEESWARM:
        return False
    mask = metafeature_mask(feature_names)
    if mask.sum() == 0:
        return False
    plt.figure(figsize=(7.2, 5.2))
    shap.summary_plot(
        shap_values[:, mask],
        X_transformed[:, mask],
        feature_names=[name for name, keep in zip(readable_names, mask) if keep],
        show=False,
        max_display=min(TOP_N_BAR, int(mask.sum())),
    )
    ax = plt.gca()
    ax.set_title(f"SHAP | {method} | OOD ROC-AUC", fontsize=12)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def save_dependence_plot(
    method_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    feature_names: list[str],
    readable_names: list[str],
    X_transformed: np.ndarray,
    shap_values: np.ndarray,
    output_path: Path,
    method: str,
) -> tuple[bool, str | None]:
    if len(method_df) < MIN_ROWS_FOR_DEPENDENCE:
        return False, None
    numeric_top = importance_df.loc[
        importance_df["predictor_type"] == "metafeature"
    ].copy()
    for _, row in numeric_top.iterrows():
        raw_name = str(row["feature_raw"])
        if raw_name not in feature_names:
            continue
        idx = feature_names.index(raw_name)
        raw_col = raw_name.replace("num__", "", 1)
        if method_df[raw_col].nunique(dropna=True) < 3:
            continue
        plt.figure(figsize=(7.0, 5.0))
        shap.dependence_plot(
            raw_name,
            shap_values,
            X_transformed,
            feature_names=feature_names,
            display_features=None,
            interaction_index=None,
            show=False,
            dot_size=28,
            alpha=0.85,
        )
        ax = plt.gca()
        ax.set_title(
            f"SHAP dependence | {method} | {row['feature_display']}", fontsize=12
        )
        ax.set_xlabel(row["feature_display"])
        ax.set_ylabel("SHAP value")
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return True, str(row["feature_display"])
    return False, None


def method_output_dir(method: str) -> Path:
    return OUTPUT_ROOT / method


def output_script_name() -> str:
    dataset_name = INPUT_ROOT.name.strip().lower() or "dataset"
    return f"run_shap_by_method_{dataset_name}.py"


def ensure_output_structure() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for method in METHOD_ORDER:
        method_output_dir(method).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "combined_summary").mkdir(parents=True, exist_ok=True)


def save_method_result(result: MethodResult, output_dir: Path) -> None:
    result.shap_importance.to_csv(output_dir / "shap_importance.csv", index=False)

    summary_lines = [
        f"Method: {result.method}",
        f"Rows used: {result.n_rows}",
        f"Numeric predictors: {', '.join(result.numeric_features) if result.numeric_features else 'None'}",
        f"Categorical predictors: {', '.join(result.categorical_features) if result.categorical_features else 'None'}",
        f"Included readable features: {', '.join(result.included_display_features) if result.included_display_features else 'None'}",
        f"Excluded columns: {', '.join(result.excluded_columns) if result.excluded_columns else 'None'}",
        f"Generated beeswarm: {'yes' if result.generated_beeswarm else 'no'}",
        f"Generated dependence plot: {'yes' if result.generated_dependence else 'no'}",
        f"Top metafeature for dependence plot: {result.top_metafeature or 'not generated'}",
    ]
    (output_dir / "method_summary.txt").write_text("\n".join(summary_lines))


def create_combined_summary(
    method_results: list[MethodResult], combined_dir: Path
) -> None:
    metafeature_tables = []
    for result in method_results:
        df = result.shap_importance.loc[
            result.shap_importance["predictor_type"] == "metafeature",
            ["feature_display", "mean_abs_shap"],
        ].copy()
        df = df.rename(columns={"mean_abs_shap": result.method})
        metafeature_tables.append(df)

    if not metafeature_tables:
        return

    merged = metafeature_tables[0]
    for table in metafeature_tables[1:]:
        merged = merged.merge(table, on="feature_display", how="outer")
    for method in METHOD_ORDER:
        if method not in merged.columns:
            merged[method] = np.nan
    merged["strongest_across_methods"] = merged[METHOD_ORDER].max(axis=1)
    merged = merged.sort_values(
        "strongest_across_methods", ascending=False
    ).reset_index(drop=True)
    merged.to_csv(
        combined_dir / "shap_importance_comparison_by_method.csv", index=False
    )
    merged.to_excel(
        combined_dir / "shap_importance_comparison_by_method.xlsx", index=False
    )

    heatmap_df = merged.set_index("feature_display")[METHOD_ORDER].head(TOP_N_HEATMAP)
    plt.figure(figsize=(8.4, 6.4))
    if sns is not None:
        sns.heatmap(
            heatmap_df,
            cmap="YlGnBu",
            annot=True,
            fmt=".3f",
            cbar_kws={"label": "Mean |SHAP|"},
        )
    else:
        plt.imshow(heatmap_df.to_numpy(), aspect="auto", cmap="YlGnBu")
        plt.colorbar(label="Mean |SHAP|")
        plt.xticks(np.arange(len(heatmap_df.columns)), heatmap_df.columns, rotation=0)
        plt.yticks(np.arange(len(heatmap_df.index)), heatmap_df.index)
    plt.title("Metafeature × method SHAP importance | OOD ROC-AUC", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        combined_dir / "shap_importance_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    top_rows = []
    for result in method_results:
        top = (
            result.shap_importance.loc[
                result.shap_importance["predictor_type"] == "metafeature"
            ]
            .head(5)
            .copy()
        )
        top["method"] = result.method
        top_rows.append(top)
    if top_rows:
        pd.concat(top_rows, ignore_index=True).to_csv(
            combined_dir / "top_metafeatures_by_method.csv", index=False
        )


def write_readme(
    output_root: Path,
    files_used: list[dict[str, str]],
    target_name: str,
    global_excluded: list[str],
    method_results: list[MethodResult],
) -> None:
    lines = []
    lines.append("README_SHAP2")
    lines.append("")
    lines.append("Files used:")
    for item in files_used:
        lines.append(f"- {item['role']}: {item['path']}")
    lines.append("")
    lines.append(f"Target predicted: {target_name}")
    lines.append("")
    lines.append("Excluded columns / rules:")
    for item in global_excluded:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Included metafeatures:")
    all_features = sorted(
        {name for result in method_results for name in result.included_display_features}
    )
    for name in all_features:
        lines.append(f"- {name}")
    lines.append("")
    lines.append("Generated plots:")
    lines.append("- For each method: 1 SHAP bar plot with the top metafeatures.")
    lines.append(
        "- For each method: 1 SHAP beeswarm plot only if the number of observations was sufficient."
    )
    lines.append(
        "- For each method: at most 1 SHAP dependence plot for the most important metafeature, only when it had enough variation."
    )
    lines.append("- Combined summary: 1 metafeature × method heatmap.")
    lines.append("")
    lines.append("Method-specific notes:")
    for result in method_results:
        lines.append(
            f"- {result.method}: rows={result.n_rows}, beeswarm={'yes' if result.generated_beeswarm else 'no'}, dependence={'yes' if result.generated_dependence else 'no'}, top_dependence_feature={result.top_metafeature or 'not generated'}."
        )
    lines.append("")
    if sns is None:
        lines.append(
            "Technical note: seaborn was not available in the runtime, so the combined heatmap used a matplotlib fallback."
        )
    (output_root / "README_SHAP2.txt").write_text("\n".join(lines))


def main() -> None:
    ensure_output_structure()

    meta_table_path, meta_role = auto_find_meta_table(INPUT_ROOT)
    meta_df = load_table(meta_table_path)
    metric_df = meta_df.loc[meta_df["metric"] == TARGET_METRIC].copy()
    metric_df = metric_df.loc[metric_df["method"].isin(METHOD_ORDER)].reset_index(
        drop=True
    )

    files_used = [{"role": meta_role, "path": str(meta_table_path)}]
    global_excluded = sorted(LEAKAGE_COLUMNS.union(REDUNDANT_COLUMNS))

    method_results: list[MethodResult] = []
    for method in METHOD_ORDER:
        method_df = (
            metric_df.loc[metric_df["method"] == method].copy().reset_index(drop=True)
        )
        out_dir = method_output_dir(method)

        if method_df.empty:
            (out_dir / "method_summary.txt").write_text(
                f"Method {method} had no OOD ROC-AUC rows."
            )
            continue

        numeric_features, categorical_features, display_map, excluded = (
            infer_method_predictors(method_df)
        )
        if not numeric_features and not categorical_features:
            (out_dir / "method_summary.txt").write_text(
                f"Method {method} had no usable predictors after filtering."
            )
            continue

        (
            X,
            y,
            preprocessor,
            X_transformed,
            feature_names,
            readable_names,
            model,
            shap_values,
        ) = fit_method_model(
            method_df=method_df,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        importance_df = build_importance_table(
            feature_names, readable_names, shap_values
        )
        bar_path = out_dir / "shap_barplot.png"
        save_barplot(importance_df, bar_path, method)

        beeswarm_path = out_dir / "shap_beeswarm.png"
        generated_beeswarm = save_beeswarm(
            method_df=method_df,
            feature_names=feature_names,
            readable_names=readable_names,
            X_transformed=X_transformed,
            shap_values=shap_values,
            output_path=beeswarm_path,
            method=method,
        )

        dependence_path = out_dir / "shap_dependence_top_feature.png"
        generated_dependence, top_dep_feature = save_dependence_plot(
            method_df=method_df,
            importance_df=importance_df,
            feature_names=feature_names,
            readable_names=readable_names,
            X_transformed=X_transformed,
            shap_values=shap_values,
            output_path=dependence_path,
            method=method,
        )

        result = MethodResult(
            method=method,
            n_rows=len(method_df),
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            included_display_features=sorted(
                {display_map[col] for col in numeric_features}
            ),
            excluded_columns=sorted(excluded),
            shap_importance=importance_df,
            generated_beeswarm=generated_beeswarm,
            generated_dependence=generated_dependence,
            top_metafeature=top_dep_feature,
        )
        save_method_result(result, out_dir)
        method_results.append(result)

    combined_dir = OUTPUT_ROOT / "combined_summary"
    create_combined_summary(method_results, combined_dir)
    write_readme(
        output_root=OUTPUT_ROOT,
        files_used=files_used,
        target_name="OOD ROC-AUC",
        global_excluded=global_excluded,
        method_results=method_results,
    )

    (OUTPUT_ROOT / "files_used.json").write_text(json.dumps(files_used, indent=2))
    shutil.copy2(Path(__file__), OUTPUT_ROOT / output_script_name())

    print("SHAP2 analysis complete")
    print(f"- output root: {OUTPUT_ROOT}")
    for result in method_results:
        print(
            f"- {result.method}: rows={result.n_rows}, beeswarm={'yes' if result.generated_beeswarm else 'no'}, dependence={'yes' if result.generated_dependence else 'no'}"
        )


if __name__ == "__main__":
    main()
