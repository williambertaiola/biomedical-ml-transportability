"""Transportability experiment: abide2 combat lodo pipeline."""

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
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

PERFORMANCE_METRICS = ["accuracy", "balanced_accuracy", "f1", "auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "f1": "F1",
    "auc": "AUC",
}
GREEN = "#2e8b57"
RED = "#c0392b"
GREY = "#d9d9d9"


@dataclass
class RunConfig:
    input_path: str
    output_dir: str
    model_name: str = "logistic_regression"
    random_seed: int = 42
    plot_top_features: int = 4


@dataclass
class ABIDE2Schema:
    id_col: str
    domain_col: str
    target_source_col: str
    target_source_code_col: str
    target_col: str
    target_label_col: str
    target_mapping: dict[str, int]
    combat_features: list[str]
    numeric_passthrough: list[str]
    categorical_passthrough: list[str]
    leakage_exclusion_cols: list[str]

    @property
    def model_numeric_features(self) -> list[str]:
        return [*self.combat_features, *self.numeric_passthrough]

    @property
    def model_features(self) -> list[str]:
        return [*self.model_numeric_features, *self.categorical_passthrough]


def bootstrap_neurocombat():
    candidates = [
        Path(_configured_path("legacy_local_py_pkgs")),
        Path(__file__).resolve().parent
        / "combat_transportability_project"
        / "local_py_pkgs",
        Path.cwd() / "combat_transportability_project" / "local_py_pkgs",
    ]
    for candidate in candidates:
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            break
    from neuroCombat import neuroCombat

    return neuroCombat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict LODO ComBat pipeline for ABIDE II core multisite dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to abide2_core_multisite_transportability_dataset.csv.",
    )
    parser.add_argument(
        "--output", required=True, help="Directory where all outputs will be written."
    )
    parser.add_argument(
        "--model",
        default="logistic_regression",
        choices=("logistic_regression", "random_forest"),
        help="Downstream model used for baseline and ComBat evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--plot-top-features",
        type=int,
        default=4,
        help="Number of site-sensitive harmonized features shown in the distribution plot.",
    )
    return parser


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("abide2_combat_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "execution_log.txt")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    last_exception: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception as exc:
            last_exception = exc
    raise RuntimeError(f"Unable to read {path}: {last_exception}")


def get_rg_cmap(reverse: bool = False):
    cmap = plt.get_cmap("RdYlGn_r" if reverse else "RdYlGn").copy()
    cmap.set_bad(color=GREY)
    return cmap


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_abide2_dataset(input_path: Path) -> tuple[pd.DataFrame, ABIDE2Schema]:
    df = read_csv_with_fallback(input_path)
    required_columns = {
        "subject_id",
        "site",
        "disease_level",
        "disease_level_code",
        "age_at_scan",
        "sex_code",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise RuntimeError(f"ABIDE2 input missing required columns: {sorted(missing)}")

    work = df.copy()
    work["target"] = pd.to_numeric(work["disease_level_code"], errors="coerce").map(
        {2: 0, 1: 1}
    )
    work = work.loc[work["target"].isin([0, 1])].copy().reset_index(drop=True)
    work["target"] = work["target"].astype(int)
    work["target_label"] = np.where(work["target"] == 1, "ASD", "Control")
    work["subject_id"] = work["subject_id"].astype(str)
    work["site"] = work["site"].astype(str)

    combat_features = [
        "fiq",
        "viq",
        "piq",
        "srs_total_raw",
        "srs_total_t",
        "srs_awareness_raw",
        "srs_awareness_t",
        "srs_cognition_raw",
        "srs_cognition_t",
        "srs_communication_raw",
        "srs_communication_t",
        "srs_motivation_raw",
        "srs_motivation_t",
        "srs_mannerisms_raw",
        "srs_mannerisms_t",
    ]
    numeric_passthrough = ["age_at_scan"]
    categorical_passthrough = [
        "sex_code",
        "handedness_category_code",
        "fiq_test_type",
        "viq_test_type",
        "piq_test_type",
        "srs_edition",
        "srs_version",
        "srs_informant",
    ]
    leakage_exclusion_cols = [
        "current_med_status_code",
        "eye_status_at_scan_code",
        "disease_level",
        "disease_level_code",
    ]

    all_needed = [*combat_features, *numeric_passthrough, *categorical_passthrough]
    missing_features = [column for column in all_needed if column not in work.columns]
    if missing_features:
        raise RuntimeError(
            f"ABIDE2 input missing expected feature columns: {missing_features}"
        )

    for column in [*combat_features, *numeric_passthrough]:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    schema = ABIDE2Schema(
        id_col="subject_id",
        domain_col="site",
        target_source_col="disease_level",
        target_source_code_col="disease_level_code",
        target_col="target",
        target_label_col="target_label",
        target_mapping={
            "Control": 0,
            "ASD": 1,
            "disease_level_code_2": 0,
            "disease_level_code_1": 1,
        },
        combat_features=combat_features,
        numeric_passthrough=numeric_passthrough,
        categorical_passthrough=categorical_passthrough,
        leakage_exclusion_cols=leakage_exclusion_cols,
    )
    return work, schema


def summarize_dataset(df: pd.DataFrame, schema: ABIDE2Schema) -> dict[str, Any]:
    site_counts = (
        df[schema.domain_col]
        .astype(str)
        .value_counts()
        .sort_index()
        .rename_axis(schema.domain_col)
        .reset_index(name="n_samples")
    )
    class_counts = pd.crosstab(
        df[schema.domain_col].astype(str), df[schema.target_col].astype(int)
    ).reindex(
        columns=[0, 1],
        fill_value=0,
    )
    class_by_site = class_counts.rename_axis(index=schema.domain_col).reset_index()
    target_distribution = (
        df[schema.target_source_col]
        .value_counts()
        .rename_axis(schema.target_source_col)
        .reset_index(name="n_samples")
    )
    single_class_sites = class_counts.index[class_counts.min(axis=1) == 0].tolist()
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "site_counts": site_counts,
        "class_by_site": class_by_site,
        "target_distribution": target_distribution,
        "single_class_holdout_sites": single_class_sites,
        "n_combat_features": int(len(schema.combat_features)),
        "n_numeric_passthrough": int(len(schema.numeric_passthrough)),
        "n_categorical_passthrough": int(len(schema.categorical_passthrough)),
    }


def feature_roles_to_frame(schema: ABIDE2Schema) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for column in schema.combat_features:
        rows.append({"column": column, "role": "combat_numeric"})
    for column in schema.numeric_passthrough:
        rows.append({"column": column, "role": "numeric_passthrough"})
    for column in schema.categorical_passthrough:
        rows.append({"column": column, "role": "categorical_passthrough"})
    for column in schema.leakage_exclusion_cols:
        if column not in {schema.target_col, schema.id_col, schema.domain_col}:
            rows.append({"column": column, "role": "leakage_excluded"})
    rows.extend(
        [
            {"column": schema.id_col, "role": "id"},
            {"column": schema.domain_col, "role": "domain"},
            {"column": schema.target_source_col, "role": "target_source"},
            {"column": schema.target_source_code_col, "role": "target_source_code"},
            {"column": schema.target_col, "role": "target"},
            {"column": schema.target_label_col, "role": "target_label"},
        ]
    )
    return (
        pd.DataFrame(rows)
        .drop_duplicates()
        .sort_values(["role", "column"])
        .reset_index(drop=True)
    )


def build_fill_values(
    train_df: pd.DataFrame, schema: ABIDE2Schema
) -> tuple[dict[str, float], dict[str, str]]:
    numeric_fill: dict[str, float] = {}
    categorical_fill: dict[str, str] = {}
    for column in schema.model_numeric_features:
        numeric = pd.to_numeric(train_df[column], errors="coerce")
        numeric_fill[column] = float(numeric.median()) if numeric.notna().any() else 0.0
    for column in schema.categorical_passthrough:
        series = train_df[column].dropna()
        categorical_fill[column] = (
            str(series.mode().iloc[0]) if not series.empty else "missing"
        )
    return numeric_fill, categorical_fill


def apply_fill_values(
    df: pd.DataFrame,
    schema: ABIDE2Schema,
    numeric_fill: dict[str, float],
    categorical_fill: dict[str, str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for column in schema.model_numeric_features:
        out[column] = (
            pd.to_numeric(df[column], errors="coerce")
            .fillna(numeric_fill[column])
            .astype(float)
        )
    for column in schema.categorical_passthrough:
        filled = (
            df[column]
            .astype("object")
            .where(df[column].notna(), categorical_fill[column])
        )
        out[column] = filled.astype(str)
    return out


def run_neurocombat(
    train_frame: pd.DataFrame,
    batch_train: pd.Series,
    age_train: pd.Series,
    sex_train: pd.Series,
    label_train: pd.Series,
    neurocombat,
) -> pd.DataFrame:
    if train_frame.empty:
        return train_frame.copy()
    covars = pd.DataFrame(
        {
            "batch": batch_train.astype(str).to_numpy(),
            "age_at_scan": pd.to_numeric(age_train, errors="coerce").to_numpy(
                dtype=float
            ),
            "sex_code": sex_train.astype(str).to_numpy(),
            "label": label_train.astype(str).to_numpy(),
        }
    )
    ref_batch = str(batch_train.astype(str).value_counts().idxmax())
    result = neurocombat(
        dat=train_frame.to_numpy(dtype=float).T,
        covars=covars,
        batch_col="batch",
        categorical_cols=["sex_code", "label"],
        continuous_cols=["age_at_scan"],
        ref_batch=ref_batch,
    )
    return pd.DataFrame(
        result["data"].T, index=train_frame.index, columns=train_frame.columns
    )


def build_design_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: ABIDE2Schema,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    transformers: list[tuple[str, Any, list[str]]] = []
    if schema.model_numeric_features:
        transformers.append(("num", StandardScaler(), schema.model_numeric_features))
    if schema.categorical_passthrough:
        transformers.append(("cat", one_hot_encoder(), schema.categorical_passthrough))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    x_train = preprocessor.fit_transform(train_df)
    x_test = preprocessor.transform(test_df)
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"feature_{idx}" for idx in range(x_train.shape[1])]
    return np.asarray(x_train), np.asarray(x_test), feature_names


def build_model(model_name: str, seed: int):
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=seed
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=seed,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": safe_auc(y_true, y_prob),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_lodo(
    df: pd.DataFrame,
    schema: ABIDE2Schema,
    config: RunConfig,
    neurocombat,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    holdout_domains = sorted(df[schema.domain_col].astype(str).unique().tolist())

    for fold_index, holdout_domain in enumerate(holdout_domains, start=1):
        train_df = df[df[schema.domain_col].astype(str) != holdout_domain].reset_index(
            drop=True
        )
        test_df = df[df[schema.domain_col].astype(str) == holdout_domain].reset_index(
            drop=True
        )

        numeric_fill, categorical_fill = build_fill_values(train_df, schema)
        train_filled = apply_fill_values(
            train_df, schema, numeric_fill, categorical_fill
        )
        test_filled = apply_fill_values(test_df, schema, numeric_fill, categorical_fill)

        train_combat = train_filled.copy()
        train_combat.loc[:, schema.combat_features] = run_neurocombat(
            train_frame=train_filled[schema.combat_features],
            batch_train=train_df[schema.domain_col],
            age_train=train_filled["age_at_scan"],
            sex_train=train_filled["sex_code"],
            label_train=train_df[schema.target_col],
            neurocombat=neurocombat,
        )

        y_train = train_df[schema.target_col].to_numpy(dtype=int)
        y_test = test_df[schema.target_col].to_numpy(dtype=int)

        logger.info(
            "Fold %s/%s | holdout=%s | n_train=%s | n_test=%s | combat_features=%s | test_classes=%s",
            fold_index,
            len(holdout_domains),
            holdout_domain,
            len(train_df),
            len(test_df),
            len(schema.combat_features),
            np.unique(y_test).size,
        )

        pipeline_frames = {
            "baseline_raw": (
                train_filled[schema.model_features],
                test_filled[schema.model_features],
            ),
            "combat_train_only": (
                train_combat[schema.model_features],
                test_filled[schema.model_features],
            ),
        }

        for pipeline_name, (train_features, test_features) in pipeline_frames.items():
            x_train, x_test, feature_names = build_design_matrix(
                train_features, test_features, schema
            )
            model = build_model(config.model_name, config.random_seed)
            model.fit(x_train, y_train)
            y_prob = model.predict_proba(x_test)[:, 1]
            metrics = compute_metrics(y_test, y_prob)
            metrics_rows.append(
                {
                    "fold_index": fold_index,
                    "holdout_domain": holdout_domain,
                    "pipeline": pipeline_name,
                    "model": config.model_name,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "n_test_classes": int(np.unique(y_test).size),
                    "n_model_features": int(len(feature_names)),
                    "n_combat_features": int(len(schema.combat_features)),
                    "test_harmonized": False,
                    **metrics,
                }
            )
            prediction_rows.extend(
                {
                    schema.id_col: sample_id,
                    "holdout_domain": holdout_domain,
                    "pipeline": pipeline_name,
                    "y_true": int(y_true),
                    "y_prob": float(prob),
                    "y_pred": int(prob >= 0.5),
                }
                for sample_id, y_true, prob in zip(
                    test_df[schema.id_col], y_test, y_prob, strict=False
                )
            )

    return pd.DataFrame(metrics_rows), pd.DataFrame(prediction_rows)


def summarize_metrics(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        metrics_df.groupby(["pipeline", "model"], dropna=False)[PERFORMANCE_METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part).strip("_") for part in summary.columns.to_flat_index()
    ]

    comparison_rows: list[dict[str, Any]] = []
    for metric in PERFORMANCE_METRICS:
        raw_mean = float(
            metrics_df.loc[metrics_df["pipeline"] == "baseline_raw", metric].mean()
        )
        combat_mean = float(
            metrics_df.loc[metrics_df["pipeline"] == "combat_train_only", metric].mean()
        )
        comparison_rows.append(
            {
                "metric": metric,
                "baseline_raw_mean": raw_mean,
                "combat_train_only_mean": combat_mean,
                "delta_combat_minus_raw": combat_mean - raw_mean,
            }
        )
    return summary, pd.DataFrame(comparison_rows)


def fit_global_harmonization(
    df: pd.DataFrame,
    schema: ABIDE2Schema,
    neurocombat,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric_fill, categorical_fill = build_fill_values(df, schema)
    filled = apply_fill_values(df, schema, numeric_fill, categorical_fill)
    harmonized_matrix = run_neurocombat(
        train_frame=filled[schema.combat_features],
        batch_train=df[schema.domain_col],
        age_train=filled["age_at_scan"],
        sex_train=filled["sex_code"],
        label_train=df[schema.target_col],
        neurocombat=neurocombat,
    )

    harmonized_df = df.copy()
    for column in schema.combat_features:
        harmonized_df[column] = harmonized_matrix[column]
        harmonized_df.loc[df[column].isna(), column] = np.nan

    return harmonized_df, filled[schema.combat_features].copy(), harmonized_matrix


def compute_site_distance_matrix(
    matrix: pd.DataFrame, domains: pd.Series
) -> pd.DataFrame:
    scaled = StandardScaler().fit_transform(matrix)
    scaled_frame = pd.DataFrame(scaled, index=matrix.index, columns=matrix.columns)
    centroids = scaled_frame.groupby(domains.astype(str)).mean()
    labels = centroids.index.tolist()
    distances = np.zeros((len(labels), len(labels)), dtype=float)
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            distances[i, j] = float(
                np.linalg.norm(centroids.loc[label_i] - centroids.loc[label_j])
            )
    return pd.DataFrame(distances, index=labels, columns=labels)


def select_plot_features(
    matrix: pd.DataFrame, domains: pd.Series, top_k: int
) -> list[str]:
    site_means = matrix.groupby(domains.astype(str)).mean()
    between_site_var = site_means.var(axis=0).sort_values(ascending=False)
    return between_site_var.head(min(top_k, len(between_site_var))).index.tolist()


def plot_pca_by_domain(
    raw_matrix: pd.DataFrame,
    harmonized_matrix: pd.DataFrame,
    domains: pd.Series,
    output_path: Path,
) -> None:
    colors = plt.get_cmap("tab20")
    unique_domains = sorted(domains.astype(str).unique().tolist())
    domain_to_color = {
        domain: colors(idx % 20) for idx, domain in enumerate(unique_domains)
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, title, matrix in zip(
        axes,
        ("Raw Features", "Global ComBat Harmonized"),
        (raw_matrix, harmonized_matrix),
        strict=False,
    ):
        pca_input = StandardScaler().fit_transform(matrix)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(pca_input)
        for domain in unique_domains:
            mask = domains.astype(str) == domain
            axis.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=30,
                alpha=0.85,
                color=domain_to_color[domain],
                label=domain,
            )
        axis.set_title(title)
        axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(7, len(unique_domains)),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_site_distributions(
    raw_matrix: pd.DataFrame,
    harmonized_matrix: pd.DataFrame,
    domains: pd.Series,
    features: list[str],
    output_path: Path,
) -> None:
    unique_domains = sorted(domains.astype(str).unique().tolist())
    fig, axes = plt.subplots(
        len(features), 2, figsize=(16, max(4, 3 * len(features))), squeeze=False
    )

    for row_idx, feature in enumerate(features):
        for col_idx, (title, matrix) in enumerate(
            (("Raw", raw_matrix), ("ComBat", harmonized_matrix)), start=0
        ):
            axis = axes[row_idx, col_idx]
            grouped = [
                matrix.loc[domains.astype(str) == domain, feature].to_numpy(dtype=float)
                for domain in unique_domains
            ]
            axis.boxplot(grouped, tick_labels=unique_domains, patch_artist=True)
            axis.set_title(f"{feature} | {title}")
            axis.tick_params(axis="x", rotation=30)
            axis.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distance_heatmaps(
    raw_distances: pd.DataFrame, harmonized_distances: pd.DataFrame, output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for axis, title, matrix in zip(
        axes,
        ("Raw Site Distances", "ComBat Site Distances"),
        (raw_distances, harmonized_distances),
        strict=False,
    ):
        image = axis.imshow(
            matrix.to_numpy(dtype=float), cmap=get_rg_cmap(reverse=True)
        )
        axis.set_title(title)
        axis.set_xticks(range(len(matrix.columns)))
        axis.set_xticklabels(matrix.columns.tolist(), rotation=30, ha="right")
        axis.set_yticks(range(len(matrix.index)))
        axis.set_yticklabels(matrix.index.tolist())
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(
                    j,
                    i,
                    f"{matrix.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_metric_tables(
    metrics_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = (
        metrics_df.loc[
            metrics_df["pipeline"] == "baseline_raw",
            ["holdout_domain", *PERFORMANCE_METRICS],
        ]
        .set_index("holdout_domain")
        .sort_index()
    )
    combat = (
        metrics_df.loc[
            metrics_df["pipeline"] == "combat_train_only",
            ["holdout_domain", *PERFORMANCE_METRICS],
        ]
        .set_index("holdout_domain")
        .sort_index()
    )
    delta = combat - baseline
    return baseline, combat, delta


def annotate_matrix(axis, matrix: pd.DataFrame, signed: bool = False) -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            text = (
                "NA"
                if pd.isna(value)
                else (f"{value:+.3f}" if signed else f"{value:.3f}")
            )
            axis.text(j, i, text, ha="center", va="center", color="black", fontsize=8)


def plot_metric_heatmaps(
    baseline_table: pd.DataFrame,
    combat_table: pd.DataFrame,
    delta_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for axis, title, matrix in zip(
        axes,
        ("Baseline Raw Metrics", "ComBat Train Only Metrics"),
        (baseline_table, combat_table),
        strict=False,
    ):
        image = axis.imshow(
            matrix.to_numpy(dtype=float),
            cmap=get_rg_cmap(),
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_xticks(range(len(matrix.columns)))
        axis.set_xticklabels(
            [METRIC_LABELS.get(column, column) for column in matrix.columns],
            rotation=25,
            ha="right",
        )
        axis.set_yticks(range(len(matrix.index)))
        axis.set_yticklabels(matrix.index.tolist())
        annotate_matrix(axis, matrix)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(
        output_dir / "metric_heatmap_baseline_vs_combat.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    max_abs_delta = (
        float(np.nanmax(np.abs(delta_table.to_numpy(dtype=float))))
        if not delta_table.empty
        else 0.0
    )
    max_abs_delta = max(max_abs_delta, 1e-6)
    fig, axis = plt.subplots(figsize=(9.5, 6))
    image = axis.imshow(
        delta_table.to_numpy(dtype=float),
        cmap=get_rg_cmap(),
        vmin=-max_abs_delta,
        vmax=max_abs_delta,
        aspect="auto",
    )
    axis.set_title("ComBat - Baseline: Delta Metrics by Holdout Site")
    axis.set_xticks(range(len(delta_table.columns)))
    axis.set_xticklabels(
        [METRIC_LABELS.get(column, column) for column in delta_table.columns],
        rotation=25,
        ha="right",
    )
    axis.set_yticks(range(len(delta_table.index)))
    axis.set_yticklabels(delta_table.index.tolist())
    annotate_matrix(axis, delta_table, signed=True)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(
        output_dir / "metric_delta_heatmap_combat_minus_baseline.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_metric_mean_bars(comparison_df: pd.DataFrame, output_path: Path) -> None:
    metrics = comparison_df["metric"].tolist()
    baseline_means = comparison_df["baseline_raw_mean"].to_numpy(dtype=float)
    combat_means = comparison_df["combat_train_only_mean"].to_numpy(dtype=float)
    positions = np.arange(len(metrics))
    width = 0.35

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        positions - width / 2,
        baseline_means,
        width=width,
        color=RED,
        alpha=0.85,
        label="Baseline Raw",
    )
    axis.bar(
        positions + width / 2,
        combat_means,
        width=width,
        color=GREEN,
        alpha=0.85,
        label="ComBat Train Only",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [METRIC_LABELS.get(metric, metric) for metric in metrics],
        rotation=20,
        ha="right",
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Mean Score")
    axis.set_title("Mean Performance: Baseline vs ComBat")
    axis.grid(alpha=0.2, axis="y")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_delta_bars(comparison_df: pd.DataFrame, output_path: Path) -> None:
    metrics = comparison_df["metric"].tolist()
    deltas = comparison_df["delta_combat_minus_raw"].to_numpy(dtype=float)
    positions = np.arange(len(metrics))
    colors = [GREEN if value >= 0 else RED for value in deltas]

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(positions, deltas, color=colors, alpha=0.9)
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [METRIC_LABELS.get(metric, metric) for metric in metrics],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel("Delta ComBat - Baseline")
    axis.set_title("Average Improvement by Metric")
    axis.grid(alpha=0.2, axis="y")
    for idx, value in enumerate(deltas):
        axis.text(
            idx,
            value,
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_by_holdout(metrics_df: pd.DataFrame, output_path: Path) -> None:
    holdouts = sorted(metrics_df["holdout_domain"].astype(str).unique().tolist())
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    axes = axes.flatten()

    for axis, metric in zip(axes, PERFORMANCE_METRICS, strict=False):
        baseline_values: list[float] = []
        combat_values: list[float] = []
        for holdout in holdouts:
            baseline_values.append(
                float(
                    metrics_df.loc[
                        (metrics_df["holdout_domain"] == holdout)
                        & (metrics_df["pipeline"] == "baseline_raw"),
                        metric,
                    ].iloc[0]
                )
            )
            combat_values.append(
                float(
                    metrics_df.loc[
                        (metrics_df["holdout_domain"] == holdout)
                        & (metrics_df["pipeline"] == "combat_train_only"),
                        metric,
                    ].iloc[0]
                )
            )
        x = np.arange(len(holdouts))
        axis.plot(
            x, baseline_values, marker="o", color=RED, linewidth=2, label="Baseline Raw"
        )
        axis.plot(
            x,
            combat_values,
            marker="o",
            color=GREEN,
            linewidth=2,
            label="ComBat Train Only",
        )
        for idx, (base_value, combat_value) in enumerate(
            zip(baseline_values, combat_values, strict=False)
        ):
            if not (np.isnan(base_value) or np.isnan(combat_value)):
                axis.vlines(
                    idx,
                    ymin=min(base_value, combat_value),
                    ymax=max(base_value, combat_value),
                    color="#666666",
                    alpha=0.35,
                )
        axis.set_title(METRIC_LABELS.get(metric, metric))
        axis.set_xticks(x)
        axis.set_xticklabels(holdouts, rotation=30, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.2, axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Per-Holdout Comparison: Baseline vs ComBat", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    if df.empty:
        return "_Empty table_"
    headers = [str(column) for column in df.columns.tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        formatted: list[str] = []
        for value in row:
            if pd.isna(value):
                formatted.append("NA")
            elif isinstance(value, float):
                formatted.append(f"{value:.{float_digits}f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    config: RunConfig,
    schema: ABIDE2Schema,
    dataset_summary: dict[str, Any],
    metrics_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> None:
    leakage_notes = [
        "LODO rigoroso: ogni fold allena su N-1 siti e testa sul sito hold-out.",
        "Imputazione, encoding, scaling e ComBat sono fittati solo sui dati di training fold.",
        "ComBat usa `age_at_scan`, `sex_code` e la label di training come covariate per preservare il segnale clinico e diagnostico.",
        "Il sito hold-out non viene armonizzato, per evitare leakage: i parametri ComBat non vengono mai stimati usando il sito test.",
        "Il file `abide2_combat_harmonized.csv` e i plot globali sono artefatti descrittivi costruiti sull'intero dataset e non vengono usati per stimare le metriche LODO.",
        "Nei siti mono-classe (`ABIDEII-KUL_3`, `ABIDEII-NYU_2`) la ROC AUC non e' definita e viene salvata come `NA`.",
    ]

    report = (
        f"""# ABIDE II ComBat LODO Report

## Setup
- Input: `{config.input_path}`
- Output: `{config.output_dir}`
- Modello downstream: `{config.model_name}`
- Random seed: `{config.random_seed}`

## Colonne e target
- ID campione: `{schema.id_col}`
- Dominio/site: `{schema.domain_col}`
- Target sorgente: `{schema.target_source_col}`
- Target codice sorgente: `{schema.target_source_code_col}`
- Target binario creato: `{schema.target_col}`
- Mapping target: `{json.dumps(schema.target_mapping, ensure_ascii=False)}`
- Covariata continua preservata: `{", ".join(schema.numeric_passthrough)}`
- Covariate categoriche preservate: `sex_code`
- Feature armonizzate con ComBat: `{", ".join(schema.combat_features)}`
- Feature categoriche pass-through: `{", ".join(schema.categorical_passthrough)}`
- Colonne escluse per leakage o ruolo ausiliario: `{", ".join(schema.leakage_exclusion_cols)}`

## Analisi iniziale
- Shape: `{tuple(dataset_summary["shape"])}`
- Numero feature armonizzate: `{dataset_summary["n_combat_features"]}`
- Numero feature numeriche pass-through: `{dataset_summary["n_numeric_passthrough"]}`
- Numero feature categoriche pass-through: `{dataset_summary["n_categorical_passthrough"]}`
- Siti hold-out con una sola classe test: `{", ".join(dataset_summary["single_class_holdout_sites"]) or "nessuno"}`

### Campioni per sito
{markdown_table(dataset_summary["site_counts"])}

### Classi binarie per sito
{markdown_table(dataset_summary["class_by_site"])}

### Distribuzione `disease_level`
{markdown_table(dataset_summary["target_distribution"])}

## Come ho gestito il leakage
"""
        + "\n".join(f"- {note}" for note in leakage_notes)
        + f"""

## Risultati LODO
### Medie e deviazioni standard
{markdown_table(metrics_summary)}

### Confronto raw vs ComBat
{markdown_table(comparison)}

### Metriche per fold
{markdown_table(fold_metrics)}

## Grafici generati
- `plots/pca_by_site_raw_vs_combat.png`: PCA colorata per sito prima e dopo armonizzazione globale.
- `plots/site_distributions_raw_vs_combat.png`: boxplot delle feature piu' site-sensitive prima e dopo ComBat.
- `plots/site_distance_heatmap_raw_vs_combat.png`: heatmap rosso-verde delle distanze tra siti.
- `plots/metric_heatmap_baseline_vs_combat.png`: heatmap rosso-verde di accuracy, balanced accuracy, F1 e AUC per sito hold-out e pipeline.
- `plots/metric_delta_heatmap_combat_minus_baseline.png`: heatmap del delta `ComBat - baseline`.
- `plots/metric_mean_barplot.png`: confronto delle metriche medie raw vs ComBat.
- `plots/metric_delta_barplot.png`: miglioramento medio per metrica.
- `plots/metric_by_holdout_comparison.png`: confronto per sito hold-out su tutte le metriche principali.

## Interpretazione
- In ABIDE II l'eterogeneita' multisito e' reale e il task cross-site e' piu' difficile di TCGA, quindi il confronto raw vs ComBat e' informativo.
- Se ComBat migliora alcuni siti ma non altri, il pattern va interpretato insieme al fatto che il sito hold-out resta raw sotto vincolo no-leakage.
- Le feature SRS e IQ sono dati clinico-comportamentali: l'armonizzazione va letta come correzione del sito, non come sostituto di una vera pipeline di imaging.
"""
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def save_run_manifest(
    output_dir: Path, config: RunConfig, schema: ABIDE2Schema
) -> None:
    manifest = {
        "config": asdict(config),
        "schema": asdict(schema),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_pipeline(config: RunConfig) -> None:
    neurocombat = bootstrap_neurocombat()
    output_dir = ensure_directory(Path(config.output_dir))
    plots_dir = ensure_directory(output_dir / "plots")
    logger = setup_logger(output_dir)

    logger.info("Reading dataset from %s", config.input_path)
    df, schema = load_abide2_dataset(Path(config.input_path))
    dataset_summary = summarize_dataset(df, schema)
    logger.info(
        "Detected ABIDE2 schema | id=%s | domain=%s | target_source=%s | combat_features=%s",
        schema.id_col,
        schema.domain_col,
        schema.target_source_col,
        ",".join(schema.combat_features),
    )

    dataset_summary_payload = {
        "input_path": config.input_path,
        "shape": dataset_summary["shape"],
        "n_combat_features": dataset_summary["n_combat_features"],
        "n_numeric_passthrough": dataset_summary["n_numeric_passthrough"],
        "n_categorical_passthrough": dataset_summary["n_categorical_passthrough"],
        "single_class_holdout_sites": dataset_summary["single_class_holdout_sites"],
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    dataset_summary["site_counts"].to_csv(output_dir / "site_counts.csv", index=False)
    dataset_summary["class_by_site"].to_csv(
        output_dir / "class_distribution_by_site.csv", index=False
    )
    dataset_summary["target_distribution"].to_csv(
        output_dir / "target_distribution.csv", index=False
    )
    feature_roles_to_frame(schema).to_csv(output_dir / "feature_roles.csv", index=False)

    metrics_df, predictions_df = evaluate_lodo(df, schema, config, neurocombat, logger)
    metrics_summary, comparison = summarize_metrics(metrics_df)
    metrics_df.to_csv(output_dir / "metrics_per_fold.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    metrics_summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    comparison.to_csv(output_dir / "raw_vs_combat_comparison.csv", index=False)
    baseline_metric_table, combat_metric_table, delta_metric_table = (
        build_metric_tables(metrics_df)
    )
    baseline_metric_table.to_csv(output_dir / "metric_table_baseline.csv")
    combat_metric_table.to_csv(output_dir / "metric_table_combat.csv")
    delta_metric_table.to_csv(
        output_dir / "metric_table_delta_combat_minus_baseline.csv"
    )

    harmonized_df, raw_matrix, harmonized_matrix = fit_global_harmonization(
        df, schema, neurocombat
    )
    harmonized_df.to_csv(output_dir / "abide2_combat_harmonized.csv", index=False)
    raw_distances = compute_site_distance_matrix(raw_matrix, df[schema.domain_col])
    harmonized_distances = compute_site_distance_matrix(
        harmonized_matrix, df[schema.domain_col]
    )
    raw_distances.to_csv(output_dir / "site_distance_matrix_raw.csv")
    harmonized_distances.to_csv(output_dir / "site_distance_matrix_combat.csv")

    plot_features = select_plot_features(
        raw_matrix, df[schema.domain_col], config.plot_top_features
    )
    pd.DataFrame({"plot_feature": plot_features}).to_csv(
        output_dir / "site_sensitive_plot_features.csv", index=False
    )
    plot_pca_by_domain(
        raw_matrix,
        harmonized_matrix,
        df[schema.domain_col],
        plots_dir / "pca_by_site_raw_vs_combat.png",
    )
    plot_site_distributions(
        raw_matrix,
        harmonized_matrix,
        df[schema.domain_col],
        plot_features,
        plots_dir / "site_distributions_raw_vs_combat.png",
    )
    plot_distance_heatmaps(
        raw_distances,
        harmonized_distances,
        plots_dir / "site_distance_heatmap_raw_vs_combat.png",
    )
    plot_metric_heatmaps(
        baseline_metric_table,
        combat_metric_table,
        delta_metric_table,
        plots_dir,
    )
    plot_metric_mean_bars(
        comparison,
        plots_dir / "metric_mean_barplot.png",
    )
    plot_metric_delta_bars(
        comparison,
        plots_dir / "metric_delta_barplot.png",
    )
    plot_metric_by_holdout(
        metrics_df,
        plots_dir / "metric_by_holdout_comparison.png",
    )

    save_run_manifest(output_dir, config, schema)
    write_report(
        output_dir=output_dir,
        config=config,
        schema=schema,
        dataset_summary=dataset_summary,
        metrics_summary=metrics_summary,
        comparison=comparison,
        fold_metrics=metrics_df,
    )
    shutil.copy2(Path(__file__), output_dir / Path(__file__).name)
    logger.info("Pipeline completed successfully. Outputs written to %s", output_dir)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = RunConfig(
        input_path=args.input,
        output_dir=args.output,
        model_name=args.model,
        random_seed=args.seed,
        plot_top_features=args.plot_top_features,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
