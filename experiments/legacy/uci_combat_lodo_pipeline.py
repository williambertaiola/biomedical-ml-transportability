"""Transportability experiment: uci combat lodo pipeline."""

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
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
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
PIPELINE_LABELS = {
    "baseline_raw": "Baseline Raw",
    "combat_train_only": "ComBat Train Only",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "f1": "F1",
    "auc": "AUC",
}
GREEN = "#2e8b57"
RED = "#c0392b"


TARGET_KEYWORDS = (
    "target",
    "label",
    "class",
    "outcome",
    "diagnosis",
    "diagnostic",
    "status",
    "case",
    "disease",
    "phenotype",
)
DOMAIN_KEYWORDS = (
    "domain",
    "site",
    "hospital",
    "center",
    "centre",
    "source",
    "batch",
    "scanner",
    "cohort",
)
ID_KEYWORDS = (
    "subject",
    "sample",
    "patient",
    "record",
    "study",
    "visit",
    "case",
    "participant",
    "id",
)
AGE_KEYWORDS = ("age",)
SEX_KEYWORDS = ("sex", "gender")


@dataclass
class RunConfig:
    input_path: str
    output_dir: str
    model_name: str = "logistic_regression"
    random_seed: int = 42
    low_cardinality_threshold: int = 10
    plot_top_features: int = 4


@dataclass
class DetectedSchema:
    id_col: str
    target_col: str
    domain_col: str
    target_mapping: dict[str, int]
    positive_label: str
    target_like_exclusions: list[str] = field(default_factory=list)
    covariate_continuous: list[str] = field(default_factory=list)
    covariate_categorical: list[str] = field(default_factory=list)

    @property
    def covariate_cols(self) -> list[str]:
        return [*self.covariate_continuous, *self.covariate_categorical]


@dataclass
class FeatureRoles:
    combat_features: list[str]
    numeric_passthrough: list[str]
    categorical_passthrough: list[str]
    excluded_features: list[str]
    covariate_continuous: list[str]
    covariate_categorical: list[str]

    @property
    def numeric_model_features(self) -> list[str]:
        return [*self.combat_features, *self.numeric_passthrough]

    @property
    def model_features(self) -> list[str]:
        return [*self.numeric_model_features, *self.categorical_passthrough]


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
        description="Strict LODO ComBat pipeline for tabular multi-site datasets."
    )
    parser.add_argument("--input", required=True, help="Path to the input CSV dataset.")
    parser.add_argument(
        "--output", required=True, help="Directory where all outputs will be written."
    )
    parser.add_argument(
        "--model",
        default="logistic_regression",
        choices=("logistic_regression", "random_forest"),
        help="Baseline downstream model used in the LODO evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--low-cardinality-threshold",
        type=int,
        default=10,
        help="Numeric integer-like columns with <= threshold unique values are treated as categorical passthrough features.",
    )
    parser.add_argument(
        "--plot-top-features",
        type=int,
        default=4,
        help="Number of high site-shift features to show in the site-distribution plot.",
    )
    return parser


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("uci_combat_pipeline")
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


def normalize_name(column: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(column).strip().lower()).strip("_")


def keyword_score(name: str, keywords: tuple[str, ...]) -> int:
    score = 0
    for keyword in keywords:
        if name == keyword:
            score += 10
        elif name.startswith(f"{keyword}_") or name.endswith(f"_{keyword}"):
            score += 7
        elif keyword in name:
            score += 4
    return score


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def is_integer_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return False
    return bool(np.allclose(numeric, np.round(numeric), atol=1e-8))


def score_target_column(df: pd.DataFrame, column: str) -> int:
    name = normalize_name(column)
    series = df[column]
    n_unique = int(series.nunique(dropna=True))
    score = keyword_score(name, TARGET_KEYWORDS)
    if n_unique == 2:
        score += 8
    elif 2 < n_unique <= 10:
        score += 3
    if pd.api.types.is_bool_dtype(series):
        score += 4
    return score


def score_domain_column(df: pd.DataFrame, column: str) -> int:
    name = normalize_name(column)
    series = df[column]
    n_unique = int(series.nunique(dropna=True))
    score = keyword_score(name, DOMAIN_KEYWORDS)
    if 2 <= n_unique <= min(100, max(5, len(df) // 2)):
        score += 4
    if not pd.api.types.is_numeric_dtype(series):
        score += 3
    return score


def score_id_column(df: pd.DataFrame, column: str) -> int:
    name = normalize_name(column)
    series = df[column]
    n_unique = int(series.nunique(dropna=True))
    uniqueness_ratio = n_unique / max(len(df), 1)
    score = keyword_score(name, ID_KEYWORDS)
    if uniqueness_ratio >= 0.95:
        score += 8
    if name.endswith("id") or "_id" in name:
        score += 4
    return score


def detect_best_column(df: pd.DataFrame, scorer, kind: str) -> str:
    scored = sorted(
        ((scorer(df, column), column) for column in df.columns),
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )
    if not scored or scored[0][0] <= 0:
        raise RuntimeError(
            f"Unable to identify a plausible {kind} column automatically."
        )
    return scored[0][1]


def detect_covariates(
    df: pd.DataFrame, excluded: set[str]
) -> tuple[list[str], list[str]]:
    continuous: list[str] = []
    categorical: list[str] = []
    for column in df.columns:
        if column in excluded:
            continue
        name = normalize_name(column)
        if (
            keyword_score(name, AGE_KEYWORDS) <= 0
            and keyword_score(name, SEX_KEYWORDS) <= 0
        ):
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if keyword_score(name, SEX_KEYWORDS) > 0:
            categorical.append(column)
        elif pd.api.types.is_numeric_dtype(df[column]) or numeric.notna().mean() > 0.8:
            continuous.append(column)
        else:
            categorical.append(column)
    return continuous, categorical


def build_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, DetectedSchema]:
    target_col = detect_best_column(df, score_target_column, "target")
    domain_col = detect_best_column(df, score_domain_column, "domain")
    id_col = detect_best_column(df, score_id_column, "ID")

    work = df.copy()
    if id_col not in work.columns:
        id_col = "row_id"
        work[id_col] = [f"row_{idx:05d}" for idx in range(1, len(work) + 1)]

    target_series = work[target_col]
    if pd.api.types.is_numeric_dtype(target_series):
        target_numeric = pd.to_numeric(target_series, errors="coerce")
        valid_values = sorted(target_numeric.dropna().unique().tolist())
    else:
        valid_values = sorted(target_series.dropna().astype(str).unique().tolist())
    if len(valid_values) != 2:
        raise RuntimeError(
            f"Target column '{target_col}' must be binary for this pipeline. Found {len(valid_values)} classes."
        )

    target_mapping = {str(value): idx for idx, value in enumerate(valid_values)}
    if pd.api.types.is_numeric_dtype(target_series):
        work[target_col] = (
            pd.to_numeric(target_series, errors="coerce")
            .astype(str)
            .map(target_mapping)
        )
    else:
        work[target_col] = target_series.astype(str).map(target_mapping)
    work[target_col] = work[target_col].astype(int)
    work[domain_col] = work[domain_col].astype(str)
    work[id_col] = work[id_col].astype(str)

    target_like_exclusions = [
        column
        for column in work.columns
        if column != target_col
        and keyword_score(normalize_name(column), TARGET_KEYWORDS) > 0
    ]
    covariate_continuous, covariate_categorical = detect_covariates(
        work,
        excluded={id_col, target_col, domain_col, *target_like_exclusions},
    )

    positive_label = str(valid_values[-1])
    schema = DetectedSchema(
        id_col=id_col,
        target_col=target_col,
        domain_col=domain_col,
        target_mapping=target_mapping,
        positive_label=positive_label,
        target_like_exclusions=target_like_exclusions,
        covariate_continuous=covariate_continuous,
        covariate_categorical=covariate_categorical,
    )
    return work, schema


def infer_feature_roles(
    df: pd.DataFrame, schema: DetectedSchema, low_cardinality_threshold: int
) -> FeatureRoles:
    excluded = {
        schema.id_col,
        schema.target_col,
        schema.domain_col,
        *schema.target_like_exclusions,
    }
    combat_features: list[str] = []
    numeric_passthrough: list[str] = []
    categorical_passthrough: list[str] = []

    for column in df.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        is_numeric = (
            pd.api.types.is_numeric_dtype(df[column]) or numeric.notna().mean() > 0.8
        )
        if column in schema.covariate_continuous:
            numeric_passthrough.append(column)
            continue
        if column in schema.covariate_categorical:
            categorical_passthrough.append(column)
            continue
        if not is_numeric:
            categorical_passthrough.append(column)
            continue
        n_unique = int(numeric.nunique(dropna=True))
        if is_integer_like(df[column]) and n_unique <= low_cardinality_threshold:
            categorical_passthrough.append(column)
        else:
            combat_features.append(column)

    return FeatureRoles(
        combat_features=combat_features,
        numeric_passthrough=numeric_passthrough,
        categorical_passthrough=categorical_passthrough,
        excluded_features=sorted(excluded),
        covariate_continuous=schema.covariate_continuous.copy(),
        covariate_categorical=schema.covariate_categorical.copy(),
    )


def build_fill_values(
    train_df: pd.DataFrame, roles: FeatureRoles
) -> tuple[dict[str, float], dict[str, str]]:
    numeric_fill: dict[str, float] = {}
    categorical_fill: dict[str, str] = {}
    for column in roles.numeric_model_features:
        numeric = pd.to_numeric(train_df[column], errors="coerce")
        fill_value = float(numeric.median()) if numeric.notna().any() else 0.0
        numeric_fill[column] = fill_value
    for column in roles.categorical_passthrough:
        series = train_df[column].dropna()
        fill_value = str(series.mode().iloc[0]) if not series.empty else "missing"
        categorical_fill[column] = fill_value
    return numeric_fill, categorical_fill


def apply_fill_values(
    df: pd.DataFrame,
    roles: FeatureRoles,
    numeric_fill: dict[str, float],
    categorical_fill: dict[str, str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for column in roles.numeric_model_features:
        out[column] = (
            pd.to_numeric(df[column], errors="coerce")
            .fillna(numeric_fill[column])
            .astype(float)
        )
    for column in roles.categorical_passthrough:
        filled = (
            df[column]
            .astype("object")
            .where(df[column].notna(), categorical_fill[column])
        )
        out[column] = filled.astype(str)
    return out


def build_combat_covariates(
    df_filled: pd.DataFrame, roles: FeatureRoles
) -> tuple[pd.DataFrame, list[str], list[str]]:
    covars = pd.DataFrame(index=df_filled.index)
    categorical_cols: list[str] = []
    continuous_cols: list[str] = []
    for column in roles.covariate_continuous:
        if column in df_filled.columns:
            covars[column] = pd.to_numeric(df_filled[column], errors="coerce").astype(
                float
            )
            continuous_cols.append(column)
    for column in roles.covariate_categorical:
        if column in df_filled.columns:
            covars[column] = df_filled[column].astype(str)
            categorical_cols.append(column)
    return covars, categorical_cols, continuous_cols


def run_neurocombat(
    train_frame: pd.DataFrame,
    batch_train: pd.Series,
    covariates: pd.DataFrame,
    categorical_covariates: list[str],
    continuous_covariates: list[str],
    neurocombat,
) -> pd.DataFrame:
    if train_frame.empty:
        return train_frame.copy()
    covars = covariates.copy()
    covars["batch"] = batch_train.astype(str).to_numpy()
    covar_order = ["batch", *covariates.columns.tolist()]
    covars = covars[covar_order]
    ref_batch = str(batch_train.astype(str).value_counts().idxmax())
    result = neurocombat(
        dat=train_frame.to_numpy(dtype=float).T,
        covars=covars,
        batch_col="batch",
        categorical_cols=categorical_covariates or None,
        continuous_cols=continuous_covariates or None,
        ref_batch=ref_batch,
    )
    return pd.DataFrame(
        result["data"].T, index=train_frame.index, columns=train_frame.columns
    )


def build_design_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    roles: FeatureRoles,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    transformers: list[tuple[str, Any, list[str]]] = []
    if roles.numeric_model_features:
        transformers.append(("num", StandardScaler(), roles.numeric_model_features))
    if roles.categorical_passthrough:
        transformers.append(("cat", one_hot_encoder(), roles.categorical_passthrough))
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


def feature_roles_to_frame(roles: FeatureRoles) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for column in roles.combat_features:
        rows.append({"column": column, "role": "combat_numeric"})
    for column in roles.numeric_passthrough:
        rows.append({"column": column, "role": "numeric_passthrough"})
    for column in roles.categorical_passthrough:
        rows.append({"column": column, "role": "categorical_passthrough"})
    for column in roles.excluded_features:
        rows.append({"column": column, "role": "excluded"})
    return pd.DataFrame(rows).sort_values(["role", "column"]).reset_index(drop=True)


def summarize_dataset(df: pd.DataFrame, schema: DetectedSchema) -> dict[str, Any]:
    site_counts = (
        df[schema.domain_col]
        .astype(str)
        .value_counts()
        .sort_index()
        .rename_axis(schema.domain_col)
        .reset_index(name="n_samples")
    )
    class_by_site = (
        pd.crosstab(
            df[schema.domain_col].astype(str), df[schema.target_col].astype(int)
        )
        .rename_axis(index=schema.domain_col)
        .reset_index()
    )
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": df.columns.tolist(),
        "site_counts": site_counts,
        "class_by_site": class_by_site,
    }


def evaluate_lodo(
    df: pd.DataFrame,
    schema: DetectedSchema,
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
        roles = infer_feature_roles(train_df, schema, config.low_cardinality_threshold)
        if not roles.combat_features:
            raise RuntimeError("No continuous numeric features were found for ComBat.")

        numeric_fill, categorical_fill = build_fill_values(train_df, roles)
        train_filled = apply_fill_values(
            train_df, roles, numeric_fill, categorical_fill
        )
        test_filled = apply_fill_values(test_df, roles, numeric_fill, categorical_fill)

        train_combat = train_filled.copy()
        covariates_train, categorical_covars, continuous_covars = (
            build_combat_covariates(train_filled, roles)
        )
        train_combat.loc[:, roles.combat_features] = run_neurocombat(
            train_frame=train_filled[roles.combat_features],
            batch_train=train_df[schema.domain_col],
            covariates=covariates_train,
            categorical_covariates=categorical_covars,
            continuous_covariates=continuous_covars,
            neurocombat=neurocombat,
        )

        y_train = train_df[schema.target_col].to_numpy(dtype=int)
        y_test = test_df[schema.target_col].to_numpy(dtype=int)

        pipeline_frames = {
            "baseline_raw": (
                train_filled[roles.model_features],
                test_filled[roles.model_features],
            ),
            "combat_train_only": (
                train_combat[roles.model_features],
                test_filled[roles.model_features],
            ),
        }

        logger.info(
            "Fold %s/%s | holdout=%s | n_train=%s | n_test=%s | combat_features=%s | covariates=%s",
            fold_index,
            len(holdout_domains),
            holdout_domain,
            len(train_df),
            len(test_df),
            len(roles.combat_features),
            ",".join([*roles.covariate_continuous, *roles.covariate_categorical])
            or "none",
        )

        for pipeline_name, (train_features, test_features) in pipeline_frames.items():
            x_train, x_test, feature_names = build_design_matrix(
                train_features, test_features, roles
            )
            model = build_model(config.model_name, config.random_seed)
            model.fit(x_train, y_train)
            y_prob = model.predict_proba(x_test)[:, 1]
            fold_metrics = compute_metrics(y_test, y_prob)
            metrics_rows.append(
                {
                    "fold_index": fold_index,
                    "holdout_domain": holdout_domain,
                    "pipeline": pipeline_name,
                    "model": config.model_name,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "n_model_features": int(len(feature_names)),
                    "n_combat_features": int(len(roles.combat_features)),
                    "test_harmonized": False,
                    **fold_metrics,
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

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    return metrics_df, predictions_df


def summarize_metrics(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        metrics_df.groupby(["pipeline", "model"], dropna=False)[
            ["auc", "accuracy", "balanced_accuracy", "f1"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part).strip("_") for part in summary.columns.to_flat_index()
    ]

    comparison_rows: list[dict[str, Any]] = []
    metrics = ["auc", "accuracy", "balanced_accuracy", "f1"]
    for metric in metrics:
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
    comparison = pd.DataFrame(comparison_rows)
    return summary, comparison


def fit_global_harmonization(
    df: pd.DataFrame,
    schema: DetectedSchema,
    roles: FeatureRoles,
    neurocombat,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric_fill, categorical_fill = build_fill_values(df, roles)
    filled = apply_fill_values(df, roles, numeric_fill, categorical_fill)
    covariates, categorical_covars, continuous_covars = build_combat_covariates(
        filled, roles
    )
    harmonized_matrix = run_neurocombat(
        train_frame=filled[roles.combat_features],
        batch_train=df[schema.domain_col],
        covariates=covariates,
        categorical_covariates=categorical_covars,
        continuous_covariates=continuous_covars,
        neurocombat=neurocombat,
    )

    harmonized_df = df.copy()
    for column in roles.combat_features:
        harmonized_df[column] = harmonized_matrix[column]
        harmonized_df.loc[df[column].isna(), column] = np.nan

    return harmonized_df, filled[roles.combat_features].copy(), harmonized_matrix


def compute_site_distance_matrix(
    matrix: pd.DataFrame, domains: pd.Series
) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
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
    if site_means.empty:
        return matrix.columns.tolist()[:top_k]
    score = site_means.var(axis=0).sort_values(ascending=False)
    return score.head(min(top_k, len(score))).index.tolist()


def plot_pca_by_domain(
    raw_matrix: pd.DataFrame,
    harmonized_matrix: pd.DataFrame,
    domains: pd.Series,
    output_path: Path,
) -> None:
    colors = plt.get_cmap("tab10")
    unique_domains = sorted(domains.astype(str).unique().tolist())
    domain_to_color = {
        domain: colors(idx % 10) for idx, domain in enumerate(unique_domains)
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False, sharey=False)
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
                s=32,
                alpha=0.8,
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
        ncol=min(4, len(unique_domains)),
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
        len(features), 2, figsize=(14, max(4, 3 * len(features))), squeeze=False
    )

    for row_idx, feature in enumerate(features):
        for col_idx, (title, matrix) in enumerate(
            (("Raw", raw_matrix), ("ComBat", harmonized_matrix)),
            start=0,
        ):
            axis = axes[row_idx, col_idx]
            grouped = [
                matrix.loc[domains.astype(str) == domain, feature].to_numpy(dtype=float)
                for domain in unique_domains
            ]
            axis.boxplot(grouped, tick_labels=unique_domains, patch_artist=True)
            axis.set_title(f"{feature} | {title}")
            axis.tick_params(axis="x", rotation=25)
            axis.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distance_heatmaps(
    raw_distances: pd.DataFrame, harmonized_distances: pd.DataFrame, output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, title, matrix in zip(
        axes,
        ("Raw Site Distances", "ComBat Site Distances"),
        (raw_distances, harmonized_distances),
        strict=False,
    ):
        image = axis.imshow(matrix.to_numpy(dtype=float), cmap="RdYlGn_r")
        axis.set_title(title)
        axis.set_xticks(range(len(matrix.columns)))
        axis.set_xticklabels(matrix.columns.tolist(), rotation=25, ha="right")
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
                    color="white",
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


def plot_metric_heatmaps(
    baseline_table: pd.DataFrame,
    combat_table: pd.DataFrame,
    delta_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, title, matrix in zip(
        axes,
        ("Baseline Raw Metrics", "ComBat Train Only Metrics"),
        (baseline_table, combat_table),
        strict=False,
    ):
        image = axis.imshow(
            matrix.to_numpy(dtype=float),
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_xticks(range(len(matrix.columns)))
        axis.set_xticklabels(
            [METRIC_LABELS.get(col, col) for col in matrix.columns],
            rotation=25,
            ha="right",
        )
        axis.set_yticks(range(len(matrix.index)))
        axis.set_yticklabels(matrix.index.tolist())
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(
                    j,
                    i,
                    f"{matrix.iloc[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
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
    fig, axis = plt.subplots(figsize=(8.5, 5))
    image = axis.imshow(
        delta_table.to_numpy(dtype=float),
        cmap="RdYlGn",
        vmin=-max_abs_delta,
        vmax=max_abs_delta,
        aspect="auto",
    )
    axis.set_title("ComBat - Baseline: Delta Metrics by Holdout Site")
    axis.set_xticks(range(len(delta_table.columns)))
    axis.set_xticklabels(
        [METRIC_LABELS.get(col, col) for col in delta_table.columns],
        rotation=25,
        ha="right",
    )
    axis.set_yticks(range(len(delta_table.index)))
    axis.set_yticklabels(delta_table.index.tolist())
    for i in range(delta_table.shape[0]):
        for j in range(delta_table.shape[1]):
            axis.text(
                j,
                i,
                f"{delta_table.iloc[i, j]:+.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
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
    colors = [GREEN if value >= 0 else RED for value in deltas]
    positions = np.arange(len(metrics))

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
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()

    for axis, metric in zip(axes, PERFORMANCE_METRICS, strict=False):
        baseline_values = []
        combat_values = []
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
            axis.vlines(
                idx,
                ymin=min(base_value, combat_value),
                ymax=max(base_value, combat_value),
                color="#666666",
                alpha=0.35,
            )
        axis.set_title(METRIC_LABELS.get(metric, metric))
        axis.set_xticks(x)
        axis.set_xticklabels(holdouts, rotation=25, ha="right")
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
    schema: DetectedSchema,
    roles: FeatureRoles,
    dataset_summary: dict[str, Any],
    metrics_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> None:
    site_counts = dataset_summary["site_counts"]
    class_by_site = dataset_summary["class_by_site"]
    leakage_notes = [
        "LODO rigoroso: ogni fold allena su N-1 siti e testa sul sito hold-out.",
        "Le mediane/moda usate per l'imputazione sono stimate solo sul training fold.",
        "ComBat viene fittato solo sui campioni di training fold.",
        "Il sito hold-out non viene armonizzato, per evitare leakage: i metodi ComBat standard non possono stimare i parametri di un batch completamente nuovo senza usare quel batch in fit.",
        "Il file `uci_combat_harmonized.csv` e i plot globali sono artefatti descrittivi costruiti sul dataset completo e non sono usati per stimare le metriche LODO.",
    ]

    report = (
        f"""# UCI ComBat LODO Report

## Setup
- Input: `{config.input_path}`
- Output: `{config.output_dir}`
- Modello downstream: `{config.model_name}`
- Random seed: `{config.random_seed}`

## Rilevamento automatico colonne
- ID: `{schema.id_col}`
- Target: `{schema.target_col}`
- Domain/site: `{schema.domain_col}`
- Mapping target: `{json.dumps(schema.target_mapping, ensure_ascii=False)}`
- Covariate continue preservate: `{", ".join(schema.covariate_continuous) or "nessuna"}`
- Covariate categoriche preservate: `{", ".join(schema.covariate_categorical) or "nessuna"}`
- Colonne escluse per evitare leakage: `{", ".join(schema.target_like_exclusions) or "nessuna"}`

## Analisi iniziale
- Shape: `{tuple(dataset_summary["shape"])}`
- Numero colonne: `{len(dataset_summary["columns"])}`
- Feature armonizzate con ComBat: `{", ".join(roles.combat_features) or "nessuna"}`
- Feature numeriche pass-through: `{", ".join(roles.numeric_passthrough) or "nessuna"}`
- Feature categoriche pass-through: `{", ".join(roles.categorical_passthrough) or "nessuna"}`

### Campioni per sito
{markdown_table(site_counts)}

### Distribuzione classi per sito
{markdown_table(class_by_site)}

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
- `plots/metric_heatmap_baseline_vs_combat.png`: heatmap rosso-verde di accuracy, balanced accuracy, F1 e AUC per fold e pipeline.
- `plots/metric_delta_heatmap_combat_minus_baseline.png`: heatmap del delta `ComBat - baseline` per capire dove migliora o peggiora.
- `plots/metric_mean_barplot.png`: confronto delle metriche medie raw vs ComBat.
- `plots/metric_delta_barplot.png`: miglioramento medio per metrica.
- `plots/metric_by_holdout_comparison.png`: confronto per sito hold-out su tutte le metriche principali.

## Interpretazione
- Se `combat_train_only` migliora, significa che ridurre il site-effect nei soli domini sorgente aiuta la generalizzazione verso il sito lasciato fuori.
- Se `combat_train_only` peggiora, il motivo principale da considerare e' il mismatch tra training armonizzato e sito hold-out lasciato raw sotto vincolo no-leakage.
- In questo dataset tabulare misto, ComBat viene applicato solo alle feature continue per evitare trasformazioni poco interpretabili sulle variabili cliniche codificate.
"""
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def save_run_manifest(
    output_dir: Path, config: RunConfig, schema: DetectedSchema, roles: FeatureRoles
) -> None:
    manifest = {
        "config": asdict(config),
        "schema": asdict(schema),
        "feature_roles": {
            "combat_features": roles.combat_features,
            "numeric_passthrough": roles.numeric_passthrough,
            "categorical_passthrough": roles.categorical_passthrough,
            "excluded_features": roles.excluded_features,
        },
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
    raw_df = read_csv_with_fallback(Path(config.input_path))
    df, schema = build_schema(raw_df)
    dataset_summary = summarize_dataset(df, schema)
    roles = infer_feature_roles(df, schema, config.low_cardinality_threshold)
    if not roles.combat_features:
        raise RuntimeError(
            "No continuous numeric features available for ComBat after feature-role detection."
        )

    logger.info(
        "Detected columns | id=%s | target=%s | domain=%s | combat_features=%s",
        schema.id_col,
        schema.target_col,
        schema.domain_col,
        ",".join(roles.combat_features),
    )

    dataset_summary_json = {
        "shape": dataset_summary["shape"],
        "columns": dataset_summary["columns"],
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    dataset_summary["site_counts"].to_csv(output_dir / "site_counts.csv", index=False)
    dataset_summary["class_by_site"].to_csv(
        output_dir / "class_distribution_by_site.csv", index=False
    )
    feature_roles_to_frame(roles).to_csv(output_dir / "feature_roles.csv", index=False)

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
        df, schema, roles, neurocombat
    )
    harmonized_df.to_csv(output_dir / "uci_combat_harmonized.csv", index=False)
    raw_distances = compute_site_distance_matrix(raw_matrix, df[schema.domain_col])
    harmonized_distances = compute_site_distance_matrix(
        harmonized_matrix, df[schema.domain_col]
    )
    raw_distances.to_csv(output_dir / "site_distance_matrix_raw.csv")
    harmonized_distances.to_csv(output_dir / "site_distance_matrix_combat.csv")

    plot_features = select_plot_features(
        raw_matrix, df[schema.domain_col], config.plot_top_features
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

    save_run_manifest(output_dir, config, schema, roles)
    write_report(
        output_dir,
        config,
        schema,
        roles,
        dataset_summary,
        metrics_summary,
        comparison,
        metrics_df,
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
        low_cardinality_threshold=args.low_cardinality_threshold,
        plot_top_features=args.plot_top_features,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
