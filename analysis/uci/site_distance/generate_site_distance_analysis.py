"""Post-experiment analysis: generate site distance analysis."""

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
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon, pdist
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "codex_site_distance_mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RANDOM_STATE = 42
METHOD_ORDER = ["M0", "M1", "M2", "M3", "M4"]
METHOD_COLORS = {
    "M0": "#4C78A8",
    "M1": "#F58518",
    "M2": "#54A24B",
    "M3": "#E45756",
    "M4": "#72B7B2",
}
METHOD_FULL_LABELS = {
    "M0": "M0: Baseline",
    "M1": "M1: ComBat",
    "M2": "M2: Weighting",
    "M3": "M3: Transductive",
    "M4": "M4: ComBat + Transductive",
}
DISTANCE_COLUMNS = [
    "euclidean_centroid_distance",
    "mahalanobis_distance",
    "wasserstein_mean_distance",
    "jensen_shannon_mean_divergence",
    "mmd_rbf",
]
PERFORMANCE_METRICS = {
    "roc_auc": {
        "ood": "roc_auc_ood",
        "id": "roc_auc_id",
        "gap": "gap_roc_auc",
        "label": "ROC-AUC",
    },
    "f1": {
        "ood": "f1_ood",
        "id": "f1_id",
        "gap": "gap_f1",
        "label": "F1-score",
    },
    "balanced_accuracy": {
        "ood": "balanced_accuracy_ood",
        "id": "balanced_accuracy_id",
        "gap": "gap_balanced_accuracy",
        "label": "Balanced Accuracy",
    },
}
OUTCOME_VARIABLES = [
    "mean_ood_performance",
    "id_ood_gap",
    "std_ood_performance",
    "mean_ece",
]
NON_FEATURE_TOKENS = {
    "target",
    "label",
    "outcome",
    "site",
    "domain",
    "subject",
    "patient",
    "sample_id",
    "bootstrap",
    "split",
    "prediction",
    "pred_",
    "prob",
    "metric",
    "score",
    "fold",
}


@dataclass(frozen=True)
class OutputCandidate:
    output_dir: Path
    manifest_path: Path
    config_snapshot_path: Path
    config_snapshot: dict[str, Any]
    results_summary_path: Path | None
    results_bootstrap_raw_path: Path | None
    is_bootstrap: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate site distance analysis from an existing transportability output folder"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Dataset root containing methods/outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where site-distance outputs will be written",
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


def save_figure(
    fig: plt.Figure,
    output_path: Path,
    *,
    rect: tuple[float, float, float, float] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def add_bottom_legend(
    fig: plt.Figure,
    handles: list[Any],
    labels: list[str],
    *,
    ncol: int,
    y: float = 0.005,
) -> None:
    if not handles or not labels:
        return
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        fontsize=9,
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def resolve_existing_path(path_str: str, *, search_root: Path) -> Path:
    candidate = Path(path_str)
    if candidate.exists():
        return candidate
    filename = candidate.name
    for base in [search_root, search_root.parent]:
        matches = sorted(base.rglob(filename))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not resolve dataset path from snapshot: {path_str}")


def resolve_domain_column_name(df: pd.DataFrame, requested_domain_column: str) -> str:
    if requested_domain_column in df.columns:
        return requested_domain_column
    aliases = [
        requested_domain_column,
        "site",
        "domain_id",
        "domain",
        "center",
        "batch",
    ]
    for alias in aliases:
        if alias in df.columns:
            return alias
    raise KeyError(
        f"Domain column '{requested_domain_column}' not found. "
        f"Available columns include: {', '.join(df.columns[:20])}"
    )


def resolve_target_column_name(df: pd.DataFrame, requested_target_column: str) -> str:
    if requested_target_column in df.columns:
        return requested_target_column

    normalized = requested_target_column.strip().lower()
    alias_candidates = {
        "target": [
            "target",
            "target_code",
            "label",
            "label_code",
            "class",
            "class_code",
            "outcome",
            "outcome_code",
            "diagnosis_code",
            "disease_level_code",
        ]
    }
    for candidate in alias_candidates.get(normalized, []):
        if candidate in df.columns:
            return candidate

    binary_candidates: list[str] = []
    for column in df.columns:
        numeric_values = pd.to_numeric(df[column], errors="coerce")
        if numeric_values.isna().any():
            continue
        unique_values = set(numeric_values.astype(int).unique().tolist())
        if unique_values.issubset({0, 1, 2, -1}) and len(unique_values) > 1:
            binary_candidates.append(column)

    if len(binary_candidates) == 1:
        return binary_candidates[0]

    raise KeyError(
        f"Target column '{requested_target_column}' not found. "
        f"Available columns include: {', '.join(df.columns[:20])}"
    )


def normalize_binary_target_values(
    df: pd.DataFrame, target_column: str
) -> pd.DataFrame:
    numeric_target = pd.to_numeric(df[target_column], errors="raise")
    unique_values = set(numeric_target.dropna().astype(int).unique().tolist())
    if unique_values.issubset({0, 1}):
        normalized = numeric_target.astype(int)
    elif unique_values == {1, 2}:
        normalized = numeric_target.map({1: 1, 2: 0}).astype(int)
    elif unique_values == {-1, 1}:
        normalized = numeric_target.map({-1: 0, 1: 1}).astype(int)
    else:
        raise ValueError(
            f"Target column '{target_column}' is not binary after numeric conversion. "
            f"Observed values: {sorted(unique_values)}"
        )
    normalized_df = df.copy()
    normalized_df[target_column] = normalized
    return normalized_df


def load_dataset(
    snapshot: dict[str, Any], *, search_root: Path
) -> tuple[pd.DataFrame, str, str, str | None, Path]:
    dataset_path = resolve_existing_path(
        str(snapshot["input_path"]), search_root=search_root
    )
    requested_domain_column = str(snapshot["domain_column"])
    requested_target_column = str(snapshot["target_column"])
    requested_id_column = str(snapshot.get("id_column", "")) or None

    df = pd.read_csv(dataset_path, low_memory=False)
    domain_column = resolve_domain_column_name(df, requested_domain_column)
    target_column = resolve_target_column_name(df, requested_target_column)
    df = normalize_binary_target_values(df, target_column)
    id_column = requested_id_column if requested_id_column in df.columns else None
    return df, domain_column, target_column, id_column, dataset_path


def discover_output_candidates(input_root: Path) -> list[OutputCandidate]:
    candidates: list[OutputCandidate] = []
    for config_snapshot_path in sorted(input_root.rglob("config_snapshot.json")):
        output_dir = config_snapshot_path.parent
        manifest_path = output_dir / "tables" / "target_site_split_manifest.csv"
        if not manifest_path.exists():
            continue
        snapshot = json.loads(config_snapshot_path.read_text(encoding="utf-8"))
        results_summary_path = output_dir / "tables" / "target_site_results.csv"
        results_bootstrap_raw_path = (
            output_dir / "tables" / "target_site_results_bootstrap_raw.csv"
        )
        is_bootstrap = (
            "bootstrap" in output_dir.name.lower()
            or (output_dir / "tables" / "bootstrap_iterations.csv").exists()
            or (
                isinstance(snapshot.get("n_bootstrap"), int)
                and int(snapshot["n_bootstrap"]) > 1
            )
        )
        candidates.append(
            OutputCandidate(
                output_dir=output_dir,
                manifest_path=manifest_path,
                config_snapshot_path=config_snapshot_path,
                config_snapshot=snapshot,
                results_summary_path=results_summary_path
                if results_summary_path.exists()
                else None,
                results_bootstrap_raw_path=results_bootstrap_raw_path
                if results_bootstrap_raw_path.exists()
                else None,
                is_bootstrap=is_bootstrap,
            )
        )
    if not candidates:
        raise FileNotFoundError(
            f"No transportability output folders with split manifests found under {input_root}"
        )
    return candidates


def candidate_score(
    candidate: OutputCandidate, *, want_bootstrap: bool
) -> tuple[int, int, int, str]:
    output_name = candidate.output_dir.name.lower()
    method_count = len(candidate.config_snapshot.get("enabled_methods", []))
    score = 0
    if candidate.is_bootstrap == want_bootstrap:
        score += 100
    if "m4" in output_name:
        score += 20
    if "transductive" in output_name:
        score += 10
    if "combat" in output_name:
        score += 5
    if candidate.results_bootstrap_raw_path is not None:
        score += 5
    return (score, method_count, len(output_name), str(candidate.output_dir))


def choose_candidate(
    candidates: list[OutputCandidate], *, want_bootstrap: bool
) -> OutputCandidate | None:
    matching = [
        candidate
        for candidate in candidates
        if candidate.is_bootstrap == want_bootstrap
    ]
    if not matching:
        return None
    return sorted(
        matching,
        key=lambda candidate: candidate_score(candidate, want_bootstrap=want_bootstrap),
        reverse=True,
    )[0]


def site_order_from_candidates(
    *,
    standard_candidate: OutputCandidate | None,
    bootstrap_candidate: OutputCandidate | None,
    dataset_df: pd.DataFrame,
    domain_column: str,
) -> list[str]:
    for candidate in [standard_candidate, bootstrap_candidate]:
        if candidate is None:
            continue
        configured_order = [
            str(site)
            for site in candidate.config_snapshot.get("target_site_order", [])
            if str(site)
        ]
        if configured_order:
            return configured_order
        manifest_df = pd.read_csv(candidate.manifest_path)
        if "evaluation_target_site" in manifest_df.columns:
            return (
                manifest_df["evaluation_target_site"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
    return dataset_df[domain_column].dropna().astype(str).drop_duplicates().tolist()


def select_numeric_feature_columns(
    dataset_df: pd.DataFrame,
    *,
    domain_column: str,
    target_column: str,
    id_column: str | None,
) -> list[str]:
    numeric_columns = dataset_df.select_dtypes(include=[np.number]).columns.tolist()
    excluded_columns = {domain_column, target_column}
    if id_column:
        excluded_columns.add(id_column)

    feature_columns: list[str] = []
    for column in numeric_columns:
        if column in excluded_columns:
            continue
        normalized = column.strip().lower()
        if any(token in normalized for token in NON_FEATURE_TOKENS):
            continue
        feature_columns.append(column)
    if not feature_columns:
        raise ValueError(
            "No usable numeric feature columns were found for the site distance analysis."
        )
    return feature_columns


def pretty_target_site(site: str, display_map: dict[str, str]) -> str:
    if site in display_map:
        return display_map[site]
    return site.replace("_", " ").title().replace(" Va", " VA")


def pretty_model_name(model_name: str, model_display_map: dict[str, str]) -> str:
    return model_display_map.get(model_name, model_name)


def should_apply_pca(*, dataset_name: str, n_features: int) -> bool:
    return ("tcga" in dataset_name.lower()) or n_features > 100


def fit_source_target_preprocessor(
    dataset_df: pd.DataFrame,
    *,
    target_site: str,
    domain_column: str,
    feature_columns: list[str],
    dataset_name: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_mask = dataset_df[domain_column].astype(str) != target_site
    target_mask = dataset_df[domain_column].astype(str) == target_site
    source_features = dataset_df.loc[source_mask, feature_columns].copy()
    target_features = dataset_df.loc[target_mask, feature_columns].copy()

    imputer = SimpleImputer(strategy="median")
    source_imputed = imputer.fit_transform(source_features)
    target_imputed = imputer.transform(target_features)

    scaler = StandardScaler()
    source_scaled = scaler.fit_transform(source_imputed)
    target_scaled = scaler.transform(target_imputed)

    metadata: dict[str, Any] = {
        "n_features_used": len(feature_columns),
        "feature_columns": feature_columns,
        "pca_applied": False,
        "pca_n_components": 0,
        "pca_explained_variance_ratio": 0.0,
    }

    if should_apply_pca(dataset_name=dataset_name, n_features=source_scaled.shape[1]):
        n_components = min(50, source_scaled.shape[0] - 1, source_scaled.shape[1])
        if n_components >= 1 and n_components < source_scaled.shape[1]:
            pca = PCA(
                n_components=n_components,
                svd_solver="randomized",
                random_state=random_state,
            )
            source_scaled = pca.fit_transform(source_scaled)
            target_scaled = pca.transform(target_scaled)
            metadata["pca_applied"] = True
            metadata["pca_n_components"] = int(n_components)
            metadata["pca_explained_variance_ratio"] = float(
                np.sum(pca.explained_variance_ratio_)
            )
            metadata["feature_columns"] = [f"PC{i + 1}" for i in range(n_components)]

    return source_scaled, target_scaled, metadata


def euclidean_centroid_distance(
    source_array: np.ndarray, target_array: np.ndarray
) -> float:
    source_centroid = np.mean(source_array, axis=0)
    target_centroid = np.mean(target_array, axis=0)
    return float(np.linalg.norm(source_centroid - target_centroid))


def mahalanobis_distance(source_array: np.ndarray, target_array: np.ndarray) -> float:
    source_centroid = np.mean(source_array, axis=0)
    target_centroid = np.mean(target_array, axis=0)
    diff = (target_centroid - source_centroid).reshape(-1, 1)

    if source_array.shape[1] == 1:
        variance = (
            float(np.var(source_array[:, 0], ddof=1))
            if source_array.shape[0] > 1
            else 1.0
        )
        variance = variance if np.isfinite(variance) and variance > 0 else 1.0
        return float(abs(diff[0, 0]) / math.sqrt(variance))

    covariance = np.cov(source_array, rowvar=False)
    covariance = np.atleast_2d(covariance)
    diagonal_mean = float(np.mean(np.diag(covariance))) if covariance.size else 1.0
    regularization = (
        1e-6
        if not np.isfinite(diagonal_mean) or diagonal_mean <= 0
        else diagonal_mean * 1e-6
    )
    covariance_reg = covariance + np.eye(covariance.shape[0]) * regularization
    covariance_inv = np.linalg.pinv(covariance_reg)
    squared_distance = np.asarray(diff.T @ covariance_inv @ diff, dtype=float).reshape(
        -1
    )[0]
    return float(math.sqrt(max(float(squared_distance), 0.0)))


def wasserstein_mean_distance(
    source_array: np.ndarray, target_array: np.ndarray
) -> float:
    distances = [
        float(
            wasserstein_distance(
                source_array[:, feature_idx], target_array[:, feature_idx]
            )
        )
        for feature_idx in range(source_array.shape[1])
    ]
    return float(np.mean(distances)) if distances else 0.0


def histogram_edges(
    source_values: np.ndarray, target_values: np.ndarray
) -> np.ndarray | None:
    combined = np.concatenate([source_values, target_values])
    finite_combined = combined[np.isfinite(combined)]
    if finite_combined.size == 0:
        return None
    if np.allclose(finite_combined.min(), finite_combined.max()):
        return None
    try:
        edges = np.histogram_bin_edges(finite_combined, bins="fd")
    except Exception:
        bin_count = int(min(20, max(5, round(np.sqrt(finite_combined.size)))))
        edges = np.histogram_bin_edges(finite_combined, bins=bin_count)
    if len(edges) < 2 or np.allclose(edges[0], edges[-1]):
        return None
    return edges


def jensen_shannon_mean_divergence(
    source_array: np.ndarray, target_array: np.ndarray
) -> float:
    divergences: list[float] = []
    for feature_idx in range(source_array.shape[1]):
        source_values = source_array[:, feature_idx]
        target_values = target_array[:, feature_idx]
        edges = histogram_edges(source_values, target_values)
        if edges is None:
            divergences.append(0.0)
            continue
        source_hist, _ = np.histogram(source_values, bins=edges)
        target_hist, _ = np.histogram(target_values, bins=edges)
        if source_hist.sum() == 0 or target_hist.sum() == 0:
            divergences.append(0.0)
            continue
        source_prob = source_hist / source_hist.sum()
        target_prob = target_hist / target_hist.sum()
        js_distance = float(jensenshannon(source_prob, target_prob, base=2.0))
        divergences.append(js_distance**2)
    return float(np.mean(divergences)) if divergences else 0.0


def median_heuristic_sigma(sample_array: np.ndarray) -> float:
    if sample_array.shape[0] < 2:
        return 1.0
    pairwise_distances = pdist(sample_array, metric="euclidean")
    positive_distances = pairwise_distances[pairwise_distances > 0]
    if positive_distances.size == 0:
        return 1.0
    sigma = float(np.median(positive_distances))
    return sigma if np.isfinite(sigma) and sigma > 0 else 1.0


def subsample_array(
    values: np.ndarray, *, max_samples: int, random_state: int
) -> np.ndarray:
    if values.shape[0] <= max_samples:
        return values
    rng = np.random.default_rng(random_state)
    indices = rng.choice(values.shape[0], size=max_samples, replace=False)
    return values[indices]


def mmd_rbf(
    source_array: np.ndarray, target_array: np.ndarray, *, random_state: int
) -> float:
    source_sub = subsample_array(
        source_array, max_samples=300, random_state=random_state
    )
    target_sub = subsample_array(
        target_array, max_samples=300, random_state=random_state + 1
    )
    combined = np.vstack([source_sub, target_sub])
    sigma = median_heuristic_sigma(combined)
    gamma = 1.0 / (2.0 * sigma * sigma)
    kernel_xx = rbf_kernel(source_sub, source_sub, gamma=gamma)
    kernel_yy = rbf_kernel(target_sub, target_sub, gamma=gamma)
    kernel_xy = rbf_kernel(source_sub, target_sub, gamma=gamma)
    mmd_value = float(kernel_xx.mean() + kernel_yy.mean() - 2.0 * kernel_xy.mean())
    return max(mmd_value, 0.0)


def compute_site_distance_row(
    dataset_df: pd.DataFrame,
    *,
    dataset_name: str,
    target_site: str,
    domain_column: str,
    feature_columns: list[str],
    random_state: int,
) -> dict[str, Any]:
    target_mask = dataset_df[domain_column].astype(str) == target_site
    source_mask = ~target_mask
    source_array, target_array, metadata = fit_source_target_preprocessor(
        dataset_df,
        target_site=target_site,
        domain_column=domain_column,
        feature_columns=feature_columns,
        dataset_name=dataset_name,
        random_state=random_state,
    )

    return {
        "dataset": dataset_name,
        "target_site": target_site,
        "n_source": int(source_mask.sum()),
        "n_target": int(target_mask.sum()),
        "n_features_used": int(metadata["n_features_used"]),
        "pca_applied": bool(metadata["pca_applied"]),
        "pca_n_components": int(metadata["pca_n_components"]),
        "pca_explained_variance_ratio": float(metadata["pca_explained_variance_ratio"]),
        "euclidean_centroid_distance": euclidean_centroid_distance(
            source_array, target_array
        ),
        "mahalanobis_distance": mahalanobis_distance(source_array, target_array),
        "wasserstein_mean_distance": wasserstein_mean_distance(
            source_array, target_array
        ),
        "jensen_shannon_mean_divergence": jensen_shannon_mean_divergence(
            source_array, target_array
        ),
        "mmd_rbf": mmd_rbf(source_array, target_array, random_state=random_state),
        "feature_columns_used": "|".join(metadata["feature_columns"]),
    }


def aggregate_performance_results(
    *,
    standard_candidate: OutputCandidate | None,
    bootstrap_candidate: OutputCandidate | None,
) -> pd.DataFrame:
    if (
        bootstrap_candidate is not None
        and bootstrap_candidate.results_bootstrap_raw_path is not None
    ):
        raw_df = pd.read_csv(bootstrap_candidate.results_bootstrap_raw_path)
        group_columns = [
            "method",
            "model",
            "model_display",
            "target_site",
            "target_site_display",
        ]
        rows: list[dict[str, Any]] = []
        for group_keys, group_df in raw_df.groupby(group_columns, dropna=False):
            group_record = dict(zip(group_columns, group_keys, strict=False))
            group_record["n_bootstrap"] = (
                int(group_df["bootstrap_index"].nunique())
                if "bootstrap_index" in group_df.columns
                else np.nan
            )
            for metric_columns in PERFORMANCE_METRICS.values():
                group_record[metric_columns["ood"]] = float(
                    group_df[metric_columns["ood"]].mean()
                )
                group_record[f"{metric_columns['ood']}_std"] = float(
                    group_df[metric_columns["ood"]].std(ddof=0)
                )
                group_record[metric_columns["id"]] = float(
                    group_df[metric_columns["id"]].mean()
                )
                if metric_columns["gap"] in group_df.columns:
                    group_record[metric_columns["gap"]] = float(
                        group_df[metric_columns["gap"]].mean()
                    )
                else:
                    group_record[metric_columns["gap"]] = float(
                        group_record[metric_columns["id"]]
                        - group_record[metric_columns["ood"]]
                    )
            if "ece_ood" in group_df.columns:
                group_record["ece_ood"] = float(group_df["ece_ood"].mean())
                group_record["ece_ood_std"] = float(group_df["ece_ood"].std(ddof=0))
            rows.append(group_record)
        return pd.DataFrame(rows)

    if (
        bootstrap_candidate is not None
        and bootstrap_candidate.results_summary_path is not None
    ):
        summary_df = pd.read_csv(bootstrap_candidate.results_summary_path).copy()
        if "ece_ood_std" not in summary_df.columns:
            summary_df["ece_ood_std"] = np.nan
        if "n_bootstrap" not in summary_df.columns:
            summary_df["n_bootstrap"] = np.nan
        for metric_columns in PERFORMANCE_METRICS.values():
            if f"{metric_columns['ood']}_std" not in summary_df.columns:
                summary_df[f"{metric_columns['ood']}_std"] = np.nan
            if (
                metric_columns["gap"] not in summary_df.columns
                and metric_columns["id"] in summary_df.columns
            ):
                summary_df[metric_columns["gap"]] = (
                    summary_df[metric_columns["id"]] - summary_df[metric_columns["ood"]]
                )
        return summary_df

    if standard_candidate is None or standard_candidate.results_summary_path is None:
        raise FileNotFoundError(
            "Could not find results tables for the site distance analysis."
        )

    summary_df = pd.read_csv(standard_candidate.results_summary_path).copy()
    for metric_columns in PERFORMANCE_METRICS.values():
        summary_df[f"{metric_columns['ood']}_std"] = np.nan
        if (
            metric_columns["gap"] not in summary_df.columns
            and metric_columns["id"] in summary_df.columns
        ):
            summary_df[metric_columns["gap"]] = (
                summary_df[metric_columns["id"]] - summary_df[metric_columns["ood"]]
            )
    if "ece_ood_std" not in summary_df.columns:
        summary_df["ece_ood_std"] = np.nan
    summary_df["n_bootstrap"] = np.nan
    return summary_df


def build_site_distance_vs_performance(
    *,
    performance_summary_df: pd.DataFrame,
    site_distance_df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    merge_columns = [
        "target_site",
        "n_source",
        "n_target",
        "n_features_used",
        "pca_applied",
        "pca_n_components",
        "pca_explained_variance_ratio",
        *DISTANCE_COLUMNS,
    ]
    merged = performance_summary_df.merge(
        site_distance_df[merge_columns], on="target_site", how="left"
    )

    rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        for metric_name, metric_columns in PERFORMANCE_METRICS.items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "target_site": row["target_site"],
                    "target_site_display": row.get(
                        "target_site_display", row["target_site"]
                    ),
                    "model": row["model"],
                    "model_display": row.get("model_display", row["model"]),
                    "metric": metric_name,
                    "method": row["method"],
                    "n_source": row.get("n_source"),
                    "n_target": row.get("n_target"),
                    "n_features_used": row.get("n_features_used"),
                    "pca_applied": row.get("pca_applied"),
                    "pca_n_components": row.get("pca_n_components"),
                    "pca_explained_variance_ratio": row.get(
                        "pca_explained_variance_ratio"
                    ),
                    "euclidean_centroid_distance": row.get(
                        "euclidean_centroid_distance"
                    ),
                    "mahalanobis_distance": row.get("mahalanobis_distance"),
                    "wasserstein_mean_distance": row.get("wasserstein_mean_distance"),
                    "jensen_shannon_mean_divergence": row.get(
                        "jensen_shannon_mean_divergence"
                    ),
                    "mmd_rbf": row.get("mmd_rbf"),
                    "mean_ood_performance": row.get(metric_columns["ood"]),
                    "std_ood_performance": row.get(f"{metric_columns['ood']}_std"),
                    "mean_id_performance": row.get(metric_columns["id"]),
                    "id_ood_gap": row.get(metric_columns["gap"]),
                    "mean_ece": row.get("ece_ood"),
                }
            )
    return pd.DataFrame(rows)


def correlation_stats(
    x: pd.Series, y: pd.Series
) -> tuple[float, float, float, float, int]:
    paired = pd.concat([x, y], axis=1).dropna()
    if paired.empty:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0)
    x_values = paired.iloc[:, 0].astype(float)
    y_values = paired.iloc[:, 1].astype(float)
    n_sites = int(len(paired))
    if n_sites < 2 or x_values.nunique() < 2 or y_values.nunique() < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"), n_sites)
    pearson_r_value, pearson_p_value = pearsonr(x_values, y_values)
    spearman_r_value, spearman_p_value = spearmanr(x_values, y_values)
    return (
        float(pearson_r_value),
        float(pearson_p_value),
        float(spearman_r_value),
        float(spearman_p_value),
        n_sites,
    )


def build_site_distance_correlations(
    site_distance_vs_performance_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_keys, group_df in site_distance_vs_performance_df.groupby(
        ["dataset", "model", "metric", "method"], dropna=False
    ):
        dataset_name, model_name, metric_name, method_name = group_keys
        for distance_metric in DISTANCE_COLUMNS:
            for outcome_variable in OUTCOME_VARIABLES:
                (
                    pearson_r_value,
                    pearson_p_value,
                    spearman_r_value,
                    spearman_p_value,
                    n_sites,
                ) = correlation_stats(
                    group_df[distance_metric],
                    group_df[outcome_variable],
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "metric": metric_name,
                        "method": method_name,
                        "distance_metric": distance_metric,
                        "outcome_variable": outcome_variable,
                        "pearson_r": pearson_r_value,
                        "pearson_p": pearson_p_value,
                        "spearman_rho": spearman_r_value,
                        "spearman_p": spearman_p_value,
                        "n_sites": n_sites,
                    }
                )
    return pd.DataFrame(rows)


def write_single_sheet_excel(
    output_path: Path, sheet_name: str, dataframe: pd.DataFrame
) -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        bundled_site_packages = Path(_configured_path("optional_site_packages"))
        if (
            bundled_site_packages.exists()
            and str(bundled_site_packages) not in sys.path
        ):
            sys.path.insert(0, str(bundled_site_packages))
        import openpyxl  # noqa: F401

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        dataframe.to_excel(writer, sheet_name=sheet_name, index=False)


def correlation_annotation(x: pd.Series, y: pd.Series) -> str:
    pearson_r_value, pearson_p_value, spearman_r_value, spearman_p_value, n_sites = (
        correlation_stats(x, y)
    )
    if n_sites == 0:
        return "No valid pairs"
    return (
        f"n={n_sites}\n"
        f"Pearson r={pearson_r_value:.3f} (p={pearson_p_value:.3f})\n"
        f"Spearman rho={spearman_r_value:.3f} (p={spearman_p_value:.3f})"
    )


def regression_line(ax: plt.Axes, x: pd.Series, y: pd.Series) -> None:
    paired = pd.concat([x, y], axis=1).dropna()
    if len(paired) < 2 or paired.iloc[:, 0].nunique() < 2:
        return
    x_values = paired.iloc[:, 0].to_numpy(dtype=float)
    y_values = paired.iloc[:, 1].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x_values, y_values, deg=1)
    x_line = np.linspace(x_values.min(), x_values.max(), 100)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, color="#222222", linestyle="--", linewidth=1.2, alpha=0.9)


def annotate_site_points(
    ax: plt.Axes, subset: pd.DataFrame, *, x_column: str, y_column: str
) -> None:
    for _, row in subset.iterrows():
        ax.annotate(
            str(row["target_site_display"]),
            (float(row[x_column]), float(row[y_column])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color="#333333",
            alpha=0.85,
        )


def plot_distance_scatter(
    *,
    site_distance_vs_performance_df: pd.DataFrame,
    output_path: Path,
    distance_metric: str,
    model_name: str,
    metric_name: str,
    y_column: str,
    y_label: str,
    title_prefix: str,
    display_map: dict[str, str],
    model_display_map: dict[str, str],
) -> None:
    subset = site_distance_vs_performance_df.loc[
        (site_distance_vs_performance_df["model"] == model_name)
        & (site_distance_vs_performance_df["metric"] == metric_name)
        & (site_distance_vs_performance_df[distance_metric].notna())
        & (site_distance_vs_performance_df[y_column].notna())
    ].copy()
    if subset.empty:
        return

    subset["target_site_display"] = subset["target_site"].map(
        lambda site: pretty_target_site(str(site), display_map)
    )
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    for method_name in METHOD_ORDER:
        method_subset = subset.loc[subset["method"] == method_name].copy()
        if method_subset.empty:
            continue
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=METHOD_COLORS.get(method_name, "#777777"),
                markeredgecolor="#303030",
                markersize=8,
            )
        )
        legend_labels.append(METHOD_FULL_LABELS.get(method_name, method_name))
        ax.scatter(
            method_subset[distance_metric],
            method_subset[y_column],
            color=METHOD_COLORS.get(method_name, "#777777"),
            edgecolor="#303030",
            linewidth=0.7,
            alpha=0.9,
            s=70.0 + method_subset["n_target"].fillna(0).astype(float) * 1.1,
        )
        annotate_site_points(
            ax, method_subset, x_column=distance_metric, y_column=y_column
        )

    regression_line(ax, subset[distance_metric], subset[y_column])

    metric_label = PERFORMANCE_METRICS[metric_name]["label"]
    ax.set_title(
        f"{title_prefix} | {pretty_model_name(model_name, model_display_map)} | {metric_label}"
    )
    ax.set_xlabel(distance_metric.replace("_", " ").title())
    ax.set_ylabel(y_label)
    ax.grid(False)
    text = correlation_annotation(subset[distance_metric], subset[y_column])
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#999999",
            "alpha": 0.85,
        },
    )

    add_bottom_legend(
        fig, legend_handles, legend_labels, ncol=max(1, len(legend_labels))
    )
    save_figure(fig, output_path, rect=(0, 0.1, 1, 1))


def plot_site_distance_barplot(
    *,
    site_distance_df: pd.DataFrame,
    distance_metric: str,
    output_path: Path,
    display_map: dict[str, str],
) -> None:
    plot_df = site_distance_df.sort_values(distance_metric, ascending=True).reset_index(
        drop=True
    )
    labels = [
        pretty_target_site(str(site), display_map) for site in plot_df["target_site"]
    ]
    values = plot_df[distance_metric].to_numpy(dtype=float)
    colors = plt.get_cmap("RdYlGn_r")(np.linspace(0.15, 0.85, len(plot_df)))

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(max(8.5, len(plot_df) * 2.2), 5.8))
    bars = ax.bar(
        np.arange(len(plot_df)),
        values,
        color=colors,
        edgecolor="#404040",
        linewidth=0.7,
    )
    ax.set_title(
        f"Site distance by target site | {distance_metric.replace('_', ' ').title()}"
    )
    ax.set_xlabel("Target site")
    ax.set_ylabel(distance_metric.replace("_", " ").title())
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.grid(False)

    y_max = float(np.nanmax(values)) if len(values) else 1.0
    ax.set_ylim(0.0, y_max * 1.15 if np.isfinite(y_max) and y_max > 0 else 1.0)
    for bar, value in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(0.01, y_max * 0.02),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_figure(fig, output_path)


def plot_site_distance_correlation_heatmap(
    *,
    site_distance_vs_performance_df: pd.DataFrame,
    output_path: Path,
) -> None:
    correlation_columns = DISTANCE_COLUMNS + [
        "mean_ood_performance",
        "std_ood_performance",
        "id_ood_gap",
        "mean_ece",
    ]
    heatmap_df = site_distance_vs_performance_df[correlation_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    correlation_matrix = heatmap_df.corr(method="pearson")

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    image = ax.imshow(
        correlation_matrix.to_numpy(dtype=float),
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
    )
    labels = [column.replace("_", "\n") for column in correlation_matrix.columns]
    ax.set_title("Site distance correlation heatmap")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.grid(False)

    values = correlation_matrix.to_numpy(dtype=float)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            label = "NA" if pd.isna(value) else f"{value:.2f}"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="#111111",
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.set_ylabel("Pearson correlation", rotation=90)
    save_figure(fig, output_path)


def generate_figures(
    *,
    site_distance_df: pd.DataFrame,
    site_distance_vs_performance_df: pd.DataFrame,
    output_dir: Path,
    display_map: dict[str, str],
    model_display_map: dict[str, str],
) -> None:
    figures_dir = output_dir / "figures"

    for distance_metric in DISTANCE_COLUMNS:
        plot_site_distance_barplot(
            site_distance_df=site_distance_df,
            distance_metric=distance_metric,
            output_path=figures_dir / f"site_distance_by_target_{distance_metric}.png",
            display_map=display_map,
        )

    for model_name in (
        site_distance_vs_performance_df["model"].dropna().astype(str).unique()
    ):
        for metric_name in PERFORMANCE_METRICS:
            for distance_metric in DISTANCE_COLUMNS:
                plot_distance_scatter(
                    site_distance_vs_performance_df=site_distance_vs_performance_df,
                    output_path=figures_dir
                    / f"distance_vs_ood_{distance_metric}_{model_name.lower()}_{metric_name}.png",
                    distance_metric=distance_metric,
                    model_name=model_name,
                    metric_name=metric_name,
                    y_column="mean_ood_performance",
                    y_label=f"Mean OOD {PERFORMANCE_METRICS[metric_name]['label']}",
                    title_prefix="Site distance vs OOD performance",
                    display_map=display_map,
                    model_display_map=model_display_map,
                )
                plot_distance_scatter(
                    site_distance_vs_performance_df=site_distance_vs_performance_df,
                    output_path=figures_dir
                    / f"distance_vs_id_ood_gap_{distance_metric}_{model_name.lower()}_{metric_name}.png",
                    distance_metric=distance_metric,
                    model_name=model_name,
                    metric_name=metric_name,
                    y_column="id_ood_gap",
                    y_label=f"ID-OOD gap ({PERFORMANCE_METRICS[metric_name]['label']})",
                    title_prefix="Site distance vs ID-OOD gap",
                    display_map=display_map,
                    model_display_map=model_display_map,
                )
                plot_distance_scatter(
                    site_distance_vs_performance_df=site_distance_vs_performance_df,
                    output_path=figures_dir
                    / f"distance_vs_ood_std_{distance_metric}_{model_name.lower()}_{metric_name}.png",
                    distance_metric=distance_metric,
                    model_name=model_name,
                    metric_name=metric_name,
                    y_column="std_ood_performance",
                    y_label=f"STD OOD {PERFORMANCE_METRICS[metric_name]['label']}",
                    title_prefix="Site distance vs OOD variability",
                    display_map=display_map,
                    model_display_map=model_display_map,
                )

    plot_site_distance_correlation_heatmap(
        site_distance_vs_performance_df=site_distance_vs_performance_df,
        output_path=figures_dir / "site_distance_correlation_heatmap.png",
    )


def write_readme(
    output_dir: Path,
    *,
    dataset_name: str,
    feature_columns: list[str],
    site_distance_df: pd.DataFrame,
) -> None:
    pca_used = bool(site_distance_df["pca_applied"].fillna(False).astype(bool).any())
    pca_line = (
        "PCA was not applied."
        if not pca_used
        else (
            "PCA was applied for at least one target-site comparison. "
            "See the table columns `pca_applied`, `pca_n_components` and `pca_explained_variance_ratio`."
        )
    )
    if len(feature_columns) <= 25:
        feature_text = ", ".join(feature_columns)
    else:
        feature_text = (
            ", ".join(feature_columns[:25])
            + f", ... ({len(feature_columns)} total features)"
        )

    readme_text = f"""Site distance analysis outputs for {dataset_name}

This folder contains a dedicated site-distance analysis built from the existing cross-site pipeline outputs.

Distance metrics:
- Euclidean centroid distance: distance between the standardized source centroid and target centroid.
- Mahalanobis distance: centroid distance using source covariance with regularized pseudo-inverse.
- Wasserstein mean distance: average univariate Wasserstein distance across processed features.
- Jensen-Shannon mean divergence: average feature-wise Jensen-Shannon divergence after histogram binning.
- MMD RBF: Maximum Mean Discrepancy with an RBF kernel and median-heuristic bandwidth.

Preprocessing:
- only numeric feature columns were used;
- non-feature columns such as target, site/domain and subject id were excluded;
- missing values were imputed with source-domain medians only;
- standardization was fit on the source domains only and then applied to source and target.
- {pca_line}

Feature columns used:
- {feature_text}

Files:
- site_distance_by_target.csv / .xlsx
  One row per target site with sample sizes and the full-target source-vs-target distance metrics.

- site_distance_vs_performance.csv / .xlsx
  Site distance metrics joined with the existing cross-site OOD results for every method/model/metric.

- site_distance_correlations.csv / .xlsx
  Pearson and Spearman correlations between each distance metric and each outcome:
  mean OOD performance, ID-OOD gap, OOD variability and ECE.

- figures/
  Dedicated scatterplots, barplots and a correlation heatmap.

How to read the figures:
- larger site distance means the target site is more different from the pooled source domains.
- if higher distance aligns with lower OOD performance, that suggests a transportability penalty under domain shift.
- if higher distance aligns with larger ID-OOD gap or larger OOD variability, that suggests less robust transfer.

No original pipeline files or existing figures were modified.
"""
    (output_dir / "README.txt").write_text(readme_text, encoding="utf-8")


def write_metadata(
    output_dir: Path,
    *,
    input_root: Path,
    standard_candidate: OutputCandidate | None,
    bootstrap_candidate: OutputCandidate | None,
    dataset_path: Path,
    domain_column: str,
    target_column: str,
    feature_columns: list[str],
) -> None:
    metadata = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "standard_manifest": str(standard_candidate.manifest_path)
        if standard_candidate is not None
        else "",
        "bootstrap_manifest": str(bootstrap_candidate.manifest_path)
        if bootstrap_candidate is not None
        else "",
        "standard_results": str(standard_candidate.results_summary_path)
        if standard_candidate and standard_candidate.results_summary_path
        else "",
        "bootstrap_results_raw": str(bootstrap_candidate.results_bootstrap_raw_path)
        if bootstrap_candidate and bootstrap_candidate.results_bootstrap_raw_path
        else "",
        "dataset_path": str(dataset_path),
        "domain_column": domain_column,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "random_state": RANDOM_STATE,
    }
    (output_dir / "site_distance_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = discover_output_candidates(input_root)
    standard_candidate = choose_candidate(candidates, want_bootstrap=False)
    bootstrap_candidate = choose_candidate(candidates, want_bootstrap=True)
    if standard_candidate is None and bootstrap_candidate is None:
        raise RuntimeError(
            "Could not find a standard or bootstrap candidate for the site distance analysis."
        )

    base_candidate = standard_candidate or bootstrap_candidate
    assert base_candidate is not None
    dataset_df, domain_column, target_column, id_column, dataset_path = load_dataset(
        base_candidate.config_snapshot,
        search_root=input_root,
    )
    dataset_name = str(
        base_candidate.config_snapshot.get("dataset_name", input_root.name)
    )
    site_order = site_order_from_candidates(
        standard_candidate=standard_candidate,
        bootstrap_candidate=bootstrap_candidate,
        dataset_df=dataset_df,
        domain_column=domain_column,
    )
    feature_columns = select_numeric_feature_columns(
        dataset_df,
        domain_column=domain_column,
        target_column=target_column,
        id_column=id_column,
    )

    site_distance_rows = [
        compute_site_distance_row(
            dataset_df,
            dataset_name=dataset_name,
            target_site=target_site,
            domain_column=domain_column,
            feature_columns=feature_columns,
            random_state=RANDOM_STATE,
        )
        for target_site in site_order
    ]
    site_distance_df = pd.DataFrame(site_distance_rows)

    performance_summary_df = aggregate_performance_results(
        standard_candidate=standard_candidate,
        bootstrap_candidate=bootstrap_candidate,
    )
    site_distance_vs_performance_df = build_site_distance_vs_performance(
        performance_summary_df=performance_summary_df,
        site_distance_df=site_distance_df,
        dataset_name=dataset_name,
    )
    site_distance_correlations_df = build_site_distance_correlations(
        site_distance_vs_performance_df
    )

    site_distance_df.to_csv(output_dir / "site_distance_by_target.csv", index=False)
    site_distance_vs_performance_df.to_csv(
        output_dir / "site_distance_vs_performance.csv", index=False
    )
    site_distance_correlations_df.to_csv(
        output_dir / "site_distance_correlations.csv", index=False
    )

    write_single_sheet_excel(
        output_dir / "site_distance_by_target.xlsx", "site_distance", site_distance_df
    )
    write_single_sheet_excel(
        output_dir / "site_distance_vs_performance.xlsx",
        "distance_vs_perf",
        site_distance_vs_performance_df,
    )
    write_single_sheet_excel(
        output_dir / "site_distance_correlations.xlsx",
        "distance_corr",
        site_distance_correlations_df,
    )

    display_map: dict[str, str] = {}
    model_display_map: dict[str, str] = {}
    for candidate in [bootstrap_candidate, standard_candidate]:
        if candidate is None or candidate.results_summary_path is None:
            continue
        preview_df = pd.read_csv(candidate.results_summary_path, nrows=300)
        if {"target_site", "target_site_display"}.issubset(preview_df.columns):
            display_map.update(
                {
                    str(row["target_site"]): str(row["target_site_display"])
                    for _, row in preview_df[["target_site", "target_site_display"]]
                    .dropna()
                    .drop_duplicates()
                    .iterrows()
                }
            )
        if {"model", "model_display"}.issubset(preview_df.columns):
            model_display_map.update(
                {
                    str(row["model"]): str(row["model_display"])
                    for _, row in preview_df[["model", "model_display"]]
                    .dropna()
                    .drop_duplicates()
                    .iterrows()
                }
            )

    generate_figures(
        site_distance_df=site_distance_df,
        site_distance_vs_performance_df=site_distance_vs_performance_df,
        output_dir=output_dir,
        display_map=display_map,
        model_display_map=model_display_map,
    )
    write_readme(
        output_dir,
        dataset_name=dataset_name,
        feature_columns=feature_columns,
        site_distance_df=site_distance_df,
    )
    write_metadata(
        output_dir,
        input_root=input_root,
        standard_candidate=standard_candidate,
        bootstrap_candidate=bootstrap_candidate,
        dataset_path=dataset_path,
        domain_column=domain_column,
        target_column=target_column,
        feature_columns=feature_columns,
    )
    print(f"Wrote site distance analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
