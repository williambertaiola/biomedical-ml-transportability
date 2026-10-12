"""Post-experiment analysis: generate entropy analysis."""

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
from scipy.stats import pearsonr, spearmanr

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "codex_entropy_mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


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
        description="Generate entropy analysis from an existing transportability output folder"
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
        help="Directory where entropy outputs will be written",
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
    alias_candidates = [
        requested_domain_column,
        "site",
        "domain_id",
        "domain",
        "center",
        "batch",
    ]
    for candidate in alias_candidates:
        if candidate in df.columns:
            return candidate
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
) -> tuple[pd.DataFrame, str, str, Path]:
    dataset_path = resolve_existing_path(
        str(snapshot["input_path"]), search_root=search_root
    )
    requested_domain_column = str(snapshot["domain_column"])
    requested_target_column = str(snapshot["target_column"])
    df = pd.read_csv(dataset_path, low_memory=False)
    domain_column = resolve_domain_column_name(df, requested_domain_column)
    target_column = resolve_target_column_name(df, requested_target_column)
    df = normalize_binary_target_values(df, target_column)
    return df, domain_column, target_column, dataset_path


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


def parse_balance(value: Any) -> dict[str, float]:
    if pd.isna(value):
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text:
        return {}
    return json.loads(text)


def count_for_class(balance: dict[str, Any], class_id: int) -> int:
    value = balance.get(f"n_class_{class_id}")
    if value is None or pd.isna(value):
        return 0
    return int(value)


def entropy_from_counts(
    class_counts: list[int], *, n_classes: int
) -> tuple[float, float]:
    total = int(sum(class_counts))
    if total <= 0:
        return (0.0, 0.0)
    probabilities = [count / total for count in class_counts if count > 0]
    entropy = float(
        -sum(probability * math.log2(probability) for probability in probabilities)
    )
    if n_classes <= 1:
        return (entropy, 0.0)
    normalized_entropy = float(entropy / math.log2(n_classes))
    return (entropy, normalized_entropy)


def prevalence_safe(count_positive: int, total: int) -> float:
    if total <= 0:
        return float("nan")
    return float(count_positive / total)


def pretty_target_site(site: str, display_map: dict[str, str]) -> str:
    if site in display_map:
        return display_map[site]
    return site.replace("_", " ").title().replace(" Va", " VA")


def pretty_model_name(model_name: str, model_display_map: dict[str, str]) -> str:
    return model_display_map.get(model_name, model_name)


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


def dataset_level_site_counts(
    dataset_df: pd.DataFrame,
    *,
    domain_column: str,
    target_column: str,
    site_order: list[str],
    n_classes: int,
) -> pd.DataFrame:
    total_n = len(dataset_df)
    total_class_0 = int((dataset_df[target_column].astype(int) == 0).sum())
    total_class_1 = int((dataset_df[target_column].astype(int) == 1).sum())
    rows: list[dict[str, Any]] = []
    for site in site_order:
        site_df = dataset_df.loc[dataset_df[domain_column].astype(str) == site].copy()
        class_0_count = int((site_df[target_column].astype(int) == 0).sum())
        class_1_count = int((site_df[target_column].astype(int) == 1).sum())
        n_total = len(site_df)
        entropy, normalized_entropy = entropy_from_counts(
            [class_0_count, class_1_count], n_classes=n_classes
        )
        rows.append(
            {
                "target_site": site,
                "n_total_target": n_total,
                "class_0_count": class_0_count,
                "class_1_count": class_1_count,
                "class_0_ratio": prevalence_safe(class_0_count, n_total),
                "class_1_ratio": prevalence_safe(class_1_count, n_total),
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "n_source": total_n - n_total,
                "source_class_0_count": total_class_0 - class_0_count,
                "source_class_1_count": total_class_1 - class_1_count,
            }
        )
    return pd.DataFrame(rows)


def build_entropy_row(
    *,
    dataset_name: str,
    target_site: str,
    split_type: str,
    n_total: int,
    class_0_count: int,
    class_1_count: int,
    n_classes: int,
    bootstrap_iteration: int | None = None,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    entropy, normalized_entropy = entropy_from_counts(
        [class_0_count, class_1_count], n_classes=n_classes
    )
    return {
        "dataset": dataset_name,
        "target_site": target_site,
        "split_type": split_type,
        "bootstrap_iteration": bootstrap_iteration,
        "bootstrap_seed": bootstrap_seed,
        "n_total": int(n_total),
        "class_0_count": int(class_0_count),
        "class_1_count": int(class_1_count),
        "class_0_ratio": prevalence_safe(int(class_0_count), int(n_total)),
        "class_1_ratio": prevalence_safe(int(class_1_count), int(n_total)),
        "entropy": float(entropy),
        "normalized_entropy": float(normalized_entropy),
    }


def append_manifest_split_rows(
    *,
    manifest_df: pd.DataFrame,
    dataset_name: str,
    n_classes: int,
    rows: list[dict[str, Any]],
    use_bootstrap_columns: bool,
) -> None:
    for row in manifest_df.to_dict(orient="records"):
        bootstrap_iteration = (
            int(row["bootstrap_index"])
            if use_bootstrap_columns and pd.notna(row.get("bootstrap_index"))
            else None
        )
        bootstrap_seed = (
            int(row["bootstrap_seed"])
            if use_bootstrap_columns and pd.notna(row.get("bootstrap_seed"))
            else None
        )
        target_site = str(row["evaluation_target_site"])

        source_balance = parse_balance(row.get("source_class_balance"))
        train_balance = parse_balance(row.get("target_train_1_class_balance"))
        test_balance = parse_balance(row.get("target_test_1_class_balance"))

        rows.append(
            build_entropy_row(
                dataset_name=dataset_name,
                target_site=target_site,
                split_type="source",
                n_total=int(row["n_source_samples"]),
                class_0_count=count_for_class(source_balance, 0),
                class_1_count=count_for_class(source_balance, 1),
                n_classes=n_classes,
                bootstrap_iteration=bootstrap_iteration,
                bootstrap_seed=bootstrap_seed,
            )
        )
        rows.append(
            build_entropy_row(
                dataset_name=dataset_name,
                target_site=target_site,
                split_type="train_1",
                n_total=int(row["n_target_train_1"]),
                class_0_count=count_for_class(train_balance, 0),
                class_1_count=count_for_class(train_balance, 1),
                n_classes=n_classes,
                bootstrap_iteration=bootstrap_iteration,
                bootstrap_seed=bootstrap_seed,
            )
        )
        rows.append(
            build_entropy_row(
                dataset_name=dataset_name,
                target_site=target_site,
                split_type="test_1",
                n_total=int(row["n_target_test_1"]),
                class_0_count=count_for_class(test_balance, 0),
                class_1_count=count_for_class(test_balance, 1),
                n_classes=n_classes,
                bootstrap_iteration=bootstrap_iteration,
                bootstrap_seed=bootstrap_seed,
            )
        )


def build_entropy_long_table(
    *,
    dataset_name: str,
    site_counts_df: pd.DataFrame,
    standard_manifest_df: pd.DataFrame | None,
    bootstrap_manifest_df: pd.DataFrame | None,
    n_classes: int,
    site_order: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for row in site_counts_df.to_dict(orient="records"):
        rows.append(
            build_entropy_row(
                dataset_name=dataset_name,
                target_site=str(row["target_site"]),
                split_type="full_target",
                n_total=int(row["n_total_target"]),
                class_0_count=int(row["class_0_count"]),
                class_1_count=int(row["class_1_count"]),
                n_classes=n_classes,
            )
        )

    if bootstrap_manifest_df is not None:
        append_manifest_split_rows(
            manifest_df=bootstrap_manifest_df,
            dataset_name=dataset_name,
            n_classes=n_classes,
            rows=rows,
            use_bootstrap_columns=True,
        )
    elif standard_manifest_df is not None:
        append_manifest_split_rows(
            manifest_df=standard_manifest_df,
            dataset_name=dataset_name,
            n_classes=n_classes,
            rows=rows,
            use_bootstrap_columns=False,
        )

    entropy_df = pd.DataFrame(rows)
    entropy_df["target_site"] = pd.Categorical(
        entropy_df["target_site"], categories=site_order, ordered=True
    )
    entropy_df = entropy_df.sort_values(
        ["target_site", "split_type", "bootstrap_iteration"], kind="stable"
    ).reset_index(drop=True)
    entropy_df["target_site"] = entropy_df["target_site"].astype(str)
    return entropy_df


def build_entropy_summary(
    *,
    entropy_long_df: pd.DataFrame,
    dataset_name: str,
    site_counts_df: pd.DataFrame,
    site_order: list[str],
) -> pd.DataFrame:
    site_counts_lookup = site_counts_df.set_index("target_site")
    summary_rows: list[dict[str, Any]] = []
    for target_site in site_order:
        site_rows = entropy_long_df.loc[
            entropy_long_df["target_site"] == target_site
        ].copy()
        full_target_row = site_rows.loc[site_rows["split_type"] == "full_target"].iloc[
            0
        ]
        source_rows = site_rows.loc[site_rows["split_type"] == "source"].copy()
        train_rows = site_rows.loc[site_rows["split_type"] == "train_1"].copy()
        test_rows = site_rows.loc[site_rows["split_type"] == "test_1"].copy()

        summary_rows.append(
            {
                "dataset": dataset_name,
                "target_site": target_site,
                "n_total_target": int(
                    site_counts_lookup.loc[target_site, "n_total_target"]
                ),
                "n_source": int(site_counts_lookup.loc[target_site, "n_source"]),
                "mean_n_train_1": float(train_rows["n_total"].mean())
                if not train_rows.empty
                else float("nan"),
                "mean_n_test_1": float(test_rows["n_total"].mean())
                if not test_rows.empty
                else float("nan"),
                "full_target_entropy": float(full_target_row["entropy"]),
                "full_target_normalized_entropy": float(
                    full_target_row["normalized_entropy"]
                ),
                "mean_train_1_entropy": float(train_rows["entropy"].mean())
                if not train_rows.empty
                else float("nan"),
                "std_train_1_entropy": float(train_rows["entropy"].std(ddof=0))
                if not train_rows.empty
                else float("nan"),
                "mean_test_1_entropy": float(test_rows["entropy"].mean())
                if not test_rows.empty
                else float("nan"),
                "std_test_1_entropy": float(test_rows["entropy"].std(ddof=0))
                if not test_rows.empty
                else float("nan"),
                "mean_source_entropy": float(source_rows["entropy"].mean())
                if not source_rows.empty
                else float("nan"),
                "std_source_entropy": float(source_rows["entropy"].std(ddof=0))
                if not source_rows.empty
                else float("nan"),
            }
        )
    return pd.DataFrame(summary_rows)


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
        agg_spec: dict[str, list[str]] = {
            "ece_ood": ["mean"],
        }
        for metric_base, metric_columns in PERFORMANCE_METRICS.items():
            del metric_base
            agg_spec[metric_columns["ood"]] = [
                "mean",
                lambda series: series.std(ddof=0),
            ]
            agg_spec[metric_columns["id"]] = ["mean"]
            agg_spec[metric_columns["gap"]] = ["mean"]

        aggregated = raw_df.groupby(group_columns, dropna=False).agg(agg_spec)
        aggregated.columns = [
            (
                f"{column}_std"
                if stat_name == "<lambda_0>"
                else f"{column}_{stat_name}"
                if stat_name != "mean"
                else column
            )
            for column, stat_name in aggregated.columns
        ]
        aggregated = aggregated.reset_index()
        n_bootstrap = (
            raw_df.groupby(group_columns, dropna=False)["bootstrap_index"]
            .nunique()
            .reset_index(name="n_bootstrap")
        )
        aggregated = aggregated.merge(n_bootstrap, on=group_columns, how="left")
        return aggregated

    if (
        bootstrap_candidate is not None
        and bootstrap_candidate.results_summary_path is not None
    ):
        return pd.read_csv(bootstrap_candidate.results_summary_path)

    if standard_candidate is None or standard_candidate.results_summary_path is None:
        raise FileNotFoundError(
            "Could not find results tables for the entropy-vs-OOD analysis."
        )

    summary_df = pd.read_csv(standard_candidate.results_summary_path).copy()
    for metric_columns in PERFORMANCE_METRICS.values():
        summary_df[f"{metric_columns['ood']}_std"] = np.nan
    summary_df["ece_ood_mean"] = summary_df["ece_ood"]
    summary_df["ece_ood_std"] = np.nan
    summary_df["n_bootstrap"] = np.nan
    return summary_df


def build_entropy_vs_ood_results(
    *,
    performance_summary_df: pd.DataFrame,
    entropy_summary_df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    entropy_lookup = entropy_summary_df[
        ["target_site", "n_total_target", "full_target_normalized_entropy"]
    ].rename(columns={"full_target_normalized_entropy": "normalized_entropy"})
    merged = performance_summary_df.merge(entropy_lookup, on="target_site", how="left")

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
                    "normalized_entropy": row.get("normalized_entropy"),
                    "n_total_target": row.get("n_total_target"),
                    "mean_ood_performance": row.get(metric_columns["ood"]),
                    "std_ood_performance": row.get(f"{metric_columns['ood']}_std"),
                    "mean_id_performance": row.get(metric_columns["id"]),
                    "id_ood_gap": row.get(metric_columns["gap"]),
                    "mean_ece": row.get("ece_ood_mean", row.get("ece_ood")),
                    "std_ece": row.get("ece_ood_std"),
                    "n_bootstrap": row.get("n_bootstrap"),
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
    pearson_r, pearson_p = pearsonr(x_values, y_values)
    spearman_r, spearman_p = spearmanr(x_values, y_values)
    return (
        float(pearson_r),
        float(pearson_p),
        float(spearman_r),
        float(spearman_p),
        n_sites,
    )


def build_entropy_correlations(entropy_vs_ood_df: pd.DataFrame) -> pd.DataFrame:
    variable_pairs = [
        ("normalized_entropy", "mean_ood_performance"),
        ("normalized_entropy", "std_ood_performance"),
        ("normalized_entropy", "id_ood_gap"),
        ("normalized_entropy", "mean_ece"),
    ]
    rows: list[dict[str, Any]] = []
    group_columns = ["dataset", "model", "metric", "method"]
    for group_keys, group_df in entropy_vs_ood_df.groupby(group_columns, dropna=False):
        dataset_name, model_name, metric_name, method_name = group_keys
        for variable_x, variable_y in variable_pairs:
            pearson_r, pearson_p, spearman_r, spearman_p, n_sites = correlation_stats(
                group_df[variable_x],
                group_df[variable_y],
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "metric": metric_name,
                    "method": method_name,
                    "variable_x": variable_x,
                    "variable_y": variable_y,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
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


def save_figure(
    fig: plt.Figure,
    output_path: Path,
    *,
    bottom_rect: tuple[float, float, float, float] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if bottom_rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=bottom_rect)
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


def plot_normalized_entropy_by_site(
    *,
    entropy_summary_df: pd.DataFrame,
    output_path: Path,
    display_map: dict[str, str],
) -> None:
    apply_plot_style()
    plot_df = entropy_summary_df.copy()
    x_labels = [
        pretty_target_site(site, display_map) for site in plot_df["target_site"]
    ]
    values = plot_df["full_target_normalized_entropy"].to_numpy(dtype=float)
    n_values = plot_df["n_total_target"].to_numpy(dtype=int)
    colors = plt.get_cmap("RdYlGn")(values)

    fig, ax = plt.subplots(figsize=(max(8.5, len(plot_df) * 2.2), 5.8))
    bars = ax.bar(
        np.arange(len(plot_df)),
        values,
        color=colors,
        edgecolor="#404040",
        linewidth=0.7,
    )
    ax.set_title("Normalized entropy by target site")
    ax.set_ylabel("Normalized entropy")
    ax.set_xlabel("Target site")
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(x_labels, rotation=0, ha="center")
    ax.grid(False)

    for bar, entropy_value, n_total in zip(bars, values, n_values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(1.05, entropy_value + 0.02),
            f"{entropy_value:.3f}\nn={n_total}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_figure(fig, output_path)


def plot_class_balance_by_site(
    *,
    site_counts_df: pd.DataFrame,
    output_path: Path,
    display_map: dict[str, str],
) -> None:
    apply_plot_style()
    plot_df = site_counts_df.copy()
    x_labels = [
        pretty_target_site(site, display_map) for site in plot_df["target_site"]
    ]
    class_0_ratio = plot_df["class_0_ratio"].to_numpy(dtype=float)
    class_1_ratio = plot_df["class_1_ratio"].to_numpy(dtype=float)
    n_values = plot_df["n_total_target"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(max(8.5, len(plot_df) * 2.2), 5.8))
    bar_positions = np.arange(len(plot_df))
    class_0_handle = Patch(facecolor="#B8B8B8", edgecolor="#404040")
    class_1_handle = Patch(facecolor="#54A24B", edgecolor="#404040")
    ax.bar(
        bar_positions,
        class_0_ratio,
        color="#B8B8B8",
        edgecolor="#404040",
        linewidth=0.7,
    )
    ax.bar(
        bar_positions,
        class_1_ratio,
        bottom=class_0_ratio,
        color="#54A24B",
        edgecolor="#404040",
        linewidth=0.7,
    )
    ax.set_title("Class balance by target site")
    ax.set_ylabel("Class proportion")
    ax.set_xlabel("Target site")
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(x_labels, rotation=0, ha="center")
    ax.grid(False)

    for position, n_total in zip(bar_positions, n_values, strict=False):
        ax.text(position, 1.02, f"n={n_total}", ha="center", va="bottom", fontsize=9)
    add_bottom_legend(
        fig,
        [class_0_handle, class_1_handle],
        ["Class 0 ratio", "Class 1 ratio"],
        ncol=2,
    )
    save_figure(fig, output_path, bottom_rect=(0, 0.08, 1, 1))


def annotate_scatter_points(ax: plt.Axes, subset: pd.DataFrame) -> None:
    for _, row in subset.iterrows():
        ax.annotate(
            str(row["target_site_display"]),
            (float(row["normalized_entropy"]), float(row["_scatter_y"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color="#333333",
            alpha=0.8,
        )


def plot_entropy_scatter(
    *,
    entropy_vs_ood_df: pd.DataFrame,
    output_path: Path,
    display_map: dict[str, str],
    model_display_map: dict[str, str],
    model_name: str,
    metric_name: str,
    y_column: str,
    title_suffix: str,
    y_label: str,
    add_zero_line: bool = False,
) -> None:
    subset = entropy_vs_ood_df.loc[
        (entropy_vs_ood_df["model"] == model_name)
        & (entropy_vs_ood_df["metric"] == metric_name)
        & (entropy_vs_ood_df[y_column].notna())
    ].copy()
    if subset.empty:
        return

    subset["target_site_display"] = subset["target_site"].map(
        lambda site: pretty_target_site(str(site), display_map)
    )
    subset["_scatter_y"] = subset[y_column].astype(float)
    subset["_scatter_size"] = (
        60.0 + subset["n_total_target"].fillna(0).astype(float) * 0.45
    )

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9.4, 6.1))
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
            method_subset["normalized_entropy"],
            method_subset["_scatter_y"],
            s=method_subset["_scatter_size"],
            color=METHOD_COLORS.get(method_name, "#777777"),
            edgecolor="#303030",
            linewidth=0.7,
            alpha=0.9,
        )
        annotate_scatter_points(ax, method_subset)

    if add_zero_line:
        ax.axhline(0.0, color="#777777", linewidth=0.9, linestyle="--")

    metric_label = PERFORMANCE_METRICS[metric_name]["label"]
    ax.set_title(
        f"{title_suffix} | {pretty_model_name(model_name, model_display_map)} | {metric_label}"
    )
    ax.set_xlabel("Full target normalized entropy")
    ax.set_ylabel(y_label)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(False)
    add_bottom_legend(
        fig, legend_handles, legend_labels, ncol=max(1, len(legend_labels))
    )
    save_figure(fig, output_path, bottom_rect=(0, 0.1, 1, 1))


def generate_figures(
    *,
    entropy_summary_df: pd.DataFrame,
    site_counts_df: pd.DataFrame,
    entropy_vs_ood_df: pd.DataFrame,
    output_dir: Path,
    display_map: dict[str, str],
    model_display_map: dict[str, str],
) -> None:
    figures_dir = output_dir / "figures"
    plot_normalized_entropy_by_site(
        entropy_summary_df=entropy_summary_df,
        output_path=figures_dir / "normalized_entropy_by_site.png",
        display_map=display_map,
    )
    plot_class_balance_by_site(
        site_counts_df=site_counts_df,
        output_path=figures_dir / "class_balance_by_site.png",
        display_map=display_map,
    )

    for model_name in entropy_vs_ood_df["model"].dropna().astype(str).unique():
        for metric_name in PERFORMANCE_METRICS:
            plot_entropy_scatter(
                entropy_vs_ood_df=entropy_vs_ood_df,
                output_path=figures_dir
                / f"entropy_vs_ood_{model_name.lower()}_{metric_name}.png",
                display_map=display_map,
                model_display_map=model_display_map,
                model_name=model_name,
                metric_name=metric_name,
                y_column="mean_ood_performance",
                title_suffix="Entropy vs mean OOD performance",
                y_label=f"Mean OOD {PERFORMANCE_METRICS[metric_name]['label']}",
            )
            plot_entropy_scatter(
                entropy_vs_ood_df=entropy_vs_ood_df,
                output_path=figures_dir
                / f"entropy_vs_ood_std_{model_name.lower()}_{metric_name}.png",
                display_map=display_map,
                model_display_map=model_display_map,
                model_name=model_name,
                metric_name=metric_name,
                y_column="std_ood_performance",
                title_suffix="Entropy vs OOD variability",
                y_label=f"STD OOD {PERFORMANCE_METRICS[metric_name]['label']}",
            )
            plot_entropy_scatter(
                entropy_vs_ood_df=entropy_vs_ood_df,
                output_path=figures_dir
                / f"entropy_vs_id_ood_gap_{model_name.lower()}_{metric_name}.png",
                display_map=display_map,
                model_display_map=model_display_map,
                model_name=model_name,
                metric_name=metric_name,
                y_column="id_ood_gap",
                title_suffix="Entropy vs ID-OOD gap",
                y_label=f"ID-OOD gap ({PERFORMANCE_METRICS[metric_name]['label']})",
                add_zero_line=True,
            )


def write_readme(output_dir: Path, *, dataset_name: str) -> None:
    readme_text = f"""Entropy analysis outputs for {dataset_name}

This folder contains a dedicated entropy analysis built from the existing cross-site pipeline outputs.

Files:
- entropy_by_site_and_split.csv / .xlsx
  Long-format table with sample size, class balance, entropy and normalized entropy for:
  full_target, source, train_1 and test_1. When bootstrap outputs are available, train_1/test_1/source
  are reported for each bootstrap iteration.

- entropy_summary_by_site.csv / .xlsx
  Site-level summary with full-target entropy plus mean/std entropy across train_1, test_1 and source splits.

- entropy_vs_ood_results.csv / .xlsx
  Site-level link between full-target normalized entropy and the cross-site OOD results already produced by the pipeline.

- entropy_correlations.csv / .xlsx
  Pearson and Spearman correlations between normalized entropy and:
  mean OOD performance, OOD variability, ID-OOD gap and ECE.

- figures/
  Dedicated PNG figures for entropy, class balance and entropy-vs-OOD relationships.

Interpretation:
- normalized entropy near 1 indicates a balanced binary class distribution.
- normalized entropy near 0 indicates strong class imbalance.
- lower entropy can be compared against OOD mean performance, OOD variability, ID-OOD gap and calibration.

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
    }
    (output_dir / "entropy_metadata.json").write_text(
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
            "Could not find a standard or bootstrap candidate for the entropy analysis."
        )

    base_candidate = standard_candidate or bootstrap_candidate
    assert base_candidate is not None

    dataset_df, domain_column, target_column, dataset_path = load_dataset(
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
    n_classes = int(dataset_df[target_column].nunique())

    site_counts_df = dataset_level_site_counts(
        dataset_df,
        domain_column=domain_column,
        target_column=target_column,
        site_order=site_order,
        n_classes=n_classes,
    )

    standard_manifest_df = (
        pd.read_csv(standard_candidate.manifest_path)
        if standard_candidate is not None
        else None
    )
    bootstrap_manifest_df = (
        pd.read_csv(bootstrap_candidate.manifest_path)
        if bootstrap_candidate is not None
        else None
    )

    entropy_long_df = build_entropy_long_table(
        dataset_name=dataset_name,
        site_counts_df=site_counts_df,
        standard_manifest_df=standard_manifest_df,
        bootstrap_manifest_df=bootstrap_manifest_df,
        n_classes=n_classes,
        site_order=site_order,
    )
    entropy_summary_df = build_entropy_summary(
        entropy_long_df=entropy_long_df,
        dataset_name=dataset_name,
        site_counts_df=site_counts_df,
        site_order=site_order,
    )

    performance_summary_df = aggregate_performance_results(
        standard_candidate=standard_candidate,
        bootstrap_candidate=bootstrap_candidate,
    )
    entropy_vs_ood_df = build_entropy_vs_ood_results(
        performance_summary_df=performance_summary_df,
        entropy_summary_df=entropy_summary_df,
        dataset_name=dataset_name,
    )
    entropy_correlations_df = build_entropy_correlations(entropy_vs_ood_df)

    entropy_long_df.to_csv(output_dir / "entropy_by_site_and_split.csv", index=False)
    entropy_summary_df.to_csv(output_dir / "entropy_summary_by_site.csv", index=False)
    entropy_vs_ood_df.to_csv(output_dir / "entropy_vs_ood_results.csv", index=False)
    entropy_correlations_df.to_csv(output_dir / "entropy_correlations.csv", index=False)

    write_single_sheet_excel(
        output_dir / "entropy_by_site_and_split.xlsx",
        "entropy_by_split",
        entropy_long_df,
    )
    write_single_sheet_excel(
        output_dir / "entropy_summary_by_site.xlsx",
        "entropy_summary",
        entropy_summary_df,
    )
    write_single_sheet_excel(
        output_dir / "entropy_vs_ood_results.xlsx", "entropy_vs_ood", entropy_vs_ood_df
    )
    write_single_sheet_excel(
        output_dir / "entropy_correlations.xlsx",
        "entropy_correlations",
        entropy_correlations_df,
    )

    display_map: dict[str, str] = {}
    model_display_map: dict[str, str] = {}
    for candidate in [bootstrap_candidate, standard_candidate]:
        if candidate is None:
            continue
        if candidate.results_summary_path is None:
            continue
        results_df = pd.read_csv(candidate.results_summary_path, nrows=200)
        if {"target_site", "target_site_display"}.issubset(results_df.columns):
            display_map.update(
                {
                    str(row["target_site"]): str(row["target_site_display"])
                    for _, row in results_df[["target_site", "target_site_display"]]
                    .dropna()
                    .drop_duplicates()
                    .iterrows()
                }
            )
        if {"model", "model_display"}.issubset(results_df.columns):
            model_display_map.update(
                {
                    str(row["model"]): str(row["model_display"])
                    for _, row in results_df[["model", "model_display"]]
                    .dropna()
                    .drop_duplicates()
                    .iterrows()
                }
            )

    generate_figures(
        entropy_summary_df=entropy_summary_df,
        site_counts_df=site_counts_df,
        entropy_vs_ood_df=entropy_vs_ood_df,
        output_dir=output_dir,
        display_map=display_map,
        model_display_map=model_display_map,
    )
    write_readme(output_dir, dataset_name=dataset_name)
    write_metadata(
        output_dir,
        input_root=input_root,
        standard_candidate=standard_candidate,
        bootstrap_candidate=bootstrap_candidate,
        dataset_path=dataset_path,
        domain_column=domain_column,
        target_column=target_column,
    )
    print(f"Wrote entropy analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
