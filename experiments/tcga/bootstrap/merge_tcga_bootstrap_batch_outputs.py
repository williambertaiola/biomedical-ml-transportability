"""Transportability experiment: merge tcga bootstrap batch outputs."""

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


import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = _REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tcga_transportability.bootstrap_reporting import (
    aggregate_target_site_results,
    save_bootstrap_outputs,
    write_bootstrap_markdown_report,
)
from tcga_transportability.config import ExperimentConfig


def build_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tcga_transportability_bootstrap_merge")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge batched TCGA bootstrap outputs into a single final package"
    )
    parser.add_argument(
        "--input-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more batch output directories produced by run_tcga_bootstrap_resampling.py",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Merged output directory"
    )
    return parser.parse_args()


def _load_csv(batch_dir: Path, relative_path: str) -> pd.DataFrame:
    path = batch_dir / relative_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _build_config_from_snapshot(snapshot: dict, output_dir: Path) -> ExperimentConfig:
    config = ExperimentConfig(
        input_path=Path(snapshot["input_path"]),
        output_dir=output_dir,
    )
    config.dataset_name = snapshot.get("dataset_name", config.dataset_name)
    config.enabled_methods = tuple(
        snapshot.get("enabled_methods", config.enabled_methods)
    )
    config.target_sites = tuple()
    config.target_column = snapshot.get("target_column", config.target_column)
    config.target_label_column = snapshot.get(
        "target_label_column", config.target_label_column
    )
    config.domain_column = snapshot.get("domain_column", config.domain_column)
    config.id_column = snapshot.get("id_column", config.id_column)
    config.gene_prefix = snapshot.get("gene_prefix", config.gene_prefix)
    config.target_test_size = float(
        snapshot.get("target_test_size", config.target_test_size)
    )
    config.random_seed = int(snapshot.get("random_seed", config.random_seed))
    config.ece_bins = int(snapshot.get("ece_bins", config.ece_bins))
    config.n_jobs = int(snapshot.get("n_jobs", config.n_jobs))
    config.threshold = float(snapshot.get("threshold", config.threshold))
    config.min_site_samples = int(
        snapshot.get("min_site_samples", config.min_site_samples)
    )
    config.min_site_class_count = int(
        snapshot.get("min_site_class_count", config.min_site_class_count)
    )
    config.combat_top_k_variance_genes = int(
        snapshot.get("combat_top_k_variance_genes", config.combat_top_k_variance_genes)
    )
    config.top_k_variance_genes = int(
        snapshot.get("top_k_variance_genes", config.top_k_variance_genes)
    )
    config.top_k_model_genes = int(
        snapshot.get("top_k_model_genes", config.top_k_model_genes)
    )
    config.combat_label_covariate = snapshot.get(
        "combat_label_covariate", config.combat_label_covariate
    )
    config.m2_weight_temperature = float(
        snapshot.get("m2_weight_temperature", config.m2_weight_temperature)
    )
    config.m2_min_domain_weight = float(
        snapshot.get("m2_min_domain_weight", config.m2_min_domain_weight)
    )
    config.m3_target_blend_weight = float(
        snapshot.get("m3_target_blend_weight", config.m3_target_blend_weight)
    )
    return config


def _build_model_summary_from_target_results(
    raw_target_results_df: pd.DataFrame,
) -> pd.DataFrame:
    if raw_target_results_df.empty:
        return pd.DataFrame()

    per_bootstrap = (
        raw_target_results_df.groupby(
            ["bootstrap_index", "bootstrap_seed", "method", "model", "model_display"],
            dropna=False,
        )[
            [
                "decision_threshold",
                "roc_auc_id",
                "roc_auc_ood",
                "gap_roc_auc",
                "f1_id",
                "f1_ood",
                "gap_f1",
                "balanced_accuracy_id",
                "balanced_accuracy_ood",
                "gap_balanced_accuracy",
                "ece_id",
                "ece_ood",
            ]
        ]
        .mean()
        .reset_index()
    )
    n_target_sites = (
        raw_target_results_df.groupby(
            ["bootstrap_index", "bootstrap_seed", "method", "model", "model_display"],
            dropna=False,
        )["target_site"]
        .nunique()
        .reset_index(name="n_target_sites")
    )
    per_bootstrap = per_bootstrap.merge(
        n_target_sites,
        on=["bootstrap_index", "bootstrap_seed", "method", "model", "model_display"],
        how="left",
    )
    return per_bootstrap.rename(
        columns={
            "decision_threshold": "decision_threshold_mean",
            "roc_auc_id": "roc_auc_id_mean",
            "roc_auc_ood": "roc_auc_ood_mean",
            "gap_roc_auc": "gap_roc_auc_mean",
            "f1_id": "f1_id_mean",
            "f1_ood": "f1_ood_mean",
            "gap_f1": "gap_f1_mean",
            "balanced_accuracy_id": "balanced_accuracy_id_mean",
            "balanced_accuracy_ood": "balanced_accuracy_ood_mean",
            "gap_balanced_accuracy": "gap_balanced_accuracy_mean",
            "ece_id": "ece_id_mean",
            "ece_ood": "ece_ood_mean",
        }
    )


def _aggregate_model_summary(raw_model_summary_df: pd.DataFrame) -> pd.DataFrame:
    if raw_model_summary_df.empty:
        return pd.DataFrame()
    group_columns = ["method", "model", "model_display"]
    metric_columns = [
        column for column in raw_model_summary_df.columns if column.endswith("_mean")
    ]
    static_columns = [
        column
        for column in raw_model_summary_df.columns
        if column
        not in set(
            group_columns + metric_columns + ["bootstrap_index", "bootstrap_seed"]
        )
    ]

    def collapse_static(series: pd.Series):
        non_null = series.dropna()
        if non_null.empty:
            return ""
        unique_values = non_null.astype(str).unique().tolist()
        if len(unique_values) == 1:
            value = non_null.iloc[0]
            return value.item() if hasattr(value, "item") else value
        return "<varies_across_bootstrap>"

    static_df = (
        raw_model_summary_df.groupby(group_columns, dropna=False)[static_columns]
        .agg(collapse_static)
        .reset_index()
    )
    mean_df = (
        raw_model_summary_df.groupby(group_columns, dropna=False)[metric_columns]
        .mean()
        .reset_index()
    )
    std_df = (
        raw_model_summary_df.groupby(group_columns, dropna=False)[metric_columns]
        .std(ddof=0)
        .fillna(0.0)
        .rename(columns={metric: f"{metric}_std" for metric in metric_columns})
        .reset_index()
    )
    median_df = (
        raw_model_summary_df.groupby(group_columns, dropna=False)[metric_columns]
        .median()
        .rename(columns={metric: f"{metric}_median" for metric in metric_columns})
        .reset_index()
    )
    q1_df = (
        raw_model_summary_df.groupby(group_columns, dropna=False)[metric_columns]
        .quantile(0.25)
        .rename(columns={metric: f"{metric}_q1" for metric in metric_columns})
        .reset_index()
    )
    q3_df = (
        raw_model_summary_df.groupby(group_columns, dropna=False)[metric_columns]
        .quantile(0.75)
        .rename(columns={metric: f"{metric}_q3" for metric in metric_columns})
        .reset_index()
    )
    count_df = (
        raw_model_summary_df.groupby(group_columns, dropna=False)
        .size()
        .reset_index(name="n_bootstrap")
    )

    summary_df = (
        static_df.merge(mean_df, on=group_columns, how="left")
        .merge(std_df, on=group_columns, how="left")
        .merge(median_df, on=group_columns, how="left")
        .merge(q1_df, on=group_columns, how="left")
        .merge(q3_df, on=group_columns, how="left")
        .merge(count_df, on=group_columns, how="left")
        .sort_values(["method", "model"])
        .reset_index(drop=True)
    )
    for metric in metric_columns:
        q1_column = f"{metric}_q1"
        q3_column = f"{metric}_q3"
        summary_df[f"{metric}_iqr"] = summary_df[q3_column] - summary_df[q1_column]
    return summary_df


def main() -> None:
    args = parse_args()
    input_dirs = [path.resolve() for path in args.input_dirs]
    output_dir = args.output_dir.resolve()

    if not input_dirs:
        raise ValueError("At least one batch output directory is required.")

    logger = build_logger(output_dir / "logs" / "merge.log")
    logger.info("Merging %s TCGA bootstrap batch directories", len(input_dirs))

    snapshots = []
    for batch_dir in input_dirs:
        snapshot_path = batch_dir / "config_snapshot.json"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Missing config snapshot: {snapshot_path}")
        snapshots.append(json.loads(snapshot_path.read_text(encoding="utf-8")))

    first_snapshot = snapshots[0]
    config = _build_config_from_snapshot(first_snapshot, output_dir)
    config.enabled_methods = tuple(
        first_snapshot.get("enabled_methods", config.enabled_methods)
    )

    target_site_results_raw = pd.concat(
        [
            _load_csv(batch_dir, "tables/target_site_results_bootstrap_raw.csv")
            for batch_dir in input_dirs
        ],
        ignore_index=True,
    )
    id_validation_raw = pd.concat(
        [
            _load_csv(batch_dir, "tables/id_validation_fold_metrics.csv")
            for batch_dir in input_dirs
        ],
        ignore_index=True,
    )
    prediction_records_raw = pd.concat(
        [
            _load_csv(batch_dir, "tables/prediction_records.csv")
            for batch_dir in input_dirs
        ],
        ignore_index=True,
    )
    split_manifest_raw = pd.concat(
        [
            _load_csv(batch_dir, "tables/target_site_split_manifest.csv")
            for batch_dir in input_dirs
        ],
        ignore_index=True,
    )
    split_assignments_raw = pd.concat(
        [
            _load_csv(batch_dir, "tables/target_site_split_assignments.csv")
            for batch_dir in input_dirs
        ],
        ignore_index=True,
    )
    m2_pairwise_raw = pd.concat(
        [
            _load_csv(batch_dir, "tables/m2_pairwise_source_weighting.csv")
            for batch_dir in input_dirs
        ],
        ignore_index=True,
    )
    bootstrap_iterations_df = (
        pd.concat(
            [
                _load_csv(batch_dir, "tables/bootstrap_iterations.csv")
                for batch_dir in input_dirs
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values(["bootstrap_index", "bootstrap_seed"])
        .reset_index(drop=True)
    )

    target_site_results_summary = aggregate_target_site_results(target_site_results_raw)
    model_summary_raw = _build_model_summary_from_target_results(
        target_site_results_raw
    )
    model_summary_summary = _aggregate_model_summary(model_summary_raw)

    n_bootstrap = int(len(bootstrap_iterations_df))
    bootstrap_seeds = bootstrap_iterations_df["bootstrap_seed"].astype(int).tolist()
    bootstrap_metadata = {
        "dataset_name": config.dataset_name,
        "n_bootstrap": n_bootstrap,
        "base_seed": int(min(bootstrap_seeds))
        if bootstrap_seeds
        else config.random_seed,
        "seed_step": int(bootstrap_seeds[1] - bootstrap_seeds[0])
        if len(bootstrap_seeds) > 1
        else 1,
        "bootstrap_seeds": bootstrap_seeds,
        "iterations": bootstrap_iterations_df.to_dict(orient="records"),
        "target_site_batches": [str(batch_dir) for batch_dir in input_dirs],
    }

    save_bootstrap_outputs(
        results_summary_df=target_site_results_summary,
        results_raw_df=target_site_results_raw,
        model_summary_df=model_summary_summary,
        model_summary_raw_df=model_summary_raw,
        id_validation_fold_metrics_df=id_validation_raw,
        prediction_records_df=prediction_records_raw,
        split_manifest_df=split_manifest_raw,
        split_assignments_df=split_assignments_raw,
        m2_pairwise_weighting_df=m2_pairwise_raw,
        bootstrap_iterations_df=bootstrap_iterations_df,
        config=config,
        bootstrap_metadata=bootstrap_metadata,
        logger=logger,
    )

    write_bootstrap_markdown_report(
        config=config,
        output_path=output_dir / "reports" / "bootstrap_transportability_report.md",
        bootstrap_metadata=bootstrap_metadata,
        results_summary_df=target_site_results_summary,
        model_summary_df=model_summary_summary,
        split_manifest_df=split_manifest_raw,
    )

    config_snapshot = config.to_snapshot()
    config_snapshot["n_bootstrap"] = n_bootstrap
    config_snapshot["seed_step"] = bootstrap_metadata["seed_step"]
    config_snapshot["bootstrap_seeds"] = bootstrap_seeds
    config_snapshot["merged_from_batches"] = [
        str(batch_dir) for batch_dir in input_dirs
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info("Merged TCGA bootstrap outputs saved under %s", output_dir)


if __name__ == "__main__":
    main()
