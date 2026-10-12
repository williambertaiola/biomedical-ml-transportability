"""Transportability experiment: run uci bootstrap resampling."""

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
import sys
import tempfile
from pathlib import Path

import pandas as pd

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = _REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uci_transportability.bootstrap_reporting import (
    aggregate_model_summary,
    aggregate_target_site_results,
    save_bootstrap_outputs,
    write_bootstrap_markdown_report,
)
from uci_transportability.config import ExperimentConfig
from uci_transportability.experiment import run_transportability_experiment


def build_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("uci_transportability_bootstrap")
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


def default_config(output_dir: Path) -> ExperimentConfig:
    """Resolve the original preprocessed input through the portable path registry."""
    return ExperimentConfig(
        input_path=Path(
            _configured_path(
                "results_data_uci_heart_preprocessing_data_processed_uci_heart_analysis_ready.csv"
            )
        ),
        schema_path=Path(
            _configured_path(
                "results_data_uci_heart_preprocessing_data_processed_uci_heart_feature_schema.json"
            )
        ),
        output_dir=output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UCI Heart Disease bootstrap/repeated resampling pipeline"
    )
    parser.add_argument(
        "--input", type=Path, default=None, help="Path to the analysis-ready UCI CSV"
    )
    parser.add_argument(
        "--schema", type=Path, default=None, help="Path to the UCI feature schema JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPOSITORY_ROOT / "results" / "UCI" / "Methods" / "bootstrap_outputs",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Increment applied to the seed at each bootstrap",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1,
        help="Number of bootstrap/repeated-split iterations",
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        help="Number of bins for Expected Calibration Error",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="n_jobs for Random Forest"
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="M0,M1,M2,M3,M4",
        help="Comma-separated list of enabled methods",
    )
    return parser.parse_args()


def _append_bootstrap_columns(df, *, bootstrap_index: int, bootstrap_seed: int):
    if df is None or df.empty:
        return df
    enriched = df.copy()
    enriched.insert(0, "bootstrap_index", bootstrap_index)
    enriched.insert(1, "bootstrap_seed", bootstrap_seed)
    return enriched


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    config = default_config(output_dir)

    if args.input is not None:
        config.input_path = args.input.resolve()
    if args.schema is not None:
        config.schema_path = args.schema.resolve()
    config.random_seed = int(args.seed)
    config.ece_bins = int(args.ece_bins)
    config.n_jobs = int(args.n_jobs)
    requested_methods = tuple(
        method.strip().upper()
        for method in str(args.methods).split(",")
        if method.strip()
    )
    supported_methods = {"M0", "M1", "M2", "M3", "M4"}
    unsupported_methods = sorted(set(requested_methods).difference(supported_methods))
    if unsupported_methods:
        raise ValueError(f"Unsupported methods for this runner: {unsupported_methods}")
    if not requested_methods:
        raise ValueError("At least one method must be provided via --methods.")
    config.enabled_methods = requested_methods

    logger = build_logger(output_dir / "logs" / "run.log")
    logger.info("Starting UCI bootstrap transportability experiment")
    logger.info("Input dataset: %s", config.input_path)
    logger.info("Feature schema: %s", config.schema_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("Enabled methods: %s", ", ".join(config.enabled_methods))
    logger.info(
        "n_bootstrap=%s | base_seed=%s | seed_step=%s",
        args.n_bootstrap,
        args.seed,
        args.seed_step,
    )

    target_site_results_tables = []
    model_summary_tables = []
    id_validation_tables = []
    prediction_tables = []
    split_manifest_tables = []
    split_assignment_tables = []
    m2_pairwise_tables = []
    bootstrap_iterations: list[dict[str, int]] = []

    for bootstrap_index in range(1, int(args.n_bootstrap) + 1):
        bootstrap_seed = int(args.seed) + (bootstrap_index - 1) * int(args.seed_step)
        logger.info(
            "Running bootstrap iteration %s/%s with seed %s",
            bootstrap_index,
            args.n_bootstrap,
            bootstrap_seed,
        )

        iteration_config = default_config(output_dir)
        iteration_config.input_path = config.input_path
        iteration_config.schema_path = config.schema_path
        iteration_config.dataset_name = config.dataset_name
        iteration_config.enabled_methods = config.enabled_methods
        iteration_config.target_column = config.target_column
        iteration_config.domain_column = config.domain_column
        iteration_config.id_column = config.id_column
        iteration_config.target_test_size = config.target_test_size
        iteration_config.random_seed = bootstrap_seed
        iteration_config.ece_bins = config.ece_bins
        iteration_config.n_jobs = config.n_jobs
        iteration_config.threshold = config.threshold
        iteration_config.combat_continuous_covariates = tuple(
            config.combat_continuous_covariates
        )
        iteration_config.combat_categorical_covariates = tuple(
            config.combat_categorical_covariates
        )
        iteration_config.combat_low_cardinality_threshold = (
            config.combat_low_cardinality_threshold
        )
        iteration_config.m2_weight_temperature = config.m2_weight_temperature
        iteration_config.m2_min_domain_weight = config.m2_min_domain_weight
        iteration_config.m3_target_blend_weight = config.m3_target_blend_weight

        outputs = run_transportability_experiment(
            config=iteration_config, logger=logger
        )
        target_site_results_tables.append(
            _append_bootstrap_columns(
                outputs.target_site_results,
                bootstrap_index=bootstrap_index,
                bootstrap_seed=bootstrap_seed,
            )
        )
        model_summary_tables.append(
            _append_bootstrap_columns(
                outputs.model_summary,
                bootstrap_index=bootstrap_index,
                bootstrap_seed=bootstrap_seed,
            )
        )
        id_validation_tables.append(
            _append_bootstrap_columns(
                outputs.id_validation_fold_metrics,
                bootstrap_index=bootstrap_index,
                bootstrap_seed=bootstrap_seed,
            )
        )
        prediction_tables.append(
            _append_bootstrap_columns(
                outputs.prediction_records,
                bootstrap_index=bootstrap_index,
                bootstrap_seed=bootstrap_seed,
            )
        )
        split_manifest_tables.append(
            _append_bootstrap_columns(
                outputs.split_manifest,
                bootstrap_index=bootstrap_index,
                bootstrap_seed=bootstrap_seed,
            )
        )
        split_assignment_tables.append(
            _append_bootstrap_columns(
                outputs.split_assignments,
                bootstrap_index=bootstrap_index,
                bootstrap_seed=bootstrap_seed,
            )
        )
        m2_pairwise_tables.append(
            _append_bootstrap_columns(
                outputs.m2_pairwise_weighting,
                bootstrap_index=bootstrap_index,
                bootstrap_seed=bootstrap_seed,
            )
        )
        bootstrap_iterations.append(
            {
                "bootstrap_index": bootstrap_index,
                "bootstrap_seed": bootstrap_seed,
            }
        )

    target_site_results_raw = (
        pd.concat(target_site_results_tables, ignore_index=True)
        if target_site_results_tables
        else pd.DataFrame()
    )
    model_summary_raw = (
        pd.concat(model_summary_tables, ignore_index=True)
        if model_summary_tables
        else pd.DataFrame()
    )
    id_validation_raw = (
        pd.concat(id_validation_tables, ignore_index=True)
        if id_validation_tables
        else pd.DataFrame()
    )
    prediction_records_raw = (
        pd.concat(prediction_tables, ignore_index=True)
        if prediction_tables
        else pd.DataFrame()
    )
    split_manifest_raw = (
        pd.concat(split_manifest_tables, ignore_index=True)
        if split_manifest_tables
        else pd.DataFrame()
    )
    split_assignments_raw = (
        pd.concat(split_assignment_tables, ignore_index=True)
        if split_assignment_tables
        else pd.DataFrame()
    )
    m2_pairwise_raw = (
        pd.concat(m2_pairwise_tables, ignore_index=True)
        if m2_pairwise_tables
        else pd.DataFrame()
    )

    target_site_results_summary = aggregate_target_site_results(target_site_results_raw)
    model_summary_summary = aggregate_model_summary(model_summary_raw)
    bootstrap_iterations_df = pd.DataFrame(bootstrap_iterations)
    bootstrap_metadata = {
        "dataset_name": config.dataset_name,
        "n_bootstrap": int(args.n_bootstrap),
        "base_seed": int(args.seed),
        "seed_step": int(args.seed_step),
        "bootstrap_seeds": [row["bootstrap_seed"] for row in bootstrap_iterations],
        "iterations": bootstrap_iterations,
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
    config_snapshot["n_bootstrap"] = int(args.n_bootstrap)
    config_snapshot["seed_step"] = int(args.seed_step)
    config_snapshot["bootstrap_seeds"] = [
        row["bootstrap_seed"] for row in bootstrap_iterations
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info("Finished UCI bootstrap transportability experiment")


if __name__ == "__main__":
    main()
