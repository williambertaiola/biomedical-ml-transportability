"""Transportability experiment: merge abide bootstrap batch outputs."""

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

from abide_transportability.bootstrap_reporting import (
    aggregate_model_summary,
    aggregate_target_site_results,
    save_bootstrap_outputs,
    write_bootstrap_markdown_report,
)
from abide_transportability.config import ExperimentConfig


def build_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("abide_transportability_bootstrap_merge")
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
        description="Merge batched ABIDE bootstrap outputs into a single final package"
    )
    parser.add_argument(
        "--input-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more batch output directories produced by run_abide_bootstrap_resampling.py",
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
        schema_path=Path(snapshot["schema_path"]),
        output_dir=output_dir,
    )
    config.dataset_name = snapshot.get("dataset_name", config.dataset_name)
    config.enabled_methods = tuple(
        snapshot.get("enabled_methods", config.enabled_methods)
    )
    config.target_sites = tuple()
    config.target_column = snapshot.get("target_column", config.target_column)
    config.domain_column = snapshot.get("domain_column", config.domain_column)
    config.id_column = snapshot.get("id_column", config.id_column)
    config.target_test_size = float(
        snapshot.get("target_test_size", config.target_test_size)
    )
    config.random_seed = int(snapshot.get("random_seed", config.random_seed))
    config.ece_bins = int(snapshot.get("ece_bins", config.ece_bins))
    config.n_jobs = int(snapshot.get("n_jobs", config.n_jobs))
    config.threshold = float(snapshot.get("threshold", config.threshold))
    config.min_target_class_count = int(
        snapshot.get("min_target_class_count", config.min_target_class_count)
    )
    config.combat_continuous_covariates = tuple(
        snapshot.get(
            "combat_continuous_covariates", config.combat_continuous_covariates
        )
    )
    config.combat_categorical_covariates = tuple(
        snapshot.get(
            "combat_categorical_covariates", config.combat_categorical_covariates
        )
    )
    config.combat_low_cardinality_threshold = int(
        snapshot.get(
            "combat_low_cardinality_threshold", config.combat_low_cardinality_threshold
        )
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


def main() -> None:
    args = parse_args()
    input_dirs = [path.resolve() for path in args.input_dirs]
    output_dir = args.output_dir.resolve()

    if not input_dirs:
        raise ValueError("At least one batch output directory is required.")

    logger = build_logger(output_dir / "logs" / "merge.log")
    logger.info("Merging %s ABIDE bootstrap batch directories", len(input_dirs))

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
    model_summary_raw = pd.concat(
        [
            _load_csv(batch_dir, "tables/model_summary_bootstrap_raw.csv")
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
    model_summary_summary = aggregate_model_summary(model_summary_raw)

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

    logger.info("Merged ABIDE bootstrap outputs saved under %s", output_dir)


if __name__ == "__main__":
    main()
