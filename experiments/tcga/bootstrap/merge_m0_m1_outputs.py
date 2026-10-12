"""Transportability experiment: merge m0 m1 outputs."""

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
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = _REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from tcga_transportability.config import ExperimentConfig
from tcga_transportability.data import load_dataset
from tcga_transportability.experiment import (
    ExperimentOutputs,
    _build_model_summary,
    retune_metrics_from_saved_predictions,
)
from tcga_transportability.reporting import save_outputs, write_markdown_report


BATCH_ROOT = Path(_configured_path("results_tcga_methods__m0_m1_target_batches_v2"))
FINAL_OUTPUT_DIR = Path(
    _configured_path("results_tcga_methods_tcga_brca_m1_combat_outputs")
)
INPUT_CSV = Path(
    _configured_path(
        "results_data_tcga_brca_preprocessing_data_processed_tcga_brca_stage_transportability_dataset.csv"
    )
)


def build_logger() -> logging.Logger:
    logger = logging.getLogger("tcga_merge_m0_m1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(logging.StreamHandler())
    return logger


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return pd.read_csv(path)


def load_batch_frames(batch_root: Path) -> dict[str, pd.DataFrame]:
    batch_dirs = sorted(path for path in batch_root.iterdir() if path.is_dir())
    if not batch_dirs:
        raise FileNotFoundError(f"No M1 batch directories found under {batch_root}")

    table_names = [
        "target_site_results.csv",
        "id_validation_fold_metrics.csv",
        "prediction_records.csv",
        "target_site_split_manifest.csv",
        "target_site_split_assignments.csv",
    ]
    combined: dict[str, list[pd.DataFrame]] = {name: [] for name in table_names}
    for batch_dir in batch_dirs:
        tables_dir = batch_dir / "tables"
        for name in table_names:
            combined[name].append(_read_csv(tables_dir / name))

    return {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in combined.items()
    }


def main() -> None:
    logger = build_logger()
    config = ExperimentConfig(
        input_path=INPUT_CSV,
        output_dir=FINAL_OUTPUT_DIR,
        enabled_methods=("M0", "M1"),
    )

    dataset_df, schema = load_dataset(config, logger)
    frames = load_batch_frames(BATCH_ROOT)
    combined_results = (
        frames["target_site_results.csv"]
        .sort_values(["method", "model", "target_site"])
        .reset_index(drop=True)
    )
    combined_folds = (
        frames["id_validation_fold_metrics.csv"]
        .sort_values(["method", "model", "evaluation_target_site", "fold_index"])
        .reset_index(drop=True)
    )
    combined_predictions = frames["prediction_records.csv"].reset_index(drop=True)
    manifest_sorted = (
        frames["target_site_split_manifest.csv"]
        .drop_duplicates(subset=["evaluation_target_site"])
        .sort_values("evaluation_target_site")
        .reset_index(drop=True)
    )
    assignments_sorted = (
        frames["target_site_split_assignments.csv"]
        .drop_duplicates(subset=["evaluation_target_site", "partition", "subject_id"])
        .sort_values(["evaluation_target_site", "partition", "subject_id"])
        .reset_index(drop=True)
    )

    expected_sites = sorted(
        dataset_df[config.domain_column].dropna().astype(str).unique().tolist()
    )
    observed_sites = sorted(
        combined_results["target_site"].dropna().astype(str).unique().tolist()
    )
    if observed_sites != expected_sites:
        raise ValueError(
            f"Batch target-site coverage mismatch. Expected {expected_sites}, got {observed_sites}"
        )

    observed_methods = sorted(
        combined_results["method"].dropna().astype(str).unique().tolist()
    )
    if observed_methods != ["M0", "M1"]:
        raise ValueError(
            f"Expected merged methods ['M0', 'M1'], got {observed_methods}"
        )

    retuned_results, retuned_folds, retuned_predictions = (
        retune_metrics_from_saved_predictions(
            target_site_results=combined_results,
            id_validation_fold_metrics=combined_folds,
            prediction_records=combined_predictions,
            config=config,
        )
    )

    outputs = ExperimentOutputs(
        dataset_df=dataset_df,
        schema=schema,
        target_site_results=retuned_results,
        id_validation_fold_metrics=retuned_folds,
        prediction_records=retuned_predictions,
        split_manifest=manifest_sorted,
        split_assignments=assignments_sorted,
        model_summary=_build_model_summary(retuned_results),
        m2_pairwise_weighting=pd.DataFrame(),
    )

    save_outputs(outputs=outputs, config=config, logger=logger)
    write_markdown_report(
        outputs=outputs,
        config=config,
        output_path=FINAL_OUTPUT_DIR / "reports" / "transportability_report.md",
    )
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_OUTPUT_DIR / "config_snapshot.json").write_text(
        json.dumps(config.to_snapshot(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Merged M0 + M1 batch outputs under %s", FINAL_OUTPUT_DIR)


if __name__ == "__main__":
    main()
