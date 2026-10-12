"""Preprocessing: run pipeline."""

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
import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingestion.loaders import load_sources  # noqa: E402
from preprocessing.audit import prepare_analysis_dataset  # noqa: E402
from preprocessing.reporting import build_report  # noqa: E402
from preprocessing.schema import build_feature_schema  # noqa: E402
from preprocessing.splits import build_lodo_manifest  # noqa: E402
from utils.config import PipelineConfig  # noqa: E402
from utils.filesystem import ensure_project_dirs, write_json  # noqa: E402
from utils.logging_utils import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UCI Heart Disease preprocessing.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default_config.json",
        help="Path to the JSON config file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PipelineConfig.from_json(args.config.resolve())
    ensure_project_dirs(config.project_root)

    logger = setup_logger(config.logs_dir / "uci_preprocessing.log")
    logger.info("Starting UCI Heart Disease preprocessing pipeline.")
    logger.info("Using config: %s", args.config.resolve())

    np.random.seed(config.random_seed)
    run_timestamp = datetime.now(tz=ZoneInfo(config.timezone))

    write_json(
        config.config_dir / "resolved_run_config.json", config.to_serializable_dict()
    )

    unified_df, source_manifest, ingestion_warnings = load_sources(
        input_paths=config.input_paths,
        expected_rows_by_domain=config.expected_rows_by_domain,
        min_valid_raw_fraction=config.min_valid_raw_fraction,
    )

    for warning in ingestion_warnings:
        logger.warning(warning)
    logger.info(
        "Loaded %s rows across %s sources.", unified_df.shape[0], len(source_manifest)
    )

    audit_result = prepare_analysis_dataset(
        unified_df=unified_df,
        high_missingness_warning=config.high_missingness_warning,
        min_domain_samples_warning=config.min_domain_samples_warning,
    )
    for warning in audit_result.warnings:
        logger.warning(warning)

    feature_schema = build_feature_schema()
    split_manifest = build_lodo_manifest(
        df=audit_result.analysis_ready_df,
        domain_column=feature_schema["domain_column"],
        target_column=feature_schema["target_column"],
    )

    interim_path = config.interim_dir / "uci_heart_unified.csv"
    analysis_ready_path = config.processed_dir / "uci_heart_analysis_ready.csv"
    schema_path = config.processed_dir / "uci_heart_feature_schema.json"
    splits_path = config.splits_dir / "lodo_domains.json"
    source_manifest_path = config.raw_dir / "source_files_manifest.json"
    ingestion_summary_path = config.interim_dir / "uci_heart_ingestion_summary.json"
    quality_checks_path = config.processed_dir / "uci_heart_quality_checks.json"
    report_path = config.reports_dir / "uci_preprocessing_report.md"

    audit_result.interim_df.to_csv(interim_path, index=False)
    audit_result.analysis_ready_df.to_csv(analysis_ready_path, index=False)
    write_json(schema_path, feature_schema)
    write_json(splits_path, split_manifest)
    write_json(source_manifest_path, source_manifest)
    write_json(
        ingestion_summary_path,
        {
            "n_loaded_rows_before_cleaning": int(unified_df.shape[0]),
            "n_rows_after_cleaning": int(audit_result.analysis_ready_df.shape[0]),
            "ingestion_warnings": ingestion_warnings,
            "removal_log": audit_result.removal_log,
        },
    )
    write_json(quality_checks_path, audit_result.quality_checks)

    output_files = [
        interim_path,
        analysis_ready_path,
        schema_path,
        splits_path,
        source_manifest_path,
        ingestion_summary_path,
        quality_checks_path,
        report_path,
        config.logs_dir / "uci_preprocessing.log",
        config.config_dir / "resolved_run_config.json",
        PROJECT_ROOT / "src" / "preprocessing" / "example_lodo_usage.py",
    ]

    report_text = build_report(
        run_timestamp=run_timestamp,
        config_payload=config.to_serializable_dict(),
        source_manifest=source_manifest,
        ingestion_warnings=ingestion_warnings,
        audit_result=audit_result,
        feature_schema=feature_schema,
        split_manifest=split_manifest,
        output_files=output_files,
    )
    report_path.write_text(report_text, encoding="utf-8")

    logger.info("Wrote interim dataset to %s", interim_path)
    logger.info("Wrote analysis-ready dataset to %s", analysis_ready_path)
    logger.info("Wrote feature schema to %s", schema_path)
    logger.info("Wrote LODO split manifest to %s", splits_path)
    logger.info("Wrote report to %s", report_path)
    logger.info("Pipeline completed successfully.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        logging.exception("Pipeline failed: %s", exc)
        raise
