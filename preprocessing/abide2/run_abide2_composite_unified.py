"""Preprocessing: run abide2 composite unified."""

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
from thesis_tools.paths import resolve_config_paths


import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.abide2_composite_unified import (
    file_manifest,
    init_logger,
    run_composite_unified_processing,
    save_dataframe,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a composite-enriched ABIDE II unified subject database."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "abide2_composite_unified_config.json",
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return resolve_config_paths(json.load(handle))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = Path(config["output_root"])
    input_root = Path(config["input_root"])
    composite_path = Path(config["composite_phenotypic_path"])
    long_path = Path(config["long_phenotypic_path"])

    logger = init_logger(output_root / "logs" / "abide2_composite_unified.log")
    logger.info("Starting composite-enriched ABIDE II unified processing")

    manifest = {
        "input_root": file_manifest(input_root),
        "qap_root": file_manifest(input_root / "ABIDEII_MRI_Quality_Metrics"),
        "raw_bni_root": file_manifest(input_root / "ABIDEII-BNI_1"),
        "composite_phenotypic_path": file_manifest(composite_path),
        "long_phenotypic_path": file_manifest(long_path),
        "config_path": str(args.config),
    }
    write_json(manifest, output_root / "data" / "raw" / "composite_input_manifest.json")

    artifacts = run_composite_unified_processing(config, logger)

    final_path = save_dataframe(
        artifacts.final_df,
        output_root
        / "data"
        / "processed"
        / "abide2_composite_subject_level_unified_database.csv",
        index=False,
    )
    save_dataframe(
        artifacts.source_coverage,
        output_root / "reports" / "tables" / "source_coverage_composite.csv",
        index=False,
    )
    save_dataframe(
        artifacts.site_summary,
        output_root / "reports" / "tables" / "site_summary_composite.csv",
        index=False,
    )
    save_dataframe(
        artifacts.disease_summary,
        output_root / "reports" / "tables" / "disease_summary_composite.csv",
        index=False,
    )
    save_dataframe(
        artifacts.missingness,
        output_root / "reports" / "tables" / "column_missingness_composite_unified.csv",
        index=False,
    )
    save_dataframe(
        artifacts.dropped_columns,
        output_root / "reports" / "tables" / "dropped_columns_composite_unified.csv",
        index=False,
    )
    save_dataframe(
        artifacts.file_inventory,
        output_root / "reports" / "tables" / "folder_inventory.csv",
        index=False,
    )
    save_dataframe(
        artifacts.site_aliases,
        output_root / "reports" / "tables" / "site_aliases_composite.csv",
        index=False,
    )

    report_path = output_root / "reports" / "abide2_composite_unified_report.md"
    report_path.write_text(artifacts.report_text, encoding="utf-8")

    logger.info("Saved composite-enriched unified database to %s", final_path)
    logger.info("Saved composite-enriched report to %s", report_path)
    logger.info("Finished composite-enriched ABIDE II unified processing")


if __name__ == "__main__":
    main()
