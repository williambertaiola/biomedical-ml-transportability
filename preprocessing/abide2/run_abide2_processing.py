"""Preprocessing: run abide2 processing."""

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

from src.abide2_processing import (
    file_manifest,
    init_logger,
    run_processing,
    save_dataframe,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ABIDE II BNI subject-level preprocessing."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "abide2_processing_config.json",
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

    logger = init_logger(output_root / "logs" / "abide2_processing.log")
    logger.info("Starting ABIDE II preprocessing")

    manifest = {
        "phenotypic_file": file_manifest(Path(config["phenotypic_file"])),
        "qap_dir": file_manifest(Path(config["qap_dir"])),
        "config_path": str(args.config),
    }
    write_json(manifest, output_root / "data" / "raw" / "input_manifest.json")

    artifacts = run_processing(config, logger)

    final_path = save_dataframe(
        artifacts.final_df,
        output_root / "data" / "processed" / "abide2_bni_subject_level_database.csv",
        index=False,
    )
    save_dataframe(
        artifacts.file_overview,
        output_root / "reports" / "tables" / "file_overview.csv",
        index=False,
    )
    save_dataframe(
        artifacts.qap_coverage,
        output_root / "reports" / "tables" / "qap_coverage.csv",
        index=False,
    )
    save_dataframe(
        artifacts.missingness,
        output_root / "reports" / "tables" / "column_missingness.csv",
        index=False,
    )
    save_dataframe(
        artifacts.dropped_columns,
        output_root / "reports" / "tables" / "dropped_columns.csv",
        index=False,
    )

    report_path = output_root / "reports" / "abide2_processing_report.md"
    report_path.write_text(artifacts.report_text, encoding="utf-8")

    logger.info("Saved final database to %s", final_path)
    logger.info("Saved report to %s", report_path)
    logger.info("Finished ABIDE II preprocessing")


if __name__ == "__main__":
    main()
