"""Preprocessing: run abide2 transportability rich."""

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

from src.abide2_core_transportability import init_logger, save_dataframe, write_json
from src.abide2_transportability_rich import run_rich_transportability_processing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strict and rich ABIDE II transportability datasets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "abide2_transportability_rich_config.json",
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
    master_path = Path(config["master_table_path"])

    logger = init_logger(output_root / "logs" / "abide2_transportability_rich.log")
    logger.info("Starting ABIDE II strict+rich transportability dataset build")

    artifacts = run_rich_transportability_processing(master_path, logger)

    rich_path = save_dataframe(
        artifacts.rich_df,
        output_root
        / "data"
        / "processed"
        / "abide2_core_multisite_transportability_dataset.csv",
        index=False,
    )
    strict_path = save_dataframe(
        artifacts.strict_df,
        output_root
        / "data"
        / "processed"
        / "abide2_core_multisite_transportability_dataset_strict.csv",
        index=False,
    )
    save_dataframe(
        artifacts.rich_feature_manifest,
        output_root
        / "reports"
        / "tables"
        / "abide2_core_multisite_feature_manifest.csv",
        index=False,
    )
    save_dataframe(
        artifacts.rich_row_missingness_summary,
        output_root
        / "reports"
        / "tables"
        / "abide2_core_multisite_row_missingness_summary.csv",
        index=False,
    )
    save_dataframe(
        artifacts.rich_site_completeness,
        output_root
        / "reports"
        / "tables"
        / "abide2_core_multisite_site_completeness.csv",
        index=False,
    )
    write_json(
        artifacts.blueprint,
        output_root
        / "data"
        / "processed"
        / "abide2_core_multisite_preprocessing_blueprint.json",
    )

    report_path = (
        output_root / "reports" / "abide2_core_multisite_transportability_report.md"
    )
    report_path.write_text(artifacts.report_text, encoding="utf-8")

    logger.info("Saved rich dataset to %s", rich_path)
    logger.info("Saved strict dataset to %s", strict_path)
    logger.info("Saved report to %s", report_path)
    logger.info("Finished ABIDE II strict+rich transportability dataset build")


if __name__ == "__main__":
    main()
