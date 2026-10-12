"""Transportability experiment: run m0 baseline."""

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

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = _REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uci_transportability.config import ExperimentConfig
from uci_transportability.experiment import run_transportability_experiment
from uci_transportability.reporting import save_outputs, write_markdown_report


def build_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("uci_transportability_m0_m1")
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
        description="UCI Heart Disease cross-site transportability pipeline (M0-M4)"
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
        default=_REPOSITORY_ROOT / "results" / "UCI" / "Methods" / "baseline_outputs",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
        help="Comma-separated list of enabled methods (supported here: M0,M1,M2,M3,M4)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_config(args.output_dir.resolve())

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

    logger = build_logger(config.output_dir / "logs" / "run.log")
    logger.info("Starting UCI Heart Disease transportability experiment")
    logger.info("Input dataset: %s", config.input_path)
    logger.info("Feature schema: %s", config.schema_path)
    logger.info("Output directory: %s", config.output_dir)
    logger.info("Enabled methods: %s", ", ".join(config.enabled_methods))

    outputs = run_transportability_experiment(config=config, logger=logger)
    save_outputs(outputs=outputs, config=config, logger=logger)
    write_markdown_report(
        outputs=outputs,
        config=config,
        output_path=config.output_dir / "reports" / "transportability_report.md",
    )

    config_snapshot = config.to_snapshot()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config_snapshot.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info("Finished UCI Heart Disease transportability experiment")


if __name__ == "__main__":
    main()
