"""Transportability experiment: run tcga baseline."""

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

from tcga_transportability.config import ExperimentConfig
from tcga_transportability.experiment import run_transportability_experiment
from tcga_transportability.reporting import save_outputs, write_markdown_report


def build_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tcga_transportability")
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
                "results_data_tcga_brca_preprocessing_data_processed_tcga_brca_stage_transportability_dataset.csv"
            )
        ),
        output_dir=output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TCGA-BRCA cross-site transportability pipeline (M0/M1/M2/M3/M4)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to the TCGA-BRCA stage transportability CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPOSITORY_ROOT / "results" / "TCGA" / "Methods" / "baseline_outputs",
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
        "--combat-top-k-variance-genes",
        type=int,
        default=500,
        help="Train-only top-variance gene pool size used before ComBat harmonization",
    )
    parser.add_argument(
        "--top-k-variance-genes",
        type=int,
        default=2000,
        help="Training-only variance prefilter size applied before model selection",
    )
    parser.add_argument(
        "--top-k-model-genes",
        type=int,
        default=500,
        help="Training-only ANOVA feature selection size applied before the model",
    )
    parser.add_argument(
        "--m2-weight-temperature",
        type=float,
        default=0.12,
        help="Softmax temperature used to turn M2 pairwise transfer scores into source-domain weights",
    )
    parser.add_argument(
        "--m2-min-domain-weight",
        type=float,
        default=0.25,
        help="Minimum source-domain weight enforced before M2 renormalization",
    )
    parser.add_argument(
        "--m3-target-blend-weight",
        type=float,
        default=0.5,
        help="Blend weight assigned to the target train_1 adapter in M3 probability blending",
    )
    parser.add_argument(
        "--min-site-samples",
        type=int,
        default=30,
        help="Minimum number of samples required for a site to be included",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="M0",
        help="Comma-separated list of enabled methods (currently supported: M0, M1, M2, M3, M4)",
    )
    parser.add_argument(
        "--target-sites",
        type=str,
        default="",
        help="Optional comma-separated subset of target sites to evaluate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_config(args.output_dir.resolve())

    if args.input is not None:
        config.input_path = args.input.resolve()
    config.random_seed = int(args.seed)
    config.ece_bins = int(args.ece_bins)
    config.n_jobs = int(args.n_jobs)
    config.combat_top_k_variance_genes = int(args.combat_top_k_variance_genes)
    config.top_k_variance_genes = int(args.top_k_variance_genes)
    config.top_k_model_genes = int(args.top_k_model_genes)
    config.m2_weight_temperature = float(args.m2_weight_temperature)
    config.m2_min_domain_weight = float(args.m2_min_domain_weight)
    config.m3_target_blend_weight = float(args.m3_target_blend_weight)
    config.min_site_samples = int(args.min_site_samples)

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
    requested_target_sites = tuple(
        site.strip() for site in str(args.target_sites).split(",") if site.strip()
    )
    config.target_sites = requested_target_sites

    logger = build_logger(config.output_dir / "logs" / "run.log")
    logger.info("Starting TCGA-BRCA transportability experiment")
    logger.info("Input dataset: %s", config.input_path)
    logger.info("Output directory: %s", config.output_dir)
    logger.info("Enabled methods: %s", ", ".join(config.enabled_methods))
    if config.target_sites:
        logger.info("Restricted target sites: %s", ", ".join(config.target_sites))

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

    logger.info("Finished TCGA-BRCA transportability experiment")


if __name__ == "__main__":
    main()
