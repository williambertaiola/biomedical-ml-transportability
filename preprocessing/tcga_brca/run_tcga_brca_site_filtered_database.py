"""Preprocessing: run tcga brca site filtered database."""

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

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.site_filtered_database import (
    build_subject_gene_database,
    determine_kept_sites,
    filter_genes,
    load_expression_subset,
    render_site_filtered_report,
    save_site_filtered_outputs,
)
from src.utils.io import init_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a TCGA-BRCA site-filtered subject-gene database."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "tcga_brca_site_filtered_process_config.json",
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
    logger = init_logger(
        output_root / "logs" / "tcga_brca_site_filtered_processing.log"
    )

    samples_df = pd.read_csv(
        config["input_samples_metadata"],
        dtype={"sample_type_code": str, "tss": str},
    )
    logger.info("Loaded standardized sample metadata: %s rows", len(samples_df))

    site_filter_result = determine_kept_sites(
        samples_df,
        allowed_tumor_levels=config["allowed_tumor_levels"],
        site_anchor=config["site_anchor"],
    )
    logger.info(
        "Kept %s sites with threshold >= %s samples",
        len(site_filter_result.kept_sites),
        site_filter_result.site_threshold,
    )

    filtered_expression = load_expression_subset(
        Path(config["input_expression_matrix"]),
        keep_barcodes=site_filter_result.filtered_samples["sample_barcode"].tolist(),
    )
    logger.info(
        "Loaded expression subset: %s samples x %s genes", *filtered_expression.shape
    )

    gene_filter_result = filter_genes(
        filtered_expression,
        min_detected_samples=int(config["min_detected_samples"]),
        min_variance=float(config["min_variance"]),
    )
    logger.info(
        "Gene filtering completed: %s -> %s genes",
        gene_filter_result.summary["genes_before_filtering"],
        gene_filter_result.summary["genes_after_filtering"],
    )

    subject_gene_database = build_subject_gene_database(
        site_filter_result.filtered_samples,
        gene_filter_result.filtered_expression,
    )
    logger.info(
        "Final subject-gene database shape: %s rows x %s columns",
        *subject_gene_database.shape,
    )

    saved_paths = save_site_filtered_outputs(
        output_root=output_root,
        site_filter_result=site_filter_result,
        gene_filter_result=gene_filter_result,
        subject_gene_database=subject_gene_database,
    )

    report_text = render_site_filtered_report(
        site_filter_result=site_filter_result,
        gene_filter_result=gene_filter_result,
        subject_gene_database=subject_gene_database,
        saved_paths=saved_paths,
    )
    report_path = output_root / "reports" / "tcga_brca_site_filtered_processing.md"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved processing report to %s", report_path)


if __name__ == "__main__":
    main()
