"""Preprocessing: run tcga brca initial audit."""

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
import gc
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit.barcodes import build_sample_metadata
from src.audit.clinical import load_subject_level_clinical, scan_metadata_inventory
from src.audit.reporting import render_audit_report
from src.audit.summaries import compute_expression_qc, create_audit_tables_and_figures
from src.ingestion.expression import load_and_standardize_expression
from src.preprocessing.blueprint import (
    build_downstream_metadata_table,
    build_preprocessing_blueprint,
)
from src.utils.io import file_manifest, init_logger, save_dataframe, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initial exploratory audit for TCGA-BRCA expression data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tcga_brca_audit_config.json",
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return resolve_config_paths(json.load(handle))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config.get("seed", 42))
    np.random.seed(seed)

    output_root = Path(config["output_root"])
    expression_path = Path(config["expression_matrix"])
    clinical_dir = Path(config["clinical_dir"]) if config.get("clinical_dir") else None

    logger = init_logger(output_root / "logs" / "tcga_brca_initial_audit.log")
    logger.info("Starting TCGA-BRCA initial audit")

    manifest_payload = {
        "expression_matrix": file_manifest(expression_path),
        "clinical_dir": file_manifest(clinical_dir)
        if clinical_dir
        else {"exists": False},
        "config_path": str(args.config),
        "seed": seed,
    }
    write_json(manifest_payload, output_root / "data" / "raw" / "input_manifest.json")

    expression_result = load_and_standardize_expression(
        expression_path, logger, seed=seed
    )
    standardized_expression = expression_result.standardized_df.copy()
    standardized_expression_export = standardized_expression.reset_index()
    standardized_expression_path = save_dataframe(
        standardized_expression_export,
        output_root / "data" / "interim" / "tcga_brca_expression_standardized.csv",
        index=False,
    )
    logger.info(
        "Saved standardized expression matrix to %s", standardized_expression_path
    )

    clinical_subject_df, clinical_summary = load_subject_level_clinical(clinical_dir)
    metadata_inventory = scan_metadata_inventory(clinical_dir)

    samples_df, domain_field_summary = build_sample_metadata(
        list(standardized_expression.index), clinical_subject_df
    )
    logger.info(
        "Primary domain source: %s",
        samples_df["domain_source"].iloc[0] if not samples_df.empty else "none",
    )

    sample_qc, gene_qc, qc_summary = compute_expression_qc(standardized_expression)
    samples_df = samples_df.merge(sample_qc, how="left", on="sample_barcode")

    standardized_samples_path = save_dataframe(
        samples_df.sort_values(["sample_type_code", "subject_id", "sample_barcode"]),
        output_root / "data" / "interim" / "tcga_brca_samples_standardized.csv",
        index=False,
    )
    logger.info("Saved standardized sample metadata to %s", standardized_samples_path)

    del standardized_expression
    del standardized_expression_export
    gc.collect()

    audit_artifacts = create_audit_tables_and_figures(
        samples_df,
        sample_qc,
        gene_qc,
        qc_summary,
        output_root / "reports" / "tables",
        output_root / "reports" / "figures",
        value_type_guess=str(expression_result.value_profile["value_type_guess"]),
    )

    blueprint = build_preprocessing_blueprint(
        samples_df, audit_artifacts.summary_metrics, expression_result.value_profile
    )
    downstream_metadata = build_downstream_metadata_table(samples_df, blueprint)
    downstream_metadata_path = save_dataframe(
        downstream_metadata,
        output_root
        / "data"
        / "processed"
        / "tcga_brca_downstream_metadata_blueprint.csv",
        index=False,
    )
    logger.info("Saved downstream metadata blueprint to %s", downstream_metadata_path)

    report_text = render_audit_report(
        expression_result=expression_result,
        samples_df=samples_df,
        clinical_summary=clinical_summary,
        metadata_inventory=metadata_inventory,
        audit_artifacts=audit_artifacts,
        blueprint=blueprint,
    )
    report_path = output_root / "reports" / "tcga_brca_initial_audit.md"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved Markdown audit report to %s", report_path)

    blueprint_payload = {
        "domain_field_summary": domain_field_summary,
        "blueprint": blueprint,
        "expression_value_profile": expression_result.value_profile,
        "audit_summary_metrics": audit_artifacts.summary_metrics,
    }
    write_json(
        blueprint_payload,
        output_root / "data" / "processed" / "tcga_brca_preprocessing_blueprint.json",
    )
    logger.info("Finished TCGA-BRCA initial audit")


if __name__ == "__main__":
    main()
