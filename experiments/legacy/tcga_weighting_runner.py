"""Transportability experiment: tcga weighting runner."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from weighting_pipeline import (
    RunConfig,
    log_final_console_summary,
    resolve_input_path,
    run_lodo_experiment,
    save_core_outputs,
    save_run_metadata,
)
from weighting_utils import (
    DEFAULT_DISTANCE_CMAP,
    DEFAULT_PERFORMANCE_CMAP,
    DatasetSchema,
    audit_payload_from_dataframe,
    binarize_target,
    compute_domain_distance_matrix,
    compute_pairwise_transfer,
    copy_source_files,
    ensure_directory,
    frame_to_text_block,
    init_logger,
    plot_domain_sample_distribution,
    plot_heatmap_from_matrix,
    plot_metric_boxplot,
    plot_metric_mean_bars,
    plot_primary_metric_by_domain,
    plot_similarity_weights,
    prepare_modeling_dataframe,
    read_table,
    schema_to_dataframe,
    write_report,
    aggregate_metric_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TCGA BRCA weighting pipeline with the same LODO protocol used for UCI and ABIDE II."
    )
    parser.add_argument(
        "--input", required=True, help="Path to the TCGA preprocessed file or folder."
    )
    parser.add_argument(
        "--output", required=True, help="Directory where all results will be saved."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--similarity-alpha",
        type=float,
        default=2.0,
        help="Alpha for similarity-based weighting.",
    )
    return parser


def build_tcga_schema(df: pd.DataFrame) -> DatasetSchema:
    target_candidates = [
        column
        for column in ["sample_type_label", "tumor_level"]
        if column in df.columns
    ]
    if not target_candidates:
        raise ValueError(
            "TCGA input does not contain `sample_type_label` or `tumor_level`."
        )
    target_col = (
        "sample_type_label"
        if "sample_type_label" in df.columns
        else target_candidates[0]
    )

    if "site" not in df.columns:
        raise ValueError(
            "TCGA input does not contain the expected domain column `site`."
        )

    binary_target, target_mapping, target_note, target_binarized = binarize_target(
        df[target_col],
        full_df=df,
        column_name=target_col,
    )
    if binary_target.isna().any():
        raise ValueError(
            "Target binarization produced missing values, so the TCGA workflow cannot continue safely."
        )

    gene_cols = [column for column in df.columns if column.startswith("ENSG")]
    if not gene_cols:
        raise ValueError(
            "No ENSG gene-expression columns were found in the TCGA input."
        )

    id_cols = [
        column for column in ["sample_barcode", "subject_id"] if column in df.columns
    ]
    leakage_exclude_cols = [
        column
        for column in ["tumor_level", "domain_primary", "domain_source"]
        if column in df.columns and column != target_col
    ]

    return DatasetSchema(
        target_col=target_col,
        domain_col="site",
        id_cols=id_cols,
        leakage_exclude_cols=leakage_exclude_cols,
        numeric_candidates=gene_cols,
        categorical_candidates=[],
        feature_candidates=gene_cols,
        target_mapping=target_mapping,
        target_note=target_note,
        target_binarized=target_binarized,
    )


def main() -> None:
    args = build_parser().parse_args()
    resolved_input_path = resolve_input_path(args.input)
    config = RunConfig(
        input_path=str(resolved_input_path),
        output_dir=args.output,
        random_seed=args.seed,
        similarity_alpha=args.similarity_alpha,
        primary_metric="balanced_accuracy",
        model_name="LogisticRegression",
    )

    output_dir = ensure_directory(config.output_dir)
    logger = init_logger(output_dir / "execution_log.txt")
    logger.info("Starting TCGA weighting runner.")
    logger.info("Input dataset: %s", Path(config.input_path).resolve())
    logger.info("Output directory: %s", output_dir.resolve())

    raw_df = read_table(config.input_path)
    logger.info("Loaded raw TCGA dataset with shape %s", raw_df.shape)

    schema = build_tcga_schema(raw_df)
    clean_df = prepare_modeling_dataframe(raw_df, schema, logger)
    feature_schema_df = schema_to_dataframe(schema)
    audit_payload, missing_df, target_distribution_df, domain_distribution_df = (
        audit_payload_from_dataframe(
            raw_df,
            clean_df,
            schema,
            config.input_path,
        )
    )

    logger.info("Detected target column: %s", schema.target_col)
    logger.info("Detected domain column: %s", schema.domain_col)
    logger.info("Target note: %s", schema.target_note)
    logger.info("Using %s gene-expression features.", len(schema.numeric_candidates))
    logger.info(
        "Excluded ID-like columns: %s", schema.id_cols if schema.id_cols else "none"
    )
    logger.info(
        "Excluded leakage-prone columns: %s",
        schema.leakage_exclude_cols if schema.leakage_exclude_cols else "none",
    )

    lodo_results, domain_weights_df, dropped_features_df = run_lodo_experiment(
        clean_df,
        target_col=schema.target_col,
        domain_col=schema.domain_col,
        schema=schema,
        config=config,
        logger=logger,
    )
    summary_df = aggregate_metric_summary(lodo_results)
    domain_distance_matrix = compute_domain_distance_matrix(clean_df, schema)
    pairwise_transfer_df = compute_pairwise_transfer(
        clean_df,
        schema,
        random_seed=config.random_seed,
        logger=logger,
    )
    pairwise_transfer_matrix = (
        pairwise_transfer_df.pivot(
            index="source_domain", columns="target_domain", values=config.primary_metric
        )
        if not pairwise_transfer_df.empty
        else pd.DataFrame()
    )

    save_core_outputs(
        output_dir=output_dir,
        audit_payload=audit_payload,
        missing_df=missing_df,
        target_distribution_df=target_distribution_df,
        domain_distribution_df=domain_distribution_df,
        feature_schema_df=feature_schema_df,
        lodo_results=lodo_results,
        summary_df=summary_df,
        domain_weights_df=domain_weights_df,
        dropped_features_df=dropped_features_df,
        pairwise_transfer_df=pairwise_transfer_df,
        domain_distance_matrix=domain_distance_matrix,
    )
    save_run_metadata(output_dir, config, schema, clean_df)

    plot_metric_boxplot(lodo_results, output_dir / "metric_boxplot.png")
    plot_metric_mean_bars(summary_df, output_dir / "metric_mean_barplot.png")
    plot_primary_metric_by_domain(
        lodo_results,
        output_dir / f"{config.primary_metric}_by_holdout_domain.png",
        primary_metric=config.primary_metric,
    )
    plot_similarity_weights(
        domain_weights_df, output_dir / "similarity_weights_by_fold.png"
    )
    plot_domain_sample_distribution(
        domain_distribution_df, output_dir / "domain_sample_distribution.png"
    )
    plot_heatmap_from_matrix(
        domain_distance_matrix,
        output_dir / "domain_distance_heatmap.png",
        title="Pairwise Domain Distance Heatmap",
        cmap=DEFAULT_DISTANCE_CMAP,
    )
    if not pairwise_transfer_matrix.empty:
        plot_heatmap_from_matrix(
            pairwise_transfer_matrix,
            output_dir / "train_domain_vs_test_domain_heatmap.png",
            title=f"Single-Source Transfer Heatmap ({config.primary_metric})",
            cmap=DEFAULT_PERFORMANCE_CMAP,
        )
        pairwise_transfer_matrix.to_csv(
            output_dir / "train_domain_vs_test_domain_matrix.csv", index=True
        )

    write_report(
        output_dir / "report.md",
        audit_payload=audit_payload,
        schema=schema,
        lodo_results=lodo_results,
        summary_df=summary_df,
        domain_weights_df=domain_weights_df,
        pairwise_transfer_df=pairwise_transfer_df,
        domain_distance_matrix=domain_distance_matrix,
        primary_metric=config.primary_metric,
        model_name=config.model_name,
        similarity_alpha=config.similarity_alpha,
    )

    copy_source_files(
        output_dir,
        [
            Path(__file__).resolve(),
            Path(__file__).resolve().with_name("weighting_pipeline.py"),
            Path(__file__).resolve().with_name("weighting_utils.py"),
        ],
    )

    logger.info("Summary of per-fold results:")
    logger.info("\n%s", frame_to_text_block(lodo_results.round(4)))
    logger.info("Summary across weighting methods:")
    logger.info("\n%s", frame_to_text_block(summary_df.round(4)))
    log_final_console_summary(summary_df, config, clean_df, schema, logger)


if __name__ == "__main__":
    main()
