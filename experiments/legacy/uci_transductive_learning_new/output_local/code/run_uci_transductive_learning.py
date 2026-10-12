"""Transportability experiment: run uci transductive learning."""

#!/usr/bin/env python3
from __future__ import annotations

"""
Complete leakage-safe transductive learning workflow for the preprocessed UCI
Heart Disease dataset in a cross-site transportability setting.

Workflow overview:
1. Load the preprocessed CSV and infer target/site/subject columns.
2. Print and save an initial dataset inspection.
3. Run leave-one-site-out evaluation.
4. For each fold, fit preprocessing on source sites only.
5. Apply CORAL using source features and the unlabeled target site.
6. Train tabular models on the aligned source representation.
7. Optionally add conservative pseudo-labels from the target site.
8. Evaluate only at the end with the true target labels.
9. Save tables, plots, fitted models and an automatic report.
"""

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
Path("/tmp/mplconfig").mkdir(parents=True, exist_ok=True)

import joblib
import numpy as np
import pandas as pd

from transductive_uci.adaptation import prepare_fold_data
from transductive_uci.analysis import (
    build_error_analysis_table,
    build_site_difficulty_ranking,
    build_variant_comparison_table,
    summarize_site_distances,
)
from transductive_uci.config import (
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_DIR,
    ExperimentConfig,
)
from transductive_uci.data_utils import (
    build_dataset_tables,
    format_initial_inspection,
    infer_columns,
    load_data,
    split_loso,
)
from transductive_uci.models import build_model_registry, run_transductive_fold
from transductive_uci.plotting import (
    create_legacy_report_compatibility_plots,
    create_summary_plots,
    plot_confusion_matrix_png,
)
from transductive_uci.reporting import (
    OutputPaths,
    RunLogger,
    build_summary_table,
    build_thesis_ready_table,
    ensure_output_structure,
    save_config,
    save_reports,
    snapshot_code,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe cross-site transductive learning on preprocessed UCI Heart Disease."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--coral-regularization", type=float, default=1e-3)
    parser.add_argument("--pseudo-threshold", type=float, default=0.95)
    parser.add_argument("--pseudo-min-samples", type=int, default=8)
    parser.add_argument("--pseudo-max-fraction", type=float, default=0.25)
    parser.add_argument("--pseudo-class-cap-ratio", type=float, default=0.50)
    parser.add_argument("--pseudo-sample-weight", type=float, default=0.35)
    parser.add_argument("--disable-pseudolabeling", action="store_true")
    parser.add_argument("--disable-model-saving", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)

    config = ExperimentConfig(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        random_state=args.random_state,
        coral_regularization=args.coral_regularization,
        pseudo_labeling_enabled=not args.disable_pseudolabeling,
        pseudo_threshold=args.pseudo_threshold,
        pseudo_min_samples=args.pseudo_min_samples,
        pseudo_max_fraction=args.pseudo_max_fraction,
        pseudo_class_cap_ratio=args.pseudo_class_cap_ratio,
        pseudo_sample_weight=args.pseudo_sample_weight,
        save_models=not args.disable_model_saving,
    )

    logger = RunLogger()
    output_paths = ensure_output_structure(config.output_dir)
    save_config(config.to_serializable_dict(), output_paths.root / "config_used.json")

    logger.log("[START] Running UCI cross-site transductive learning pipeline.")
    logger.log(f"[CONFIG] input_csv={config.input_csv}")
    logger.log(f"[CONFIG] output_dir={config.output_dir}")
    logger.log(f"[CONFIG] pseudo_labeling_enabled={config.pseudo_labeling_enabled}")
    logger.log(f"[CONFIG] pseudo_threshold={config.pseudo_threshold}")

    df = load_data(config.input_csv)
    inferred = infer_columns(df)
    dataset_tables = build_dataset_tables(df, inferred)
    for line in format_initial_inspection(df, inferred):
        logger.log(line)

    for name, table in dataset_tables.items():
        table.to_csv(output_paths.tables / f"{name}.csv", index=False)

    model_registry = build_model_registry(config)
    logger.log(f"[MODELS] {', '.join(model.name for model in model_registry)}")

    fold_metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    confidence_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    feature_importance_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    feature_shift_frames: list[pd.DataFrame] = []

    for target_site, source_df, target_df in split_loso(df, inferred.domain_column):
        logger.log(
            f"[FOLD] target_site={target_site} | source_n={len(source_df)} | target_n={len(target_df)}"
        )
        prepared = prepare_fold_data(
            source_df=source_df,
            target_df=target_df,
            inferred=inferred,
            target_site=target_site,
            config=config,
        )
        distance_row, feature_shift_df = summarize_site_distances(
            prepared, source_df, target_df, inferred
        )
        distance_rows.append(distance_row)
        feature_shift_frames.append(feature_shift_df)

        for model_spec in model_registry:
            logger.log(f"[MODEL][{target_site}] {model_spec.name}")
            fold_output = run_transductive_fold(
                model_spec=model_spec, prepared=prepared, config=config
            )
            fold_metric_rows.extend(fold_output.metric_rows)
            prediction_rows.extend(fold_output.prediction_rows)
            confidence_rows.extend(fold_output.confidence_rows)
            threshold_rows.extend(fold_output.threshold_rows)
            feature_importance_rows.extend(fold_output.feature_importance_rows)

            for confusion_artifact in fold_output.confusion_artifacts:
                plot_confusion_matrix_png(
                    matrix=confusion_artifact.matrix,
                    title=confusion_artifact.title,
                    output_path=output_paths.confusion_matrices
                    / f"confusion_{confusion_artifact.target_site}_{confusion_artifact.model_name}_{confusion_artifact.variant}.png",
                )

            if config.save_models:
                for model_bundle in fold_output.model_bundles:
                    joblib.dump(
                        model_bundle.bundle,
                        output_paths.models
                        / f"{model_bundle.target_site}_{model_bundle.model_name}_{model_bundle.variant}.joblib",
                    )

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    confidence_df = pd.DataFrame(confidence_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    feature_importance_df = pd.DataFrame(feature_importance_rows)
    distance_df = pd.DataFrame(distance_rows)
    feature_shift_df = (
        pd.concat(feature_shift_frames, axis=0, ignore_index=True)
        if feature_shift_frames
        else pd.DataFrame()
    )

    summary_df = build_summary_table(
        fold_metrics_df, primary_metric=config.primary_summary_metric
    )
    thesis_ready_df = build_thesis_ready_table(summary_df)
    variant_comparison_df = build_variant_comparison_table(fold_metrics_df)
    site_ranking_df = build_site_difficulty_ranking(
        fold_metrics_df, primary_metric=config.primary_summary_metric
    )
    error_analysis_df = build_error_analysis_table(predictions_df)

    fold_metrics_df.to_csv(output_paths.tables / "fold_metrics.csv", index=False)
    predictions_df.to_csv(output_paths.tables / "prediction_records.csv", index=False)
    confidence_df.to_csv(output_paths.tables / "confidence_records.csv", index=False)
    threshold_df.to_csv(
        output_paths.tables / "pseudo_threshold_counts.csv", index=False
    )
    feature_importance_df.to_csv(
        output_paths.tables / "feature_importance.csv", index=False
    )
    distance_df.to_csv(output_paths.tables / "site_distance_summary.csv", index=False)
    feature_shift_df.to_csv(
        output_paths.tables / "feature_shift_by_site.csv", index=False
    )
    summary_df.to_csv(output_paths.tables / "summary_metrics.csv", index=False)
    thesis_ready_df.to_csv(
        output_paths.tables / "thesis_ready_summary.csv", index=False
    )
    variant_comparison_df.to_csv(
        output_paths.tables / "variant_comparison.csv", index=False
    )
    site_ranking_df.to_csv(
        output_paths.tables / "site_difficulty_ranking.csv", index=False
    )
    error_analysis_df.to_csv(output_paths.tables / "error_analysis.csv", index=False)

    create_summary_plots(
        fold_metrics_df=fold_metrics_df,
        variant_comparison_df=variant_comparison_df,
        predictions_df=predictions_df,
        confidence_df=confidence_df,
        threshold_df=threshold_df,
        summary_df=summary_df,
        distance_df=distance_df,
        feature_shift_df=feature_shift_df,
        raw_df=df,
        inferred=inferred,
        plots_dir=output_paths.plots,
        ece_bins=config.ece_bins,
        top_shift_features=config.feature_shift_top_n,
        pseudo_threshold=config.pseudo_threshold,
    )
    create_legacy_report_compatibility_plots(
        fold_metrics_df=fold_metrics_df,
        predictions_df=predictions_df,
        output_dir=output_paths.root,
    )

    save_reports(
        output_paths=output_paths,
        input_csv=config.input_csv,
        inferred=inferred,
        dataset_tables=dataset_tables,
        model_names=[model.name for model in model_registry],
        summary_df=summary_df,
        site_ranking_df=site_ranking_df,
        variant_comparison_df=variant_comparison_df,
        distance_df=distance_df,
        error_analysis_df=error_analysis_df,
    )
    snapshot_code(output_paths, Path(__file__).resolve())
    logger.log(f"[DONE] Results saved in {output_paths.root}")
    logger.save(output_paths.logs / "run.log")


if __name__ == "__main__":
    main()
