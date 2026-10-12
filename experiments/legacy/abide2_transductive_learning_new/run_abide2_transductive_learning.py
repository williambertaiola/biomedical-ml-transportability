"""Transportability experiment: run abide2 transductive learning."""

#!/usr/bin/env python3
from __future__ import annotations

"""
ABIDE II multisite cross-site transductive learning pipeline.

Workflow overview:
1. Automatically identify the main dataset file inside the input directory.
2. Load the dataset and infer target/domain identifiers.
3. Print a dataset audit with shape, column preview and domain/target distributions.
4. Run leave-one-domain-out evaluation.
5. Fit all preprocessing steps on source domains only:
   imputation, variance filtering, scaling and optional PCA.
6. Adapt source features to the unlabeled target domain with CORAL.
7. Train tabular models and optionally add conservative pseudo-labels.
8. Evaluate only at the end with the true target labels.
9. Save tables, models, plots and an automatic report.
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

from transductive_abide.analysis import (
    build_error_analysis_table,
    build_site_difficulty_ranking,
    build_variant_comparison_table,
    summarize_site_distances,
)
from transductive_abide.config import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    ExperimentConfig,
)
from transductive_abide.data_utils import (
    build_dataset_tables,
    ensure_subject_ids,
    format_initial_inspection,
    infer_columns,
    load_data,
    split_loso,
)
from transductive_abide.metrics import top_label_calibration_curve
from transductive_abide.models import build_model_registry, run_transductive_fold
from transductive_abide.plotting import create_all_plots, plot_confusion_matrix_png
from transductive_abide.preprocessing import prepare_fold_data
from transductive_abide.reporting import (
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
        description="Leakage-safe ABIDE II multisite transductive learning."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-variable-features", type=int, default=64)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--disable-pca", action="store_true")
    parser.add_argument("--pseudo-threshold", type=float, default=0.98)
    parser.add_argument("--pseudo-min-samples", type=int, default=10)
    parser.add_argument("--pseudo-max-fraction", type=float, default=0.15)
    parser.add_argument("--pseudo-class-cap-ratio", type=float, default=0.35)
    parser.add_argument("--pseudo-sample-weight", type=float, default=0.25)
    parser.add_argument("--disable-pseudolabeling", action="store_true")
    parser.add_argument("--disable-model-saving", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)

    config = ExperimentConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        random_state=args.random_state,
        top_variable_features=args.top_variable_features,
        pca_components=args.pca_components,
        use_pca=not args.disable_pca,
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

    logger.log("[START] Running ABIDE II transductive learning pipeline.")
    logger.log(f"[CONFIG] input_path={config.input_path}")
    logger.log(f"[CONFIG] output_dir={config.output_dir}")
    logger.log(
        f"[CONFIG] use_pca={config.use_pca} | pca_components={config.pca_components}"
    )
    logger.log(f"[CONFIG] top_variable_features={config.top_variable_features}")
    logger.log(f"[CONFIG] pseudo_labeling_enabled={config.pseudo_labeling_enabled}")
    logger.log(f"[CONFIG] pseudo_threshold={config.pseudo_threshold}")

    df, dataset_file = load_data(config.input_path)
    df, inferred = infer_columns(
        df,
        random_state=config.random_state,
        artificial_domain_count=config.artificial_domain_count,
    )
    dataset_tables = build_dataset_tables(df, inferred)
    for line in format_initial_inspection(df, inferred):
        logger.log(line)
    logger.log(f"[DATA] dataset_file={dataset_file}")

    for name, table in dataset_tables.items():
        table.to_csv(output_paths.tables / f"{name}.csv", index=False)

    model_registry = build_model_registry(config, n_classes=len(inferred.class_labels))
    logger.log(f"[MODELS] {', '.join(model.name for model in model_registry)}")

    fold_metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    confidence_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    feature_importance_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    feature_shift_frames: list[pd.DataFrame] = []
    alignment_plot_frames: list[pd.DataFrame] = []
    fold_overview_rows: list[dict[str, object]] = []

    for target_domain, source_df, target_df in split_loso(df, inferred.domain_column):
        source_ids = ensure_subject_ids(source_df, inferred)
        target_ids = ensure_subject_ids(target_df, inferred)
        overlap = set(source_ids.tolist()).intersection(set(target_ids.tolist()))
        if overlap:
            raise RuntimeError(
                f"[LEAKAGE ERROR] Subject overlap detected between source and target for domain {target_domain}."
            )

        fold_overview_rows.append(
            {
                "target_domain": target_domain,
                "source_n": int(len(source_df)),
                "target_n": int(len(target_df)),
                "target_class_distribution": dict(
                    target_df[inferred.target_column]
                    .astype("string")
                    .value_counts(dropna=False)
                ),
            }
        )
        logger.log(
            f"[FOLD] target_domain={target_domain} | source_n={len(source_df)} | target_n={len(target_df)} "
            f"| target_classes={target_df[inferred.target_column].astype('string').value_counts().to_dict()}"
        )

        prepared = prepare_fold_data(
            source_df=source_df,
            target_df=target_df,
            inferred=inferred,
            target_domain=target_domain,
            config=config,
        )
        distance_artifacts = summarize_site_distances(
            prepared=prepared,
            source_df=source_df,
            target_df=target_df,
            max_top_features=config.feature_shift_save_top_n,
            max_points_plot=config.max_points_per_site_plot,
            random_state=config.random_state,
        )
        distance_rows.append(distance_artifacts.summary_row)
        feature_shift_frames.append(distance_artifacts.feature_shift_df)
        alignment_plot_frames.append(distance_artifacts.alignment_plot_df)

        for model_spec in model_registry:
            logger.log(f"[MODEL][{target_domain}] {model_spec.name}")
            fold_output = run_transductive_fold(
                model_spec=model_spec,
                prepared=prepared,
                class_labels=inferred.class_labels,
                config=config,
            )
            fold_metric_rows.extend(fold_output.metric_rows)
            prediction_rows.extend(fold_output.prediction_rows)
            confidence_rows.extend(fold_output.confidence_rows)
            threshold_rows.extend(fold_output.threshold_rows)
            feature_importance_rows.extend(fold_output.feature_importance_rows)

            for confusion_artifact in fold_output.confusion_artifacts:
                plot_confusion_matrix_png(
                    matrix=confusion_artifact.matrix,
                    class_labels=confusion_artifact.class_labels,
                    title=confusion_artifact.title,
                    output_path=output_paths.confusion_matrices
                    / f"confusion_{confusion_artifact.target_domain}_{confusion_artifact.model_name}_{confusion_artifact.variant}.png",
                )

            if config.save_models:
                for bundle in fold_output.model_bundles:
                    joblib.dump(
                        bundle.bundle,
                        output_paths.models
                        / f"{bundle.target_domain}_{bundle.model_name}_{bundle.variant}.joblib",
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
    alignment_plot_df = (
        pd.concat(alignment_plot_frames, axis=0, ignore_index=True)
        if alignment_plot_frames
        else pd.DataFrame()
    )
    fold_overview_df = pd.DataFrame(fold_overview_rows)

    calibration_rows: list[pd.DataFrame] = []
    for (model_name, variant), group in predictions_df.groupby(
        ["model_name", "variant"], dropna=False
    ):
        prob_columns = [
            column for column in group.columns if column.startswith("prob_")
        ]
        if not prob_columns:
            continue
        calibration_rows.append(
            top_label_calibration_curve(
                y_true=group["y_true"].to_numpy(dtype=int),
                y_prob=group.loc[:, prob_columns].to_numpy(dtype=float),
                n_bins=config.ece_bins,
                metadata={"model_name": model_name, "variant": variant},
            )
        )
    calibration_df = (
        pd.concat(calibration_rows, axis=0, ignore_index=True)
        if calibration_rows
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
    fold_metrics_df.to_csv(output_paths.tables / "metrics.csv", index=False)
    predictions_df.to_csv(output_paths.tables / "predictions.csv", index=False)
    confidence_df.to_csv(output_paths.tables / "confidence_records.csv", index=False)
    threshold_df.to_csv(
        output_paths.tables / "pseudo_threshold_counts.csv", index=False
    )
    feature_importance_df.to_csv(
        output_paths.tables / "feature_importance.csv", index=False
    )
    distance_df.to_csv(output_paths.tables / "site_distance_summary.csv", index=False)
    feature_shift_df.to_csv(
        output_paths.tables / "feature_shift_top_by_domain.csv", index=False
    )
    alignment_plot_df.to_csv(
        output_paths.tables / "alignment_plot_records.csv", index=False
    )
    calibration_df.to_csv(
        output_paths.tables / "calibration_curve_data.csv", index=False
    )
    fold_overview_df.to_csv(output_paths.tables / "fold_overview.csv", index=False)
    summary_df.to_csv(output_paths.tables / "summary.csv", index=False)
    thesis_ready_df.to_csv(
        output_paths.tables / "thesis_ready_summary.csv", index=False
    )
    variant_comparison_df.to_csv(
        output_paths.tables / "variant_comparison.csv", index=False
    )
    site_ranking_df.to_csv(
        output_paths.tables / "domain_difficulty_ranking.csv", index=False
    )
    error_analysis_df.to_csv(output_paths.tables / "error_analysis.csv", index=False)

    create_all_plots(
        fold_metrics_df=fold_metrics_df,
        variant_comparison_df=variant_comparison_df,
        predictions_df=predictions_df,
        confidence_df=confidence_df,
        threshold_df=threshold_df,
        summary_df=summary_df,
        distance_df=distance_df,
        feature_shift_df=feature_shift_df,
        calibration_df=calibration_df,
        alignment_plot_df=alignment_plot_df,
        plots_dir=output_paths.plots,
        pseudo_threshold=config.pseudo_threshold,
    )

    save_reports(
        output_paths=output_paths,
        dataset_file=dataset_file,
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
