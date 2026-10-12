"""Transportability experiment: weighting pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")

import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")

from weighting_utils import (
    DEFAULT_DISTANCE_CMAP,
    DEFAULT_PERFORMANCE_CMAP,
    METHOD_ORDER,
    METRIC_COLUMNS,
    aggregate_metric_summary,
    audit_payload_from_dataframe,
    build_domain_weight_rows,
    compute_domain_balanced_weights,
    compute_domain_distance_matrix,
    compute_metrics,
    compute_pairwise_transfer,
    compute_similarity_weights,
    copy_source_files,
    detect_dataset_schema,
    ensure_directory,
    fit_logistic_regression,
    frame_to_text_block,
    init_logger,
    plot_domain_sample_distribution,
    plot_heatmap_from_matrix,
    plot_metric_boxplot,
    plot_metric_mean_bars,
    plot_primary_metric_by_domain,
    plot_similarity_weights,
    prepare_column_spec,
    prepare_modeling_dataframe,
    read_table,
    save_json,
    schema_to_dataframe,
    transform_fold,
    write_report,
)


@dataclass
class RunConfig:
    input_path: str
    output_dir: str
    random_seed: int = 42
    similarity_alpha: float = 2.0
    low_cardinality_threshold: int = 10
    primary_metric: str = "balanced_accuracy"
    model_name: str = "LogisticRegression"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LODO weighting pipeline for cross-site transportability on tabular biomedical datasets."
    )
    parser.add_argument(
        "--input", required=True, help="Path to the preprocessed input dataset."
    )
    parser.add_argument(
        "--output", required=True, help="Directory where all results will be saved."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--similarity-alpha",
        type=float,
        default=2.0,
        help="Sharpness parameter used in the similarity-based weighting rule.",
    )
    parser.add_argument(
        "--low-cardinality-threshold",
        type=int,
        default=10,
        help="Numeric integer-like columns with <= threshold unique values are treated as categorical.",
    )
    return parser


def resolve_input_path(input_path: str | Path) -> Path:
    candidate = Path(input_path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {candidate}")

    supported_suffixes = {".csv", ".tsv", ".txt", ".parquet"}
    files = [
        path
        for path in sorted(candidate.rglob("*"))
        if path.is_file() and path.suffix.lower() in supported_suffixes
    ]
    if not files:
        raise FileNotFoundError(
            f"No supported dataset files found inside directory: {candidate}"
        )

    preferred_patterns = [
        "transportability_dataset",
        "multisite_transportability_dataset",
        "preprocessed",
        "processed",
    ]
    ranked: list[tuple[tuple[int, int, int], Path]] = []
    for file_path in files:
        lowered = file_path.name.lower()
        pattern_score = (
            max(
                (100 - idx * 10)
                for idx, pattern in enumerate(preferred_patterns)
                if pattern in lowered
            )
            if any(pattern in lowered for pattern in preferred_patterns)
            else 0
        )
        ranked.append(
            (
                (pattern_score, int(file_path.stat().st_size), -len(file_path.name)),
                file_path,
            )
        )

    return max(ranked, key=lambda item: item[0])[1]


def run_lodo_experiment(
    df: pd.DataFrame,
    *,
    target_col: str,
    domain_col: str,
    schema,
    config: RunConfig,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_rows: list[dict[str, Any]] = []
    domain_weight_rows: list[dict[str, Any]] = []
    dropped_feature_rows: list[dict[str, Any]] = []
    holdout_domains = sorted(df[domain_col].astype(str).unique().tolist())

    logger.info(
        "Starting LODO evaluation across %s domains: %s",
        len(holdout_domains),
        ", ".join(holdout_domains),
    )

    for fold_index, holdout_domain in enumerate(holdout_domains, start=1):
        train_df = df[df[domain_col].astype(str) != holdout_domain].reset_index(
            drop=True
        )
        test_df = df[df[domain_col].astype(str) == holdout_domain].reset_index(
            drop=True
        )

        if train_df[target_col].nunique(dropna=True) < 2:
            raise ValueError(
                f"Fold with holdout domain '{holdout_domain}' has only one class in the source training data."
            )

        column_spec = prepare_column_spec(train_df, schema)
        _, X_train, X_test = transform_fold(train_df, test_df, column_spec)
        y_train = train_df[target_col].to_numpy(dtype=int)
        y_test = test_df[target_col].to_numpy(dtype=int)
        train_domains = train_df[domain_col].astype(str).reset_index(drop=True)
        source_domains = sorted(train_domains.unique().tolist())

        logger.info(
            "LODO fold %s/%s | holdout_domain=%s | n_train=%s | n_test=%s | usable_features=%s | transformed_features=%s",
            fold_index,
            len(holdout_domains),
            holdout_domain,
            len(train_df),
            len(test_df),
            len(column_spec.feature_cols),
            X_train.shape[1],
        )

        raw_domain_weights = {domain: 1.0 for domain in source_domains}
        raw_sample_weights = np.ones(len(train_df), dtype=float)
        balanced_sample_weights, balanced_domain_weights = (
            compute_domain_balanced_weights(train_domains)
        )
        (
            similarity_sample_weights,
            similarity_domain_weights,
            similarity_distances,
            similarity_scale,
        ) = compute_similarity_weights(
            X_train,
            train_domains,
            X_test,
            alpha=config.similarity_alpha,
        )

        logger.info(
            "Similarity distances to holdout %s: %s",
            holdout_domain,
            ", ".join(
                f"{domain}={distance:.4f}"
                for domain, distance in similarity_distances.items()
            ),
        )

        weight_registry = {
            "raw": {
                "sample_weights": raw_sample_weights,
                "domain_weights": raw_domain_weights,
                "distances": None,
                "scale": None,
            },
            "domain_balanced": {
                "sample_weights": balanced_sample_weights,
                "domain_weights": balanced_domain_weights,
                "distances": None,
                "scale": None,
            },
            "similarity_based": {
                "sample_weights": similarity_sample_weights,
                "domain_weights": similarity_domain_weights,
                "distances": similarity_distances,
                "scale": similarity_scale,
            },
        }

        for method in METHOD_ORDER:
            weights_payload = weight_registry[method]
            model = fit_logistic_regression(
                X_train,
                y_train,
                sample_weight=weights_payload["sample_weights"],
                random_seed=config.random_seed,
            )
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, y_prob)

            results_rows.append(
                {
                    "fold_index": fold_index,
                    "holdout_domain": holdout_domain,
                    "method": method,
                    "model": config.model_name,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "n_source_domains": int(len(source_domains)),
                    "source_domains": "|".join(source_domains),
                    "positive_rate_train": float(train_df[target_col].mean()),
                    "positive_rate_test": float(test_df[target_col].mean()),
                    "n_features_before_encoding": int(len(column_spec.feature_cols)),
                    "n_numeric_features": int(len(column_spec.numeric_cols)),
                    "n_categorical_features": int(len(column_spec.categorical_cols)),
                    "n_transformed_features": int(X_train.shape[1]),
                    **metrics,
                }
            )

            domain_weight_rows.extend(
                build_domain_weight_rows(
                    holdout_domain=holdout_domain,
                    method=method,
                    train_domains=train_domains,
                    domain_weights=weights_payload["domain_weights"],
                    distances=weights_payload["distances"],
                    similarity_scale=weights_payload["scale"],
                )
            )

        for dropped in column_spec.dropped_cols:
            dropped_feature_rows.append(
                {
                    "fold_index": fold_index,
                    "holdout_domain": holdout_domain,
                    **dropped,
                }
            )

    return (
        pd.DataFrame(results_rows),
        pd.DataFrame(domain_weight_rows),
        pd.DataFrame(dropped_feature_rows),
    )


def save_core_outputs(
    *,
    output_dir: Path,
    audit_payload: dict[str, Any],
    missing_df: pd.DataFrame,
    target_distribution_df: pd.DataFrame,
    domain_distribution_df: pd.DataFrame,
    feature_schema_df: pd.DataFrame,
    lodo_results: pd.DataFrame,
    summary_df: pd.DataFrame,
    domain_weights_df: pd.DataFrame,
    dropped_features_df: pd.DataFrame,
    pairwise_transfer_df: pd.DataFrame,
    domain_distance_matrix: pd.DataFrame,
) -> None:
    save_json(output_dir / "dataset_audit.json", audit_payload)
    missing_df.to_csv(output_dir / "missing_by_column.csv", index=False)
    target_distribution_df.to_csv(output_dir / "target_distribution.csv", index=False)
    domain_distribution_df.to_csv(output_dir / "domain_distribution.csv", index=False)
    feature_schema_df.to_csv(output_dir / "feature_schema.csv", index=False)
    lodo_results.to_csv(output_dir / "lodo_results.csv", index=False)
    summary_df.to_csv(output_dir / "weighting_summary.csv", index=False)
    domain_weights_df.to_csv(output_dir / "domain_weights.csv", index=False)
    dropped_features_df.to_csv(output_dir / "dropped_features_by_fold.csv", index=False)
    pairwise_transfer_df.to_csv(
        output_dir / "pairwise_transfer_results.csv", index=False
    )
    domain_distance_matrix.to_csv(output_dir / "domain_distance_matrix.csv", index=True)


def save_run_metadata(
    output_dir: Path, config: RunConfig, schema, df: pd.DataFrame
) -> None:
    metadata = {
        "config": asdict(config),
        "schema": asdict(schema),
        "dataset_shape_after_cleaning": list(df.shape),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def log_final_console_summary(
    summary_df: pd.DataFrame, config: RunConfig, df: pd.DataFrame, schema, logger
) -> None:
    primary_rows = summary_df[
        summary_df["metric"] == config.primary_metric
    ].sort_values("mean", ascending=False)
    best_primary_method = (
        primary_rows.iloc[0]["method"] if not primary_rows.empty else "n/a"
    )
    best_primary_value = (
        primary_rows.iloc[0]["mean"] if not primary_rows.empty else math.nan
    )

    overall_best_row = (
        summary_df.sort_values("mean", ascending=False).iloc[0]
        if not summary_df.empty
        else None
    )
    overall_best_metric = (
        overall_best_row["metric"] if overall_best_row is not None else "n/a"
    )
    overall_best_method = (
        overall_best_row["method"] if overall_best_row is not None else "n/a"
    )
    overall_best_value = (
        overall_best_row["mean"] if overall_best_row is not None else math.nan
    )

    summary_lines = [
        "",
        "================ FINAL CONSOLE SUMMARY ================",
        f"Numero domini: {df[schema.domain_col].nunique(dropna=True)}",
        f"Target usato: {schema.target_col}",
        f"Modello usato: {config.model_name}",
        f"Metodi confrontati: {', '.join(METHOD_ORDER)}",
        f"Metrica primaria per ranking: {config.primary_metric}",
        f"Miglior metodo medio OOD ({config.primary_metric}): {best_primary_method} ({best_primary_value:.4f})",
        f"Miglior coppia metodo-metrica media: {overall_best_method} | {overall_best_metric} = {overall_best_value:.4f}",
        "=======================================================",
    ]
    for line in summary_lines:
        logger.info(line)


def main() -> None:
    args = build_parser().parse_args()
    resolved_input_path = resolve_input_path(args.input)
    config = RunConfig(
        input_path=str(resolved_input_path),
        output_dir=args.output,
        random_seed=args.seed,
        similarity_alpha=args.similarity_alpha,
        low_cardinality_threshold=args.low_cardinality_threshold,
    )

    output_dir = ensure_directory(config.output_dir)
    logger = init_logger(output_dir / "execution_log.txt")
    logger.info("Starting weighting pipeline.")
    logger.info("Input dataset: %s", Path(config.input_path).resolve())
    logger.info("Output directory: %s", output_dir.resolve())

    raw_df = read_table(config.input_path)
    logger.info("Loaded raw dataset with shape %s", raw_df.shape)

    schema = detect_dataset_schema(
        raw_df, low_cardinality_threshold=config.low_cardinality_threshold
    )
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
    logger.info(
        "Detected %s numeric and %s categorical feature candidates.",
        len(schema.numeric_candidates),
        len(schema.categorical_candidates),
    )
    logger.info(
        "Detected ID-like columns excluded from modeling: %s",
        schema.id_cols if schema.id_cols else "none",
    )
    logger.info(
        "Detected leakage-prone auxiliary columns excluded from modeling: %s",
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
