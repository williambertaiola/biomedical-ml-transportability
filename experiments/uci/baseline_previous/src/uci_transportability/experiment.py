"""Transportability experiment: experiment."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from uci_transportability.config import (
    ExperimentConfig,
    pretty_model_name,
    pretty_target_site,
)
from uci_transportability.data import (
    TargetSiteSplit,
    build_all_target_splits,
    load_dataset,
)
from uci_transportability.metrics import compute_binary_metrics, id_ood_gap
from uci_transportability.models import (
    build_model_specs,
    build_pipeline,
    safe_predict_proba,
)


@dataclass
class ExperimentOutputs:
    dataset_df: pd.DataFrame
    schema: dict[str, Any]
    target_site_results: pd.DataFrame
    id_validation_fold_metrics: pd.DataFrame
    prediction_records: pd.DataFrame
    split_manifest: pd.DataFrame
    split_assignments: pd.DataFrame
    model_summary: pd.DataFrame


def _fallback_id_splits(
    source_df: pd.DataFrame, config: ExperimentConfig
) -> list[tuple[Any, Any]]:
    class_counts = source_df[config.target_column].value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    n_splits = max(2, min(3, min_class_count))
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=config.random_seed
    )
    return list(splitter.split(source_df, source_df[config.target_column]))


def run_source_id_validation(
    *,
    source_df: pd.DataFrame,
    schema: dict[str, Any],
    model_name: str,
    config: ExperimentConfig,
    logger: logging.Logger,
    evaluation_target_site: str,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    groups = source_df[config.domain_column].astype(str)
    unique_groups = groups.nunique()
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    if unique_groups >= 2:
        split_iterator = LeaveOneGroupOut().split(
            source_df, source_df[config.target_column], groups=groups
        )
        split_name = "LeaveOneGroupOut"
    else:
        split_iterator = _fallback_id_splits(source_df, config)
        split_name = "StratifiedKFoldFallback"

    model_specs = build_model_specs(config)
    model_spec = model_specs[model_name]

    for fold_idx, (train_idx, valid_idx) in enumerate(split_iterator, start=1):
        train_df = source_df.iloc[train_idx].reset_index(drop=True)
        valid_df = source_df.iloc[valid_idx].reset_index(drop=True)
        pipeline = build_pipeline(schema, model_spec)
        pipeline.fit(train_df, train_df[config.target_column].astype(int))
        y_prob = safe_predict_proba(pipeline, valid_df)
        metrics = compute_binary_metrics(
            valid_df[config.target_column].to_numpy(dtype=int),
            y_prob,
            threshold=config.threshold,
            ece_bins=config.ece_bins,
        )

        held_out_domains = sorted(
            valid_df[config.domain_column].dropna().astype(str).unique().tolist()
        )
        held_out_label = "|".join(held_out_domains)
        fold_rows.append(
            {
                "method": config.method_name,
                "model": model_name,
                "model_display": pretty_model_name(model_name),
                "evaluation_target_site": evaluation_target_site,
                "evaluation_target_site_display": pretty_target_site(
                    evaluation_target_site
                ),
                "id_validation_scheme": split_name,
                "fold_index": fold_idx,
                "held_out_source_domain": held_out_label,
                "n_train": int(len(train_df)),
                "n_valid": int(len(valid_df)),
                "roc_auc_id_fold": metrics["roc_auc"],
                "f1_id_fold": metrics["f1"],
                "balanced_accuracy_id_fold": metrics["balanced_accuracy"],
                "ece_id_fold": metrics["ece"],
            }
        )

        predictions.append(
            pd.DataFrame(
                {
                    "method": config.method_name,
                    "model": model_name,
                    "model_display": pretty_model_name(model_name),
                    "evaluation_target_site": evaluation_target_site,
                    "evaluation_target_site_display": pretty_target_site(
                        evaluation_target_site
                    ),
                    "evaluation_split": "id_validation",
                    "fold_index": fold_idx,
                    "evaluation_domain": held_out_label,
                    "subject_id": valid_df[config.id_column].astype(str).to_numpy(),
                    "domain_id": valid_df[config.domain_column].astype(str).to_numpy(),
                    "y_true": valid_df[config.target_column].astype(int).to_numpy(),
                    "y_prob": y_prob,
                    "y_pred": (y_prob >= config.threshold).astype(int),
                }
            )
        )

    prediction_df = (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    )
    id_metrics = compute_binary_metrics(
        prediction_df["y_true"].to_numpy(dtype=int),
        prediction_df["y_prob"].to_numpy(dtype=float),
        threshold=config.threshold,
        ece_bins=config.ece_bins,
    )
    id_metrics["id_validation_scheme"] = split_name
    return id_metrics, pd.DataFrame(fold_rows), prediction_df


def run_target_ood_evaluation(
    *,
    source_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    schema: dict[str, Any],
    model_name: str,
    config: ExperimentConfig,
    evaluation_target_site: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    model_spec = build_model_specs(config)[model_name]
    pipeline = build_pipeline(schema, model_spec)
    pipeline.fit(source_df, source_df[config.target_column].astype(int))
    y_prob = safe_predict_proba(pipeline, target_test_df)
    metrics = compute_binary_metrics(
        target_test_df[config.target_column].to_numpy(dtype=int),
        y_prob,
        threshold=config.threshold,
        ece_bins=config.ece_bins,
    )
    prediction_df = pd.DataFrame(
        {
            "method": config.method_name,
            "model": model_name,
            "model_display": pretty_model_name(model_name),
            "evaluation_target_site": evaluation_target_site,
            "evaluation_target_site_display": pretty_target_site(
                evaluation_target_site
            ),
            "evaluation_split": "ood_test_1",
            "fold_index": pd.NA,
            "evaluation_domain": evaluation_target_site,
            "subject_id": target_test_df[config.id_column].astype(str).to_numpy(),
            "domain_id": target_test_df[config.domain_column].astype(str).to_numpy(),
            "y_true": target_test_df[config.target_column].astype(int).to_numpy(),
            "y_prob": y_prob,
            "y_pred": (y_prob >= config.threshold).astype(int),
        }
    )
    return metrics, prediction_df


def _run_single_target(
    *,
    split: TargetSiteSplit,
    schema: dict[str, Any],
    config: ExperimentConfig,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    result_rows: list[dict[str, Any]] = []
    fold_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []

    source_domains = sorted(
        split.source_df[config.domain_column].dropna().astype(str).unique().tolist()
    )
    for model_name in build_model_specs(config):
        logger.info("Running %s on target site %s", model_name, split.target_site)
        id_metrics, id_fold_df, id_prediction_df = run_source_id_validation(
            source_df=split.source_df,
            schema=schema,
            model_name=model_name,
            config=config,
            logger=logger,
            evaluation_target_site=split.target_site,
        )
        ood_metrics, ood_prediction_df = run_target_ood_evaluation(
            source_df=split.source_df,
            target_test_df=split.target_test_df,
            schema=schema,
            model_name=model_name,
            config=config,
            evaluation_target_site=split.target_site,
        )

        result_rows.append(
            {
                "method": config.method_name,
                "model": model_name,
                "model_display": pretty_model_name(model_name),
                "target_site": split.target_site,
                "target_site_display": pretty_target_site(split.target_site),
                "source_domains": "|".join(source_domains),
                "n_source_domains": int(len(source_domains)),
                "n_source_samples": int(len(split.source_df)),
                "n_target_train_1": int(len(split.target_train_df)),
                "n_target_test_1": int(len(split.target_test_df)),
                "id_validation_scheme": id_metrics["id_validation_scheme"],
                "roc_auc_id": id_metrics["roc_auc"],
                "roc_auc_ood": ood_metrics["roc_auc"],
                "gap_roc_auc": id_ood_gap(
                    id_metrics["roc_auc"], ood_metrics["roc_auc"]
                ),
                "f1_id": id_metrics["f1"],
                "f1_ood": ood_metrics["f1"],
                "gap_f1": id_ood_gap(id_metrics["f1"], ood_metrics["f1"]),
                "balanced_accuracy_id": id_metrics["balanced_accuracy"],
                "balanced_accuracy_ood": ood_metrics["balanced_accuracy"],
                "gap_balanced_accuracy": id_ood_gap(
                    id_metrics["balanced_accuracy"],
                    ood_metrics["balanced_accuracy"],
                ),
                "ece_id": id_metrics["ece"],
                "ece_ood": ood_metrics["ece"],
            }
        )

        fold_tables.append(id_fold_df)
        prediction_tables.append(id_prediction_df)
        prediction_tables.append(ood_prediction_df)

    return result_rows, fold_tables, prediction_tables


def _build_model_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    summary = (
        results_df.groupby(["method", "model", "model_display"], dropna=False)
        .agg(
            n_target_sites=("target_site", "nunique"),
            roc_auc_id_mean=("roc_auc_id", "mean"),
            roc_auc_ood_mean=("roc_auc_ood", "mean"),
            gap_roc_auc_mean=("gap_roc_auc", "mean"),
            f1_id_mean=("f1_id", "mean"),
            f1_ood_mean=("f1_ood", "mean"),
            gap_f1_mean=("gap_f1", "mean"),
            balanced_accuracy_id_mean=("balanced_accuracy_id", "mean"),
            balanced_accuracy_ood_mean=("balanced_accuracy_ood", "mean"),
            gap_balanced_accuracy_mean=("gap_balanced_accuracy", "mean"),
            ece_id_mean=("ece_id", "mean"),
            ece_ood_mean=("ece_ood", "mean"),
        )
        .reset_index()
        .sort_values(["method", "model"])
        .reset_index(drop=True)
    )
    return summary


def run_baseline_experiment(
    config: ExperimentConfig, logger: logging.Logger
) -> ExperimentOutputs:
    dataset_df, schema = load_dataset(config, logger)
    target_splits = build_all_target_splits(dataset_df, config=config, logger=logger)

    result_rows: list[dict[str, Any]] = []
    id_fold_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    split_manifest_rows = [split.manifest_row for split in target_splits]
    split_assignment_tables = [split.assignment_df for split in target_splits]

    for split in target_splits:
        rows, fold_tables, predictions = _run_single_target(
            split=split,
            schema=schema,
            config=config,
            logger=logger,
        )
        result_rows.extend(rows)
        id_fold_tables.extend(fold_tables)
        prediction_tables.extend(predictions)

    target_site_results = (
        pd.DataFrame(result_rows)
        .sort_values(["model", "target_site"])
        .reset_index(drop=True)
    )
    id_validation_fold_metrics = (
        pd.concat(id_fold_tables, ignore_index=True)
        if id_fold_tables
        else pd.DataFrame()
    )
    prediction_records = (
        pd.concat(prediction_tables, ignore_index=True)
        if prediction_tables
        else pd.DataFrame()
    )
    split_manifest = (
        pd.DataFrame(split_manifest_rows)
        .sort_values("evaluation_target_site")
        .reset_index(drop=True)
    )
    split_assignments = (
        pd.concat(split_assignment_tables, ignore_index=True)
        if split_assignment_tables
        else pd.DataFrame()
    )
    model_summary = _build_model_summary(target_site_results)

    return ExperimentOutputs(
        dataset_df=dataset_df,
        schema=schema,
        target_site_results=target_site_results,
        id_validation_fold_metrics=id_validation_fold_metrics,
        prediction_records=prediction_records,
        split_manifest=split_manifest,
        split_assignments=split_assignments,
        model_summary=model_summary,
    )
