"""Transportability pipeline module: experiment."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.model_selection import train_test_split

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
from uci_transportability.harmonization import prepare_method_train_test_frames
from uci_transportability.metrics import compute_binary_metrics, id_ood_gap
from uci_transportability.models import (
    build_model_specs,
    build_pipeline,
    safe_predict_proba,
)
from uci_transportability.transductive import predict_transductive_blend
from uci_transportability.weighting import compute_m2_domain_weighting


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
    m2_pairwise_weighting: pd.DataFrame


def _method_diagnostics(method_details: dict[str, Any]) -> dict[str, Any]:
    return {
        "harmonization": method_details.get("harmonization", "none"),
        "combat_backend": method_details.get("combat_backend", "none"),
        "combat_reference_batch": method_details.get("combat_reference_batch", ""),
        "n_combat_features": method_details.get("n_combat_features", 0),
        "combat_features": method_details.get("combat_features", ""),
        "combat_skipped_features": method_details.get("combat_skipped_features", ""),
        "weighting": method_details.get("weighting", "none"),
        "weight_backend": method_details.get("weight_backend", "none"),
        "domain_weights": method_details.get("domain_weights", ""),
        "domain_transfer_scores": method_details.get("domain_transfer_scores", ""),
        "sample_weight_min": method_details.get("sample_weight_min", 1.0),
        "sample_weight_mean": method_details.get("sample_weight_mean", 1.0),
        "sample_weight_max": method_details.get("sample_weight_max", 1.0),
        "transductive": method_details.get("transductive", "none"),
        "transductive_backend": method_details.get("transductive_backend", "none"),
        "transductive_blend_weight": method_details.get(
            "transductive_blend_weight", 0.0
        ),
        "n_transductive_adaptation_samples": method_details.get(
            "n_transductive_adaptation_samples", 0
        ),
        "transductive_adapter_status": method_details.get(
            "transductive_adapter_status", "none"
        ),
    }


def _apply_m2_weighting(
    *,
    method_name: str,
    train_df: pd.DataFrame,
    schema: dict[str, Any],
    config: ExperimentConfig,
    context: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any], pd.DataFrame]:
    if method_name != "M2":
        return None, {}, pd.DataFrame()

    weighting_result = compute_m2_domain_weighting(
        train_df=train_df,
        schema=schema,
        config=config,
    )
    pairwise = weighting_result.pairwise_scores.copy()
    if not pairwise.empty:
        for key, value in context.items():
            pairwise[key] = value
    return weighting_result.sample_weight, weighting_result.details, pairwise


def _fit_pipeline_with_optional_weights(
    pipeline, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None
) -> None:
    if sample_weight is None:
        pipeline.fit(X, y)
    else:
        pipeline.fit(X, y, model__sample_weight=sample_weight)


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


def _split_transductive_adaptation_test(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[config.target_column]
    class_counts = y.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    use_stratify = y.nunique() > 1 and min_class_count >= 2
    adaptation_df, evaluation_df = train_test_split(
        df,
        test_size=config.target_test_size,
        random_state=config.random_seed,
        shuffle=True,
        stratify=y if use_stratify else None,
    )
    return adaptation_df.reset_index(drop=True), evaluation_df.reset_index(drop=True)


def run_source_id_validation(
    *,
    method_name: str,
    source_df: pd.DataFrame,
    schema: dict[str, Any],
    model_name: str,
    config: ExperimentConfig,
    logger: logging.Logger,
    evaluation_target_site: str,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate in-distribution metrics using the original source validation scheme."""
    groups = source_df[config.domain_column].astype(str)
    unique_groups = groups.nunique()
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    pairwise_weighting_tables: list[pd.DataFrame] = []

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
        held_out_domains = sorted(
            valid_df[config.domain_column].dropna().astype(str).unique().tolist()
        )
        held_out_label = "|".join(held_out_domains)

        if method_name in {"M3", "M4"}:
            adaptation_df, evaluation_df = _split_transductive_adaptation_test(
                valid_df, config
            )
            source_train_df = train_df
            harmonization_details: dict[str, Any] = {}
            if method_name == "M4":
                source_train_df, _, harmonization_details = (
                    prepare_method_train_test_frames(
                        method_name=method_name,
                        train_df=train_df,
                        test_df=evaluation_df,
                        schema=schema,
                        config=config,
                    )
                )
            y_prob, transductive_details = predict_transductive_blend(
                method_name=method_name,
                source_df=source_train_df,
                adaptation_df=adaptation_df,
                evaluation_df=evaluation_df,
                schema=schema,
                model_spec=model_spec,
                config=config,
            )
            method_details = {**harmonization_details, **transductive_details}
            metrics = compute_binary_metrics(
                evaluation_df[config.target_column].to_numpy(dtype=int),
                y_prob,
                threshold=config.threshold,
                ece_bins=config.ece_bins,
            )
            fold_rows.append(
                {
                    "method": method_name,
                    "model": model_name,
                    "model_display": pretty_model_name(model_name),
                    "evaluation_target_site": evaluation_target_site,
                    "evaluation_target_site_display": pretty_target_site(
                        evaluation_target_site
                    ),
                    "id_validation_scheme": f"{split_name}_TransductiveTrain1Test1",
                    "fold_index": fold_idx,
                    "held_out_source_domain": held_out_label,
                    "n_train": int(len(train_df)),
                    "n_valid": int(len(evaluation_df)),
                    **_method_diagnostics(method_details),
                    "roc_auc_id_fold": metrics["roc_auc"],
                    "f1_id_fold": metrics["f1"],
                    "balanced_accuracy_id_fold": metrics["balanced_accuracy"],
                    "ece_id_fold": metrics["ece"],
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "method": method_name,
                        "model": model_name,
                        "model_display": pretty_model_name(model_name),
                        "evaluation_target_site": evaluation_target_site,
                        "evaluation_target_site_display": pretty_target_site(
                            evaluation_target_site
                        ),
                        "evaluation_split": "id_validation",
                        "fold_index": fold_idx,
                        "evaluation_domain": held_out_label,
                        "subject_id": evaluation_df[config.id_column]
                        .astype(str)
                        .to_numpy(),
                        "domain_id": evaluation_df[config.domain_column]
                        .astype(str)
                        .to_numpy(),
                        "y_true": evaluation_df[config.target_column]
                        .astype(int)
                        .to_numpy(),
                        "y_prob": y_prob,
                        "y_pred": (y_prob >= config.threshold).astype(int),
                    }
                )
            )
            continue

        train_prepared, valid_prepared, method_details = (
            prepare_method_train_test_frames(
                method_name=method_name,
                train_df=train_df,
                test_df=valid_df,
                schema=schema,
                config=config,
            )
        )
        sample_weight, weighting_details, pairwise_weighting = _apply_m2_weighting(
            method_name=method_name,
            train_df=train_df,
            schema=schema,
            config=config,
            context={
                "model": model_name,
                "evaluation_target_site": evaluation_target_site,
                "evaluation_split": "id_validation_weighting",
                "fold_index": fold_idx,
                "held_out_source_domain": held_out_label,
            },
        )
        method_details.update(weighting_details)
        if not pairwise_weighting.empty:
            pairwise_weighting_tables.append(pairwise_weighting)
        pipeline = build_pipeline(schema, model_spec)
        _fit_pipeline_with_optional_weights(
            pipeline,
            train_prepared,
            train_df[config.target_column].astype(int),
            sample_weight,
        )
        y_prob = safe_predict_proba(pipeline, valid_prepared)
        metrics = compute_binary_metrics(
            valid_df[config.target_column].to_numpy(dtype=int),
            y_prob,
            threshold=config.threshold,
            ece_bins=config.ece_bins,
        )

        fold_rows.append(
            {
                "method": method_name,
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
                **_method_diagnostics(method_details),
                "roc_auc_id_fold": metrics["roc_auc"],
                "f1_id_fold": metrics["f1"],
                "balanced_accuracy_id_fold": metrics["balanced_accuracy"],
                "ece_id_fold": metrics["ece"],
            }
        )

        predictions.append(
            pd.DataFrame(
                {
                    "method": method_name,
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
    id_metrics["id_validation_scheme"] = (
        f"{split_name}_TransductiveTrain1Test1"
        if method_name in {"M3", "M4"}
        else split_name
    )
    pairwise_weighting_df = (
        pd.concat(pairwise_weighting_tables, ignore_index=True)
        if pairwise_weighting_tables
        else pd.DataFrame()
    )
    return id_metrics, pd.DataFrame(fold_rows), prediction_df, pairwise_weighting_df


def run_target_ood_evaluation(
    *,
    method_name: str,
    source_df: pd.DataFrame,
    target_train_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    schema: dict[str, Any],
    model_name: str,
    config: ExperimentConfig,
    evaluation_target_site: str,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Evaluate the held-out target subset with the configured adaptation strategy."""
    model_spec = build_model_specs(config)[model_name]

    if method_name in {"M3", "M4"}:
        source_train_df = source_df
        harmonization_details: dict[str, Any] = {}
        if method_name == "M4":
            source_train_df, _, harmonization_details = (
                prepare_method_train_test_frames(
                    method_name=method_name,
                    train_df=source_df,
                    test_df=target_test_df,
                    schema=schema,
                    config=config,
                )
            )
        y_prob, transductive_details = predict_transductive_blend(
            method_name=method_name,
            source_df=source_train_df,
            adaptation_df=target_train_df,
            evaluation_df=target_test_df,
            schema=schema,
            model_spec=model_spec,
            config=config,
        )
        method_details = {**harmonization_details, **transductive_details}
        metrics = compute_binary_metrics(
            target_test_df[config.target_column].to_numpy(dtype=int),
            y_prob,
            threshold=config.threshold,
            ece_bins=config.ece_bins,
        )
        prediction_df = pd.DataFrame(
            {
                "method": method_name,
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
                "domain_id": target_test_df[config.domain_column]
                .astype(str)
                .to_numpy(),
                "y_true": target_test_df[config.target_column].astype(int).to_numpy(),
                "y_prob": y_prob,
                "y_pred": (y_prob >= config.threshold).astype(int),
            }
        )
        return metrics, prediction_df, method_details, pd.DataFrame()

    source_prepared, target_test_prepared, method_details = (
        prepare_method_train_test_frames(
            method_name=method_name,
            train_df=source_df,
            test_df=target_test_df,
            schema=schema,
            config=config,
        )
    )
    sample_weight, weighting_details, pairwise_weighting = _apply_m2_weighting(
        method_name=method_name,
        train_df=source_df,
        schema=schema,
        config=config,
        context={
            "model": model_name,
            "evaluation_target_site": evaluation_target_site,
            "evaluation_split": "ood_training_weighting",
            "fold_index": pd.NA,
            "held_out_source_domain": "",
        },
    )
    method_details.update(weighting_details)
    pipeline = build_pipeline(schema, model_spec)
    _fit_pipeline_with_optional_weights(
        pipeline,
        source_prepared,
        source_df[config.target_column].astype(int),
        sample_weight,
    )
    y_prob = safe_predict_proba(pipeline, target_test_prepared)
    metrics = compute_binary_metrics(
        target_test_df[config.target_column].to_numpy(dtype=int),
        y_prob,
        threshold=config.threshold,
        ece_bins=config.ece_bins,
    )
    prediction_df = pd.DataFrame(
        {
            "method": method_name,
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
    return metrics, prediction_df, method_details, pairwise_weighting


def _run_single_target(
    *,
    split: TargetSiteSplit,
    schema: dict[str, Any],
    config: ExperimentConfig,
    logger: logging.Logger,
) -> tuple[
    list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]
]:
    result_rows: list[dict[str, Any]] = []
    fold_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    pairwise_weighting_tables: list[pd.DataFrame] = []

    source_domains = sorted(
        split.source_df[config.domain_column].dropna().astype(str).unique().tolist()
    )
    for method_name in config.enabled_methods:
        for model_name in build_model_specs(config):
            logger.info(
                "Running %s | %s on target site %s",
                method_name,
                model_name,
                split.target_site,
            )
            id_metrics, id_fold_df, id_prediction_df, id_pairwise_weighting_df = (
                run_source_id_validation(
                    method_name=method_name,
                    source_df=split.source_df,
                    schema=schema,
                    model_name=model_name,
                    config=config,
                    logger=logger,
                    evaluation_target_site=split.target_site,
                )
            )
            (
                ood_metrics,
                ood_prediction_df,
                method_details,
                ood_pairwise_weighting_df,
            ) = run_target_ood_evaluation(
                method_name=method_name,
                source_df=split.source_df,
                target_train_df=split.target_train_df,
                target_test_df=split.target_test_df,
                schema=schema,
                model_name=model_name,
                config=config,
                evaluation_target_site=split.target_site,
            )
            if not id_pairwise_weighting_df.empty:
                pairwise_weighting_tables.append(id_pairwise_weighting_df)
            if not ood_pairwise_weighting_df.empty:
                pairwise_weighting_tables.append(ood_pairwise_weighting_df)

            result_rows.append(
                {
                    "method": method_name,
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
                    **_method_diagnostics(method_details),
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

    return result_rows, fold_tables, prediction_tables, pairwise_weighting_tables


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


def run_transportability_experiment(
    config: ExperimentConfig, logger: logging.Logger
) -> ExperimentOutputs:
    """Evaluate the configured methods with the original source and target partitions."""
    dataset_df, schema = load_dataset(config, logger)
    target_splits = build_all_target_splits(dataset_df, config=config, logger=logger)

    result_rows: list[dict[str, Any]] = []
    id_fold_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    m2_pairwise_tables: list[pd.DataFrame] = []
    split_manifest_rows = [split.manifest_row for split in target_splits]
    split_assignment_tables = [split.assignment_df for split in target_splits]

    for split in target_splits:
        rows, fold_tables, predictions, pairwise_tables = _run_single_target(
            split=split,
            schema=schema,
            config=config,
            logger=logger,
        )
        result_rows.extend(rows)
        id_fold_tables.extend(fold_tables)
        prediction_tables.extend(predictions)
        m2_pairwise_tables.extend(pairwise_tables)

    target_site_results = (
        pd.DataFrame(result_rows)
        .sort_values(["method", "model", "target_site"])
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
    m2_pairwise_weighting = (
        pd.concat(m2_pairwise_tables, ignore_index=True)
        if m2_pairwise_tables
        else pd.DataFrame()
    )

    return ExperimentOutputs(
        dataset_df=dataset_df,
        schema=schema,
        target_site_results=target_site_results,
        id_validation_fold_metrics=id_validation_fold_metrics,
        prediction_records=prediction_records,
        split_manifest=split_manifest,
        split_assignments=split_assignments,
        model_summary=model_summary,
        m2_pairwise_weighting=m2_pairwise_weighting,
    )


def run_baseline_experiment(
    config: ExperimentConfig, logger: logging.Logger
) -> ExperimentOutputs:
    return run_transportability_experiment(config=config, logger=logger)
