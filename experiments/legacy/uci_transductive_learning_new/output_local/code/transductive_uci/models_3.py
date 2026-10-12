"""Transportability experiment: models 3."""

from __future__ import annotations

import importlib.util
import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .adaptation import PreparedFoldData
from .analysis import extract_feature_importance
from .config import ExperimentConfig
from .metrics import compute_metrics


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    family: str


@dataclass
class SavedModelBundle:
    model_name: str
    variant: str
    target_site: str
    bundle: dict[str, Any]


@dataclass
class ConfusionArtifact:
    model_name: str
    variant: str
    target_site: str
    title: str
    matrix: np.ndarray


@dataclass
class FoldRunOutput:
    metric_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]
    confidence_rows: list[dict[str, Any]]
    threshold_rows: list[dict[str, Any]]
    feature_importance_rows: list[dict[str, Any]]
    confusion_artifacts: list[ConfusionArtifact]
    model_bundles: list[SavedModelBundle]


@dataclass
class PseudoLabelResult:
    final_model: Any
    selected_indices: np.ndarray
    selected_mask: np.ndarray
    confidence: np.ndarray
    predicted_labels: np.ndarray
    threshold_table: pd.DataFrame
    n_raw_candidates: int
    mean_selected_confidence: float
    was_applied: bool
    skip_reason: str


def build_model_registry(config: ExperimentConfig) -> list[ModelSpec]:
    models: list[ModelSpec] = [
        ModelSpec(
            name="logistic_regression",
            estimator=LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=5000,
                random_state=config.random_state,
            ),
            family="linear",
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=config.random_state,
                n_jobs=config.n_jobs,
            ),
            family="tree",
        ),
    ]

    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        models.append(
            ModelSpec(
                name="xgboost",
                estimator=XGBClassifier(
                    n_estimators=350,
                    learning_rate=0.05,
                    max_depth=3,
                    min_child_weight=2,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=config.random_state,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=config.n_jobs,
                ),
                family="boosting",
            )
        )
    elif importlib.util.find_spec("lightgbm") is not None:
        from lightgbm import LGBMClassifier

        models.append(
            ModelSpec(
                name="lightgbm",
                estimator=LGBMClassifier(
                    n_estimators=350,
                    learning_rate=0.05,
                    max_depth=-1,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=config.random_state,
                    n_jobs=config.n_jobs,
                ),
                family="boosting",
            )
        )
    else:
        models.append(
            ModelSpec(
                name="hist_gradient_boosting",
                estimator=HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=4,
                    max_iter=350,
                    min_samples_leaf=12,
                    l2_regularization=0.05,
                    random_state=config.random_state,
                ),
                family="boosting_fallback",
            )
        )

    return models


def run_transductive_fold(
    model_spec: ModelSpec,
    prepared: PreparedFoldData,
    config: ExperimentConfig,
) -> FoldRunOutput:
    source_weights, class_weight_map = compute_balanced_sample_weights(
        prepared.y_source
    )
    base_model = fit_estimator(
        model_spec.estimator,
        prepared.X_source_aligned,
        prepared.y_source,
        sample_weight=source_weights,
    )
    base_prob = predict_positive_proba(base_model, prepared.X_target_eval)
    base_pred = (base_prob >= 0.5).astype(int)
    base_metrics = compute_metrics(
        prepared.y_target, base_prob, ece_bins=config.ece_bins
    )
    base_confidence = np.maximum(base_prob, 1.0 - base_prob)

    threshold_table = compute_threshold_acceptance_table(
        confidence=base_confidence,
        predicted_labels=base_pred,
        source_class_counts=pd.Series(prepared.y_source).value_counts().to_dict(),
        config=config,
        target_site=prepared.target_site,
        model_name=model_spec.name,
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    threshold_rows = threshold_table.to_dict("records")
    feature_importance_rows: list[dict[str, Any]] = []
    confusion_artifacts: list[ConfusionArtifact] = []
    model_bundles: list[SavedModelBundle] = []

    selected_mask_placeholder = np.zeros_like(base_pred, dtype=bool)
    confidence_selection_mask = selected_mask_placeholder.copy()
    metric_rows.append(
        _build_metric_row(
            metrics=base_metrics,
            model_name=model_spec.name,
            variant="coral_only",
            target_site=prepared.target_site,
            pseudo_selected_count=0,
            pseudo_selected_fraction=0.0,
            pseudo_mean_confidence=np.nan,
            pseudo_was_applied=False,
            pseudo_skip_reason="not_requested_for_variant",
        )
    )
    prediction_rows.extend(
        _build_prediction_rows(
            target_site=prepared.target_site,
            model_name=model_spec.name,
            variant="coral_only",
            subject_ids=prepared.target_ids,
            y_true=prepared.y_target,
            y_prob=base_prob,
            selected_mask=selected_mask_placeholder,
        )
    )
    feature_importance_rows.extend(
        extract_feature_importance(
            base_model,
            prepared.feature_names,
            model_name=model_spec.name,
            variant="coral_only",
            target_site=prepared.target_site,
        ).to_dict("records")
    )
    confusion_artifacts.append(
        ConfusionArtifact(
            model_name=model_spec.name,
            variant="coral_only",
            target_site=prepared.target_site,
            title=f"{model_spec.name} | CORAL only | target={prepared.target_site}",
            matrix=np.array(
                [
                    [base_metrics["tn"], base_metrics["fp"]],
                    [base_metrics["fn"], base_metrics["tp"]],
                ],
                dtype=int,
            ),
        )
    )
    model_bundles.append(
        SavedModelBundle(
            model_name=model_spec.name,
            variant="coral_only",
            target_site=prepared.target_site,
            bundle={
                "model": base_model,
                "feature_names": prepared.feature_names,
                "preprocessor": prepared.preprocessor,
                "variance_filter": prepared.variance_filter,
                "coral_aligner": prepared.coral_aligner,
                "metadata": {
                    "target_site": prepared.target_site,
                    "variant": "coral_only",
                    "model_name": model_spec.name,
                },
            },
        )
    )

    if config.pseudo_labeling_enabled:
        pseudo_result = run_pseudolabeling(
            model_spec=model_spec,
            base_model=base_model,
            X_source_aligned=prepared.X_source_aligned,
            y_source=prepared.y_source,
            X_target_eval=prepared.X_target_eval,
            source_weights=source_weights,
            class_weight_map=class_weight_map,
            config=config,
            target_site=prepared.target_site,
        )
        final_prob = predict_positive_proba(
            pseudo_result.final_model, prepared.X_target_eval
        )
        final_metrics = compute_metrics(
            prepared.y_target, final_prob, ece_bins=config.ece_bins
        )
        confidence_selection_mask = pseudo_result.selected_mask
        metric_rows.append(
            _build_metric_row(
                metrics=final_metrics,
                model_name=model_spec.name,
                variant="coral_plus_pseudolabel",
                target_site=prepared.target_site,
                pseudo_selected_count=int(pseudo_result.selected_mask.sum()),
                pseudo_selected_fraction=float(pseudo_result.selected_mask.mean()),
                pseudo_mean_confidence=pseudo_result.mean_selected_confidence,
                pseudo_was_applied=pseudo_result.was_applied,
                pseudo_skip_reason=pseudo_result.skip_reason,
            )
        )
        prediction_rows.extend(
            _build_prediction_rows(
                target_site=prepared.target_site,
                model_name=model_spec.name,
                variant="coral_plus_pseudolabel",
                subject_ids=prepared.target_ids,
                y_true=prepared.y_target,
                y_prob=final_prob,
                selected_mask=pseudo_result.selected_mask,
            )
        )
        feature_importance_rows.extend(
            extract_feature_importance(
                pseudo_result.final_model,
                prepared.feature_names,
                model_name=model_spec.name,
                variant="coral_plus_pseudolabel",
                target_site=prepared.target_site,
            ).to_dict("records")
        )
        confusion_artifacts.append(
            ConfusionArtifact(
                model_name=model_spec.name,
                variant="coral_plus_pseudolabel",
                target_site=prepared.target_site,
                title=f"{model_spec.name} | CORAL + pseudo-labeling | target={prepared.target_site}",
                matrix=np.array(
                    [
                        [final_metrics["tn"], final_metrics["fp"]],
                        [final_metrics["fn"], final_metrics["tp"]],
                    ],
                    dtype=int,
                ),
            )
        )
        model_bundles.append(
            SavedModelBundle(
                model_name=model_spec.name,
                variant="coral_plus_pseudolabel",
                target_site=prepared.target_site,
                bundle={
                    "model": pseudo_result.final_model,
                    "feature_names": prepared.feature_names,
                    "preprocessor": prepared.preprocessor,
                    "variance_filter": prepared.variance_filter,
                    "coral_aligner": prepared.coral_aligner,
                    "metadata": {
                        "target_site": prepared.target_site,
                        "variant": "coral_plus_pseudolabel",
                        "model_name": model_spec.name,
                        "pseudo_selected_count": int(pseudo_result.selected_mask.sum()),
                        "pseudo_was_applied": pseudo_result.was_applied,
                        "pseudo_skip_reason": pseudo_result.skip_reason,
                    },
                },
            )
        )

    confidence_rows.extend(
        _build_confidence_rows(
            target_site=prepared.target_site,
            model_name=model_spec.name,
            subject_ids=prepared.target_ids,
            y_true=prepared.y_target,
            y_prob=base_prob,
            selected_mask=confidence_selection_mask,
            threshold=config.pseudo_threshold,
        )
    )

    return FoldRunOutput(
        metric_rows=metric_rows,
        prediction_rows=prediction_rows,
        confidence_rows=confidence_rows,
        threshold_rows=threshold_rows,
        feature_importance_rows=feature_importance_rows,
        confusion_artifacts=confusion_artifacts,
        model_bundles=model_bundles,
    )


def run_pseudolabeling(
    model_spec: ModelSpec,
    base_model: Any,
    X_source_aligned: np.ndarray,
    y_source: np.ndarray,
    X_target_eval: np.ndarray,
    source_weights: np.ndarray,
    class_weight_map: dict[int, float],
    config: ExperimentConfig,
    target_site: str,
) -> PseudoLabelResult:
    del target_site  # metadata already stored outside the function

    base_prob = predict_positive_proba(base_model, X_target_eval)
    confidence = np.maximum(base_prob, 1.0 - base_prob)
    predicted_labels = (base_prob >= 0.5).astype(int)
    source_class_counts = pd.Series(y_source).value_counts().to_dict()

    threshold_table = compute_threshold_acceptance_table(
        confidence=confidence,
        predicted_labels=predicted_labels,
        source_class_counts=source_class_counts,
        config=config,
        target_site="",
        model_name=model_spec.name,
    )

    raw_candidate_mask = confidence >= config.pseudo_threshold
    raw_candidate_count = int(raw_candidate_mask.sum())
    selected_indices = select_pseudo_indices(
        confidence=confidence,
        predicted_labels=predicted_labels,
        source_class_counts=source_class_counts,
        threshold=config.pseudo_threshold,
        config=config,
    )
    selected_mask = np.zeros(X_target_eval.shape[0], dtype=bool)
    selected_mask[selected_indices] = True

    if raw_candidate_count == 0:
        return PseudoLabelResult(
            final_model=base_model,
            selected_indices=selected_indices,
            selected_mask=selected_mask,
            confidence=confidence,
            predicted_labels=predicted_labels,
            threshold_table=threshold_table,
            n_raw_candidates=0,
            mean_selected_confidence=np.nan,
            was_applied=False,
            skip_reason="no_high_confidence_candidates",
        )

    if len(selected_indices) < config.pseudo_min_samples:
        return PseudoLabelResult(
            final_model=base_model,
            selected_indices=selected_indices,
            selected_mask=selected_mask,
            confidence=confidence,
            predicted_labels=predicted_labels,
            threshold_table=threshold_table,
            n_raw_candidates=raw_candidate_count,
            mean_selected_confidence=float(confidence[selected_indices].mean())
            if len(selected_indices)
            else np.nan,
            was_applied=False,
            skip_reason="below_minimum_selected_samples",
        )

    X_augmented = np.vstack([X_source_aligned, X_target_eval[selected_indices]])
    y_pseudo = predicted_labels[selected_indices]
    y_augmented = np.concatenate([y_source, y_pseudo])
    pseudo_weights = np.array(
        [
            config.pseudo_sample_weight * class_weight_map.get(int(label), 1.0)
            for label in y_pseudo
        ],
        dtype=float,
    )
    augmented_weights = np.concatenate([source_weights, pseudo_weights])
    final_model = fit_estimator(
        model_spec.estimator,
        X_augmented,
        y_augmented,
        sample_weight=augmented_weights,
    )

    return PseudoLabelResult(
        final_model=final_model,
        selected_indices=selected_indices,
        selected_mask=selected_mask,
        confidence=confidence,
        predicted_labels=predicted_labels,
        threshold_table=threshold_table,
        n_raw_candidates=raw_candidate_count,
        mean_selected_confidence=float(confidence[selected_indices].mean()),
        was_applied=True,
        skip_reason="applied",
    )


def compute_balanced_sample_weights(
    y: np.ndarray,
) -> tuple[np.ndarray, dict[int, float]]:
    y = np.asarray(y, dtype=int)
    class_counts = pd.Series(y).value_counts().to_dict()
    n_samples = max(len(y), 1)
    n_classes = max(len(class_counts), 1)
    class_weight_map = {
        int(label): float(n_samples / (n_classes * count))
        for label, count in class_counts.items()
    }
    sample_weights = np.array(
        [class_weight_map[int(label)] for label in y], dtype=float
    )
    return sample_weights, class_weight_map


def fit_estimator(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
):
    model = clone(estimator)
    fit_signature = inspect.signature(model.fit)
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None and "sample_weight" in fit_signature.parameters:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X, y, **fit_kwargs)
    return model


def predict_positive_proba(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = np.asarray(model.predict_proba(X), dtype=float)
        if prob.ndim == 1:
            return prob
        if prob.shape[1] == 2:
            return prob[:, 1]
        return prob[:, -1]

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-decision))

    raise ValueError(
        f"Model does not provide probabilistic outputs: {type(model).__name__}"
    )


def select_pseudo_indices(
    confidence: np.ndarray,
    predicted_labels: np.ndarray,
    source_class_counts: dict[int, int],
    threshold: float,
    config: ExperimentConfig,
) -> np.ndarray:
    candidate_indices = np.where(confidence >= threshold)[0]
    if candidate_indices.size == 0:
        return np.array([], dtype=int)

    max_total = max(1, int(np.floor(config.pseudo_max_fraction * len(confidence))))
    selected_frames: list[pd.DataFrame] = []

    candidate_df = pd.DataFrame(
        {
            "index": candidate_indices,
            "predicted_label": predicted_labels[candidate_indices],
            "confidence": confidence[candidate_indices],
        }
    ).sort_values("confidence", ascending=False)

    for label, group in candidate_df.groupby("predicted_label", dropna=False):
        source_count = int(source_class_counts.get(int(label), 0))
        class_cap = max(
            1, int(np.ceil(config.pseudo_class_cap_ratio * max(source_count, 1)))
        )
        selected_frames.append(group.head(class_cap))

    selected_df = pd.concat(selected_frames, axis=0, ignore_index=True)
    selected_df = selected_df.sort_values("confidence", ascending=False).head(max_total)
    return selected_df["index"].to_numpy(dtype=int)


def compute_threshold_acceptance_table(
    confidence: np.ndarray,
    predicted_labels: np.ndarray,
    source_class_counts: dict[int, int],
    config: ExperimentConfig,
    target_site: str,
    model_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in config.confidence_threshold_grid:
        selected_indices = select_pseudo_indices(
            confidence=confidence,
            predicted_labels=predicted_labels,
            source_class_counts=source_class_counts,
            threshold=threshold,
            config=config,
        )
        raw_candidates = int(np.sum(confidence >= threshold))
        rows.append(
            {
                "target_site": target_site,
                "model_name": model_name,
                "threshold": float(threshold),
                "n_raw_candidates": raw_candidates,
                "n_selected_after_caps": int(len(selected_indices)),
                "selected_fraction": float(
                    len(selected_indices) / max(len(confidence), 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_metric_row(
    metrics: dict[str, Any],
    model_name: str,
    variant: str,
    target_site: str,
    pseudo_selected_count: int,
    pseudo_selected_fraction: float,
    pseudo_mean_confidence: float,
    pseudo_was_applied: bool,
    pseudo_skip_reason: str,
) -> dict[str, Any]:
    row = {
        "model_name": model_name,
        "variant": variant,
        "target_site": target_site,
        "pseudo_selected_count": int(pseudo_selected_count),
        "pseudo_selected_fraction": float(pseudo_selected_fraction),
        "pseudo_mean_confidence": float(pseudo_mean_confidence)
        if pd.notna(pseudo_mean_confidence)
        else np.nan,
        "pseudo_was_applied": bool(pseudo_was_applied),
        "pseudo_skip_reason": pseudo_skip_reason,
    }
    row.update(metrics)
    return row


def _build_prediction_rows(
    target_site: str,
    model_name: str,
    variant: str,
    subject_ids: pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selected_mask: np.ndarray,
) -> list[dict[str, Any]]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    rows: list[dict[str, Any]] = []

    for idx, subject_id in enumerate(subject_ids.astype(str).tolist()):
        rows.append(
            {
                "target_site": target_site,
                "model_name": model_name,
                "variant": variant,
                "subject_id": subject_id,
                "y_true": int(y_true[idx]),
                "y_pred": int(y_pred[idx]),
                "y_prob": float(y_prob[idx]),
                "confidence": float(confidence[idx]),
                "selected_for_pseudolabeling": int(selected_mask[idx]),
                "is_correct": int(y_true[idx] == y_pred[idx]),
            }
        )
    return rows


def _build_confidence_rows(
    target_site: str,
    model_name: str,
    subject_ids: pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selected_mask: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    rows: list[dict[str, Any]] = []

    for idx, subject_id in enumerate(subject_ids.astype(str).tolist()):
        rows.append(
            {
                "target_site": target_site,
                "model_name": model_name,
                "subject_id": subject_id,
                "y_true": int(y_true[idx]),
                "pseudo_predicted_label": int(y_pred[idx]),
                "pseudo_probability": float(y_prob[idx]),
                "confidence": float(confidence[idx]),
                "selected_for_pseudolabeling": int(selected_mask[idx]),
                "threshold": float(threshold),
            }
        )
    return rows
