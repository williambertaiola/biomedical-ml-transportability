"""Transportability experiment: models."""

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

from .analysis import extract_feature_importance
from .config import ExperimentConfig
from .metrics import compute_metrics
from .preprocessing import PreparedFoldData


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    family: str


@dataclass
class SavedModelBundle:
    model_name: str
    variant: str
    target_domain: str
    bundle: dict[str, Any]


@dataclass
class ConfusionArtifact:
    model_name: str
    variant: str
    target_domain: str
    class_labels: list[str]
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


def build_model_registry(config: ExperimentConfig, n_classes: int) -> list[ModelSpec]:
    models: list[ModelSpec] = [
        ModelSpec(
            name="logistic_regression",
            estimator=LogisticRegression(
                C=0.1,
                max_iter=6000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=config.random_state,
            ),
            family="linear",
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=config.random_state,
                n_jobs=config.n_jobs,
            ),
            family="tree",
        ),
    ]

    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
        xgb_kwargs = dict(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective=objective,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )
        if n_classes > 2:
            xgb_kwargs["num_class"] = n_classes
        models.append(
            ModelSpec(
                name="xgboost",
                estimator=XGBClassifier(**xgb_kwargs),
                family="boosting",
            )
        )
    elif importlib.util.find_spec("lightgbm") is not None:
        from lightgbm import LGBMClassifier

        objective = "multiclass" if n_classes > 2 else "binary"
        models.append(
            ModelSpec(
                name="lightgbm",
                estimator=LGBMClassifier(
                    n_estimators=350,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective=objective,
                    class_weight="balanced",
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
    class_labels: list[str],
    config: ExperimentConfig,
) -> FoldRunOutput:
    source_weights, class_weight_map = compute_balanced_sample_weights(
        prepared.y_source, n_classes=len(class_labels)
    )
    base_model = fit_estimator(
        model_spec.estimator,
        prepared.X_source_aligned,
        prepared.y_source,
        sample_weight=source_weights,
    )
    base_prob = predict_probabilities(
        base_model, prepared.X_target_eval, n_classes=len(class_labels)
    )
    base_pred = np.argmax(base_prob, axis=1)
    base_confidence = np.max(base_prob, axis=1)
    base_metrics = compute_metrics(
        prepared.y_target,
        base_prob,
        class_labels=class_labels,
        ece_bins=config.ece_bins,
    )

    threshold_table = compute_threshold_acceptance_table(
        confidence=base_confidence,
        predicted_labels=base_pred,
        source_class_counts=pd.Series(prepared.y_source).value_counts().to_dict(),
        config=config,
        target_domain=prepared.target_domain,
        model_name=model_spec.name,
        class_labels=class_labels,
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    threshold_rows = threshold_table.to_dict("records")
    feature_importance_rows: list[dict[str, Any]] = []
    confusion_artifacts: list[ConfusionArtifact] = []
    model_bundles: list[SavedModelBundle] = []

    selected_mask_placeholder = np.zeros(prepared.y_target.shape[0], dtype=bool)
    confidence_selection_mask = selected_mask_placeholder.copy()

    metric_rows.append(
        build_metric_row(
            metrics=base_metrics,
            model_name=model_spec.name,
            variant="coral_only",
            target_domain=prepared.target_domain,
            pseudo_selected_count=0,
            pseudo_selected_fraction=0.0,
            pseudo_mean_confidence=np.nan,
            pseudo_was_applied=False,
            pseudo_skip_reason="not_requested_for_variant",
        )
    )
    prediction_rows.extend(
        build_prediction_rows(
            target_domain=prepared.target_domain,
            model_name=model_spec.name,
            variant="coral_only",
            subject_ids=prepared.target_ids,
            y_true=prepared.y_target,
            y_prob=base_prob,
            selected_mask=selected_mask_placeholder,
            class_labels=class_labels,
        )
    )
    feature_importance_rows.extend(
        extract_feature_importance(
            model=base_model,
            model_name=model_spec.name,
            variant="coral_only",
            target_domain=prepared.target_domain,
            selected_feature_names=prepared.selected_feature_names,
            pca_feature_names=prepared.pca_feature_names,
            pca=prepared.source_preprocessor.pca,
        ).to_dict("records")
    )
    confusion_artifacts.append(
        ConfusionArtifact(
            model_name=model_spec.name,
            variant="coral_only",
            target_domain=prepared.target_domain,
            class_labels=class_labels,
            title=f"{model_spec.name} | CORAL only | target={prepared.target_domain}",
            matrix=confusion_matrix_from_predictions(
                prepared.y_target, base_pred, len(class_labels)
            ),
        )
    )
    model_bundles.append(
        SavedModelBundle(
            model_name=model_spec.name,
            variant="coral_only",
            target_domain=prepared.target_domain,
            bundle={
                "model": base_model,
                "class_labels": class_labels,
                "selected_feature_names": prepared.selected_feature_names,
                "pca_feature_names": prepared.pca_feature_names,
                "source_preprocessor": prepared.source_preprocessor,
                "coral_aligner": prepared.coral_aligner,
                "metadata": {
                    "model_name": model_spec.name,
                    "variant": "coral_only",
                    "target_domain": prepared.target_domain,
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
            class_labels=class_labels,
            target_domain=prepared.target_domain,
        )
        final_prob = predict_probabilities(
            pseudo_result.final_model,
            prepared.X_target_eval,
            n_classes=len(class_labels),
        )
        final_pred = np.argmax(final_prob, axis=1)
        final_metrics = compute_metrics(
            prepared.y_target,
            final_prob,
            class_labels=class_labels,
            ece_bins=config.ece_bins,
        )
        confidence_selection_mask = pseudo_result.selected_mask

        metric_rows.append(
            build_metric_row(
                metrics=final_metrics,
                model_name=model_spec.name,
                variant="coral_plus_pseudolabel",
                target_domain=prepared.target_domain,
                pseudo_selected_count=int(pseudo_result.selected_mask.sum()),
                pseudo_selected_fraction=float(pseudo_result.selected_mask.mean()),
                pseudo_mean_confidence=pseudo_result.mean_selected_confidence,
                pseudo_was_applied=pseudo_result.was_applied,
                pseudo_skip_reason=pseudo_result.skip_reason,
            )
        )
        prediction_rows.extend(
            build_prediction_rows(
                target_domain=prepared.target_domain,
                model_name=model_spec.name,
                variant="coral_plus_pseudolabel",
                subject_ids=prepared.target_ids,
                y_true=prepared.y_target,
                y_prob=final_prob,
                selected_mask=pseudo_result.selected_mask,
                class_labels=class_labels,
            )
        )
        feature_importance_rows.extend(
            extract_feature_importance(
                model=pseudo_result.final_model,
                model_name=model_spec.name,
                variant="coral_plus_pseudolabel",
                target_domain=prepared.target_domain,
                selected_feature_names=prepared.selected_feature_names,
                pca_feature_names=prepared.pca_feature_names,
                pca=prepared.source_preprocessor.pca,
            ).to_dict("records")
        )
        confusion_artifacts.append(
            ConfusionArtifact(
                model_name=model_spec.name,
                variant="coral_plus_pseudolabel",
                target_domain=prepared.target_domain,
                class_labels=class_labels,
                title=f"{model_spec.name} | CORAL + pseudo-labeling | target={prepared.target_domain}",
                matrix=confusion_matrix_from_predictions(
                    prepared.y_target, final_pred, len(class_labels)
                ),
            )
        )
        model_bundles.append(
            SavedModelBundle(
                model_name=model_spec.name,
                variant="coral_plus_pseudolabel",
                target_domain=prepared.target_domain,
                bundle={
                    "model": pseudo_result.final_model,
                    "class_labels": class_labels,
                    "selected_feature_names": prepared.selected_feature_names,
                    "pca_feature_names": prepared.pca_feature_names,
                    "source_preprocessor": prepared.source_preprocessor,
                    "coral_aligner": prepared.coral_aligner,
                    "metadata": {
                        "model_name": model_spec.name,
                        "variant": "coral_plus_pseudolabel",
                        "target_domain": prepared.target_domain,
                        "pseudo_selected_count": int(pseudo_result.selected_mask.sum()),
                        "pseudo_was_applied": bool(pseudo_result.was_applied),
                        "pseudo_skip_reason": pseudo_result.skip_reason,
                    },
                },
            )
        )

    confidence_rows.extend(
        build_confidence_rows(
            target_domain=prepared.target_domain,
            model_name=model_spec.name,
            subject_ids=prepared.target_ids,
            y_true=prepared.y_target,
            y_prob=base_prob,
            selected_mask=confidence_selection_mask,
            class_labels=class_labels,
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
    class_labels: list[str],
    target_domain: str,
) -> PseudoLabelResult:
    del target_domain
    base_prob = predict_probabilities(
        base_model, X_target_eval, n_classes=len(class_labels)
    )
    confidence = np.max(base_prob, axis=1)
    predicted_labels = np.argmax(base_prob, axis=1)
    source_class_counts = pd.Series(y_source).value_counts().to_dict()

    threshold_table = compute_threshold_acceptance_table(
        confidence=confidence,
        predicted_labels=predicted_labels,
        source_class_counts=source_class_counts,
        config=config,
        target_domain="",
        model_name=model_spec.name,
        class_labels=class_labels,
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
        dtype=np.float64,
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
    y: np.ndarray, n_classes: int
) -> tuple[np.ndarray, dict[int, float]]:
    y = np.asarray(y, dtype=int)
    class_counts = pd.Series(y).value_counts().to_dict()
    n_samples = max(len(y), 1)
    class_weight_map = {
        class_index: float(n_samples / (n_classes * class_counts.get(class_index, 1)))
        for class_index in range(n_classes)
        if class_counts.get(class_index, 0) > 0
    }
    sample_weights = np.array(
        [class_weight_map[int(label)] for label in y], dtype=np.float64
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


def predict_probabilities(model: Any, X: np.ndarray, n_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = np.asarray(model.predict_proba(X), dtype=np.float64)
        if prob.ndim == 1:
            prob = np.column_stack([1.0 - prob, prob])
        return prob

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X), dtype=np.float64)
        if decision.ndim == 1:
            prob_pos = 1.0 / (1.0 + np.exp(-decision))
            return np.column_stack([1.0 - prob_pos, prob_pos])
        shifted = decision - decision.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    raise ValueError(
        f"Model does not provide probabilistic predictions: {type(model).__name__}"
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
    candidate_df = pd.DataFrame(
        {
            "index": candidate_indices,
            "predicted_label": predicted_labels[candidate_indices],
            "confidence": confidence[candidate_indices],
        }
    ).sort_values("confidence", ascending=False)

    selected_parts: list[pd.DataFrame] = []
    for label, group in candidate_df.groupby("predicted_label", dropna=False):
        source_count = int(source_class_counts.get(int(label), 0))
        class_cap = max(
            1, int(np.ceil(config.pseudo_class_cap_ratio * max(source_count, 1)))
        )
        selected_parts.append(group.head(class_cap))

    selected_df = pd.concat(selected_parts, axis=0, ignore_index=True)
    selected_df = selected_df.sort_values("confidence", ascending=False).head(max_total)
    return selected_df["index"].to_numpy(dtype=int)


def compute_threshold_acceptance_table(
    confidence: np.ndarray,
    predicted_labels: np.ndarray,
    source_class_counts: dict[int, int],
    config: ExperimentConfig,
    target_domain: str,
    model_name: str,
    class_labels: list[str],
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
        row: dict[str, Any] = {
            "target_domain": target_domain,
            "model_name": model_name,
            "threshold": float(threshold),
            "n_raw_candidates": raw_candidates,
            "n_selected_after_caps": int(len(selected_indices)),
            "selected_fraction": float(len(selected_indices) / max(len(confidence), 1)),
        }
        if len(selected_indices) > 0:
            selected_labels = (
                pd.Series(predicted_labels[selected_indices]).value_counts().to_dict()
            )
            for class_index, class_label in enumerate(class_labels):
                row[f"selected_{sanitize_label(class_label)}"] = int(
                    selected_labels.get(class_index, 0)
                )
        rows.append(row)
    return pd.DataFrame(rows)


def build_metric_row(
    metrics: dict[str, Any],
    model_name: str,
    variant: str,
    target_domain: str,
    pseudo_selected_count: int,
    pseudo_selected_fraction: float,
    pseudo_mean_confidence: float,
    pseudo_was_applied: bool,
    pseudo_skip_reason: str,
) -> dict[str, Any]:
    row = {
        "model_name": model_name,
        "variant": variant,
        "target_domain": target_domain,
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


def build_prediction_rows(
    target_domain: str,
    model_name: str,
    variant: str,
    subject_ids: pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selected_mask: np.ndarray,
    class_labels: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = np.argmax(y_prob, axis=1)
    confidence = np.max(y_prob, axis=1)
    inverse_labels = {idx: label for idx, label in enumerate(class_labels)}

    for idx, subject_id in enumerate(subject_ids.astype("string").tolist()):
        row = {
            "target_domain": target_domain,
            "model_name": model_name,
            "variant": variant,
            "subject_id": subject_id,
            "y_true": int(y_true[idx]),
            "y_pred": int(y_pred[idx]),
            "y_true_label": inverse_labels[int(y_true[idx])],
            "y_pred_label": inverse_labels[int(y_pred[idx])],
            "confidence": float(confidence[idx]),
            "selected_for_pseudolabeling": int(selected_mask[idx]),
            "is_correct": int(y_true[idx] == y_pred[idx]),
        }
        for class_index, class_label in enumerate(class_labels):
            row[f"prob_{sanitize_label(class_label)}"] = float(y_prob[idx, class_index])
        rows.append(row)
    return rows


def build_confidence_rows(
    target_domain: str,
    model_name: str,
    subject_ids: pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selected_mask: np.ndarray,
    class_labels: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = np.argmax(y_prob, axis=1)
    confidence = np.max(y_prob, axis=1)
    inverse_labels = {idx: label for idx, label in enumerate(class_labels)}

    for idx, subject_id in enumerate(subject_ids.astype("string").tolist()):
        rows.append(
            {
                "target_domain": target_domain,
                "model_name": model_name,
                "subject_id": subject_id,
                "y_true": int(y_true[idx]),
                "y_true_label": inverse_labels[int(y_true[idx])],
                "pseudo_predicted_label": int(y_pred[idx]),
                "pseudo_predicted_label_text": inverse_labels[int(y_pred[idx])],
                "confidence": float(confidence[idx]),
                "selected_for_pseudolabeling": int(selected_mask[idx]),
                "threshold": float(threshold),
            }
        )
    return rows


def confusion_matrix_from_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_value, pred_value in zip(y_true, y_pred):
        matrix[int(true_value), int(pred_value)] += 1
    return matrix


def sanitize_label(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()
