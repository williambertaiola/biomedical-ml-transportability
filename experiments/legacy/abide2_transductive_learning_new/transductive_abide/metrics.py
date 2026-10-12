"""Transportability experiment: metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_labels: list[str],
    ece_bins: int = 10,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = np.argmax(y_prob, axis=1)
    classes = np.arange(len(class_labels), dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(_safe_balanced_accuracy(y_true, y_pred)),
        "precision_macro": float(
            precision_score(
                y_true, y_pred, labels=classes, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true, y_pred, labels=classes, average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        ),
        "roc_auc_macro": float(safe_multiclass_roc_auc(y_true, y_prob, classes)),
        "pr_auc_macro": float(safe_multiclass_pr_auc(y_true, y_prob, classes)),
        "brier_score": float(multiclass_brier_score(y_true, y_prob, classes)),
        "ece": float(expected_calibration_error(y_true, y_prob, n_bins=ece_bins)),
        "n_samples": int(len(y_true)),
        "n_true_classes": int(np.unique(y_true).size),
        "predicted_class_entropy": float(class_entropy(y_pred, classes)),
    }

    for row_idx, true_class in enumerate(class_labels):
        for col_idx, pred_class in enumerate(class_labels):
            metrics[
                f"cm_true_{sanitize_label(true_class)}_pred_{sanitize_label(pred_class)}"
            ] = int(cm[row_idx, col_idx])
    return metrics


def top_label_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = np.argmax(y_prob, axis=1)
    confidence = np.max(y_prob, axis=1)
    correctness = (y_true == y_pred).astype(int)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, object]] = []

    for idx in range(n_bins):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)

        row = {
            "bin_left": float(left),
            "bin_right": float(right),
            "bin_mid": float((left + right) / 2.0),
            "mean_confidence": np.nan,
            "empirical_accuracy": np.nan,
            "n_samples": int(mask.sum()),
        }
        if metadata:
            row.update(metadata)

        if np.any(mask):
            row["mean_confidence"] = float(confidence[mask].mean())
            row["empirical_accuracy"] = float(correctness[mask].mean())
        rows.append(row)

    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = np.argmax(y_prob, axis=1)
    confidence = np.max(y_prob, axis=1)
    correctness = (y_true == y_pred).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = max(len(y_true), 1)

    for idx in range(n_bins):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if not np.any(mask):
            continue
        avg_confidence = float(confidence[mask].mean())
        avg_accuracy = float(correctness[mask].mean())
        ece += (mask.sum() / total) * abs(avg_accuracy - avg_confidence)
    return float(ece)


def safe_multiclass_roc_auc(
    y_true: np.ndarray, y_prob: np.ndarray, classes: np.ndarray
) -> float:
    valid_scores: list[float] = []
    for class_index in classes:
        binary_true = (y_true == class_index).astype(int)
        if binary_true.min() == binary_true.max():
            continue
        valid_scores.append(float(roc_auc_score(binary_true, y_prob[:, class_index])))
    if len(valid_scores) < 2:
        return float("nan")
    return float(np.mean(valid_scores))


def safe_multiclass_pr_auc(
    y_true: np.ndarray, y_prob: np.ndarray, classes: np.ndarray
) -> float:
    valid_scores: list[float] = []
    for class_index in classes:
        binary_true = (y_true == class_index).astype(int)
        if binary_true.min() == binary_true.max():
            continue
        valid_scores.append(
            float(average_precision_score(binary_true, y_prob[:, class_index]))
        )
    if len(valid_scores) < 2:
        return float("nan")
    return float(np.mean(valid_scores))


def multiclass_brier_score(
    y_true: np.ndarray, y_prob: np.ndarray, classes: np.ndarray
) -> float:
    one_hot = np.zeros((len(y_true), len(classes)), dtype=np.float64)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def class_entropy(y_pred: np.ndarray, classes: np.ndarray) -> float:
    counts = np.bincount(y_pred, minlength=len(classes)).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


def _safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        present_classes = np.unique(y_true)
        recalls = []
        for class_index in present_classes:
            mask = y_true == class_index
            recalls.append(float(np.mean(y_pred[mask] == class_index)))
        return float(np.mean(recalls)) if recalls else float("nan")


def sanitize_label(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()
