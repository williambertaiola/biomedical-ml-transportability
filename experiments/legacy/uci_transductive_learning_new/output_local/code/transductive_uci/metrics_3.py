"""Transportability experiment: metrics 3."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, ece_bins: int = 10
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_prob),
        "pr_auc": _safe_pr_auc(y_true, y_prob),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": float(expected_calibration_error(y_true, y_prob, n_bins=ece_bins)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_samples": int(len(y_true)),
        "target_prevalence": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(y_pred)),
    }
    return metrics


def calibration_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, ece_bins: int = 10
) -> dict[str, float]:
    return {
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": float(expected_calibration_error(y_true, y_prob, n_bins=ece_bins)),
    }


def calibration_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    model_name: str | None = None,
    variant: str | None = None,
    target_site: str | None = None,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, object]] = []

    for idx in range(n_bins):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        n_bin = int(mask.sum())
        if n_bin == 0:
            rows.append(
                {
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "bin_mid": float((left + right) / 2.0),
                    "mean_predicted_probability": np.nan,
                    "empirical_positive_rate": np.nan,
                    "n_samples": 0,
                    "model_name": model_name,
                    "variant": variant,
                    "target_site": target_site,
                }
            )
            continue

        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "bin_mid": float((left + right) / 2.0),
                "mean_predicted_probability": float(y_prob[mask].mean()),
                "empirical_positive_rate": float(y_true[mask].mean()),
                "n_samples": n_bin,
                "model_name": model_name,
                "variant": variant,
                "target_site": target_site,
            }
        )

    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = max(len(y_true), 1)

    for idx in range(n_bins):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        if not np.any(mask):
            continue

        avg_confidence = float(y_prob[mask].mean())
        avg_accuracy = float(y_true[mask].mean())
        ece += (mask.sum() / total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))
