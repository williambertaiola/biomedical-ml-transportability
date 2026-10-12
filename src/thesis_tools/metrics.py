"""Shared binary metrics; copied exactly from the original UCI/ABIDE/TCGA implementations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return math.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    error = 0.0
    total = len(y_true)
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        accuracy = float(np.mean(y_true[mask]))
        confidence = float(np.mean(y_prob[mask]))
        error += (np.sum(mask) / total) * abs(accuracy - confidence)
    return float(error)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    ece_bins: int = 10,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": math.nan,
        "f1": math.nan,
        "balanced_accuracy": math.nan,
        "ece": math.nan,
    }

    if len(y_true) == 0:
        return metrics

    if len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))

    try:
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        metrics["f1"] = math.nan

    try:
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        metrics["balanced_accuracy"] = math.nan

    metrics["ece"] = compute_ece(y_true, y_prob, n_bins=ece_bins)
    return metrics


def id_ood_gap(id_value: float, ood_value: float) -> float:
    if pd.isna(id_value) or pd.isna(ood_value):
        return math.nan
    return float(id_value - ood_value)
