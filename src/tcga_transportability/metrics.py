"""Transportability pipeline module: metrics."""

from __future__ import annotations

from thesis_tools.metrics import compute_ece, compute_binary_metrics, id_ood_gap

import math

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def select_operating_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    default_threshold: float = 0.5,
) -> dict[str, float | str]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0 or len(y_prob) == 0:
        return {
            "decision_threshold": float(default_threshold),
            "threshold_selection_metric": "balanced_accuracy",
            "threshold_selection_tiebreak_metric": "f1",
            "threshold_selection_value": math.nan,
            "threshold_selection_f1": math.nan,
        }

    finite_mask = np.isfinite(y_prob)
    if not np.all(finite_mask):
        y_true = y_true[finite_mask]
        y_prob = y_prob[finite_mask]

    if len(y_true) == 0 or len(y_prob) == 0:
        return {
            "decision_threshold": float(default_threshold),
            "threshold_selection_metric": "balanced_accuracy",
            "threshold_selection_tiebreak_metric": "f1",
            "threshold_selection_value": math.nan,
            "threshold_selection_f1": math.nan,
        }

    max_prob = float(np.max(y_prob))
    above_max_threshold = float(np.nextafter(max_prob, 2.0))
    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.asarray([0.0], dtype=float),
                y_prob.astype(float, copy=False),
                np.asarray([above_max_threshold], dtype=float),
            ]
        )
    )

    best_record: tuple[float, float, float, float] | None = None
    best_threshold = float(default_threshold)
    for threshold in candidate_thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        try:
            balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
        except Exception:
            balanced_accuracy = math.nan
        try:
            f1_value = float(f1_score(y_true, y_pred, zero_division=0))
        except Exception:
            f1_value = math.nan

        score_tuple = (
            balanced_accuracy if not math.isnan(balanced_accuracy) else -math.inf,
            f1_value if not math.isnan(f1_value) else -math.inf,
            -abs(float(threshold) - float(default_threshold)),
            -float(threshold),
        )
        if best_record is None or score_tuple > best_record:
            best_record = score_tuple
            best_threshold = float(threshold)

    selected_pred = (y_prob >= best_threshold).astype(int)
    return {
        "decision_threshold": best_threshold,
        "threshold_selection_metric": "balanced_accuracy",
        "threshold_selection_tiebreak_metric": "f1",
        "threshold_selection_value": float(
            balanced_accuracy_score(y_true, selected_pred)
        ),
        "threshold_selection_f1": float(
            f1_score(y_true, selected_pred, zero_division=0)
        ),
    }
