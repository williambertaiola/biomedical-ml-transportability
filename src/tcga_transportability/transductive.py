"""Transportability pipeline module: transductive."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tcga_transportability.config import ExperimentConfig
from tcga_transportability.models import ModelSpec, build_pipeline, safe_predict_proba


def predict_transductive_blend(
    *,
    method_name: str,
    source_df: pd.DataFrame,
    adaptation_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    schema: dict[str, Any],
    model_spec: ModelSpec,
    config: ExperimentConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_pipeline = build_pipeline(schema, model_spec, config)
    source_pipeline.fit(source_df, source_df[config.target_column].astype(int))
    source_prob = safe_predict_proba(source_pipeline, evaluation_df)

    method_label = str(method_name).upper()
    transductive_name = (
        "m4_combat_source_plus_target_train_1_adapter"
        if method_label == "M4"
        else "m3_source_plus_target_train_1_adapter"
    )
    details: dict[str, Any] = {
        "transductive": transductive_name,
        "transductive_backend": "same_model_probability_blend",
        "transductive_blend_weight": float(config.m3_target_blend_weight),
        "n_transductive_adaptation_samples": int(len(adaptation_df)),
        "transductive_adapter_status": "source_only_fallback",
    }

    if len(adaptation_df) == 0 or adaptation_df[config.target_column].nunique() < 2:
        return source_prob, details

    adapter_pipeline = build_pipeline(schema, model_spec, config)
    adapter_pipeline.fit(adaptation_df, adaptation_df[config.target_column].astype(int))
    adapter_prob = safe_predict_proba(adapter_pipeline, evaluation_df)
    alpha = float(np.clip(config.m3_target_blend_weight, 0.0, 1.0))
    blended_prob = (1.0 - alpha) * source_prob + alpha * adapter_prob
    details["transductive_adapter_status"] = "adapted_on_train_1"
    return np.clip(blended_prob, 0.0, 1.0), details
