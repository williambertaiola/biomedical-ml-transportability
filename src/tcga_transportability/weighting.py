"""Transportability pipeline module: weighting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tcga_transportability.config import ExperimentConfig
from tcga_transportability.metrics import compute_binary_metrics
from tcga_transportability.models import ModelSpec, build_pipeline, safe_predict_proba


EPSILON = 1e-8


@dataclass(frozen=True)
class DomainWeightingResult:
    sample_weight: np.ndarray
    domain_weights: dict[str, float]
    transfer_scores: dict[str, float]
    pairwise_scores: pd.DataFrame
    details: dict[str, Any]


def _m2_reference_model(config: ExperimentConfig) -> ModelSpec:
    return ModelSpec(
        name="M2ReferenceLogisticRegression",
        estimator=LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            random_state=config.random_seed,
        ),
        scale_numeric=True,
    )


def _score_transfer(
    *,
    train_domain_df: pd.DataFrame,
    test_domain_df: pd.DataFrame,
    schema: dict[str, Any],
    config: ExperimentConfig,
) -> dict[str, float]:
    y_train = train_domain_df[config.target_column].astype(int).to_numpy()
    y_test = test_domain_df[config.target_column].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2 or len(test_domain_df) == 0:
        return {
            "roc_auc": np.nan,
            "balanced_accuracy": np.nan,
            "transfer_score": np.nan,
        }

    pipeline = build_pipeline(schema, _m2_reference_model(config), config)
    pipeline.fit(train_domain_df, y_train)
    y_prob = safe_predict_proba(pipeline, test_domain_df)
    metrics = compute_binary_metrics(
        y_test,
        y_prob,
        threshold=config.threshold,
        ece_bins=config.ece_bins,
    )
    transfer_score = metrics["roc_auc"]
    if not np.isfinite(transfer_score):
        transfer_score = metrics["balanced_accuracy"]
    return {
        "roc_auc": float(metrics["roc_auc"])
        if np.isfinite(metrics["roc_auc"])
        else np.nan,
        "balanced_accuracy": (
            float(metrics["balanced_accuracy"])
            if np.isfinite(metrics["balanced_accuracy"])
            else np.nan
        ),
        "transfer_score": float(transfer_score)
        if np.isfinite(transfer_score)
        else np.nan,
    }


def _estimate_pairwise_source_transfer(
    *,
    train_df: pd.DataFrame,
    schema: dict[str, Any],
    config: ExperimentConfig,
) -> pd.DataFrame:
    domains = sorted(
        train_df[config.domain_column].dropna().astype(str).unique().tolist()
    )
    rows: list[dict[str, Any]] = []
    for train_domain in domains:
        train_domain_df = train_df.loc[
            train_df[config.domain_column].astype(str) == train_domain
        ].reset_index(drop=True)
        for test_domain in domains:
            if test_domain == train_domain:
                continue
            test_domain_df = train_df.loc[
                train_df[config.domain_column].astype(str) == test_domain
            ].reset_index(drop=True)
            scores = _score_transfer(
                train_domain_df=train_domain_df,
                test_domain_df=test_domain_df,
                schema=schema,
                config=config,
            )
            rows.append(
                {
                    "train_source_domain": train_domain,
                    "validation_source_domain": test_domain,
                    "n_train_domain": int(len(train_domain_df)),
                    "n_validation_domain": int(len(test_domain_df)),
                    **scores,
                }
            )
    return pd.DataFrame(rows)


def _softmax_domain_weights(
    scores: dict[str, float], *, temperature: float, min_weight: float
) -> dict[str, float]:
    domains = list(scores)
    values = np.asarray([scores[domain] for domain in domains], dtype=float)
    finite_mask = np.isfinite(values)
    if len(domains) == 0:
        return {}
    if finite_mask.sum() < 2 or float(np.nanmax(values) - np.nanmin(values)) < EPSILON:
        return {domain: 1.0 for domain in domains}

    fill_value = float(np.nanmean(values[finite_mask]))
    values = np.where(finite_mask, values, fill_value)
    temp = max(float(temperature), EPSILON)
    logits = (values - float(np.max(values))) / temp
    raw = np.exp(logits)
    weights = raw / raw.sum() * len(domains)
    weights = np.clip(weights, float(min_weight), None)
    weights = weights / weights.sum() * len(domains)
    return {
        domain: float(weight) for domain, weight in zip(domains, weights, strict=False)
    }


def _sample_weights_from_domain_weights(
    train_df: pd.DataFrame,
    domain_weights: dict[str, float],
    config: ExperimentConfig,
) -> np.ndarray:
    domains = train_df[config.domain_column].astype(str)
    counts = domains.value_counts().to_dict()
    weights = np.ones(len(train_df), dtype=float)
    for domain, domain_weight in domain_weights.items():
        mask = domains.to_numpy() == domain
        weights[mask] = float(domain_weight) / max(float(counts.get(domain, 1)), 1.0)
    weights = weights / max(float(weights.mean()), EPSILON)
    return weights


def compute_m2_domain_weighting(
    *,
    train_df: pd.DataFrame,
    schema: dict[str, Any],
    config: ExperimentConfig,
) -> DomainWeightingResult:
    """Estimate source-domain weights using the original pairwise transfer scores."""
    domains = sorted(
        train_df[config.domain_column].dropna().astype(str).unique().tolist()
    )
    if len(domains) < 2:
        sample_weight = np.ones(len(train_df), dtype=float)
        domain_weights = {domain: 1.0 for domain in domains}
        return DomainWeightingResult(
            sample_weight=sample_weight,
            domain_weights=domain_weights,
            transfer_scores={domain: np.nan for domain in domains},
            pairwise_scores=pd.DataFrame(),
            details={
                "weighting": "m2_uniform_single_source_fallback",
                "weight_backend": "source_only_dashi_like_pairwise_transfer",
                "domain_weights": "|".join(f"{domain}:1.000000" for domain in domains),
                "domain_transfer_scores": "",
                "sample_weight_min": float(sample_weight.min())
                if len(sample_weight)
                else np.nan,
                "sample_weight_mean": float(sample_weight.mean())
                if len(sample_weight)
                else np.nan,
                "sample_weight_max": float(sample_weight.max())
                if len(sample_weight)
                else np.nan,
            },
        )

    pairwise_scores = _estimate_pairwise_source_transfer(
        train_df=train_df,
        schema=schema,
        config=config,
    )
    transfer_scores = (
        pairwise_scores.groupby("train_source_domain", dropna=False)["transfer_score"]
        .mean()
        .reindex(domains)
        .to_dict()
    )
    transfer_scores = {
        str(domain): float(score) if np.isfinite(score) else np.nan
        for domain, score in transfer_scores.items()
    }
    domain_weights = _softmax_domain_weights(
        transfer_scores,
        temperature=config.m2_weight_temperature,
        min_weight=config.m2_min_domain_weight,
    )
    sample_weight = _sample_weights_from_domain_weights(
        train_df, domain_weights, config
    )

    details = {
        "weighting": "m2_source_only_dashi_like",
        "weight_backend": "pairwise_source_transfer_softmax",
        "domain_weights": "|".join(
            f"{domain}:{domain_weights[domain]:.6f}" for domain in domains
        ),
        "domain_transfer_scores": "|".join(
            f"{domain}:{transfer_scores[domain]:.6f}"
            if np.isfinite(transfer_scores[domain])
            else f"{domain}:nan"
            for domain in domains
        ),
        "sample_weight_min": float(sample_weight.min())
        if len(sample_weight)
        else np.nan,
        "sample_weight_mean": float(sample_weight.mean())
        if len(sample_weight)
        else np.nan,
        "sample_weight_max": float(sample_weight.max())
        if len(sample_weight)
        else np.nan,
    }
    return DomainWeightingResult(
        sample_weight=sample_weight,
        domain_weights=domain_weights,
        transfer_scores=transfer_scores,
        pairwise_scores=pairwise_scores,
        details=details,
    )
