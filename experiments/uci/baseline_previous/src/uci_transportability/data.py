"""Transportability experiment: data."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from uci_transportability.config import ExperimentConfig, TARGET_SITE_ORDER


@dataclass
class TargetSiteSplit:
    target_site: str
    source_df: pd.DataFrame
    target_train_df: pd.DataFrame
    target_test_df: pd.DataFrame
    assignment_df: pd.DataFrame
    manifest_row: dict[str, Any]
    warnings: list[str]


def _class_balance(series: pd.Series) -> dict[str, Any]:
    return {
        "n_class_0": int((series == 0).sum()),
        "n_class_1": int((series == 1).sum()),
        "positive_prevalence": round(float(series.mean()), 6),
    }


def load_schema(schema_path: Path) -> dict[str, Any]:
    return json.loads(schema_path.read_text(encoding="utf-8"))


def load_dataset(
    config: ExperimentConfig, logger: logging.Logger
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(config.input_path)
    schema = load_schema(config.schema_path)

    required_columns = {
        config.id_column,
        config.domain_column,
        config.target_column,
        *schema["predictive_feature_columns"],
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in input dataset: {missing}")

    df = df.copy()
    df[config.domain_column] = df[config.domain_column].astype("string")
    df[config.id_column] = df[config.id_column].astype("string")
    df[config.target_column] = df[config.target_column].astype(int)

    unique_targets = sorted(df[config.target_column].dropna().unique().tolist())
    if unique_targets != [0, 1]:
        raise ValueError(
            f"The rebuilt M0 baseline expects a binary target encoded as 0/1. Found: {unique_targets}"
        )

    observed_sites = sorted(
        df[config.domain_column].dropna().astype(str).unique().tolist()
    )
    logger.info(
        "Loaded dataset with shape %s and %s observed sites: %s",
        df.shape,
        len(observed_sites),
        ", ".join(observed_sites),
    )
    return df, schema


def _split_target_site(
    domain_df: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings_out: list[str] = []
    y = domain_df[config.target_column]
    class_counts = y.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    use_stratify = y.nunique() > 1 and min_class_count >= 2

    if not use_stratify:
        warnings_out.append("target_split_used_unstratified_shuffle_fallback")

    train_df, test_df = train_test_split(
        domain_df,
        test_size=config.target_test_size,
        random_state=config.random_seed,
        shuffle=True,
        stratify=y if use_stratify else None,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), warnings_out


def build_target_site_split(
    df: pd.DataFrame,
    *,
    target_site: str,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> TargetSiteSplit:
    target_mask = df[config.domain_column].astype(str) == str(target_site)
    target_df = df.loc[target_mask].copy().reset_index(drop=True)
    source_df = df.loc[~target_mask].copy().reset_index(drop=True)

    target_train_df, target_test_df, warnings_out = _split_target_site(
        target_df, config
    )

    assignment_df = pd.concat(
        [
            source_df[
                [config.id_column, config.domain_column, config.target_column]
            ].assign(
                evaluation_target_site=target_site,
                partition="source",
            ),
            target_train_df[
                [config.id_column, config.domain_column, config.target_column]
            ].assign(
                evaluation_target_site=target_site,
                partition="target_train_1",
            ),
            target_test_df[
                [config.id_column, config.domain_column, config.target_column]
            ].assign(
                evaluation_target_site=target_site,
                partition="target_test_1",
            ),
        ],
        ignore_index=True,
    )

    manifest_row = {
        "evaluation_target_site": target_site,
        "n_source_samples": int(len(source_df)),
        "n_target_train_1": int(len(target_train_df)),
        "n_target_test_1": int(len(target_test_df)),
        "n_source_domains": int(source_df[config.domain_column].nunique()),
        "source_domains": "|".join(
            sorted(
                source_df[config.domain_column].dropna().astype(str).unique().tolist()
            )
        ),
        "source_class_balance": json.dumps(
            _class_balance(source_df[config.target_column]), sort_keys=True
        ),
        "target_train_1_class_balance": json.dumps(
            _class_balance(target_train_df[config.target_column]), sort_keys=True
        ),
        "target_test_1_class_balance": json.dumps(
            _class_balance(target_test_df[config.target_column]), sort_keys=True
        ),
        "warnings": " | ".join(warnings_out),
    }

    logger.info(
        "Prepared target site %s | source=%s | train_1=%s | test_1=%s",
        target_site,
        len(source_df),
        len(target_train_df),
        len(target_test_df),
    )

    return TargetSiteSplit(
        target_site=target_site,
        source_df=source_df,
        target_train_df=target_train_df,
        target_test_df=target_test_df,
        assignment_df=assignment_df,
        manifest_row=manifest_row,
        warnings=warnings_out,
    )


def build_all_target_splits(
    df: pd.DataFrame,
    *,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> list[TargetSiteSplit]:
    observed_sites = sorted(
        df[config.domain_column].dropna().astype(str).unique().tolist()
    )
    ordered_sites = [site for site in TARGET_SITE_ORDER if site in observed_sites]
    extra_sites = [site for site in observed_sites if site not in ordered_sites]
    all_sites = ordered_sites + extra_sites
    return [
        build_target_site_split(
            df, target_site=target_site, config=config, logger=logger
        )
        for target_site in all_sites
    ]
