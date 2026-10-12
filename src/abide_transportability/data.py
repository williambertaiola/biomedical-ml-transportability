"""Transportability pipeline module: data."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from abide_transportability.config import ExperimentConfig, TARGET_SITE_ORDER


STRICT_NUMERIC_FEATURES = ("age_at_scan", "fiq")
STRICT_CATEGORICAL_FEATURES = (
    "sex_code",
    "handedness_category_code",
    "eye_status_at_scan_code",
)
RAW_TARGET_CODE_COLUMN = "disease_level_code"
RAW_TARGET_LABEL_COLUMN = "disease_level"


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
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    if "predictive_feature_columns" in payload:
        return payload
    if "strict_variant" not in payload:
        raise ValueError(f"Unsupported ABIDE schema/blueprint format: {schema_path}")

    feature_columns = payload["strict_variant"]["feature_columns"]
    return {
        "predictive_feature_columns": feature_columns,
        "numeric_features": [
            column for column in STRICT_NUMERIC_FEATURES if column in feature_columns
        ],
        "categorical_features": [
            column
            for column in STRICT_CATEGORICAL_FEATURES
            if column in feature_columns
        ],
        "recommended_for": payload["strict_variant"].get("recommended_for", ""),
        "source_blueprint_path": str(schema_path),
    }


def _build_binary_target(df: pd.DataFrame, config: ExperimentConfig) -> pd.Series:
    if config.target_column in df.columns:
        target = pd.to_numeric(df[config.target_column], errors="coerce")
        if target.isna().any():
            raise ValueError(
                f"Column `{config.target_column}` contains non-numeric target values."
            )
        return target.astype(int)

    if RAW_TARGET_CODE_COLUMN in df.columns:
        raw_codes = pd.to_numeric(df[RAW_TARGET_CODE_COLUMN], errors="coerce")
        if raw_codes.isna().any():
            raise ValueError(
                f"Column `{RAW_TARGET_CODE_COLUMN}` contains non-numeric values."
            )
        unique_codes = sorted(raw_codes.dropna().astype(int).unique().tolist())
        if not set(unique_codes).issubset({1, 2}):
            raise ValueError(
                "ABIDE target construction expected `disease_level_code` encoded as 1=ASD, 2=Control. "
                f"Found codes: {unique_codes}"
            )
        return raw_codes.astype(int).eq(1).astype(int)

    if RAW_TARGET_LABEL_COLUMN in df.columns:
        labels = df[RAW_TARGET_LABEL_COLUMN].astype("string").str.strip().str.lower()
        if labels.isna().any():
            raise ValueError(
                f"Column `{RAW_TARGET_LABEL_COLUMN}` contains missing target labels."
            )
        allowed = {"asd", "control"}
        unexpected = sorted(set(labels.dropna().unique().tolist()).difference(allowed))
        if unexpected:
            raise ValueError(
                "ABIDE target construction expected `disease_level` labels ASD/Control. "
                f"Found unexpected labels: {unexpected}"
            )
        return labels.eq("asd").astype(int)

    raise ValueError(
        f"Unable to derive binary target `{config.target_column}`. "
        f"Expected one of `{RAW_TARGET_CODE_COLUMN}` or `{RAW_TARGET_LABEL_COLUMN}` in the input dataset."
    )


def _ordered_candidate_sites(df: pd.DataFrame, config: ExperimentConfig) -> list[str]:
    observed_sites = sorted(
        df[config.domain_column].dropna().astype(str).unique().tolist()
    )
    if config.target_sites:
        return [site for site in config.target_sites if site in observed_sites]
    ordered_sites = [site for site in TARGET_SITE_ORDER if site in observed_sites]
    extra_sites = [site for site in observed_sites if site not in ordered_sites]
    return ordered_sites + extra_sites


def summarize_target_site_eligibility(
    df: pd.DataFrame,
    *,
    config: ExperimentConfig,
) -> tuple[list[str], list[dict[str, Any]]]:
    eligible_sites: list[str] = []
    excluded_sites: list[dict[str, Any]] = []

    for site in _ordered_candidate_sites(df, config):
        site_df = df.loc[df[config.domain_column].astype(str) == str(site)].copy()
        y = site_df[config.target_column].astype(int)
        class_counts = y.value_counts().sort_index().to_dict()
        min_class_count = int(y.value_counts().min()) if not y.empty else 0
        exclusion_reasons: list[str] = []
        if y.nunique() < 2:
            exclusion_reasons.append("single_class_target_site")
        if min_class_count < config.min_target_class_count:
            exclusion_reasons.append(
                f"min_class_count_below_{config.min_target_class_count}"
            )

        if exclusion_reasons:
            excluded_sites.append(
                {
                    "site": str(site),
                    "n_samples": int(len(site_df)),
                    "class_counts": class_counts,
                    "reasons": exclusion_reasons,
                }
            )
            continue

        eligible_sites.append(str(site))

    return eligible_sites, excluded_sites


def load_dataset(
    config: ExperimentConfig, logger: logging.Logger
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate the preprocessed dataset without changing feature values."""
    df = pd.read_csv(config.input_path)
    schema = load_schema(config.schema_path)

    df = df.copy()
    df[config.target_column] = _build_binary_target(df, config)

    required_columns = {
        config.id_column,
        config.domain_column,
        config.target_column,
        *schema["predictive_feature_columns"],
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in input dataset: {missing}")

    df[config.domain_column] = df[config.domain_column].astype("string")
    df[config.id_column] = df[config.id_column].astype("string")
    df[config.target_column] = df[config.target_column].astype(int)

    unique_targets = sorted(df[config.target_column].dropna().unique().tolist())
    if unique_targets != [0, 1]:
        raise ValueError(
            f"The ABIDE transportability pipeline expects a binary target encoded as 0/1. Found: {unique_targets}"
        )

    observed_sites = sorted(
        df[config.domain_column].dropna().astype(str).unique().tolist()
    )
    eligible_sites, excluded_sites = summarize_target_site_eligibility(
        df, config=config
    )
    logger.info(
        "Loaded dataset with shape %s and %s observed sites: %s",
        df.shape,
        len(observed_sites),
        ", ".join(observed_sites),
    )
    logger.info(
        "Eligible ABIDE target sites: %s | evaluated sites: %s | excluded as targets: %s",
        len(eligible_sites),
        ", ".join(eligible_sites),
        "; ".join(
            f"{row['site']} ({','.join(row['reasons'])}; counts={row['class_counts']})"
            for row in excluded_sites
        )
        if excluded_sites
        else "none",
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
    """Partition one target site using the configured original seed and split rules."""
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
    """Build the configured target-site evaluations in the original site order."""
    eligible_sites, excluded_sites = summarize_target_site_eligibility(
        df, config=config
    )
    for row in excluded_sites:
        logger.warning(
            "Excluding site %s from target evaluation | n=%s | class_counts=%s | reasons=%s",
            row["site"],
            row["n_samples"],
            row["class_counts"],
            ",".join(row["reasons"]),
        )
    if not eligible_sites:
        raise ValueError(
            "No eligible ABIDE target sites available after applying target-site eligibility checks."
        )
    return [
        build_target_site_split(
            df, target_site=target_site, config=config, logger=logger
        )
        for target_site in eligible_sites
    ]
