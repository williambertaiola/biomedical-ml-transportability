"""Transportability pipeline module: data."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from tcga_transportability.config import ExperimentConfig


METADATA_COLUMNS = [
    "sample_barcode",
    "project_code",
    "tss",
    "participant_id",
    "subject_id",
    "sample_id",
    "sample_type_code",
    "sample_type_label",
    "vial",
    "portion",
    "analyte",
    "plate",
    "center",
    "barcode_scope",
    "barcode_parse_status",
    "barcode_parse_note",
    "clinical_case_id",
    "clinical_primary_site",
    "clinical_ethnicity",
    "clinical_gender",
    "clinical_race",
    "clinical_ajcc_pathologic_stage",
    "clinical_primary_diagnosis",
    "clinical_site_of_resection_or_biopsy",
    "clinical_tissue_or_organ_of_origin",
    "clinical_record_count",
    "domain_primary",
    "domain_source",
    "domain_secondary_candidates",
    "total_signal",
    "nonzero_gene_count",
    "zero_fraction",
    "low_total_signal_outlier",
    "site",
    "ajcc_pathologic_stage_raw",
    "target_label",
    "target",
]


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


def load_dataset(
    config: ExperimentConfig, logger: logging.Logger
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate the preprocessed dataset without changing feature values."""
    header_df = pd.read_csv(config.input_path, nrows=0)
    all_columns = header_df.columns.astype(str).tolist()
    gene_columns = [
        column for column in all_columns if str(column).startswith(config.gene_prefix)
    ]
    if not gene_columns:
        raise ValueError(
            f"No gene columns detected with prefix '{config.gene_prefix}'."
        )

    dtype_map: dict[str, str] = {column: "float32" for column in gene_columns}
    if config.domain_column in all_columns:
        dtype_map[config.domain_column] = "string"
    if config.id_column in all_columns:
        dtype_map[config.id_column] = "string"

    df = pd.read_csv(
        config.input_path,
        low_memory=False,
        dtype=dtype_map,
        memory_map=True,
    )
    required_columns = {config.id_column, config.domain_column, config.target_column}
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in input dataset: {missing}")

    df = df.copy()
    df[config.domain_column] = df[config.domain_column].astype("string")
    df[config.id_column] = df[config.id_column].astype("string")
    df[config.target_column] = pd.to_numeric(
        df[config.target_column], errors="raise"
    ).astype(int)

    valid_sites: list[str] = []
    dropped_sites: list[str] = []
    for site, site_df in df.groupby(config.domain_column, dropna=False):
        site_name = str(site)
        if site_name == "<NA>":
            dropped_sites.append(site_name)
            continue
        if len(site_df) < config.min_site_samples:
            dropped_sites.append(site_name)
            continue
        counts = site_df[config.target_column].value_counts()
        if (
            site_df[config.target_column].nunique() < 2
            or int(counts.min()) < config.min_site_class_count
        ):
            dropped_sites.append(site_name)
            continue
        valid_sites.append(site_name)

    filtered_df = (
        df.loc[df[config.domain_column].astype(str).isin(valid_sites)]
        .copy()
        .reset_index(drop=True)
    )
    if filtered_df.empty:
        raise ValueError(
            "No valid TCGA sites remain after minimum sample/class filtering."
        )

    logger.info(
        "Loaded dataset with shape %s and %s valid sites: %s",
        filtered_df.shape,
        len(valid_sites),
        ", ".join(sorted(valid_sites)),
    )
    if dropped_sites:
        logger.info(
            "Dropped sites during TCGA filtering: %s", ", ".join(sorted(dropped_sites))
        )

    schema = {
        "gene_features": gene_columns,
        "metadata_columns": [
            column for column in METADATA_COLUMNS if column in filtered_df.columns
        ],
        "excluded_columns": [
            column for column in METADATA_COLUMNS if column in filtered_df.columns
        ],
    }
    return filtered_df, schema


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
    target_sites: tuple[str, ...] | None = None,
) -> list[TargetSiteSplit]:
    """Build the configured target-site evaluations in the original site order."""
    observed_sites = sorted(
        df[config.domain_column].dropna().astype(str).unique().tolist()
    )
    if target_sites:
        allowed_sites = set(target_sites)
        observed_sites = [site for site in observed_sites if site in allowed_sites]
    return [
        build_target_site_split(
            df, target_site=target_site, config=config, logger=logger
        )
        for target_site in observed_sites
    ]
