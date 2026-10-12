"""Preprocessing: splits."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _class_balance(series: pd.Series) -> dict[str, Any]:
    return {
        "n_class_0": int((series == 0).sum()),
        "n_class_1": int((series == 1).sum()),
        "positive_prevalence": round(float(series.mean()), 4),
    }


def build_lodo_manifest(
    df: pd.DataFrame, domain_column: str, target_column: str
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    observed_domains = sorted(df[domain_column].dropna().astype(str).unique().tolist())

    for fold_id, held_out_domain in enumerate(observed_domains, start=1):
        train_df = df.loc[df[domain_column] != held_out_domain].copy()
        test_df = df.loc[df[domain_column] == held_out_domain].copy()

        manifest.append(
            {
                "fold_id": fold_id,
                "held_out_domain": held_out_domain,
                "train_domains": sorted(
                    train_df[domain_column].dropna().astype(str).unique().tolist()
                ),
                "n_train": int(train_df.shape[0]),
                "n_test": int(test_df.shape[0]),
                "class_balance_train": _class_balance(train_df[target_column]),
                "class_balance_test": _class_balance(test_df[target_column]),
            }
        )

    return manifest
