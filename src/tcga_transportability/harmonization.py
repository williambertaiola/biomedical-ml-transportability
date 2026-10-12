"""Transportability pipeline module: harmonization."""

from __future__ import annotations

from contextlib import redirect_stdout
from functools import lru_cache
from io import StringIO
from typing import Any, Callable

import numpy as np
import pandas as pd

from tcga_transportability.config import ExperimentConfig


def _preview_features(features: list[str], limit: int = 12) -> str:
    if len(features) <= limit:
        return "|".join(features)
    return "|".join([*features[:limit], f"...(+{len(features) - limit} more)"])


@lru_cache(maxsize=1)
def _load_neurocombat() -> Callable[..., dict[str, Any]] | None:
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    try:
        from neuroCombat import neuroCombat
    except ImportError:
        return None
    return neuroCombat


def _choose_reference_batch(batch_train: pd.Series) -> str:
    return str(batch_train.astype(str).value_counts().idxmax())


def _select_combat_gene_pool(
    *,
    train_df: pd.DataFrame,
    schema: dict[str, Any],
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, list[str], pd.Series]:
    gene_columns = list(schema["gene_features"])
    gene_matrix = train_df[gene_columns].to_numpy(dtype=np.float32, copy=False)

    if np.isnan(gene_matrix).any():
        fill_values_array = np.nanmedian(gene_matrix, axis=0)
        fill_values_array = np.where(
            np.isnan(fill_values_array), 0.0, fill_values_array
        ).astype(np.float32, copy=False)
        filled_matrix = np.where(
            np.isnan(gene_matrix), fill_values_array, gene_matrix
        ).astype(np.float32, copy=False)
    else:
        fill_values_array = np.zeros(gene_matrix.shape[1], dtype=np.float32)
        filled_matrix = gene_matrix

    variances = np.var(filled_matrix, axis=0, ddof=0)
    candidate_idx = np.flatnonzero(variances > config.min_variance_threshold)
    if candidate_idx.size == 0:
        candidate_idx = np.asarray([int(np.argmax(variances))], dtype=int)

    ordered_idx = candidate_idx[np.argsort(variances[candidate_idx])[::-1]]
    if (
        config.combat_top_k_variance_genes > 0
        and ordered_idx.size > config.combat_top_k_variance_genes
    ):
        ordered_idx = ordered_idx[: config.combat_top_k_variance_genes]

    selected_genes = [gene_columns[int(idx)] for idx in ordered_idx.tolist()]
    selected_fill_values = pd.Series(
        fill_values_array[ordered_idx],
        index=selected_genes,
        dtype=np.float32,
    )
    selected_matrix = pd.DataFrame(
        filled_matrix[:, ordered_idx],
        index=train_df.index,
        columns=selected_genes,
        copy=True,
    )
    return selected_matrix, selected_genes, selected_fill_values


def _fit_neurocombat_train_only(
    *,
    train_matrix: pd.DataFrame,
    batch_train: pd.Series,
    label_train: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    neurocombat = _load_neurocombat()
    if neurocombat is None:
        raise ImportError(
            "neuroCombat is not available in the current Python environment."
        )

    batch_series = batch_train.astype(str).reset_index(drop=True)
    label_series = label_train.astype(str).reset_index(drop=True)
    reference_batch = _choose_reference_batch(batch_series)
    covars = pd.DataFrame(
        {
            "batch": batch_series.to_numpy(),
            "label": label_series.to_numpy(),
        }
    )

    with redirect_stdout(StringIO()):
        result = neurocombat(
            dat=train_matrix.reset_index(drop=True).to_numpy(dtype=float).T,
            covars=covars,
            batch_col="batch",
            categorical_cols=["label"],
            continuous_cols=None,
            ref_batch=reference_batch,
        )

    harmonized = pd.DataFrame(
        result["data"].T, index=train_matrix.index, columns=train_matrix.columns
    )
    return harmonized, {
        "combat_backend": "neuroCombat",
        "combat_reference_batch": reference_batch,
    }


def prepare_method_train_test_frames(
    *,
    method_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: dict[str, Any],
    config: ExperimentConfig,
    cache: dict[Any, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    schema_out = dict(schema)

    if method_name in {"M0", "M2", "M3"}:
        gene_features = list(schema_out["gene_features"])
        train_out = train_df[gene_features].copy()
        test_out = test_df[gene_features].copy()
        return (
            train_out,
            test_out,
            {
                "harmonization": "none",
                "combat_backend": "none",
                "combat_reference_batch": "",
                "n_combat_features": 0,
                "combat_features": "",
                "combat_feature_pool_strategy": "none",
                "combat_label_covariate": "",
            },
            schema_out,
        )

    if method_name not in {"M1", "M4"}:
        raise ValueError(
            f"Unsupported harmonization method in the current TCGA pipeline: {method_name}"
        )

    cache_key = None
    if cache is not None:
        cache_key = (
            method_name,
            tuple(train_df[config.id_column].astype(str).tolist()),
        )
        cached = cache.get(cache_key)
        if cached is not None:
            selected_genes = cached["selected_genes"]
            schema_out["gene_features"] = list(selected_genes)
            train_out = cached["harmonized_train_matrix"].copy()
            test_out = pd.DataFrame(
                test_df[selected_genes].to_numpy(dtype=np.float32, copy=True),
                index=test_df.index,
                columns=selected_genes,
            )
            return train_out, test_out, dict(cached["method_details"]), schema_out

    train_matrix, selected_genes, _ = _select_combat_gene_pool(
        train_df=train_df, schema=schema, config=config
    )
    if not selected_genes:
        train_out = train_df[schema_out["gene_features"]].copy()
        test_out = test_df[schema_out["gene_features"]].copy()
        method_details = {
            "harmonization": "combat_skipped_no_features",
            "combat_backend": "none",
            "combat_reference_batch": "",
            "n_combat_features": 0,
            "combat_features": "",
            "combat_feature_pool_strategy": "train_only_variance_prefilter",
            "combat_label_covariate": config.combat_label_covariate,
        }
        return train_out, test_out, method_details, schema_out

    harmonized_train_matrix, runtime_details = _fit_neurocombat_train_only(
        train_matrix=train_matrix,
        batch_train=train_df[config.domain_column],
        label_train=train_df[config.combat_label_covariate],
    )

    method_details = {
        "harmonization": "combat_train_only",
        "combat_backend": runtime_details.get("combat_backend", "unknown"),
        "combat_reference_batch": runtime_details.get("combat_reference_batch", ""),
        "n_combat_features": int(len(selected_genes)),
        "combat_features": _preview_features(selected_genes),
        "combat_feature_pool_strategy": "train_only_variance_prefilter",
        "combat_label_covariate": config.combat_label_covariate,
    }
    if cache is not None and cache_key is not None:
        cache[cache_key] = {
            "selected_genes": list(selected_genes),
            "harmonized_train_matrix": harmonized_train_matrix.astype(np.float32),
            "method_details": dict(method_details),
        }

    schema_out["gene_features"] = selected_genes
    train_out = harmonized_train_matrix.astype(np.float32)
    test_out = pd.DataFrame(
        test_df[selected_genes].to_numpy(dtype=np.float32, copy=True),
        index=test_df.index,
        columns=selected_genes,
    )
    return train_out, test_out, method_details, schema_out
