"""Transportability experiment: splits."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedShuffleSplit

    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

from .utils import ensure_dir, log, safe_slug


def add_domain_columns(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    out = df.copy()
    splits_cfg = config.get("splits", {})

    primary = splits_cfg.get("domain_column", "site_id")
    if primary not in out.columns:
        if "site_id" in out.columns:
            primary = "site_id"
        else:
            raise ValueError(
                "No domain column found. Configure splits.domain_column or provide site_id."
            )
    out["domain_site"] = out[primary].astype("string").fillna("UNKNOWN_SITE")

    lodo_cols = splits_cfg.get("lodo_domain_columns", [primary])
    existing = [c for c in lodo_cols if c in out.columns]
    if not existing:
        existing = [primary]
    out["domain_lodo"] = (
        out[existing]
        .astype("string")
        .fillna("NA")
        .agg("__".join, axis=1)
        .str.replace(r"\s+", "_", regex=True)
    )
    return out


def _choose_val_mask(
    data: pd.DataFrame,
    diagnosis_col: str,
    domain_col: str,
    val_fraction: float,
    seed: int,
) -> pd.Series:
    n = len(data)
    if n == 0:
        return pd.Series(False, index=data.index)
    val_n_target = max(1, int(round(n * val_fraction)))

    domains = data[domain_col].astype("string").value_counts()
    # Prefer domain-based validation if possible.
    if len(domains) >= 2:
        best_domain = min(domains.index, key=lambda d: abs(domains[d] - val_n_target))
        return data[domain_col].astype("string") == best_domain

    # Fallback to random/stratified row split.
    if (
        diagnosis_col in data.columns
        and data[diagnosis_col].notna().sum() > 0
        and HAVE_SKLEARN
    ):
        y = data[diagnosis_col].fillna(-1)
        if len(pd.Series(y).value_counts()) >= 2:
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=val_fraction, random_state=seed
            )
            train_idx, val_idx = next(sss.split(np.zeros(len(y)), y))
            mask = pd.Series(False, index=data.index)
            mask.iloc[val_idx] = True
            return mask

    rng = np.random.default_rng(seed)
    take = min(val_n_target, n)
    idx = rng.choice(data.index.to_numpy(), size=take, replace=False)
    return data.index.isin(idx)


def _build_imputation_values(
    train_df: pd.DataFrame, features: List[str]
) -> pd.DataFrame:
    rows = []
    for c in features:
        if c not in train_df.columns:
            continue
        s = train_df[c]
        if pd.api.types.is_numeric_dtype(s):
            val = float(pd.to_numeric(s, errors="coerce").median(skipna=True))
            kind = "numeric_median"
        else:
            mode = s.astype("string").dropna().mode()
            val = mode.iloc[0] if len(mode) else "UNK"
            kind = "categorical_mode"
        rows.append({"feature": c, "imputation_type": kind, "impute_value": val})
    return pd.DataFrame(rows)


def _apply_imputation(df: pd.DataFrame, imputation_table: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for row in imputation_table.itertuples(index=False):
        c = row.feature
        if c not in out.columns:
            continue
        if row.imputation_type == "numeric_median":
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(
                float(row.impute_value)
            )
        else:
            out[c] = out[c].astype("string").fillna(str(row.impute_value))
    return out


def _create_leave_one_domain_splits(
    df: pd.DataFrame,
    domain_col: str,
    strategy_label: str,
    filename_prefix: str,
    output_dir: Path,
    diagnosis_col: str,
    feature_cols_for_impute: List[str],
    val_fraction: float,
    seed: int,
    min_subjects_per_test_domain: int,
) -> Tuple[pd.DataFrame, List[Path]]:
    ensure_dir(output_dir)
    domain_counts = df[domain_col].astype("string").value_counts()
    eligible_domains = [
        d for d, n in domain_counts.items() if int(n) >= min_subjects_per_test_domain
    ]

    rows = []
    generated_files: List[Path] = []

    if len(eligible_domains) < 2:
        log(
            f"{strategy_label.upper()} skipped: only {len(eligible_domains)} eligible domains "
            f"(min_subjects_per_test_domain={min_subjects_per_test_domain})."
        )
        return pd.DataFrame(), generated_files

    for fold_i, test_domain in enumerate(sorted(eligible_domains)):
        fold_df = df.copy()
        test_mask = fold_df[domain_col].astype("string") == str(test_domain)
        trainval_mask = ~test_mask
        trainval_df = fold_df.loc[trainval_mask].copy()

        val_mask_local = _choose_val_mask(
            trainval_df,
            diagnosis_col=diagnosis_col,
            domain_col=domain_col,
            val_fraction=val_fraction,
            seed=seed + fold_i,
        )
        val_indices = trainval_df.index[val_mask_local]

        split = pd.Series("train", index=fold_df.index, dtype="string")
        split.loc[test_mask] = "test"
        split.loc[val_indices] = "val"
        fold_df["split"] = split
        fold_df["split_strategy"] = strategy_label
        fold_df["test_domain"] = str(test_domain)

        # Train-only imputation.
        train_df = fold_df.loc[fold_df["split"] == "train"].copy()
        imputation_table = _build_imputation_values(train_df, feature_cols_for_impute)
        fold_df_imp = _apply_imputation(fold_df, imputation_table)

        test_slug = safe_slug(str(test_domain))
        fold_name = f"{filename_prefix}{test_slug}"
        split_path = output_dir / f"{fold_name}.csv"
        impute_path = output_dir / f"{fold_name}_imputation.csv"

        fold_df_imp.to_csv(split_path, index=False)
        imputation_table.to_csv(impute_path, index=False)
        generated_files.extend([split_path, impute_path])

        counts = fold_df_imp["split"].value_counts().to_dict()
        rows.append(
            {
                "strategy": strategy_label,
                "domain_col": domain_col,
                "test_domain": str(test_domain),
                "n_total": int(len(fold_df_imp)),
                "n_train": int(counts.get("train", 0)),
                "n_val": int(counts.get("val", 0)),
                "n_test": int(counts.get("test", 0)),
                "split_file": str(split_path),
                "imputation_file": str(impute_path),
            }
        )

    return pd.DataFrame(rows), generated_files


def generate_splits(
    dataset_df: pd.DataFrame,
    model_feature_cols: List[str],
    config: Dict,
    output_splits_dir: Path,
) -> Tuple[pd.DataFrame, List[Path]]:
    ensure_dir(output_splits_dir)
    splits_cfg = config.get("splits", {})
    val_fraction = float(splits_cfg.get("val_fraction", 0.2))
    seed = int(
        splits_cfg.get("random_seed", config.get("project", {}).get("random_seed", 42))
    )
    min_subjects = int(splits_cfg.get("min_subjects_per_test_domain", 10))

    diagnosis_col = splits_cfg.get("diagnosis_column", "diagnosis")
    if diagnosis_col not in dataset_df.columns:
        diagnosis_col = (
            "diagnosis" if "diagnosis" in dataset_df.columns else "diagnosis_label"
        )

    # Do not impute target/domain/id columns.
    blocked = {
        "subject_id",
        "diagnosis",
        "diagnosis_label",
        "site_id",
        "domain_site",
        "domain_lodo",
    }
    impute_features = [
        c for c in model_feature_cols if c in dataset_df.columns and c not in blocked
    ]

    summaries = []
    files = []

    strategies = splits_cfg.get("strategies", ["loso", "lodo"])
    if "loso" in strategies:
        loso_df, loso_files = _create_leave_one_domain_splits(
            df=dataset_df,
            domain_col="domain_site",
            strategy_label="loso",
            filename_prefix="loso_site_",
            output_dir=output_splits_dir,
            diagnosis_col=diagnosis_col,
            feature_cols_for_impute=impute_features,
            val_fraction=val_fraction,
            seed=seed,
            min_subjects_per_test_domain=min_subjects,
        )
        if not loso_df.empty:
            summaries.append(loso_df)
        files.extend(loso_files)

    if "lodo" in strategies:
        lodo_df, lodo_files = _create_leave_one_domain_splits(
            df=dataset_df,
            domain_col="domain_lodo",
            strategy_label="lodo",
            filename_prefix="lodo_domain_",
            output_dir=output_splits_dir,
            diagnosis_col=diagnosis_col,
            feature_cols_for_impute=impute_features,
            val_fraction=val_fraction,
            seed=seed + 1337,
            min_subjects_per_test_domain=min_subjects,
        )
        if not lodo_df.empty:
            summaries.append(lodo_df)
        files.extend(lodo_files)

    if summaries:
        summary_df = pd.concat(summaries, ignore_index=True)
    else:
        summary_df = pd.DataFrame(
            columns=[
                "strategy",
                "domain_col",
                "test_domain",
                "n_total",
                "n_train",
                "n_val",
                "n_test",
                "split_file",
                "imputation_file",
            ]
        )
    return summary_df, files
