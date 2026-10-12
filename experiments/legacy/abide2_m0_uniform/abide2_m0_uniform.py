"""Transportability experiment: abide2 m0 uniform."""

from __future__ import annotations

# Locate the repository without depending on the current working directory.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys

_REPOSITORY_ROOT = next(
    parent
    for parent in _RepositoryPath(__file__).resolve().parents
    if (parent / ".thesis-root").is_file()
)
_repository_sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))
from thesis_tools.paths import configured_path as _configured_path


import argparse
import importlib.util
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


DEFAULT_INPUT_PATH = _configured_path(
    "legacy_preprocessed_abideii_abide2_core_multisite_transportability_dataset.csv"
)
DEFAULT_WORKSPACE_OUTPUT = _configured_path("legacy_output")
DEFAULT_FINAL_OUTPUT = _configured_path("legacy_cartella_senza_nome")


@dataclass
class AbideConfig:
    input_path: str = DEFAULT_INPUT_PATH
    workspace_output_dir: str = DEFAULT_WORKSPACE_OUTPUT
    final_output_dir: str = DEFAULT_FINAL_OUTPUT
    dataset_name: str = "ABIDE II"
    target_col: str = "disease_level_code"
    target_label_col: str = "disease_level"
    domain_col: str = "site"
    id_col: str = "subject_id"
    positive_label: int = 1
    random_seed: int = 42
    min_domain_size: int = 20
    save_pdf_plots: bool = True
    suspicious_f1_threshold: float = 0.05
    imbalance_warning_threshold: float = 0.30
    threshold_grid: list[float] | None = None


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_name(value: str) -> str:
    return "".join(
        char.lower() if char.isalnum() else "_" for char in str(value)
    ).strip("_")


def load_baseline_core():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "uci_m0_uniform_single"
        / "uci_m0_uniform.py"
    )
    spec = importlib.util.spec_from_file_location("baseline_m0_core", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clean_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_and_filter_abide(
    config: AbideConfig,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(config.input_path)
    total_subjects = int(df[config.id_col].nunique())
    site_counts_before = (
        df.groupby(config.domain_col)
        .agg(
            n_subjects=(config.id_col, "nunique"),
            n_rows=(config.id_col, "size"),
            n_asd=(
                config.target_col,
                lambda s: int((s == config.positive_label).sum()),
            ),
            n_control=(
                config.target_col,
                lambda s: int((s != config.positive_label).sum()),
            ),
        )
        .reset_index()
        .sort_values(["n_subjects", config.domain_col], ascending=[False, True])
        .reset_index(drop=True)
    )
    class_table = pd.crosstab(
        df[config.domain_col], df[config.target_label_col]
    ).sort_index()
    single_class_sites = class_table.index[(class_table == 0).any(axis=1)].tolist()

    filtered_df = (
        df.loc[~df[config.domain_col].isin(single_class_sites)]
        .copy()
        .reset_index(drop=True)
    )
    site_counts_after = (
        filtered_df.groupby(config.domain_col)
        .agg(
            n_subjects=(config.id_col, "nunique"),
            n_rows=(config.id_col, "size"),
            n_asd=(
                config.target_col,
                lambda s: int((s == config.positive_label).sum()),
            ),
            n_control=(
                config.target_col,
                lambda s: int((s != config.positive_label).sum()),
            ),
        )
        .reset_index()
        .sort_values(["n_subjects", config.domain_col], ascending=[False, True])
        .reset_index(drop=True)
    )

    audit = {
        "n_subjects_total": total_subjects,
        "n_rows_total": int(len(df)),
        "n_sites_total": int(df[config.domain_col].nunique()),
        "n_sites_retained": int(filtered_df[config.domain_col].nunique()),
        "n_subjects_retained": int(filtered_df[config.id_col].nunique()),
        "n_rows_retained": int(len(filtered_df)),
        "excluded_single_class_sites": single_class_sites,
        "class_counts_total": filtered_df[config.target_label_col]
        .value_counts()
        .to_dict(),
    }
    return filtered_df, audit, site_counts_before, site_counts_after


def write_filtered_input(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "abide2_filtered_input.csv"
    df.to_csv(path, index=False)
    return path


def build_core_config(
    core_module, config: AbideConfig, input_path: Path, output_dir: Path
):
    return core_module.ExperimentConfig(
        dataset_name=config.dataset_name,
        input_path=str(input_path),
        output_dir=str(output_dir),
        target_col=config.target_col,
        domain_col=config.domain_col,
        id_cols=[config.id_col],
        extra_exclude_cols=[config.target_label_col],
        positive_label=config.positive_label,
        random_seed=config.random_seed,
        enable_xgboost=False,
        save_pdf_plots=config.save_pdf_plots,
        min_domain_size=config.min_domain_size,
        enable_pairwise_site_transfer=True,
        feature_selection_method="none",
    )


def run_main_baseline(
    core_module, config: AbideConfig, filtered_input_path: Path, output_dir: Path
) -> None:
    core_config = build_core_config(
        core_module, config, filtered_input_path, output_dir
    )
    core_module.run_experiment(core_config)


def run_shuffle_test(
    core_module, config: AbideConfig, filtered_df: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.random_seed)
    shuffled_df = filtered_df.copy()
    shuffled_df[config.target_col] = rng.permutation(
        shuffled_df[config.target_col].to_numpy()
    )
    shuffled_input = output_dir / "abide2_filtered_input_shuffled.csv"
    shuffled_df.to_csv(shuffled_input, index=False)
    shuffle_output = output_dir / "_shuffle_tmp"

    shuffle_config = build_core_config(
        core_module, config, shuffled_input, shuffle_output
    )
    shuffle_config.enable_pairwise_site_transfer = False
    shuffle_config.save_pdf_plots = False
    core_module.run_experiment(shuffle_config)

    shuffle_fold_results = pd.read_csv(shuffle_output / "fold_results.csv")
    shuffle_model_summary = pd.read_csv(shuffle_output / "model_summary.csv")
    shuffle_fold_results.to_csv(
        output_dir / "shuffle_test_fold_results.csv", index=False
    )
    shuffle_model_summary.to_csv(
        output_dir / "shuffle_test_model_summary.csv", index=False
    )

    shutil.rmtree(shuffle_output, ignore_errors=True)
    if shuffled_input.exists():
        shuffled_input.unlink()
    return shuffle_fold_results, shuffle_model_summary


def build_leakage_checks(df: pd.DataFrame, config: AbideConfig) -> pd.DataFrame:
    excluded = {
        config.id_col: "subject identifier",
        config.domain_col: "domain/site",
        config.target_col: "numeric target",
        config.target_label_col: "text target label",
    }
    rows: list[dict[str, str]] = []
    for column, reason in excluded.items():
        if column in df.columns:
            rows.append({"column_name": column, "status": "excluded", "reason": reason})

    candidate_features = [col for col in df.columns if col not in excluded]
    suspicious_patterns = [
        ("site", "site leaked into features"),
        ("subject", "identifier leaked into features"),
        ("disease", "target-like label leaked into features"),
        ("diagn", "target-like label leaked into features"),
    ]
    flagged = False
    for column in candidate_features:
        for pattern, reason in suspicious_patterns:
            if pattern in column.lower():
                rows.append(
                    {
                        "column_name": column,
                        "status": "flagged_feature",
                        "reason": reason,
                    }
                )
                flagged = True
                break
    if not flagged:
        rows.append(
            {
                "column_name": "*feature_set*",
                "status": "pass",
                "reason": "no site/ID/target-like feature columns detected",
            }
        )
    return pd.DataFrame(rows)


def binarize_target_frame(df: pd.DataFrame, config: AbideConfig) -> pd.DataFrame:
    out = df.copy()
    out[config.target_col] = (out[config.target_col] == config.positive_label).astype(
        int
    )
    return out


def get_threshold_grid(config: AbideConfig) -> np.ndarray:
    if config.threshold_grid:
        return np.array(config.threshold_grid, dtype=float)
    return np.linspace(0.05, 0.95, 19)


def compute_binary_diagnostics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "n_true_negative": int((y_true == 0).sum()),
        "n_true_positive": int((y_true == 1).sum()),
        "n_pred_negative": int((y_pred == 0).sum()),
        "n_pred_positive": int((y_pred == 1).sum()),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balacc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mean_prob": float(np.mean(y_prob)),
        "median_prob": float(np.median(y_prob)),
        "min_prob": float(np.min(y_prob)),
        "max_prob": float(np.max(y_prob)),
        "predicted_single_class": bool(len(np.unique(y_pred)) == 1),
    }
    metrics["auc"] = (
        float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) == 2
        else math.nan
    )
    return metrics


def build_lodo_site_diagnostics(
    output_dir: Path, config: AbideConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_results = pd.read_csv(output_dir / "fold_results.csv")
    prediction_df = pd.read_csv(output_dir / "prediction_records.csv")
    ood_predictions = prediction_df.loc[prediction_df["split"] == "ood"].copy()
    rows: list[dict[str, Any]] = []

    for (model_name, test_domain), group in ood_predictions.groupby(
        ["model", "outer_test_domain"], dropna=False
    ):
        diagnostics = compute_binary_diagnostics(
            group["y_true"].to_numpy(), group["y_prob"].to_numpy(), threshold=0.5
        )
        diagnostics.update({"model": model_name, "test_domain": test_domain})
        fold_match = fold_results.loc[
            (fold_results["model"] == model_name)
            & (fold_results["test_domain"] == test_domain)
        ]
        if not fold_match.empty:
            row = fold_match.iloc[0].to_dict()
            row.update(diagnostics)
        else:
            row = diagnostics
        row["suspicious_case"] = bool(
            diagnostics["f1"] <= config.suspicious_f1_threshold
            or diagnostics["predicted_single_class"]
        )
        rows.append(row)

    site_diag = (
        pd.DataFrame(rows).sort_values(["test_domain", "model"]).reset_index(drop=True)
    )
    confusion_cols = [
        "model",
        "test_domain",
        "n_samples",
        "n_true_negative",
        "n_true_positive",
        "n_pred_negative",
        "n_pred_positive",
        "positive_rate_true",
        "positive_rate_pred",
        "tn",
        "fp",
        "fn",
        "tp",
        "predicted_single_class",
        "accuracy",
        "balacc",
        "f1",
        "auc",
    ]
    confusion_df = site_diag[confusion_cols].copy()
    return site_diag, confusion_df


def build_pairwise_site_diagnostics(
    output_dir: Path, config: AbideConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_df = pd.read_csv(output_dir / "train_test_site_results.csv")
    prediction_df = pd.read_csv(output_dir / "train_test_site_prediction_records.csv")
    rows: list[dict[str, Any]] = []

    for (model_name, train_domain, test_domain), group in prediction_df.groupby(
        ["model", "train_domain", "test_domain"], dropna=False
    ):
        diagnostics = compute_binary_diagnostics(
            group["y_true"].to_numpy(), group["y_prob"].to_numpy(), threshold=0.5
        )
        diagnostics.update(
            {
                "model": model_name,
                "train_domain": train_domain,
                "test_domain": test_domain,
            }
        )
        match = result_df.loc[
            (result_df["model"] == model_name)
            & (result_df["train_domain"] == train_domain)
            & (result_df["test_domain"] == test_domain)
        ]
        if not match.empty:
            row = match.iloc[0].to_dict()
            row.update(diagnostics)
        else:
            row = diagnostics
        row["strong_class_imbalance_test"] = bool(
            (diagnostics["positive_rate_true"] <= config.imbalance_warning_threshold)
            or (
                diagnostics["positive_rate_true"]
                >= 1 - config.imbalance_warning_threshold
            )
        )
        row["suspicious_case"] = bool(
            diagnostics["f1"] <= config.suspicious_f1_threshold
            or diagnostics["predicted_single_class"]
        )
        rows.append(row)

    pairwise_diag = (
        pd.DataFrame(rows)
        .sort_values(["test_domain", "train_domain", "model"])
        .reset_index(drop=True)
    )
    confusion_cols = [
        "model",
        "train_domain",
        "test_domain",
        "is_diagonal",
        "n_samples",
        "n_true_negative",
        "n_true_positive",
        "n_pred_negative",
        "n_pred_positive",
        "positive_rate_true",
        "positive_rate_pred",
        "tn",
        "fp",
        "fn",
        "tp",
        "predicted_single_class",
        "accuracy",
        "balacc",
        "f1",
        "auc",
    ]
    confusion_df = pairwise_diag[confusion_cols].copy()
    suspicious_df = (
        pairwise_diag.loc[pairwise_diag["suspicious_case"]]
        .copy()
        .reset_index(drop=True)
    )
    return pairwise_diag, confusion_df, suspicious_df


def classify_suspicious_case(row: pd.Series) -> str:
    reasons: list[str] = []
    if bool(row.get("predicted_single_class", False)):
        reasons.append("single-class prediction")
    if pd.notna(row.get("threshold_gain_f1")) and row["threshold_gain_f1"] >= 0.20:
        reasons.append("training-only threshold tuning materially recovers F1")
    if pd.notna(row.get("auc")) and row["auc"] >= 0.70:
        reasons.append("ranking signal present, threshold issue plausible")
    elif pd.notna(row.get("auc")) and row["auc"] <= 0.55:
        reasons.append("near-chance ranking, genuine transport failure")
    else:
        reasons.append("weak ranking and threshold mismatch")
    if bool(row.get("strong_class_imbalance_test", False)):
        reasons.append("test-site class imbalance may contribute")
    if pd.notna(row.get("n_train")) and row["n_train"] < 60:
        reasons.append("small source site")
    return "; ".join(reasons)


def select_training_only_threshold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    core_module,
    core_config,
    config: AbideConfig,
) -> dict[str, Any]:
    logger = logging.getLogger("abide_threshold_diagnostics")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    models = core_module.build_models(core_config, logger)
    estimator = models[model_name]
    train_df = binarize_target_frame(train_df, config)
    test_df = binarize_target_frame(test_df, config)
    y_train = train_df[config.target_col].astype(int)
    class_counts = y_train.value_counts()
    min_class = int(class_counts.min()) if not class_counts.empty else 0
    n_splits = min(3, len(train_df), min_class)
    if y_train.nunique() < 2 or n_splits < 2:
        return {
            "train_cv_best_threshold": math.nan,
            "train_cv_best_f1": math.nan,
            "train_cv_best_balacc": math.nan,
            "test_f1_threshold_0_5": math.nan,
            "test_f1_train_tuned_threshold": math.nan,
            "test_balacc_train_tuned_threshold": math.nan,
            "test_pred_positive_train_tuned_threshold": math.nan,
            "threshold_gain_f1": math.nan,
            "threshold_gain_balacc": math.nan,
        }

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=config.random_seed
    )
    oof_probs: list[np.ndarray] = []
    oof_true: list[np.ndarray] = []
    for inner_train_idx, inner_val_idx in splitter.split(train_df, y_train):
        inner_train_df = train_df.iloc[inner_train_idx].reset_index(drop=True)
        inner_val_df = train_df.iloc[inner_val_idx].reset_index(drop=True)
        spec = core_module.prepare_columns(inner_train_df, core_config)
        if not spec.feature_cols:
            continue
        pipeline = core_module.build_pipeline(spec, estimator, core_config)
        pipeline, _ = core_module.fit_with_warnings(
            pipeline, inner_train_df, inner_train_df[config.target_col].astype(int)
        )
        probs = core_module.safe_predict_proba(pipeline, inner_val_df)
        oof_probs.append(probs)
        oof_true.append(inner_val_df[config.target_col].astype(int).to_numpy())

    if not oof_probs:
        return {
            "train_cv_best_threshold": math.nan,
            "train_cv_best_f1": math.nan,
            "train_cv_best_balacc": math.nan,
            "test_f1_threshold_0_5": math.nan,
            "test_f1_train_tuned_threshold": math.nan,
            "test_balacc_train_tuned_threshold": math.nan,
            "test_pred_positive_train_tuned_threshold": math.nan,
            "threshold_gain_f1": math.nan,
            "threshold_gain_balacc": math.nan,
        }

    y_oof = np.concatenate(oof_true)
    p_oof = np.concatenate(oof_probs)
    best_tuple = None
    best_threshold = 0.5
    for threshold in get_threshold_grid(config):
        diag = compute_binary_diagnostics(y_oof, p_oof, threshold=float(threshold))
        candidate = (diag["f1"], diag["balacc"], -abs(float(threshold) - 0.5))
        if best_tuple is None or candidate > best_tuple:
            best_tuple = candidate
            best_threshold = float(threshold)

    full_spec = core_module.prepare_columns(train_df, core_config)
    pipeline = core_module.build_pipeline(full_spec, estimator, core_config)
    pipeline, _ = core_module.fit_with_warnings(
        pipeline, train_df, train_df[config.target_col].astype(int)
    )
    test_prob = core_module.safe_predict_proba(pipeline, test_df)
    default_diag = compute_binary_diagnostics(
        test_df[config.target_col].astype(int).to_numpy(), test_prob, threshold=0.5
    )
    tuned_diag = compute_binary_diagnostics(
        test_df[config.target_col].astype(int).to_numpy(),
        test_prob,
        threshold=best_threshold,
    )
    oof_diag = compute_binary_diagnostics(y_oof, p_oof, threshold=best_threshold)

    return {
        "train_cv_best_threshold": best_threshold,
        "train_cv_best_f1": oof_diag["f1"],
        "train_cv_best_balacc": oof_diag["balacc"],
        "test_f1_threshold_0_5": default_diag["f1"],
        "test_f1_train_tuned_threshold": tuned_diag["f1"],
        "test_balacc_train_tuned_threshold": tuned_diag["balacc"],
        "test_pred_positive_train_tuned_threshold": tuned_diag["n_pred_positive"],
        "threshold_gain_f1": tuned_diag["f1"] - default_diag["f1"],
        "threshold_gain_balacc": tuned_diag["balacc"] - default_diag["balacc"],
    }


def add_threshold_diagnostics_to_suspicious_cases(
    suspicious_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    core_module,
    config: AbideConfig,
    input_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    if suspicious_df.empty:
        return suspicious_df
    core_config = build_core_config(core_module, config, input_path, output_dir)
    rows: list[dict[str, Any]] = []

    for row in suspicious_df.itertuples(index=False):
        train_df = filtered_df.loc[
            filtered_df[config.domain_col] == row.train_domain
        ].reset_index(drop=True)
        test_df = filtered_df.loc[
            filtered_df[config.domain_col] == row.test_domain
        ].reset_index(drop=True)
        threshold_diag = select_training_only_threshold(
            train_df=train_df,
            test_df=test_df,
            model_name=row.model,
            core_module=core_module,
            core_config=core_config,
            config=config,
        )
        row_dict = row._asdict()
        row_dict.update(threshold_diag)
        row_dict["suspected_cause"] = classify_suspicious_case(pd.Series(row_dict))
        rows.append(row_dict)

    return pd.DataFrame(rows)


def summarise_suspicious_cases(suspicious_df: pd.DataFrame) -> pd.DataFrame:
    if suspicious_df.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "n_suspicious_cases",
                "n_single_class_predictions",
                "n_threshold_issue_plausible",
                "n_near_chance_failures",
                "mean_auc",
                "mean_balacc",
                "mean_f1",
                "mean_threshold_gain_f1",
                "max_threshold_gain_f1",
            ]
        )
    summary = (
        suspicious_df.assign(
            threshold_issue_plausible=suspicious_df["suspected_cause"]
            .fillna("")
            .str.contains(
                "threshold issue plausible|materially recovers F1", regex=True
            ),
            near_chance_failure=suspicious_df["suspected_cause"]
            .fillna("")
            .str.contains("genuine transport failure", regex=True),
        )
        .groupby("model", dropna=False)
        .agg(
            n_suspicious_cases=("model", "size"),
            n_single_class_predictions=("predicted_single_class", "sum"),
            n_threshold_issue_plausible=("threshold_issue_plausible", "sum"),
            n_near_chance_failures=("near_chance_failure", "sum"),
            mean_auc=("auc", "mean"),
            mean_balacc=("balacc", "mean"),
            mean_f1=("f1", "mean"),
            mean_threshold_gain_f1=("threshold_gain_f1", "mean"),
            max_threshold_gain_f1=("threshold_gain_f1", "max"),
        )
        .reset_index()
        .sort_values(["n_suspicious_cases", "mean_auc"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return summary


def _score_colors(
    values: np.ndarray, cmap_name: str = "RdYlGn", vmin: float = 0.0, vmax: float = 1.0
) -> list[Any]:
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return [
        cmap(norm(float(v))) if pd.notna(v) else (0.8, 0.8, 0.8, 1.0) for v in values
    ]


def save_figure(fig: plt.Figure, path_no_suffix: Path, save_pdf: bool) -> None:
    fig.savefig(path_no_suffix.with_suffix(".png"), dpi=320, bbox_inches="tight")
    if save_pdf:
        fig.savefig(path_no_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metric_by_domain(
    fold_results: pd.DataFrame,
    metric_column: str,
    title: str,
    ylabel: str,
    output_path_no_suffix: Path,
    save_pdf: bool,
) -> None:
    pivot = fold_results.pivot_table(
        index="test_domain", columns="model", values=metric_column, aggfunc="mean"
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.index) * 1.2), 6.5))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))
    for idx, model_name in enumerate(pivot.columns):
        values = pivot[model_name].to_numpy(dtype=float)
        xpos = x + idx * width - (len(pivot.columns) - 1) * width / 2
        bars = ax.bar(
            xpos, values, width=width, label=model_name, color=_score_colors(values)
        )
        for bar, value in zip(bars, values):
            if pd.notna(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Held-out site", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, title="Model")
    save_figure(fig, output_path_no_suffix, save_pdf)


def plot_model_comparison(
    model_summary: pd.DataFrame, output_path_no_suffix: Path, save_pdf: bool
) -> None:
    if model_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = model_summary[
        ["model", "accuracy_ood_mean", "balacc_ood_mean", "auc_ood_mean", "f1_ood_mean"]
    ]
    x = np.arange(len(plot_df))
    metrics = [
        ("accuracy_ood_mean", "Accuracy OOD"),
        ("balacc_ood_mean", "Balanced Accuracy OOD"),
        ("auc_ood_mean", "ROC-AUC OOD"),
        ("f1_ood_mean", "F1 OOD"),
    ]
    width = 0.18
    for idx, (column, label) in enumerate(metrics):
        values = plot_df[column].to_numpy(dtype=float)
        xpos = x + idx * width - (len(metrics) - 1) * width / 2
        bars = ax.bar(
            xpos, values, width=width, label=label, color=_score_colors(values)
        )
        for bar, value in zip(bars, values):
            if pd.notna(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_title(
        "ABIDE II baseline M0: mean model performance", fontsize=16, weight="bold"
    )
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"].tolist(), rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, output_path_no_suffix, save_pdf)


def heatmap_with_annotations(
    data: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path_no_suffix: Path,
    save_pdf: bool,
) -> None:
    if data.empty:
        return
    matrix = data.to_numpy(dtype=float)
    fig, ax = plt.subplots(
        figsize=(max(8, data.shape[1] * 1.2), max(5, data.shape[0] * 0.65))
    )
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(data.columns.tolist(), rotation=35, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index.tolist())
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            label = "NA" if np.isnan(value) else f"{value:.2f}"
            color = "black" if np.isnan(value) or value < 0.55 else "white"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    cbar = fig.colorbar(image, ax=ax)
    cbar.ax.set_ylabel("Score", rotation=90)
    save_figure(fig, output_path_no_suffix, save_pdf)


def plot_site_accuracy_ranking(
    domain_summary: pd.DataFrame, output_path_no_suffix: Path, save_pdf: bool
) -> None:
    overall = (
        domain_summary.groupby("test_domain")
        .agg(mean_accuracy_ood=("accuracy_ood_mean", "mean"))
        .reset_index()
        .sort_values("mean_accuracy_ood", ascending=True)
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(max(10, len(overall) * 1.0), 6))
    x = np.arange(len(overall))
    values = overall["mean_accuracy_ood"].to_numpy(dtype=float)
    bars = ax.bar(x, values, color=_score_colors(values))
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title(
        "ABIDE II baseline M0: site ranking by mean OOD accuracy",
        fontsize=16,
        weight="bold",
    )
    ax.set_xlabel("Held-out site", fontsize=12)
    ax.set_ylabel("Mean OOD accuracy", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(overall["test_domain"].tolist(), rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path_no_suffix, save_pdf)


def plot_site_model_ood_heatmap(
    fold_results: pd.DataFrame, output_path_no_suffix: Path, save_pdf: bool
) -> None:
    metrics = [
        ("auc_ood", "ROC-AUC"),
        ("balacc_ood", "Balanced accuracy"),
        ("f1_ood", "F1-score"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    for ax, (metric_col, label) in zip(axes, metrics):
        pivot = fold_results.pivot_table(
            index="test_domain", columns="model", values=metric_col, aggfunc="mean"
        )
        matrix = pivot.to_numpy(dtype=float)
        image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.tolist(), rotation=30, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist())
        if ax is axes[0]:
            ax.set_ylabel("Held-out site", fontsize=12)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                label_text = "NA" if np.isnan(value) else f"{value:.3f}"
                color = "black" if np.isnan(value) or value < 0.55 else "white"
                ax.text(
                    col_idx,
                    row_idx,
                    label_text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )
    cbar = fig.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.8, location="bottom", pad=0.12
    )
    cbar.ax.set_xlabel("Performance", fontsize=12)
    fig.suptitle(
        "ABIDE II baseline M0: site × model OOD performance",
        fontsize=18,
        weight="bold",
        y=1.02,
    )
    save_figure(fig, output_path_no_suffix, save_pdf)


def plot_suspicious_case_counts(
    suspicious_summary: pd.DataFrame, output_path_no_suffix: Path, save_pdf: bool
) -> None:
    if suspicious_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(suspicious_summary))
    values = suspicious_summary["n_suspicious_cases"].to_numpy(dtype=float)
    bars = ax.bar(x, values, color="#d95f02")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.3,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(
        "Suspicious pairwise transfer cases by model", fontsize=16, weight="bold"
    )
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Count of suspicious cases", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(suspicious_summary["model"].tolist(), rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path_no_suffix, save_pdf)


def plot_threshold_recovery(
    suspicious_df: pd.DataFrame, output_path_no_suffix: Path, save_pdf: bool
) -> None:
    if suspicious_df.empty:
        return
    summary = (
        suspicious_df.groupby("model", dropna=False)
        .agg(
            mean_default_f1=("test_f1_threshold_0_5", "mean"),
            mean_tuned_f1=("test_f1_train_tuned_threshold", "mean"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary))
    width = 0.32
    defaults = summary["mean_default_f1"].to_numpy(dtype=float)
    tuned = summary["mean_tuned_f1"].to_numpy(dtype=float)
    bars_default = ax.bar(
        x - width / 2,
        defaults,
        width=width,
        label="Default threshold 0.5",
        color="#d73027",
    )
    bars_tuned = ax.bar(
        x + width / 2,
        tuned,
        width=width,
        label="Training-only tuned threshold",
        color="#1a9850",
    )
    for bars in (bars_default, bars_tuned):
        for bar in bars:
            value = bar.get_height()
            if pd.notna(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_title(
        "Effect of training-only threshold tuning on suspicious pairwise cases",
        fontsize=16,
        weight="bold",
    )
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean F1-score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"].tolist(), rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, output_path_no_suffix, save_pdf)


def plot_pairwise_heatmaps(
    pairwise_results: pd.DataFrame, plots_dir: Path, save_pdf: bool
) -> None:
    for model_name in sorted(pairwise_results["model"].unique().tolist()):
        model_df = pairwise_results.loc[pairwise_results["model"] == model_name].copy()
        for metric, label in [
            ("accuracy", "Accuracy"),
            ("auc", "ROC-AUC"),
            ("f1", "F1-score"),
        ]:
            pivot = (
                model_df.pivot_table(
                    index="train_domain",
                    columns="test_domain",
                    values=metric,
                    aggfunc="mean",
                )
                .sort_index()
                .sort_index(axis=1)
            )
            heatmap_with_annotations(
                pivot,
                title=f"{model_name}: train site → test site ({label})",
                xlabel="Test site",
                ylabel="Train site",
                output_path_no_suffix=plots_dir
                / f"train_test_heatmap_{sanitize_name(metric)}_{sanitize_name(model_name)}",
                save_pdf=save_pdf,
            )


def regenerate_plots(output_dir: Path, save_pdf: bool) -> None:
    plots_dir = ensure_dir(output_dir / "plots")
    fold_results = pd.read_csv(output_dir / "fold_results.csv")
    model_summary = pd.read_csv(output_dir / "model_summary.csv")
    domain_summary = pd.read_csv(output_dir / "domain_summary.csv")
    pairwise_results = pd.read_csv(output_dir / "train_test_site_results.csv")

    plot_metric_by_domain(
        fold_results,
        "accuracy_ood",
        "ABIDE II baseline M0: OOD accuracy by held-out site",
        "Accuracy",
        plots_dir / "accuracy_ood_by_domain",
        save_pdf,
    )
    plot_metric_by_domain(
        fold_results,
        "auc_ood",
        "ABIDE II baseline M0: OOD ROC-AUC by held-out site",
        "ROC-AUC",
        plots_dir / "auc_ood_by_domain",
        save_pdf,
    )
    plot_metric_by_domain(
        fold_results,
        "f1_ood",
        "ABIDE II baseline M0: OOD F1-score by held-out site",
        "F1-score",
        plots_dir / "f1_ood_by_domain",
        save_pdf,
    )

    for metric, label, filename in [
        ("accuracy_ood", "Heatmap: OOD accuracy", "heatmap_accuracy_ood"),
        ("auc_ood", "Heatmap: OOD ROC-AUC", "heatmap_auc_ood"),
        ("f1_ood", "Heatmap: OOD F1-score", "heatmap_f1_ood"),
    ]:
        pivot = fold_results.pivot_table(
            index="model", columns="test_domain", values=metric, aggfunc="mean"
        )
        heatmap_with_annotations(
            pivot, label, "Held-out site", "Model", plots_dir / filename, save_pdf
        )

    plot_model_comparison(
        model_summary, plots_dir / "mean_performance_by_model", save_pdf
    )
    plot_site_accuracy_ranking(
        domain_summary, plots_dir / "site_accuracy_ranking", save_pdf
    )
    plot_site_model_ood_heatmap(
        fold_results, plots_dir / "heatmap_site_model_ood", save_pdf
    )
    plot_pairwise_heatmaps(pairwise_results, plots_dir, save_pdf)

    suspicious_summary_path = output_dir / "suspicious_pairwise_summary_by_model.csv"
    suspicious_cases_path = output_dir / "suspicious_pairwise_cases.csv"
    if suspicious_summary_path.exists():
        suspicious_summary = pd.read_csv(suspicious_summary_path)
        plot_suspicious_case_counts(
            suspicious_summary, plots_dir / "suspicious_pairwise_case_counts", save_pdf
        )
    if suspicious_cases_path.exists():
        suspicious_cases = pd.read_csv(suspicious_cases_path)
        plot_threshold_recovery(
            suspicious_cases,
            plots_dir / "suspicious_pairwise_threshold_recovery",
            save_pdf,
        )


def dataframe_block(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows available._"
    view = df.head(max_rows) if max_rows is not None else df
    return "```text\n" + view.to_string(index=False) + "\n```"


def write_custom_report(
    config: AbideConfig,
    audit: dict[str, Any],
    site_counts_before: pd.DataFrame,
    site_counts_after: pd.DataFrame,
    leakage_checks: pd.DataFrame,
    output_dir: Path,
) -> None:
    model_summary = pd.read_csv(output_dir / "model_summary.csv")
    domain_summary = pd.read_csv(output_dir / "domain_summary.csv")
    shuffle_summary = pd.read_csv(output_dir / "shuffle_test_model_summary.csv")
    lodo_site_diagnostics = pd.read_csv(output_dir / "lodo_site_diagnostics.csv")
    suspicious_pairwise_cases = pd.read_csv(
        output_dir / "suspicious_pairwise_cases.csv"
    )
    low_f1_pairwise_cases = pd.read_csv(output_dir / "low_f1_pairwise_cases.csv")
    suspicious_pairwise_summary = pd.read_csv(
        output_dir / "suspicious_pairwise_summary_by_model.csv"
    )

    hardest_sites = (
        domain_summary.groupby("test_domain")
        .agg(
            mean_accuracy_ood=("accuracy_ood_mean", "mean"),
            mean_auc_ood=("auc_ood_mean", "mean"),
            mean_f1_ood=("f1_ood_mean", "mean"),
        )
        .reset_index()
        .sort_values(["mean_accuracy_ood", "mean_auc_ood"], ascending=[True, True])
        .head(5)
    )
    lodo_suspicious = lodo_site_diagnostics.loc[
        lodo_site_diagnostics["suspicious_case"]
    ].copy()
    pairwise_threshold_issue = suspicious_pairwise_cases.loc[
        suspicious_pairwise_cases["suspected_cause"]
        .fillna("")
        .str.contains("threshold issue plausible|materially recovers F1", regex=True)
    ].copy()
    pairwise_collapse = suspicious_pairwise_cases.loc[
        suspicious_pairwise_cases["suspected_cause"]
        .fillna("")
        .str.contains("genuine transport failure", regex=True)
    ].copy()
    top_suspicious_cases = (
        suspicious_pairwise_cases.sort_values(
            ["f1", "auc", "threshold_gain_f1"],
            ascending=[True, False, False],
            na_position="last",
        )
        .loc[
            :,
            [
                "model",
                "train_domain",
                "test_domain",
                "n_samples",
                "positive_rate_true",
                "positive_rate_pred",
                "f1",
                "balacc",
                "auc",
                "threshold_gain_f1",
                "suspected_cause",
            ],
        ]
        .head(12)
    )

    report_lines = [
        "# ABIDE II Baseline M0",
        "",
        "## Dataset overview",
        f"- Input path: `{config.input_path}`",
        f"- Total subjects before site filtering: `{audit['n_subjects_total']}`",
        f"- Total rows before site filtering: `{audit['n_rows_total']}`",
        f"- Total sites before filtering: `{audit['n_sites_total']}`",
        f"- Sites excluded from supervised evaluation because they contain a single class only: `{audit['excluded_single_class_sites']}`",
        f"- Retained subjects: `{audit['n_subjects_retained']}`",
        f"- Retained sites: `{audit['n_sites_retained']}`",
        f"- Class counts after filtering: `{audit['class_counts_total']}`",
        "",
        "## Site summary before filtering",
        dataframe_block(site_counts_before.round(4)),
        "",
        "## Site summary after filtering",
        dataframe_block(site_counts_after.round(4)),
        "",
        "## Leakage safeguards",
        "- LODO outer split by site.",
        "- All preprocessing is fit on the training fold only.",
        "- Subject ID, site, and textual diagnosis labels are excluded from the feature matrix.",
        "- Missing values are imputed inside the fold; categorical variables are one-hot encoded inside the fold; numeric variables are scaled inside the fold.",
        "- Shuffle-label sanity test is included and written separately.",
        "- Additional diagnostics inspect whether very low F1 values arise from true OOD collapse, single-class prediction, or threshold/calibration mismatch.",
        "",
        "## Debugging findings on F1 = 0 cases",
        f"- Suspicious LODO outer-fold cases (`F1 <= {config.suspicious_f1_threshold}` or single-class prediction): `{len(lodo_suspicious)}`.",
        f"- Actual LODO low-F1 cases (`F1 <= {config.suspicious_f1_threshold}`): `{len(lodo_site_diagnostics.loc[lodo_site_diagnostics['f1'] <= config.suspicious_f1_threshold])}`.",
        f"- Suspicious pairwise train-site -> test-site cases: `{len(suspicious_pairwise_cases)}`.",
        f"- Actual pairwise low-F1 cases (`F1 <= {config.suspicious_f1_threshold}`): `{len(low_f1_pairwise_cases)}`.",
        f"- Pairwise cases plausibly driven by threshold/calibration mismatch: `{len(pairwise_threshold_issue)}`.",
        f"- Pairwise cases that look like genuine near-chance transport failure: `{len(pairwise_collapse)}`.",
        "- The main LODO baseline is therefore more stable than the harsher single-site transfer matrix. Zero-F1 cells in the pairwise heatmaps should be interpreted as stress-test failures, not as evidence that the outer baseline itself is broken.",
        "",
        "## Metric implementation check",
        "- Accuracy uses `accuracy_score` on binary predictions.",
        "- Balanced accuracy uses `balanced_accuracy_score` on binary predictions.",
        "- F1 uses `f1_score(..., zero_division=0)`, so it correctly returns `0` when the model predicts no positives while positives are present.",
        "- ROC-AUC is computed only when both classes are present in the evaluation set.",
        "- Extra diagnostics report the true and predicted class mix per site, plus confusion-matrix counts, so collapse vs threshold issues can be distinguished explicitly.",
        "",
        "## Model summary",
        dataframe_block(model_summary.round(4)),
        "",
        "## Hardest held-out sites",
        dataframe_block(hardest_sites.round(4)),
        "",
        "## LODO site diagnostics",
        dataframe_block(
            lodo_site_diagnostics.loc[
                :,
                [
                    "model",
                    "test_domain",
                    "n_samples",
                    "positive_rate_true",
                    "positive_rate_pred",
                    "n_pred_positive",
                    "n_pred_negative",
                    "accuracy",
                    "balacc",
                    "f1",
                    "auc",
                    "predicted_single_class",
                ],
            ].round(4),
            max_rows=30,
        ),
        "",
        "## Suspicious pairwise cases by model",
        dataframe_block(suspicious_pairwise_summary.round(4)),
        "",
        "## Example suspicious pairwise cases",
        dataframe_block(top_suspicious_cases.round(4)),
        "",
        "## Shuffle-label sanity check",
        dataframe_block(shuffle_summary.round(4)),
        "",
        "## Leakage checks",
        dataframe_block(leakage_checks),
        "",
        "## Interpretation",
        "- This baseline is intentionally raw M0: no ComBat, no reweighting, no DASHI-guided weighting, and no domain adaptation.",
        "- The retained ABIDE sites now all contain both ASD and control subjects, so fold-level AUC and F1 are interpretable across sites.",
        "- If OOD performance varies substantially across sites, that variability is the raw transportability signal to compare later against harmonization and domain-aware methods.",
        "- When AUC remains moderate-to-good but F1 collapses to zero with all-negative predictions, the issue is likely threshold/calibration mismatch rather than complete lack of ranking signal.",
        "- When both AUC and balanced accuracy collapse toward chance, the failure is more consistent with genuine loss of transportability between sites.",
    ]
    (output_dir / "baseline_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


def write_readme(config: AbideConfig, output_dir: Path) -> None:
    readme = f"""# ABIDE II baseline M0

This folder contains the cross-site baseline M0 for ABIDE II.

## Modeling setup

- Target: `disease_level_code` (`1 = ASD`, `2 = Control`, with ASD treated as the positive class)
- Domain: `site`
- Validation: leave-one-domain-out (LODO)
- Models: Logistic Regression, Random Forest, HistGradientBoosting
- Pairwise train-site -> test-site heatmaps are included
- Shuffle-label sanity test is included

## Run

```bash
python3 abide2_m0_uniform.py \\
  --input "{config.input_path}" \\
  --workspace-output "{config.workspace_output_dir}" \\
  --final-output "{config.final_output_dir}"
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> AbideConfig:
    parser = argparse.ArgumentParser(
        description="ABIDE II baseline M0 with LODO and train->test heatmaps"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Path to the ABIDE II preprocessed CSV",
    )
    parser.add_argument(
        "--workspace-output",
        default=DEFAULT_WORKSPACE_OUTPUT,
        help="Workspace output directory",
    )
    parser.add_argument(
        "--final-output", default=DEFAULT_FINAL_OUTPUT, help="Final output directory"
    )
    args = parser.parse_args()
    return AbideConfig(
        input_path=args.input,
        workspace_output_dir=args.workspace_output,
        final_output_dir=args.final_output,
    )


def main() -> None:
    config = parse_args()
    core = load_baseline_core()
    workspace_output = Path(config.workspace_output_dir)
    clean_output_dir(workspace_output)

    filtered_df, audit, site_counts_before, site_counts_after = load_and_filter_abide(
        config
    )
    filtered_input_path = write_filtered_input(filtered_df, workspace_output)

    run_main_baseline(core, config, filtered_input_path, workspace_output)
    run_shuffle_test(core, config, filtered_df, workspace_output)

    lodo_site_diagnostics, lodo_confusion = build_lodo_site_diagnostics(
        workspace_output, config
    )
    pairwise_site_diagnostics, pairwise_confusion, suspicious_pairwise_cases = (
        build_pairwise_site_diagnostics(workspace_output, config)
    )
    suspicious_pairwise_cases = add_threshold_diagnostics_to_suspicious_cases(
        suspicious_pairwise_cases,
        filtered_df,
        core,
        config,
        filtered_input_path,
        workspace_output,
    )
    suspicious_pairwise_summary = summarise_suspicious_cases(suspicious_pairwise_cases)
    suspicious_lodo_cases = (
        lodo_site_diagnostics.loc[lodo_site_diagnostics["suspicious_case"]]
        .copy()
        .reset_index(drop=True)
    )
    low_f1_lodo_cases = (
        lodo_site_diagnostics.loc[
            lodo_site_diagnostics["f1"] <= config.suspicious_f1_threshold
        ]
        .copy()
        .reset_index(drop=True)
    )
    low_f1_pairwise_cases = (
        pairwise_site_diagnostics.loc[
            pairwise_site_diagnostics["f1"] <= config.suspicious_f1_threshold
        ]
        .copy()
        .reset_index(drop=True)
    )

    leakage_checks = build_leakage_checks(filtered_df, config)
    leakage_checks.to_csv(workspace_output / "leakage_checks.csv", index=False)
    site_counts_before.to_csv(
        workspace_output / "site_subject_summary_before_filter.csv", index=False
    )
    site_counts_after.to_csv(
        workspace_output / "site_subject_summary_after_filter.csv", index=False
    )
    lodo_site_diagnostics.to_csv(
        workspace_output / "lodo_site_diagnostics.csv", index=False
    )
    lodo_confusion.to_csv(workspace_output / "lodo_confusion_matrices.csv", index=False)
    pairwise_site_diagnostics.to_csv(
        workspace_output / "pairwise_site_diagnostics.csv", index=False
    )
    pairwise_confusion.to_csv(
        workspace_output / "pairwise_confusion_matrices.csv", index=False
    )
    suspicious_lodo_cases.to_csv(
        workspace_output / "suspicious_lodo_cases.csv", index=False
    )
    low_f1_lodo_cases.to_csv(workspace_output / "low_f1_lodo_cases.csv", index=False)
    suspicious_pairwise_cases.to_csv(
        workspace_output / "suspicious_pairwise_cases.csv", index=False
    )
    low_f1_pairwise_cases.to_csv(
        workspace_output / "low_f1_pairwise_cases.csv", index=False
    )
    suspicious_pairwise_summary.to_csv(
        workspace_output / "suspicious_pairwise_summary_by_model.csv", index=False
    )
    with (workspace_output / "abide2_audit_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)

    regenerate_plots(workspace_output, config.save_pdf_plots)
    write_custom_report(
        config,
        audit,
        site_counts_before,
        site_counts_after,
        leakage_checks,
        workspace_output,
    )
    write_readme(config, workspace_output)
    with (workspace_output / "abide2_m0_uniform_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(asdict(config), handle, indent=2, sort_keys=True)
    shutil.copy2(Path(__file__), workspace_output / "abide2_m0_uniform.py")


if __name__ == "__main__":
    main()
