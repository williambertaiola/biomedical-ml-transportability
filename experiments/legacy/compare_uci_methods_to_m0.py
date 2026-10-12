"""Transportability experiment: compare uci methods to m0."""

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
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path("/tmp/mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DIRECT_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]
DASHI_SHARED_METRICS = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
DASHI_SITE_METRICS = ["accuracy", "balanced_accuracy", "f1"]

BASELINE_MODEL_MAP = {
    "LogisticRegression": "logistic_regression",
    "RandomForest": "random_forest",
    "HistGradientBoosting": "hist_gradient_boosting",
    "logistic": "logistic_regression",
    "random_forest": "random_forest",
    "hist_gradient_boosting": "hist_gradient_boosting",
}

SITE_MAP = {
    "Cleveland": "cleveland",
    "Hungarian": "hungarian",
    "Switzerland": "switzerland",
    "VA": "long_beach_va",
    "cleveland": "cleveland",
    "hungarian": "hungarian",
    "switzerland": "switzerland",
    "long_beach_va": "long_beach_va",
}

GROUP_ORDER = ["logistic_regression", "random_forest", "hist_gradient_boosting"]

DIRECT_LABELS = {
    "baseline_m0__logistic_regression": "M0 Baseline",
    "baseline_m0__random_forest": "M0 Baseline",
    "baseline_m0__hist_gradient_boosting": "M0 Baseline",
    "weighting__raw": "Weighting: Raw",
    "weighting__domain_balanced": "Weighting: Domain Balanced",
    "weighting__similarity_based": "Weighting: Similarity Based",
    "combat__baseline_raw": "ComBat: Raw",
    "combat__combat_train_only": "ComBat: Train Only",
    "transductive__logistic_regression__coral_only": "Transductive: CORAL Only",
    "transductive__logistic_regression__coral_plus_pseudolabel": "Transductive: CORAL + Pseudo-labeling",
    "transductive__random_forest__coral_only": "Transductive: CORAL Only",
    "transductive__random_forest__coral_plus_pseudolabel": "Transductive: CORAL + Pseudo-labeling",
    "transductive__hist_gradient_boosting__coral_only": "Transductive: CORAL Only",
    "transductive__hist_gradient_boosting__coral_plus_pseudolabel": "Transductive: CORAL + Pseudo-labeling",
}

DIRECT_SORT_ORDER = {
    "baseline_m0__logistic_regression": 0,
    "weighting__raw": 1,
    "weighting__domain_balanced": 2,
    "weighting__similarity_based": 3,
    "combat__baseline_raw": 4,
    "combat__combat_train_only": 5,
    "transductive__logistic_regression__coral_only": 6,
    "transductive__logistic_regression__coral_plus_pseudolabel": 7,
    "baseline_m0__random_forest": 0,
    "transductive__random_forest__coral_only": 1,
    "transductive__random_forest__coral_plus_pseudolabel": 2,
    "baseline_m0__hist_gradient_boosting": 0,
    "transductive__hist_gradient_boosting__coral_only": 1,
    "transductive__hist_gradient_boosting__coral_plus_pseudolabel": 2,
}

DASHI_LABELS = {
    "baseline_m0_single_source__hist_gradient_boosting": "M0 Single-Source Baseline",
    "dashi_supervised__hist_gradient_boosting_off_diagonal": "DASHI Supervised Off-Diagonal",
}


def normalize_site(value: str) -> str:
    return SITE_MAP.get(str(value), str(value).strip().lower())


def normalize_model(value: str) -> str:
    return BASELINE_MODEL_MAP.get(str(value), str(value).strip().lower())


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_binary_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_pred = frame["y_pred"].to_numpy(dtype=int)
    y_prob = frame["y_prob"].to_numpy(dtype=float)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_prob),
    }


def order_direct_rows(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["sort_key"] = ordered["method_id"].map(DIRECT_SORT_ORDER).fillna(999)
    ordered = ordered.sort_values(["comparison_group", "sort_key", "method_id"]).drop(
        columns="sort_key"
    )
    return ordered.reset_index(drop=True)


def load_baseline_direct(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = base_dir / "baseline" / "model_summary.csv"
    summary_df = pd.read_csv(summary_path)
    overall_rows: list[dict[str, object]] = []
    for _, row in summary_df.iterrows():
        group = normalize_model(row["model"])
        overall_rows.append(
            {
                "comparison_scope": "direct_lodo",
                "comparison_group": group,
                "category": "baseline_m0",
                "method_id": f"baseline_m0__{group}",
                "label": DIRECT_LABELS[f"baseline_m0__{group}"],
                "site": "overall",
                "accuracy": float(row["accuracy_ood_mean"]),
                "balanced_accuracy": float(row["balacc_ood_mean"]),
                "precision": float(row["precision_ood_mean"]),
                "recall": float(row["recall_ood_mean"]),
                "f1": float(row["f1_ood_mean"]),
                "roc_auc": float(row["auc_ood_mean"]),
            }
        )

    pred_path = base_dir / "baseline" / "prediction_records.csv"
    pred_df = pd.read_csv(pred_path)
    pred_df = pred_df.loc[pred_df["split"].astype(str).str.lower() == "ood"].copy()
    pred_df["comparison_group"] = pred_df["model"].map(normalize_model)
    pred_df["site"] = pred_df["outer_test_domain"].map(normalize_site)
    site_rows: list[dict[str, object]] = []
    for (group, site), group_df in pred_df.groupby(
        ["comparison_group", "site"], dropna=False
    ):
        metrics = compute_binary_metrics(group_df)
        site_rows.append(
            {
                "comparison_scope": "direct_lodo",
                "comparison_group": group,
                "category": "baseline_m0",
                "method_id": f"baseline_m0__{group}",
                "label": DIRECT_LABELS[f"baseline_m0__{group}"],
                "site": site,
                **metrics,
            }
        )
    return pd.DataFrame(overall_rows), pd.DataFrame(site_rows)


def load_weighting(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_path = base_dir / "weighting" / "lodo_results.csv"
    results_df = pd.read_csv(results_path)
    results_df["comparison_group"] = "logistic_regression"
    results_df["site"] = results_df["holdout_domain"].map(normalize_site)
    results_df["method_id"] = results_df["method"].map(
        lambda value: f"weighting__{value}"
    )
    results_df["label"] = results_df["method_id"].map(DIRECT_LABELS)
    overall = (
        results_df.groupby(["comparison_group", "method_id", "label"], dropna=False)[
            DIRECT_METRICS
        ]
        .mean()
        .reset_index()
    )
    overall["comparison_scope"] = "direct_lodo"
    overall["category"] = "weighting"
    overall["site"] = "overall"
    site = results_df[
        ["comparison_group", "method_id", "label", "site", *DIRECT_METRICS]
    ].copy()
    site["comparison_scope"] = "direct_lodo"
    site["category"] = "weighting"
    return overall[
        [
            "comparison_scope",
            "comparison_group",
            "category",
            "method_id",
            "label",
            "site",
            *DIRECT_METRICS,
        ]
    ], site[
        [
            "comparison_scope",
            "comparison_group",
            "category",
            "method_id",
            "label",
            "site",
            *DIRECT_METRICS,
        ]
    ]


def load_combat(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = base_dir / "combat" / "predictions.csv"
    pred_df = pd.read_csv(pred_path)
    pred_df["comparison_group"] = "logistic_regression"
    pred_df["site"] = pred_df["holdout_domain"].map(normalize_site)
    pred_df["method_id"] = pred_df["pipeline"].map(lambda value: f"combat__{value}")
    pred_df["label"] = pred_df["method_id"].map(DIRECT_LABELS)

    site_rows: list[dict[str, object]] = []
    for (method_id, label, site), group_df in pred_df.groupby(
        ["method_id", "label", "site"], dropna=False
    ):
        metrics = compute_binary_metrics(group_df)
        site_rows.append(
            {
                "comparison_scope": "direct_lodo",
                "comparison_group": "logistic_regression",
                "category": "combat",
                "method_id": method_id,
                "label": label,
                "site": site,
                **metrics,
            }
        )

    site = pd.DataFrame(site_rows)
    overall = (
        site.groupby(
            ["comparison_group", "category", "method_id", "label"], dropna=False
        )[DIRECT_METRICS]
        .mean()
        .reset_index()
    )
    overall["comparison_scope"] = "direct_lodo"
    overall["site"] = "overall"
    return overall[
        [
            "comparison_scope",
            "comparison_group",
            "category",
            "method_id",
            "label",
            "site",
            *DIRECT_METRICS,
        ]
    ], site[
        [
            "comparison_scope",
            "comparison_group",
            "category",
            "method_id",
            "label",
            "site",
            *DIRECT_METRICS,
        ]
    ]


def load_transductive(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_path = base_dir / "transductive_learning" / "tables" / "fold_metrics.csv"
    fold_df = pd.read_csv(fold_path)
    fold_df["comparison_group"] = fold_df["model_name"].map(normalize_model)
    fold_df["site"] = fold_df["target_site"].map(normalize_site)
    fold_df["method_id"] = fold_df.apply(
        lambda row: f"transductive__{row['comparison_group']}__{row['variant']}",
        axis=1,
    )
    fold_df["label"] = fold_df["method_id"].map(DIRECT_LABELS)

    overall = (
        fold_df.groupby(["comparison_group", "method_id", "label"], dropna=False)[
            DIRECT_METRICS
        ]
        .mean()
        .reset_index()
    )
    overall["comparison_scope"] = "direct_lodo"
    overall["category"] = "transductive"
    overall["site"] = "overall"

    site = fold_df[
        ["comparison_group", "method_id", "label", "site", *DIRECT_METRICS]
    ].copy()
    site["comparison_scope"] = "direct_lodo"
    site["category"] = "transductive"
    return overall[
        [
            "comparison_scope",
            "comparison_group",
            "category",
            "method_id",
            "label",
            "site",
            *DIRECT_METRICS,
        ]
    ], site[
        [
            "comparison_scope",
            "comparison_group",
            "category",
            "method_id",
            "label",
            "site",
            *DIRECT_METRICS,
        ]
    ]


def load_dashi_secondary(
    base_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dashi_path = (
        base_dir / "dashi" / "tables" / "supervised_multibatch_metrics_long.tsv"
    )
    dashi_df = pd.read_csv(dashi_path, sep="\t")
    dashi_df = dashi_df.loc[
        (dashi_df["split"].astype(str).str.lower() == "test")
        & (dashi_df["train_batches"] != dashi_df["test_batch"])
    ].copy()
    pivot = dashi_df.pivot_table(
        index=["train_batches", "test_batch"],
        columns="metric",
        values="value",
        aggfunc="first",
    )
    pivot = pivot.reset_index()
    pivot["test_batch"] = pivot["test_batch"].map(normalize_site)
    pivot["accuracy"] = pd.to_numeric(pivot["ACCURACY"], errors="coerce")
    pivot["balanced_accuracy"] = (
        pd.to_numeric(pivot["RECALL_0"], errors="coerce")
        + pd.to_numeric(pivot["RECALL_1"], errors="coerce")
    ) / 2.0
    pivot["precision"] = pd.to_numeric(pivot["PRECISION_1"], errors="coerce")
    pivot["recall"] = pd.to_numeric(pivot["RECALL_1"], errors="coerce")
    pivot["f1"] = pd.to_numeric(pivot["F1-SCORE_1"], errors="coerce")
    pivot["roc_auc"] = pd.to_numeric(pivot["AUC_1"], errors="coerce")

    baseline_results_path = (
        base_dir / "baseline" / "training_domain_experiment_results.csv"
    )
    baseline_results = pd.read_csv(baseline_results_path)
    baseline_results = baseline_results.loc[
        (baseline_results["model"] == "hist_gradient_boosting")
        & (baseline_results["n_train_sites"] == 1)
    ].copy()
    baseline_results["site"] = baseline_results["test_site"].map(normalize_site)
    baseline_results = baseline_results.rename(columns={"auc": "roc_auc"})

    dashi_overall = pd.DataFrame(
        [
            {
                "comparison_scope": "secondary_single_source",
                "comparison_group": "hist_gradient_boosting",
                "category": "baseline_m0_single_source",
                "method_id": "baseline_m0_single_source__hist_gradient_boosting",
                "label": DASHI_LABELS[
                    "baseline_m0_single_source__hist_gradient_boosting"
                ],
                "site": "overall",
                "accuracy": float(baseline_results["accuracy"].mean()),
                "balanced_accuracy": float(
                    baseline_results["balanced_accuracy"].mean()
                ),
                "f1": float(baseline_results["f1"].mean()),
                "roc_auc": float(baseline_results["roc_auc"].mean()),
                "roc_auc_valid_pairs": int(baseline_results["roc_auc"].notna().sum()),
            },
            {
                "comparison_scope": "secondary_single_source",
                "comparison_group": "hist_gradient_boosting",
                "category": "dashi_supervised",
                "method_id": "dashi_supervised__hist_gradient_boosting_off_diagonal",
                "label": DASHI_LABELS[
                    "dashi_supervised__hist_gradient_boosting_off_diagonal"
                ],
                "site": "overall",
                "accuracy": float(pivot["accuracy"].mean()),
                "balanced_accuracy": float(pivot["balanced_accuracy"].mean()),
                "f1": float(pivot["f1"].mean()),
                "roc_auc": float(pivot["roc_auc"].mean()),
                "roc_auc_valid_pairs": int(pivot["roc_auc"].notna().sum()),
            },
        ]
    )

    baseline_site = (
        baseline_results.groupby("site", dropna=False)[
            ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
        ]
        .mean()
        .reset_index()
    )
    baseline_site["comparison_scope"] = "secondary_single_source"
    baseline_site["comparison_group"] = "hist_gradient_boosting"
    baseline_site["category"] = "baseline_m0_single_source"
    baseline_site["method_id"] = "baseline_m0_single_source__hist_gradient_boosting"
    baseline_site["label"] = DASHI_LABELS[
        "baseline_m0_single_source__hist_gradient_boosting"
    ]

    dashi_site = (
        pivot.groupby("test_batch", dropna=False)[
            ["accuracy", "balanced_accuracy", "f1"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"test_batch": "site"})
    )
    dashi_site["comparison_scope"] = "secondary_single_source"
    dashi_site["comparison_group"] = "hist_gradient_boosting"
    dashi_site["category"] = "dashi_supervised"
    dashi_site["method_id"] = "dashi_supervised__hist_gradient_boosting_off_diagonal"
    dashi_site["label"] = DASHI_LABELS[
        "dashi_supervised__hist_gradient_boosting_off_diagonal"
    ]

    dashi_absolute_full = pd.DataFrame(
        [
            {
                "comparison_scope": "secondary_single_source",
                "comparison_group": "hist_gradient_boosting",
                "category": "dashi_supervised",
                "method_id": "dashi_supervised__hist_gradient_boosting_off_diagonal",
                "label": DASHI_LABELS[
                    "dashi_supervised__hist_gradient_boosting_off_diagonal"
                ],
                "site": "overall",
                "accuracy": float(pivot["accuracy"].mean()),
                "balanced_accuracy": float(pivot["balanced_accuracy"].mean()),
                "precision": float(pivot["precision"].mean()),
                "recall": float(pivot["recall"].mean()),
                "f1": float(pivot["f1"].mean()),
                "roc_auc": float(pivot["roc_auc"].mean()),
                "roc_auc_valid_pairs": int(pivot["roc_auc"].notna().sum()),
            }
        ]
    )

    return dashi_overall, baseline_site, dashi_site, dashi_absolute_full


def compute_delta_vs_baseline(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["comparison_group", "site"]
    for keys, subset in frame.groupby(group_cols, dropna=False):
        subset = subset.copy()
        baseline = subset.loc[
            subset["category"].astype(str).str.startswith("baseline_m0")
        ]
        if baseline.empty:
            continue
        baseline_row = baseline.iloc[0]
        for _, row in subset.iterrows():
            record = {
                "comparison_scope": row["comparison_scope"],
                "comparison_group": row["comparison_group"],
                "category": row["category"],
                "method_id": row["method_id"],
                "label": row["label"],
                "site": row["site"],
                "reference_method_id": baseline_row["method_id"],
                "reference_label": baseline_row["label"],
            }
            for metric in metrics:
                record[metric] = (
                    float(row[metric]) if pd.notna(row[metric]) else float("nan")
                )
                record[f"delta_{metric}_vs_m0"] = (
                    float(row[metric] - baseline_row[metric])
                    if pd.notna(row[metric]) and pd.notna(baseline_row[metric])
                    else float("nan")
                )
            rows.append(record)
    return pd.DataFrame(rows)


def plot_delta_heatmap(
    frame: pd.DataFrame,
    value_columns: list[str],
    title: str,
    output_path: Path,
    label_map: dict[str, str],
) -> None:
    if frame.empty:
        return

    working = frame.copy()
    working["label"] = working["method_id"].map(label_map).fillna(working["label"])
    working = working.set_index("label")[value_columns]
    matrix = working.to_numpy(dtype=float)
    mask = np.isnan(matrix) | np.isclose(matrix, 0.0, atol=5e-6)
    masked = np.ma.masked_array(matrix, mask=mask)

    finite = matrix[~np.isnan(matrix)]
    vmax = np.max(np.abs(finite)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)

    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad(color="white")

    fig_width = max(8.0, 1.45 * len(value_columns) + 2.4)
    fig_height = max(2.8, 0.65 * len(working.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    im = ax.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.set_ylabel("Delta vs M0", rotation=270, labelpad=18)

    ax.set_xticks(np.arange(len(value_columns)))
    ax.set_xticklabels(
        [
            column.replace("delta_", "").replace("_vs_m0", "").replace("_", " ").title()
            for column in value_columns
        ],
        rotation=25,
        ha="right",
    )
    ax.set_yticks(np.arange(len(working.index)))
    ax.set_yticklabels(list(working.index))
    ax.set_title(title, pad=14)

    ax.set_xticks(np.arange(-0.5, len(value_columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(working.index), 1), minor=True)
    ax.grid(which="minor", color="#d9d9d9", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "" if math.isnan(value) else f"{value:.4f}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(
    output_dir: Path,
    direct_overall_delta: pd.DataFrame,
    dashi_overall_delta: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# UCI Methods vs External Baseline M0")
    lines.append("")
    lines.append("## Scope")
    lines.append(
        "- This comparison layer keeps the original method outputs unchanged and adds an external comparison against the user-defined Baseline M0."
    )
    lines.append(
        "- Primary direct comparison: Baseline M0 LODO with all three source sites (`n_train_sites = 3`) vs weighting, ComBat, and transductive learning."
    )
    lines.append(
        "- Secondary comparison: DASHI supervised off-diagonal multi-batch transfer vs the single-source HistGradientBoosting branch of Baseline M0 (`n_train_sites = 1`)."
    )
    lines.append(
        "- All values are reported in English and restricted to thesis-relevant metrics only."
    )
    lines.append("")
    lines.append("## Direct Comparison Rules")
    lines.append(
        "- `weighting` and `ComBat` are compared against the `LogisticRegression` branch of Baseline M0."
    )
    lines.append(
        "- `transductive learning` is compared against the matching Baseline M0 model (`logistic_regression`, `random_forest`, or `hist_gradient_boosting`)."
    )
    lines.append(
        "- `DASHI` is not mixed into the primary LODO table because its supervised output is a source-target multi-batch transfer experiment rather than a three-source LODO pipeline."
    )
    lines.append("")
    lines.append("## Main Findings")

    for group in GROUP_ORDER:
        group_df = direct_overall_delta.loc[
            (direct_overall_delta["comparison_group"] == group)
            & (direct_overall_delta["category"] != "baseline_m0")
        ].copy()
        if group_df.empty:
            continue
        best_f1 = group_df.sort_values("delta_f1_vs_m0", ascending=False).iloc[0]
        best_auc = group_df.sort_values("delta_roc_auc_vs_m0", ascending=False).iloc[0]
        lines.append(
            f"- `{group}`: best delta F1 vs M0 was `{best_f1['label']}` ({best_f1['delta_f1_vs_m0']:+.4f}); best delta ROC AUC vs M0 was `{best_auc['label']}` ({best_auc['delta_roc_auc_vs_m0']:+.4f})."
        )

    if not dashi_overall_delta.empty:
        dashi_row = dashi_overall_delta.loc[
            dashi_overall_delta["category"] == "dashi_supervised"
        ].iloc[0]
        lines.append(
            "- `DASHI` secondary comparison: relative to the single-source M0 HistGradientBoosting reference, DASHI "
            f"changed accuracy by {dashi_row['delta_accuracy_vs_m0']:+.4f}, balanced accuracy by {dashi_row['delta_balanced_accuracy_vs_m0']:+.4f}, "
            f"F1 by {dashi_row['delta_f1_vs_m0']:+.4f}, and ROC AUC by {dashi_row['delta_roc_auc_vs_m0']:+.4f}."
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("- Exact `0.0000` delta cells are rendered in white in the heatmaps.")
    lines.append(
        "- The DASHI overall ROC AUC summary is based on the valid off-diagonal pairs provided by the original DASHI supervised output."
    )
    lines.append(
        "- No original baseline, weighting, ComBat, transductive, or DASHI files were modified."
    )
    lines.append("")
    lines.append("## Files")
    lines.append(
        "- `tables/direct_lodo_absolute_metrics.csv`: direct comparable absolute metrics."
    )
    lines.append(
        "- `tables/direct_lodo_delta_vs_m0.csv`: direct comparable deltas against external M0."
    )
    lines.append(
        "- `tables/direct_lodo_site_absolute_metrics.csv`: site-level direct comparable metrics."
    )
    lines.append(
        "- `tables/direct_lodo_site_delta_vs_m0.csv`: site-level direct comparable deltas."
    )
    lines.append(
        "- `tables/dashi_secondary_overall_shared_absolute.csv`: DASHI secondary absolute metrics on the shared metric set."
    )
    lines.append(
        "- `tables/dashi_secondary_overall_shared_delta_vs_m0.csv`: DASHI secondary deltas against M0 single-source HistGradientBoosting."
    )
    lines.append(
        "- `plots/`: green-yellow-red delta heatmaps for each direct-comparison group and for DASHI."
    )
    lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_output(base_dir: Path, output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline_overall, baseline_site = load_baseline_direct(base_dir)
    weighting_overall, weighting_site = load_weighting(base_dir)
    combat_overall, combat_site = load_combat(base_dir)
    transductive_overall, transductive_site = load_transductive(base_dir)

    direct_overall = pd.concat(
        [baseline_overall, weighting_overall, combat_overall, transductive_overall],
        ignore_index=True,
    )
    direct_site = pd.concat(
        [baseline_site, weighting_site, combat_site, transductive_site],
        ignore_index=True,
    )

    direct_overall = order_direct_rows(direct_overall)
    direct_site = order_direct_rows(direct_site)
    direct_overall_delta = order_direct_rows(
        compute_delta_vs_baseline(direct_overall, DIRECT_METRICS)
    )
    direct_site_delta = order_direct_rows(
        compute_delta_vs_baseline(direct_site, DIRECT_METRICS)
    )

    dashi_overall, dashi_baseline_site, dashi_site, dashi_full = load_dashi_secondary(
        base_dir
    )
    dashi_overall_delta = compute_delta_vs_baseline(dashi_overall, DASHI_SHARED_METRICS)
    dashi_site_frame = pd.concat([dashi_baseline_site, dashi_site], ignore_index=True)
    dashi_site_delta = compute_delta_vs_baseline(dashi_site_frame, DASHI_SITE_METRICS)

    direct_overall.to_csv(tables_dir / "direct_lodo_absolute_metrics.csv", index=False)
    direct_overall_delta.to_csv(tables_dir / "direct_lodo_delta_vs_m0.csv", index=False)
    direct_site.to_csv(
        tables_dir / "direct_lodo_site_absolute_metrics.csv", index=False
    )
    direct_site_delta.to_csv(
        tables_dir / "direct_lodo_site_delta_vs_m0.csv", index=False
    )
    dashi_overall.to_csv(
        tables_dir / "dashi_secondary_overall_shared_absolute.csv", index=False
    )
    dashi_overall_delta.to_csv(
        tables_dir / "dashi_secondary_overall_shared_delta_vs_m0.csv", index=False
    )
    dashi_site_frame.to_csv(
        tables_dir / "dashi_secondary_site_shared_absolute.csv", index=False
    )
    dashi_site_delta.to_csv(
        tables_dir / "dashi_secondary_site_shared_delta_vs_m0.csv", index=False
    )
    dashi_full.to_csv(
        tables_dir / "dashi_secondary_overall_full_dashi_metrics.csv", index=False
    )

    for group in GROUP_ORDER:
        group_df = direct_overall_delta.loc[
            direct_overall_delta["comparison_group"] == group
        ].copy()
        if group_df.empty:
            continue
        value_cols = [f"delta_{metric}_vs_m0" for metric in DIRECT_METRICS]
        plot_delta_heatmap(
            group_df,
            value_cols,
            title=f"Delta vs External M0 ({group.replace('_', ' ').title()})",
            output_path=plots_dir / f"direct_delta_heatmap_{group}.png",
            label_map=DIRECT_LABELS,
        )

    plot_delta_heatmap(
        dashi_overall_delta,
        [f"delta_{metric}_vs_m0" for metric in DASHI_SHARED_METRICS],
        title="DASHI Secondary Delta vs M0 Single-Source HistGradientBoosting",
        output_path=plots_dir / "dashi_secondary_delta_heatmap.png",
        label_map=DASHI_LABELS,
    )

    write_report(output_dir, direct_overall_delta, dashi_overall_delta)

    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# UCI External Baseline Comparison",
                "",
                "- `report.md`: English summary of how each method compares to the external Baseline M0.",
                "- `tables/`: absolute metrics and deltas against M0.",
                "- `plots/`: green-yellow-red delta heatmaps with exact `0.0000` cells shown in white.",
                "",
                "This folder is additive: it does not overwrite or reinterpret the original outputs inside `baseline/`, `dashi/`, `transductive_learning/`, `weighting/`, or `combat/`.",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare UCI method outputs against the external Baseline M0."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(_configured_path("legacy_uci_final")),
        help="Root directory containing the reorganized UCI outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(_configured_path("legacy_uci_m0_comparison_output")),
        help="Directory where the additive comparison package will be created.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_output(args.base_dir, args.output_dir)


if __name__ == "__main__":
    main()
