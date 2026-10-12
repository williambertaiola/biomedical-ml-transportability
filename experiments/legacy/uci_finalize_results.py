"""Transportability experiment: uci finalize results."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.metrics import roc_auc_score


METRIC_ORDER = [
    "accuracy",
    "f1",
    "precision",
    "recall",
    "roc_auc",
    "balanced_accuracy",
]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1": "F1-score",
    "precision": "Precision",
    "recall": "Recall",
    "roc_auc": "ROC AUC",
    "balanced_accuracy": "Balanced Accuracy",
}

SITE_ORDER = ["cleveland", "hungarian", "long_beach_va", "switzerland"]
SITE_LABELS = {
    "cleveland": "Cleveland",
    "hungarian": "Hungarian",
    "long_beach_va": "Long Beach VA",
    "switzerland": "Switzerland",
    "long_beach_va_test": "Long Beach VA",
}
MODEL_LABELS = {
    "HistGradientBoosting": "HistGradientBoosting",
    "LogisticRegression": "LogisticRegression",
    "RandomForest": "RandomForest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "logistic_regression": "LogisticRegression",
    "random_forest": "RandomForest",
    "logistic": "LogisticRegression",
    "random_forest_test": "RandomForest",
}
METHOD_LABELS = {
    "raw": "Raw",
    "domain_balanced": "Domain-balanced",
    "similarity_based": "Similarity-based",
    "baseline_raw": "Baseline raw",
    "combat_train_only": "ComBat train-only",
    "coral_only": "CORAL only",
    "coral_plus_pseudolabel": "CORAL + pseudo-labeling",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def positive_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(
        "thesis_green_yellow_red_family",
        ["#c62828", "#f9e547", "#2e7d32"],
    )
    cmap = cmap.copy()
    cmap.set_bad(color="white")
    return cmap


def delta_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(
        "thesis_delta",
        ["#b71c1c", "#ffffff", "#1b5e20"],
    )
    cmap = cmap.copy()
    cmap.set_bad(color="white")
    return cmap


def format_value(value: float | int | str | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    return f"{format_value(mean, digits)} ± {format_value(std, digits)}"


def pretty_site(value: str) -> str:
    key = str(value).strip().lower()
    return SITE_LABELS.get(key, value)


def pretty_model(value: str) -> str:
    return MODEL_LABELS.get(str(value), str(value))


def pretty_method(value: str) -> str:
    return METHOD_LABELS.get(str(value), str(value).replace("_", " ").title())


def markdown_table(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, classes="report-table")


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=10)


def annotate_heatmap(ax: plt.Axes, data: np.ndarray, digits: int = 3) -> None:
    finite_values = data[np.isfinite(data)]
    threshold = np.nanmedian(finite_values) if finite_values.size else 0.5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isnan(value):
                text = "NA"
                color = "#333333"
            else:
                text = f"{value:.{digits}f}"
                if np.isclose(value, 0.0):
                    color = "#333333"
                else:
                    color = "white" if value >= threshold else "#111111"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)


def plot_positive_heatmap(df: pd.DataFrame, title: str, output_path: Path) -> None:
    data = df.astype(float).to_numpy()
    masked = np.ma.masked_where(np.isclose(data, 0.0), data)
    fig_width = max(6.0, 1.2 * len(df.columns) + 2.8)
    fig_height = max(3.0, 0.6 * len(df.index) + 2.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(masked, aspect="auto", cmap=positive_cmap(), vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(
        [METRIC_LABELS.get(col, col) for col in df.columns], rotation=25, ha="right"
    )
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    style_axis(ax, title)
    annotate_heatmap(ax, data)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.set_ylabel("Metric value", rotation=90, va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_delta_heatmap(df: pd.DataFrame, title: str, output_path: Path) -> None:
    data = df.astype(float).to_numpy()
    masked = np.ma.masked_where(np.isclose(data, 0.0), data)
    vmax = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 1.0
    vmax = max(vmax, 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig_width = max(6.0, 1.2 * len(df.columns) + 2.8)
    fig_height = max(2.8, 0.6 * len(df.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(masked, aspect="auto", cmap=delta_cmap(), norm=norm)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(
        [METRIC_LABELS.get(col, col) for col in df.columns], rotation=25, ha="right"
    )
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    style_axis(ax, title)
    annotate_heatmap(ax, data)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.set_ylabel("Delta", rotation=90, va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False)


def write_text(text: str, output_path: Path) -> None:
    output_path.write_text(text, encoding="utf-8")


def copy_visible_files(source_dir: Path, destination_dir: Path) -> None:
    ensure_dir(destination_dir)
    for item in source_dir.iterdir():
        if item.name.startswith(".") or item.name == ".idea":
            continue
        if item.is_file():
            shutil.copy2(item, destination_dir / item.name)


def copy_database_assets(
    raw_dir: Path, preprocessed_csv: Path, database_dir: Path
) -> None:
    raw_output_dir = ensure_dir(database_dir / "raw_data")
    preprocessed_output_dir = ensure_dir(database_dir / "preprocessed")
    copy_visible_files(raw_dir, raw_output_dir)
    shutil.copy2(preprocessed_csv, preprocessed_output_dir / preprocessed_csv.name)
    preprocessing_report = preprocessed_csv.with_name("uci_preprocessing_report.html")
    if preprocessing_report.exists():
        shutil.copy2(
            preprocessing_report, preprocessed_output_dir / preprocessing_report.name
        )


def write_section_readme(
    section_dir: Path, title: str, description: str, extra_lines: list[str]
) -> None:
    lines = [f"# {title}", "", description, "", "## Contents"]
    lines.extend([f"- {line}" for line in extra_lines])
    write_text("\n".join(lines) + "\n", section_dir / "README.md")


def compute_classification_metrics(
    y_true: Iterable[int], y_pred: Iterable[int], y_prob: Iterable[float]
) -> dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    accuracy = float((tp + tn) / max(len(y_true_arr), 1))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )
    if len(np.unique(y_true_arr)) >= 2:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    else:
        roc_auc = np.nan
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "balanced_accuracy": balanced_accuracy,
    }


def summarize_mean_std(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        record: dict[str, object] = {col: row[col] for col in id_cols}
        for metric in METRIC_ORDER:
            record[f"{metric}_mean"] = row[f"{metric}_mean"]
            record[f"{metric}_std"] = row[f"{metric}_std"]
        records.append(record)
    return pd.DataFrame(records)


def add_presentation_columns(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for _, row in df.iterrows():
        record: dict[str, str] = {col: str(row[col]) for col in id_cols}
        for metric in METRIC_ORDER:
            record[METRIC_LABELS[metric]] = format_mean_std(
                row[f"{metric}_mean"], row[f"{metric}_std"]
            )
        records.append(record)
    return pd.DataFrame(records)


def mean_metric_view(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    columns = [f"{metric}_mean" for metric in METRIC_ORDER]
    renamed = {f"{metric}_mean": metric for metric in METRIC_ORDER}
    return df.set_index(index_col)[columns].rename(columns=renamed)


def build_baseline_tables(
    base_dir: Path, tables_dir: Path, plots_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_summary = pd.read_csv(base_dir / "model_summary.csv")
    baseline_summary = pd.DataFrame(
        {
            "Model": model_summary["model"].map(pretty_model),
            "accuracy_mean": model_summary["accuracy_ood_mean"],
            "accuracy_std": model_summary["accuracy_ood_std"],
            "f1_mean": model_summary["f1_ood_mean"],
            "f1_std": model_summary["f1_ood_std"],
            "precision_mean": model_summary["precision_ood_mean"],
            "precision_std": model_summary["precision_ood_std"],
            "recall_mean": model_summary["recall_ood_mean"],
            "recall_std": model_summary["recall_ood_std"],
            "roc_auc_mean": model_summary["auc_ood_mean"],
            "roc_auc_std": model_summary["auc_ood_std"],
            "balanced_accuracy_mean": model_summary["balacc_ood_mean"],
            "balanced_accuracy_std": model_summary["balacc_ood_std"],
        }
    ).sort_values(
        ["balanced_accuracy_mean", "roc_auc_mean"], ascending=False, ignore_index=True
    )

    prediction_records = pd.read_csv(base_dir / "prediction_records.csv")
    prediction_records = prediction_records[
        (prediction_records["split"] == "ood")
        & (
            prediction_records["outer_test_domain"]
            == prediction_records["evaluation_domain"]
        )
    ].copy()
    pair_rows = []
    for (holdout_domain, model), subset in prediction_records.groupby(
        ["outer_test_domain", "model"], sort=False
    ):
        metrics = compute_classification_metrics(
            subset["y_true"], subset["y_pred"], subset["y_prob"]
        )
        pair_rows.append(
            {
                "holdout_domain": pretty_site(holdout_domain),
                "model": pretty_model(model),
                **metrics,
            }
        )
    baseline_pairs = pd.DataFrame(pair_rows).sort_values(
        ["holdout_domain", "model"], ignore_index=True
    )

    baseline_summary_csv = baseline_summary.copy()
    write_csv(baseline_summary_csv, tables_dir / "baseline_m0_summary.csv")
    write_csv(baseline_pairs, tables_dir / "baseline_m0_holdout_domain_results.csv")

    presentation = add_presentation_columns(baseline_summary, ["Model"])
    write_csv(presentation, tables_dir / "baseline_m0_summary_presentation.csv")

    heatmap_df = mean_metric_view(baseline_summary, "Model")
    plot_positive_heatmap(
        heatmap_df,
        "Baseline M0: LODO summary metrics",
        plots_dir / "baseline_m0_summary_heatmap.png",
    )
    return baseline_summary, baseline_pairs, presentation


def build_dashi_tables(
    dashi_dir: Path, tables_dir: Path, plots_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_df = pd.read_csv(
        dashi_dir / "tables" / "supervised_multibatch_metrics_long.tsv", sep="\t"
    )
    mapping = {
        "ACCURACY": "accuracy",
        "F1-SCORE_1": "f1",
        "PRECISION_1": "precision",
        "RECALL_1": "recall",
        "AUC_1": "roc_auc",
        "RECALL_MACRO": "balanced_accuracy",
    }
    metric_df = long_df[long_df["metric"].isin(mapping)].copy()
    metric_df["metric_name"] = metric_df["metric"].map(mapping)
    pairwise = (
        metric_df.pivot_table(
            index=["train_batches", "test_batch"],
            columns="metric_name",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"train_batches": "train_domain", "test_batch": "test_domain"})
    )
    pairwise["roc_auc_source"] = np.where(
        pairwise["roc_auc"].isna(), "balanced_accuracy_surrogate", "reported"
    )
    pairwise["roc_auc"] = pairwise["roc_auc"].fillna(pairwise["balanced_accuracy"])
    pairwise["train_domain"] = pairwise["train_domain"].map(pretty_site)
    pairwise["test_domain"] = pairwise["test_domain"].map(pretty_site)
    pairwise["pair_type"] = np.where(
        pairwise["train_domain"] == pairwise["test_domain"],
        "within-domain",
        "cross-domain",
    )
    pairwise = pairwise[
        ["train_domain", "test_domain", "pair_type", *METRIC_ORDER, "roc_auc_source"]
    ].sort_values(["train_domain", "test_domain"], ignore_index=True)

    unsupervised = pd.read_csv(dashi_dir / "tables" / "msv_metrics_by_source.csv")
    gpd = pd.read_csv(dashi_dir / "tables" / "msv_gpd.tsv", sep="\t")
    dashi_unsupervised = pd.DataFrame(
        {
            "Source domain": unsupervised["source"].map(pretty_site),
            "SPO": unsupervised["SPO"],
            "nBySource": unsupervised["nBySource"],
            "GPD": unsupervised["GPD"],
            "Anchor feature": gpd["anchor_feature"].iloc[0],
        }
    )

    summary_rows = []
    for pair_type, subset in pairwise.groupby("pair_type", sort=False):
        row: dict[str, object] = {"Subset": pair_type.replace("-", " ").title()}
        for metric in METRIC_ORDER:
            row[f"{metric}_mean"] = subset[metric].mean()
            row[f"{metric}_std"] = subset[metric].std(ddof=0)
        summary_rows.append(row)
    dashi_summary = pd.DataFrame(summary_rows)

    write_csv(pairwise, tables_dir / "dashi_supervised_pairwise_results.csv")
    write_csv(dashi_summary, tables_dir / "dashi_supervised_summary.csv")
    write_csv(dashi_unsupervised, tables_dir / "dashi_unsupervised_summary.csv")

    summary_presentation = add_presentation_columns(dashi_summary, ["Subset"])
    write_csv(
        summary_presentation, tables_dir / "dashi_supervised_summary_presentation.csv"
    )

    for metric in METRIC_ORDER:
        matrix = pairwise.pivot(
            index="train_domain", columns="test_domain", values=metric
        )
        matrix = matrix.reindex(
            index=[pretty_site(site) for site in SITE_ORDER],
            columns=[pretty_site(site) for site in SITE_ORDER],
        )
        plot_positive_heatmap(
            matrix,
            f"DASHI supervised characterization: {METRIC_LABELS[metric]}",
            plots_dir / f"dashi_{metric}_matrix_heatmap.png",
        )
        matrix_out = matrix.reset_index().rename(
            columns={"train_domain": "Train domain"}
        )
        write_csv(matrix_out, tables_dir / f"dashi_{metric}_matrix.csv")

    return dashi_summary, dashi_unsupervised, summary_presentation


def build_transductive_tables(
    trans_dir: Path, tables_dir: Path, plots_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(trans_dir / "tables" / "summary_metrics.csv")
    trans_summary = pd.DataFrame(
        {
            "Configuration": summary.apply(
                lambda row: (
                    f"{pretty_model(row['model_name'])} | {pretty_method(row['variant'])}"
                ),
                axis=1,
            ),
            "accuracy_mean": summary["accuracy_mean"],
            "accuracy_std": summary["accuracy_std"],
            "f1_mean": summary["f1_mean"],
            "f1_std": summary["f1_std"],
            "precision_mean": summary["precision_mean"],
            "precision_std": summary["precision_std"],
            "recall_mean": summary["recall_mean"],
            "recall_std": summary["recall_std"],
            "roc_auc_mean": summary["roc_auc_mean"],
            "roc_auc_std": summary["roc_auc_std"],
            "balanced_accuracy_mean": summary["balanced_accuracy_mean"],
            "balanced_accuracy_std": summary["balanced_accuracy_std"],
        }
    ).sort_values(
        ["balanced_accuracy_mean", "roc_auc_mean"], ascending=False, ignore_index=True
    )

    fold_metrics = pd.read_csv(trans_dir / "tables" / "fold_metrics.csv")
    fold_metrics["Configuration"] = fold_metrics.apply(
        lambda row: (
            f"{pretty_model(row['model_name'])} | {pretty_method(row['variant'])}"
        ),
        axis=1,
    )
    fold_metrics["target_site"] = fold_metrics["target_site"].map(pretty_site)
    fold_results = fold_metrics.rename(columns={"target_site": "holdout_domain"})[
        ["holdout_domain", "Configuration", *METRIC_ORDER]
    ].sort_values(["holdout_domain", "Configuration"], ignore_index=True)

    variant_comparison = pd.read_csv(trans_dir / "tables" / "variant_comparison.csv")
    delta_rows = []
    for model_name, subset in variant_comparison.groupby("model_name", sort=False):
        row = {"Model": pretty_model(model_name)}
        for metric in METRIC_ORDER:
            row[metric] = subset[f"delta_{metric}"].mean()
        delta_rows.append(row)
    delta_df = pd.DataFrame(delta_rows).sort_values("Model").reset_index(drop=True)

    write_csv(trans_summary, tables_dir / "transductive_learning_summary.csv")
    write_csv(
        add_presentation_columns(trans_summary, ["Configuration"]),
        tables_dir / "transductive_learning_summary_presentation.csv",
    )
    write_csv(
        fold_results, tables_dir / "transductive_learning_holdout_domain_results.csv"
    )
    write_csv(delta_df, tables_dir / "transductive_learning_pseudolabel_delta.csv")

    plot_positive_heatmap(
        mean_metric_view(trans_summary, "Configuration"),
        "Transductive learning: summary metrics",
        plots_dir / "transductive_learning_summary_heatmap.png",
    )
    plot_delta_heatmap(
        delta_df.set_index("Model")[METRIC_ORDER],
        "Transductive learning: pseudo-labeling delta vs CORAL only",
        plots_dir / "transductive_learning_pseudolabel_delta_heatmap.png",
    )
    return trans_summary, fold_results, delta_df


def build_weighting_tables(
    weight_dir: Path, tables_dir: Path, plots_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(weight_dir / "weighting_summary.csv")
    summary_pivot = summary.pivot(
        index="method", columns="metric", values=["mean", "std"]
    )
    rows = []
    for method in summary_pivot.index:
        row: dict[str, object] = {"Method": pretty_method(method)}
        for metric in METRIC_ORDER:
            row[f"{metric}_mean"] = summary_pivot.loc[method, ("mean", metric)]
            row[f"{metric}_std"] = summary_pivot.loc[method, ("std", metric)]
        rows.append(row)
    weighting_summary = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy_mean", "roc_auc_mean"], ascending=False, ignore_index=True
    )

    lodo = pd.read_csv(weight_dir / "lodo_results.csv")
    weighting_holdout = lodo.copy()
    weighting_holdout["holdout_domain"] = weighting_holdout["holdout_domain"].map(
        pretty_site
    )
    weighting_holdout["Method"] = weighting_holdout["method"].map(pretty_method)
    weighting_holdout = weighting_holdout[
        ["holdout_domain", "Method", *METRIC_ORDER]
    ].sort_values(["holdout_domain", "Method"], ignore_index=True)

    delta_rows = []
    raw_reference = weighting_summary.loc[weighting_summary["Method"] == "Raw"].iloc[0]
    for _, row in weighting_summary.iterrows():
        if row["Method"] == "Raw":
            continue
        delta = {"Method": row["Method"]}
        for metric in METRIC_ORDER:
            delta[metric] = row[f"{metric}_mean"] - raw_reference[f"{metric}_mean"]
        delta_rows.append(delta)
    weighting_delta = pd.DataFrame(delta_rows)

    write_csv(weighting_summary, tables_dir / "weighting_summary.csv")
    write_csv(
        add_presentation_columns(weighting_summary, ["Method"]),
        tables_dir / "weighting_summary_presentation.csv",
    )
    write_csv(weighting_holdout, tables_dir / "weighting_holdout_domain_results.csv")
    if not weighting_delta.empty:
        write_csv(weighting_delta, tables_dir / "weighting_delta_vs_raw.csv")

    plot_positive_heatmap(
        mean_metric_view(weighting_summary, "Method"),
        "Weighting: summary metrics",
        plots_dir / "weighting_summary_heatmap.png",
    )
    if not weighting_delta.empty:
        plot_delta_heatmap(
            weighting_delta.set_index("Method")[METRIC_ORDER],
            "Weighting: delta vs raw reference",
            plots_dir / "weighting_delta_vs_raw_heatmap.png",
        )
    return weighting_summary, weighting_holdout, weighting_delta


def build_combat_tables(
    combat_dir: Path, tables_dir: Path, plots_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(combat_dir / "predictions.csv")
    rows = []
    for (pipeline, holdout_domain), subset in predictions.groupby(
        ["pipeline", "holdout_domain"], sort=False
    ):
        metrics = compute_classification_metrics(
            subset["y_true"], subset["y_pred"], subset["y_prob"]
        )
        row = {
            "Pipeline": pretty_method(pipeline),
            "holdout_domain": pretty_site(holdout_domain),
            **metrics,
        }
        rows.append(row)
    combat_holdout = pd.DataFrame(rows).sort_values(
        ["holdout_domain", "Pipeline"], ignore_index=True
    )

    summary_rows = []
    for pipeline, subset in combat_holdout.groupby("Pipeline", sort=False):
        row: dict[str, object] = {"Pipeline": pipeline}
        for metric in METRIC_ORDER:
            row[f"{metric}_mean"] = subset[metric].mean()
            row[f"{metric}_std"] = subset[metric].std(ddof=0)
        summary_rows.append(row)
    combat_summary = pd.DataFrame(summary_rows).sort_values(
        ["balanced_accuracy_mean", "roc_auc_mean"], ascending=False, ignore_index=True
    )

    baseline_row = combat_summary.loc[
        combat_summary["Pipeline"] == "Baseline raw"
    ].iloc[0]
    combat_rows = []
    for _, row in combat_summary.iterrows():
        if row["Pipeline"] == "Baseline raw":
            continue
        delta = {"Pipeline": row["Pipeline"]}
        for metric in METRIC_ORDER:
            delta[metric] = row[f"{metric}_mean"] - baseline_row[f"{metric}_mean"]
        combat_rows.append(delta)
    combat_delta = pd.DataFrame(combat_rows)

    write_csv(combat_summary, tables_dir / "combat_summary.csv")
    write_csv(
        add_presentation_columns(combat_summary, ["Pipeline"]),
        tables_dir / "combat_summary_presentation.csv",
    )
    write_csv(combat_holdout, tables_dir / "combat_holdout_domain_results.csv")
    if not combat_delta.empty:
        write_csv(combat_delta, tables_dir / "combat_delta_vs_baseline.csv")

    plot_positive_heatmap(
        mean_metric_view(combat_summary, "Pipeline"),
        "ComBat: summary metrics",
        plots_dir / "combat_summary_heatmap.png",
    )
    if not combat_delta.empty:
        plot_delta_heatmap(
            combat_delta.set_index("Pipeline")[METRIC_ORDER],
            "ComBat: delta vs baseline raw",
            plots_dir / "combat_delta_vs_baseline_heatmap.png",
        )
    return combat_summary, combat_holdout, combat_delta


def build_cross_method_tables(
    baseline_summary: pd.DataFrame,
    dashi_summary: pd.DataFrame,
    trans_summary: pd.DataFrame,
    weighting_summary: pd.DataFrame,
    combat_summary: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_best = baseline_summary.sort_values(
        ["balanced_accuracy_mean", "roc_auc_mean"], ascending=False
    ).iloc[0]
    trans_best = trans_summary.sort_values(
        ["balanced_accuracy_mean", "roc_auc_mean"], ascending=False
    ).iloc[0]
    weighting_best = weighting_summary.sort_values(
        ["balanced_accuracy_mean", "roc_auc_mean"], ascending=False
    ).iloc[0]
    combat_adapted = combat_summary.loc[
        combat_summary["Pipeline"] == "ComBat train-only"
    ].iloc[0]

    lodo_rows = [
        {
            "Method family": "Baseline M0",
            **{
                f"{metric}_mean": baseline_best[f"{metric}_mean"]
                for metric in METRIC_ORDER
            },
        },
        {
            "Method family": "Transductive learning",
            **{
                f"{metric}_mean": trans_best[f"{metric}_mean"]
                for metric in METRIC_ORDER
            },
        },
        {
            "Method family": "Weighting",
            **{
                f"{metric}_mean": weighting_best[f"{metric}_mean"]
                for metric in METRIC_ORDER
            },
        },
        {
            "Method family": "ComBat",
            **{
                f"{metric}_mean": combat_adapted[f"{metric}_mean"]
                for metric in METRIC_ORDER
            },
        },
    ]
    lodo_comparison = pd.DataFrame(lodo_rows)
    lodo_comparison_plot = lodo_comparison.rename(
        columns={f"{metric}_mean": metric for metric in METRIC_ORDER}
    )

    dashi_cross = dashi_summary.loc[dashi_summary["Subset"] == "Cross Domain"].copy()
    dashi_cross.insert(0, "Method family", "DASHI supervised characterization")
    dashi_cross = dashi_cross[
        [
            "Method family",
            *[f"{metric}_mean" for metric in METRIC_ORDER],
            *[f"{metric}_std" for metric in METRIC_ORDER],
        ]
    ]

    write_csv(lodo_comparison, tables_dir / "cross_method_lodo_comparison.csv")
    write_csv(dashi_cross, tables_dir / "cross_method_dashi_characterization.csv")

    plot_positive_heatmap(
        lodo_comparison_plot.set_index("Method family")[METRIC_ORDER],
        "Cross-method overview: best LODO configuration per family",
        plots_dir / "cross_method_lodo_comparison_heatmap.png",
    )
    return lodo_comparison, dashi_cross


def build_dataset_tables(
    raw_dir: Path,
    preprocessed_csv: Path,
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(preprocessed_csv)
    target_series = df["target"]
    if not pd.api.types.is_numeric_dtype(target_series):
        target_numeric = (
            target_series.astype(str)
            .str.lower()
            .map({"negative": 0, "positive": 1, "0": 0, "1": 1})
        )
    else:
        target_numeric = pd.to_numeric(target_series, errors="coerce")
    feature_columns = [
        col
        for col in df.columns
        if col not in {"subject_id", "target", "domain_id", "target_original"}
    ]
    dataset_summary = pd.DataFrame(
        {
            "Field": [
                "Raw data directory",
                "Preprocessed file",
                "Number of samples",
                "Number of columns",
                "Predictive features retained",
                "Domain column",
                "Target column",
                "Number of domains",
            ],
            "Value": [
                str(raw_dir),
                str(preprocessed_csv),
                len(df),
                len(df.columns),
                len(feature_columns),
                "domain_id",
                "target",
                df["domain_id"].nunique(),
            ],
        }
    )
    site_distribution = (
        df.groupby("domain_id")
        .agg(n_samples=("subject_id", "count"))
        .reset_index()
        .rename(columns={"domain_id": "Domain"})
    )
    site_positive_rate = (
        pd.DataFrame({"domain_id": df["domain_id"], "target_numeric": target_numeric})
        .groupby("domain_id", as_index=False)
        .agg(positive_rate=("target_numeric", "mean"))
        .rename(columns={"domain_id": "Domain"})
    )
    site_distribution = site_distribution.merge(
        site_positive_rate, on="Domain", how="left"
    )
    site_distribution["Domain"] = site_distribution["Domain"].map(pretty_site)
    site_distribution["positive_rate"] = site_distribution["positive_rate"].map(
        lambda value: format_value(value, 3)
    )

    feature_overview = pd.DataFrame(
        {
            "Category": [
                "Numeric features",
                "Categorical features",
                "Excluded ID-like column",
                "Excluded auxiliary column",
            ],
            "Count / value": ["5", "8", "subject_id", "target_original"],
        }
    )

    write_csv(dataset_summary, tables_dir / "dataset_summary.csv")
    write_csv(site_distribution, tables_dir / "dataset_site_distribution.csv")
    write_csv(feature_overview, tables_dir / "dataset_feature_overview.csv")
    return dataset_summary, site_distribution, feature_overview


def build_markdown_report(
    output_dir: Path,
    dataset_summary: pd.DataFrame,
    site_distribution: pd.DataFrame,
    feature_overview: pd.DataFrame,
    baseline_presentation: pd.DataFrame,
    dashi_summary_presentation: pd.DataFrame,
    dashi_unsupervised: pd.DataFrame,
    trans_summary: pd.DataFrame,
    weighting_summary: pd.DataFrame,
    combat_summary: pd.DataFrame,
    lodo_comparison: pd.DataFrame,
) -> str:
    trans_present = add_presentation_columns(trans_summary, ["Configuration"])
    weighting_present = add_presentation_columns(weighting_summary, ["Method"])
    combat_present = add_presentation_columns(combat_summary, ["Pipeline"])

    lodo_report = lodo_comparison.copy()
    for metric in METRIC_ORDER:
        lodo_report[METRIC_LABELS[metric]] = lodo_report[f"{metric}_mean"].map(
            lambda value: format_value(value, 3)
        )
    lodo_report = lodo_report[["Method family", *METRIC_LABELS.values()]]

    dashi_unsupervised_report = dashi_unsupervised.copy()
    for column in ["SPO", "GPD"]:
        dashi_unsupervised_report[column] = dashi_unsupervised_report[column].map(
            lambda value: format_value(value, 3)
        )

    report = f"""# UCI Heart Disease: Harmonized Results Package

This folder consolidates the UCI Heart Disease results into a single thesis-ready package with consistent English wording, complete core metrics, and a uniform green-yellow-red palette. Exact `0.00` values are rendered in white in all heatmaps.

## Folder organization

- `database/`: raw files, preprocessed dataset, and dataset-level summary tables.
- `baseline/`: harmonized Baseline M0 tables and plots.
- `dashi/`: harmonized DASHI unsupervised and supervised outputs.
- `transductive_learning/`: harmonized transductive learning tables and plots.
- `weighting/`: harmonized weighting tables and plots.
- `combat/`: harmonized ComBat tables and plots.
- `overview/`: cross-method comparison tables and plots.

## 1. Dataset and preprocessing

{markdown_table(dataset_summary)}

### Site distribution

{markdown_table(site_distribution)}

### Feature overview

{markdown_table(feature_overview)}

## 2. Baseline M0

Evaluation protocol: strict leave-one-domain-out validation across the four UCI sites.

{markdown_table(baseline_presentation)}

![Baseline summary heatmap](baseline/plots/baseline_m0_summary_heatmap.png)

## 3. DASHI

Methodological note: DASHI supervised outputs are retained as cross-domain characterization matrices, not as the same LODO benchmark used for the downstream adaptation methods.

### Unsupervised summary

{markdown_table(dashi_unsupervised_report)}

### Supervised summary

Balanced Accuracy was derived from macro recall because the outcome is binary. The original DASHI export omitted ROC AUC values for all `Switzerland` target-batch evaluations; these cells were completed with the hard-label ROC AUC surrogate, which equals balanced accuracy in the binary setting.

{markdown_table(dashi_summary_presentation)}

![DASHI Accuracy](dashi/plots/dashi_accuracy_matrix_heatmap.png)
![DASHI F1-score](dashi/plots/dashi_f1_matrix_heatmap.png)
![DASHI Precision](dashi/plots/dashi_precision_matrix_heatmap.png)
![DASHI Recall](dashi/plots/dashi_recall_matrix_heatmap.png)
![DASHI ROC AUC](dashi/plots/dashi_roc_auc_matrix_heatmap.png)
![DASHI Balanced Accuracy](dashi/plots/dashi_balanced_accuracy_matrix_heatmap.png)

## 4. Transductive learning

{markdown_table(trans_present)}

![Transductive learning summary](transductive_learning/plots/transductive_learning_summary_heatmap.png)
![Transductive learning pseudo-label delta](transductive_learning/plots/transductive_learning_pseudolabel_delta_heatmap.png)

## 5. Weighting

{markdown_table(weighting_present)}

![Weighting summary](weighting/plots/weighting_summary_heatmap.png)
![Weighting delta vs raw](weighting/plots/weighting_delta_vs_raw_heatmap.png)

## 6. ComBat

Precision and recall were reconstructed from the saved fold-level predictions so that the ComBat section uses the same complete metric set as the other methods.

{markdown_table(combat_present)}

![ComBat summary](combat/plots/combat_summary_heatmap.png)
![ComBat delta vs baseline](combat/plots/combat_delta_vs_baseline_heatmap.png)

## 7. Cross-method overview

The comparison below includes the best LODO configuration for each downstream family. DASHI is kept separate because its supervised results are source-to-target characterization matrices rather than the same LODO benchmarking protocol.

{markdown_table(lodo_report)}

![Cross-method LODO comparison](overview/plots/cross_method_lodo_comparison_heatmap.png)

## Notes

- All detailed machine-readable tables are stored in the `data/` subfolder of each section.
- All plots were regenerated with a single visual standard and a shared metric order: Accuracy, F1-score, Precision, Recall, ROC AUC, and Balanced Accuracy.
- Only material directly relevant to the thesis results section was retained in this final package.
"""
    write_text(report, output_dir / "README.md")
    return report


def build_html_report(
    output_dir: Path,
    dataset_summary: pd.DataFrame,
    site_distribution: pd.DataFrame,
    feature_overview: pd.DataFrame,
    baseline_presentation: pd.DataFrame,
    dashi_summary_presentation: pd.DataFrame,
    dashi_unsupervised: pd.DataFrame,
    trans_summary: pd.DataFrame,
    weighting_summary: pd.DataFrame,
    combat_summary: pd.DataFrame,
    lodo_comparison: pd.DataFrame,
) -> None:
    trans_present = add_presentation_columns(trans_summary, ["Configuration"])
    weighting_present = add_presentation_columns(weighting_summary, ["Method"])
    combat_present = add_presentation_columns(combat_summary, ["Pipeline"])

    lodo_report = lodo_comparison.copy()
    for metric in METRIC_ORDER:
        lodo_report[METRIC_LABELS[metric]] = lodo_report[f"{metric}_mean"].map(
            lambda value: format_value(value, 3)
        )
    lodo_report = lodo_report[["Method family", *METRIC_LABELS.values()]]

    dashi_unsupervised_report = dashi_unsupervised.copy()
    for column in ["SPO", "GPD"]:
        dashi_unsupervised_report[column] = dashi_unsupervised_report[column].map(
            lambda value: format_value(value, 3)
        )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>UCI Heart Disease Harmonized Results</title>
  <style>
    body {{
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      margin: 32px auto;
      max-width: 1120px;
      line-height: 1.55;
      color: #1f2933;
      background: #ffffff;
      padding: 0 20px 40px 20px;
    }}
    h1, h2, h3 {{
      color: #7f1d1d;
    }}
    h1 {{
      margin-bottom: 10px;
    }}
    .lead {{
      margin-bottom: 26px;
    }}
    .report-table {{
      border-collapse: collapse;
      margin: 14px 0 24px 0;
      width: 100%;
      font-size: 0.95rem;
    }}
    .report-table th, .report-table td {{
      border: 1px solid #d5dbe0;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .report-table th {{
      background: #f3f4f6;
    }}
    img {{
      width: 100%;
      max-width: 980px;
      display: block;
      margin: 12px 0 22px 0;
      border: 1px solid #e5e7eb;
    }}
    .note {{
      background: #f9fafb;
      border-left: 4px solid #b91c1c;
      padding: 12px 14px;
      margin: 14px 0 22px 0;
    }}
    ul {{
      padding-left: 18px;
    }}
  </style>
</head>
<body>
  <h1>UCI Heart Disease: Harmonized Results Package</h1>
  <p class="lead">This report consolidates the UCI Heart Disease results into a thesis-ready package with consistent English wording, complete core metrics, and a uniform green-yellow-red palette. Exact <code>0.00</code> values are rendered in white in all heatmaps.</p>

  <h2>Folder organization</h2>
  <ul>
    <li><code>database/</code>: raw files, preprocessed dataset, and dataset-level summary tables.</li>
    <li><code>baseline/</code>: harmonized Baseline M0 tables and plots.</li>
    <li><code>dashi/</code>: harmonized DASHI unsupervised and supervised outputs.</li>
    <li><code>transductive_learning/</code>: harmonized transductive learning tables and plots.</li>
    <li><code>weighting/</code>: harmonized weighting tables and plots.</li>
    <li><code>combat/</code>: harmonized ComBat tables and plots.</li>
    <li><code>overview/</code>: cross-method comparison tables and plots.</li>
  </ul>

  <h2>1. Dataset and preprocessing</h2>
  {html_table(dataset_summary)}
  <h3>Site distribution</h3>
  {html_table(site_distribution)}
  <h3>Feature overview</h3>
  {html_table(feature_overview)}

  <h2>2. Baseline M0</h2>
  <p>Evaluation protocol: strict leave-one-domain-out validation across the four UCI sites.</p>
  {html_table(baseline_presentation)}
  <img src="baseline/plots/baseline_m0_summary_heatmap.png" alt="Baseline M0 heatmap" />

  <h2>3. DASHI</h2>
  <div class="note">DASHI supervised outputs are retained as cross-domain characterization matrices, not as the same LODO benchmark used for the downstream adaptation methods.</div>
  <h3>Unsupervised summary</h3>
  {html_table(dashi_unsupervised_report)}
  <h3>Supervised summary</h3>
  <div class="note">Balanced Accuracy was derived from macro recall because the outcome is binary. The original DASHI export omitted ROC AUC values for all Switzerland target-batch evaluations; these cells were completed with the hard-label ROC AUC surrogate, which equals balanced accuracy in the binary setting.</div>
  {html_table(dashi_summary_presentation)}
  <img src="dashi/plots/dashi_accuracy_matrix_heatmap.png" alt="DASHI Accuracy heatmap" />
  <img src="dashi/plots/dashi_f1_matrix_heatmap.png" alt="DASHI F1 heatmap" />
  <img src="dashi/plots/dashi_precision_matrix_heatmap.png" alt="DASHI Precision heatmap" />
  <img src="dashi/plots/dashi_recall_matrix_heatmap.png" alt="DASHI Recall heatmap" />
  <img src="dashi/plots/dashi_roc_auc_matrix_heatmap.png" alt="DASHI ROC AUC heatmap" />
  <img src="dashi/plots/dashi_balanced_accuracy_matrix_heatmap.png" alt="DASHI Balanced Accuracy heatmap" />

  <h2>4. Transductive learning</h2>
  {html_table(trans_present)}
  <img src="transductive_learning/plots/transductive_learning_summary_heatmap.png" alt="Transductive learning heatmap" />
  <img src="transductive_learning/plots/transductive_learning_pseudolabel_delta_heatmap.png" alt="Transductive learning delta heatmap" />

  <h2>5. Weighting</h2>
  {html_table(weighting_present)}
  <img src="weighting/plots/weighting_summary_heatmap.png" alt="Weighting heatmap" />
  <img src="weighting/plots/weighting_delta_vs_raw_heatmap.png" alt="Weighting delta heatmap" />

  <h2>6. ComBat</h2>
  <p>Precision and recall were reconstructed from the saved fold-level predictions so that the ComBat section uses the same complete metric set as the other methods.</p>
  {html_table(combat_present)}
  <img src="combat/plots/combat_summary_heatmap.png" alt="ComBat heatmap" />
  <img src="combat/plots/combat_delta_vs_baseline_heatmap.png" alt="ComBat delta heatmap" />

  <h2>7. Cross-method overview</h2>
  <p>The comparison below includes the best LODO configuration for each downstream family. DASHI is kept separate because its supervised results are source-to-target characterization matrices rather than the same LODO benchmarking protocol.</p>
  {html_table(lodo_report)}
  <img src="overview/plots/cross_method_lodo_comparison_heatmap.png" alt="Cross-method comparison heatmap" />

  <h2>Notes</h2>
  <ul>
    <li>All detailed machine-readable tables are stored in the <code>data/</code> subfolder of each section.</li>
    <li>All plots were regenerated with a single visual standard and a shared metric order: Accuracy, F1-score, Precision, Recall, ROC AUC, and Balanced Accuracy.</li>
    <li>Only material directly relevant to the thesis results section was retained in this final package.</li>
  </ul>
</body>
</html>
"""
    write_text(html_content, output_dir / "report.html")


def build_manifest(output_dir: Path, paths: dict[str, str]) -> None:
    manifest = {
        "dataset": "UCI Heart Disease",
        "generated_files": sorted(
            [
                str(path.relative_to(output_dir))
                for path in output_dir.rglob("*")
                if path.is_file()
            ]
        ),
        "source_paths": paths,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a harmonized UCI final results package."
    )
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--preprocessed-csv", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--dashi-dir", required=True, type=Path)
    parser.add_argument("--transductive-dir", required=True, type=Path)
    parser.add_argument("--weight-dir", required=True, type=Path)
    parser.add_argument("--combat-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    database_dir = ensure_dir(output_dir / "database")
    baseline_dir = ensure_dir(output_dir / "baseline")
    dashi_dir = ensure_dir(output_dir / "dashi")
    transductive_dir = ensure_dir(output_dir / "transductive_learning")
    weighting_dir = ensure_dir(output_dir / "weighting")
    combat_dir = ensure_dir(output_dir / "combat")
    overview_dir = ensure_dir(output_dir / "overview")

    database_data_dir = ensure_dir(database_dir / "data")
    baseline_data_dir = ensure_dir(baseline_dir / "data")
    baseline_plots_dir = ensure_dir(baseline_dir / "plots")
    dashi_data_dir = ensure_dir(dashi_dir / "data")
    dashi_plots_dir = ensure_dir(dashi_dir / "plots")
    transductive_data_dir = ensure_dir(transductive_dir / "data")
    transductive_plots_dir = ensure_dir(transductive_dir / "plots")
    weighting_data_dir = ensure_dir(weighting_dir / "data")
    weighting_plots_dir = ensure_dir(weighting_dir / "plots")
    combat_data_dir = ensure_dir(combat_dir / "data")
    combat_plots_dir = ensure_dir(combat_dir / "plots")
    overview_data_dir = ensure_dir(overview_dir / "data")
    overview_plots_dir = ensure_dir(overview_dir / "plots")

    copy_database_assets(args.raw_dir, args.preprocessed_csv, database_dir)

    dataset_summary, site_distribution, feature_overview = build_dataset_tables(
        args.raw_dir,
        args.preprocessed_csv,
        database_data_dir,
    )
    baseline_summary, _, baseline_presentation = build_baseline_tables(
        args.baseline_dir, baseline_data_dir, baseline_plots_dir
    )
    dashi_summary, dashi_unsupervised, dashi_summary_presentation = build_dashi_tables(
        args.dashi_dir, dashi_data_dir, dashi_plots_dir
    )
    trans_summary, _, _ = build_transductive_tables(
        args.transductive_dir, transductive_data_dir, transductive_plots_dir
    )
    weighting_summary, _, _ = build_weighting_tables(
        args.weight_dir, weighting_data_dir, weighting_plots_dir
    )
    combat_summary, _, _ = build_combat_tables(
        args.combat_dir, combat_data_dir, combat_plots_dir
    )
    lodo_comparison, _ = build_cross_method_tables(
        baseline_summary,
        dashi_summary,
        trans_summary,
        weighting_summary,
        combat_summary,
        overview_data_dir,
        overview_plots_dir,
    )

    build_markdown_report(
        output_dir,
        dataset_summary,
        site_distribution,
        feature_overview,
        baseline_presentation,
        dashi_summary_presentation,
        dashi_unsupervised,
        trans_summary,
        weighting_summary,
        combat_summary,
        lodo_comparison,
    )
    build_html_report(
        output_dir,
        dataset_summary,
        site_distribution,
        feature_overview,
        baseline_presentation,
        dashi_summary_presentation,
        dashi_unsupervised,
        trans_summary,
        weighting_summary,
        combat_summary,
        lodo_comparison,
    )
    write_section_readme(
        database_dir,
        "Database",
        "This folder contains the raw UCI source files, the preprocessed dataset used for downstream analyses, and dataset-level summary tables.",
        [
            "`raw_data/`: original visible raw files copied from the source directory.",
            "`preprocessed/`: preprocessed CSV and preprocessing report.",
            "`data/`: harmonized dataset summary tables.",
        ],
    )
    write_section_readme(
        baseline_dir,
        "Baseline M0",
        "This folder contains the harmonized Baseline M0 outputs for the strict LODO evaluation.",
        [
            "`data/`: summary and holdout-domain tables.",
            "`plots/`: harmonized baseline heatmaps.",
        ],
    )
    write_section_readme(
        dashi_dir,
        "DASHI",
        "This folder contains the harmonized DASHI outputs retained for the thesis package.",
        [
            "`data/`: supervised matrices, pairwise tables, and unsupervised summaries.",
            "`plots/`: harmonized DASHI heatmaps.",
        ],
    )
    write_section_readme(
        transductive_dir,
        "Transductive Learning",
        "This folder contains the harmonized transductive learning results.",
        [
            "`data/`: summary, holdout-domain, and pseudo-label delta tables.",
            "`plots/`: harmonized transductive learning heatmaps.",
        ],
    )
    write_section_readme(
        weighting_dir,
        "Weighting",
        "This folder contains the harmonized weighting results.",
        [
            "`data/`: summary, holdout-domain, and delta-versus-raw tables.",
            "`plots/`: harmonized weighting heatmaps.",
        ],
    )
    write_section_readme(
        combat_dir,
        "ComBat",
        "This folder contains the harmonized ComBat results.",
        [
            "`data/`: summary, holdout-domain, and delta-versus-baseline tables.",
            "`plots/`: harmonized ComBat heatmaps.",
        ],
    )
    write_section_readme(
        overview_dir,
        "Overview",
        "This folder contains the cross-method comparison outputs used to compare the downstream method families.",
        [
            "`data/`: cross-method summary tables.",
            "`plots/`: cross-method comparison heatmap.",
        ],
    )
    build_manifest(
        output_dir,
        {
            "raw_dir": str(args.raw_dir),
            "preprocessed_csv": str(args.preprocessed_csv),
            "baseline_dir": str(args.baseline_dir),
            "dashi_dir": str(args.dashi_dir),
            "transductive_dir": str(args.transductive_dir),
            "weight_dir": str(args.weight_dir),
            "combat_dir": str(args.combat_dir),
        },
    )


if __name__ == "__main__":
    main()
