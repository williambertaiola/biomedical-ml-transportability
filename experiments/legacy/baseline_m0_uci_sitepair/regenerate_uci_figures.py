"""Transportability experiment: regenerate uci figures."""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from src.baseline_m0 import (
    ExperimentConfig,
    aggregate_warning_list,
    build_models,
    build_pipeline,
    compute_metrics,
    ensure_directory,
    fit_with_warnings,
    init_logger,
    load_dataset,
    prepare_columns,
    safe_predict_proba,
)

matplotlib.use("Agg")

OUTPUT_DIR = PROJECT_ROOT / "output" / "UCI"
CONFIG_PATH = PROJECT_ROOT / "configs" / "uci_m0_config.json"

DOMAIN_DISPLAY = {
    "cleveland": "Cleveland",
    "hungarian": "Hungarian",
    "long_beach_va": "VA",
    "switzerland": "Switzerland",
}

MODEL_DISPLAY = {
    "LogisticRegression": "logistic",
    "RandomForest": "random_forest",
    "HistGradientBoosting": "hist_gradient_boosting",
}

DOMAIN_ORDER = ["Cleveland", "Hungarian", "Switzerland", "VA"]


def pretty_domain(domain: str) -> str:
    return DOMAIN_DISPLAY.get(str(domain), str(domain))


def pretty_model(model: str) -> str:
    return MODEL_DISPLAY.get(str(model), str(model))


def performance_cmap():
    return plt.get_cmap("RdYlGn")


def gap_cmap():
    return plt.get_cmap("RdYlGn_r")


def save_figure(fig: plt.Figure, path_no_suffix: Path) -> None:
    ensure_directory(path_no_suffix.parent)
    fig.savefig(path_no_suffix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_no_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        if math.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01 * max(1.0, abs(height)),
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_grouped_bar_redgreen(
    df: pd.DataFrame,
    *,
    value_col: str,
    x_col: str,
    group_col: str,
    title: str,
    ylabel: str,
    output_name: str,
    good_high: bool = True,
) -> None:
    if df.empty or value_col not in df.columns:
        return

    pivot = df.pivot_table(
        index=x_col, columns=group_col, values=value_col, aggfunc="mean"
    )
    if pivot.empty:
        return

    if x_col == "test_domain":
        ordered_index = [site for site in DOMAIN_ORDER if site in pivot.index]
        pivot = pivot.reindex(ordered_index)

    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))
    values = pivot.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return

    cmap = performance_cmap() if good_high else gap_cmap()
    norm = colors.Normalize(
        vmin=float(finite_values.min()), vmax=float(finite_values.max())
    )

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.index) * 1.4), 6))
    for model_idx, model_name in enumerate(pivot.columns):
        series = pivot[model_name].to_numpy(dtype=float)
        color_values = cmap(norm(np.nan_to_num(series, nan=float(finite_values.min()))))
        bars = ax.bar(
            x + model_idx * width - (len(pivot.columns) - 1) * width / 2,
            series,
            width,
            label=model_name,
            color=color_values,
            edgecolor="black",
            linewidth=0.6,
        )
        annotate_bars(ax, bars)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(
        "Holdout site" if x_col == "test_domain" else x_col.replace("_", " ").title(),
        fontsize=13,
    )
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model", loc="best")
    save_figure(fig, OUTPUT_DIR / "plots" / output_name)


def heatmap_with_annotations(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    title: str,
    cmap,
    vmin: float,
    vmax: float,
    fmt: str = ".3f",
) -> any:
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = "NA" if np.isnan(value) else format(value, fmt)
            text_color = "black"
            ax.text(j, i, label, ha="center", va="center", color=text_color, fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#888888")
    return im


def plot_site_model_ood_heatmap(fold_results: pd.DataFrame) -> None:
    metrics = [
        ("auc_ood", "auc_ood"),
        ("balacc_ood", "balacc_ood"),
        ("f1_ood", "f1_ood"),
    ]
    plot_df = fold_results.copy()
    plot_df["test_domain_display"] = plot_df["test_domain"].map(pretty_domain)
    plot_df["model_display"] = plot_df["model"].map(pretty_model)

    model_order = [
        pretty_model(model)
        for model in sorted(plot_df["model"].dropna().unique().tolist())
    ]
    row_order = [
        site
        for site in reversed(DOMAIN_ORDER)
        if site in plot_df["test_domain_display"].unique()
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 8), constrained_layout=True)
    fig.suptitle("Heatmap: site x model OOD performance", fontsize=22, y=1.02)
    all_values = []
    for metric_col, _ in metrics:
        all_values.extend(plot_df[metric_col].dropna().tolist())
    vmin, vmax = float(min(all_values)), float(max(all_values))

    for ax, (metric_col, metric_title) in zip(axes, metrics):
        pivot = plot_df.pivot_table(
            index="test_domain_display",
            columns="model_display",
            values=metric_col,
            aggfunc="mean",
        ).reindex(index=row_order, columns=model_order)
        im = heatmap_with_annotations(
            ax,
            pivot.to_numpy(dtype=float),
            pivot.index.tolist(),
            pivot.columns.tolist(),
            title=metric_title,
            cmap=performance_cmap(),
            vmin=vmin,
            vmax=vmax,
        )
        if ax is axes[0]:
            ax.set_ylabel("Holdout site", fontsize=18)
    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(), shrink=0.6, location="bottom", pad=0.08
    )
    cbar.set_label("Value", fontsize=13)
    save_figure(fig, OUTPUT_DIR / "plots" / "heatmap_site_model_ood")


def run_training_domain_experiments(
    df: pd.DataFrame, config: ExperimentConfig, logger
) -> pd.DataFrame:
    models = build_models(config, logger)
    domains = sorted(df[config.domain_col].dropna().astype(str).unique().tolist())
    rows: list[dict[str, object]] = []

    for test_domain in domains:
        source_domains = [domain for domain in domains if domain != test_domain]
        test_df = df.loc[df[config.domain_col].astype(str) == test_domain].reset_index(
            drop=True
        )
        for n_train_sites in range(1, len(source_domains) + 1):
            for combo in combinations(source_domains, n_train_sites):
                train_df = df.loc[
                    df[config.domain_col].astype(str).isin(combo)
                ].reset_index(drop=True)
                train_sites_label = " + ".join(
                    pretty_domain(domain) for domain in combo
                )
                for model_name, estimator in models.items():
                    warnings_out: list[str] = []
                    row = {
                        "test_site": pretty_domain(test_domain),
                        "test_site_raw": test_domain,
                        "train_sites": train_sites_label,
                        "train_sites_raw": "|".join(combo),
                        "n_train_sites": n_train_sites,
                        "model": pretty_model(model_name),
                        "n_train_samples": int(len(train_df)),
                        "n_test_samples": int(len(test_df)),
                        "positive_rate_train": float(
                            train_df[config.target_col].mean()
                        ),
                        "positive_rate_test": float(test_df[config.target_col].mean()),
                    }

                    if train_df[config.target_col].nunique() < 2:
                        warnings_out.append("single_class_in_training")
                        metric_values = {
                            metric: math.nan
                            for metric in [
                                "accuracy",
                                "balanced_accuracy",
                                "auc",
                                "f1",
                                "brier",
                            ]
                        }
                        row.update(metric_values)
                        row["issues"] = aggregate_warning_list(warnings_out)
                        rows.append(row)
                        continue

                    spec = prepare_columns(train_df, config)
                    warnings_out.extend(spec.warnings)
                    if not spec.feature_cols:
                        warnings_out.append("no_usable_features")
                        metric_values = {
                            metric: math.nan
                            for metric in [
                                "accuracy",
                                "balanced_accuracy",
                                "auc",
                                "f1",
                                "brier",
                            ]
                        }
                        row.update(metric_values)
                        row["issues"] = aggregate_warning_list(warnings_out)
                        rows.append(row)
                        continue

                    pipeline = build_pipeline(spec, estimator, config)
                    try:
                        pipeline, fit_warnings = fit_with_warnings(
                            pipeline, train_df, train_df[config.target_col].astype(int)
                        )
                        warnings_out.extend(fit_warnings)
                        y_prob = safe_predict_proba(pipeline, test_df)
                        metric_values, metric_warnings = compute_metrics(
                            test_df[config.target_col].to_numpy(dtype=int),
                            y_prob,
                            config=config,
                        )
                        warnings_out.extend(metric_warnings)
                    except Exception as exc:
                        warnings_out.append(f"fit_or_predict_failed:{exc}")
                        metric_values = {
                            "accuracy": math.nan,
                            "balacc": math.nan,
                            "auc": math.nan,
                            "f1": math.nan,
                            "brier": math.nan,
                            "precision": math.nan,
                            "recall": math.nan,
                            "ece": math.nan,
                        }

                    row["accuracy"] = metric_values.get("accuracy", math.nan)
                    row["balanced_accuracy"] = metric_values.get("balacc", math.nan)
                    row["auc"] = metric_values.get("auc", math.nan)
                    row["f1"] = metric_values.get("f1", math.nan)
                    row["brier"] = metric_values.get("brier", math.nan)
                    row["issues"] = aggregate_warning_list(warnings_out)
                    rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_DIR / "training_domain_experiment_results.csv", index=False)

    summary_by_model = (
        result_df.groupby(["model", "n_train_sites"], dropna=False)
        .agg(
            n_experiments=("model", "size"),
            mean_accuracy=("accuracy", "mean"),
            sd_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            sd_balanced_accuracy=("balanced_accuracy", "std"),
            mean_auc=("auc", "mean"),
            sd_auc=("auc", "std"),
        )
        .reset_index()
    )
    summary_by_model.to_csv(
        OUTPUT_DIR / "training_domain_experiment_summary_by_model.csv", index=False
    )

    summary_by_site = (
        result_df.groupby(["test_site", "model", "n_train_sites"], dropna=False)
        .agg(
            n_experiments=("model", "size"),
            mean_accuracy=("accuracy", "mean"),
            sd_accuracy=("accuracy", "std"),
            mean_auc=("auc", "mean"),
            sd_auc=("auc", "std"),
        )
        .reset_index()
    )
    summary_by_site.to_csv(
        OUTPUT_DIR / "training_domain_experiment_summary_by_site.csv", index=False
    )
    return result_df


def plot_training_domain_all_combinations_heatmap(combo_df: pd.DataFrame) -> None:
    if combo_df.empty:
        return

    test_sites = [
        site for site in DOMAIN_ORDER if site in combo_df["test_site"].unique()
    ]
    models = [
        model
        for model in ["logistic", "random_forest", "hist_gradient_boosting"]
        if model in combo_df["model"].unique()
    ]
    all_values = combo_df["accuracy"].dropna().to_numpy(dtype=float)
    vmin, vmax = float(all_values.min()), float(all_values.max())

    fig, axes = plt.subplots(
        len(test_sites), 1, figsize=(12, 3.2 * len(test_sites)), constrained_layout=True
    )
    if len(test_sites) == 1:
        axes = [axes]
    fig.suptitle(
        "Exhaustive training-site combinations for each held-out test site",
        fontsize=18,
        y=1.01,
    )

    im = None
    for ax, test_site in zip(axes, test_sites):
        subset = combo_df.loc[combo_df["test_site"] == test_site].copy()
        subset["row_label"] = subset.apply(
            lambda row: (
                f"Train {int(row['n_train_sites'])} site(s) | {row['train_sites']}"
            ),
            axis=1,
        )
        order_df = (
            subset[["row_label", "n_train_sites", "train_sites"]]
            .drop_duplicates()
            .sort_values(["n_train_sites", "train_sites"], ascending=[False, True])
        )
        row_order = order_df["row_label"].tolist()
        pivot = subset.pivot_table(
            index="row_label", columns="model", values="accuracy", aggfunc="mean"
        ).reindex(index=row_order, columns=models)
        im = heatmap_with_annotations(
            ax,
            pivot.to_numpy(dtype=float),
            pivot.index.tolist(),
            pivot.columns.tolist(),
            title=test_site,
            cmap=performance_cmap(),
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel("Training combination", fontsize=10)
    cbar = fig.colorbar(im, ax=axes, shrink=0.5, location="bottom", pad=0.04)
    cbar.set_label("Accuracy", fontsize=12)
    save_figure(
        fig,
        OUTPUT_DIR
        / "plots"
        / "transportability_training_domains_all_combinations_heatmap",
    )


def plot_training_domains_by_site_heatmap(combo_df: pd.DataFrame) -> None:
    if combo_df.empty:
        return
    summary = (
        combo_df.groupby(["test_site", "model", "n_train_sites"], dropna=False)[
            "accuracy"
        ]
        .mean()
        .reset_index(name="mean_accuracy")
    )
    test_sites = [
        site for site in reversed(DOMAIN_ORDER) if site in summary["test_site"].unique()
    ]
    models = [
        model
        for model in ["logistic", "random_forest", "hist_gradient_boosting"]
        if model in summary["model"].unique()
    ]
    n_train_values = sorted(summary["n_train_sites"].dropna().unique().tolist())
    all_values = summary["mean_accuracy"].dropna().to_numpy(dtype=float)
    vmin, vmax = float(all_values.min()), float(all_values.max())

    fig, axes = plt.subplots(
        1, len(models), figsize=(5.5 * len(models), 8), constrained_layout=True
    )
    if len(models) == 1:
        axes = [axes]
    fig.suptitle(
        "Transportability improvement by held-out site as training domains increase",
        fontsize=20,
        y=1.02,
    )

    im = None
    for ax, model in zip(axes, models):
        subset = summary.loc[summary["model"] == model].copy()
        pivot = subset.pivot_table(
            index="test_site",
            columns="n_train_sites",
            values="mean_accuracy",
            aggfunc="mean",
        ).reindex(index=test_sites, columns=n_train_values)
        im = heatmap_with_annotations(
            ax,
            pivot.to_numpy(dtype=float),
            pivot.index.tolist(),
            [str(v) for v in pivot.columns.tolist()],
            title=model,
            cmap=performance_cmap(),
            vmin=vmin,
            vmax=vmax,
        )
        if ax is axes[0]:
            ax.set_ylabel("Held-out test site", fontsize=18)
        ax.set_xlabel("Number of training sites", fontsize=12)
    cbar = fig.colorbar(im, ax=axes, shrink=0.55, location="bottom", pad=0.05)
    cbar.set_label("Mean accuracy", fontsize=12)
    save_figure(
        fig, OUTPUT_DIR / "plots" / "transportability_training_domains_by_site_heatmap"
    )


def plot_training_domains_heatmap(combo_df: pd.DataFrame) -> None:
    if combo_df.empty:
        return
    summary = (
        combo_df.groupby(["model", "n_train_sites"], dropna=False)["accuracy"]
        .mean()
        .reset_index(name="mean_accuracy")
    )
    models = [
        model
        for model in ["hist_gradient_boosting", "random_forest", "logistic"]
        if model in summary["model"].unique()
    ]
    n_train_values = sorted(summary["n_train_sites"].dropna().unique().tolist())
    pivot = summary.pivot_table(
        index="model", columns="n_train_sites", values="mean_accuracy", aggfunc="mean"
    ).reindex(index=models, columns=n_train_values)
    values = pivot.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return
    vmin, vmax = float(finite_values.min()), float(finite_values.max())

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    im = heatmap_with_annotations(
        ax,
        values,
        pivot.index.tolist(),
        [str(v) for v in pivot.columns.tolist()],
        title="Effect of training domain diversity on transportability",
        cmap=performance_cmap(),
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Number of training sites", fontsize=14)
    ax.set_ylabel("Model", fontsize=14)
    cbar = fig.colorbar(im, ax=ax, shrink=0.72, location="bottom", pad=0.12)
    cbar.set_label("Mean accuracy", fontsize=12)
    save_figure(fig, OUTPUT_DIR / "plots" / "transportability_training_domains_heatmap")


def plot_pairwise_heatmaps_redgreen(pairwise_df: pd.DataFrame) -> None:
    if pairwise_df.empty:
        return
    pairwise_df = pairwise_df.copy()
    pairwise_df["train_domain_display"] = pairwise_df["train_domain"].map(pretty_domain)
    pairwise_df["test_domain_display"] = pairwise_df["test_domain"].map(pretty_domain)
    pairwise_df["model_display"] = pairwise_df["model"].map(pretty_model)

    model_order = [
        pretty_model(m) for m in sorted(pairwise_df["model"].dropna().unique().tolist())
    ]
    row_order = [
        site
        for site in DOMAIN_ORDER
        if site in pairwise_df["train_domain_display"].unique()
    ]
    col_order = [
        site
        for site in DOMAIN_ORDER
        if site in pairwise_df["test_domain_display"].unique()
    ]

    for metric_col, prefix in [("auc", "auc"), ("f1", "f1")]:
        finite = pairwise_df[metric_col].dropna().to_numpy(dtype=float)
        if finite.size == 0:
            continue
        vmin, vmax = float(finite.min()), float(finite.max())
        for model in model_order:
            subset = pairwise_df.loc[pairwise_df["model_display"] == model]
            pivot = subset.pivot_table(
                index="train_domain_display",
                columns="test_domain_display",
                values=metric_col,
                aggfunc="mean",
            ).reindex(index=row_order, columns=col_order)
            fig, ax = plt.subplots(figsize=(8, 6))
            im = heatmap_with_annotations(
                ax,
                pivot.to_numpy(dtype=float),
                pivot.index.tolist(),
                pivot.columns.tolist(),
                title=f"Train site -> test site ({prefix.upper()}): {model}",
                cmap=performance_cmap(),
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("Test site", fontsize=12)
            ax.set_ylabel("Train site", fontsize=12)
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(prefix.upper(), fontsize=11)
            save_figure(
                fig, OUTPUT_DIR / "plots" / f"train_test_heatmap_{prefix}_{model}"
            )


def plot_simple_heatmap_redgreen(
    df: pd.DataFrame, metric_column: str, output_name: str, title: str
) -> None:
    if df.empty or metric_column not in df.columns:
        return
    plot_df = df.copy()
    plot_df["test_domain_display"] = plot_df["test_domain"].map(pretty_domain)
    plot_df["model_display"] = plot_df["model"].map(pretty_model)
    pivot = plot_df.pivot_table(
        index="model_display",
        columns="test_domain_display",
        values=metric_column,
        aggfunc="mean",
    ).reindex(
        index=[
            m
            for m in ["logistic", "random_forest", "hist_gradient_boosting"]
            if m in plot_df["model_display"].unique()
        ],
        columns=[
            d for d in DOMAIN_ORDER if d in plot_df["test_domain_display"].unique()
        ],
    )
    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    cmap = gap_cmap() if metric_column.startswith("gap_") else performance_cmap()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = heatmap_with_annotations(
        ax,
        values,
        pivot.index.tolist(),
        pivot.columns.tolist(),
        title=title,
        cmap=cmap,
        vmin=float(finite.min()),
        vmax=float(finite.max()),
    )
    ax.set_xlabel("Holdout site", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label(metric_column, fontsize=10)
    save_figure(fig, OUTPUT_DIR / "plots" / output_name)


def regenerate_existing_redgreen_plots(
    fold_results: pd.DataFrame, model_summary: pd.DataFrame, pairwise_df: pd.DataFrame
) -> None:
    plot_df = fold_results.copy()
    plot_df["test_domain"] = plot_df["test_domain"].map(pretty_domain)
    plot_df["model"] = plot_df["model"].map(pretty_model)
    model_summary = model_summary.copy()
    model_summary["model"] = model_summary["model"].map(pretty_model)

    plot_grouped_bar_redgreen(
        plot_df,
        value_col="auc_ood",
        x_col="test_domain",
        group_col="model",
        title="OOD ROC-AUC by holdout site and model",
        ylabel="ROC-AUC",
        output_name="auc_ood_by_domain",
        good_high=True,
    )
    plot_grouped_bar_redgreen(
        plot_df,
        value_col="f1_ood",
        x_col="test_domain",
        group_col="model",
        title="OOD F1 by holdout site and model",
        ylabel="F1-score",
        output_name="f1_ood_by_domain",
        good_high=True,
    )
    plot_grouped_bar_redgreen(
        plot_df,
        value_col="accuracy_ood",
        x_col="test_domain",
        group_col="model",
        title="OOD accuracy by holdout site and model",
        ylabel="Accuracy",
        output_name="accuracy_ood_by_domain",
        good_high=True,
    )
    plot_grouped_bar_redgreen(
        plot_df,
        value_col="gap_auc",
        x_col="test_domain",
        group_col="model",
        title="ID-OOD ROC-AUC gap by holdout site and model",
        ylabel="Gap (ID - OOD)",
        output_name="gap_auc_by_domain",
        good_high=False,
    )
    plot_grouped_bar_redgreen(
        plot_df,
        value_col="gap_f1",
        x_col="test_domain",
        group_col="model",
        title="ID-OOD F1 gap by holdout site and model",
        ylabel="Gap (ID - OOD)",
        output_name="gap_f1_by_domain",
        good_high=False,
    )
    plot_grouped_bar_redgreen(
        plot_df,
        value_col="gap_accuracy",
        x_col="test_domain",
        group_col="model",
        title="ID-OOD accuracy gap by holdout site and model",
        ylabel="Gap (ID - OOD)",
        output_name="gap_accuracy_by_domain",
        good_high=False,
    )

    mean_df = model_summary.melt(
        id_vars=["model"],
        value_vars=[
            "auc_ood_mean",
            "accuracy_ood_mean",
            "f1_ood_mean",
            "balacc_ood_mean",
        ],
        var_name="metric",
        value_name="value",
    )
    mean_df["metric"] = (
        mean_df["metric"].str.replace("_ood_mean", "", regex=False).str.upper()
    )
    mean_df["metric"] = mean_df["metric"].replace({"BALACC": "BALACC"})
    plot_grouped_bar_redgreen(
        mean_df.rename(columns={"metric": "metric_label"}),
        value_col="value",
        x_col="metric_label",
        group_col="model",
        title="Mean OOD performance by model",
        ylabel="Score",
        output_name="mean_performance_by_model",
        good_high=True,
    )

    plot_simple_heatmap_redgreen(
        fold_results, "auc_ood", "heatmap_auc_ood", "Heatmap: OOD ROC-AUC"
    )
    plot_simple_heatmap_redgreen(
        fold_results, "f1_ood", "heatmap_f1_ood", "Heatmap: OOD F1"
    )
    plot_simple_heatmap_redgreen(
        fold_results, "accuracy_ood", "heatmap_accuracy_ood", "Heatmap: OOD accuracy"
    )
    plot_site_model_ood_heatmap(fold_results)
    plot_pairwise_heatmaps_redgreen(pairwise_df)


def main() -> None:
    ensure_directory(OUTPUT_DIR / "plots")
    logger = init_logger(OUTPUT_DIR / "logs_figures.txt")
    config = ExperimentConfig.from_json(CONFIG_PATH)
    config.output_dir = str(OUTPUT_DIR)

    fold_results = pd.read_csv(OUTPUT_DIR / "fold_results.csv")
    model_summary = pd.read_csv(OUTPUT_DIR / "model_summary.csv")
    pairwise_df = pd.read_csv(OUTPUT_DIR / "train_test_site_results.csv")

    dataset_df, _, _ = load_dataset(config, logger)
    combo_df = run_training_domain_experiments(dataset_df, config, logger)

    regenerate_existing_redgreen_plots(fold_results, model_summary, pairwise_df)
    plot_training_domain_all_combinations_heatmap(combo_df)
    plot_training_domains_by_site_heatmap(combo_df)
    plot_training_domains_heatmap(combo_df)


if __name__ == "__main__":
    main()
