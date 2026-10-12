"""Transportability experiment: pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .bootstrap import bootstrap_environment

PROJECT_ROOT = bootstrap_environment()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .datasets import DatasetBundle, load_requested_datasets, save_audit


def _bootstrap_optional_harmonizers() -> dict[str, Any]:
    from neuroCombat import neuroCombat
    from inmoose.pycombat import pycombat_norm, pycombat_seq

    return {
        "neuroCombat": neuroCombat,
        "pycombat_norm": pycombat_norm,
        "pycombat_seq": pycombat_seq,
    }


HARMONIZERS = _bootstrap_optional_harmonizers()


def _safe_auc(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob_pos))


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prob_pos: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auc": _safe_auc(y_true, prob_pos),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _largest_batch(batch: pd.Series) -> str:
    return str(batch.value_counts().idxmax())


def _harmonize_train_only_standard(
    train_frame: pd.DataFrame,
    batch_train: pd.Series,
    y_train: pd.Series,
    method: str,
) -> pd.DataFrame:
    if train_frame.shape[1] == 0:
        return train_frame.copy()

    train_matrix = train_frame.T
    covars = pd.DataFrame(
        {
            "batch": batch_train.astype(str).to_numpy(),
            "label": y_train.astype(str).to_numpy(),
        }
    )
    ref_batch = _largest_batch(batch_train)
    if method == "neuroCombat":
        result = HARMONIZERS["neuroCombat"](
            dat=train_matrix,
            covars=covars,
            batch_col="batch",
            categorical_cols=["label"],
            continuous_cols=None,
            ref_batch=ref_batch,
        )
        adjusted = pd.DataFrame(
            result["data"].T, index=train_frame.index, columns=train_frame.columns
        )
        return adjusted
    if method == "pycombat_norm":
        adjusted = HARMONIZERS["pycombat_norm"](
            counts=train_matrix,
            batch=batch_train.astype(str).tolist(),
            covar_mod=pd.DataFrame({"label": y_train.astype(str).to_numpy()}),
            ref_batch=ref_batch,
        )
        if isinstance(adjusted, pd.DataFrame):
            return adjusted.T.reindex(
                index=train_frame.index, columns=train_frame.columns
            )
        return pd.DataFrame(
            np.asarray(adjusted).T, index=train_frame.index, columns=train_frame.columns
        )
    raise KeyError(f"Unsupported standard harmonization method: {method}")


def _log_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    lib_size = counts.sum(axis=1).replace(0, np.nan)
    cpm = counts.div(lib_size, axis=0) * 1_000_000.0
    return np.log2(cpm + 1.0)


def _select_top_variance_features(train_frame: pd.DataFrame, top_k: int) -> list[str]:
    if train_frame.shape[1] <= top_k:
        return train_frame.columns.tolist()
    variances = train_frame.var(axis=0, ddof=1).sort_values(ascending=False)
    return variances.head(top_k).index.tolist()


def _prepare_tabular_fold(
    bundle: DatasetBundle,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    harmonize: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    all_features = [*bundle.harmonizable_features, *bundle.passthrough_features]
    train_raw = train_df[all_features].copy()
    test_raw = test_df[all_features].copy()

    initial_imputer = SimpleImputer(strategy="median")
    train_imp = pd.DataFrame(
        initial_imputer.fit_transform(train_raw),
        columns=all_features,
        index=train_df.index,
    )
    test_imp = pd.DataFrame(
        initial_imputer.transform(test_raw), columns=all_features, index=test_df.index
    )

    if harmonize and bundle.harmonizable_features:
        train_harm = _harmonize_train_only_standard(
            train_frame=train_imp[bundle.harmonizable_features],
            batch_train=train_df[bundle.domain_col],
            y_train=train_df[bundle.label_col],
            method="neuroCombat" if bundle.dataset_id == "abide2" else "pycombat_norm",
        )
        test_harm = test_imp[bundle.harmonizable_features].copy()
        train_final = pd.concat(
            [train_harm, train_imp[bundle.passthrough_features]], axis=1
        )
        test_final = pd.concat(
            [test_harm, test_imp[bundle.passthrough_features]], axis=1
        )
        test_harmonized = False
    else:
        train_final = train_imp
        test_final = test_imp
        test_harmonized = False

    final_imputer = SimpleImputer(strategy="median")
    train_final = pd.DataFrame(
        final_imputer.fit_transform(train_final),
        columns=train_final.columns,
        index=train_final.index,
    )
    test_final = pd.DataFrame(
        final_imputer.transform(test_final),
        columns=test_final.columns,
        index=test_final.index,
    )

    selector_k = min(30, train_final.shape[1])
    if selector_k < train_final.shape[1]:
        selector = SelectKBest(score_func=f_classif, k=selector_k)
        train_selected = selector.fit_transform(
            train_final, train_df[bundle.label_col].to_numpy(dtype=int)
        )
        test_selected = selector.transform(test_final)
        selected_features = train_final.columns[selector.get_support()].tolist()
        train_final = pd.DataFrame(
            train_selected, columns=selected_features, index=train_final.index
        )
        test_final = pd.DataFrame(
            test_selected, columns=selected_features, index=test_final.index
        )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_final)
    x_test = scaler.transform(test_final)

    return (
        x_train,
        x_test,
        train_final.columns.tolist(),
        {"test_harmonized": test_harmonized},
    )


def _prepare_tcga_fold(
    bundle: DatasetBundle,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    harmonize: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    feature_columns = bundle.feature_columns
    train_expr = train_df[feature_columns].copy()
    test_expr = test_df[feature_columns].copy()

    if bundle.extra["tcga_input_is_raw_counts"]:
        min_positive = max(5, int(np.ceil(0.10 * len(train_df))))
        keep_mask = (train_expr > 1).sum(axis=0) >= min_positive
        kept_columns = keep_mask[keep_mask].index.tolist()
        if len(kept_columns) < 50:
            raise RuntimeError("Too few TCGA genes remain after count filtering.")
        train_expr = train_expr[kept_columns]
        test_expr = test_expr[kept_columns]

        log_train = _log_cpm(train_expr)
        selected = _select_top_variance_features(
            log_train, top_k=min(2000, log_train.shape[1])
        )
        train_expr = train_expr[selected]
        test_expr = test_expr[selected]
        log_train = _log_cpm(train_expr)
        log_test = _log_cpm(test_expr)

        if harmonize:
            ref_batch = _largest_batch(train_df[bundle.domain_col])
            adjusted = HARMONIZERS["pycombat_seq"](
                counts=train_expr.T,
                batch=train_df[bundle.domain_col].astype(str).tolist(),
                covar_mod=pd.DataFrame(
                    {"label": train_df[bundle.label_col].astype(str).to_numpy()}
                ),
                ref_batch=ref_batch,
            )
            harm_train_counts = adjusted.T.reindex(
                index=train_expr.index, columns=train_expr.columns
            )
            train_final = _log_cpm(harm_train_counts)
            test_final = log_test
            test_harmonized = False
        else:
            train_final = log_train
            test_final = log_test
            test_harmonized = False
    else:
        selected = _select_top_variance_features(
            train_expr, top_k=min(2000, train_expr.shape[1])
        )
        train_final = train_expr[selected].copy()
        test_final = test_expr[selected].copy()
        test_harmonized = False

    selector_k = min(500, train_final.shape[1])
    if selector_k < train_final.shape[1]:
        selector = SelectKBest(score_func=f_classif, k=selector_k)
        train_selected = selector.fit_transform(
            train_final, train_df[bundle.label_col].to_numpy(dtype=int)
        )
        test_selected = selector.transform(test_final)
        selected_features = train_final.columns[selector.get_support()].tolist()
        train_final = pd.DataFrame(
            train_selected, columns=selected_features, index=train_final.index
        )
        test_final = pd.DataFrame(
            test_selected, columns=selected_features, index=test_final.index
        )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_final)
    x_test = scaler.transform(test_final)
    return (
        x_train,
        x_test,
        train_final.columns.tolist(),
        {"test_harmonized": test_harmonized},
    )


def _prepare_fold(
    bundle: DatasetBundle,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    harmonize: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    if bundle.dataset_id == "tcga_brca":
        return _prepare_tcga_fold(bundle, train_df, test_df, harmonize)
    return _prepare_tabular_fold(bundle, train_df, test_df, harmonize)


def _build_models(seed: int = 42) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            max_iter=300,
            random_state=seed,
        ),
    }


def _fit_predict_model(
    model_name: str,
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    if model_name == "hist_gradient_boosting":
        model.fit(x_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(x_train, y_train)
    prob_pos = model.predict_proba(x_test)[:, 1]
    pred = (prob_pos >= 0.5).astype(int)
    return pred, prob_pos


def _save_domain_count_plot(bundle: DatasetBundle, plots_dir: Path) -> None:
    counts = bundle.data[bundle.domain_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(counts.index.astype(str), counts.to_numpy(), color="#4472c4")
    ax.set_title(f"{bundle.display_name}: samples per domain")
    ax.set_ylabel("n samples")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(plots_dir / "domain_counts.png", dpi=180)
    plt.close(fig)


def _save_label_by_domain_plot(bundle: DatasetBundle, plots_dir: Path) -> None:
    table = pd.crosstab(
        bundle.data[bundle.domain_col], bundle.data[bundle.label_col]
    ).sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(table))
    colors = ["#8ecae6", "#d90429"]
    for idx, column in enumerate(table.columns):
        values = table[column].to_numpy()
        ax.bar(
            table.index.astype(str),
            values,
            bottom=bottom,
            label=f"class_{column}",
            color=colors[idx % len(colors)],
        )
        bottom += values
    ax.set_title(f"{bundle.display_name}: label distribution by domain")
    ax.set_ylabel("n samples")
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "label_by_domain.png", dpi=180)
    plt.close(fig)


def _save_pca_plots(bundle: DatasetBundle, plots_dir: Path) -> None:
    if bundle.dataset_id == "tcga_brca":
        feature_columns = _select_top_variance_features(
            bundle.data[bundle.feature_columns],
            top_k=min(500, len(bundle.feature_columns)),
        )
    else:
        feature_columns = bundle.feature_columns
    x = bundle.data[feature_columns].copy()
    x = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(x), columns=feature_columns
    )
    x_scaled = StandardScaler().fit_transform(x)
    coords = PCA(n_components=2, random_state=42).fit_transform(x_scaled)
    domain_codes = pd.Categorical(bundle.data[bundle.domain_col].astype(str))
    label_codes = pd.Categorical(bundle.data[bundle.label_col].astype(str))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter0 = axes[0].scatter(
        coords[:, 0], coords[:, 1], c=domain_codes.codes, cmap="tab20", s=18
    )
    axes[0].set_title(f"{bundle.display_name}: PCA by domain")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    scatter1 = axes[1].scatter(
        coords[:, 0], coords[:, 1], c=label_codes.codes, cmap="Set1", s=18
    )
    axes[1].set_title(f"{bundle.display_name}: PCA by label")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(plots_dir / "pca_raw.png", dpi=180)
    plt.close(fig)

    if (
        bundle.harmonization_allowed
        and bundle.dataset_id in {"abide2", "uci_heart"}
        and bundle.harmonizable_features
    ):
        x_imp = pd.DataFrame(
            SimpleImputer(strategy="median").fit_transform(
                bundle.data[feature_columns]
            ),
            columns=feature_columns,
            index=bundle.data.index,
        )
        harmonized = _harmonize_train_only_standard(
            train_frame=x_imp[bundle.harmonizable_features],
            batch_train=bundle.data[bundle.domain_col],
            y_train=bundle.data[bundle.label_col],
            method="neuroCombat" if bundle.dataset_id == "abide2" else "pycombat_norm",
        )
        x_harm = pd.concat(
            [
                harmonized,
                x_imp[
                    [
                        c
                        for c in feature_columns
                        if c not in bundle.harmonizable_features
                    ]
                ],
            ],
            axis=1,
        )
        x_harm = x_harm[feature_columns]
        coords_h = PCA(n_components=2, random_state=42).fit_transform(
            StandardScaler().fit_transform(x_harm)
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(
            coords_h[:, 0], coords_h[:, 1], c=domain_codes.codes, cmap="tab20", s=18
        )
        axes[0].set_title(
            f"{bundle.display_name}: PCA after descriptive global harmonization"
        )
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        axes[1].scatter(
            coords_h[:, 0], coords_h[:, 1], c=label_codes.codes, cmap="Set1", s=18
        )
        axes[1].set_title(
            f"{bundle.display_name}: label structure after descriptive global harmonization"
        )
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        fig.tight_layout()
        fig.savefig(plots_dir / "pca_harmonized_descriptive.png", dpi=180)
        plt.close(fig)


def _save_metric_plots(metrics_per_domain: pd.DataFrame, plots_dir: Path) -> None:
    if metrics_per_domain.empty:
        return
    for metric in ("accuracy", "balanced_accuracy", "auc", "f1"):
        fig, ax = plt.subplots(figsize=(11, 4))
        pivot = metrics_per_domain.pivot_table(
            index="holdout_domain", columns=["pipeline", "model"], values=metric
        )
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"OOD {metric} by holdout domain")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=60)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric}_by_domain.png", dpi=180)
        plt.close(fig)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    columns = [str(col) for col in df.columns]
    body_rows = [[str(value) for value in row] for row in df.to_numpy().tolist()]
    widths = [
        max(len(columns[idx]), *(len(row[idx]) for row in body_rows))
        if body_rows
        else len(columns[idx])
        for idx in range(len(columns))
    ]
    header = (
        "| "
        + " | ".join(columns[idx].ljust(widths[idx]) for idx in range(len(columns)))
        + " |"
    )
    separator = (
        "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    )
    rows = [
        "| "
        + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns)))
        + " |"
        for row in body_rows
    ]
    return "\n".join([header, separator, *rows])


def _write_report(
    bundle: DatasetBundle,
    dataset_dir: Path,
    metrics_per_domain: pd.DataFrame,
    metrics_summary: pd.DataFrame,
) -> None:
    report_lines = [
        f"# {bundle.display_name}",
        "",
        "## Audit",
        f"- Domain kind: `{bundle.domain_kind}`",
        f"- Domain real: `{bundle.domain_real}`",
        f"- Harmonization requested: `{bundle.harmonization_method}`",
        f"- Harmonization applicable on current local input: `{bundle.harmonization_allowed}`",
        f"- Harmonization rationale: {bundle.harmonization_reason}",
        f"- Interpretability: {bundle.interpretability_note}",
        "",
        "## Dataset final",
        f"- Samples: {bundle.audit.get('n_samples', len(bundle.data))}",
        f"- Features: {bundle.audit.get('n_features', len(bundle.feature_columns))}",
        f"- Domains: {json.dumps(bundle.audit.get('site_counts', bundle.audit.get('domain_counts', {})), indent=2)}",
        f"- Classes: {json.dumps(bundle.audit.get('class_counts', {}), indent=2)}",
        "",
        "## Leakage control",
        "- Leave-one-domain-out split by real domain.",
        "- Imputation, scaling, feature selection and harmonization are fitted on training domains only.",
        "- Held-out domains are never used to estimate ComBat parameters.",
        "- Because ComBat-family methods cannot estimate parameters for a completely unseen holdout batch without using test data, the holdout domain is left unharmonized under strict no-leakage LODO.",
        "",
        "## Plots",
        "- `plots/pca_raw.png`: descriptive PCA before harmonization.",
        "- `plots/pca_harmonized_descriptive.png`: descriptive global harmonization view only when the requested method is applicable; this figure is not used for model fitting or evaluation.",
        "- `plots/*_by_domain.png`: OOD metric comparison by held-out domain.",
        "",
        "## Metrics summary",
    ]
    if metrics_summary.empty:
        report_lines.append("- No summary metrics available.")
    else:
        report_lines.append(_dataframe_to_markdown(metrics_summary))
    report_lines.extend(["", "## Holdout domain metrics"])
    if metrics_per_domain.empty:
        report_lines.append("- No fold-level metrics available.")
    else:
        report_lines.append(_dataframe_to_markdown(metrics_per_domain))
    (dataset_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


def _evaluate_bundle(bundle: DatasetBundle, output_root: Path) -> None:
    dataset_dir = output_root / bundle.dataset_id
    plots_dir = dataset_dir / "plots"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_audit(bundle, dataset_dir)
    _save_domain_count_plot(bundle, plots_dir)
    _save_label_by_domain_plot(bundle, plots_dir)
    _save_pca_plots(bundle, plots_dir)

    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    models = _build_models()
    pipelines = [("baseline", False)]
    if bundle.harmonization_allowed:
        pipelines.append(("harmonized_train_only", True))

    holdout_domains = sorted(
        bundle.data[bundle.domain_col].astype(str).unique().tolist()
    )
    for holdout_domain in holdout_domains:
        train_df = bundle.data.loc[
            bundle.data[bundle.domain_col].astype(str) != holdout_domain
        ].copy()
        test_df = bundle.data.loc[
            bundle.data[bundle.domain_col].astype(str) == holdout_domain
        ].copy()
        y_train = train_df[bundle.label_col].to_numpy(dtype=int)
        y_test = test_df[bundle.label_col].to_numpy(dtype=int)

        for pipeline_name, do_harmonize in pipelines:
            x_train, x_test, selected_features, prep_info = _prepare_fold(
                bundle, train_df, test_df, do_harmonize
            )
            for model_name, model in _build_models().items():
                pred, prob_pos = _fit_predict_model(
                    model_name, model, x_train, y_train, x_test
                )
                metric_row = {
                    "dataset": bundle.dataset_id,
                    "pipeline": pipeline_name,
                    "model": model_name,
                    "holdout_domain": holdout_domain,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "n_features": int(len(selected_features)),
                    "test_harmonized": prep_info["test_harmonized"],
                    **_compute_metrics(y_test, pred, prob_pos),
                }
                metrics_rows.append(metric_row)
                for sample_id, truth, pred_label, prob in zip(
                    test_df[bundle.sample_id_col].astype(str).tolist(),
                    y_test.tolist(),
                    pred.tolist(),
                    prob_pos.tolist(),
                ):
                    prediction_rows.append(
                        {
                            "dataset": bundle.dataset_id,
                            "pipeline": pipeline_name,
                            "model": model_name,
                            "holdout_domain": holdout_domain,
                            "sample_id": sample_id,
                            "domain": holdout_domain,
                            "y_true": truth,
                            "y_pred": int(pred_label),
                            "prob_positive": float(prob),
                        }
                    )

    metrics_per_domain = (
        pd.DataFrame(metrics_rows)
        .sort_values(["pipeline", "model", "holdout_domain"])
        .reset_index(drop=True)
    )
    metrics_summary = (
        metrics_per_domain.groupby(["pipeline", "model"], dropna=False)[
            ["accuracy", "balanced_accuracy", "auc", "f1"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    metrics_summary.columns = [
        "_".join([str(part) for part in col if str(part) != ""]).rstrip("_")
        for col in metrics_summary.columns.to_flat_index()
    ]
    predictions = (
        pd.DataFrame(prediction_rows)
        .sort_values(["pipeline", "model", "holdout_domain", "sample_id"])
        .reset_index(drop=True)
    )

    metrics_per_domain.to_csv(dataset_dir / "metrics_per_domain.csv", index=False)
    metrics_summary.to_csv(dataset_dir / "metrics_summary.csv", index=False)
    predictions.to_csv(dataset_dir / "predictions.csv", index=False)
    _save_metric_plots(metrics_per_domain, plots_dir)
    _write_report(bundle, dataset_dir, metrics_per_domain, metrics_summary)


def run_datasets(dataset_ids: list[str], output_root: Path) -> None:
    bundles = load_requested_datasets(dataset_ids)
    for bundle in bundles:
        _evaluate_bundle(bundle, output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline vs ComBat transportability benchmark"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["abide2", "uci_heart", "tcga_brca"],
        choices=["abide2", "uci_heart", "tcga_brca"],
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs"),
        help="Directory where dataset-specific outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_datasets(args.datasets, Path(args.output_root))


if __name__ == "__main__":
    main()
