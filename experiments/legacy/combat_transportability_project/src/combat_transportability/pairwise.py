"""Transportability experiment: pairwise."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .bootstrap import bootstrap_environment

PROJECT_ROOT = bootstrap_environment()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .datasets import DatasetBundle, load_requested_datasets
from .pipeline import (
    _build_models,
    _fit_predict_model,
    _harmonize_train_only_standard,
    _log_cpm,
    _select_top_variance_features,
)


def _prepare_global_tabular_frames(
    bundle: DatasetBundle,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    harmonize: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    if harmonize and bundle.harmonization_allowed and bundle.harmonizable_features:
        train_harm = _harmonize_train_only_standard(
            train_frame=train_imp[bundle.harmonizable_features],
            batch_train=train_df[bundle.domain_col],
            y_train=train_df[bundle.label_col],
            method="neuroCombat" if bundle.dataset_id == "abide2" else "pycombat_norm",
        )
        train_final = pd.concat(
            [train_harm, train_imp[bundle.passthrough_features]], axis=1
        )
        test_final = pd.concat(
            [
                test_imp[bundle.harmonizable_features],
                test_imp[bundle.passthrough_features],
            ],
            axis=1,
        )
        train_final = train_final[all_features]
        test_final = test_final[all_features]
    else:
        train_final = train_imp
        test_final = test_imp

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
    return train_final, test_final


def _prepare_global_tcga_frames(
    bundle: DatasetBundle,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    harmonize: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = bundle.feature_columns
    train_expr = train_df[feature_columns].copy()
    test_expr = test_df[feature_columns].copy()

    if bundle.extra["tcga_input_is_raw_counts"]:
        min_positive = max(5, int(np.ceil(0.10 * len(train_df))))
        keep_mask = (train_expr > 1).sum(axis=0) >= min_positive
        kept_columns = keep_mask[keep_mask].index.tolist()
        if len(kept_columns) < 50:
            raise RuntimeError(
                "Too few TCGA genes remain after training-only count filtering."
            )
        train_expr = train_expr[kept_columns]
        test_expr = test_expr[kept_columns]

        log_train = _log_cpm(train_expr)
        selected = _select_top_variance_features(
            log_train, top_k=min(2000, log_train.shape[1])
        )
        train_expr = train_expr[selected]
        test_expr = test_expr[selected]

        if harmonize and bundle.harmonization_allowed:
            from inmoose.pycombat import pycombat_seq

            adjusted = pycombat_seq(
                counts=train_expr.T,
                batch=train_df[bundle.domain_col].astype(str).tolist(),
                covar_mod=pd.DataFrame(
                    {"label": train_df[bundle.label_col].astype(str).to_numpy()}
                ),
                ref_batch=str(train_df[bundle.domain_col].value_counts().idxmax()),
            )
            if isinstance(adjusted, pd.DataFrame):
                harm_train_counts = adjusted.T.reindex(
                    index=train_expr.index, columns=train_expr.columns
                )
            else:
                harm_train_counts = pd.DataFrame(
                    np.asarray(adjusted).T,
                    index=train_expr.index,
                    columns=train_expr.columns,
                )
            train_final = _log_cpm(harm_train_counts)
            test_final = _log_cpm(test_expr)
        else:
            train_final = _log_cpm(train_expr)
            test_final = _log_cpm(test_expr)
        return train_final, test_final

    selected = _select_top_variance_features(
        train_expr, top_k=min(2000, train_expr.shape[1])
    )
    return train_expr[selected].copy(), test_expr[selected].copy()


def _prepare_global_frames(
    bundle: DatasetBundle,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    harmonize: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bundle.dataset_id == "tcga_brca":
        return _prepare_global_tcga_frames(bundle, train_df, test_df, harmonize)
    return _prepare_global_tabular_frames(bundle, train_df, test_df, harmonize)


def _select_and_scale_source_specific(
    bundle: DatasetBundle,
    source_train_frame: pd.DataFrame,
    source_y: np.ndarray,
    test_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if source_train_frame.shape[1] == 0:
        raise RuntimeError("No features available for source-specific training.")

    target_k = 500 if bundle.dataset_id == "tcga_brca" else 30
    selector_k = min(target_k, source_train_frame.shape[1])
    if selector_k < source_train_frame.shape[1]:
        selector = SelectKBest(score_func=f_classif, k=selector_k)
        source_selected = selector.fit_transform(source_train_frame, source_y)
        test_selected = selector.transform(test_frame)
        selected_features = source_train_frame.columns[selector.get_support()].tolist()
    else:
        source_selected = source_train_frame.to_numpy()
        test_selected = test_frame.to_numpy()
        selected_features = source_train_frame.columns.tolist()

    scaler = StandardScaler()
    x_source = scaler.fit_transform(source_selected)
    x_test = scaler.transform(test_selected)
    return x_source, x_test, selected_features


def _plot_heatmap(matrix: pd.DataFrame, title: str, output_path: Path) -> None:
    fig_w = max(6.5, 1.0 + 0.65 * len(matrix.columns))
    fig_h = max(6.0, 1.0 + 0.55 * len(matrix.index))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot_values = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f2f2f2")
    im = ax.imshow(plot_values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=60, ha="right")
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_xlabel("Test domain")
    ax.set_ylabel("Train domain")
    ax.set_title(title)

    for row_idx, row_name in enumerate(matrix.index):
        for col_idx, col_name in enumerate(matrix.columns):
            value = matrix.loc[row_name, col_name]
            if pd.notna(value):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_pairwise_note(bundle: DatasetBundle, output_dir: Path) -> None:
    lines = [
        f"# Pairwise Train-to-Test Heatmaps for {bundle.display_name}",
        "",
        "- This view is stricter than the main LODO benchmark.",
        "- Each matrix cell trains on a single source domain and tests on a different holdout domain.",
        "- Rows are training domains; columns are test domains.",
        "- Under strict no-leakage ComBat, the holdout domain is never harmonized.",
        "- To make ComBat estimable, harmonization parameters are fitted on all non-holdout training domains, then the classifier is trained only on the selected row-domain.",
        f"- Requested harmonization: `{bundle.harmonization_method}`",
        f"- Harmonization applicable on current local input: `{bundle.harmonization_allowed}`",
        f"- Domain is real: `{bundle.domain_real}` ({bundle.domain_kind})",
        "",
        "This matrix is therefore closer to the low cross-site transfer setting seen in DASHI than the main multi-source holdout benchmark.",
    ]
    (output_dir / "pairwise_method_note.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _evaluate_pairwise_bundle(bundle: DatasetBundle, output_root: Path) -> None:
    dataset_dir = output_root / bundle.dataset_id / "pairwise_site_transfer"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _write_pairwise_note(bundle, dataset_dir)

    domains = sorted(bundle.data[bundle.domain_col].astype(str).unique().tolist())
    pipelines: list[tuple[str, bool]] = [("baseline", False)]
    if bundle.harmonization_allowed:
        pipelines.append(("harmonized_train_only", True))

    rows: list[dict[str, Any]] = []
    for test_domain in domains:
        train_df = bundle.data.loc[
            bundle.data[bundle.domain_col].astype(str) != test_domain
        ].copy()
        test_df = bundle.data.loc[
            bundle.data[bundle.domain_col].astype(str) == test_domain
        ].copy()
        y_test = test_df[bundle.label_col].to_numpy(dtype=int)

        for pipeline_name, do_harmonize in pipelines:
            train_global, test_global = _prepare_global_frames(
                bundle, train_df, test_df, do_harmonize
            )
            for train_domain in domains:
                if train_domain == test_domain:
                    continue
                source_mask = train_df[bundle.domain_col].astype(str) == train_domain
                if not bool(source_mask.any()):
                    continue
                source_y = train_df.loc[source_mask, bundle.label_col].to_numpy(
                    dtype=int
                )
                if len(np.unique(source_y)) < 2:
                    continue
                source_train_frame = train_global.loc[source_mask].copy()

                for model_name, model in _build_models().items():
                    try:
                        x_source, x_test, selected_features = (
                            _select_and_scale_source_specific(
                                bundle=bundle,
                                source_train_frame=source_train_frame,
                                source_y=source_y,
                                test_frame=test_global,
                            )
                        )
                        pred, _ = _fit_predict_model(
                            model_name, model, x_source, source_y, x_test
                        )
                        accuracy = float(accuracy_score(y_test, pred))
                        error = ""
                    except Exception as exc:
                        accuracy = float("nan")
                        selected_features = []
                        error = str(exc)

                    rows.append(
                        {
                            "dataset": bundle.dataset_id,
                            "pipeline": pipeline_name,
                            "model": model_name,
                            "train_domain": train_domain,
                            "test_domain": test_domain,
                            "n_train_source": int(source_mask.sum()),
                            "n_test_target": int(len(test_df)),
                            "n_features": int(len(selected_features)),
                            "accuracy": accuracy,
                            "error": error,
                        }
                    )

    results = (
        pd.DataFrame(rows)
        .sort_values(["pipeline", "model", "train_domain", "test_domain"])
        .reset_index(drop=True)
    )
    results.to_csv(dataset_dir / "pairwise_accuracy_long.csv", index=False)

    summary = (
        results.groupby(["pipeline", "model"], dropna=False)["accuracy"]
        .agg(
            off_diagonal_mean="mean",
            off_diagonal_std="std",
            off_diagonal_median="median",
        )
        .reset_index()
    )
    summary.to_csv(dataset_dir / "pairwise_accuracy_summary.csv", index=False)

    for pipeline_name in results["pipeline"].dropna().unique().tolist():
        subset_pipeline = results.loc[results["pipeline"] == pipeline_name].copy()
        for model_name in subset_pipeline["model"].dropna().unique().tolist():
            subset = subset_pipeline.loc[subset_pipeline["model"] == model_name].copy()
            matrix = subset.pivot(
                index="train_domain", columns="test_domain", values="accuracy"
            ).reindex(index=domains, columns=domains)
            stem = f"pairwise_accuracy_{pipeline_name}_{model_name}"
            matrix.to_csv(dataset_dir / f"{stem}.csv")
            _plot_heatmap(
                matrix=matrix,
                title=f"{bundle.display_name}: {pipeline_name} - {model_name}",
                output_path=dataset_dir / f"{stem}.png",
            )

        avg_matrix = (
            subset_pipeline.groupby(["train_domain", "test_domain"], dropna=False)[
                "accuracy"
            ]
            .mean()
            .reset_index()
            .pivot(index="train_domain", columns="test_domain", values="accuracy")
            .reindex(index=domains, columns=domains)
        )
        avg_stem = f"pairwise_accuracy_{pipeline_name}_mean_over_models"
        avg_matrix.to_csv(dataset_dir / f"{avg_stem}.csv")
        _plot_heatmap(
            matrix=avg_matrix,
            title=f"{bundle.display_name}: {pipeline_name} - mean over models",
            output_path=dataset_dir / f"{avg_stem}.png",
        )


def run_pairwise_heatmaps(dataset_ids: list[str], output_root: Path) -> None:
    bundles = load_requested_datasets(dataset_ids)
    for bundle in bundles:
        _evaluate_pairwise_bundle(bundle, output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pairwise train-domain to test-domain heatmaps"
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
    run_pairwise_heatmaps(args.datasets, Path(args.output_root))


if __name__ == "__main__":
    main()
