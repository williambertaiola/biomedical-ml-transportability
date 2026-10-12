"""Transportability experiment: supervised tcga."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common_pipeline import (
    assert_local_module,
    configure_local_dashi_submodule,
    dataset_output_dir,
    save_plotly_figure,
    write_text,
)
from dataset_loaders import prepare_tcga_dataset

LOCAL_SUP_PATH = Path(__file__).resolve().parent / "supervised_characterization"
TCGA_MIN_SOURCE_SAMPLES = 20
TCGA_MIN_CLASS_COUNT = 4
TCGA_GLOBAL_GENE_POOL = 5000
TCGA_TRAIN_TOP_FEATURES = 500


def main() -> None:
    configure_local_dashi_submodule("supervised_characterization", LOCAL_SUP_PATH)

    from dashi.supervised_characterization.arrange_metrics import (
        arrange_performance_metrics,
    )
    from dashi.supervised_characterization.estimate_models import (
        estimate_multibatch_models,
    )
    from dashi.supervised_characterization.plot_performance import (
        plot_multibatch_performance,
    )

    assert_local_module(estimate_multibatch_models, LOCAL_SUP_PATH)
    assert_local_module(plot_multibatch_performance, LOCAL_SUP_PATH)
    assert_local_module(arrange_performance_metrics, LOCAL_SUP_PATH)

    output_dir = dataset_output_dir("tcga", "supervised")
    model_df, feature_pool, metadata = prepare_tcga_dataset(
        min_source_samples=TCGA_MIN_SOURCE_SAMPLES,
        require_both_binary_classes=True,
        min_binary_class_count=TCGA_MIN_CLASS_COUNT,
        global_top_variable_genes=TCGA_GLOBAL_GENE_POOL,
    )

    metrics = estimate_multibatch_models(
        data=model_df,
        inputs_numerical_column_names=feature_pool,
        inputs_categorical_column_names=None,
        output_regression_column_name=None,
        output_classification_column_name="target",
        date_column_name=None,
        period=None,
        source_column_name="source",
        learning_strategy="from_scratch",
        model_type="histogram_gradient_boosting",
        feature_selection_top_k=TCGA_TRAIN_TOP_FEATURES,
        split_random_seed=42,
        test_ratio=0.2,
        shuffle_within_batch=True,
        stratify_classification_split=True,
    )

    metrics_rows: list[dict] = []
    for (train_batches, test_batch, split_name), metric_values in metrics.items():
        train_repr = "|".join(str(batch) for batch in train_batches)
        for metric_name, metric_value in metric_values.items():
            metrics_rows.append(
                {
                    "train_batches": train_repr,
                    "test_batch": str(test_batch),
                    "split": split_name,
                    "metric": metric_name,
                    "value": metric_value,
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(
        output_dir / "supervised_multibatch_metrics_long.tsv", sep="\t", index=False
    )

    available_metrics = sorted(
        {metric for values in metrics.values() for metric in values.keys()}
    )
    pd.DataFrame({"metric": available_metrics}).to_csv(
        output_dir / "supervised_available_metrics.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame({"feature": feature_pool}).to_csv(
        output_dir / "supervised_feature_pool.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        {
            "source": list(metadata["source_counts"].keys()),
            "n_samples": list(metadata["source_counts"].values()),
        }
    ).to_csv(output_dir / "supervised_source_counts.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "target": list(metadata["binary_target_counts"].keys()),
            "n_samples": list(metadata["binary_target_counts"].values()),
        }
    ).to_csv(output_dir / "supervised_target_counts.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "sample_type": list(metadata["sample_type_counts"].keys()),
            "n_samples": list(metadata["sample_type_counts"].values()),
        }
    ).to_csv(output_dir / "supervised_sample_type_counts.tsv", sep="\t", index=False)

    metrics_to_plot = [
        "ACCURACY",
        "F1-SCORE_MACRO",
        "RECALL_MACRO",
        "PRECISION_MACRO",
        "AUC_MACRO",
        "PR-AUC_MACRO",
        "LOGLOSS",
    ]

    metric_summaries: list[str] = []
    for metric_name in metrics_to_plot:
        if metric_name not in available_metrics:
            continue

        performance_frame = arrange_performance_metrics(
            metrics=metrics, metric_name=metric_name
        )
        performance_frame.to_csv(
            output_dir
            / f"supervised_{metric_name.lower().replace('-', '_')}_matrix.tsv",
            sep="\t",
            index=True,
        )

        figure = plot_multibatch_performance(metrics=metrics, metric_name=metric_name)
        save_plotly_figure(
            figure,
            output_dir
            / f"supervised_{metric_name.lower().replace('-', '_')}_heatmap.html",
        )

        values = pd.to_numeric(performance_frame.stack(), errors="coerce")
        metric_summaries.append(
            f"- {metric_name}: mean={values.mean():.4f}, median={values.median():.4f}, min={values.min():.4f}, max={values.max():.4f}"
        )

    majority_baseline = model_df["target"].value_counts(normalize=True).max()
    summary = "\n".join(
        [
            "# TCGA-BRCA Supervised",
            "",
            f"- Input: `{metadata['input_path']}`",
            "- Source corrette: TSS TCGA reali.",
            "- Task corretto: classificazione binaria `tumor` vs `normal`.",
            f"- Samples analizzati: {metadata['n_samples']}",
            f"- Pool globale di geni: {metadata['n_features']}",
            f"- Top feature per training batch: {TCGA_TRAIN_TOP_FEATURES}",
            f"- Baseline maggioritaria globale: {majority_baseline:.4f}",
            "",
            "## Source mantenute",
            *(
                f"- {source}: {count}"
                for source, count in metadata["source_counts"].items()
            ),
            "",
            "## Target binario",
            *(
                f"- {target}: {count}"
                for target, count in metadata["binary_target_counts"].items()
            ),
            "",
            "## Sample type originali",
            *(
                f"- {sample_type}: {count}"
                for sample_type, count in metadata["sample_type_counts"].items()
            ),
            "",
            "## Sintesi metriche",
            *metric_summaries,
        ]
    )
    write_text(summary, output_dir / "analysis_summary.md")


if __name__ == "__main__":
    main()
