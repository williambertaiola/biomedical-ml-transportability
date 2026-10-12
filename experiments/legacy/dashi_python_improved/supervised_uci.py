"""Transportability experiment: supervised uci."""

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
from dataset_loaders import UCI_SOURCE_COLUMN, UCI_TARGET_COLUMN, prepare_uci_supervised

LOCAL_SUP_PATH = Path(__file__).resolve().parent / "supervised_characterization"


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

    output_dir = dataset_output_dir("uci", "supervised")
    model_df, numeric_features, metadata = prepare_uci_supervised()

    metrics = estimate_multibatch_models(
        data=model_df,
        inputs_numerical_column_names=numeric_features,
        inputs_categorical_column_names=None,
        output_regression_column_name=None,
        output_classification_column_name=UCI_TARGET_COLUMN,
        date_column_name=None,
        period=None,
        source_column_name=UCI_SOURCE_COLUMN,
        learning_strategy="from_scratch",
        model_type="histogram_gradient_boosting",
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

    pd.DataFrame(
        {
            "source": list(metadata["source_counts"].keys()),
            "n_samples": list(metadata["source_counts"].values()),
        }
    ).to_csv(output_dir / "supervised_source_counts.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "target": list(metadata["target_counts"].keys()),
            "n_samples": list(metadata["target_counts"].values()),
        }
    ).to_csv(output_dir / "supervised_target_counts.tsv", sep="\t", index=False)

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

    majority_baseline = model_df[UCI_TARGET_COLUMN].value_counts(normalize=True).max()
    summary = "\n".join(
        [
            "# UCI Supervised",
            "",
            f"- Input: `{metadata['input_path']}`",
            f"- Samples analizzati: {metadata['n_samples']}",
            f"- Feature numeriche: {metadata['n_features']}",
            f"- Baseline maggioritaria globale: {majority_baseline:.4f}",
            "",
            "## Conteggi per source",
            *(
                f"- {source}: {count}"
                for source, count in metadata["source_counts"].items()
            ),
            "",
            "## Conteggi target",
            *(
                f"- {target}: {count}"
                for target, count in metadata["target_counts"].items()
            ),
            "",
            "## Sintesi metriche",
            *metric_summaries,
        ]
    )
    write_text(summary, output_dir / "analysis_summary.md")


if __name__ == "__main__":
    main()
