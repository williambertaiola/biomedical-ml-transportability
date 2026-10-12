"""Transportability experiment: unsupervised tcga."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans

from common_pipeline import (
    assert_local_module,
    configure_local_dashi_submodule,
    dataset_output_dir,
    save_plotly_figure,
    save_plotly_figure_list,
    write_text,
)
from dataset_loaders import prepare_tcga_dataset

LOCAL_UNSUP_PATH = Path(__file__).resolve().parent / "unsupervised_characterization"
TCGA_MIN_SOURCE_SAMPLES = 20
TCGA_MIN_CLASS_COUNT = 4
TCGA_UNSUP_TOP_GENES = 1500


def _sphere_surface(center: np.ndarray, radius: float, color: str) -> go.Surface:
    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    return go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.22,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        hoverinfo="skip",
        name="Sfera",
    )


def _cluster_3d_with_spheres(
    points: np.ndarray, point_labels: list[str], title: str, output_path: Path
) -> None:
    if points.shape[0] < 2:
        raise ValueError(f"Punti insufficienti per clustering: {title}")

    k = min(3, points.shape[0])
    model = KMeans(n_clusters=k, random_state=112, n_init=20)
    cluster_id = model.fit_predict(points)
    centers = model.cluster_centers_

    cluster_colors = ["#4c78a8", "#f58518", "#54a24b"]
    sphere_colors = ["#cfe8ff", "#ffe7cf", "#d9f5d6"]
    fig = go.Figure()

    for cluster_index in range(k):
        mask = cluster_id == cluster_index
        idx = np.where(mask)[0]
        cluster_points = points[mask]
        centroid = centers[cluster_index]

        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        radius = (
            float(np.percentile(dists, 95))
            if len(dists) > 1
            else float(dists.max() if len(dists) else 0.0)
        )
        if radius <= 0:
            radius = 0.02

        fig.add_trace(
            _sphere_surface(
                centroid, radius, sphere_colors[cluster_index % len(sphere_colors)]
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                z=cluster_points[:, 2],
                mode="markers+text",
                text=[point_labels[i] for i in idx],
                textposition="top center",
                name=f"Cluster {cluster_index + 1}",
                marker=dict(
                    size=5, color=cluster_colors[cluster_index % len(cluster_colors)]
                ),
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode="markers",
            name="Centroidi",
            marker=dict(size=10, symbol="diamond", color="black"),
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="D1", yaxis_title="D2", zaxis_title="D3"),
        template="plotly_white",
    )
    save_plotly_figure(fig, output_path)


def main() -> None:
    configure_local_dashi_submodule("unsupervised_characterization", LOCAL_UNSUP_PATH)

    from dashi.unsupervised_characterization.data_source_map.data_source_map import (
        estimate_conditional_data_source_map,
        estimate_multivariate_data_source_map,
        estimate_univariate_data_source_map,
    )
    from dashi.unsupervised_characterization.data_source_map.data_source_map_plotter import (
        plot_conditional_data_source_map,
        plot_multivariate_data_source_map,
        plot_univariate_data_source_map,
    )
    from dashi.unsupervised_characterization.variability_metrics.estimate_msv_metrics import (
        estimate_MSV_metrics,
    )
    from dashi.unsupervised_characterization.variability_metrics.plot_msv import (
        plot_MSV,
    )

    assert_local_module(estimate_univariate_data_source_map, LOCAL_UNSUP_PATH)
    assert_local_module(estimate_multivariate_data_source_map, LOCAL_UNSUP_PATH)
    assert_local_module(estimate_conditional_data_source_map, LOCAL_UNSUP_PATH)
    assert_local_module(estimate_MSV_metrics, LOCAL_UNSUP_PATH)

    output_dir = dataset_output_dir("tcga", "unsupervised")
    analysis_df, feature_columns, metadata = prepare_tcga_dataset(
        min_source_samples=TCGA_MIN_SOURCE_SAMPLES,
        require_both_binary_classes=True,
        min_binary_class_count=TCGA_MIN_CLASS_COUNT,
        global_top_variable_genes=TCGA_UNSUP_TOP_GENES,
    )
    first_feature = feature_columns[0]

    univariate_dsm = estimate_univariate_data_source_map(
        data=analysis_df[[first_feature, "source"]].copy(),
        source_column="source",
    )
    multivariate_dsm = estimate_multivariate_data_source_map(
        data=analysis_df[feature_columns + ["source"]].copy(),
        source_column_name="source",
        dimensions=2,
        dim_reduction="PCA",
        scale=True,
        scatter_plot=False,
    )
    conditional_dsm = estimate_conditional_data_source_map(
        data=analysis_df[feature_columns + ["source", "label"]].copy(),
        source_column_name="source",
        label_column_name="label",
        dimensions=2,
        dim_reduction="PCA",
        scale=True,
        scatter_plot=False,
    )

    save_plotly_figure(
        plot_univariate_data_source_map(univariate_dsm),
        output_dir / "data_source_map_univariate.html",
    )
    save_plotly_figure(
        plot_multivariate_data_source_map(multivariate_dsm),
        output_dir / "data_source_map_multivariate.html",
    )
    save_plotly_figure_list(
        plot_conditional_data_source_map(conditional_dsm),
        output_dir / "data_source_map_conditional",
    )

    msv_metrics = estimate_MSV_metrics(multivariate_dsm)
    save_plotly_figure(
        plot_MSV(msv_metrics=msv_metrics, dimensions=2), output_dir / "msv_plot_2d.html"
    )
    if len(msv_metrics.sources) >= 4:
        save_plotly_figure(
            plot_MSV(msv_metrics=msv_metrics, dimensions=3),
            output_dir / "msv_plot_3d.html",
        )
        _cluster_3d_with_spheres(
            points=msv_metrics.vertices[:, :3],
            point_labels=[str(source) for source in msv_metrics.sources],
            title="MSV 3D - KMeans (k=3) con Sfere",
            output_path=output_dir / "msv_plot_3d_kmeans_sfere.html",
        )

    pd.DataFrame({"GPD": [msv_metrics.GPD]}).to_csv(
        output_dir / "msv_gpd.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        {
            "source": msv_metrics.sources,
            "SPO": msv_metrics.SPO,
            "nBySource": msv_metrics.nBySource,
        }
    ).to_csv(output_dir / "msv_source_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "source": list(metadata["source_counts"].keys()),
            "n_samples": list(metadata["source_counts"].values()),
        }
    ).to_csv(output_dir / "source_counts.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "target": list(metadata["binary_target_counts"].keys()),
            "n_samples": list(metadata["binary_target_counts"].values()),
        }
    ).to_csv(output_dir / "binary_target_counts.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "sample_type": list(metadata["sample_type_counts"].keys()),
            "n_samples": list(metadata["sample_type_counts"].values()),
        }
    ).to_csv(output_dir / "sample_type_counts.tsv", sep="\t", index=False)

    summary = "\n".join(
        [
            "# TCGA-BRCA Unsupervised",
            "",
            f"- Input: `{metadata['input_path']}`",
            "- Preprocessing: log-CPM per campione, selezione globale dei 1500 geni a maggiore varianza.",
            f"- Samples analizzati: {metadata['n_samples']}",
            f"- Feature finali: {metadata['n_features']}",
            f"- Source reali (TSS) incluse: {len(metadata['source_counts'])}",
            f"- GPD: {float(msv_metrics.GPD):.6f}",
            "- Analisi temporale rimossa: nel file non c'è una cronologia reale utilizzabile per DTM/IGT.",
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
        ]
    )
    write_text(summary, output_dir / "analysis_summary.md")


if __name__ == "__main__":
    main()
