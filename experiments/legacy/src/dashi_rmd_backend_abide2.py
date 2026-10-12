"""Transportability experiment: dashi rmd backend abide2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import AgglomerativeClustering

import dashi

QAP_FEATURE_CANDIDATES = [
    "SNR",
    "EFC",
    "Smoothness_of_Voxels",
    "FBER",
    "GSR",
    "mFD",
    "PercentFD_greater_than_0.20",
    "NumFD_greater_than_0.20",
    "DVARS",
    "Outliers",
    "GCORR",
    "Median_Distance_Index",
]


def _select_cluster_count(n_points: int) -> int:
    # Heuristic for source-level MSV points:
    # keep at least 2 clusters when possible, avoid over-fragmentation for small N.
    if n_points <= 2:
        return 1
    if n_points <= 6:
        return 2
    return 3


def _cluster_circle_traces(
    center: np.ndarray,
    radius: float,
    color: str,
    name: str,
) -> list[go.Scatter3d]:
    theta = np.linspace(0, 2 * np.pi, 120)
    cx, cy, cz = center

    # Three orthogonal circles (XY, XZ, YZ) to make cluster envelopes clear in 3D.
    circles = [
        (
            cx + radius * np.cos(theta),
            cy + radius * np.sin(theta),
            np.full_like(theta, cz),
        ),
        (
            cx + radius * np.cos(theta),
            np.full_like(theta, cy),
            cz + radius * np.sin(theta),
        ),
        (
            np.full_like(theta, cx),
            cy + radius * np.cos(theta),
            cz + radius * np.sin(theta),
        ),
    ]
    traces: list[go.Scatter3d] = []
    for i, (x, y, z) in enumerate(circles):
        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color=color, width=6 if i == 0 else 4, dash="solid"),
                name=name,
                showlegend=(i == 0),
                hoverinfo="skip",
                opacity=0.9 if i == 0 else 0.75,
            )
        )
    return traces


def build_msv_cluster_figure(msv: Any) -> tuple[go.Figure | None, pd.DataFrame | None]:
    vertices = np.asarray(getattr(msv, "vertices", []), dtype=float)
    sources = [str(s) for s in np.asarray(getattr(msv, "sources", []))]
    spo = np.asarray(getattr(msv, "SPO", []), dtype=float)
    n_by_source = np.asarray(getattr(msv, "nBySource", []), dtype=float)

    if (
        vertices.ndim != 2
        or vertices.shape[0] == 0
        or len(sources) != vertices.shape[0]
    ):
        return None, None

    if vertices.shape[1] < 3:
        padded = np.zeros((vertices.shape[0], 3), dtype=float)
        padded[:, : vertices.shape[1]] = vertices
        points = padded
    else:
        points = vertices[:, :3]

    n_clusters = _select_cluster_count(points.shape[0])
    if n_clusters <= 1:
        labels = np.zeros(points.shape[0], dtype=int)
    else:
        labels = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward"
        ).fit_predict(points)

    if n_by_source.size != points.shape[0]:
        n_by_source = np.ones(points.shape[0], dtype=float)
    n_norm = n_by_source / max(float(np.nanmax(n_by_source)), 1.0)
    marker_sizes = 8 + 20 * n_norm

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers+text",
            text=sources,
            textposition="top center",
            marker=dict(
                size=marker_sizes,
                color=spo,
                colorscale="RdYlBu",
                reversescale=True,
                opacity=0.9,
                colorbar=dict(title="SPO"),
                line=dict(width=1, color="white"),
            ),
            name="Sources",
            showlegend=False,
            hovertemplate=(
                "<b>%{text}</b><br>D1: %{x:.3f}<br>D2: %{y:.3f}<br>D3: %{z:.3f}"
                "<br>SPO: %{marker.color:.3f}<extra></extra>"
            ),
        )
    )

    cluster_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    cluster_rows: list[dict[str, Any]] = []
    for cluster_id in sorted(np.unique(labels)):
        idx = np.where(labels == cluster_id)[0]
        pts = points[idx]
        cluster_name = f"Cluster {int(cluster_id) + 1}"
        color = cluster_palette[int(cluster_id) % len(cluster_palette)]

        center = pts.mean(axis=0)
        if pts.shape[0] >= 2:
            radius = max(
                float(np.max(np.linalg.norm(pts - center, axis=1))) * 1.35, 0.06
            )
        else:
            radius = 0.07
        for tr in _cluster_circle_traces(center, radius, color, cluster_name):
            fig.add_trace(tr)

        for i in idx:
            cluster_rows.append(
                {
                    "source": sources[i],
                    "cluster": int(cluster_id) + 1,
                    "D1": float(points[i, 0]),
                    "D2": float(points[i, 1]),
                    "D3": float(points[i, 2]),
                    "SPO": float(spo[i]) if i < len(spo) else np.nan,
                    "n_samples": float(n_by_source[i])
                    if i < len(n_by_source)
                    else np.nan,
                }
            )

    fig.update_layout(
        title=f"MSV 3D plot with clustered circles (k={n_clusters})",
        template="plotly_white",
        scene=dict(
            xaxis=dict(title="D1"),
            yaxis=dict(title="D2"),
            zaxis=dict(title="D3"),
        ),
        legend_title="Clusters",
        margin=dict(l=0, r=0, b=0, t=35),
    )

    cluster_table = (
        pd.DataFrame(cluster_rows)
        .sort_values(["cluster", "source"])
        .reset_index(drop=True)
    )
    return fig, cluster_table


def _detect_real_temporal_columns(df: pd.DataFrame) -> list[str]:
    # Only treat a column as "real temporal" if it can be parsed as datetime
    # and carries non-trivial temporal variation.
    temporal_keywords = ("date", "time", "timestamp", "year", "visit")
    candidate_cols = [
        c for c in df.columns if any(k in c.lower() for k in temporal_keywords)
    ]

    real_cols: list[str] = []
    for col in candidate_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() >= 10 and parsed.nunique(dropna=True) > 1:
            real_cols.append(col)
    return real_cols


def prepare_abide2_data(
    data_root: str | Path,
) -> tuple[
    pd.DataFrame,
    list[str],
    dict[str, pd.DataFrame],
    pd.DataFrame,
    list[str],
    dict[str, Any],
]:
    notes: list[str] = []
    data_root = Path(data_root)
    functional_path = data_root / "ABIDEII_MRI_Quality_Metrics" / "functional_qap.csv"

    if not functional_path.exists():
        raise FileNotFoundError(f"Missing required file: {functional_path}")

    df = pd.read_csv(functional_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not c.lower().startswith("unnamed")]]

    real_temporal_columns = _detect_real_temporal_columns(df)
    temporal_metadata: dict[str, Any] = {
        "real_temporal_available": bool(real_temporal_columns),
        "real_temporal_columns": real_temporal_columns,
    }

    required_cols = {"Site_ID", "Sub_ID"}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        raise ValueError(
            f"Missing required columns in functional_qap.csv: {sorted(missing_required)}"
        )

    if "Unique_ID" in df.columns:
        sample_id = df["Unique_ID"].astype(str).str.strip()
    else:
        sample_id = (
            df["Site_ID"].astype(str).str.strip()
            + "_"
            + df["Sub_ID"].astype(str).str.strip()
            + "_"
            + np.arange(len(df)).astype(str)
        )

    df["sample_id"] = sample_id
    df["source"] = df["Site_ID"].astype(str).str.strip()
    df = df[
        df["source"].notna() & (df["source"] != "") & (df["source"] != "nan")
    ].copy()

    feature_cols = [c for c in QAP_FEATURE_CANDIDATES if c in df.columns]
    if not feature_cols:
        raise ValueError("No expected QAP feature columns were found.")

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_pct = (
        (df[feature_cols].isna().mean() * 100)
        .rename("missing_pct")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    for col in feature_cols:
        median_value = df[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        df[col] = df[col].fillna(median_value).astype(float)

    if "mFD" in df.columns:
        motion_col = "mFD"
    elif "PercentFD_greater_than_0.20" in df.columns:
        motion_col = "PercentFD_greater_than_0.20"
    else:
        motion_col = feature_cols[0]
        notes.append(
            "Neither `mFD` nor `PercentFD_greater_than_0.20` was available; target proxy built from the first numeric feature."
        )

    motion_values = pd.to_numeric(df[motion_col], errors="coerce")
    motion_threshold = (
        float(motion_values.median())
        if motion_values.notna().any()
        else float(df[feature_cols[0]].median())
    )
    df["target"] = (motion_values.fillna(motion_threshold) >= motion_threshold).astype(
        int
    )

    if df["target"].nunique() < 2:
        # Robust fallback if the threshold creates a degenerate target.
        rank = df[feature_cols[0]].rank(method="average")
        df["target"] = (rank >= rank.median()).astype(int)
        notes.append(
            "Initial proxy target was degenerate; rebuilt target with a rank-based split."
        )

    source_counts = (
        df.groupby("source")["target"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={0: "target_0", 1: "target_1"})
    )
    source_counts["n_samples"] = source_counts.sum(axis=1)

    min_site_n = 30
    min_class_n = 5
    eligible_sources = source_counts[
        (source_counts["n_samples"] >= min_site_n)
        & (source_counts.get("target_0", 0) >= min_class_n)
        & (source_counts.get("target_1", 0) >= min_class_n)
    ].index.tolist()

    if len(eligible_sources) < 2:
        eligible_sources = (
            source_counts.sort_values("n_samples", ascending=False)
            .head(4)
            .index.tolist()
        )
        notes.append(
            "Insufficient balanced sources for strict filtering; fallback to top sources by sample count."
        )

    df = df[df["source"].isin(eligible_sources)].copy()
    df = df.sort_values(["source", "sample_id"]).reset_index(drop=True)

    final_df = df[["sample_id", "source", *feature_cols, "target"]].copy()
    if temporal_metadata["real_temporal_available"]:
        chosen_col = real_temporal_columns[0]
        final_df["real_temporal_date"] = pd.to_datetime(df[chosen_col], errors="coerce")

    summary_shape = pd.DataFrame(
        [
            {
                "dataset": "abide2_functional_qap_for_dashi",
                "n_rows": int(final_df.shape[0]),
                "n_columns": int(final_df.shape[1]),
            }
        ]
    )
    summary_samples = (
        final_df["source"]
        .value_counts()
        .rename_axis("source")
        .reset_index(name="n_samples")
        .sort_values("source")
    )
    summary_target = (
        final_df.groupby("source")["target"]
        .value_counts()
        .rename("n_samples")
        .reset_index()
    )
    summary_target["pct_within_source"] = (
        summary_target["n_samples"]
        / summary_target.groupby("source")["n_samples"].transform("sum")
        * 100
    )

    notes.append(
        f"Proxy target defined as `high_motion` from {motion_col} using median threshold={motion_threshold:.6f}."
    )

    summaries = {
        "shape": summary_shape,
        "samples": summary_samples,
        "target": summary_target,
        "missing_pct": missing_pct,
    }
    return (
        final_df,
        feature_cols,
        summaries,
        source_counts.reset_index(),
        notes,
        temporal_metadata,
    )


def generate_dashi_results(data_root: str | Path) -> dict[str, Any]:
    result: dict[str, Any] = {"errors": {}, "notes": []}

    data, feature_cols, summaries, source_balance, prep_notes, temporal_metadata = (
        prepare_abide2_data(data_root)
    )
    result["summary_shape"] = summaries["shape"]
    result["summary_samples"] = summaries["samples"]
    result["summary_target"] = summaries["target"]
    result["missing_pct"] = summaries["missing_pct"]
    result["source_balance"] = source_balance
    result["feature_cols"] = feature_cols
    result["temporal_metadata"] = temporal_metadata
    result["notes"].extend(prep_notes)

    # Data Source Maps + MSV
    try:
        source_df = data[["source", *feature_cols]].copy()
        source_df["source"] = source_df["source"].astype("object")

        source_map = dashi.estimate_multivariate_data_source_map(
            data=source_df,
            source_column_name="source",
            dimensions=2,
            dim_reduction="PCA",
            kde_resolution=20,
            scale=True,
            scatter_plot=False,
            verbose=False,
        )
        msv = dashi.estimate_MSV_metrics(source_map)

        source_counts = (
            source_df["source"]
            .value_counts()
            .reindex(msv.sources)
            .fillna(0)
            .to_numpy(dtype=float)
        )
        n_by_source = np.asarray(msv.nBySource, dtype=float)
        if (
            n_by_source.size == 0
            or np.isnan(n_by_source).any()
            or np.allclose(n_by_source, 0)
        ):
            msv.nBySource = source_counts

        result["fig_source_map"] = dashi.plot_multivariate_data_source_map(
            source_map, absolute=False
        )
        result["fig_msv"] = dashi.plot_MSV(msv, dimensions=3)
        # Secondary MSV visualization in the same 3D space with cluster envelopes.
        fig_msv_clustered, msv_cluster_table = build_msv_cluster_figure(msv)
        result["fig_msv_clustered"] = fig_msv_clustered
        result["msv_cluster_table"] = msv_cluster_table
        result["msv_table"] = pd.DataFrame(
            {
                "source": [str(s) for s in np.asarray(msv.sources)],
                "SPO": np.asarray(msv.SPO, dtype=float),
                "n_samples": source_counts,
                "GPD": float(msv.GPD) if msv.GPD is not None else np.nan,
            }
        )
    except Exception as exc:  # pragma: no cover
        result["errors"]["source_msv"] = str(exc)
        result["fig_source_map"] = None
        result["fig_msv"] = None
        result["fig_msv_clustered"] = None

    # Temporal map + IGT (use real temporal variable if available; otherwise synthetic index)
    try:
        temporal_df = data[feature_cols].copy()
        if (
            temporal_metadata.get("real_temporal_available", False)
            and "real_temporal_date" in data.columns
        ):
            valid_real = pd.to_datetime(data["real_temporal_date"], errors="coerce")
            if valid_real.notna().sum() >= 2 and valid_real.nunique(dropna=True) > 1:
                temporal_df["real_temporal_date"] = valid_real
                date_column_name = "real_temporal_date"
                result["notes"].append(
                    "Temporal outputs use real temporal variable(s): "
                    + ", ".join(temporal_metadata.get("real_temporal_columns", []))
                    + "."
                )
            else:
                temporal_df["synthetic_batch_index"] = pd.Timestamp(
                    "2010-01-01"
                ) + pd.to_timedelta(np.arange(len(temporal_df)), unit="D")
                date_column_name = "synthetic_batch_index"
                result["notes"].append(
                    "Temporal candidate column(s) detected but not usable as real timeline; synthetic batch index used."
                )
        else:
            temporal_df["synthetic_batch_index"] = pd.Timestamp(
                "2010-01-01"
            ) + pd.to_timedelta(np.arange(len(temporal_df)), unit="D")
            date_column_name = "synthetic_batch_index"
            result["notes"].append(
                "No real timestamp/date variable is available in ABIDE II functional QAP; temporal outputs use a synthetic batch index from sample order."
            )

        temporal_map = dashi.estimate_multivariate_data_temporal_map(
            data=temporal_df,
            date_column_name=date_column_name,
            dimensions=3,
            period="month",
            dim_reduction="PCA",
            kde_resolution=20,
            scale=True,
            scatter_plot=False,
            verbose=False,
        )
        igt = dashi.estimate_igt_projection(temporal_map, dimensions=3)

        result["fig_temporal_heatmap"] = dashi.plot_multivariate_data_temporal_map(
            temporal_map, absolute=False, log_transform=False
        )
        result["fig_igt"] = dashi.plot_IGT_projection(
            igt, dimensions=3, trajectory=True
        )
    except Exception as exc:  # pragma: no cover
        result["errors"]["temporal"] = str(exc)
        result["fig_temporal_heatmap"] = None
        result["fig_igt"] = None

    # Supervised heatmap
    try:
        supervised_df = data[["source", *feature_cols, "target"]].copy()
        supervised_df["source"] = supervised_df["source"].astype("object")
        supervised_df["target"] = supervised_df["target"].astype(int)

        metrics = dashi.estimate_multibatch_models(
            data=supervised_df,
            inputs_numerical_column_names=feature_cols,
            inputs_categorical_column_names=None,
            output_classification_column_name="target",
            output_regression_column_name=None,
            source_column_name="source",
            date_column_name=None,
            period=None,
            learning_strategy="from_scratch",
            model_type="histogram_gradient_boosting",
        )

        all_metric_names = sorted(
            {
                metric
                for metric_dict in metrics.values()
                for metric in metric_dict.keys()
            }
        )
        if "AUC" in all_metric_names:
            metric_name = "AUC"
        elif "ACCURACY" in all_metric_names:
            metric_name = "ACCURACY"
        else:
            metric_name = all_metric_names[0]

        result["fig_performance_heatmap"] = dashi.plot_multibatch_performance(
            metrics=metrics,
            metric_name=metric_name,
        )
        result["performance_matrix"] = dashi.arrange_performance_metrics(
            metrics=metrics, metric_name=metric_name
        )
        result["metric_name"] = metric_name
    except Exception as exc:  # pragma: no cover
        result["errors"]["supervised"] = str(exc)
        result["fig_performance_heatmap"] = None

    return result


DASHI_RESULTS: dict[str, Any] | None = None


def dashi_generate_and_store(data_root: str | Path) -> bool:
    global DASHI_RESULTS
    DASHI_RESULTS = generate_dashi_results(data_root)
    return True
