"""Transportability experiment: dashi rmd backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import prince
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import AgglomerativeClustering

import dashi

UCI_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

NUMERICAL_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def _read_source(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path, header=None, names=UCI_COLUMNS, na_values="?")
    df["source"] = source_name
    return df


def _safe_mode(series: pd.Series, fallback: float = 0.0) -> float:
    clean = series.dropna()
    if clean.empty:
        return float(fallback)
    mode_values = clean.mode(dropna=True)
    if mode_values.empty:
        return float(clean.iloc[0])
    return float(mode_values.iloc[0])


def _cluster_points(points: np.ndarray, n_clusters: int = 2) -> np.ndarray:
    if points.shape[0] <= 1:
        return np.zeros(points.shape[0], dtype=int)
    k = int(max(1, min(n_clusters, points.shape[0])))
    if k == 1:
        return np.zeros(points.shape[0], dtype=int)
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return model.fit_predict(points)


def _cluster_sphere_surface(
    center: np.ndarray,
    radius: float,
    color: str,
    name: str,
) -> go.Surface:
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 18)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.16,
        showscale=False,
        name=name,
        showlegend=True,
        surfacecolor=np.zeros_like(x),
        colorscale=[[0.0, color], [1.0, color]],
        hoverinfo="skip",
    )


def build_msv_cluster_figure(
    msv: Any,
    title: str,
) -> tuple[go.Figure | None, pd.DataFrame | None]:
    vertices = np.asarray(getattr(msv, "vertices", []), dtype=float)
    sources = [str(s) for s in np.asarray(getattr(msv, "sources", []))]
    spo = np.asarray(getattr(msv, "SPO", []), dtype=float)
    n_by_source = np.asarray(getattr(msv, "nBySource", []), dtype=float)

    if (
        vertices.ndim != 2
        or vertices.shape[0] == 0
        or vertices.shape[0] != len(sources)
    ):
        return None, None

    if vertices.shape[1] < 3:
        padded = np.zeros((vertices.shape[0], 3), dtype=float)
        padded[:, : vertices.shape[1]] = vertices
        points = padded
    else:
        points = vertices[:, :3]

    labels = _cluster_points(points, n_clusters=2)

    if n_by_source.size != points.shape[0]:
        n_by_source = np.ones(points.shape[0], dtype=float)
    n_norm = n_by_source / max(float(np.nanmax(n_by_source)), 1.0)
    sizes = 8 + 20 * n_norm

    cluster_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

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
                size=sizes,
                color=spo,
                colorscale="RdYlBu",
                reversescale=True,
                opacity=0.9,
                colorbar=dict(title="SPO"),
                line=dict(width=1, color="white"),
            ),
            name="Sources",
            hovertemplate=(
                "<b>%{text}</b><br>D1: %{x:.3f}<br>D2: %{y:.3f}<br>D3: %{z:.3f}"
                "<br>SPO: %{marker.color:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    cluster_rows: list[dict[str, Any]] = []
    for cluster_id in sorted(np.unique(labels)):
        idx = np.where(labels == cluster_id)[0]
        pts = points[idx]
        color = cluster_colors[int(cluster_id) % len(cluster_colors)]
        cluster_name = f"Cluster {int(cluster_id) + 1}"

        if pts.shape[0] >= 4:
            try:
                hull = ConvexHull(pts)
                simplices = hull.simplices
                fig.add_trace(
                    go.Mesh3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        i=simplices[:, 0],
                        j=simplices[:, 1],
                        k=simplices[:, 2],
                        opacity=0.18,
                        color=color,
                        name=cluster_name,
                        hoverinfo="skip",
                    )
                )
            except (QhullError, ValueError):
                center = pts.mean(axis=0)
                radius = float(np.max(np.linalg.norm(pts - center, axis=1)))
                radius = max(radius * 1.35, 0.06)
                fig.add_trace(
                    _cluster_sphere_surface(center, radius, color, cluster_name)
                )
        elif pts.shape[0] >= 2:
            center = pts.mean(axis=0)
            radius = float(np.max(np.linalg.norm(pts - center, axis=1)))
            radius = max(radius * 1.45, 0.06)
            fig.add_trace(_cluster_sphere_surface(center, radius, color, cluster_name))
        else:
            center = pts[0]
            fig.add_trace(_cluster_sphere_surface(center, 0.07, color, cluster_name))

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
        title=title,
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


def prepare_uci_heart_data(
    data_dir: str | Path = "data",
) -> tuple[pd.DataFrame, list[str], dict[str, pd.DataFrame]]:
    data_dir = Path(data_dir)
    sources = [
        ("Cleveland", "processed.cleveland.data"),
        ("Hungary", "processed.hungarian.data"),
        ("Switzerland", "processed.switzerland.data"),
        ("VA Long Beach", "processed.va.data"),
    ]

    frames = [
        _read_source(data_dir / filename, source_name)
        for source_name, filename in sources
    ]
    df = pd.concat(frames, ignore_index=True)

    df["sample_id"] = [f"S{i:05d}" for i in range(1, len(df) + 1)]
    df["target"] = np.where(pd.to_numeric(df["num"], errors="coerce") > 0, 1, 0).astype(
        int
    )

    feature_cols = [c for c in UCI_COLUMNS if c != "num"]
    for col in feature_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        med = values.median()
        if pd.isna(med):
            med = 0.0
        df[col] = values.fillna(med).astype(float)

    df["source"] = df["source"].astype("object")
    final_df = df[["sample_id", "source", *feature_cols, "target"]].copy()

    summary_shape = pd.DataFrame(
        [
            {
                "dataset": "prepared_for_dashi",
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

    summaries = {
        "shape": summary_shape,
        "samples": summary_samples,
        "target": summary_target,
    }
    return final_df, feature_cols, summaries


def prepare_uci_heart_data_imputed(
    data_dir: str | Path = "data",
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
    list[str],
    dict[str, pd.DataFrame],
    dict[str, Any],
]:
    data_dir = Path(data_dir)
    sources = [
        ("Cleveland", "processed.cleveland.data"),
        ("Hungary", "processed.hungarian.data"),
        ("Switzerland", "processed.switzerland.data"),
        ("VA Long Beach", "processed.va.data"),
    ]

    frames = [
        _read_source(data_dir / filename, source_name)
        for source_name, filename in sources
    ]
    df = pd.concat(frames, ignore_index=True)

    df["sample_id"] = [f"S{i:05d}" for i in range(1, len(df) + 1)]

    target_num = pd.to_numeric(df["num"], errors="coerce")
    valid_target = target_num.notna()
    removed_missing_target = int((~valid_target).sum())
    df = df.loc[valid_target].copy()
    target_num = target_num.loc[valid_target]
    df["target"] = (target_num > 0).astype(int)

    numerical_cols = [c for c in NUMERICAL_COLUMNS if c in df.columns]
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    feature_cols = [
        c
        for c in UCI_COLUMNS
        if c != "num" and c in (numerical_cols + categorical_cols)
    ]

    missing_before = (
        df[feature_cols]
        .isna()
        .mean()
        .mul(100)
        .rename("missing_pct")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )

    imputation_rows: list[dict[str, Any]] = []
    for col in numerical_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        n_missing = int(values.isna().sum())
        fill_value = values.mean()
        if pd.isna(fill_value):
            fill_value = values.median()
        if pd.isna(fill_value):
            fill_value = 0.0
        df[col] = values.fillna(fill_value).astype(float)
        imputation_rows.append(
            {
                "feature": col,
                "variable_type": "numerical",
                "missing_before": n_missing,
                "strategy": "mean",
                "fill_value": float(fill_value),
            }
        )

    for col in categorical_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        n_missing = int(values.isna().sum())
        fill_value = _safe_mode(values, fallback=0.0)
        df[col] = values.fillna(fill_value).round().astype("Int64").astype("object")
        imputation_rows.append(
            {
                "feature": col,
                "variable_type": "categorical",
                "missing_before": n_missing,
                "strategy": "mode",
                "fill_value": float(fill_value),
            }
        )

    df["source"] = df["source"].astype("object")
    final_df = df[["sample_id", "source", *feature_cols, "target"]].copy()

    summary_shape = pd.DataFrame(
        [
            {
                "dataset": "prepared_for_dashi_imputed",
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

    summaries = {
        "shape": summary_shape,
        "samples": summary_samples,
        "target": summary_target,
        "missing_before_imputation": missing_before,
        "imputation_table": pd.DataFrame(imputation_rows)
        .sort_values("feature")
        .reset_index(drop=True),
    }
    meta = {
        "removed_rows_missing_target": removed_missing_target,
        "numerical_features": numerical_cols,
        "categorical_features": categorical_cols,
    }
    return final_df, feature_cols, numerical_cols, categorical_cols, summaries, meta


def build_famd_figure(
    data: pd.DataFrame, feature_cols: list[str]
) -> tuple[go.Figure, pd.DataFrame]:
    famd_input = data[feature_cols].copy()
    for col in CATEGORICAL_COLUMNS:
        if col in famd_input.columns:
            famd_input[col] = famd_input[col].astype("category")

    famd = prince.FAMD(n_components=3, random_state=42)
    coords = famd.fit_transform(famd_input)
    coords.columns = ["FAMD1", "FAMD2", "FAMD3"]

    embed_df = pd.concat(
        [
            data[["sample_id", "source"]].reset_index(drop=True),
            coords.reset_index(drop=True),
        ],
        axis=1,
    )

    fig = go.Figure()
    for source in sorted(embed_df["source"].astype(str).unique()):
        sub = embed_df[embed_df["source"].astype(str) == source]
        fig.add_trace(
            go.Scatter3d(
                x=sub["FAMD1"],
                y=sub["FAMD2"],
                z=sub["FAMD3"],
                mode="markers",
                name=str(source),
                marker=dict(size=4, opacity=0.7),
                hovertemplate="<b>%{text}</b><br>FAMD1: %{x:.3f}<br>FAMD2: %{y:.3f}<br>FAMD3: %{z:.3f}<extra></extra>",
                text=sub["sample_id"],
            )
        )

    fig.update_layout(
        title="FAMD embedding (imputed dataset)",
        template="plotly_white",
        scene=dict(
            xaxis=dict(title="FAMD1"),
            yaxis=dict(title="FAMD2"),
            zaxis=dict(title="FAMD3"),
        ),
        legend_title="Source",
        margin=dict(l=0, r=0, b=0, t=35),
    )
    return fig, embed_df


def _resolve_temporal_metadata(data: pd.DataFrame) -> dict[str, Any]:
    temporal_keywords = ("date", "time", "year", "visit", "timestamp", "follow")
    real_temporal_columns = [
        c for c in data.columns if any(k in c.lower() for k in temporal_keywords)
    ]
    return {
        "real_temporal_available": bool(real_temporal_columns),
        "real_temporal_columns": real_temporal_columns,
    }


def generate_dashi_results(data_dir: str | Path = "data") -> dict[str, Any]:
    result: dict[str, Any] = {"errors": {}, "notes": []}

    data, feature_cols, summaries = prepare_uci_heart_data(data_dir)
    result["summary_shape"] = summaries["shape"]
    result["summary_samples"] = summaries["samples"]
    result["summary_target"] = summaries["target"]
    result["feature_cols"] = feature_cols

    temporal_metadata = _resolve_temporal_metadata(data)
    real_temporal_columns = temporal_metadata["real_temporal_columns"]
    result["temporal_metadata"] = temporal_metadata

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
        fig_msv_cluster, msv_cluster_table = build_msv_cluster_figure(
            msv,
            title="MSV 3D plot with clustered groups (original analysis)",
        )
        result["fig_msv_clustered"] = fig_msv_cluster
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

    # Temporal map + IGT (real temporal variable if available, otherwise synthetic index)
    try:
        temporal_df = data[feature_cols].copy()
        if real_temporal_columns:
            temporal_col = real_temporal_columns[0]
            temporal_series = pd.to_datetime(data[temporal_col], errors="coerce")
            if temporal_series.notna().sum() < 2:
                raise ValueError(
                    f"Temporal column '{temporal_col}' was detected but cannot be parsed as a usable timeline."
                )
            temporal_df["temporal_date"] = temporal_series
            date_column_name = "temporal_date"
            result["notes"].append(
                f"Temporal outputs use real temporal column '{temporal_col}' parsed from the dataset."
            )
        else:
            temporal_df["synthetic_batch_index"] = pd.Timestamp(
                "2000-01-01"
            ) + pd.to_timedelta(np.arange(len(temporal_df)), unit="D")
            date_column_name = "synthetic_batch_index"
            result["notes"].append(
                "No real timestamp/date variable is available in UCI Heart Disease; temporal outputs use a synthetic batch index from sample order."
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
        metric_name = (
            "ACCURACY" if "ACCURACY" in all_metric_names else all_metric_names[0]
        )

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


def generate_dashi_results_imputed(data_dir: str | Path = "data") -> dict[str, Any]:
    result: dict[str, Any] = {"errors": {}, "notes": []}

    data, feature_cols, numerical_cols, categorical_cols, summaries, meta = (
        prepare_uci_heart_data_imputed(data_dir)
    )
    result["summary_shape"] = summaries["shape"]
    result["summary_samples"] = summaries["samples"]
    result["summary_target"] = summaries["target"]
    result["feature_cols"] = feature_cols
    result["numerical_cols"] = numerical_cols
    result["categorical_cols"] = categorical_cols
    result["missing_before_imputation"] = summaries["missing_before_imputation"]
    result["imputation_table"] = summaries["imputation_table"]
    result["preprocessing_meta"] = meta
    result["notes"].append(
        "Analysis on imputed dataset: missing values were imputed before FAMD, MSV, IGT, clustering, and supervised performance analysis."
    )

    temporal_metadata = _resolve_temporal_metadata(data)
    real_temporal_columns = temporal_metadata["real_temporal_columns"]
    result["temporal_metadata"] = temporal_metadata

    # FAMD embedding
    try:
        fig_famd, famd_embedding = build_famd_figure(data, feature_cols)
        result["fig_famd"] = fig_famd
        result["famd_embedding"] = famd_embedding
    except Exception as exc:  # pragma: no cover
        result["errors"]["famd"] = str(exc)
        result["fig_famd"] = None

    # Data Source Maps + MSV using mixed-data reduction
    try:
        source_df = data[["source", *feature_cols]].copy()
        source_df["source"] = source_df["source"].astype("object")

        source_reduction = "FAMD"
        try:
            source_map = dashi.estimate_multivariate_data_source_map(
                data=source_df,
                source_column_name="source",
                dimensions=2,
                dim_reduction=source_reduction,
                kde_resolution=20,
                scale=True,
                scatter_plot=False,
                verbose=False,
            )
        except Exception:
            source_reduction = "PCA"
            source_map = dashi.estimate_multivariate_data_source_map(
                data=source_df,
                source_column_name="source",
                dimensions=2,
                dim_reduction=source_reduction,
                kde_resolution=20,
                scale=True,
                scatter_plot=False,
                verbose=False,
            )
            result["notes"].append(
                "FAMD unavailable in source map; fallback to PCA for imputed analysis."
            )

        result["source_dim_reduction"] = source_reduction
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
        fig_msv_cluster, msv_cluster_table = build_msv_cluster_figure(
            msv,
            title="MSV 3D plot with clustered groups (imputed analysis)",
        )
        result["fig_msv_clustered"] = fig_msv_cluster
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

    # Temporal map + IGT
    try:
        temporal_df = data[feature_cols].copy()
        if real_temporal_columns:
            temporal_col = real_temporal_columns[0]
            temporal_series = pd.to_datetime(data[temporal_col], errors="coerce")
            if temporal_series.notna().sum() < 2:
                raise ValueError(
                    f"Temporal column '{temporal_col}' was detected but cannot be parsed as a usable timeline."
                )
            temporal_df["temporal_date"] = temporal_series
            date_column_name = "temporal_date"
            result["notes"].append(
                f"Temporal outputs use real temporal column '{temporal_col}' parsed from the dataset (imputed analysis)."
            )
        else:
            temporal_df["synthetic_batch_index"] = pd.Timestamp(
                "2000-01-01"
            ) + pd.to_timedelta(np.arange(len(temporal_df)), unit="D")
            date_column_name = "synthetic_batch_index"
            result["notes"].append(
                "No real timestamp/date variable is available in UCI Heart Disease; imputed analysis uses synthetic batch index from sample order."
            )

        temporal_reduction = "FAMD"
        try:
            temporal_map = dashi.estimate_multivariate_data_temporal_map(
                data=temporal_df,
                date_column_name=date_column_name,
                dimensions=3,
                period="month",
                dim_reduction=temporal_reduction,
                kde_resolution=20,
                scale=True,
                scatter_plot=False,
                verbose=False,
            )
        except Exception:
            temporal_reduction = "PCA"
            temporal_map = dashi.estimate_multivariate_data_temporal_map(
                data=temporal_df,
                date_column_name=date_column_name,
                dimensions=3,
                period="month",
                dim_reduction=temporal_reduction,
                kde_resolution=20,
                scale=True,
                scatter_plot=False,
                verbose=False,
            )
            result["notes"].append(
                "FAMD unavailable in temporal map; fallback to PCA for imputed analysis."
            )

        result["temporal_dim_reduction"] = temporal_reduction
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

    # Supervised heatmap (mixed numerical/categorical)
    try:
        supervised_df = data[["source", *feature_cols, "target"]].copy()
        supervised_df["source"] = supervised_df["source"].astype("object")
        supervised_df["target"] = supervised_df["target"].astype(int)

        metrics = dashi.estimate_multibatch_models(
            data=supervised_df,
            inputs_numerical_column_names=numerical_cols,
            inputs_categorical_column_names=categorical_cols
            if len(categorical_cols) > 0
            else None,
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
        metric_name = (
            "ACCURACY" if "ACCURACY" in all_metric_names else all_metric_names[0]
        )

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


def dashi_generate_and_store(data_dir: str | Path = "data") -> bool:
    global DASHI_RESULTS
    baseline = generate_dashi_results(data_dir)
    imputed = generate_dashi_results_imputed(data_dir)
    combined = dict(baseline)
    combined.update({f"imputed_{k}": v for k, v in imputed.items()})
    DASHI_RESULTS = combined
    return True
