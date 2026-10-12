"""Transportability experiment: dashi rmd backend tcga brca."""

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


from pathlib import Path
from typing import Any
import json
import re
import warnings
import os

import numpy as np
import pandas as pd

import dashi
from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import plotly.graph_objects as go

DATA_ROOT_CANDIDATES = [
    Path(_configured_path("tcga_raw_root")),
    Path(_configured_path("tcga_raw_root")),
]


def _select_cluster_count(n_points: int) -> int:
    # Source-level MSV points are usually few; keep a compact and interpretable k.
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
                line=dict(color=color, width=6 if i == 0 else 4),
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


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    if data_root is not None:
        p = Path(data_root)
        if (p / "TCGA-BRCA.star_counts.tsv").exists():
            return p

    import os

    tcga_env = os.getenv("TCGA_BRCA_DATA_ROOT", "").strip()
    if tcga_env:
        p = Path(tcga_env)
        if (p / "TCGA-BRCA.star_counts.tsv").exists():
            return p

    for p in DATA_ROOT_CANDIDATES:
        if (p / "TCGA-BRCA.star_counts.tsv").exists():
            return p

    raise FileNotFoundError(
        "TCGA-BRCA data root not found. Set TCGA_BRCA_DATA_ROOT or provide data_root explicitly."
    )


def _clean_missing_tokens(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    missing_tokens = {
        "",
        "nan",
        "none",
        "null",
        "not reported",
        "'--",
        "--",
        "na",
        "n/a",
    }
    mask = out.str.lower().isin(missing_tokens)
    out = out.mask(mask)
    out = out.replace("<NA>", np.nan)
    return out


def _first_informative(series: pd.Series) -> Any:
    vals = series.dropna()
    if vals.empty:
        return np.nan
    vc = vals.value_counts()
    return vc.index[0]


def _parse_sample_metadata(sample_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sid in sample_ids:
        parts = str(sid).split("-")
        case_id = "-".join(parts[:3]) if len(parts) >= 3 else np.nan
        source = parts[1] if len(parts) >= 2 else np.nan
        sample_type_code = parts[3][:2] if len(parts) >= 4 else np.nan
        rows.append(
            {
                "sample_id": sid,
                "case_id": case_id,
                "source": source,
                "sample_type_code": sample_type_code,
            }
        )
    return pd.DataFrame(rows)


def _load_case_clinical_metadata(clinical_path: Path) -> pd.DataFrame:
    df = pd.read_csv(clinical_path, sep="\t", low_memory=False)

    required = [
        "cases.submitter_id",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.year_of_diagnosis",
        "cases.index_date",
        "diagnoses.days_to_last_follow_up",
        "demographic.days_to_death",
        "demographic.vital_status",
    ]
    cols = [c for c in required if c in df.columns]
    cdf = df[cols].copy()

    rename_map = {
        "cases.submitter_id": "case_id",
        "diagnoses.ajcc_pathologic_stage": "ajcc_pathologic_stage",
        "diagnoses.year_of_diagnosis": "year_of_diagnosis",
        "cases.index_date": "index_date",
        "diagnoses.days_to_last_follow_up": "days_to_last_follow_up",
        "demographic.days_to_death": "days_to_death",
        "demographic.vital_status": "vital_status",
    }
    cdf = cdf.rename(columns=rename_map)

    for col in cdf.columns:
        if col != "case_id":
            cdf[col] = _clean_missing_tokens(cdf[col])

    case_level = cdf.groupby("case_id", as_index=False).agg(_first_informative)
    return case_level


def _encode_stage_binary(stage_value: Any) -> float:
    if pd.isna(stage_value):
        return np.nan
    s = str(stage_value).strip()
    if not s:
        return np.nan

    # Late stage first to avoid "Stage III" matching "Stage I".
    if re.search(r"^Stage\s*IV", s, flags=re.IGNORECASE) or re.search(
        r"^Stage\s*III", s, flags=re.IGNORECASE
    ):
        return 1.0
    if re.search(r"^Stage\s*II", s, flags=re.IGNORECASE) or re.search(
        r"^Stage\s*I(?!I|V)", s, flags=re.IGNORECASE
    ):
        return 0.0
    return np.nan


def _choose_temporal_variable(df: pd.DataFrame) -> tuple[str | None, pd.Series, str]:
    # Returns (temporal_name, temporal_datetime, note)
    candidates: list[tuple[str, pd.Series, str]] = []

    if "year_of_diagnosis" in df.columns:
        y = pd.to_numeric(df["year_of_diagnosis"], errors="coerce")
        y = y.where((y >= 1900) & (y <= 2100))
        dt = pd.to_datetime(y.astype("Int64").astype(str), format="%Y", errors="coerce")
        candidates.append(("year_of_diagnosis", dt, "calendar year of diagnosis"))

    if "index_date" in df.columns:
        dt = pd.to_datetime(df["index_date"], errors="coerce")
        candidates.append(("index_date", dt, "clinical index date"))

    if "days_to_last_follow_up" in df.columns:
        d = pd.to_numeric(df["days_to_last_follow_up"], errors="coerce")
        d = d.where(d >= 0)
        # Real duration variable; converted to date-like axis for DASHI.
        # Build datetime safely to avoid pandas RuntimeWarning on NaN -> timedelta cast.
        dt = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
        valid = d.notna()
        if valid.any():
            d_days = d.loc[valid].round().astype("int64")
            dt.loc[valid] = pd.Timestamp("2000-01-01") + pd.to_timedelta(
                d_days, unit="D"
            )
        candidates.append(("days_to_last_follow_up", dt, "follow-up duration in days"))

    # Prefer clinically interpretable calendar-time metadata for temporal drift.
    # Fallback to follow-up duration only if true calendar fields are unavailable.
    preferred_order = ["year_of_diagnosis", "index_date", "days_to_last_follow_up"]
    by_name = {name: (dt, note) for name, dt, note in candidates}

    for name in preferred_order:
        if name not in by_name:
            continue
        dt, note = by_name[name]
        non_missing = int(dt.notna().sum())
        unique = int(dt.nunique(dropna=True))
        if non_missing >= 100 and unique >= 5:
            return name, dt, note

    return None, pd.Series([pd.NaT] * len(df)), ""


def _compute_gene_stats(
    star_counts_path: Path,
    sample_ids: list[str],
    expr_threshold: float,
    min_expr_fraction: float,
    low_variance_quantile: float,
    top_n_genes: int,
    chunksize: int = 2500,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    usecols = ["Ensembl_ID", *sample_ids]

    gene_ids: list[str] = []
    nonzero_frac: list[float] = []
    variances: list[float] = []
    means: list[float] = []

    for chunk in pd.read_csv(
        star_counts_path, sep="\t", usecols=usecols, chunksize=chunksize
    ):
        gid = chunk.iloc[:, 0].astype(str).tolist()
        vals = (
            chunk.iloc[:, 1:]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32, copy=False)
        )

        nz_frac = np.mean(vals > expr_threshold, axis=1)
        var_vals = np.nanvar(vals, axis=1)
        mean_vals = np.nanmean(vals, axis=1)

        gene_ids.extend(gid)
        nonzero_frac.extend(nz_frac.tolist())
        variances.extend(var_vals.tolist())
        means.extend(mean_vals.tolist())

    stats = pd.DataFrame(
        {
            "gene_id": gene_ids,
            "nonzero_fraction": np.asarray(nonzero_frac, dtype=float),
            "variance": np.asarray(variances, dtype=float),
            "mean_expression": np.asarray(means, dtype=float),
        }
    )

    keep_expr = stats["nonzero_fraction"] >= float(min_expr_fraction)
    expr_pass = stats.loc[keep_expr].copy()

    if expr_pass.empty:
        raise ValueError("No genes passed low-expression filtering.")

    var_threshold = float(expr_pass["variance"].quantile(low_variance_quantile))
    keep_var = expr_pass["variance"] > var_threshold
    var_pass = expr_pass.loc[keep_var].copy()

    if var_pass.empty:
        raise ValueError("No genes passed low-variance filtering.")

    selected = var_pass.nlargest(top_n_genes, columns="variance").copy()
    selected = selected.sort_values("variance", ascending=False).reset_index(drop=True)

    summary = {
        "n_genes_initial": int(stats.shape[0]),
        "n_genes_after_low_expression": int(expr_pass.shape[0]),
        "n_genes_removed_low_expression": int(stats.shape[0] - expr_pass.shape[0]),
        "variance_threshold_quantile": float(low_variance_quantile),
        "variance_threshold_value": var_threshold,
        "n_genes_after_low_variance": int(var_pass.shape[0]),
        "n_genes_removed_low_variance": int(expr_pass.shape[0] - var_pass.shape[0]),
        "n_genes_selected_final": int(selected.shape[0]),
        "expression_threshold": float(expr_threshold),
        "min_expression_fraction": float(min_expr_fraction),
    }

    return stats, selected, summary


def _extract_selected_gene_matrix(
    star_counts_path: Path,
    sample_ids: list[str],
    selected_genes: set[str],
    chunksize: int = 2500,
) -> pd.DataFrame:
    usecols = ["Ensembl_ID", *sample_ids]
    blocks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        star_counts_path, sep="\t", usecols=usecols, chunksize=chunksize
    ):
        filt = chunk[chunk["Ensembl_ID"].astype(str).isin(selected_genes)]
        if not filt.empty:
            blocks.append(filt)

    if not blocks:
        raise ValueError("Selected genes were not found in star counts file.")

    genes_df = pd.concat(blocks, ignore_index=True)
    genes_df = genes_df.drop_duplicates(subset=["Ensembl_ID"], keep="first")
    genes_df = genes_df.set_index("Ensembl_ID")
    genes_df = genes_df.apply(pd.to_numeric, errors="coerce")

    matrix = genes_df.T
    matrix.index.name = "sample_id"
    matrix.columns = [str(c) for c in matrix.columns]

    # preserve requested sample order
    matrix = matrix.reindex(sample_ids)
    return matrix


def _build_external_performance(
    data: pd.DataFrame,
    feature_cols: list[str],
    output_tables_dir: Path,
) -> tuple[pd.DataFrame, go.Figure, dict[str, Any]]:
    model = Pipeline(
        steps=[
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    sources = sorted(data["source"].astype(str).unique().tolist())
    rows: list[dict[str, Any]] = []

    for tr in sources:
        train_df = data[data["source"] == tr]
        X_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["target"].to_numpy(dtype=int)
        train_unique = np.unique(y_train)

        for te in sources:
            test_df = data[data["source"] == te]
            X_test = test_df[feature_cols].to_numpy(dtype=float)
            y_test = test_df["target"].to_numpy(dtype=int)
            test_unique = np.unique(y_test)

            overlap = len(
                set(train_df["sample_id"]).intersection(set(test_df["sample_id"]))
            )

            result = {
                "train_source": tr,
                "test_source": te,
                "n_train": int(train_df.shape[0]),
                "n_test": int(test_df.shape[0]),
                "n_train_class_0": int(np.sum(y_train == 0)),
                "n_train_class_1": int(np.sum(y_train == 1)),
                "n_test_class_0": int(np.sum(y_test == 0)),
                "n_test_class_1": int(np.sum(y_test == 1)),
                "sample_overlap": int(overlap),
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "f1": np.nan,
                "roc_auc": np.nan,
                "status": "ok",
            }

            if overlap > 0:
                result["status"] = "overlap_detected"
                rows.append(result)
                continue

            if train_unique.size < 2:
                result["status"] = "train_single_class"
                rows.append(result)
                continue

            if test_unique.size < 2:
                result["status"] = "test_single_class"
                rows.append(result)
                continue

            clf = clone(model)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            result["accuracy"] = float(accuracy_score(y_test, y_pred))
            result["balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
            result["f1"] = float(f1_score(y_test, y_pred, zero_division=0))

            try:
                y_prob = clf.predict_proba(X_test)[:, 1]
                result["roc_auc"] = float(roc_auc_score(y_test, y_prob))
            except Exception:
                result["roc_auc"] = np.nan
                result["status"] = "auc_unavailable"

            rows.append(result)

    perf = pd.DataFrame(rows)
    perf.to_csv(output_tables_dir / "external_performance_pairwise.csv", index=False)

    pivot_bal = perf.pivot(
        index="train_source", columns="test_source", values="balanced_accuracy"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_bal.to_numpy(dtype=float),
            x=pivot_bal.columns.tolist(),
            y=pivot_bal.index.tolist(),
            colorscale="RdYlGn",
            zmin=0.0,
            zmax=1.0,
            colorbar={"title": "Balanced ACC"},
            hovertemplate="train=%{y}<br>test=%{x}<br>balanced_acc=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="External Cross-Source Performance Heatmap (Balanced Accuracy)",
        xaxis_title="Test Source",
        yaxis_title="Train Source",
        margin={"l": 70, "r": 20, "t": 50, "b": 70},
    )

    diag = {
        "pairs_total": int(perf.shape[0]),
        "pairs_with_overlap": int((perf["sample_overlap"] > 0).sum()),
        "pairs_with_metric_1_balanced_acc": int(
            np.sum(np.isclose(perf["balanced_accuracy"], 1.0, equal_nan=False))
        ),
        "pairs_with_metric_0_balanced_acc": int(
            np.sum(np.isclose(perf["balanced_accuracy"], 0.0, equal_nan=False))
        ),
        "status_counts": perf["status"].value_counts(dropna=False).to_dict(),
    }

    with open(
        output_tables_dir / "external_performance_diagnostics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(diag, f, indent=2)

    return perf, fig, diag


def generate_dashi_results(
    data_root: str | Path | None = None,
    output_root: str | Path = "outputs/tcga_brca_dashi_rebuild",
) -> dict[str, Any]:
    result: dict[str, Any] = {"errors": {}, "notes": []}
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in cast",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Could not find the number of physical cores for the following reason",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="resource_tracker: There appear to be .* leaked semaphore objects to clean up at shutdown",
        category=UserWarning,
    )

    data_root_path = resolve_data_root(data_root)
    star_counts_path = data_root_path / "TCGA-BRCA.star_counts.tsv"
    clinical_path = data_root_path / "clinical.cart.2026-03-09 (1)" / "clinical.tsv"

    if not clinical_path.exists():
        raise FileNotFoundError(f"Missing required clinical file: {clinical_path}")

    output_root = Path(output_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    reports_dir = output_root / "reports"
    data_dir = output_root / "data"
    for d in (tables_dir, figures_dir, reports_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # A. Sample-level metadata
    # --------------------------
    header = pd.read_csv(star_counts_path, sep="\t", nrows=0)
    sample_ids = header.columns[1:].tolist()
    sample_meta = _parse_sample_metadata(sample_ids)

    case_meta = _load_case_clinical_metadata(clinical_path)

    sample_meta = sample_meta.sort_values("sample_id").reset_index(drop=True)
    sample_meta["is_primary_tumor"] = sample_meta["sample_type_code"].eq("01")

    # sample-level audit table
    audit = sample_meta[
        ["sample_id", "case_id", "source", "sample_type_code", "is_primary_tumor"]
    ].copy()
    audit["keep"] = True
    audit["removal_reason"] = ""

    # filter: primary tumor only
    non_primary = ~audit["is_primary_tumor"]
    audit.loc[non_primary, "keep"] = False
    audit.loc[non_primary, "removal_reason"] = "non_primary_tumor"

    # deduplicate aliquots/samples: keep one sample per case among primary tumors
    primary_keep = audit[audit["keep"]].copy().sort_values("sample_id")
    dup_case = primary_keep.duplicated(subset=["case_id"], keep="first")
    dup_ids = primary_keep.loc[dup_case, "sample_id"]
    if not dup_ids.empty:
        audit.loc[audit["sample_id"].isin(dup_ids), "keep"] = False
        audit.loc[audit["sample_id"].isin(dup_ids), "removal_reason"] = (
            "duplicate_case_sample"
        )

    # merge clinical metadata
    kept_ids_step = audit.loc[audit["keep"], "sample_id"].tolist()
    merged = audit[audit["sample_id"].isin(kept_ids_step)].merge(
        case_meta, on="case_id", how="left"
    )

    merged["target"] = merged["ajcc_pathologic_stage"].map(_encode_stage_binary)

    temporal_name, temporal_dt, temporal_note = _choose_temporal_variable(merged)
    merged["temporal_date"] = temporal_dt

    # missing source/target/temporal filtering
    missing_source = merged["source"].isna() | (
        merged["source"].astype(str).str.strip() == ""
    )
    missing_target = merged["target"].isna()
    missing_temporal = merged["temporal_date"].isna()

    for mask, reason in [
        (missing_source, "missing_source"),
        (missing_target, "missing_target"),
        (missing_temporal, "missing_temporal"),
    ]:
        rm_ids = merged.loc[mask, "sample_id"]
        if not rm_ids.empty:
            audit.loc[audit["sample_id"].isin(rm_ids), "keep"] = False
            audit.loc[
                audit["sample_id"].isin(rm_ids) & (audit["removal_reason"] == ""),
                "removal_reason",
            ] = reason
            audit.loc[
                audit["sample_id"].isin(rm_ids)
                & (audit["removal_reason"] != "")
                & (audit["removal_reason"] != reason),
                "removal_reason",
            ] = (
                audit.loc[
                    audit["sample_id"].isin(rm_ids)
                    & (audit["removal_reason"] != "")
                    & (audit["removal_reason"] != reason),
                    "removal_reason",
                ]
                + ";"
                + reason
            )

    # provisional kept dataframe after metadata filters
    keep_ids_meta = audit.loc[audit["keep"], "sample_id"].tolist()
    meta_df = merged[merged["sample_id"].isin(keep_ids_meta)].copy()

    # per-source balance filter for supervised stability
    source_balance = (
        meta_df.groupby("source")["target"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={0.0: "target_0", 1.0: "target_1"})
    )
    source_balance["n_samples"] = source_balance.sum(axis=1)

    min_source_n = 25
    min_class_n = 5
    eligible_sources = source_balance[
        (source_balance["n_samples"] >= min_source_n)
        & (source_balance.get("target_0", 0) >= min_class_n)
        & (source_balance.get("target_1", 0) >= min_class_n)
    ].index.tolist()

    bad_source_ids = meta_df.loc[~meta_df["source"].isin(eligible_sources), "sample_id"]
    if not bad_source_ids.empty:
        audit.loc[audit["sample_id"].isin(bad_source_ids), "keep"] = False
        audit.loc[
            audit["sample_id"].isin(bad_source_ids) & (audit["removal_reason"] == ""),
            "removal_reason",
        ] = "source_unbalanced_or_too_small"
        audit.loc[
            audit["sample_id"].isin(bad_source_ids)
            & (audit["removal_reason"] != "")
            & (audit["removal_reason"] != "source_unbalanced_or_too_small"),
            "removal_reason",
        ] = (
            audit.loc[
                audit["sample_id"].isin(bad_source_ids)
                & (audit["removal_reason"] != "")
                & (audit["removal_reason"] != "source_unbalanced_or_too_small"),
                "removal_reason",
            ]
            + ";source_unbalanced_or_too_small"
        )

    keep_ids = audit.loc[audit["keep"], "sample_id"].tolist()

    final_meta = merged[merged["sample_id"].isin(keep_ids)].copy()
    final_meta = final_meta[
        [
            "sample_id",
            "case_id",
            "source",
            "target",
            "temporal_date",
            "year_of_diagnosis",
            "vital_status",
            "ajcc_pathologic_stage",
        ]
    ]
    final_meta = final_meta.sort_values(["source", "sample_id"]).reset_index(drop=True)

    # preprocessing summary/audit outputs
    audit.loc[audit["keep"], "removal_reason"] = "kept"
    audit.to_csv(tables_dir / "sample_audit.csv", index=False)

    step_summary = [
        {"step": "initial_samples", "n_samples": int(len(sample_ids))},
        {
            "step": "after_primary_tumor_filter",
            "n_samples": int(np.sum(audit["is_primary_tumor"])),
        },
        {
            "step": "after_case_deduplication",
            "n_samples": int(len(set(audit[audit["is_primary_tumor"]]["case_id"]))),
        },
        {
            "step": "after_target_temporal_source_filter",
            "n_samples": int(len(keep_ids_meta)),
        },
        {"step": "after_source_balance_filter", "n_samples": int(len(keep_ids))},
    ]
    step_summary_df = pd.DataFrame(step_summary)
    step_summary_df.to_csv(tables_dir / "preprocessing_step_summary.csv", index=False)

    # --------------------------
    # B. Gene filtering
    # --------------------------
    expr_threshold = 0.1
    min_expr_fraction = 0.10
    low_variance_quantile = 0.25
    top_n_genes = 1500

    gene_stats, selected_genes_df, gene_filter_summary = _compute_gene_stats(
        star_counts_path=star_counts_path,
        sample_ids=keep_ids,
        expr_threshold=expr_threshold,
        min_expr_fraction=min_expr_fraction,
        low_variance_quantile=low_variance_quantile,
        top_n_genes=top_n_genes,
    )
    gene_stats.to_csv(tables_dir / "gene_filter_metrics_all.csv", index=False)
    selected_genes_df.to_csv(
        tables_dir / "selected_genes_top_variance.csv", index=False
    )
    with open(tables_dir / "gene_filter_summary.json", "w", encoding="utf-8") as f:
        json.dump(gene_filter_summary, f, indent=2)

    selected_gene_set = set(selected_genes_df["gene_id"].tolist())
    feature_matrix = _extract_selected_gene_matrix(
        star_counts_path=star_counts_path,
        sample_ids=keep_ids,
        selected_genes=selected_gene_set,
    )

    # merge metadata + features
    dashi_df = final_meta.merge(
        feature_matrix.reset_index(), on="sample_id", how="inner"
    )
    dashi_df = dashi_df.sort_values(["source", "sample_id"]).reset_index(drop=True)

    feature_cols = [
        c
        for c in dashi_df.columns
        if c
        not in {
            "sample_id",
            "case_id",
            "source",
            "target",
            "temporal_date",
            "year_of_diagnosis",
            "vital_status",
            "ajcc_pathologic_stage",
        }
    ]

    dashi_ready = dashi_df[
        ["sample_id", "source", "temporal_date", "target", *feature_cols]
    ].copy()

    # save dashi-ready dataset
    dashi_ready.to_csv(data_dir / "tcga_brca_dashi_ready.csv", index=False)

    # key summary tables
    samples_per_source = (
        dashi_ready["source"]
        .value_counts()
        .rename_axis("source")
        .reset_index(name="n_samples")
    )
    target_global = (
        dashi_ready["target"]
        .value_counts()
        .rename_axis("target")
        .reset_index(name="n_samples")
    )
    target_by_source = (
        dashi_ready.groupby("source")["target"]
        .value_counts()
        .rename("n_samples")
        .reset_index()
    )
    target_by_source["pct_within_source"] = (
        target_by_source["n_samples"]
        / target_by_source.groupby("source")["n_samples"].transform("sum")
        * 100
    )

    samples_per_source.to_csv(tables_dir / "samples_per_source.csv", index=False)
    target_global.to_csv(tables_dir / "target_distribution_global.csv", index=False)
    target_by_source.to_csv(
        tables_dir / "target_distribution_by_source.csv", index=False
    )
    source_balance.reset_index().to_csv(
        tables_dir / "source_balance_before_source_filter.csv", index=False
    )

    temporal_info = {
        "temporal_variable": temporal_name,
        "temporal_note": temporal_note,
        "non_missing_temporal": int(dashi_ready["temporal_date"].notna().sum()),
        "n_unique_temporal": int(dashi_ready["temporal_date"].nunique(dropna=True)),
    }
    with open(tables_dir / "temporal_variable_info.json", "w", encoding="utf-8") as f:
        json.dump(temporal_info, f, indent=2)

    # --------------------------
    # C. DASHI unsupervised
    # --------------------------
    try:
        source_df = dashi_ready[["source", *feature_cols]].copy()
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
        # Secondary MSV visualization in the same 3D coordinates with clustered circles.
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
        result["msv_table"].to_csv(tables_dir / "dashi_msv_metrics.csv", index=False)
    except Exception as exc:  # pragma: no cover
        result["errors"]["source_msv"] = str(exc)
        result["fig_source_map"] = None
        result["fig_msv"] = None
        result["fig_msv_clustered"] = None

    # --------------------------
    # D. DASHI temporal + IGT
    # --------------------------
    try:
        temporal_df = dashi_ready[[*feature_cols, "temporal_date"]].copy()
        temporal_df_for_plot = temporal_df.copy()

        # Remove very sparse edge years only for temporal visualization stability.
        temporal_year = pd.to_datetime(
            temporal_df_for_plot["temporal_date"], errors="coerce"
        ).dt.year
        year_counts = temporal_year.value_counts(dropna=True)
        valid_years = set(year_counts[year_counts >= 5].index.tolist())
        if len(valid_years) >= 5:
            keep_mask = temporal_year.isin(valid_years)
            if int(keep_mask.sum()) >= int(0.8 * len(temporal_df_for_plot)):
                temporal_df_for_plot = temporal_df_for_plot.loc[keep_mask].copy()
                result["temporal_filter_note"] = (
                    "Years with fewer than 5 samples were excluded from temporal plots "
                    "to avoid unstable edge batches."
                )
            else:
                result["temporal_filter_note"] = (
                    "No sparse-year temporal filtering applied (would remove too many samples)."
                )
        else:
            result["temporal_filter_note"] = (
                "No sparse-year temporal filtering applied."
            )

        temporal_map_for_heatmap = dashi.estimate_multivariate_data_temporal_map(
            data=temporal_df_for_plot,
            date_column_name="temporal_date",
            dimensions=2,
            period="year",
            dim_reduction="PCA",
            kde_resolution=20,
            scale=True,
            scatter_plot=False,
            verbose=False,
        )
        temporal_map_for_igt = dashi.estimate_multivariate_data_temporal_map(
            data=temporal_df_for_plot,
            date_column_name="temporal_date",
            dimensions=3,
            period="year",
            dim_reduction="PCA",
            kde_resolution=20,
            scale=True,
            scatter_plot=False,
            verbose=False,
        )
        igt = dashi.estimate_igt_projection(temporal_map_for_igt, dimensions=3)

        result["fig_temporal_heatmap"] = dashi.plot_multivariate_data_temporal_map(
            temporal_map_for_heatmap, absolute=False, log_transform=False
        )
        result["fig_igt"] = dashi.plot_IGT_projection(
            igt, dimensions=3, trajectory=True
        )
    except Exception as exc:  # pragma: no cover
        result["errors"]["temporal"] = str(exc)
        result["fig_temporal_heatmap"] = None
        result["fig_igt"] = None

    # --------------------------
    # E. DASHI supervised
    # --------------------------
    dashi_metrics = None
    try:
        supervised_df = dashi_ready[["source", *feature_cols, "target"]].copy()
        supervised_df["source"] = supervised_df["source"].astype("object")
        supervised_df["target"] = supervised_df["target"].astype(int)

        dashi_metrics = dashi.estimate_multibatch_models(
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

        metric_names = sorted({m for row in dashi_metrics.values() for m in row.keys()})
        preferred = ["BALANCED_ACCURACY", "AUC", "F1", "ACCURACY"]
        metric_name = next((m for m in preferred if m in metric_names), metric_names[0])

        result["fig_performance_heatmap"] = dashi.plot_multibatch_performance(
            metrics=dashi_metrics,
            metric_name=metric_name,
        )
        result["performance_matrix"] = dashi.arrange_performance_metrics(
            metrics=dashi_metrics, metric_name=metric_name
        )
        result["metric_name"] = metric_name

        pd.DataFrame(result["performance_matrix"]).to_csv(
            tables_dir / "dashi_performance_matrix.csv"
        )
    except Exception as exc:  # pragma: no cover
        result["errors"]["supervised"] = str(exc)
        result["fig_performance_heatmap"] = None

    # --------------------------
    # F. External supervised diagnostics
    # --------------------------
    try:
        external_perf, external_fig, external_diag = _build_external_performance(
            data=dashi_ready,
            feature_cols=feature_cols,
            output_tables_dir=tables_dir,
        )
        result["external_performance_table"] = external_perf
        result["fig_external_performance_heatmap"] = external_fig
        result["external_performance_diag"] = external_diag
    except Exception as exc:  # pragma: no cover
        result["errors"]["external_supervised"] = str(exc)
        result["fig_external_performance_heatmap"] = None

    # --------------------------
    # G. Final summaries for report
    # --------------------------
    preprocessing_summary = pd.DataFrame(
        [
            {"item": "samples_initial", "value": int(len(sample_ids))},
            {"item": "samples_final", "value": int(dashi_ready.shape[0])},
            {
                "item": "genes_initial",
                "value": int(gene_filter_summary["n_genes_initial"]),
            },
            {
                "item": "genes_final",
                "value": int(gene_filter_summary["n_genes_selected_final"]),
            },
            {"item": "n_sources_final", "value": int(dashi_ready["source"].nunique())},
            {"item": "temporal_variable", "value": temporal_name or "not_available"},
            {
                "item": "target_definition",
                "value": "ajcc_pathologic_stage: early(I/II)=0 vs late(III/IV)=1",
            },
        ]
    )
    preprocessing_summary.to_csv(tables_dir / "preprocessing_summary.csv", index=False)

    result["preprocessing_summary"] = preprocessing_summary
    result["summary_shape"] = pd.DataFrame(
        [
            {
                "dataset": "tcga_brca_dashi_ready",
                "n_rows": int(dashi_ready.shape[0]),
                "n_columns": int(dashi_ready.shape[1]),
            }
        ]
    )
    result["summary_samples"] = samples_per_source
    result["summary_target"] = target_by_source
    result["missing_pct"] = (
        (dashi_ready[feature_cols].isna().mean() * 100)
        .rename("missing_pct")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    result["source_balance"] = source_balance.reset_index()
    result["feature_cols"] = feature_cols
    result["temporal_metadata"] = {
        "real_temporal_available": temporal_name is not None,
        "real_temporal_columns": [temporal_name] if temporal_name else [],
        "temporal_note": temporal_note,
    }

    result["notes"].extend(
        [
            f"Data root resolved to: {data_root_path}",
            f"Temporal variable used: {temporal_name} ({temporal_note}).",
            "Target used: AJCC pathologic stage early(I/II)=0 vs late(III/IV)=1.",
            "Primary tumor samples were retained and duplicate case-level samples were removed.",
            f"Final supervised sources: {', '.join(sorted(dashi_ready['source'].unique().tolist()))}",
        ]
    )

    artifacts = {
        "data_root": str(data_root_path),
        "star_counts_path": str(star_counts_path),
        "clinical_path": str(clinical_path),
        "output_root": str(output_root),
        "dashi_ready_csv": str(data_dir / "tcga_brca_dashi_ready.csv"),
        "sample_audit_csv": str(tables_dir / "sample_audit.csv"),
        "preprocessing_summary_csv": str(tables_dir / "preprocessing_summary.csv"),
        "samples_per_source_csv": str(tables_dir / "samples_per_source.csv"),
        "target_distribution_by_source_csv": str(
            tables_dir / "target_distribution_by_source.csv"
        ),
        "gene_filter_summary_json": str(tables_dir / "gene_filter_summary.json"),
        "selected_genes_csv": str(tables_dir / "selected_genes_top_variance.csv"),
        "external_performance_pairwise_csv": str(
            tables_dir / "external_performance_pairwise.csv"
        ),
    }
    with open(tables_dir / "artifacts.json", "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    return result


DASHI_RESULTS: dict[str, Any] | None = None


def dashi_generate_and_store(
    data_root: str | Path | None = None,
    output_root: str | Path = "outputs/tcga_brca_dashi_rebuild",
) -> bool:
    global DASHI_RESULTS
    DASHI_RESULTS = generate_dashi_results(data_root=data_root, output_root=output_root)
    return True
