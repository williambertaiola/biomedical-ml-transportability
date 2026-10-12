"""Transportability experiment: report."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    HAVE_SEABORN = True
except Exception:
    HAVE_SEABORN = False

try:
    from scipy import stats

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

from .utils import ensure_dir, log


def _setup_plot_style() -> None:
    if HAVE_SEABORN:
        sns.set_theme(style="whitegrid", context="talk")
    else:
        plt.style.use("ggplot")


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _safe_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _qap_metrics(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith(("anat_", "dti_", "func_"))]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    drop_suffixes = (
        "_n_scans_subject",
        "_quality_score",
    )
    cols = [c for c in cols if not c.endswith(drop_suffixes)]
    return cols


def _compute_site_shift(
    df: pd.DataFrame, site_col: str, metrics: List[str]
) -> pd.DataFrame:
    rows = []
    site_counts = df[site_col].value_counts(dropna=False)
    for m in metrics:
        s = pd.to_numeric(df[m], errors="coerce")
        tmp = pd.DataFrame({site_col: df[site_col], m: s}).dropna()
        if tmp.empty:
            continue
        if tmp[site_col].nunique() < 2:
            continue
        site_means = tmp.groupby(site_col)[m].mean()
        overall_std = float(tmp[m].std(ddof=1))
        between_std = float(site_means.std(ddof=1))
        shift_index = (
            between_std / overall_std
            if overall_std and not np.isnan(overall_std)
            else np.nan
        )
        row = {
            "metric": m,
            "n_non_missing": int(tmp[m].notna().sum()),
            "n_sites": int(tmp[site_col].nunique()),
            "overall_mean": float(tmp[m].mean()),
            "overall_std": overall_std,
            "between_site_std": between_std,
            "shift_index_between_overall": shift_index,
        }
        if HAVE_SCIPY:
            groups = [g[m].dropna().to_numpy() for _, g in tmp.groupby(site_col)]
            if len(groups) >= 2 and min(len(x) for x in groups) >= 3:
                try:
                    row["kruskal_p"] = float(stats.kruskal(*groups).pvalue)
                except Exception:
                    row["kruskal_p"] = np.nan
            else:
                row["kruskal_p"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("shift_index_between_overall", ascending=False)
    return out


def generate_report(
    output_dir: Path,
    full_df: pd.DataFrame,
    model_df: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    config: Dict,
) -> Dict[str, Path]:
    report_dir = ensure_dir(output_dir / "report")
    fig_dir = ensure_dir(report_dir / "figures")
    _setup_plot_style()

    site_col = (
        "site_id"
        if "site_id" in full_df.columns
        else ("domain_site" if "domain_site" in full_df.columns else None)
    )
    diagnosis_col = (
        "diagnosis_label"
        if "diagnosis_label" in full_df.columns
        else ("diagnosis" if "diagnosis" in full_df.columns else None)
    )

    if site_col is None:
        full_df = full_df.copy()
        full_df["site_id"] = "UNKNOWN_SITE"
        site_col = "site_id"

    missingness = tables.get("missingness", pd.DataFrame())
    feature_selection = tables.get("feature_selection", pd.DataFrame())
    file_overview = tables.get("file_overview", pd.DataFrame())
    audit = tables.get("audit_duplicates", pd.DataFrame())
    split_summary = tables.get("split_summary", pd.DataFrame())

    # Core descriptive tables.
    site_counts = (
        full_df[site_col]
        .astype("string")
        .fillna("UNKNOWN")
        .value_counts(dropna=False)
        .rename_axis("site_id")
        .reset_index(name="n_subjects")
    )
    site_counts["pct_subjects"] = (
        100.0 * site_counts["n_subjects"] / max(len(full_df), 1)
    )
    site_counts.to_csv(report_dir / "site_counts.csv", index=False)

    if diagnosis_col and diagnosis_col in full_df.columns:
        site_dx = pd.crosstab(
            full_df[site_col].astype("string"), full_df[diagnosis_col].astype("string")
        )
        site_dx.to_csv(report_dir / "site_by_diagnosis.csv")
    else:
        site_dx = pd.DataFrame()

    qap_cols = _qap_metrics(full_df)
    shift_df = _compute_site_shift(full_df, site_col=site_col, metrics=qap_cols)
    shift_df.to_csv(report_dir / "qap_shift_by_site.csv", index=False)

    # Figure 1: counts per site
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(site_counts["site_id"], site_counts["n_subjects"], color="#457b9d")
    ax.set_title("Subjects per site")
    ax.set_xlabel("Site")
    ax.set_ylabel("N subjects")
    ax.tick_params(axis="x", rotation=75, labelsize=9)
    _save_fig(fig, fig_dir / "01_subjects_per_site.png")

    # Figure 2: diagnosis by site (stacked)
    if not site_dx.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        site_dx.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set_title("Diagnosis distribution by site")
        ax.set_xlabel("Site")
        ax.set_ylabel("N subjects")
        ax.tick_params(axis="x", rotation=75, labelsize=9)
        _save_fig(fig, fig_dir / "02_diagnosis_by_site.png")

    # Figure 3: top missing columns
    if not missingness.empty:
        top_n = int(config.get("report", {}).get("n_top_missing_columns", 30))
        top_miss = missingness.head(top_n).copy()
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(
            top_miss["column"].astype(str), top_miss["missing_pct"], color="#e76f51"
        )
        ax.set_title("Top columns by missingness")
        ax.set_xlabel("Missing %")
        ax.invert_yaxis()
        _save_fig(fig, fig_dir / "03_top_missing_columns.png")

    # Figure 4: missingness heatmap
    if not missingness.empty:
        cols = missingness.head(25)["column"].tolist()
        sample_n = min(200, len(full_df))
        sample_df = full_df.sample(
            n=sample_n,
            random_state=int(config.get("project", {}).get("random_seed", 42)),
        )
        mat = sample_df[cols].isna().astype(int).to_numpy()
        fig, ax = plt.subplots(figsize=(12, 7))
        if HAVE_SEABORN:
            sns.heatmap(mat, cmap="magma", cbar_kws={"label": "Missing (1=yes)"}, ax=ax)
        else:
            im = ax.imshow(mat, aspect="auto", cmap="magma")
            fig.colorbar(im, ax=ax, label="Missing (1=yes)")
        ax.set_title("Missingness heatmap (sampled subjects x top missing columns)")
        ax.set_xlabel("Columns (top missing)")
        ax.set_ylabel("Sampled subjects")
        _save_fig(fig, fig_dir / "04_missingness_heatmap.png")

    # Figure 5 & 6: per-site boxplots for top shift metrics
    top_shift_metrics = (
        shift_df["metric"].head(2).tolist() if not shift_df.empty else []
    )
    for i, metric in enumerate(top_shift_metrics, start=5):
        tmp = full_df[[site_col, metric]].copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        tmp = tmp.dropna()
        if tmp.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        if HAVE_SEABORN:
            sns.boxplot(data=tmp, x=site_col, y=metric, ax=ax)
        else:
            order = list(tmp[site_col].astype(str).value_counts().index)
            data = [tmp.loc[tmp[site_col].astype(str) == s, metric] for s in order]
            ax.boxplot(data, tick_labels=order)
        ax.set_title(f"Distribution of {metric} by site")
        ax.tick_params(axis="x", rotation=75, labelsize=9)
        _save_fig(fig, fig_dir / f"{i:02d}_{metric}_by_site.png")

    # Figure 7 & 8: PCA colored by site and diagnosis
    pca_numeric = [
        c
        for c in model_df.columns
        if c.startswith(("anat_", "dti_", "func_"))
        and pd.api.types.is_numeric_dtype(model_df[c])
    ]
    if HAVE_SKLEARN and len(pca_numeric) >= 3:
        pca_df = model_df.copy()
        X = pca_df[pca_numeric].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True))
        Xs = StandardScaler().fit_transform(X)
        pcs = PCA(
            n_components=2,
            random_state=int(config.get("project", {}).get("random_seed", 42)),
        ).fit_transform(Xs)
        pca_vis = pd.DataFrame(
            {
                "PC1": pcs[:, 0],
                "PC2": pcs[:, 1],
                "site_id": full_df.loc[pca_df.index, site_col]
                .astype("string")
                .fillna("UNKNOWN"),
                "diagnosis": full_df.loc[pca_df.index, diagnosis_col]
                .astype("string")
                .fillna("UNKNOWN")
                if diagnosis_col in full_df.columns
                else "UNKNOWN",
            }
        )
        pca_vis.to_csv(report_dir / "pca_coordinates.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 7))
        if HAVE_SEABORN:
            sns.scatterplot(
                data=pca_vis,
                x="PC1",
                y="PC2",
                hue="site_id",
                s=36,
                alpha=0.85,
                ax=ax,
                legend=False,
            )
        else:
            ax.scatter(pca_vis["PC1"], pca_vis["PC2"], s=24, alpha=0.8)
        ax.set_title("PCA (QAP features) colored by site")
        _save_fig(fig, fig_dir / "07_pca_by_site.png")

        fig, ax = plt.subplots(figsize=(10, 7))
        if HAVE_SEABORN:
            sns.scatterplot(
                data=pca_vis, x="PC1", y="PC2", hue="diagnosis", s=36, alpha=0.85, ax=ax
            )
        else:
            ax.scatter(pca_vis["PC1"], pca_vis["PC2"], s=24, alpha=0.8)
        ax.set_title("PCA (QAP features) colored by diagnosis")
        _save_fig(fig, fig_dir / "08_pca_by_diagnosis.png")

    # Figure 9: duplicate scans per subject distribution
    if not audit.empty:
        dup_dist = (
            audit.groupby(["modality", "subject_id"], dropna=False)["n_scans"]
            .max()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        if HAVE_SEABORN:
            sns.histplot(
                data=dup_dist,
                x="n_scans",
                hue="modality",
                bins=20,
                multiple="stack",
                ax=ax,
            )
        else:
            ax.hist(dup_dist["n_scans"], bins=20, color="#8d99ae")
        ax.set_title("Number of scans per subject before deduplication")
        ax.set_xlabel("n_scans")
        _save_fig(fig, fig_dir / "09_duplicates_scans_distribution.png")

    # Figure 10: site x top-metric mean zscore heatmap
    if not shift_df.empty and len(site_counts) >= 2:
        top_metrics = shift_df["metric"].head(12).tolist()
        tmp = full_df[[site_col] + top_metrics].copy()
        for c in top_metrics:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        site_mean = tmp.groupby(site_col, dropna=False).mean(numeric_only=True)
        z = (site_mean - site_mean.mean(axis=0)) / site_mean.std(axis=0).replace(
            0, np.nan
        )
        z = z.fillna(0.0)
        fig, ax = plt.subplots(figsize=(12, 8))
        if HAVE_SEABORN:
            sns.heatmap(z, cmap="RdBu_r", center=0, ax=ax)
        else:
            im = ax.imshow(z.to_numpy(), cmap="coolwarm", aspect="auto")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(z.columns)))
            ax.set_xticklabels(z.columns, rotation=90, fontsize=8)
            ax.set_yticks(range(len(z.index)))
            ax.set_yticklabels(z.index, fontsize=8)
        ax.set_title("Site-level QAP shift heatmap (z-scored means)")
        _save_fig(fig, fig_dir / "10_site_shift_heatmap.png")

    # Markdown report
    n_sites = int(site_counts["site_id"].nunique()) if not site_counts.empty else 0
    is_single_site = n_sites <= 1
    kept_features = (
        feature_selection.loc[feature_selection["kept"] == True]
        if not feature_selection.empty
        else pd.DataFrame()
    )
    dropped_features = (
        feature_selection.loc[feature_selection["kept"] == False]
        if not feature_selection.empty
        else pd.DataFrame()
    )
    duplicate_subjects = (
        int((audit["n_scans"] > 1).sum())
        if not audit.empty and "n_scans" in audit.columns
        else 0
    )
    shift_preview = shift_df.head(15)
    dedup_mode = config.get("dedup", {}).get("mode", "keep_best_scan")

    site_counts_path = report_dir / "site_counts.csv"
    site_dx_path = report_dir / "site_by_diagnosis.csv"
    shift_path = report_dir / "qap_shift_by_site.csv"
    missing_path = report_dir / "missingness_by_column.csv"
    feature_sel_path = report_dir / "feature_selection.csv"
    split_summary_path = report_dir / "split_summary.csv"
    audit_path = report_dir / "audit_duplicates_summary.csv"

    missingness.to_csv(missing_path, index=False)
    feature_selection.to_csv(feature_sel_path, index=False)
    split_summary.to_csv(split_summary_path, index=False)

    audit_summary = pd.DataFrame()
    if not audit.empty:
        audit_summary = (
            audit.groupby("modality", dropna=False)["n_scans"]
            .agg(["count", "mean", "max"])
            .reset_index()
            .rename(
                columns={
                    "count": "n_subjects",
                    "mean": "mean_scans_per_subject",
                    "max": "max_scans_per_subject",
                }
            )
        )
        audit_summary.to_csv(audit_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "modality",
                "n_subjects",
                "mean_scans_per_subject",
                "max_scans_per_subject",
            ]
        ).to_csv(audit_path, index=False)

    lines = [
        "# ABIDE II Data Quality / Shift Report",
        "",
        "## 1) Dataset dimensions",
        f"- Total subjects (post-merge): **{len(full_df)}**",
        f"- Number of sites: **{n_sites}**",
        f"- Single-site warning: **{'YES' if is_single_site else 'NO'}**",
        f"- File overview: `{(output_dir / 'file_overview.csv').resolve()}`",
        "",
        "## 2) Site and class distribution",
        f"- Site counts table: `{site_counts_path.resolve()}`",
        f"- Site x diagnosis table: `{site_dx_path.resolve()}`",
        "",
        "## 3) Missingness",
        f"- Missingness by column: `{missing_path.resolve()}`",
        f"- Top missing columns shown in figure `03_top_missing_columns.png`.",
        "",
        "## 4) Duplicate scans and deduplication",
        f"- Dedup mode used: **{dedup_mode}**",
        f"- Duplicate-scan audit (full): `{(output_dir / 'audit_duplicates.csv').resolve()}`",
        f"- Duplicate summary: `{audit_path.resolve()}`",
        f"- Subjects with >1 scan entries (across modality-subject pairs): **{duplicate_subjects}**",
        "",
        "## 5) Site shift (descriptive transportability risk)",
        f"- Shift table: `{shift_path.resolve()}`",
        (
            "- Shift statistics include Kruskal p-values."
            if HAVE_SCIPY
            else "- SciPy unavailable: p-values not computed, descriptive shift index only."
        ),
        "",
        "## 6) Feature filtering",
        f"- Feature selection table: `{feature_sel_path.resolve()}`",
        f"- Kept features: **{len(kept_features)}**",
        f"- Dropped features: **{len(dropped_features)}**",
        "",
        "## 7) Domain splits (DG-ready)",
        f"- Split summary table: `{split_summary_path.resolve()}`",
        f"- Split files folder: `{(output_dir / 'splits').resolve()}`",
        "",
        "## 8) Figures",
    ]

    if not dropped_features.empty:
        dropped_preview = dropped_features.head(25)
        dropped_path = report_dir / "dropped_features_preview.csv"
        dropped_preview.to_csv(dropped_path, index=False)
        lines.append(f"- Dropped features preview: `{dropped_path.resolve()}`")

    figure_files = sorted(fig_dir.glob("*.png"))
    if figure_files:
        for p in figure_files:
            rel = Path("figures") / p.name
            lines.append(f"- ![{p.name}]({rel.as_posix()})")
    else:
        lines.append("- No figures generated.")

    if is_single_site:
        lines.extend(
            [
                "",
                "## 9) Critical note",
                "- Current export appears single-site, so LOSO/LODO across sites is not fully informative.",
                "- Use full ABIDE II multi-site phenotypic file and keep SITE_ID as domain to evaluate transportability properly.",
            ]
        )

    report_md = report_dir / "data_quality_report.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")

    log(f"Report written to: {report_md}")
    return {
        "report_md": report_md,
        "report_dir": report_dir,
        "figure_dir": fig_dir,
    }
