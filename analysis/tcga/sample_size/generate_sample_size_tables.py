"""Post-experiment analysis: generate sample size tables."""

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


import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


MAIN_COLUMNS = [
    "Target site",
    "N total target",
    "N source",
    "N train_1",
    "N test_1",
    "Target class 0",
    "Target class 1",
    "Target prevalence",
    "Source class 0",
    "Source class 1",
    "Source prevalence",
    "Train_1 class 0",
    "Train_1 class 1",
    "Train_1 prevalence",
    "Test_1 class 0",
    "Test_1 class 1",
    "Test_1 prevalence",
]


def apply_site_order(df: pd.DataFrame, site_order: list[str]) -> pd.DataFrame:
    if "Target site" not in df.columns or not site_order:
        return df
    category = pd.Categorical(df["Target site"], categories=site_order, ordered=True)
    return (
        df.assign(_site_order=category)
        .sort_values(["_site_order", "Target site"])
        .drop(columns="_site_order")
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class OutputCandidate:
    output_dir: Path
    manifest_path: Path
    config_snapshot_path: Path
    config_snapshot: dict[str, Any]
    is_bootstrap: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create sample-size tables from an existing transportability output folder"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Dataset root containing methods/outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where sample-size tables will be written",
    )
    return parser.parse_args()


def _is_bootstrap_candidate(output_dir: Path, snapshot: dict[str, Any]) -> bool:
    if "bootstrap" in output_dir.name.lower():
        return True
    if (output_dir / "tables" / "bootstrap_iterations.csv").exists():
        return True
    n_bootstrap = snapshot.get("n_bootstrap")
    return isinstance(n_bootstrap, int) and n_bootstrap > 1


def discover_output_candidates(input_root: Path) -> list[OutputCandidate]:
    candidates: list[OutputCandidate] = []
    for config_snapshot_path in sorted(input_root.rglob("config_snapshot.json")):
        output_dir = config_snapshot_path.parent
        manifest_path = output_dir / "tables" / "target_site_split_manifest.csv"
        if not manifest_path.exists():
            continue
        snapshot = json.loads(config_snapshot_path.read_text(encoding="utf-8"))
        candidates.append(
            OutputCandidate(
                output_dir=output_dir,
                manifest_path=manifest_path,
                config_snapshot_path=config_snapshot_path,
                config_snapshot=snapshot,
                is_bootstrap=_is_bootstrap_candidate(output_dir, snapshot),
            )
        )
    if not candidates:
        raise FileNotFoundError(
            f"No transportability output folders with split manifests found under {input_root}"
        )
    return candidates


def _candidate_score(
    candidate: OutputCandidate, *, want_bootstrap: bool
) -> tuple[int, int, int, str]:
    output_name = candidate.output_dir.name.lower()
    method_count = len(candidate.config_snapshot.get("enabled_methods", []))
    score = 0
    if candidate.is_bootstrap == want_bootstrap:
        score += 100
    if "m4" in output_name:
        score += 20
    if "transductive" in output_name:
        score += 10
    if "combat" in output_name:
        score += 5
    return (score, method_count, len(output_name), str(candidate.output_dir))


def choose_candidate(
    candidates: list[OutputCandidate], *, want_bootstrap: bool
) -> OutputCandidate | None:
    matching = [
        candidate
        for candidate in candidates
        if candidate.is_bootstrap == want_bootstrap
    ]
    if not matching:
        return None
    return sorted(
        matching,
        key=lambda candidate: _candidate_score(
            candidate, want_bootstrap=want_bootstrap
        ),
        reverse=True,
    )[0]


def parse_balance(value: Any) -> dict[str, float]:
    if pd.isna(value):
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text:
        return {}
    return json.loads(text)


def class_count(balance: dict[str, Any], class_id: int) -> int:
    value = balance.get(f"n_class_{class_id}")
    if value is None or pd.isna(value):
        return 0
    return int(value)


def prevalence(balance: dict[str, Any]) -> float | None:
    value = balance.get("positive_prevalence")
    if value is None or pd.isna(value):
        return None
    return float(value)


def resolve_existing_path(path_str: str, *, search_root: Path) -> Path:
    candidate = Path(path_str)
    if candidate.exists():
        return candidate
    filename = candidate.name
    search_bases = [search_root, search_root.parent]
    for base in search_bases:
        matches = sorted(base.rglob(filename))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not resolve dataset path from snapshot: {path_str}")


def load_dataset(
    snapshot: dict[str, Any], *, search_root: Path
) -> tuple[pd.DataFrame, str, str, Path]:
    dataset_path = resolve_existing_path(
        str(snapshot["input_path"]), search_root=search_root
    )
    domain_column = str(snapshot["domain_column"])
    target_column = str(snapshot["target_column"])
    df = pd.read_csv(dataset_path)
    if domain_column not in df.columns:
        raise KeyError(f"Domain column '{domain_column}' not found in {dataset_path}")
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in {dataset_path}")
    return df, domain_column, target_column, dataset_path


def total_site_counts(
    dataset_df: pd.DataFrame, *, domain_column: str, target_column: str
) -> pd.DataFrame:
    total_n = len(dataset_df)
    grouped = (
        dataset_df.groupby(domain_column, dropna=False)
        .agg(
            **{
                "N total target": (target_column, "size"),
                "Target class 0": (
                    target_column,
                    lambda values: int((pd.Series(values).astype(int) == 0).sum()),
                ),
                "Target class 1": (
                    target_column,
                    lambda values: int((pd.Series(values).astype(int) == 1).sum()),
                ),
            }
        )
        .reset_index()
        .rename(columns={domain_column: "Target site"})
    )
    grouped["Target prevalence"] = grouped["Target class 1"] / grouped["N total target"]
    grouped["N source"] = total_n - grouped["N total target"]
    grouped["Source class 0"] = (
        int((dataset_df[target_column].astype(int) == 0).sum())
        - grouped["Target class 0"]
    )
    grouped["Source class 1"] = (
        int((dataset_df[target_column].astype(int) == 1).sum())
        - grouped["Target class 1"]
    )
    grouped["Source prevalence"] = grouped["Source class 1"] / grouped["N source"]
    return grouped


def build_main_table(
    manifest_df: pd.DataFrame, site_counts_df: pd.DataFrame, *, site_order: list[str]
) -> pd.DataFrame:
    renamed = manifest_df.rename(
        columns={
            "evaluation_target_site": "Target site",
            "n_source_samples": "N source",
            "n_target_train_1": "N train_1",
            "n_target_test_1": "N test_1",
        }
    ).copy()
    merged = renamed.merge(
        site_counts_df,
        on="Target site",
        how="left",
        suffixes=("", "_dataset"),
    )

    train_balances = merged["target_train_1_class_balance"].apply(parse_balance)
    test_balances = merged["target_test_1_class_balance"].apply(parse_balance)
    source_balances = merged["source_class_balance"].apply(parse_balance)

    merged["Source class 0"] = source_balances.apply(
        lambda balance: class_count(balance, 0)
    )
    merged["Source class 1"] = source_balances.apply(
        lambda balance: class_count(balance, 1)
    )
    merged["Source prevalence"] = source_balances.apply(prevalence)
    merged["Train_1 class 0"] = train_balances.apply(
        lambda balance: class_count(balance, 0)
    )
    merged["Train_1 class 1"] = train_balances.apply(
        lambda balance: class_count(balance, 1)
    )
    merged["Train_1 prevalence"] = train_balances.apply(prevalence)
    merged["Test_1 class 0"] = test_balances.apply(
        lambda balance: class_count(balance, 0)
    )
    merged["Test_1 class 1"] = test_balances.apply(
        lambda balance: class_count(balance, 1)
    )
    merged["Test_1 prevalence"] = test_balances.apply(prevalence)

    main_df = merged[MAIN_COLUMNS].copy()
    return apply_site_order(main_df, site_order)


def build_bootstrap_table(
    bootstrap_manifest_df: pd.DataFrame,
    site_counts_df: pd.DataFrame,
    *,
    site_order: list[str],
) -> pd.DataFrame:
    bootstrap_df = bootstrap_manifest_df.rename(
        columns={
            "bootstrap_index": "Bootstrap iteration",
            "bootstrap_seed": "Bootstrap seed",
            "evaluation_target_site": "Target site",
            "n_source_samples": "N source",
            "n_target_train_1": "N train_1",
            "n_target_test_1": "N test_1",
        }
    ).copy()
    bootstrap_df = bootstrap_df.merge(
        site_counts_df, on="Target site", how="left", suffixes=("", "_dataset")
    )

    train_balances = bootstrap_df["target_train_1_class_balance"].apply(parse_balance)
    test_balances = bootstrap_df["target_test_1_class_balance"].apply(parse_balance)
    source_balances = bootstrap_df["source_class_balance"].apply(parse_balance)

    bootstrap_df["Source class 0"] = source_balances.apply(
        lambda balance: class_count(balance, 0)
    )
    bootstrap_df["Source class 1"] = source_balances.apply(
        lambda balance: class_count(balance, 1)
    )
    bootstrap_df["Source prevalence"] = source_balances.apply(prevalence)
    bootstrap_df["Train_1 class 0"] = train_balances.apply(
        lambda balance: class_count(balance, 0)
    )
    bootstrap_df["Train_1 class 1"] = train_balances.apply(
        lambda balance: class_count(balance, 1)
    )
    bootstrap_df["Train_1 prevalence"] = train_balances.apply(prevalence)
    bootstrap_df["Test_1 class 0"] = test_balances.apply(
        lambda balance: class_count(balance, 0)
    )
    bootstrap_df["Test_1 class 1"] = test_balances.apply(
        lambda balance: class_count(balance, 1)
    )
    bootstrap_df["Test_1 prevalence"] = test_balances.apply(prevalence)

    bootstrap_columns = ["Bootstrap iteration", "Bootstrap seed", *MAIN_COLUMNS]
    bootstrap_df = bootstrap_df[bootstrap_columns].copy()
    ordered_parts = []
    for bootstrap_index, group_df in bootstrap_df.groupby(
        "Bootstrap iteration", sort=True
    ):
        ordered_group_df = apply_site_order(group_df, site_order)
        ordered_parts.append(ordered_group_df)
    return (
        pd.concat(ordered_parts, ignore_index=True) if ordered_parts else bootstrap_df
    )


def build_bootstrap_summary(
    bootstrap_df: pd.DataFrame, *, site_order: list[str]
) -> pd.DataFrame:
    metric_columns = [
        "N train_1",
        "N test_1",
        "Train_1 class 0",
        "Train_1 class 1",
        "Train_1 prevalence",
        "Test_1 class 0",
        "Test_1 class 1",
        "Test_1 prevalence",
    ]
    static_columns = [
        "Target site",
        "N total target",
        "N source",
        "Target class 0",
        "Target class 1",
        "Target prevalence",
        "Source class 0",
        "Source class 1",
        "Source prevalence",
    ]
    summary_rows: list[dict[str, Any]] = []
    for target_site, group_df in bootstrap_df.groupby("Target site", dropna=False):
        row: dict[str, Any] = {
            "Target site": target_site,
            "N bootstrap": int(group_df["Bootstrap iteration"].nunique()),
        }
        for column in static_columns[1:]:
            row[column] = group_df[column].iloc[0]
        for column in metric_columns:
            row[f"{column} mean"] = float(group_df[column].mean())
            row[f"{column} std"] = float(group_df[column].std(ddof=0))
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    ordered_columns = [
        "Target site",
        "N bootstrap",
        "N total target",
        "N source",
        "N train_1 mean",
        "N train_1 std",
        "N test_1 mean",
        "N test_1 std",
        "Target class 0",
        "Target class 1",
        "Target prevalence",
        "Source class 0",
        "Source class 1",
        "Source prevalence",
        "Train_1 class 0 mean",
        "Train_1 class 0 std",
        "Train_1 class 1 mean",
        "Train_1 class 1 std",
        "Train_1 prevalence mean",
        "Train_1 prevalence std",
        "Test_1 class 0 mean",
        "Test_1 class 0 std",
        "Test_1 class 1 mean",
        "Test_1 class 1 std",
        "Test_1 prevalence mean",
        "Test_1 prevalence std",
    ]
    return apply_site_order(summary_df[ordered_columns], site_order)


def write_excel(
    output_path: Path,
    *,
    main_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame | None,
    bootstrap_summary_df: pd.DataFrame | None,
) -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        bundled_site_packages = Path(_configured_path("optional_site_packages"))
        if (
            bundled_site_packages.exists()
            and str(bundled_site_packages) not in sys.path
        ):
            sys.path.insert(0, str(bundled_site_packages))
        import openpyxl  # noqa: F401

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        main_df.to_excel(writer, sheet_name="sample_size_by_site", index=False)
        if bootstrap_df is not None:
            bootstrap_df.to_excel(
                writer, sheet_name="sample_size_by_bootstrap", index=False
            )
        if bootstrap_summary_df is not None:
            bootstrap_summary_df.to_excel(
                writer, sheet_name="sample_size_summary", index=False
            )


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = discover_output_candidates(input_root)
    standard_candidate = choose_candidate(candidates, want_bootstrap=False)
    bootstrap_candidate = choose_candidate(candidates, want_bootstrap=True)

    if standard_candidate is None and bootstrap_candidate is None:
        raise RuntimeError(
            "Could not find either a standard or a bootstrap split manifest."
        )

    base_candidate = standard_candidate or bootstrap_candidate
    assert base_candidate is not None
    dataset_df, domain_column, target_column, resolved_dataset_path = load_dataset(
        base_candidate.config_snapshot,
        search_root=input_root,
    )
    site_order = list(base_candidate.config_snapshot.get("target_site_order", []))
    site_counts_df = total_site_counts(
        dataset_df, domain_column=domain_column, target_column=target_column
    )

    if standard_candidate is not None:
        standard_manifest_df = pd.read_csv(standard_candidate.manifest_path)
    else:
        standard_manifest_df = pd.read_csv(bootstrap_candidate.manifest_path)  # type: ignore[arg-type]
        if "bootstrap_index" in standard_manifest_df.columns:
            first_bootstrap = int(standard_manifest_df["bootstrap_index"].min())
            standard_manifest_df = standard_manifest_df.loc[
                standard_manifest_df["bootstrap_index"] == first_bootstrap
            ].copy()
    main_df = build_main_table(
        standard_manifest_df, site_counts_df, site_order=site_order
    )

    bootstrap_df: pd.DataFrame | None = None
    bootstrap_summary_df: pd.DataFrame | None = None
    if bootstrap_candidate is not None:
        bootstrap_manifest_df = pd.read_csv(bootstrap_candidate.manifest_path)
        bootstrap_df = build_bootstrap_table(
            bootstrap_manifest_df, site_counts_df, site_order=site_order
        )
        bootstrap_summary_df = build_bootstrap_summary(
            bootstrap_df, site_order=site_order
        )
        bootstrap_df.to_csv(output_dir / "sample_size_by_bootstrap.csv", index=False)
        bootstrap_summary_df.to_csv(
            output_dir / "sample_size_summary_bootstrap.csv", index=False
        )

    main_df.to_csv(output_dir / "sample_size_by_site.csv", index=False)
    write_excel(
        output_dir / "sample_size_by_site.xlsx",
        main_df=main_df,
        bootstrap_df=bootstrap_df,
        bootstrap_summary_df=bootstrap_summary_df,
    )

    metadata = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "standard_manifest": str(standard_candidate.manifest_path)
        if standard_candidate is not None
        else "",
        "bootstrap_manifest": str(bootstrap_candidate.manifest_path)
        if bootstrap_candidate is not None
        else "",
        "dataset_path": str(resolved_dataset_path),
        "domain_column": domain_column,
        "target_column": target_column,
    }
    (output_dir / "sample_size_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Wrote sample-size tables to {output_dir}")


if __name__ == "__main__":
    main()
