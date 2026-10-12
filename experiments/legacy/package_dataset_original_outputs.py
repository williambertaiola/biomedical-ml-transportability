"""Transportability experiment: package dataset original outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IGNORE_NAMES = {".DS_Store", ".idea", "__pycache__"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_contents(source_dir: Path, destination_dir: Path) -> None:
    ensure_dir(destination_dir)
    for item in source_dir.iterdir():
        if item.name in IGNORE_NAMES:
            continue
        target = destination_dir / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                target,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(*IGNORE_NAMES),
            )
        else:
            shutil.copy2(item, target)


def count_files(directory: Path) -> int:
    return sum(1 for path in directory.rglob("*") if path.is_file())


def build_readme(
    dataset_name: str,
    output_dir: Path,
    source_paths: dict[str, Path],
    copied_counts: dict[str, int],
) -> None:
    lines = [
        f"# {dataset_name}: Reorganized Original Outputs",
        "",
        "This package preserves the original analytical outputs as produced in the source directories.",
        "No result tables were recomputed and no original plots were intentionally removed.",
        "Only non-analytical hidden metadata such as `.DS_Store`, `.idea`, and `__pycache__` was omitted.",
        "",
        "## Folder structure",
        "",
        "- `database/`: raw database files and preprocessing outputs.",
        "- `baseline/`: original Baseline M0 outputs.",
        "- `dashi/`: original DASHI outputs.",
        "- `transductive_learning/`: original transductive learning outputs.",
        "- `weighting/`: original weighting outputs.",
        "- `combat/`: original ComBat outputs.",
        "",
        "## Source mapping",
        "",
        f"- Database raw: `{source_paths['database_raw']}`",
        f"- Database preprocessing: `{source_paths['database_preprocessing']}`",
        f"- Baseline: `{source_paths['baseline']}`",
        f"- DASHI: `{source_paths['dashi']}`",
        f"- Transductive learning: `{source_paths['transductive_learning']}`",
        f"- Weighting: `{source_paths['weighting']}`",
        f"- ComBat: `{source_paths['combat']}`",
        "",
        "## Copied file counts",
        "",
        f"- Database raw: {copied_counts['database_raw']}",
        f"- Database preprocessing: {copied_counts['database_preprocessing']}",
        f"- Baseline: {copied_counts['baseline']}",
        f"- DASHI: {copied_counts['dashi']}",
        f"- Transductive learning: {copied_counts['transductive_learning']}",
        f"- Weighting: {copied_counts['weighting']}",
        f"- ComBat: {copied_counts['combat']}",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    output_dir: Path,
    dataset_name: str,
    source_paths: dict[str, Path],
    copied_counts: dict[str, int],
) -> None:
    manifest = {
        "dataset": dataset_name,
        "package_type": "reorganized_original_outputs",
        "source_paths": {key: str(value) for key, value in source_paths.items()},
        "copied_file_counts": copied_counts,
        "generated_files": sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*")
            if path.is_file()
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorganize original result folders without recomputing outputs."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--database-raw", required=True, type=Path)
    parser.add_argument("--database-preprocessing", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--dashi", required=True, type=Path)
    parser.add_argument("--transductive-learning", required=True, type=Path)
    parser.add_argument("--weighting", required=True, type=Path)
    parser.add_argument("--combat", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output)
    database_dir = ensure_dir(output_dir / "database")
    baseline_dir = ensure_dir(output_dir / "baseline")
    dashi_dir = ensure_dir(output_dir / "dashi")
    transductive_dir = ensure_dir(output_dir / "transductive_learning")
    weighting_dir = ensure_dir(output_dir / "weighting")
    combat_dir = ensure_dir(output_dir / "combat")

    copy_contents(args.database_raw, ensure_dir(database_dir / "raw"))
    copy_contents(
        args.database_preprocessing, ensure_dir(database_dir / "preprocessing")
    )
    copy_contents(args.baseline, baseline_dir)
    copy_contents(args.dashi, dashi_dir)
    copy_contents(args.transductive_learning, transductive_dir)
    copy_contents(args.weighting, weighting_dir)
    copy_contents(args.combat, combat_dir)

    source_paths = {
        "database_raw": args.database_raw,
        "database_preprocessing": args.database_preprocessing,
        "baseline": args.baseline,
        "dashi": args.dashi,
        "transductive_learning": args.transductive_learning,
        "weighting": args.weighting,
        "combat": args.combat,
    }
    copied_counts = {
        "database_raw": count_files(database_dir / "raw"),
        "database_preprocessing": count_files(database_dir / "preprocessing"),
        "baseline": count_files(baseline_dir),
        "dashi": count_files(dashi_dir),
        "transductive_learning": count_files(transductive_dir),
        "weighting": count_files(weighting_dir),
        "combat": count_files(combat_dir),
    }
    build_readme(args.dataset_name, output_dir, source_paths, copied_counts)
    build_manifest(output_dir, args.dataset_name, source_paths, copied_counts)


if __name__ == "__main__":
    main()
