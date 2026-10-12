"""Compare original and refactored experiments using private archive inputs.

Reference code, biomedical inputs, and intermediate comparison tables are kept
in a temporary directory and removed on exit. Only table sizes and comparison
status are printed or written to the optional JSON report.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "uci": (
        "UCI_Heart",
        "uci_heart_analysis_ready.csv",
        "uci_heart_feature_schema.json",
    ),
    "abide": (
        "ABIDE2",
        "abide2_core_multisite_transportability_dataset_strict.csv",
        "abide2_core_multisite_preprocessing_blueprint.json",
    ),
    "tcga": ("TCGA_BRCA", "tcga_brca_stage_transportability_dataset.csv", None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, help="Read-only original thesis archive"
    )
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--methods", default="M0,M1,M2,M3,M4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-sites",
        default="",
        help="Optional ABIDE/TCGA subset; UCI evaluates all four sites",
    )
    parser.add_argument(
        "--report", type=Path, help="Optional aggregate JSON comparison report"
    )
    parser.add_argument(
        "--worker-mode", choices=("original", "refactored"), help=argparse.SUPPRESS
    )
    parser.add_argument("--working-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.dataset == "uci" and args.target_sites:
        parser.error("The original UCI configuration has no target-site subset option.")
    if not args.worker_mode and args.archive is None:
        parser.error("--archive is required")
    return args


def run_worker(args: argparse.Namespace) -> None:
    """Execute each version in a separate interpreter to isolate its imports."""
    import pandas as pd

    workspace = args.working_dir
    package_root = (
        workspace / "original_src"
        if args.worker_mode == "original"
        else REPOSITORY_ROOT / "src"
    )
    sys.path.insert(0, str(package_root))
    prefix = args.dataset + "_transportability"
    config_class = importlib.import_module(prefix + ".config").ExperimentConfig
    experiment = importlib.import_module(
        prefix + ".experiment"
    ).run_transportability_experiment
    _, input_name, schema_name = DATASETS[args.dataset]
    kwargs = dict(
        input_path=workspace / "inputs" / input_name,
        output_dir=workspace / args.worker_mode,
        random_seed=args.seed,
        n_jobs=1,
        enabled_methods=tuple(
            part.strip().upper() for part in args.methods.split(",") if part.strip()
        ),
    )
    if schema_name:
        kwargs["schema_path"] = workspace / "inputs" / schema_name
    if args.target_sites:
        kwargs["target_sites"] = tuple(
            part.strip() for part in args.target_sites.split(",") if part.strip()
        )
    config = config_class(**kwargs)
    logger = logging.getLogger("archive_comparison")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    outputs = experiment(config=config, logger=logger)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for name in outputs.__dataclass_fields__:
        value = getattr(outputs, name)
        if isinstance(value, pd.DataFrame):
            value.to_pickle(config.output_dir / (name + ".pkl"))


def extract_inputs(archive: Path, workspace: Path, dataset: str) -> None:
    """Extract only the selected reference modules and required private inputs."""
    family, input_name, schema_name = DATASETS[dataset]
    preprocessing = family + "_preprocessing"
    method = family + "_bootstrap_pipeline"
    prefix = (
        f"TFM_Bertaiola/3_Results_databases/{dataset.upper()}/Methods/{method}/src/"
    )
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if not member.filename.startswith(prefix) or not member.filename.endswith(
                ".py"
            ):
                continue
            relative = PurePosixPath(member.filename[len(prefix) :])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unexpected reference module path")
            output = workspace / "original_src" / Path(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(source.read(member))
        (workspace / "inputs").mkdir()
        for filename in (input_name, schema_name):
            if filename is None:
                continue
            member = f"TFM_Bertaiola/1_Data/{preprocessing}/data/processed/{filename}"
            with (
                source.open(member) as input_stream,
                (workspace / "inputs" / filename).open("wb") as output_stream,
            ):
                while chunk := input_stream.read(1024 * 1024):
                    output_stream.write(chunk)


def compare(args: argparse.Namespace) -> dict:
    import pandas as pd

    records = []
    with tempfile.TemporaryDirectory(prefix="thesis_comparison_") as directory:
        workspace = Path(directory)
        extract_inputs(args.archive, workspace, args.dataset)
        environment = os.environ.copy()
        environment.update(
            OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"
        )
        environment.setdefault("MPLCONFIGDIR", str(workspace / "matplotlib"))
        for version in ("original", "refactored"):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--dataset",
                args.dataset,
                "--methods",
                args.methods,
                "--seed",
                str(args.seed),
                "--worker-mode",
                version,
                "--working-dir",
                str(workspace),
            ]
            if args.target_sites:
                command += ["--target-sites", args.target_sites]
            subprocess.run(command, env=environment, check=True)
        for original in sorted((workspace / "original").glob("*.pkl")):
            reference = pd.read_pickle(original)
            refactored = pd.read_pickle(workspace / "refactored" / original.name)
            pd.testing.assert_frame_equal(reference, refactored, check_exact=True)
            records.append(
                dict(
                    table=original.stem,
                    rows=len(reference),
                    columns=len(reference.columns),
                    comparison="exact",
                )
            )
    return dict(
        dataset=args.dataset,
        methods=args.methods.split(","),
        seed=args.seed,
        target_sites=args.target_sites.split(",") if args.target_sites else "all",
        tables=records,
    )


def main() -> None:
    args = parse_args()
    if args.worker_mode:
        run_worker(args)
        return
    report = compare(args)
    serialized = json.dumps(report, indent=2) + "\n"
    print(serialized)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
