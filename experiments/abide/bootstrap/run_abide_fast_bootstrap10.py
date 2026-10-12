"""Transportability experiment: run abide fast bootstrap10."""

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


import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RUNNER_PATH = PROJECT_ROOT / "run_abide_bootstrap_resampling.py"
MERGE_PATH = PROJECT_ROOT / "merge_abide_bootstrap_batch_outputs.py"

TARGET_GROUPS = {
    "g1": (
        "ABIDEII-BNI_1",
        "ABIDEII-EMC_1",
        "ABIDEII-ETH_1",
        "ABIDEII-GU_1",
        "ABIDEII-IP_1",
        "ABIDEII-IU_1",
    ),
    "g2": (
        "ABIDEII-KKI_1",
        "ABIDEII-NYU_1",
        "ABIDEII-OHSU_1",
        "ABIDEII-OILH_2",
        "ABIDEII-SDSU_1",
        "ABIDEII-SU_2",
    ),
    "g3": (
        "ABIDEII-TCD_1",
        "ABIDEII-UCD_1",
        "ABIDEII-UCLA_1",
        "ABIDEII-USM_1",
        "ABIDEII-U_MIA_1",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast ABIDE bootstrap-10 orchestration with batched parallel tasks"
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("results/ABIDE/bootstrap_fast10_work"),
        help="Root directory for temporary per-task outputs",
    )
    parser.add_argument(
        "--final-output-dir",
        type=Path,
        default=Path(
            "results/ABIDE/Methods/ABIDE2_M4_ComBat_transductive_bootstrap10_outputs"
        ),
        help="Final merged output directory",
    )
    parser.add_argument(
        "--base-seed", type=int, default=42, help="Seed used for bootstrap index 1"
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=10, help="Number of bootstrap iterations"
    )
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=6,
        help="Maximum number of concurrent batch tasks",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="n_jobs passed to Random Forest inside each task",
    )
    return parser.parse_args()


def _build_task_commands(args: argparse.Namespace) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for bootstrap_index in range(1, int(args.n_bootstrap) + 1):
        bootstrap_seed = int(args.base_seed) + bootstrap_index - 1
        for group_name, sites in TARGET_GROUPS.items():
            output_dir = (
                args.work_root / f"bootstrap_{bootstrap_index:02d}_{group_name}"
            )
            command = [
                sys.executable,
                str(RUNNER_PATH),
                "--output-dir",
                str(output_dir),
                "--n-bootstrap",
                "1",
                "--bootstrap-start-index",
                str(bootstrap_index),
                "--seed",
                str(bootstrap_seed),
                "--methods",
                "M0,M1,M2,M3,M4",
                "--target-sites",
                ",".join(sites),
                "--n-jobs",
                str(args.n_jobs),
                "--skip-plots",
                "--skip-report",
            ]
            tasks.append(
                {
                    "bootstrap_index": bootstrap_index,
                    "bootstrap_seed": bootstrap_seed,
                    "group_name": group_name,
                    "sites": sites,
                    "output_dir": output_dir,
                    "command": command,
                }
            )
    return tasks


def _run_task(task: dict[str, object]) -> dict[str, object]:
    output_dir = Path(task["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    env.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, int(os.cpu_count() or 1))))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("BLIS_NUM_THREADS", "1")
    subprocess.run(task["command"], check=True, env=env)
    return task


def main() -> None:
    args = parse_args()
    args.work_root = args.work_root.resolve()
    args.final_output_dir = args.final_output_dir.resolve()
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.final_output_dir.parent.mkdir(parents=True, exist_ok=True)

    tasks = _build_task_commands(args)
    task_manifest_path = args.work_root / "task_manifest.json"
    task_manifest_path.write_text(
        json.dumps(tasks, indent=2, default=str), encoding="utf-8"
    )

    print(
        f"Launching {len(tasks)} ABIDE fast-bootstrap tasks with parallel_jobs={args.parallel_jobs}"
    )
    completed_output_dirs: list[Path] = []
    with ThreadPoolExecutor(max_workers=int(args.parallel_jobs)) as executor:
        future_to_task = {executor.submit(_run_task, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            future.result()
            completed_output_dirs.append(Path(task["output_dir"]))
            print(
                "Completed bootstrap "
                f"{task['bootstrap_index']:02d} | {task['group_name']} | seed={task['bootstrap_seed']} | "
                f"sites={','.join(task['sites'])}"
            )

    merge_command = [
        sys.executable,
        str(MERGE_PATH),
        "--output-dir",
        str(args.final_output_dir),
        "--input-dirs",
        *[str(path) for path in sorted(completed_output_dirs)],
    ]
    print("Merging batched outputs into the final ABIDE bootstrap package")
    subprocess.run(merge_command, check=True)
    print(f"Done. Final merged output: {args.final_output_dir}")


if __name__ == "__main__":
    main()
