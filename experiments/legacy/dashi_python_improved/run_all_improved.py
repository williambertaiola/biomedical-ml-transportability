"""Transportability experiment: run all improved."""

from __future__ import annotations

import importlib
import time

from common_pipeline import get_output_root, write_text

PIPELINES = [
    ("unsupervised_uci", "UCI unsupervised"),
    ("supervised_uci", "UCI supervised"),
    ("unsupervised_abide2", "ABIDE2 unsupervised"),
    ("supervised_abide2", "ABIDE2 supervised"),
    ("unsupervised_tcga", "TCGA unsupervised"),
    ("supervised_tcga", "TCGA supervised"),
]


def main() -> None:
    output_root = get_output_root()
    logs: list[str] = []

    for module_name, label in PIPELINES:
        start = time.time()
        logs.append(f"## {label}\n")
        try:
            module = importlib.import_module(module_name)
            module.main()
            elapsed = time.time() - start
            logs.append(f"- Stato: completato\n- Tempo: {elapsed:.1f} s\n")
        except Exception as exc:  # pragma: no cover
            elapsed = time.time() - start
            logs.append(
                f"- Stato: errore\n- Tempo: {elapsed:.1f} s\n- Eccezione: {type(exc).__name__}: {exc}\n"
            )
            write_text("\n".join(logs), output_root / "run_summary.md")
            raise

    write_text("\n".join(logs), output_root / "run_summary.md")
    print(f"Pipeline completate. Output root: {output_root}")


if __name__ == "__main__":
    main()
