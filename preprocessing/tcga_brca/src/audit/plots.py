"""Preprocessing: plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def _save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    output_path: Path,
    rotation: int = 45,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(df[x].astype(str), df[y], color="#4477AA")
    plt.title(title)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    _save_current_figure(output_path)


def plot_histogram(
    values, title: str, xlabel: str, output_path: Path, bins: int = 40
) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color="#228833", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    _save_current_figure(output_path)
