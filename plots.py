"""
Plotting utilities (matplotlib only).
"""

from __future__ import annotations
from typing import Dict, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def plot_learning_curve(
    curve_by_m: Dict[float, Tuple[np.ndarray, np.ndarray]],
    fractions: np.ndarray,
    title: str,
    outpath: str
) -> None:
    """Plot learning curves with error bars for each m value."""
    plt.figure()

    # Plot mean accuracy with std error bars for each smoothing setting.
    for m, (mean, std) in curve_by_m.items():
        plt.errorbar(fractions, mean, yerr=std, marker="o", capsize=3, label=f"m={m}")

    # Label axes and add standard plot formatting.
    plt.xlabel("Training fraction (of fold training set)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.ylim(0.0, 1.0)

    # Write the figure to disk.
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_smoothing_sweep(ms: np.ndarray, mean: np.ndarray, std: np.ndarray, title: str, outpath: str) -> None:
    """Plot accuracy vs smoothing parameter m with error bars."""
    plt.figure()

    # Plot mean accuracy with std error bars across folds.
    plt.errorbar(ms, mean, yerr=std, marker="o", capsize=3)

    # Label axes and add standard plot formatting.
    plt.xlabel("Smoothing parameter m")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.ylim(0.0, 1.0)

    # Write the figure to disk.
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
