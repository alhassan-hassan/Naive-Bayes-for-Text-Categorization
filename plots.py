"""
Plotting utilities (matplotlib only).
"""

from __future__ import annotations
from typing import Dict, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_learning_curve(curve_by_m: Dict[float, Tuple[np.ndarray, np.ndarray]], fractions: np.ndarray,
                        title: str, outpath: str) -> None:
    """
    curve_by_m[m] = (mean, std)
    """
    plt.figure()
    for m, (mean, std) in curve_by_m.items():
        plt.errorbar(fractions, mean, yerr=std, marker='o', capsize=3, label=f"m={m}")
    plt.xlabel("Training fraction (of fold training set)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_smoothing_sweep(ms: np.ndarray, mean: np.ndarray, std: np.ndarray,
                         title: str, outpath: str) -> None:
    plt.figure()
    plt.errorbar(ms, mean, yerr=std, marker='o', capsize=3)
    plt.xlabel("Smoothing parameter m")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
