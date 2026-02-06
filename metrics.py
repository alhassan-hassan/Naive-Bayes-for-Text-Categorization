"""
Basic metrics.
"""

from __future__ import annotations
from typing import List


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    if len(y_true) == 0:
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)
