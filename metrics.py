"""
Basic metrics.
"""

from __future__ import annotations
from typing import List


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Return classification accuracy (fraction of correct predictions)."""
    # Handle the edge case where there are no examples.
    if len(y_true) == 0:
        return 0.0

    # Count how many predictions match the true labels.
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)

    # Accuracy is correct / total.
    return correct / len(y_true)
