"""
Stratified k-fold cross validation utilities (binary labels 0/1).
"""

from __future__ import annotations
from typing import List, Tuple, Sequence
import random


def stratified_kfold_indices(labels: Sequence[int],k: int = 10,seed: int = 0) -> List[Tuple[List[int], List[int]]]:
    """Return (train_idx, test_idx) splits for stratified k-fold CV."""
    if k < 2:
        raise ValueError("k must be >= 2")

    # Split indices by class so each fold keeps a similar class balance.
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]

    # Shuffle with a fixed seed for reproducibility.
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    # Buckets for each fold's test indices.
    pos_chunks = [[] for _ in range(k)]
    neg_chunks = [[] for _ in range(k)]

    # Distribute indices round-robin to keep fold sizes balanced.
    for j, idx in enumerate(pos):
        pos_chunks[j % k].append(idx)
    for j, idx in enumerate(neg):
        neg_chunks[j % k].append(idx)

    folds: List[Tuple[List[int], List[int]]] = []
    all_idx = set(range(len(labels)))  # all example indices

    for i in range(k):
        # Fold i test set = i-th pos chunk + i-th neg chunk.
        test = pos_chunks[i] + neg_chunks[i]
        test_set = set(test)

        # Train set = everything not in the test set.
        train = [idx for idx in all_idx if idx not in test_set]
        folds.append((train, test))

    return folds
