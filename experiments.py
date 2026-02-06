"""
Experiment runners:
- Learning curves with stratified CV (train fractions 0.1..0.9)
- Smoothing sweep (m values per spec)
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Sequence
import random
import numpy as np

from nb import NaiveBayesText
from metrics import accuracy


def evaluate_on_split(examples, train_idx: List[int], test_idx: List[int], m: float, seed: int) -> float:
    """Train on train_idx and return accuracy on test_idx."""
    # build (tokens,label) pairs
    nb = NaiveBayesText()
    train_examples = [(examples[i].tokens, examples[i].label) for i in train_idx]  
    nb.fit(train_examples, m=m)  # fit NB with smoothing m

    y_true = []
    y_pred = []
    for i in test_idx:
        # gold label
        y_true.append(examples[i].label)    
        # predicted label          
        y_pred.append(nb.predict(examples[i].tokens)) 

    return accuracy(y_true, y_pred)


def learning_curves(examples,folds: Sequence[Tuple[List[int], List[int]]],m_values=(0.0, 1.0),fractions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),seed: int = 0) -> Dict[float, Dict[float, Dict[str, List[float]]]]:
    """Compute learning curves: accuracy vs training fraction for each m."""
    results: Dict[float, Dict[float, Dict[str, List[float]]]] = {}
    # reserved for reproducibility (not strictly needed below)
    rng = random.Random(seed)  

    # Initialize storage for all (m, fraction) combinations.
    for m in m_values:
        results[m] = {}
        for frac in fractions:
            # one accuracy per fold
            results[m][frac] = {"acc": []}  

    for fold_id, (train_idx_full, test_idx) in enumerate(folds):
        # copy so shuffling doesn't affect input folds
        train_idx = list(train_idx_full)  

        # Shuffle training indices within the fold so subsets are random (per spec).
        rng_fold = random.Random(seed + 1000 * (fold_id + 1))
        rng_fold.shuffle(train_idx)

        for m in m_values:
            for frac in fractions:
                # number of training points for this fraction
                n_train = int(frac * len(train_idx)) 
                # avoid empty training sets 
                n_train = max(1, n_train)
                # prefix subset after shuffle             
                sub_train = train_idx[:n_train]       

                acc = evaluate_on_split(examples, sub_train, test_idx, m=m, seed=seed)
                # store fold accuracy
                results[m][frac]["acc"].append(acc)   

    return results


def summarize_curve(curve_results: Dict[float, Dict[str, List[float]]]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert per-fold curve results into mean/std arrays over folds."""
    # keep x-axis in increasing order
    fracs = sorted(curve_results.keys())  
    means = []
    stds = []
    for f in fracs:
        # accuracies across folds
        accs = np.array(curve_results[f]["acc"], dtype=float)  
        # mean accuracy
        means.append(accs.mean()) 
        # population std across folds                              
        stds.append(accs.std(ddof=0))                          
    return np.array(means), np.array(stds)


def smoothing_values() -> List[float]:
    """Return the required m grid: 0,0.1,...,0.9 and 1,2,...,10."""
    # 0.0..0.9
    vals = [round(0.1 * i, 1) for i in range(0, 10)]  
     # 1..10
    vals += list(range(1, 11))                       
    return [float(v) for v in vals]


def smoothing_sweep(examples,folds: Sequence[Tuple[List[int], List[int]]],m_list: List[float],seed: int = 0) -> Dict[float, List[float]]:
    """Evaluate each m on each fold using the full fold training split."""
    # m -> [acc per fold]
    res: Dict[float, List[float]] = {m: [] for m in m_list}  

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        for m in m_list:
            acc = evaluate_on_split(examples, train_idx, test_idx, m=m, seed=seed)
            # store fold accuracy for this m
            res[m].append(acc)  

    return res


def summarize_sweep(sweep_results: Dict[float, List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert sweep results into sorted m values plus mean/std arrays."""
    # x-axis (m)
    ms = np.array(sorted(sweep_results.keys()), dtype=float)      
    # mean over folds               
    means = np.array([np.mean(sweep_results[m]) for m in ms], dtype=float)      
    # std over folds 
    stds = np.array([np.std(sweep_results[m], ddof=0) for m in ms], dtype=float) 
    return ms, means, stds
