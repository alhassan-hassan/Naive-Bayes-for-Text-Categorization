"""
Extra credit experiments: optional preprocessing variants I implemented.

"""

from __future__ import annotations
from typing import Callable, List, Tuple
import re
import numpy as np

from data import load_labelled_sentences
from cv import stratified_kfold_indices
from experiments import smoothing_sweep, summarize_sweep, smoothing_values
from plots import plot_smoothing_sweep


# Regex used to trim punctuation/non-alphanumeric characters at token boundaries.
_punct_re = re.compile(r"^[\W_]+|[\W_]+$")


def tok_lower(sentence: str) -> List[str]:
    """Lowercase and split on whitespace."""
    return sentence.strip().lower().split()


def tok_lower_strip_edges(sentence: str) -> List[str]:
    """Lowercase, split on whitespace, and strip punctuation from token edges."""
    toks = sentence.strip().lower().split()
    return [_punct_re.sub("", t) for t in toks if _punct_re.sub("", t) != ""]


def tok_lower_split_punct(sentence: str) -> List[str]:
    """Lowercase and split punctuation into separate tokens (e.g., 'good!!!' -> 'good', '!', '!', '!')."""
    s = sentence.strip().lower()
    return re.findall(r"[a-z0-9]+|[^\s\w]", s)


def tok_lower_negation_bigram(sentence: str, window: int = 2) -> List[str]:
    """Lowercase and mark simple negation by joining the negator with the next token (e.g., 'not_good')."""
    toks = sentence.strip().lower().split()
    out: List[str] = []
    i = 0

    while i < len(toks):
        t = toks[i]

        # If we see a negator, combine it with the following token.
        if t in ("not", "no", "never") and i + 1 < len(toks):
            out.append(f"{t}_{toks[i+1]}")
            i += 2
            continue

        out.append(t)
        i += 1

    return out


def variants() -> List[Tuple[str, Callable[[str], List[str]] | None]]:
    """Return the set of preprocessing variants to evaluate."""
    return [
        # None means: use the default "verbatim" tokenizer in the data loader.
        ("baseline_verbatim", None),
        ("lower", tok_lower),
        ("lower_strip_edges", tok_lower_strip_edges),
        ("lower_split_punct", tok_lower_split_punct),
        ("lower_negation_bigram", tok_lower_negation_bigram),
    ]


def run_extra_credit(data_path: str, out_dir: str, seed: int = 0) -> List[Tuple[str, float, float, float]]:
    """
    Run the extra-credit sweep for each preprocessing variant.
    """
    results_table: List[Tuple[str, float, float, float]] = []

    for name, tokenizer in variants():
        # Load the dataset using the chosen tokenizer.
        examples = load_labelled_sentences(data_path, tokenizer=tokenizer)

        # Build stratified folds once per variant.
        labels = [e.label for e in examples]
        folds = stratified_kfold_indices(labels, k=10, seed=seed)

        # Sweep the assignment m values and summarize across folds.
        m_list = smoothing_values()
        sweep = smoothing_sweep(examples, folds, m_list=m_list, seed=seed)
        ms, mean, std = summarize_sweep(sweep)

        # Pick the m with the best mean accuracy.
        best_idx = int(np.argmax(mean))
        best_m = float(ms[best_idx])
        best_mean = float(mean[best_idx])
        best_std = float(std[best_idx])

        results_table.append((name, best_m, best_mean, best_std))

        # Save one sweep plot per preprocessing variant.
        plot_path = f"{out_dir}/extra_{name}_sweep.png"
        plot_smoothing_sweep(ms, mean, std, title=f"Extra credit sweep ({name})", outpath=plot_path)

    # Sort variants from best to worst by mean accuracy.
    results_table.sort(key=lambda x: x[2], reverse=True)
    return results_table
