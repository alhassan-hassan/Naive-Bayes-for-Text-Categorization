"""
Extra credit experiments: optional preprocessing variants.

IMPORTANT: The main assignment says do NOT manipulate the data.
Only use these experiments for the extra credit section.
"""

from __future__ import annotations
from typing import Callable, List, Dict, Tuple
import re
import numpy as np

from data import Example, load_labelled_sentences
from cv import stratified_kfold_indices
from experiments import smoothing_sweep, summarize_sweep, smoothing_values
from plots import plot_smoothing_sweep


# --- Preprocessing variants ---

_punct_re = re.compile(r"^[\W_]+|[\W_]+$")  # trim non-alnum at token edges


def tok_lower(sentence: str) -> List[str]:
    return sentence.strip().lower().split()


def tok_lower_strip_edges(sentence: str) -> List[str]:
    toks = sentence.strip().lower().split()
    return [_punct_re.sub("", t) for t in toks if _punct_re.sub("", t) != ""]


def tok_lower_split_punct(sentence: str) -> List[str]:
    """
    Split punctuation into separate tokens using a simple regex.
    Example: "good!!!" -> ["good", "!", "!", "!"]
    """
    s = sentence.strip().lower()
    # Keep words and punctuation as separate tokens
    return re.findall(r"[a-z0-9]+|[^\s\w]", s)


def tok_lower_negation_bigram(sentence: str, window: int = 2) -> List[str]:
    """
    Simple negation marking: join 'not'/'no' with next few tokens.
    Example: "not good at all" -> ["not_good", "at", "all"] (window=1 would only affect next token)
    """
    toks = sentence.strip().lower().split()
    out = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in ("not", "no", "never") and i + 1 < len(toks):
            # attach to next token only (window can be >1 but keep conservative)
            jmax = min(len(toks), i + 1 + window)
            # only attach to immediate next token; other tokens pass through
            out.append(f"{t}_{toks[i+1]}")
            i += 2
            continue
        out.append(t)
        i += 1
    return out


def variants() -> List[Tuple[str, Callable[[str], List[str]]]]:
    return [
        ("baseline_verbatim", None),  # special: uses default in loader
        ("lower", tok_lower),
        ("lower_strip_edges", tok_lower_strip_edges),
        ("lower_split_punct", tok_lower_split_punct),
        ("lower_negation_bigram", tok_lower_negation_bigram),
    ]


def run_extra_credit(data_path: str, out_dir: str, seed: int = 0) -> List[Tuple[str, float, float, float]]:
    """
    For each preprocessing variant, run a smoothing sweep and record best mean accuracy.
    Returns list of (variant_name, best_mean, best_std_at_best_m).
    Also saves one plot per variant.
    """
    results_table = []
    for name, tokenizer in variants():
        examples = load_labelled_sentences(data_path, tokenizer=tokenizer)
        labels = [e.label for e in examples]
        folds = stratified_kfold_indices(labels, k=10, seed=seed)

        # Sweep m to find best with this preprocessing
        m_list = smoothing_values()  # smaller set for speed in extra credit
        sweep = smoothing_sweep(examples, folds, m_list=m_list, seed=seed)
        ms, mean, std = summarize_sweep(sweep)

        # best by mean
        best_idx = int(np.argmax(mean))
        best_mean = float(mean[best_idx])
        best_std = float(std[best_idx])
        best_m = float(ms[best_idx])
        results_table.append((name, best_m, best_mean, best_std))

        plot_path = f"{out_dir}/extra_{name}_sweep.png"
        plot_smoothing_sweep(ms, mean, std, title=f"Extra credit sweep ({name})", outpath=plot_path)

    # sort by best_mean descending
    results_table.sort(key=lambda x: x[2], reverse=True)
    return results_table
