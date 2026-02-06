"""
Data loading and tokenization utilities for PP1 Naive Bayes sentiment classifier.

No preprocessing here; just verbatim tokenization and dataset loading.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
import os

@dataclass(frozen=True)
class Example:
    tokens: List[str]   # tokenized sentence
    label: int          # 1=positive, 0=negative


def tokenize_verbatim(sentence: str) -> List[str]:
    """Split on whitespace and keep tokens exactly as they appear."""
    return sentence.strip().split()


def load_labelled_sentences(
    path: str,
    tokenizer: Optional[Callable[[str], List[str]]] = None
) -> List[Example]:
    """Load a dataset file where each line is '<sentence>\\t<label>'."""
    if tokenizer is None:
        tokenizer = tokenize_verbatim 

    examples: List[Example] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            # remove trailing newline only
            line = line.rstrip("\n")  
            if not line.strip():
                # skip empty lines
                continue  

            if "\t" not in line:
                raise ValueError(f"Line {line_no} missing tab separator in {path!r}: {line!r}")

            # split from the right to protect sentence text
            sent, lab = line.rsplit("\t", 1)  
            lab = lab.strip()
            if lab not in ("0", "1"):
                raise ValueError(f"Line {line_no} invalid label {lab!r} in {path!r}")

            # tokenize sentence text
            tokens = tokenizer(sent)  
            examples.append(Example(tokens=tokens, label=int(lab)))

    return examples


def dataset_paths(data_dir: str) -> List[Tuple[str, str]]:
    """Return (dataset_name, file_path) for the three required datasets."""
    return [
        ("amazon", os.path.join(data_dir, "amazon_cells_labelled.txt")),
        ("imdb", os.path.join(data_dir, "imdb_labelled.txt")),
        ("yelp", os.path.join(data_dir, "yelp_labelled.txt")),
    ]
