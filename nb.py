"""
Naive Bayes for text categorization (bag-of-words multinomial model).

Implements:
- ML estimate (m=0)
- MAP/Dirichlet smoothing (m>0): P(w|c) = (count(w,c)+m) / (T_c + m*|V|)

Prediction in log-space:
score(c|x) = log P(c) + sum_{w in x ∩ V} log P(w|c)
Unknown words (not in V at all) are skipped, as required by the assignment.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple
import math


NEG = 0
POS = 1


@dataclass
class NBModel:
    # log priors
    log_prior: Dict[int, float]
    # vocabulary set (words seen in training)
    vocab: set
    # log P(w|c): per-class dict, missing => -inf (only possible when m=0)
    log_likelihood: Dict[int, Dict[str, float]]


class NaiveBayesText:
    def __init__(self):
        self.model: NBModel | None = None

    def fit(self, examples: Iterable[Tuple[List[str], int]], m: float = 0.0) -> NBModel:
        """
        Train Naive Bayes model with smoothing parameter m.
        """
        if m < 0:
            raise ValueError("m must be >= 0")

        class_counts = {NEG: 0, POS: 0}
        token_counts = {NEG: {}, POS: {}}  # type: ignore[var-annotated]
        total_tokens = {NEG: 0, POS: 0}
        vocab = set()

        for tokens, y in examples:
            if y not in (NEG, POS):
                raise ValueError(f"Label must be 0/1, got {y}")
            class_counts[y] += 1
            for w in tokens:
                vocab.add(w)
                d = token_counts[y]
                d[w] = d.get(w, 0) + 1
                total_tokens[y] += 1

        n = class_counts[NEG] + class_counts[POS]
        if n == 0:
            raise ValueError("No training examples")

        # Priors (no smoothing requested in spec; use ML prior)
        log_prior = {
            NEG: math.log(class_counts[NEG] / n) if class_counts[NEG] > 0 else float("-inf"),
            POS: math.log(class_counts[POS] / n) if class_counts[POS] > 0 else float("-inf"),
        }

        V = len(vocab)

        # Likelihoods
        log_likelihood = {NEG: {}, POS: {}}
        for c in (NEG, POS):
            denom = total_tokens[c] + m * V
            # denom can be 0 if a class has 0 examples and m=0; handle gracefully
            if denom <= 0:
                # no support for this class; keep empty likelihood dict
                continue
            for w in vocab:
                cnt = token_counts[c].get(w, 0)
                num = cnt + m
                if num == 0:
                    # only possible when m=0 and cnt=0
                    log_likelihood[c][w] = float("-inf")
                else:
                    log_likelihood[c][w] = math.log(num / denom)

        self.model = NBModel(log_prior=log_prior, vocab=vocab, log_likelihood=log_likelihood)
        return self.model

    def predict(self, tokens: List[str]) -> int:
        if self.model is None:
            raise RuntimeError("Model not fit")

        score_neg = self._score(tokens, NEG)
        score_pos = self._score(tokens, POS)

        # Tie-break: predict positive if equal (arbitrary but deterministic)
        return POS if score_pos >= score_neg else NEG

    def _score(self, tokens: List[str], c: int) -> float:
        assert self.model is not None
        s = self.model.log_prior.get(c, float("-inf"))
        ll = self.model.log_likelihood.get(c, {})
        vocab = self.model.vocab

        for w in tokens:
            if w not in vocab:
                # Required: skip words not seen in training at all (in any class)
                continue
            lw = ll.get(w, float("-inf"))
            # Add -inf safely: if s is already -inf, stays -inf
            if s == float("-inf") or lw == float("-inf"):
                s = float("-inf")
            else:
                s += lw
        return s
