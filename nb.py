"""
Naive Bayes for text categorization (bag-of-words multinomial model).

Implements:
- ML estimate (m=0)
- MAP/Dirichlet smoothing (m>0)

Prediction is done in log-space:
  score(c|x) = log P(c) + sum_{w in x ∩ V} log P(w|c)

Tokens not seen in training (not in V) are skipped, per the assignment.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple
import math


# Label constants (match the dataset convention).
NEG = 0
POS = 1


@dataclass
class NBModel:
    """Container for a trained Naive Bayes model."""
    log_prior: Dict[int, float]
    vocab: set
    log_likelihood: Dict[int, Dict[str, float]]


class NaiveBayesText:
    """Multinomial Naive Bayes for tokenized text."""
    def __init__(self):
        self.model: NBModel | None = None

    def fit(self, examples: Iterable[Tuple[List[str], int]], m: float = 0.0) -> NBModel:
        """Train the model from (tokens, label) examples with smoothing parameter m."""
        if m < 0:
            raise ValueError("m must be >= 0")

        # Count documents per class and tokens per class.
        class_counts = {NEG: 0, POS: 0}
        token_counts = {NEG: {}, POS: {}}  # token -> count within class
        total_tokens = {NEG: 0, POS: 0}
        vocab = set()

        # Build vocabulary and class-conditional token counts.
        for tokens, y in examples:
            if y not in (NEG, POS):
                raise ValueError(f"Label must be 0/1, got {y}")

            class_counts[y] += 1

            for w in tokens:
                vocab.add(w)
                d = token_counts[y]
                d[w] = d.get(w, 0) + 1
                total_tokens[y] += 1

        # Guard against empty training input.
        n = class_counts[NEG] + class_counts[POS]
        if n == 0:
            raise ValueError("No training examples")

        # Estimate class priors using ML.
        log_prior = {
            NEG: math.log(class_counts[NEG] / n) if class_counts[NEG] > 0 else float("-inf"),
            POS: math.log(class_counts[POS] / n) if class_counts[POS] > 0 else float("-inf"),
        }

        V = len(vocab)

        # Precompute log P(w|c) for all w in vocab and each class.
        log_likelihood: Dict[int, Dict[str, float]] = {NEG: {}, POS: {}}
        for c in (NEG, POS):
            denom = total_tokens[c] + m * V

            # If a class has no training examples, its likelihoods stay empty.
            if denom <= 0:
                continue

            for w in vocab:
                cnt = token_counts[c].get(w, 0)
                num = cnt + m

                # With m=0 this yields -inf for unseen-in-class words.
                if num == 0:
                    log_likelihood[c][w] = float("-inf")
                else:
                    log_likelihood[c][w] = math.log(num / denom)

        # Store the trained model.
        self.model = NBModel(log_prior=log_prior, vocab=vocab, log_likelihood=log_likelihood)
        return self.model

    def predict(self, tokens: List[str]) -> int:
        """Predict NEG/POS for one tokenized example."""
        if self.model is None:
            raise RuntimeError("Model not fit")

        score_neg = self._score(tokens, NEG)
        score_pos = self._score(tokens, POS)

        # Break ties deterministically.
        return POS if score_pos >= score_neg else NEG

    def _score(self, tokens: List[str], c: int) -> float:
        """Compute the log-score for class c."""
        assert self.model is not None

        s = self.model.log_prior.get(c, float("-inf"))
        ll = self.model.log_likelihood.get(c, {})
        vocab = self.model.vocab

        # Add log-likelihoods for tokens that exist in the training vocabulary.
        for w in tokens:
            if w not in vocab:
                continue

            lw = ll.get(w, float("-inf"))

            # If any term is -inf, the whole score becomes -inf.
            if s == float("-inf") or lw == float("-inf"):
                s = float("-inf")
            else:
                s += lw

        return s
