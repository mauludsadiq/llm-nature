"""MLE training for the Bigram model.

This is a concrete maximum-likelihood estimator within the model class.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .base_sets import Token, Context, Alphabet
from .corpus import Corpus
from .model import BigramLM


def train_bigram(corpus: Corpus, alpha: float = 1.0) -> BigramLM:
    """Fit a bigram model by counting N(u, v) on D.

    This is the empirical risk minimizer for the negative log-likelihood
        L(θ) = -∑_i log p_θ(y_i | x_i)
    within the bigram family.
    """
    alphabet = corpus.alphabet
    idx_map: Dict[Token, int] = alphabet.index()
    V = len(alphabet)
    counts = np.zeros((V, V), dtype=float)

    def last_token_index(ctx: Context) -> int:
        if len(ctx) == 0:
            return 0
        last = ctx[-1]
        return idx_map[Token(last)]

    for x, y in corpus.iter_pairs():
        u = last_token_index(x)
        v = idx_map[y]
        counts[u, v] += 1.0

    return BigramLM.from_counts(alphabet, counts, alpha=alpha)
