"""Evaluation metrics: cross-entropy and perplexity."""

from __future__ import annotations

import math
from typing import Iterable

from .base_sets import Token, Context
from .model import BigramLM


def cross_entropy(
    model: BigramLM,
    pairs: Iterable[tuple[Context, Token]],
) -> float:
    """Empirical cross-entropy H(p_data, p_θ) in nats.

    H = - (1/N) ∑_i log p_θ(y_i | x_i).
    """
    total = 0.0
    n = 0
    for x, y in pairs:
        total += -model.log_prob(x, y)
        n += 1
    if n == 0:
        raise ValueError("empty dataset for cross-entropy evaluation")
    return total / n


def perplexity(
    model: BigramLM,
    pairs: Iterable[tuple[Context, Token]],
) -> float:
    """Perplexity = exp(H) for convenience."""
    H = cross_entropy(model, pairs)
    return math.exp(H)
