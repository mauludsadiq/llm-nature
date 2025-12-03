"""Parametric model class p_θ(y | x) and a concrete Bigram model.

This corresponds to a restricted architecture in the formal framework:
- θ: parameters (here, bigram conditional counts)
- For each context x, we approximate p(y|x) ≈ p(y | last_token(x)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math
import numpy as np

from .base_sets import Token, Context, Alphabet


@dataclass
class BigramLM:
    """Simple bigram language model.

    Parameters:
        alphabet: Σ
        log_probs: matrix of shape (V, V) with log p(y | x_last)
                   (row: previous token index, col: current token index)
    """

    alphabet: Alphabet
    log_probs: np.ndarray  # shape (V, V)

    @classmethod
    def from_counts(
        cls,
        alphabet: Alphabet,
        counts: np.ndarray,
        alpha: float = 1.0,
    ) -> "BigramLM":
        """Build from bigram counts N(u, v) with additive smoothing α.

        p(v | u) = (N(u, v) + α) / (∑_w N(u, w) + α V).
        """
        if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
            raise ValueError("counts must be a square V×V matrix")
        V = counts.shape[0]
        smoothed = counts + alpha
        row_sums = smoothed.sum(axis=1, keepdims=True)
        probs = smoothed / row_sums
        log_probs = np.log(probs, dtype=float)
        return cls(alphabet=alphabet, log_probs=log_probs)

    def _last_token_index(self, context: Context) -> int:
        """Return the index of the last token of context, or 0 if empty.

        This is the projection from X = Σ* onto Y = Σ used by this model.
        """
        idx_map = self.alphabet.index()
        if len(context) == 0:
            # Use first token in Σ as a special start symbol.
            return 0
        last = context[-1]
        return idx_map[Token(last)]

    def log_prob(self, context: Context, next_token: Token) -> float:
        """Compute log p_θ(y | x)."""
        idx_map = self.alphabet.index()
        u = self._last_token_index(context)
        v = idx_map[self.alphabet.ensure_token(next_token)]
        return float(self.log_probs[u, v])

    def prob(self, context: Context, next_token: Token) -> float:
        return math.exp(self.log_prob(context, next_token))

    def next_token_distribution(self, context: Context):
        """Return p_θ(· | x) as a dictionary over Σ."""
        u = self._last_token_index(context)
        row = np.exp(self.log_probs[u, :])
        row = row / row.sum()
        return {tok: float(p) for tok, p in zip(self.alphabet.tokens, row)}

    def sample_next(self, context: Context, rng=None) -> Token:
        """Sample a single token from p_θ(· | x)."""
        if rng is None:
            rng = np.random.default_rng()
        u = self._last_token_index(context)
        row = np.exp(self.log_probs[u, :])
        row = row / row.sum()
        idx = int(rng.choice(len(self.alphabet.tokens), p=row))
        return self.alphabet.tokens[idx]
