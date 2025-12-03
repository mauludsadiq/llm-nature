"""Base sets and type aliases for the LLM Nature framework.

We mirror the formal objects:

- Σ  : finite token alphabet
- X  : Σ*  (set of all finite token sequences, here as Python strings)
- Y  : Σ   (single-token space)
- D  ⊂ X × Y  (corpus of (context, next-token) pairs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Iterable, NewType

Token = NewType("Token", str)
Context = NewType("Context", str)


@dataclass
class Alphabet:
    """Finite token alphabet Σ.

    In this simple implementation, Σ is a set of Unicode strings (tokens).
    """

    tokens: List[Token]

    @classmethod
    def from_text(cls, text: str) -> "Alphabet":
        # Here we use characters as tokens for simplicity.
        uniq = sorted(set(text))
        return cls(tokens=[Token(t) for t in uniq])

    def index(self) -> Dict[Token, int]:
        return {t: i for i, t in enumerate(self.tokens)}

    def __len__(self) -> int:  # |Σ|
        return len(self.tokens)

    def __contains__(self, t: Token) -> bool:
        return t in self.tokens

    def ensure_token(self, t: Token) -> Token:
        if t not in self.tokens:
            raise ValueError(f"Token {t!r} not in alphabet Σ.")
        return t
