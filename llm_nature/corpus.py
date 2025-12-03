"""Corpus representation D ⊂ X × Y.

We store contexts X and next-tokens Y as parallel lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Iterable

from .base_sets import Token, Context, Alphabet


@dataclass
class Corpus:
    contexts: List[Context]
    targets: List[Token]
    alphabet: Alphabet

    def __post_init__(self) -> None:
        if len(self.contexts) != len(self.targets):
            raise ValueError("contexts and targets must have same length")
        # Ensure all targets lie in Σ
        for t in self.targets:
            self.alphabet.ensure_token(t)

    def __len__(self) -> int:  # |D|
        return len(self.contexts)

    @classmethod
    def from_text(
        cls,
        text: str,
        context_window: int = 16,
    ) -> "Corpus":
        """Build a corpus from raw text.

        We slide a window over the text; for each position i>0 we collect:
            x_i = preceding `context_window` characters
            y_i = current character
        so that (x_i, y_i) ∈ X × Y.

        This is a concrete construction of D ⊂ Σ* × Σ.
        """
        if len(text) < 2:
            raise ValueError("text must have length ≥ 2 to build a corpus")

        alphabet = Alphabet.from_text(text)
        contexts: List[Context] = []
        targets: List[Token] = []

        for i in range(1, len(text)):
            start = max(0, i - context_window)
            ctx_str = text[start:i]
            contexts.append(Context(ctx_str))
            targets.append(Token(text[i]))

        return cls(contexts=contexts, targets=targets, alphabet=alphabet)

    def iter_pairs(self) -> Iterable[Tuple[Context, Token]]:
        for x, y in zip(self.contexts, self.targets):
            yield x, y
