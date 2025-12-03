from __future__ import annotations

import re
import random
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

BOS = "<BOS>"

def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text)

def detokenize(tokens: Sequence[str]) -> str:
    out = ""
    for tok in tokens:
        if not out:
            out = tok
        elif tok in {".", ",", ":", ";", "!", "?", ")", "]", "}"}:
            out += tok
        elif out.endswith(("(", "[", "{")):
            out += tok
        else:
            out += " " + tok
    return out

@dataclass
class WordNGramModel:
    k: int
    alpha: float
    counts: Dict[Tuple[Tuple[str, ...], str], int]
    context_counts: Dict[Tuple[str, ...], int]
    vocab: List[str]

    @classmethod
    def from_corpus(cls, text: str, k: int = 3, alpha: float = 1.0) -> "WordNGramModel":
        tokens = tokenize(text)
        vocab_counter: Counter = Counter(tokens)
        vocab = sorted(vocab_counter.keys())
        counts: Dict[Tuple[Tuple[str, ...], str], int] = defaultdict(int)
        context_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        padded: List[str] = [BOS] * k + tokens
        for i in range(k, len(padded)):
            context = tuple(padded[i - k : i])
            y = padded[i]
            counts[(context, y)] += 1
            context_counts[context] += 1
        return cls(k=k, alpha=alpha, counts=dict(counts), context_counts=dict(context_counts), vocab=vocab)

    def _normalize_context(self, history: Sequence[str]) -> Tuple[str, ...]:
        if len(history) >= self.k:
            ctx = tuple(history[-self.k :])
        else:
            ctx = tuple([BOS] * (self.k - len(history)) + list(history))
        return ctx

    def prob(self, history: Sequence[str], y: str) -> float:
        ctx = self._normalize_context(history)
        num = self.counts.get((ctx, y), 0) + self.alpha
        denom = self.context_counts.get(ctx, 0) + self.alpha * len(self.vocab)
        return num / denom

    def sample_next(self, history: Sequence[str], rng: random.Random | None = None) -> str:
        if rng is None:
            rng = random
        ctx = self._normalize_context(history)
        probs: List[float] = []
        for v in self.vocab:
            num = self.counts.get((ctx, v), 0) + self.alpha
            denom = self.context_counts.get(ctx, 0) + self.alpha * len(self.vocab)
            probs.append(num / denom)
        return rng.choices(self.vocab, weights=probs, k=1)[0]

    def generate(self, prompt: str, max_tokens: int = 40, seed: int | None = 0) -> str:
        rng = random.Random(seed) if seed is not None else random
        tokens = tokenize(prompt)
        for _ in range(max_tokens):
            nxt = self.sample_next(tokens, rng=rng)
            tokens.append(nxt)
        return detokenize(tokens)

    def log_prob_tokens(self, tokens: Sequence[str]) -> float:
        padded: List[str] = [BOS] * self.k + list(tokens)
        logp = 0.0
        for i in range(self.k, len(padded)):
            history = padded[i - self.k : i]
            y = padded[i]
            p = self.prob(history, y)
            logp += math.log(p)
        return logp

    def log_prob_text(self, text: str) -> float:
        tokens = tokenize(text)
        return self.log_prob_tokens(tokens)
