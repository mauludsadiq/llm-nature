import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

Pair = Tuple[str, str]

def build_ngram_pairs(text: str, k: int) -> List[Pair]:
    cleaned = text.replace("\r", "")
    pairs: List[Pair] = []
    if k <= 0 or len(cleaned) <= k:
        return pairs
    for i in range(k, len(cleaned)):
        x = cleaned[i - k:i]
        y = cleaned[i]
        pairs.append((x, y))
    return pairs

@dataclass
class NGramModel:
    k: int
    counts: Dict[str, Dict[str, int]]
    cond: Dict[str, Dict[str, float]]
    alphabet: List[str]

    @classmethod
    def from_pairs(cls, pairs: List[Pair], k: int, alpha: float = 1.0) -> "NGramModel":
        counts: Dict[str, Dict[str, int]] = {}
        alphabet_set = set()
        for x, y in pairs:
            for ch in x:
                alphabet_set.add(ch)
            alphabet_set.add(y)
            inner = counts.get(x)
            if inner is None:
                inner = {}
                counts[x] = inner
            inner[y] = inner.get(y, 0) + 1
        alphabet = sorted(alphabet_set)
        v = len(alphabet)
        cond: Dict[str, Dict[str, float]] = {}
        if v == 0:
            return cls(k=k, counts=counts, cond=cond, alphabet=alphabet)
        for x, inner in counts.items():
            total = float(sum(inner.values()))
            denom = total + alpha * float(v)
            cond_x: Dict[str, float] = {}
            for y in alphabet:
                c_xy = inner.get(y, 0)
                cond_x[y] = (float(c_xy) + alpha) / denom
            cond[x] = cond_x
        return cls(k=k, counts=counts, cond=cond, alphabet=alphabet)

    def prob(self, y: str, x: str) -> float:
        v = len(self.alphabet)
        if v == 0:
            return 0.0
        inner = self.cond.get(x)
        if inner is None:
            return 1.0 / float(v)
        p = inner.get(y)
        if p is None:
            return 1.0 / float(v)
        return p

    def sample(self, length: int = 400, seed: int = 262) -> str:
        rng = random.Random(seed)
        if not self.alphabet:
            return ""
        if not self.counts:
            return "".join(rng.choice(self.alphabet) for _ in range(length))
        contexts = list(self.counts.keys())
        context = rng.choice(contexts)
        out = list(context)
        while len(out) < length:
            inner = self.cond.get(context)
            if not inner:
                out.append(rng.choice(self.alphabet))
                context = "".join(out[-self.k:])
                continue
            ys = list(inner.keys())
            ps = [inner[y] for y in ys]
            r = rng.random()
            acc = 0.0
            chosen = ys[-1]
            for y, p in zip(ys, ps):
                acc += p
                if r <= acc:
                    chosen = y
                    break
            out.append(chosen)
            context = "".join(out[-self.k:])
        return "".join(out[:length])
