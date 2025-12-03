import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .metrics import cross_entropy

Pair = Tuple[str, str]

def build_pairs(text: str) -> List[Pair]:
    cleaned = text.replace("\r", "")
    pairs: List[Pair] = []
    for i in range(len(cleaned) - 1):
        x = cleaned[i]
        y = cleaned[i + 1]
        pairs.append((x, y))
    return pairs

def train_test_split(pairs: List[Pair], frac_train: float = 0.8, seed: int = 441) -> Tuple[List[Pair], List[Pair]]:
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * frac_train)
    return shuffled[:n_train], shuffled[n_train:]

@dataclass
class BigramModel:
    counts: Dict[str, Dict[str, int]]
    cond: Dict[str, Dict[str, float]]
    alphabet: List[str]

    @classmethod
    def from_pairs(cls, pairs: List[Pair], alpha: float = 1.0) -> "BigramModel":
        counts: Dict[str, Dict[str, int]] = {}
        alphabet_set = set()
        for x, y in pairs:
            alphabet_set.add(x)
            alphabet_set.add(y)
            inner = counts.get(x)
            if inner is None:
                inner = {}
                counts[x] = inner
            inner[y] = inner.get(y, 0) + 1
        alphabet = sorted(alphabet_set)
        v = len(alphabet)
        if v == 0:
            return cls(counts={}, cond={}, alphabet=[])
        cond: Dict[str, Dict[str, float]] = {}
        for x, inner in counts.items():
            total = float(sum(inner.values()))
            denom = total + alpha * float(v)
            cond_x: Dict[str, float] = {}
            for y in alphabet:
                c_xy = inner.get(y, 0)
                cond_x[y] = (float(c_xy) + alpha) / denom
            cond[x] = cond_x
        return cls(counts=counts, cond=cond, alphabet=alphabet)

    def prob(self, y: str, x: str) -> float:
        if not self.alphabet:
            return 0.0
        inner = self.cond.get(x)
        if inner is None:
            return 1.0 / float(len(self.alphabet))
        p = inner.get(y)
        if p is None:
            return 1.0 / float(len(self.alphabet))
        return p

    def sample(self, length: int = 400, seed: int = 262) -> str:
        rng = random.Random(seed)
        if not self.alphabet:
            return ""
        current = rng.choice(self.alphabet)
        out = [current]
        for _ in range(length - 1):
            inner = self.cond.get(current)
            if not inner:
                current = rng.choice(self.alphabet)
                out.append(current)
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
            current = chosen
        return "".join(out)

def build_alphabet(pairs: List[Pair]) -> List[str]:
    symbols = set()
    for x, y in pairs:
        symbols.add(x)
        symbols.add(y)
    return sorted(symbols)

def uniform_cross_entropy(pairs: List[Pair]) -> float:
    alphabet = build_alphabet(pairs)
    v = len(alphabet)
    if v == 0:
        raise ValueError("Empty alphabet")
    p = 1.0 / float(v)
    return -math.log(p)

def main() -> None:
    raw_text = (
        "Large language models are conditional token generators. "
        "They map a context to a distribution over the next token. "
        "This demo uses a simple bigram model to make that concrete. "
    )
    print("Building corpus D from raw text")
    pairs = build_pairs(raw_text)
    print(f"|D| = {len(pairs)} pairs")
    train_pairs, test_pairs = train_test_split(pairs, frac_train=0.8)
    model = BigramModel.from_pairs(train_pairs, alpha=1.0)
    h_train = cross_entropy(model.prob, train_pairs)
    h_test = cross_entropy(model.prob, test_pairs)
    pp_train = math.exp(h_train)
    pp_test = math.exp(h_test)
    h_unif = uniform_cross_entropy(test_pairs)
    pp_unif = math.exp(h_unif)
    print("Training bigram MLE model p_θ(y | x)")
    print(f"Train cross-entropy H_train ≈ {h_train:.4f} nats")
    print(f"Test  cross-entropy H_test  ≈ {h_test:.4f} nats")
    print(f"Train perplexity ≈ {pp_train:.4f}")
    print(f"Test  perplexity ≈ {pp_test:.4f}")
    print(f"Uniform baseline H_unif ≈ {h_unif:.4f} nats  PP_unif ≈ {pp_unif:.4f}")
    print(f"Information gain ΔH = H_unif - H_test ≈ {h_unif - h_test:.4f} nats")
    print()
    print("Sampling from the model")
    sample = model.sample(length=400)
    print()
    print(sample)

if __name__ == "__main__":
    main()
