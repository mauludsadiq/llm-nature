import math
from typing import List, Tuple

from .demo import build_pairs, train_test_split
from .metrics import cross_entropy
from .ngram import NGramModel, build_ngram_pairs
from .demo import BigramModel, uniform_cross_entropy

Pair = Tuple[str, str]

def uniform_cross_entropy_from_pairs_k(pairs: List[Pair], k: int) -> float:
    alphabet = set()
    for x, y in pairs:
        for ch in x:
            alphabet.add(ch)
        alphabet.add(y)
    v = len(alphabet)
    if v == 0:
        raise ValueError("Empty alphabet")
    p = 1.0 / float(v)
    return -math.log(p)

def run_for_k(raw_text: str, k: int) -> None:
    if k == 1:
        pairs = build_pairs(raw_text)
        print(f"k = {k}: building bigram pairs")
        print(f"|D_1| = {len(pairs)} pairs")
        train_pairs, test_pairs = train_test_split(pairs, frac_train=0.8)
        model = BigramModel.from_pairs(train_pairs, alpha=1.0)
        h_train = cross_entropy(model.prob, train_pairs)
        h_test = cross_entropy(model.prob, test_pairs)
        h_unif = uniform_cross_entropy(test_pairs)
    else:
        pairs = build_ngram_pairs(raw_text, k=k)
        print(f"k = {k}: building {k}-gram pairs")
        print(f"|D_{k}| = {len(pairs)} pairs")
        train_pairs, test_pairs = train_test_split(pairs, frac_train=0.8)
        model = NGramModel.from_pairs(train_pairs, k=k, alpha=1.0)
        h_train = cross_entropy(model.prob, train_pairs)
        h_test = cross_entropy(model.prob, test_pairs)
        h_unif = uniform_cross_entropy_from_pairs_k(test_pairs, k)
    pp_train = math.exp(h_train)
    pp_test = math.exp(h_test)
    pp_unif = math.exp(h_unif)
    delta_h = h_unif - h_test
    overfit_gap = h_test - h_train
    print(f"  Train H ≈ {h_train:.4f} nats, PP ≈ {pp_train:.4f}")
    print(f"  Test  H ≈ {h_test:.4f} nats, PP ≈ {pp_test:.4f}")
    print(f"  Uniform H_unif ≈ {h_unif:.4f} nats, PP_unif ≈ {pp_unif:.4f}")
    print(f"  Information gain ΔH ≈ {delta_h:.4f} nats")
    print(f"  Overfitting gap H_test - H_train ≈ {overfit_gap:.4f} nats")
    print()

def main() -> None:
    raw_text = (
        "Large language models are conditional token generators. "
        "They map a context to a distribution over the next token. "
        "This demo uses a simple n-gram character model to make that concrete. "
    )
    for k in [1, 2, 3, 4]:
        run_for_k(raw_text, k)

if __name__ == "__main__":
    main()
