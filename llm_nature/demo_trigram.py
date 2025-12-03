import math
from typing import List, Tuple

from .metrics import cross_entropy
from .ngram import NGramModel, build_ngram_pairs
from .demo import train_test_split

Pair = Tuple[str, str]

def uniform_cross_entropy_from_pairs(pairs: List[Pair]) -> float:
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

def main() -> None:
    raw_text = (
        "Large language models are conditional token generators. "
        "They map a context to a distribution over the next token. "
        "This demo uses a simple trigram character model to show longer-range structure. "
    )
    k = 3
    print("Building corpus D_k from raw text")
    pairs = build_ngram_pairs(raw_text, k=k)
    print(f"|D_k| = {len(pairs)} pairs with context length k = {k}")
    train_pairs, test_pairs = train_test_split(pairs, frac_train=0.8)
    model = NGramModel.from_pairs(train_pairs, k=k, alpha=1.0)
    h_train = cross_entropy(model.prob, train_pairs)
    h_test = cross_entropy(model.prob, test_pairs)
    pp_train = math.exp(h_train)
    pp_test = math.exp(h_test)
    h_unif = uniform_cross_entropy_from_pairs(test_pairs)
    pp_unif = math.exp(h_unif)
    print("Training trigram MLE model p_θ(y | x)")
    print(f"Train cross-entropy H_train ≈ {h_train:.4f} nats")
    print(f"Test  cross-entropy H_test  ≈ {h_test:.4f} nats")
    print(f"Train perplexity ≈ {pp_train:.4f}")
    print(f"Test  perplexity ≈ {pp_test:.4f}")
    print(f"Uniform baseline H_unif ≈ {h_unif:.4f} nats  PP_unif ≈ {pp_unif:.4f}")
    print(f"Information gain ΔH = H_unif - H_test ≈ {h_unif - h_test:.4f} nats")
    print()
    print("Sampling from the trigram model")
    sample = model.sample(length=400)
    print()
    print(sample)

if __name__ == "__main__":
    main()
