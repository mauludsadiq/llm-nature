import math
import csv
from typing import List, Tuple

from .demo import build_pairs, train_test_split
from .metrics import cross_entropy
from .ngram import NGramModel, build_ngram_pairs
from .demo import BigramModel, uniform_cross_entropy

Pair = Tuple[str, str]

def uniform_cross_entropy_from_pairs_k(pairs: List[Pair]) -> float:
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

def metrics_for_k_seed(raw_text: str, k: int, seed: int) -> dict:
    if k == 1:
        pairs = build_pairs(raw_text)
        train_pairs, test_pairs = train_test_split(pairs, frac_train=0.8, seed=seed)
        model = BigramModel.from_pairs(train_pairs, alpha=1.0)
        h_train = cross_entropy(model.prob, train_pairs)
        h_test = cross_entropy(model.prob, test_pairs)
        h_unif = uniform_cross_entropy(test_pairs)
    else:
        pairs = build_ngram_pairs(raw_text, k=k)
        train_pairs, test_pairs = train_test_split(pairs, frac_train=0.8, seed=seed)
        model = NGramModel.from_pairs(train_pairs, k=k, alpha=1.0)
        h_train = cross_entropy(model.prob, train_pairs)
        h_test = cross_entropy(model.prob, test_pairs)
        h_unif = uniform_cross_entropy_from_pairs_k(test_pairs)
    pp_train = math.exp(h_train)
    pp_test = math.exp(h_test)
    pp_unif = math.exp(h_unif)
    delta_h = h_unif - h_test
    gap = h_test - h_train
    return {
        "k": k,
        "seed": seed,
        "n_pairs": len(pairs),
        "H_train": h_train,
        "H_test": h_test,
        "PP_train": pp_train,
        "PP_test": pp_test,
        "H_unif": h_unif,
        "PP_unif": pp_unif,
        "Delta_H": delta_h,
        "Gap": gap,
    }

def main() -> None:
    raw_text = (
        "Large language models are conditional token generators. "
        "They map a context to a distribution over the next token. "
        "This demo uses a simple n-gram character model to make that concrete. "
    )
    ks = [1, 2, 3, 4]
    seeds = list(range(100, 120))
    fieldnames = [
        "k",
        "seed",
        "n_pairs",
        "H_train",
        "H_test",
        "PP_train",
        "PP_test",
        "H_unif",
        "PP_unif",
        "Delta_H",
        "Gap",
    ]
    with open("out_k_sweep_bootstrap.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in ks:
            for seed in seeds:
                row = metrics_for_k_seed(raw_text, k, seed)
                writer.writerow(row)
    print("Wrote out_k_sweep_bootstrap.csv")

if __name__ == "__main__":
    main()
