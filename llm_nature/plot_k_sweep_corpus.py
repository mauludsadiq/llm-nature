import csv
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

def load_results(repeats: List[int]) -> Dict[int, List[Tuple[int, float]]]:
    data: Dict[int, List[Tuple[int, float]]] = {}
    for k in [1, 2, 3, 4]:
        data[k] = []
    for r in repeats:
        path = f"out_k_sweep_metrics_repeat{r}.csv"
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = int(row["k"])
                n_pairs = int(row["n_pairs"])
                h_test = float(row["H_test"])
                data[k].append((n_pairs, h_test))
    for k in data:
        data[k].sort(key=lambda t: t[0])
    return data

def main() -> None:
    repeats = [1, 2, 5, 10, 20, 50, 100]
    data = load_results(repeats)
    plt.figure()
    for k in sorted(data.keys()):
        xs = [t[0] for t in data[k]]
        ys = [t[1] for t in data[k]]
        plt.plot(xs, ys, marker="o", label=f"k={k}")
    plt.xlabel("Corpus size (n_pairs for given k)")
    plt.ylabel("H_test (nats per character)")
    plt.title("Test cross-entropy vs corpus size for n-gram models")
    plt.legend()
    plt.tight_layout()
    plt.savefig("out_Htest_vs_corpus.png")
    print("Wrote out_Htest_vs_corpus.png")

if __name__ == "__main__":
    main()
