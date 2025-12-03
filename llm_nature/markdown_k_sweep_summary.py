import csv
from typing import Dict, Tuple, List, Optional

def load_all(path: str) -> Tuple[List[int], List[int], Dict[Tuple[int, int], float]]:
    repeats_set = set()
    ks_set = set()
    h_test: Dict[Tuple[int, int], float] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row["repeat"])
            k = int(row["k"])
            h = float(row["H_test"])
            repeats_set.add(r)
            ks_set.add(k)
            h_test[(r, k)] = h
    repeats = sorted(repeats_set)
    ks = sorted(ks_set)
    return repeats, ks, h_test

def estimate_H_inf_for_k(repeats: List[int], h_test: Dict[Tuple[int, int], float], k: int) -> Optional[float]:
    target = 100
    if target in repeats and (target, k) in h_test:
        return h_test[(target, k)]
    r_max = max(repeats)
    key = (r_max, k)
    if key in h_test:
        return h_test[key]
    return None

def find_crossover_repeat(repeats: List[int], h_test: Dict[Tuple[int, int], float], k: int) -> Optional[int]:
    if k == 1:
        return None
    for r in sorted(repeats):
        key_k = (r, k)
        key_1 = (r, 1)
        if key_k in h_test and key_1 in h_test:
            if h_test[key_k] < h_test[key_1]:
                return r
    return None

def main() -> None:
    repeats, ks, h_test = load_all("out_k_sweep_corpus_all.csv")
    header = "| k | H_inf (approx, nats/char) | crossover repeat N |"
    sep = "|---|---------------------------:|-------------------:|"
    print(header)
    print(sep)
    for k in ks:
        h_inf = estimate_H_inf_for_k(repeats, h_test, k)
        cross = find_crossover_repeat(repeats, h_test, k)
        if h_inf is None:
            h_str = ""
        else:
            h_str = f"{h_inf:.6f}"
        cross_str = "" if cross is None else str(cross)
        print(f"| {k} | {h_str} | {cross_str} |")

if __name__ == "__main__":
    main()
