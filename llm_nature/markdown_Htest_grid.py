import csv
from typing import Dict, Tuple, List

def load_h_test(path: str) -> Tuple[List[int], List[int], Dict[Tuple[int, int], float]]:
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

def main() -> None:
    repeats, ks, h_test = load_h_test("out_k_sweep_corpus_all.csv")
    header_cells = ["N (repeat)"] + [f"H_test(k={k})" for k in ks]
    header = "| " + " | ".join(header_cells) + " |"
    sep = "| " + " | ".join(["---"] * len(header_cells)) + " |"
    print(header)
    print(sep)
    for r in repeats:
        row_vals = [str(r)]
        for k in ks:
            val = h_test.get((r, k))
            if val is None:
                row_vals.append("")
            else:
                row_vals.append(f"{val:.4f}")
        print("| " + " | ".join(row_vals) + " |")

if __name__ == "__main__":
    main()
