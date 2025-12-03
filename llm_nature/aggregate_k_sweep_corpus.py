import csv
from typing import List

def aggregate(repeats: List[int], out_path: str) -> None:
    fieldnames = [
        "repeat",
        "k",
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
    rows = []
    for r in repeats:
        path = f"out_k_sweep_metrics_repeat{r}.csv"
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "repeat": r,
                    "k": int(row["k"]),
                    "n_pairs": int(row["n_pairs"]),
                    "H_train": float(row["H_train"]),
                    "H_test": float(row["H_test"]),
                    "PP_train": float(row["PP_train"]),
                    "PP_test": float(row["PP_test"]),
                    "H_unif": float(row["H_unif"]),
                    "PP_unif": float(row["PP_unif"]),
                    "Delta_H": float(row["Delta_H"]),
                    "Gap": float(row["Gap"]),
                })
    rows.sort(key=lambda r: (r["k"], r["n_pairs"]))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def main() -> None:
    repeats = [1, 2, 5, 10, 20, 50, 100]
    aggregate(repeats, "out_k_sweep_corpus_all.csv")
    print("Wrote out_k_sweep_corpus_all.csv")

if __name__ == "__main__":
    main()
