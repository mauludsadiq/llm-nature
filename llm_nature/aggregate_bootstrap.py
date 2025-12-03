import csv
import math
import statistics
from typing import Dict, List

def aggregate_bootstrap(in_path: str, out_path: str) -> None:
    by_k: Dict[int, List[dict]] = {}
    with open(in_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = int(row["k"])
            by_k.setdefault(k, []).append(row)
    fieldnames = [
        "k",
        "n_seeds",
        "H_train_mean",
        "H_train_std",
        "H_test_mean",
        "H_test_std",
        "Delta_H_mean",
        "Delta_H_std",
        "Gap_mean",
        "Gap_std",
        "PP_test_mean",
        "PP_test_std",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in sorted(by_k.keys()):
            rows = by_k[k]
            n = len(rows)
            def vals(name: str) -> List[float]:
                return [float(r[name]) for r in rows]
            H_train_vals = vals("H_train")
            H_test_vals = vals("H_test")
            Delta_H_vals = vals("Delta_H")
            Gap_vals = vals("Gap")
            PP_test_vals = vals("PP_test")
            def m(vs: List[float]) -> float:
                return statistics.mean(vs) if vs else math.nan
            def s(vs: List[float]) -> float:
                return statistics.stdev(vs) if len(vs) > 1 else 0.0
            row_out = {
                "k": k,
                "n_seeds": n,
                "H_train_mean": m(H_train_vals),
                "H_train_std": s(H_train_vals),
                "H_test_mean": m(H_test_vals),
                "H_test_std": s(H_test_vals),
                "Delta_H_mean": m(Delta_H_vals),
                "Delta_H_std": s(Delta_H_vals),
                "Gap_mean": m(Gap_vals),
                "Gap_std": s(Gap_vals),
                "PP_test_mean": m(PP_test_vals),
                "PP_test_std": s(PP_test_vals),
            }
            writer.writerow(row_out)

def main() -> None:
    aggregate_bootstrap("out_k_sweep_bootstrap.csv", "out_k_sweep_bootstrap_summary.csv")
    print("Wrote out_k_sweep_bootstrap_summary.csv")

if __name__ == "__main__":
    main()
