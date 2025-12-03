import csv
from pathlib import Path


def load_results(path: Path):
    by_N = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            N = int(row["N"])
            model = row["model"].strip()
            by_N.setdefault(N, {})[model] = row
    return dict(sorted(by_N.items()))


def main():
    path = Path("out_compare_ngram_transformer.csv")
    results = load_results(path)

    print("| N (repeats) | H_test (ngram) | PP (ngram) | H_test (transformer) | PP (transformer) |")
    print("|------------:|---------------:|-----------:|----------------------:|-----------------:|")

    for N, models in results.items():
        ng = models.get("ngram")
        tx = models.get("transformer")
        if ng is None or tx is None:
            continue
        h_ng = float(ng["H"])
        pp_ng = float(ng["PP"])
        h_tx = float(tx["H"])
        pp_tx = float(tx["PP"])
        print(
            f"| {N} | {h_ng:.4f} | {pp_ng:.4f} | {h_tx:.4f} | {pp_tx:.4f} |"
        )


if __name__ == "__main__":
    main()
