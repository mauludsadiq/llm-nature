from __future__ import annotations

import math


def main() -> None:
    """
    Print a markdown table for the context manifold:

    k, |Σ|^k (theoretical), unique contexts (N=100), utilization.
    Values are fixed for the current corpus and alphabet.
    """

    # Character alphabet size for data/paragraph.txt
    sigma = 26

    # Empirically observed unique k-gram counts at N=100
    rows = [
        {"k": 1, "unique": 26},
        {"k": 2, "unique": 104},
        {"k": 3, "unique": 153},
        {"k": 4, "unique": 169},
    ]

    print("| k | Σ^k (theoretical) | unique contexts (N=100) | utilization |")
    print("|---:|------------------:|------------------------:|------------:|")

    for row in rows:
        k = row["k"]
        unique = row["unique"]
        theoretical = sigma ** k
        util = 100.0 * unique / theoretical
        print(f"| {k} | {theoretical} | {unique} | {util:.2f}% |")


if __name__ == "__main__":
    main()
