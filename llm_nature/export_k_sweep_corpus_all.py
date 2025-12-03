"""
Run the full k-sweep across corpus sizes N and aggregate into one CSV.

This script:
  1. Calls export_k_sweep_csv with --repeat for N in {1,2,5,10,20,50,100}
  2. Merges the resulting out_k_sweep_metrics_repeat*.csv
     into out_k_sweep_corpus_all.csv with an added 'repeat' column.
"""

import csv
import sys
from pathlib import Path
from typing import List
import subprocess

REPEATS: List[int] = [1, 2, 5, 10, 20, 50, 100]


def run_sweeps() -> None:
    """Run export_k_sweep_csv once for each repeat N."""
    for N in REPEATS:
        cmd = [
            sys.executable,
            "-m",
            "llm_nature.export_k_sweep_csv",
            f"--repeat={N}",
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def aggregate() -> None:
    """
    Read out_k_sweep_metrics_repeat{N}.csv for all N and
    write out_k_sweep_corpus_all.csv with a 'repeat' column.
    """
    out_path = Path("out_k_sweep_corpus_all.csv")
    rows = []
    fieldnames = None

    for N in REPEATS:
        path = Path(f"out_k_sweep_metrics_repeat{N}.csv")
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping")
            continue
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = dict(row)
                row["repeat"] = str(N)
                rows.append(row)
                if fieldnames is None:
                    fieldnames = list(row.keys())

    if not rows:
        print("No rows found; nothing to write")
        return

    # Make sure 'repeat' is in fieldnames and near the front
    if "repeat" not in fieldnames:
        fieldnames = ["repeat"] + fieldnames
    else:
        # Move repeat to front
        fieldnames = ["repeat"] + [fn for fn in fieldnames if fn != "repeat"]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


def main() -> None:
    run_sweeps()
    aggregate()


if __name__ == "__main__":
    main()
