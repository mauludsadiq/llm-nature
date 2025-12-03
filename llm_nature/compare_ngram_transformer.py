"""
Compare 4-gram character model vs ToyTransformer on the same paragraph × N.

Outputs CSV:

N,model,H,PP
1,ngram,...
1,transformer,...
...
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

from .ngram import build_ngram_pairs, NGramModel
from .metrics import cross_entropy
from .transformer_train import (
    train_toy_transformer_on_text,
    estimate_cross_entropy_on_pairs,
)


def load_paragraph(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    # normalize newlines a bit, keep exactly what your other scripts use
    return text.strip("\n")


def steps_for_repeat(N: int) -> int:
    """
    Tiny heuristic for #steps as corpus size grows.
    We keep this modest so it runs on CPU in reasonable time.
    """
    if N <= 2:
        return 300
    if N <= 5:
        return 400
    if N <= 10:
        return 500
    if N <= 20:
        return 600
    if N <= 50:
        return 700
    return 800  # N=100


def main() -> None:
    data_path = Path("data/paragraph.txt")
    base_text = load_paragraph(data_path)

    Ns = [1, 2, 5, 10, 20, 50, 100]
    k = 4  # context length for the character n-gram

    print("N,model,H,PP")

    for N in Ns:
        text = base_text * N

        # ---------- N-GRAM BASELINE ----------
        pairs: List[Tuple[str, str]] = build_ngram_pairs(text, k=k)
        ngram = NGramModel.from_pairs(pairs, k=k, alpha=1.0)
        H_ngram = cross_entropy(ngram.prob, pairs)
        PP_ngram = math.exp(H_ngram)
        print(f"{N},ngram,{H_ngram:.4f},{PP_ngram:.4f}")

        # ---------- TOY TRANSFORMER ----------
        block_size = 32  # bigger than k, but still small
        steps = steps_for_repeat(N)

        model, vocab, stoi, itos = train_toy_transformer_on_text(
            text=text,
            block_size=block_size,
            steps=steps,
            batch_size=32,
            lr=3e-4,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            dropout=0.1,
            device="cpu",
            seed=441 + N,
        )

        H_tf = estimate_cross_entropy_on_pairs(model, stoi, pairs, device="cpu")
        PP_tf = math.exp(H_tf) if math.isfinite(H_tf) else float("nan")
        print(f"{N},transformer,{H_tf:.4f},{PP_tf:.4f}")


if __name__ == "__main__":
    main()
