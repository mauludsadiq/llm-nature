from __future__ import annotations

from typing import Dict, List, Tuple

import math

import torch
import torch.nn.functional as F

from .toy_transformer import ToyTransformer, ToyTransformerConfig


def build_char_vocab(text: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Build character-level vocabulary from text."""
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    return vocab, stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> torch.Tensor:
    """Encode text into tensor of token ids."""
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def estimate_cross_entropy_on_pairs(
    model: ToyTransformer,
    stoi: Dict[str, int],
    pairs: List[Tuple[str, str]],
    device: str = "cpu",
) -> float:
    """
    Estimate cross-entropy (nats/char) on a list of (context, next_char) pairs
    using the transformer as p(y | context).

    pairs: list of (x_str, y_char) where len(x_str) = k, y_char is a single character.
    """
    model.eval()
    total_nll = 0.0
    n = 0

    with torch.no_grad():
        for x_str, y_char in pairs:
            # skip if character not in vocab (shouldn't happen if vocab built from same corpus)
            if y_char not in stoi:
                continue

            x_ids = torch.tensor(
                [[stoi[ch] for ch in x_str]],
                dtype=torch.long,
                device=device,
            )
            logits, _ = model(x_ids, targets=None)  # [1, T, V]
            last_logits = logits[0, -1]  # [V]
            log_probs = F.log_softmax(last_logits, dim=-1)
            y_id = stoi[y_char]

            total_nll += -float(log_probs[y_id])
            n += 1

    if n == 0:
        return math.nan
    return total_nll / n


def train_toy_transformer_on_text(
    text: str,
    block_size: int,
    steps: int = 400,
    batch_size: int = 32,
    lr: float = 3e-4,
    d_model: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    d_ff: int = 256,
    dropout: float = 0.1,
    device: str = "cpu",
    seed: int = 441,
) -> Tuple[ToyTransformer, List[str], Dict[str, int], Dict[int, str]]:
    """
    Train a minimal GPT-style transformer on raw character text.

    Returns:
        model, vocab, stoi, itos
    """
    torch.manual_seed(seed)

    vocab, stoi, itos = build_char_vocab(text)
    data = encode(text, stoi).to(device)

    if len(data) <= block_size + 1:
        raise ValueError(
            f"Text too short for block_size={block_size}: got len={len(data)}"
        )

    config = ToyTransformerConfig(
        vocab_size=len(vocab),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        block_size=block_size,
        dropout=dropout,
    )
    model = ToyTransformer(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def get_batch() -> Tuple[torch.Tensor, torch.Tensor]:
        # random contiguous chunks of length block_size
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
        return x, y

    model.train()
    for step in range(steps):
        x, y = get_batch()
        x = x.to(device)
        y = y.to(device)

        logits, loss = model(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model, vocab, stoi, itos
