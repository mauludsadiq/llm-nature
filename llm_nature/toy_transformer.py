from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ToyTransformerConfig:
    vocab_size: int          # |Σ|
    d_model: int = 256       # embedding / hidden dim (d)
    n_layers: int = 4        # # transformer blocks
    n_heads: int = 4         # # attention heads
    d_ff: int = 1024         # feedforward hidden dim
    block_size: int = 256    # max sequence length (L)
    dropout: float = 0.1     # dropout prob
    # Extension hooks (not used yet, but here for future wiring):
    # use_rope: bool = False
    # use_kv_cache: bool = False
    # use_checkpoint: bool = False


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Implements:
        Q = X W_Q
        K = X W_K
        V = X W_V
        A = softmax( (Q K^T) / sqrt(d_head) + mask )
        Y = A V
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, block_size: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.block_size = block_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        returns: [B, T, d_model]
        """
        B, T, C = x.size()
        assert T <= self.block_size, "sequence length exceeds block_size"

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        mask = self.causal_mask[:T, :T]
        mask_value = torch.finfo(att.dtype).min
        att = att.masked_fill(mask == 0, mask_value)

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.out_proj(y))
        return y


class TransformerBlock(nn.Module):
    """
    One pre-norm transformer block:
        x -> x + Attn(LN(x))
        x -> x + MLP(LN(x))
    """

    def __init__(self, config: ToyTransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.ln_2 = nn.LayerNorm(config.d_model)

        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            block_size=config.block_size,
        )

        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ToyTransformer(nn.Module):
    """
    Minimal GPT-style transformer language model.

    forward(input_ids, targets=None):

        - input_ids: [B, T] of token indices in [0, vocab_size)
        - targets:   [B, T] or None

        returns: (logits, loss)
            logits: [B, T, vocab_size]
            loss: scalar cross-entropy if targets is not None, else None
    """

    def __init__(self, config: ToyTransformerConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def block_size(self) -> int:
        return self.config.block_size

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = input_ids.device
        B, T = input_ids.size()
        assert T <= self.config.block_size, "sequence too long"

        tok = self.token_emb(input_ids)
        pos_idx = torch.arange(T, device=device).unsqueeze(0)
        pos = self.pos_emb(pos_idx)
        x = self.drop(tok + pos)

        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(B * T, -1)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")

        return logits, loss


if __name__ == "__main__":
    config = ToyTransformerConfig(
        vocab_size=100,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        block_size=32,
        dropout=0.1,
    )
    model = ToyTransformer(config)

    B, T = 2, 16
    x = torch.randint(0, config.vocab_size, (B, T), dtype=torch.long)
    logits, loss = model(x, targets=x)

    print("logits shape:", logits.shape)
    print("loss:", float(loss))

    loss.backward()

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += float(p.grad.norm().item())
    print(f"Total gradient norm: {total_norm:.6f}")

    assert logits.shape == (B, T, config.vocab_size)
    assert isinstance(loss, torch.Tensor)
    print("All tests passed!")
