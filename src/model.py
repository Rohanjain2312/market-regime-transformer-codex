"""Transformer encoder with dual heads for regression and classification.

Features:
- Token embedding with sqrt(d_model) scaling
- Sinusoidal positional encoding (buffered)
- Pre-norm Transformer blocks with attention masking support
- Attention and embedding dropout
- Dual heads for regression (MSE) and classification (CrossEntropy)
- Returns attention weights for visualization
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PreNormTransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block with attention masking."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        attn_dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=attn_dropout,
            batch_first=True,
            need_weights=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # Attention
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        x = x + self.dropout(attn_out)

        # Feed-forward
        x_norm = self.norm2(x)
        ff_out = self.ffn(x_norm)
        x = x + self.dropout(ff_out)
        return x, attn_weights


class RegimeTransformer(nn.Module):
    """Transformer encoder with dual heads and attention outputs."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        attn_dropout: float = 0.0,
        num_regimes: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        self.layers = nn.ModuleList(
            [
                PreNormTransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.reg_head = nn.Linear(d_model, 1)
        self.cls_head = nn.Linear(d_model, num_regimes)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
            attn_mask: Optional attention mask (seq_len x seq_len) or broadcastable
            key_padding_mask: Optional boolean mask for padding (batch x seq_len)

        Returns:
            {
                "regression": regression_output (batch,),
                "classification": classification_logits (batch, num_regimes),
                "attention": last_layer_attention_weights
            }
        """
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.embedding_dropout(x)

        attn_weights = None
        for layer in self.layers:
            x, attn_weights = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # Pooling: mean over sequence (ignoring padding if provided)
        if key_padding_mask is not None:
            # key_padding_mask: True for padding positions
            valid_counts = (~key_padding_mask).sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
            mask = (~key_padding_mask).unsqueeze(-1)
            pooled = (x * mask).sum(dim=1) / valid_counts
        else:
            pooled = x.mean(dim=1)

        regression = self.reg_head(pooled).squeeze(-1)
        classification = self.cls_head(pooled)
        return {"regression": regression, "classification": classification, "attention": attn_weights}


def combined_loss(
    outputs: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    lambda_reg: float,
    lambda_cls: float,
) -> Tensor:
    """Compute combined regression + classification loss."""
    mse = nn.functional.mse_loss(outputs["regression"], targets["regression"])
    ce = nn.functional.cross_entropy(outputs["classification"], targets["classification"])
    return lambda_reg * mse + lambda_cls * ce
