"""Transformer encoder model with dual heads for regression and classification."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encodings to input tensor."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class RegimeTransformer(nn.Module):
    """Transformer encoder with dual heads for next-day return and regime prediction."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_regimes: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reg_head = nn.Linear(d_model, 1)
        self.cls_head = nn.Linear(d_model, num_regimes)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Input tensor with shape (batch, seq_len, n_features).

        Returns:
            Dictionary with regression and classification outputs.
        """
        x_proj = self.input_proj(x)
        x_pos = self.positional_encoding(x_proj)
        encoded = self.encoder(x_pos)
        pooled = encoded.mean(dim=1)  # Global average pooling over sequence
        regression = self.reg_head(pooled).squeeze(-1)
        classification = self.cls_head(pooled)
        return {"regression": regression, "classification": classification}


def combined_loss(
    outputs: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    lambda_reg: float,
    lambda_cls: float,
) -> Tensor:
    """Compute combined regression (MSE) and classification (CrossEntropy) loss."""
    mse = nn.functional.mse_loss(outputs["regression"], targets["regression"])
    ce = nn.functional.cross_entropy(outputs["classification"], targets["classification"])
    return lambda_reg * mse + lambda_cls * ce
