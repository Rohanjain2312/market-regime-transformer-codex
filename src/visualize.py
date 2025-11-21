"""Visualization utilities for attention maps and regime transitions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def _to_numpy(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def plot_attention(attention: Union[np.ndarray, torch.Tensor], save_path: Path) -> None:
    """Plot transformer attention weights and save to disk."""
    attn = _to_numpy(attention)
    if attn.ndim == 4:
        # Common shape: (num_layers, num_heads, seq_len, seq_len) or (batch, head, seq, seq)
        attn = attn.mean(axis=0)
    attn = attn.squeeze()

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(attn, cmap="viridis", ax=ax)
    ax.set_title("Transformer Attention Weights")
    ax.set_xlabel("Key positions")
    ax.set_ylabel("Query positions")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_regime_transitions(regimes: Union[np.ndarray, torch.Tensor], save_path: Path, time_index: Optional[Union[np.ndarray, list]] = None) -> None:
    """Plot regime transitions over time and save to disk."""
    reg = _to_numpy(regimes).flatten()
    t = np.arange(len(reg)) if time_index is None else np.asarray(time_index)

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.step(t, reg, where="post")
    ax.set_title("Regime Transitions")
    ax.set_xlabel("Time")
    ax.set_ylabel("Regime")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
