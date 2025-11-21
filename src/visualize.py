"""Visualization suite for the Market Regime-Switching Transformer."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def _to_numpy(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def plot_attention_heatmap(
    attention_weights: Union[np.ndarray, torch.Tensor],
    feature_names: Optional[Sequence[str]],
    save_path: Path,
) -> None:
    """Plot averaged multi-head attention heatmap."""
    attn = _to_numpy(attention_weights)
    # Expected shapes: (num_heads, seq, seq) or (batch, heads, seq, seq)
    if attn.ndim == 4:
        attn = attn.mean(axis=(0, 1)) if attn.shape[0] > 1 else attn.mean(axis=0)
    elif attn.ndim == 3:
        attn = attn.mean(axis=0)
    attn = attn.squeeze()

    plt.figure(figsize=(8, 6), dpi=200)
    sns.heatmap(attn, cmap="viridis", square=True)
    plt.title("Attention Heatmap (Averaged Heads)")
    plt.xlabel("Key timestep")
    plt.ylabel("Query timestep")
    if feature_names and len(feature_names) == attn.shape[0]:
        plt.xticks(np.arange(attn.shape[0]) + 0.5, feature_names, rotation=90)
        plt.yticks(np.arange(attn.shape[1]) + 0.5, feature_names, rotation=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_regime_transitions(
    regime_labels: Union[np.ndarray, torch.Tensor, Sequence[int]],
    price_series: pd.Series,
    save_path: Path,
) -> None:
    """Overlay regimes on price series with shaded bull/bear regions."""
    regimes = _to_numpy(regime_labels).flatten()
    dates = price_series.index
    if len(regimes) != len(price_series):
        min_len = min(len(regimes), len(price_series))
        regimes = regimes[:min_len]
        price_series = price_series.iloc[:min_len]
        dates = price_series.index

    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot(dates, price_series.values, label="Price", color="black")
    is_bull = regimes == 1
    plt.fill_between(dates, price_series.values.min(), price_series.values.max(), where=is_bull, color="green", alpha=0.1, label="Bull")
    plt.fill_between(dates, price_series.values.min(), price_series.values.max(), where=~is_bull, color="red", alpha=0.1, label="Bear")

    # Annotate transitions
    transitions = np.where(np.diff(regimes) != 0)[0]
    for t in transitions:
        plt.axvline(dates[t + 1], color="blue", linestyle="--", alpha=0.5)

    plt.title("Regime Transitions")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_distributions(df: pd.DataFrame, save_path_prefix: Path) -> None:
    """Plot histograms for each feature and a correlation heatmap."""
    save_path_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Histograms
    num_cols = len(df.columns)
    cols = 3
    rows = int(np.ceil(num_cols / cols))
    plt.figure(figsize=(cols * 4, rows * 3), dpi=200)
    for i, col in enumerate(df.columns):
        plt.subplot(rows, cols, i + 1)
        sns.histplot(df[col].dropna(), bins=50, kde=True)
        plt.title(col)
    plt.tight_layout()
    hist_path = save_path_prefix.with_suffix("_hists.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()

    # Correlation heatmap
    corr = df.corr()
    plt.figure(figsize=(8, 6), dpi=200)
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation")
    plt.tight_layout()
    corr_path = save_path_prefix.with_suffix("_corr.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()


def plot_training_curves(log_csv_path: Path) -> None:
    """Plot training/validation losses and F1 from CSV logs."""
    df = pd.read_csv(log_csv_path)
    save_dir = log_csv_path.parent

    plt.figure(figsize=(8, 4), dpi=200)
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.plot(df["epoch"], df["f1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    save_path = save_dir / "training_curves.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
