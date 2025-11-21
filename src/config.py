"""Configuration module for the Market Regime-Switching Transformer."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set seeds for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    """Configuration dataclass for training and data handling."""

    # Data paths
    raw_data_path: Path = Path("market_regime_transformer/data/raw")
    processed_data_path: Path = Path("market_regime_transformer/data/processed")

    # Hyperparameters
    batch_size: int = 64
    learning_rate: float = 1e-3
    lookback_window: int = 30
    dropout: float = 0.1
    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2

    # Loss weights
    lambda_regression: float = 0.7  # λ1
    lambda_classification: float = 0.3  # λ2

    # Training settings
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    seed: int = 42


def get_config(seed: Optional[int] = None) -> Config:
    """Create and return a Config instance with directories initialized."""
    cfg = Config()
    if seed is not None:
        cfg.seed = seed
    seed_everything(cfg.seed)
    cfg.raw_data_path.mkdir(parents=True, exist_ok=True)
    cfg.processed_data_path.mkdir(parents=True, exist_ok=True)
    return cfg
