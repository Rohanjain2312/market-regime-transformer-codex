"""Utility helpers for logging, reproducibility, and data handling."""

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def setup_logging(log_level: int = logging.INFO, log_file: Optional[Path] = None) -> logging.Logger:
    """Configure and return a root logger with console and optional file handlers."""
    logger = logging.getLogger("market_regime_transformer")
    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rolling_windows(data_length: int, splits: int, min_train_size: int = 252):
    """Yield train/val indices for rolling validation on time series."""
    min_train_size = min(min_train_size, max(1, data_length // 2))
    fold_size = max(1, (data_length - min_train_size) // max(1, splits))
    for i in range(splits):
        train_end = min_train_size + i * fold_size
        val_end = train_end + fold_size
        val_end = min(val_end, data_length)
        yield slice(0, train_end), slice(train_end, val_end)
