"""Evaluation script for the Market Regime-Switching Transformer.

Loads a trained checkpoint, computes regression and classification metrics,
plots the confusion matrix, and prints a concise summary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from torch.utils.data import DataLoader, Dataset

from .config import Config, get_config, seed_everything
from .data_loader import load_data
from .features import build_feature_windows
from .model import RegimeTransformer

logger = logging.getLogger("market_regime_transformer.evaluate")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class MarketEvalDataset(Dataset):
    """Dataset wrapper for evaluation windows."""

    def __init__(self, X: np.ndarray, y_reg: np.ndarray, y_cls: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], {"regression": self.y_reg[idx], "classification": self.y_cls[idx]}


def directional_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute sign accuracy for regression outputs."""
    pred_sign = np.sign(pred)
    true_sign = np.sign(true)
    return float((pred_sign == true_sign).mean())


def plot_confusion(cm: np.ndarray, save_path: Path) -> None:
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Regime Classification Confusion Matrix")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def evaluate_model(cfg: Config, checkpoint_path: Path) -> Tuple[Dict[str, float], np.ndarray]:
    """Run evaluation on the full dataset."""
    device = torch.device(cfg.device)

    raw_df = load_data(cfg)
    X, y_reg, y_cls = build_feature_windows(raw_df, cfg, target_col="SPY")
    dataset = MarketEvalDataset(X, y_reg, y_cls)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    feature_dim = X.shape[-1]
    model = RegimeTransformer(
        input_dim=feature_dim,
        d_model=cfg.embedding_dim,
        nhead=cfg.num_heads,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.embedding_dim * 2,
        dropout=cfg.dropout,
        num_regimes=2,
    ).to(device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    reg_preds, reg_true, cls_preds, cls_true = [], [], [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            targets = {k: v.to(device) for k, v in batch_y.items()}
            outputs = model(batch_x)
            reg_preds.append(outputs["regression"].cpu().numpy())
            reg_true.append(targets["regression"].cpu().numpy())
            cls_preds.append(outputs["classification"].argmax(dim=1).cpu().numpy())
            cls_true.append(targets["classification"].cpu().numpy())

    reg_preds_arr = np.concatenate(reg_preds).flatten()
    reg_true_arr = np.concatenate(reg_true).flatten()
    cls_preds_arr = np.concatenate(cls_preds).flatten()
    cls_true_arr = np.concatenate(cls_true).flatten()

    mae = mean_absolute_error(reg_true_arr, reg_preds_arr)
    rmse = mean_squared_error(reg_true_arr, reg_preds_arr, squared=False)
    dir_acc = directional_accuracy(reg_preds_arr, reg_true_arr)
    acc = accuracy_score(cls_true_arr, cls_preds_arr)
    f1 = f1_score(cls_true_arr, cls_preds_arr, average="macro")
    cm = confusion_matrix(cls_true_arr, cls_preds_arr)

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "directional_accuracy": float(dir_acc),
        "accuracy": float(acc),
        "f1_macro": float(f1),
    }
    return metrics, cm


def main() -> None:
    cfg = get_config()
    seed_everything(cfg.seed)
    checkpoint = cfg.processed_data_path / "best_model_split0.pt"

    metrics, cm = evaluate_model(cfg, checkpoint)
    plot_path = cfg.processed_data_path / "confusion_matrix.png"
    plot_confusion(cm, plot_path)

    logger.info("Evaluation Summary")
    for k, v in metrics.items():
        logger.info("%s: %.4f", k, v)
    logger.info("Confusion matrix saved to %s", plot_path)


if __name__ == "__main__":
    main()
