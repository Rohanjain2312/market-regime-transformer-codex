"""Training script for the Market Regime-Switching Transformer.

This module loads data, builds rolling training/validation splits,
trains the transformer model with dual heads, and tracks performance
using torchmetrics. The best-performing checkpoint is saved to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics import Accuracy, F1Score, MeanAbsoluteError, MeanSquaredError

from .config import Config, get_config, seed_everything
from .data_loader import load_data
from .features import build_feature_windows
from .model import RegimeTransformer, combined_loss

logger = logging.getLogger("market_regime_transformer.train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class MarketWindowDataset(Dataset):
    """Dataset wrapping windowed features with regression and classification targets."""

    def __init__(self, X, y_reg, y_cls):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], {"regression": self.y_reg[idx], "classification": self.y_cls[idx]}


def create_rolling_splits(length: int, n_splits: int = 3) -> Iterable[Tuple[slice, slice]]:
    """Yield rolling train/validation index slices."""
    if n_splits < 1:
        n_splits = 1
    split_size = length // (n_splits + 1)
    for i in range(n_splits):
        train_end = split_size * (i + 1)
        val_end = split_size * (i + 2)
        yield slice(0, train_end), slice(train_end, min(val_end, length))


def build_dataloaders(dataset: Dataset, cfg: Config) -> List[Tuple[DataLoader, DataLoader]]:
    """Build rolling train/validation dataloaders."""
    pairs = []
    for train_idx, val_idx in create_rolling_splits(len(dataset), n_splits=cfg.rolling_splits if hasattr(cfg, "rolling_splits") else 3):
        train_subset = Subset(dataset, range(train_idx.start, train_idx.stop))
        val_subset = Subset(dataset, range(val_idx.start, val_idx.stop))
        train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False)
        pairs.append((train_loader, val_loader))
    return pairs


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    mae: MeanAbsoluteError,
    rmse: MeanSquaredError,
    acc: Accuracy,
    f1: F1Score,
) -> Tuple[float, dict]:
    """Evaluate model over a validation loader."""
    model.eval()
    total_loss = 0.0
    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(device)
        targets = {k: v.to(device) for k, v in batch_y.items()}
        with torch.no_grad():
            outputs = model(batch_x)
            loss = combined_loss(outputs, targets, cfg.lambda_regression, cfg.lambda_classification)
        total_loss += loss.item() * batch_x.size(0)

        mae.update(outputs["regression"], targets["regression"])
        rmse.update(outputs["regression"], targets["regression"])
        acc.update(outputs["classification"], targets["classification"])
        f1.update(outputs["classification"], targets["classification"])

    val_loss = total_loss / len(data_loader.dataset)
    metrics = {
        "mae": float(mae.compute()),
        "rmse": float(rmse.compute()),
        "accuracy": float(acc.compute()),
        "f1": float(f1.compute()),
    }

    mae.reset()
    rmse.reset()
    acc.reset()
    f1.reset()
    return val_loss, metrics


def train_split(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    checkpoint_path: Path,
) -> None:
    """Train for one rolling split and save the best-performing checkpoint."""
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.1)
    mae = MeanAbsoluteError().to(device)
    rmse = MeanSquaredError(squared=False).to(device)
    acc = Accuracy(task="multiclass", num_classes=2).to(device)
    f1 = F1Score(task="multiclass", num_classes=2).to(device)

    best_score = float("inf")
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            targets = {k: v.to(device) for k, v in batch_y.items()}
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = combined_loss(outputs, targets, cfg.lambda_regression, cfg.lambda_classification)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        val_loss, metrics = evaluate(model, val_loader, cfg, device, mae, rmse, acc, f1)
        logger.info(
            "Epoch %d | Train Loss: %.4f | Val Loss: %.4f | MAE: %.4f | RMSE: %.4f | Acc: %.4f | F1: %.4f",
            epoch + 1,
            train_loss,
            val_loss,
            metrics["mae"],
            metrics["rmse"],
            metrics["accuracy"],
            metrics["f1"],
        )

        # Track best combined validation loss
        if val_loss < best_score:
            best_score = val_loss
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("Saved new best model to %s", checkpoint_path)


def main() -> None:
    cfg = get_config()
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    logger.info("Using device: %s", device)

    # Load and prepare data
    raw_df = load_data(cfg)
    X, y_reg, y_cls = build_feature_windows(raw_df, cfg, target_col="SPY")
    dataset = MarketWindowDataset(X, y_reg, y_cls)
    dataloaders = build_dataloaders(dataset, cfg)

    for split_idx, (train_loader, val_loader) in enumerate(dataloaders):
        logger.info("Starting split %d/%d", split_idx + 1, len(dataloaders))
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
        checkpoint = cfg.processed_data_path / f"best_model_split{split_idx}.pt"
        train_split(model, train_loader, val_loader, cfg, device, checkpoint)


if __name__ == "__main__":
    main()
