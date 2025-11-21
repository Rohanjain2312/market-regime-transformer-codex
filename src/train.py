"""Production-quality training loop for the Market Regime-Switching Transformer.

Features:
- AdamW optimizer with weight decay
- Linear warmup + cosine annealing scheduler
- Gradient clipping
- Early stopping (patience on val loss and classification F1)
- Expanding window rolling validation
- Saves best model, CSV logs, and learning curves
- Verbose logging and device management
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics import Accuracy, F1Score, MeanAbsoluteError, MeanSquaredError

from .config import Config, get_config, seed_everything
from .data_loader import load_data
from .features import build_feature_windows
from .model import RegimeTransformer, combined_loss
from .utils import get_device

logger = logging.getLogger("market_regime_transformer.train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@dataclass
class TrainResult:
    best_val_loss: float
    best_f1: float
    train_history: List[Dict]


class MarketWindowDataset(Dataset):
    """Windowed dataset for regression and classification targets."""

    def __init__(self, X: np.ndarray, y_reg: np.ndarray, y_cls: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], {"regression": self.y_reg[idx], "classification": self.y_cls[idx]}


def linear_warmup_cosine(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float,
    base_lr: float,
):
    """Create a per-step LR scheduler with linear warmup then cosine decay."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return max(min_lr / base_lr, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_expanding_splits(length: int, val_window: int, min_train: int) -> List[Tuple[slice, slice]]:
    """Generate expanding window splits: train [0:t], val [t:t+val_window]."""
    splits = []
    t = max(min_train, val_window)
    while t + val_window <= length:
        train_slice = slice(0, t)
        val_slice = slice(t, t + val_window)
        splits.append((train_slice, val_slice))
        t += val_window
    if not splits:
        splits.append((slice(0, max(min_train, length // 2)), slice(max(min_train, length // 2), length)))
    return splits


def prepare_dataloaders(dataset: Dataset, cfg: Config, val_window: int = 128, min_train: int = 256):
    """Create expanding-window train/val loaders."""
    splits = create_expanding_splits(len(dataset), val_window, min_train)
    loaders = []
    for train_slice, val_slice in splits:
        train_subset = Subset(dataset, range(train_slice.start, train_slice.stop))
        val_subset = Subset(dataset, range(val_slice.start, val_slice.stop))
        train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False)
        loaders.append((train_loader, val_loader))
    return loaders


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    mae_metric: MeanAbsoluteError,
    rmse_metric: MeanSquaredError,
    acc_metric: Accuracy,
    f1_metric: F1Score,
) -> Tuple[float, Dict[str, float]]:
    """Run evaluation and return loss + metrics."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            targets = {k: v.to(device) for k, v in batch_y.items()}
            outputs = model(batch_x)
            loss = combined_loss(outputs, targets, cfg.lambda_regression, cfg.lambda_classification)
            total_loss += loss.item() * batch_x.size(0)

            mae_metric.update(outputs["regression"], targets["regression"])
            rmse_metric.update(outputs["regression"], targets["regression"])
            acc_metric.update(outputs["classification"], targets["classification"])
            f1_metric.update(outputs["classification"], targets["classification"])

    val_loss = total_loss / len(data_loader.dataset)
    metrics = {
        "mae": float(mae_metric.compute()),
        "rmse": float(rmse_metric.compute()),
        "accuracy": float(acc_metric.compute()),
        "f1": float(f1_metric.compute()),
    }
    mae_metric.reset()
    rmse_metric.reset()
    acc_metric.reset()
    f1_metric.reset()
    return val_loss, metrics


def train_split(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    max_steps: int,
    checkpoint_path: Path,
) -> TrainResult:
    """Train on a single split with early stopping and LR scheduling."""
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    warmup_steps = int(0.1 * max_steps)
    scheduler = linear_warmup_cosine(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=max_steps,
        min_lr=cfg.learning_rate * 0.1,
        base_lr=cfg.learning_rate,
    )

    mae_metric = MeanAbsoluteError().to(device)
    rmse_metric = MeanSquaredError(squared=False).to(device)
    acc_metric = Accuracy(task="multiclass", num_classes=2).to(device)
    f1_metric = F1Score(task="multiclass", num_classes=2).to(device)

    best_loss = float("inf")
    best_f1 = 0.0
    best_state = None
    patience = cfg.early_stopping_patience
    wait = 0

    history: List[Dict] = []
    global_step = 0
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            scheduler.step()
            running_loss += loss.item() * batch_x.size(0)
            global_step += 1

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, metrics = evaluate(model, val_loader, cfg, device, mae_metric, rmse_metric, acc_metric, f1_metric)

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(log_entry)
        logger.info(
            "Epoch %d | Train %.4f | Val %.4f | MAE %.4f | RMSE %.4f | Acc %.4f | F1 %.4f | LR %.6f",
            epoch + 1,
            train_loss,
            val_loss,
            metrics["mae"],
            metrics["rmse"],
            metrics["accuracy"],
            metrics["f1"],
            optimizer.param_groups[0]["lr"],
        )

        improved = False
        if val_loss < best_loss:
            best_loss = val_loss
            improved = True
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            improved = True

        if improved:
            wait = 0
            best_state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "f1": metrics["f1"],
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint_path)
            logger.info("Saved new best model to %s", checkpoint_path)
        else:
            wait += 1

        if wait >= patience:
            logger.info("Early stopping triggered after %d epochs without improvement.", wait)
            break

        if global_step >= max_steps:
            logger.info("Reached max steps: %d", max_steps)
            break

    # Save learning curves
    curves_path = checkpoint_path.parent / "learning_curves.png"
    _save_learning_curves(history, curves_path)
    # Save CSV logs
    csv_path = checkpoint_path.parent / "training_log.csv"
    _save_csv_logs(history, csv_path)

    return TrainResult(best_val_loss=best_loss, best_f1=best_f1, train_history=history)


def _save_learning_curves(history: List[Dict], save_path: Path) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    f1 = [h["f1"] for h in history]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.plot(epochs, f1, label="Val F1")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def _save_csv_logs(history: List[Dict], save_path: Path) -> None:
    if not history:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main() -> TrainResult:
    cfg = get_config()
    seed_everything(cfg.seed)
    device = get_device()
    logger.info("Using device: %s", device)

    raw_df = load_data(cfg)
    X, y_reg, y_cls, _ = build_feature_windows(raw_df, cfg, target_col=getattr(cfg, "target_ticker", "SPY"))
    dataset = MarketWindowDataset(X, y_reg, y_cls)
    loaders = prepare_dataloaders(dataset, cfg, val_window=cfg.val_window, min_train=cfg.min_train_size)

    feature_dim = X.shape[-1]

    results = []
    for split_idx, (train_loader, val_loader) in enumerate(loaders):
        logger.info("Starting split %d/%d", split_idx + 1, len(loaders))
        model = RegimeTransformer(
            input_dim=feature_dim,
            d_model=cfg.embedding_dim,
            nhead=cfg.num_heads,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.embedding_dim * 2,
            dropout=cfg.dropout,
            attn_dropout=cfg.dropout,
            num_regimes=2,
        ).to(device)

        steps_per_epoch = len(train_loader)
        max_steps = max(1, cfg.epochs * steps_per_epoch)
        checkpoint = cfg.processed_data_path / f"best_model_split{split_idx}.pth"
        result = train_split(model, train_loader, val_loader, cfg, device, max_steps, checkpoint)
        results.append(result)

    # For convenience, return the last split result
    return results[-1]


if __name__ == "__main__":
    main()
