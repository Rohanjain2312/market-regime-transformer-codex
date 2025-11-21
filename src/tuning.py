"""Hyperparameter tuning with Optuna for the Regime Transformer."""

from __future__ import annotations

import json
import logging
from typing import Tuple

import numpy as np
import optuna
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

from .config import Config, get_config, seed_everything
from .data_loader import load_data
from .features import build_feature_windows
from .model import RegimeTransformer, combined_loss

logger = logging.getLogger("market_regime_transformer.tuning")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class MarketWindowDataset(Dataset):
    """Dataset wrapping windowed features with both targets."""

    def __init__(self, X: np.ndarray, y_reg: np.ndarray, y_cls: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], {"regression": self.y_reg[idx], "classification": self.y_cls[idx]}


def make_loaders(dataset: Dataset, batch_size: int, val_frac: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Split dataset into train/val loaders."""
    val_size = max(1, int(len(dataset) * val_frac))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def objective(trial: optuna.Trial, cfg: Config, dataset: Dataset, feature_dim: int, device: torch.device) -> float:
    """Optuna objective minimizing validation loss."""
    # Sample hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    nhead = trial.suggest_int("num_heads", 2, 8, step=2)
    num_layers = trial.suggest_int("num_layers", 1, 4)

    train_loader, val_loader = make_loaders(dataset, cfg.batch_size)

    model = RegimeTransformer(
        input_dim=feature_dim,
        d_model=cfg.embedding_dim,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=cfg.embedding_dim * 2,
        dropout=dropout,
        num_regimes=2,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(cfg.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            targets = {k: v.to(device) for k, v in batch_y.items()}
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = combined_loss(outputs, targets, cfg.lambda_regression, cfg.lambda_classification)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            targets = {k: v.to(device) for k, v in batch_y.items()}
            outputs = model(batch_x)
            loss = combined_loss(outputs, targets, cfg.lambda_regression, cfg.lambda_classification)
            total_loss += loss.item() * batch_x.size(0)

    val_loss = total_loss / len(val_loader.dataset)
    trial.report(val_loss, step=0)
    return val_loss


def main() -> None:
    cfg = get_config()
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    logger.info("Starting Optuna tuning on device: %s", device)

    # Load data and build windows
    raw_df = load_data(cfg)
    X, y_reg, y_cls = build_feature_windows(raw_df, cfg, target_col="SPY")
    dataset = MarketWindowDataset(X, y_reg, y_cls)
    feature_dim = X.shape[-1]

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, cfg, dataset, feature_dim, device), n_trials=20)

    logger.info("Best trial: %s", study.best_trial.number)
    logger.info("Best value (val loss): %.6f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    cfg.processed_data_path.mkdir(parents=True, exist_ok=True)
    best_trial_path = cfg.processed_data_path / "best_trial.json"
    with best_trial_path.open("w", encoding="utf-8") as f:
        json.dump({"value": study.best_value, "params": study.best_params}, f, indent=2)
    study.trials_dataframe().to_csv(cfg.processed_data_path / "optuna_trials.csv", index=False)
    logger.info("Saved Optuna results to %s", cfg.processed_data_path)


if __name__ == "__main__":
    main()
