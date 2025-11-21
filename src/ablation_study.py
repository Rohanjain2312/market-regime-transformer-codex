"""Ablation study script for the Market Regime-Switching Transformer.

Variants:
- No macro features
- No volatility features
- Fewer attention heads
- LSTM baseline

Logs performance for each variant and saves a results table to disk.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import Accuracy, F1Score, MeanAbsoluteError, MeanSquaredError

from .config import Config, get_config, seed_everything
from .data_loader import load_data
from .features import (
    macro_indicators,
    momentum_signal,
    regime_labels,
    rolling_volatility,
)
from .model import RegimeTransformer, combined_loss

logger = logging.getLogger("market_regime_transformer.ablation")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class MarketWindowDataset(Dataset):
    """Windowed dataset with regression and classification targets."""

    def __init__(self, X: np.ndarray, y_reg: np.ndarray, y_cls: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], {"regression": self.y_reg[idx], "classification": self.y_cls[idx]}


class LSTMRegime(nn.Module):
    """Simple LSTM baseline with dual heads."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, num_regimes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.cls_head = nn.Linear(hidden_dim, num_regimes)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        regression = self.reg_head(pooled).squeeze(-1)
        classification = self.cls_head(pooled)
        return {"regression": regression, "classification": classification}


def directional_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute sign accuracy for regression outputs."""
    return float((np.sign(pred) == np.sign(true)).mean())


def build_feature_windows_variant(
    data: pd.DataFrame,
    cfg: Config,
    target_col: str = "SPY",
    include_macro: bool = True,
    include_vol: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create lookback windows with optional macro/volatility exclusion."""
    df = data.sort_index()
    prices = df[target_col]
    daily_returns = prices.pct_change()
    log_returns_res = np.log(prices / prices.shift(1))

    feat_df = []
    # Base returns
    feat_df.append(daily_returns.rename("ret_1d"))
    feat_df.append(log_returns_res.rename("logret_1d"))

    if include_vol:
        feat_df.append(rolling_volatility(daily_returns, 20).rename("vol_20d"))
        feat_df.append(rolling_volatility(daily_returns, 60).rename("vol_60d"))

    feat_df.extend(
        [
            momentum_signal(prices, 5).rename("mom_5d"),
            momentum_signal(prices, 20).rename("mom_20d"),
            momentum_signal(prices, 60).rename("mom_60d"),
        ]
    )

    feature_matrix = pd.concat(feat_df, axis=1)

    if include_macro:
        macro_feats = macro_indicators(df)
        feature_matrix = feature_matrix.join(macro_feats, how="left")

    rolling_60d_return = prices.pct_change(periods=60)
    labels = regime_labels(rolling_60d_return, threshold=0.0)
    next_day_return = daily_returns.shift(-1)

    combined = feature_matrix.join({"next_return": next_day_return, "regime": labels})
    combined = combined.dropna()

    lookback = cfg.lookback_window
    feature_cols = [col for col in combined.columns if col not in {"next_return", "regime"}]
    X_list: List[np.ndarray] = []
    y_reg_list: List[float] = []
    y_cls_list: List[int] = []

    for idx in range(lookback - 1, len(combined)):
        window_slice = combined.iloc[idx - lookback + 1 : idx + 1]
        X_list.append(window_slice[feature_cols].values)
        y_reg_list.append(combined.iloc[idx]["next_return"])
        y_cls_list.append(combined.iloc[idx]["regime"])

    X = np.stack(X_list)
    y_reg = np.array(y_reg_list, dtype=np.float32)
    y_cls = np.array(y_cls_list, dtype=np.int64)
    return X, y_reg, y_cls


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> Dict[str, float]:
    """Train the model and return validation metrics."""
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    mae = MeanAbsoluteError().to(device)
    rmse = MeanSquaredError(squared=False).to(device)
    acc = Accuracy(task="multiclass", num_classes=2).to(device)
    f1 = F1Score(task="multiclass", num_classes=2).to(device)

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

    model.eval()
    reg_preds, reg_true, cls_preds, cls_true = [], [], [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            targets = {k: v.to(device) for k, v in batch_y.items()}
            outputs = model(batch_x)
            mae.update(outputs["regression"], targets["regression"])
            rmse.update(outputs["regression"], targets["regression"])
            acc.update(outputs["classification"], targets["classification"])
            f1.update(outputs["classification"], targets["classification"])

            reg_preds.append(outputs["regression"].cpu().numpy())
            reg_true.append(targets["regression"].cpu().numpy())
            cls_preds.append(outputs["classification"].argmax(dim=1).cpu().numpy())
            cls_true.append(targets["classification"].cpu().numpy())

    reg_preds_arr = np.concatenate(reg_preds).flatten()
    reg_true_arr = np.concatenate(reg_true).flatten()
    cls_preds_arr = np.concatenate(cls_preds).flatten()
    cls_true_arr = np.concatenate(cls_true).flatten()

    metrics = {
        "mae": float(mae.compute()),
        "rmse": float(rmse.compute()),
        "directional_accuracy": directional_accuracy(reg_preds_arr, reg_true_arr),
        "accuracy": float(acc.compute()),
        "f1_macro": float(f1.compute()),
    }
    mae.reset()
    rmse.reset()
    acc.reset()
    f1.reset()
    return metrics


def run_variant(
    name: str,
    cfg: Config,
    X: np.ndarray,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    model_builder,
) -> Dict[str, float]:
    """Train/evaluate a single variant and return metrics."""
    dataset = MarketWindowDataset(X, y_reg, y_cls)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device(cfg.device)
    model = model_builder().to(device)
    metrics = train_and_evaluate(model, train_loader, val_loader, cfg, device)
    logger.info("%s -> %s", name, metrics)
    return metrics


def main() -> None:
    cfg = get_config()
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    logger.info("Running ablation study on device: %s", device)
    logger.info("Config: %s", asdict(cfg))

    raw_df = load_data(cfg)

    results: List[Dict[str, float]] = []

    # Baseline features
    X_base, y_reg_base, y_cls_base = build_feature_windows_variant(raw_df, cfg, include_macro=True, include_vol=True)

    # Variant: No macro
    X_nomacro, y_reg_nomacro, y_cls_nomacro = build_feature_windows_variant(
        raw_df, cfg, include_macro=False, include_vol=True
    )

    # Variant: No volatility
    X_novol, y_reg_novol, y_cls_novol = build_feature_windows_variant(
        raw_df, cfg, include_macro=True, include_vol=False
    )

    def transformer_builder(num_heads: int) -> nn.Module:
        return RegimeTransformer(
            input_dim=X_base.shape[-1],
            d_model=cfg.embedding_dim,
            nhead=num_heads,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.embedding_dim * 2,
            dropout=cfg.dropout,
            num_regimes=2,
        )

    # Baseline transformer
    results.append(
        {"variant": "baseline"}
        | run_variant("baseline", cfg, X_base, y_reg_base, y_cls_base, lambda: transformer_builder(cfg.num_heads))
    )

    # No macro features
    results.append(
        {"variant": "no_macro"}
        | run_variant("no_macro", cfg, X_nomacro, y_reg_nomacro, y_cls_nomacro, lambda: transformer_builder(cfg.num_heads))
    )

    # No volatility features
    results.append(
        {"variant": "no_volatility"}
        | run_variant("no_volatility", cfg, X_novol, y_reg_novol, y_cls_novol, lambda: transformer_builder(cfg.num_heads))
    )

    # Fewer heads
    fewer_heads = max(1, cfg.num_heads // 2)
    results.append(
        {"variant": f"fewer_heads_{fewer_heads}"}
        | run_variant("fewer_heads", cfg, X_base, y_reg_base, y_cls_base, lambda: transformer_builder(fewer_heads))
    )

    # LSTM baseline
    def lstm_builder() -> nn.Module:
        return LSTMRegime(
            input_dim=X_base.shape[-1],
            hidden_dim=cfg.embedding_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            num_regimes=2,
        )

    results.append({"variant": "lstm_baseline"} | run_variant("lstm_baseline", cfg, X_base, y_reg_base, y_cls_base, lstm_builder))

    results_path = cfg.processed_data_path / "ablation_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(results).to_csv(results_path, index=False)
    logger.info("Ablation results saved to %s", results_path)


if __name__ == "__main__":
    main()
