"""Feature engineering utilities for the Market Regime-Switching Transformer.

This module constructs rolling technical features, macro indicators, and
weakly supervised regime labels, then converts them into lookback windows
compatible with sequence models.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .config import Config


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Calculate rolling volatility (standard deviation) of returns."""
    return returns.rolling(window).std()


def momentum_signal(prices: pd.Series, window: int) -> pd.Series:
    """Compute momentum as percentage change over the lookback window."""
    return prices.pct_change(periods=window)


def macro_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create macro indicators from CPI, interest rates, and VIX levels.

    Expected columns:
        - CPIAUCSL: Consumer Price Index (levels)
        - DGS10: 10-year Treasury yield (levels)
        - VIXCLS: VIX index (levels)
    """
    indicators = pd.DataFrame(index=df.index)
    if "CPIAUCSL" in df.columns:
        indicators["cpi_yoy"] = df["CPIAUCSL"].pct_change(periods=252)
    if "DGS10" in df.columns:
        indicators["rate_change"] = df["DGS10"].diff()
    if "VIXCLS" in df.columns:
        indicators["vix_level"] = df["VIXCLS"]
        indicators["vix_change_5d"] = df["VIXCLS"].pct_change(periods=5)
    return indicators


def regime_labels(returns_60d: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Generate binary regime labels using weak supervision."""
    return (returns_60d > threshold).astype(int)


def build_feature_windows(
    data: pd.DataFrame,
    cfg: Config,
    target_col: str = "SPY",
    regime_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create model-ready tensors.

    Args:
        data: DataFrame containing price columns and macro series.
        cfg: Configuration with lookback_window.
        target_col: Column to predict next-day return for.
        regime_threshold: Threshold for classifying the 60d rolling return regime.

    Returns:
        X: np.ndarray of shape (samples, lookback_window, feature_dim)
        y_reg: np.ndarray of next-day returns aligned with X
        y_cls: np.ndarray of regime labels aligned with X
    """
    data = data.sort_index()
    prices = data[target_col]
    daily_returns = prices.pct_change()
    log_returns_res = np.log(prices / prices.shift(1))

    feat_df = pd.DataFrame(index=data.index)
    feat_df["ret_1d"] = daily_returns
    feat_df["logret_1d"] = log_returns_res
    feat_df["vol_20d"] = rolling_volatility(daily_returns, 20)
    feat_df["vol_60d"] = rolling_volatility(daily_returns, 60)
    feat_df["mom_5d"] = momentum_signal(prices, 5)
    feat_df["mom_20d"] = momentum_signal(prices, 20)
    feat_df["mom_60d"] = momentum_signal(prices, 60)

    macro_feats = macro_indicators(data)
    feat_df = feat_df.join(macro_feats, how="left")

    rolling_60d_return = prices.pct_change(periods=60)
    labels = regime_labels(rolling_60d_return, threshold=regime_threshold)
    next_day_return = daily_returns.shift(-1)

    combined = feat_df.join(
        {
            "next_return": next_day_return,
            "regime": labels,
        }
    )
    combined = combined.dropna()

    lookback = cfg.lookback_window
    feature_cols = [col for col in combined.columns if col not in {"next_return", "regime"}]
    X_list = []
    y_reg_list = []
    y_cls_list = []

    for idx in range(lookback - 1, len(combined)):
        window_slice = combined.iloc[idx - lookback + 1 : idx + 1]
        X_list.append(window_slice[feature_cols].values)
        y_reg_list.append(combined.iloc[idx]["next_return"])
        y_cls_list.append(combined.iloc[idx]["regime"])

    X = np.stack(X_list)
    y_reg = np.array(y_reg_list, dtype=np.float32)
    y_cls = np.array(y_cls_list, dtype=np.int64)
    return X, y_reg, y_cls
