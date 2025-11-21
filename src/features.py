"""Feature engineering pipeline for the Market Regime-Switching Transformer.

This module builds robust technical and macro features, constructs weakly
supervised regime labels, applies scaling, and exports windowed tensors for
sequence models. Extensive validation ensures index safety and shape parity.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import Config


# --------------------------------------------------------------------------- #
# Helper calculations
# --------------------------------------------------------------------------- #

def _rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std()


def _momentum(prices: pd.Series, window: int) -> pd.Series:
    return prices.pct_change(periods=window)


def _rolling_sharpe(returns: pd.Series, window: int, trading_days: int = 252) -> pd.Series:
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std()
    sharpe = (mu / sigma) * np.sqrt(trading_days)
    return sharpe


def _zscore(series: pd.Series, window: int = 60) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std


def _macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute macro indicators with safe defaults."""
    feats = pd.DataFrame(index=df.index)
    if "CPIAUCSL" in df.columns:
        feats["cpi_yoy"] = df["CPIAUCSL"].pct_change(periods=252)
    if "DGS10" in df.columns:
        feats["rate_change"] = df["DGS10"].diff()
    if "VIXCLS" in df.columns:
        feats["vix_norm"] = _zscore(df["VIXCLS"].ffill(), window=60)
    if "INDPRO" in df.columns:
        feats["indpro_growth"] = df["INDPRO"].pct_change(periods=252)
    return feats


# --------------------------------------------------------------------------- #
# Core feature builder
# --------------------------------------------------------------------------- #

def build_feature_windows(
    data: pd.DataFrame,
    cfg: Config,
    target_col: str = "SPY",
    regime_threshold: float = 0.0,
    scaler_train_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Create windowed tensors with technical + macro features and labels.

    Steps:
        1) Compute technical indicators.
        2) Compute macro indicators.
        3) Concatenate features with inner join.
        4) Build next-day regression target and weakly supervised regime label.
        5) Scale features with StandardScaler (fit on training slice only).
        6) Construct lookback windows and validate shapes.

    Args:
        data: DataFrame with price columns and macro columns.
        cfg: Configuration object (requires lookback_window).
        target_col: Column used for return calculations.
        regime_threshold: Threshold for 30-day rolling return to label regime.
        scaler_train_size: Optional number of rows to fit the scaler on
            (e.g., training slice). If None, fit on all rows.

    Returns:
        X: np.ndarray, shape (samples, lookback_window, num_features)
        y_reg: np.ndarray, shape (samples,) next-day returns
        y_cls: np.ndarray, shape (samples,) binary regime label
        scaler: fitted StandardScaler instance
    """
    # 1) Base validation
    if data.index.has_duplicates:
        data = data[~data.index.duplicated(keep="first")]
    data = data.sort_index()
    if not data.index.is_monotonic_increasing:
        data = data.sort_index()

    if target_col not in data.columns:
        raise ValueError(f"Target column {target_col} not found in data.")

    prices = data[target_col].astype(float)
    returns = prices.pct_change()
    log_returns = np.log(prices / prices.shift(1))

    tech_feats = pd.DataFrame(index=data.index)
    tech_feats["log_return"] = log_returns
    tech_feats["vol_10d"] = _rolling_volatility(returns, 10)
    tech_feats["vol_20d"] = _rolling_volatility(returns, 20)
    tech_feats["vol_60d"] = _rolling_volatility(returns, 60)
    tech_feats["mom_5d"] = _momentum(prices, 5)
    tech_feats["mom_20d"] = _momentum(prices, 20)
    tech_feats["mom_60d"] = _momentum(prices, 60)
    tech_feats["sharpe_20d"] = _rolling_sharpe(returns, 20)
    tech_feats["price_z"] = _zscore(prices, window=60)

    macro_feats = _macro_features(data)

    # 3) Merge using inner join on index
    feature_df = pd.concat([tech_feats, macro_feats], axis=1, join="inner").dropna()

    # 4) Targets
    rolling_30d_return = prices.pct_change(periods=30)
    regime = (rolling_30d_return > regime_threshold).astype(int)
    next_return = returns.shift(-1)

    target_df = pd.DataFrame(
        {"next_return": next_return, "regime": regime}, index=data.index
    ).loc[feature_df.index]

    combined = pd.concat([feature_df, target_df], axis=1, join="inner").dropna()

    # Index safety
    combined = combined.reset_index(drop=True)
    if isinstance(combined.index, pd.MultiIndex):
        raise ValueError("Combined index is MultiIndex; expected flat RangeIndex.")
    combined.index = pd.RangeIndex(start=0, stop=len(combined), step=1)

    # 6) Scaling
    feat_cols = [c for c in combined.columns if c not in {"next_return", "regime"}]
    scaler = StandardScaler()
    fit_end = scaler_train_size if scaler_train_size is not None else len(combined)
    fit_end = max(cfg.lookback_window, min(fit_end, len(combined)))
    scaler.fit(combined.loc[: fit_end - 1, feat_cols])
    scaled_features = scaler.transform(combined[feat_cols])

    # Build windows
    X_list = []
    y_reg_list = []
    y_cls_list = []
    lookback = cfg.lookback_window
    for idx in range(lookback - 1, len(combined)):
        window_slice = scaled_features[idx - lookback + 1 : idx + 1]
        X_list.append(window_slice)
        y_reg_list.append(combined.iloc[idx]["next_return"])
        y_cls_list.append(combined.iloc[idx]["regime"])

    X = np.stack(X_list)
    y_reg = np.asarray(y_reg_list, dtype=np.float32)
    y_cls = np.asarray(y_cls_list, dtype=np.int64)

    # Validation checks
    assert X.shape[0] == y_reg.shape[0] == y_cls.shape[0], "Mismatched sample counts."
    assert X.shape[1] == lookback, "Lookback dimension mismatch."
    assert X.ndim == 3, "X must be 3D (samples, lookback, features)."

    return X, y_reg, y_cls, scaler
