"""Data loading utilities for market and macro series with caching."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

from .config import Config, get_config


@dataclass
class DataSources:
    price_tickers: List[str]
    macro_series: List[str]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_price_data(
    tickers: List[str], start: str, end: Optional[str], cache_path: Path
) -> pd.DataFrame:
    """Download price data from Yahoo Finance with CSV caching."""
    _ensure_dir(cache_path.parent)
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")

    end_date = end or dt.date.today().isoformat()
    data = yf.download(tickers, start=start, end=end_date, progress=False)["Adj Close"]
    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.dropna(how="all").sort_index()
    data.to_csv(cache_path, index_label="Date")
    return data


def download_macro_data(series: List[str], start: str, end: Optional[str], cache_path: Path) -> pd.DataFrame:
    """Download macro series from FRED with CSV caching."""
    _ensure_dir(cache_path.parent)
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["DATE"], index_col="DATE")

    end_date = end or dt.date.today().isoformat()
    frames = []
    for s in series:
        fred = pdr.DataReader(s, "fred", start=start, end=end_date)
        fred.columns = [s]
        frames.append(fred)
    macro = pd.concat(frames, axis=1).sort_index()
    macro.to_csv(cache_path, index_label="DATE")
    return macro


def compute_basic_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple returns and log returns for each ticker."""
    prices = prices.sort_index()
    returns = prices.pct_change()
    log_returns = np.log(prices / prices.shift(1))
    feature_frames = []
    for col in prices.columns:
        feature_frames.append(returns[col].rename(f"{col}_ret"))
        feature_frames.append(log_returns[col].rename(f"{col}_logret"))
    features = pd.concat(feature_frames, axis=1)
    return features


def merge_sources(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Merge price and macro data on date, forward-fill missing macro values."""
    macro = macro.rename_axis("Date").sort_index().ffill()
    prices = prices.sort_index()
    merged = prices.join(macro, how="inner")
    merged = merged.resample("B").last().ffill()
    return merged


def load_data(cfg: Optional[Config] = None) -> pd.DataFrame:
    """Main entry: download, cache, merge, and return cleaned DataFrame."""
    cfg = cfg or get_config()
    start_date = getattr(cfg, "start_date", "2015-01-01")
    end_date = getattr(cfg, "end_date", None)
    sources = DataSources(
        price_tickers=["^GSPC", "^IXIC", "SPY"],
        macro_series=["VIXCLS", "DGS10", "CPIAUCSL"],
    )

    price_cache = cfg.raw_data_path / "prices.csv"
    macro_cache = cfg.raw_data_path / "macro.csv"

    prices = download_price_data(sources.price_tickers, start_date, end_date, price_cache)
    macro = download_macro_data(sources.macro_series, start_date, end_date, macro_cache)

    merged = merge_sources(prices, macro)
    features = compute_basic_features(merged[sources.price_tickers])
    full = merged.join(features, how="left").dropna()
    return full


if __name__ == "__main__":
    df = load_data()
    print(df.head())
