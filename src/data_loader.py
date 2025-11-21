"""Production-grade data loading for market and macro series (Colab-compatible).

Features
--------
- Robust Yahoo Finance download with column repair and logging per ticker.
- Handles MultiIndex outputs; falls back across Adj Close / Close / adjclose variants.
- Filters out tickers with missing data.
- FRED macro downloads (CPI, rates, VIX, industrial production) with daily ffill.
- Aligns macro to prices via merge_asof on business dates.
- Caching with checksum metadata for raw price, raw macro, and merged datasets.
- Strict validation: monotonic unique index, inner overlap only, minimum length.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

from .config import Config, get_config

logger = logging.getLogger("market_regime_transformer.data_loader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _checksum(payload: Dict) -> str:
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.md5(blob).hexdigest()


def _load_cached(csv_path: Path, meta_path: Path, expected_checksum: str) -> Optional[pd.DataFrame]:
    """Load cached CSV if checksum matches; else None."""
    if not (csv_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("checksum") != expected_checksum:
            return None
        df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        return df
    except Exception as exc:  # pragma: no cover - cache robustness
        logger.warning("Cache read failed (%s), re-downloading.", exc)
        return None


def _save_cached(df: pd.DataFrame, csv_path: Path, meta_path: Path, checksum: str) -> None:
    _ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=True)
    meta_path.write_text(json.dumps({"checksum": checksum}, indent=2))


# --------------------------------------------------------------------------- #
# Price data
# --------------------------------------------------------------------------- #
def _extract_price_columns(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract a usable price column from a raw Yahoo frame."""
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()

    # Handle MultiIndex (from multi-ticker download)
    if isinstance(df.columns, pd.MultiIndex):
        # First level ticker, second level field
        if ticker in df.columns.get_level_values(0):
            df = df.loc[:, ticker]
        else:
            # Fallback: flatten by first level
            df.columns = [c[1] if isinstance(c, tuple) and len(c) > 1 else c for c in df.columns]
    # Normalize columns to lower for matching
    lower_map = {c.lower(): c for c in df.columns}
    candidates = ["adj close", "adjclose", "close"]
    selected = None
    for cand in candidates:
        if cand in lower_map:
            selected = lower_map[cand]
            break
    if selected is None:
        logger.warning("No Close/AdjClose column for %s", ticker)
        return pd.DataFrame()
    series = df[selected].rename(ticker)
    return series.to_frame()


def download_price_data(
    tickers: List[str],
    start: str,
    end: Optional[str],
    cache_dir: Path,
) -> pd.DataFrame:
    """Download price data from Yahoo Finance with robust handling and caching."""
    _ensure_dir(cache_dir)
    end_date = end or dt.date.today().isoformat()
    checksum = _checksum({"tickers": tickers, "start": start, "end": end_date})
    cache_csv = cache_dir / "prices.csv"
    cache_meta = cache_dir / "prices.meta.json"

    cached = _load_cached(cache_csv, cache_meta, checksum)
    if cached is not None:
        logger.info("Loaded prices from cache: %s", cache_csv)
        return cached

    frames = []
    for ticker in tickers:
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end_date,
                progress=False,
                auto_adjust=False,  # we want Adj Close if available
                group_by="ticker",
            )
            df = _extract_price_columns(raw, ticker)
            if df.empty or df.dropna(how="all").empty:
                logger.warning("No usable data for %s; skipping.", ticker)
                continue
            df = df.dropna()
            logger.info("Downloaded price data for %s (%d rows)", ticker, len(df))
            frames.append(df)
        except Exception as exc:  # pragma: no cover - network robustness
            logger.warning("Download failed for %s (%s); skipping.", ticker, exc)

    if not frames:
        raise RuntimeError("No price data downloaded for any ticker.")

    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.dropna(how="all")
    _save_cached(prices, cache_csv, cache_meta, checksum)
    logger.info("Saved price cache to %s", cache_csv)
    return prices


# --------------------------------------------------------------------------- #
# Macro data
# --------------------------------------------------------------------------- #
def download_macro_data(series: List[str], start: str, end: Optional[str], cache_dir: Path) -> pd.DataFrame:
    """Download FRED macro series with caching and daily forward-fill."""
    _ensure_dir(cache_dir)
    end_date = end or dt.date.today().isoformat()
    checksum = _checksum({"series": series, "start": start, "end": end_date})
    cache_csv = cache_dir / "macro.csv"
    cache_meta = cache_dir / "macro.meta.json"

    cached = _load_cached(cache_csv, cache_meta, checksum)
    if cached is not None:
        logger.info("Loaded macro from cache: %s", cache_csv)
        return cached

    frames = []
    for s in series:
        try:
            df = pdr.DataReader(s, "fred", start=start, end=end_date)
            df.columns = [s]
            frames.append(df)
            logger.info("Downloaded macro series %s (%d rows)", s, len(df))
        except Exception as exc:  # pragma: no cover - network robustness
            logger.warning("Failed to download macro series %s (%s); filling zeros.", s, exc)
            idx = pd.date_range(start=start, end=end_date, freq="D")
            frames.append(pd.DataFrame({s: 0.0}, index=idx))

    macro = pd.concat(frames, axis=1).sort_index()
    macro = macro.resample("D").ffill()
    _save_cached(macro, cache_csv, cache_meta, checksum)
    logger.info("Saved macro cache to %s", cache_csv)
    return macro


# --------------------------------------------------------------------------- #
# Merge logic
# --------------------------------------------------------------------------- #
def _merge_price_macro(prices: pd.DataFrame, macro: pd.DataFrame, min_rows: int = 1000) -> pd.DataFrame:
    """Merge price and macro data safely with validation."""
    prices = prices.sort_index()
    macro = macro.sort_index()

    # Align macro to price dates with merge_asof
    price_df = prices.copy()
    price_df["__merge_key"] = price_df.index
    macro_df = macro.copy()
    macro_df["__merge_key"] = macro_df.index

    merged = pd.merge_asof(
        price_df.sort_values("__merge_key"),
        macro_df.sort_values("__merge_key"),
        on="__merge_key",
        direction="backward",
    ).drop(columns="__merge_key")

    merged.index = prices.index
    merged = merged.dropna()
    merged = merged.loc[~merged.index.duplicated(keep="first")]
    merged = merged.sort_index()

    if not merged.index.is_monotonic_increasing:
        merged = merged.sort_index()

    if merged.index.has_duplicates:
        raise ValueError("Merged DataFrame has duplicate index entries.")

    if len(merged) < min_rows:
        raise ValueError(f"Merged DataFrame too short: {len(merged)} rows (<{min_rows}).")

    return merged


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_data(cfg: Optional[Config] = None) -> pd.DataFrame:
    """Load (or download) price + macro data, merge, validate, and return clean DataFrame."""
    cfg = cfg or get_config()
    start_date = getattr(cfg, "start_date", "2010-01-01")
    end_date = getattr(cfg, "end_date", None)

    raw_dir = cfg.raw_data_path
    processed_dir = cfg.processed_data_path
    _ensure_dir(raw_dir)
    _ensure_dir(processed_dir)

    tickers = getattr(cfg, "tickers", ["^GSPC", "^IXIC", "SPY"])
    macro_series = ["CPIAUCSL", "DGS10", "VIXCLS", "INDPRO"]

    merge_checksum = _checksum({"tickers": tickers, "macro": macro_series, "start": start_date, "end": end_date})
    merged_csv = processed_dir / "merged.csv"
    merged_meta = processed_dir / "merged.meta.json"

    cached_merged = _load_cached(merged_csv, merged_meta, merge_checksum)
    if cached_merged is not None:
        logger.info("Loaded merged dataset from cache: %s", merged_csv)
        return cached_merged

    prices = download_price_data(tickers, start_date, end_date, raw_dir)
    macro = download_macro_data(macro_series, start_date, end_date, raw_dir)
    min_rows = getattr(cfg, "min_merge_rows", 1000)
    merged = _merge_price_macro(prices, macro, min_rows=min_rows)

    _save_cached(merged, merged_csv, merged_meta, merge_checksum)
    logger.info("Saved merged dataset to %s", merged_csv)
    return merged


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.describe())
