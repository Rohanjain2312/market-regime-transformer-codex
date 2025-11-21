"""Simple backtesting framework for model-predicted daily returns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import get_config


@dataclass
class BacktestResult:
    cumulative_return: float
    max_drawdown: float
    sharpe: float
    volatility: float
    hit_rate: float
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    drawdown_curve: pd.Series


def _compute_stats(returns: pd.Series) -> Tuple[float, float, float, float]:
    """Compute cumulative return, max drawdown, sharpe, volatility."""
    equity = (1 + returns).cumprod()
    cum_ret = equity.iloc[-1] - 1
    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0.0
    return cum_ret, max_dd, sharpe, vol


def _hit_rate(pred: pd.Series, actual: pd.Series) -> float:
    if len(pred) == 0:
        return 0.0
    return float((np.sign(pred) == np.sign(actual)).mean())


def run_backtest(
    predicted_returns: pd.Series,
    actual_returns: pd.Series,
    price_series: pd.Series,
    save_dir: Path,
) -> BacktestResult:
    """Run simple long/cash strategy based on predicted returns."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Align series
    idx = predicted_returns.index.intersection(actual_returns.index).intersection(price_series.index)
    pred = predicted_returns.loc[idx]
    actual = actual_returns.loc[idx]
    prices = price_series.loc[idx]

    # Strategy: long when predicted > 0, else cash
    positions = (pred > 0).astype(float)
    strat_returns = positions * actual
    bh_returns = actual  # buy-and-hold on the same asset

    cum_ret, max_dd, sharpe, vol = _compute_stats(strat_returns)
    hit = _hit_rate(pred, actual)

    strat_equity = (1 + strat_returns).cumprod()
    bh_equity = (1 + bh_returns).cumprod()
    dd_curve = strat_equity / strat_equity.cummax() - 1

    # Save results CSV
    results_csv = save_dir / "backtest_results.csv"
    df_out = pd.DataFrame(
        {
            "predicted_return": pred,
            "actual_return": actual,
            "position": positions,
            "strategy_equity": strat_equity,
            "buy_hold_equity": bh_equity,
            "drawdown": dd_curve,
        }
    )
    df_out.to_csv(results_csv, index_label="date")

    # Plot equity curves
    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot(strat_equity.index, strat_equity.values, label="Strategy")
    plt.plot(bh_equity.index, bh_equity.values, label="Buy & Hold", linestyle="--")
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "equity_curve.png", dpi=300)
    plt.close()

    # Plot drawdown
    plt.figure(figsize=(10, 3), dpi=200)
    plt.plot(dd_curve.index, dd_curve.values, color="red")
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(save_dir / "drawdown.png", dpi=300)
    plt.close()

    return BacktestResult(
        cumulative_return=float(cum_ret),
        max_drawdown=float(max_dd),
        sharpe=float(sharpe),
        volatility=float(vol),
        hit_rate=hit,
        equity_curve=strat_equity,
        benchmark_curve=bh_equity,
        drawdown_curve=dd_curve,
    )


if __name__ == "__main__":
    cfg = get_config()
    # Example placeholder usage: requires predicted/actual returns and price series
    # Users should replace with actual model predictions and matching price data.
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(42)
    actual = pd.Series(rng.normal(0, 0.001, size=len(dates)), index=dates)
    predicted = actual.shift(1).fillna(0)  # trivial predictor
    price = pd.Series(100 * np.exp(np.cumsum(actual)), index=dates)
    result = run_backtest(predicted, actual, price, cfg.processed_data_path)
    print(result)
