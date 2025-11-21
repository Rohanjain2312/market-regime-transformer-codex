# Market Regime-Switching Transformer

Transformer-based model that predicts next-day returns (regression) and classifies market regimes (classification) using market and macroeconomic signals.

## Problem Statement

Financial markets exhibit regime shifts (risk-on/off). The goal is to jointly forecast next-day returns and detect the prevailing regime so downstream strategies can adapt position sizing and risk controls.

## Datasets

- Yahoo Finance: S&P 500 (`^GSPC`), Nasdaq (`^IXIC`), SPY ETF (`SPY`)
- FRED: VIX (`VIXCLS`), 10Y Treasury Yield (`DGS10`), CPI (`CPIAUCSL`)
- All series are merged on business dates with forward-filled macro data.

## Model Architecture (ASCII)

```
Prices + Macro --> Linear Embed --> + Positional Encoding --> Transformer Encoder (N layers)
                                           | pooled reps
                        -------------------+-------------------
                        |                                   |
                Regression Head (MSE)           Classification Head (CrossEntropy)
```

## Key Features

- Rolling volatility (20d/60d) and momentum signals (5d/20d/60d)
- Macro indicators: CPI YoY, rate changes, VIX levels/changes
- Weakly supervised regime labels from rolling 60d returns
- Dual-head transformer with combined loss `L = λ1*MSE + λ2*CrossEntropy`
- Rolling validation splits for time-series robustness
- TorchMetrics for MAE, RMSE, Accuracy, F1

## Evaluation

Regression: MAE, RMSE, Directional Accuracy  
Classification: Accuracy, F1 (macro), Confusion Matrix (plotted and saved)

## Project Structure

```
market_regime_transformer/
├─ data/                      # Raw/processed data, checkpoints, plots
├─ notebooks/                 # EDA templates
├─ src/
│  ├─ config.py               # Hyperparameters, paths, seeding
│  ├─ data_loader.py          # Yahoo/FRED download, merge, caching
│  ├─ features.py             # Feature engineering and labeling
│  ├─ model.py                # Transformer encoder + dual heads
│  ├─ train.py                # Rolling training loop with metrics
│  ├─ evaluate.py             # Offline evaluation + confusion matrix
│  └─ visualize.py            # Attention/regime plotting
└─ requirements.txt
```

## How to Run

1. Create a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train with default config:
   ```bash
   python -m src.train
   ```
4. Evaluate the saved model (default split0):
   ```bash
   python -m src.evaluate
   ```

Configuration (tickers, spans, hyperparameters, loss weights, paths) is centralized in `src/config.py`. Checkpoints and plots are stored under `data/processed` by default.
