# StockCNN — Time-Series Stock Price Forecasting

> 1-D Convolutional Neural Network for short-term stock trend prediction, with a Flask real-time dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-ff6f00?logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

StockCNN fetches historical OHLCV data from Yahoo Finance, engineers 20+ technical indicators, and trains a dual-head 1-D CNN that simultaneously predicts:

| Head | Task | Metric |
|------|------|--------|
| **Price** | Next-day normalised close (regression) | MAE |
| **Trend** | UP / DOWN direction (classification) | ~82% accuracy |

A Flask dashboard visualises predictions, back-test results, and training curves in real time.

---

## Architecture

```
Input  (batch, 60 timesteps, 20 features)
  │
  ├─ Conv1D(64, k=3, causal) → BN → MaxPool → Dropout
  ├─ Conv1D(128, k=3, causal) → BN → MaxPool → Dropout
  ├─ Conv1D(256, k=3, causal) → BN → GlobalAvgPool
  │
  ├─ Price Head: Dense(128) → Dense(64) → Dense(1, linear)
  └─ Trend Head: Dense(128) → Dense(64) → Dense(1, sigmoid)
```

---

## Features

- **Data pipeline** — `yfinance` fetch → normalization (MinMaxScaler) → sliding-window generation
- **20 features** — OHLCV + SMA/EMA, RSI-14, MACD, Bollinger Bands, ATR, OBV, log-returns, volatility
- **Chronological train/test split** — no data leakage
- **EDA** — stationarity test (ADF), return distributions, Q-Q plots, correlation heatmap
- **Training callbacks** — ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
- **Flask dashboard** — live price chart, actual vs predicted, loss/accuracy curves

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements.txt
```

### 2. Run EDA

```bash
python -m eda.analysis --ticker AAPL
# Plots saved to eda/plots/
```

### 3. Train the model

```bash
python -m model.train --ticker AAPL --epochs 100
# Best model → artifacts/AAPL_best.keras
# Metrics    → artifacts/AAPL_metrics.json
```

### 4. Start the dashboard

```bash
python app/app.py
# Open http://localhost:5000
```

Enter any ticker you have trained (e.g. `AAPL`, `TSLA`) and click **PREDICT →**.

---

## Project Structure

```
stock-price-prediction/
├── data/
│   ├── fetch_data.py        # yfinance download + caching
│   └── preprocess.py        # indicators, scaling, sliding windows
├── eda/
│   └── analysis.py          # ADF test, distributions, heatmaps
├── model/
│   ├── cnn_model.py         # Dual-head 1-D CNN (Keras)
│   ├── train.py             # Full training pipeline + CLI
│   └── predict.py           # Inference helpers
├── app/
│   ├── app.py               # Flask API + routes
│   ├── templates/index.html # Dashboard HTML
│   └── static/
│       ├── style.css        # Dark terminal aesthetic
│       └── script.js        # Chart.js visualisations
├── artifacts/               # Saved models, metrics, plots
├── requirements.txt
└── README.md
```

---

## Results

| Ticker | Trend Accuracy | Train Period |
|--------|---------------|--------------|
| AAPL   | ~82%          | 5 years      |
| TSLA   | ~79%          | 5 years      |
| GOOGL  | ~81%          | 5 years      |

> Results vary with market conditions; past accuracy does not guarantee future performance.

---

## CLI Reference

```bash
# Train with custom hyperparameters
python -m model.train \
  --ticker TSLA \
  --window  60   \   # lookback window (days)
  --horizon  1   \   # forecast horizon
  --epochs  150  \
  --batch-size 32

# EDA for any ticker
python -m eda.analysis --ticker MSFT
```

---

## Dashboard API

| Endpoint | Description |
|----------|-------------|
| `GET /api/predict/<ticker>` | Latest prediction + price history |
| `GET /api/batch/<ticker>`   | Full test-set actual vs predicted |
| `GET /api/metrics/<ticker>` | Saved metrics + training history |
| `GET /api/tickers`          | List of trained tickers |


<img width="950" height="439" alt="image" src="https://github.com/user-attachments/assets/617602d0-56bd-4b44-9da7-013d2acf1af2" />

---

## Disclaimer

This project is for **educational purposes only**. Predictions are not financial advice. Stock markets are inherently unpredictable; do not use this tool for real trading decisions.

---

## License

MIT © 2024
