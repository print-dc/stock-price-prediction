"""
preprocess.py
Preprocessing pipeline: feature engineering, normalisation, sliding-window generation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import joblib
import os

SCALER_DIR = os.path.join(os.path.dirname(__file__), "scalers")


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators and append them as new columns.

    Indicators added
    ----------------
    SMA_10, SMA_20, SMA_50   – Simple Moving Averages
    EMA_10, EMA_20            – Exponential Moving Averages
    RSI_14                    – Relative Strength Index
    MACD, MACD_Signal         – MACD line and signal line
    BB_Upper, BB_Lower        – Bollinger Bands (20-period, 2σ)
    ATR_14                    – Average True Range
    OBV                       – On-Balance Volume
    Return_1d, Return_5d      – Log returns
    Volatility_10             – Rolling 10-day return std
    """
    d = df.copy()

    # ── Moving Averages ──────────────────────────────
    for w in (10, 20, 50):
        d[f"SMA_{w}"] = d["Close"].rolling(w).mean()
    for w in (10, 20):
        d[f"EMA_{w}"] = d["Close"].ewm(span=w, adjust=False).mean()

    # ── RSI ──────────────────────────────────────────
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    d["RSI_14"] = 100 - (100 / (1 + rs))

    # ── MACD ─────────────────────────────────────────
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()

    # ── Bollinger Bands ───────────────────────────────
    sma20 = d["Close"].rolling(20).mean()
    std20 = d["Close"].rolling(20).std()
    d["BB_Upper"] = sma20 + 2 * std20
    d["BB_Lower"] = sma20 - 2 * std20

    # ── ATR ───────────────────────────────────────────
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - d["Close"].shift()).abs(),
        (d["Low"]  - d["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    d["ATR_14"] = tr.rolling(14).mean()

    # ── OBV ───────────────────────────────────────────
    obv = [0]
    for i in range(1, len(d)):
        if d["Close"].iloc[i] > d["Close"].iloc[i - 1]:
            obv.append(obv[-1] + d["Volume"].iloc[i])
        elif d["Close"].iloc[i] < d["Close"].iloc[i - 1]:
            obv.append(obv[-1] - d["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    d["OBV"] = obv

    # ── Returns & Volatility ──────────────────────────
    d["Return_1d"]    = np.log(d["Close"] / d["Close"].shift(1))
    d["Return_5d"]    = np.log(d["Close"] / d["Close"].shift(5))
    d["Volatility_10"] = d["Return_1d"].rolling(10).std()

    d.dropna(inplace=True)
    return d


# ─────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────

FEATURE_COLS: List[str] = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_10", "SMA_20", "SMA_50",
    "EMA_10", "EMA_20",
    "RSI_14",
    "MACD", "MACD_Signal",
    "BB_Upper", "BB_Lower",
    "ATR_14", "OBV",
    "Return_1d", "Return_5d", "Volatility_10",
]


def fit_scalers(df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Fit a MinMaxScaler on the feature set and persist it."""
    os.makedirs(SCALER_DIR, exist_ok=True)
    cols = [c for c in FEATURE_COLS if c in df.columns]
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    joblib.dump(scaler, os.path.join(SCALER_DIR, f"{ticker}_scaler.pkl"))
    return df, scaler


def load_scaler(ticker: str) -> MinMaxScaler:
    path = os.path.join(SCALER_DIR, f"{ticker}_scaler.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No scaler found for {ticker}. Run training first.")
    return joblib.load(path)


def inverse_transform_close(values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Inverse-transform only the Close column (index 3 in FEATURE_COLS)."""
    close_idx = FEATURE_COLS.index("Close")
    dummy = np.zeros((len(values), len(FEATURE_COLS)))
    dummy[:, close_idx] = values.ravel()
    return scaler.inverse_transform(dummy)[:, close_idx]


# ─────────────────────────────────────────────
# Sliding Window
# ─────────────────────────────────────────────

def create_windows(
    df: pd.DataFrame,
    window_size: int = 60,
    horizon: int = 1,
    target_col: str = "Close",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate overlapping windows for supervised time-series learning.

    Returns
    -------
    X       : (N, window_size, n_features)
    y_price : (N,)  – next-day normalised Close
    y_trend : (N,)  – 1 if price goes up, 0 otherwise
    """
    cols = [c for c in FEATURE_COLS if c in df.columns]
    data = df[cols].values
    close_idx = cols.index(target_col)

    X, y_price, y_trend = [], [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i : i + window_size])
        future_close  = data[i + window_size + horizon - 1, close_idx]
        current_close = data[i + window_size - 1, close_idx]
        y_price.append(future_close)
        y_trend.append(int(future_close > current_close))

    return np.array(X), np.array(y_price, dtype=np.float32), np.array(y_trend, dtype=np.int32)


def time_split(
    X: np.ndarray,
    y_price: np.ndarray,
    y_trend: np.ndarray,
    test_ratio: float = 0.2,
):
    """Chronological (non-shuffled) train / test split."""
    split = int(len(X) * (1 - test_ratio))
    return (
        X[:split], X[split:],
        y_price[:split], y_price[split:],
        y_trend[:split], y_trend[split:],
    )


# ─────────────────────────────────────────────
# Full pipeline helper
# ─────────────────────────────────────────────

def prepare_dataset(
    df: pd.DataFrame,
    ticker: str,
    window_size: int = 60,
    horizon: int = 1,
    test_ratio: float = 0.2,
):
    """End-to-end: indicators → scale → windows → split."""
    df = add_technical_indicators(df)
    df, scaler = fit_scalers(df, ticker)
    X, y_price, y_trend = create_windows(df, window_size, horizon)
    splits = time_split(X, y_price, y_trend, test_ratio)
    n_features = X.shape[2]
    print(f"[preprocess] X_train={splits[0].shape}  X_test={splits[1].shape}  features={n_features}")
    return splits, scaler, n_features
