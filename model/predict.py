"""
predict.py
Inference helpers used by the Flask app and CLI.
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.fetch_data import fetch_stock_data
from data.preprocess import (
    add_technical_indicators,
    fit_scalers,
    load_scaler,
    create_windows,
    inverse_transform_close,
    FEATURE_COLS,
)

ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")


def load_model(ticker: str):
    """Load the best saved Keras model for a ticker."""
    import tensorflow as tf
    path = os.path.join(ARTIFACTS_DIR, f"{ticker.upper()}_best.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found for {ticker}. Run `python -m model.train --ticker {ticker}` first."
        )
    return tf.keras.models.load_model(path)


def predict_latest(ticker: str, window_size: int = 60):
    """
    Fetch the most recent data and generate the next-day forecast.

    Returns
    -------
    dict with keys:
        ticker, last_close, predicted_price, predicted_change_pct,
        trend (UP/DOWN), trend_confidence, dates (list of ISO strings),
        close_history (list of floats)
    """
    ticker = ticker.upper()
    model = load_model(ticker)
    scaler = load_scaler(ticker)

    df_raw = fetch_stock_data(ticker, period="1y")
    df = add_technical_indicators(df_raw)

    # Use the already-fitted scaler (don't re-fit on new data)
    cols = [c for c in FEATURE_COLS if c in df.columns]
    import sklearn.preprocessing
    df[cols] = scaler.transform(df[cols])

    if len(df) < window_size:
        raise ValueError(f"Not enough data ({len(df)} rows) for window of {window_size}.")

    window = df[cols].values[-window_size:]           # (60, n_feat)
    X = window[np.newaxis, ...]                        # (1, 60, n_feat)

    pred_price_norm, pred_trend_prob = model.predict(X, verbose=0)
    pred_price = inverse_transform_close(pred_price_norm.ravel(), scaler)[0]

    last_close = df_raw["Close"].iloc[-1]
    change_pct = ((pred_price - last_close) / last_close) * 100

    # Historical close series for charting
    dates  = [d.strftime("%Y-%m-%d") for d in df_raw.index[-90:]]
    closes = df_raw["Close"].values[-90:].tolist()

    return {
        "ticker":                ticker,
        "last_close":            round(float(last_close), 2),
        "predicted_price":       round(float(pred_price), 2),
        "predicted_change_pct":  round(float(change_pct), 2),
        "trend":                 "UP" if pred_trend_prob[0, 0] > 0.5 else "DOWN",
        "trend_confidence":      round(float(pred_trend_prob[0, 0]) * 100, 1),
        "dates":                 dates,
        "close_history":         [round(float(c), 2) for c in closes],
    }


def predict_batch(ticker: str, window_size: int = 60):
    """
    Run the model over the entire test window and return actual vs predicted
    for the dashboard chart.
    """
    ticker = ticker.upper()
    model = load_model(ticker)
    scaler = load_scaler(ticker)

    df_raw = fetch_stock_data(ticker, period="3y")
    df = add_technical_indicators(df_raw)
    cols = [c for c in FEATURE_COLS if c in df.columns]
    df[cols] = scaler.transform(df[cols])

    X, y_price, y_trend = create_windows(df, window_size=window_size)
    split = int(len(X) * 0.8)
    X_test, yp_test = X[split:], y_price[split:]

    # Corresponding dates (offset by window_size)
    dates_all = df_raw.index.tolist()
    test_dates = dates_all[split + window_size: split + window_size + len(X_test)]

    pred_price_norm, _ = model.predict(X_test, verbose=0)
    pred_prices = inverse_transform_close(pred_price_norm.ravel(), scaler)
    true_prices = inverse_transform_close(yp_test, scaler)

    return {
        "dates":      [d.strftime("%Y-%m-%d") for d in test_dates],
        "actual":     [round(float(v), 2) for v in true_prices],
        "predicted":  [round(float(v), 2) for v in pred_prices],
    }
