"""
train.py
Full training pipeline: fetch → preprocess → train → evaluate → save.

Usage
-----
    python -m model.train --ticker AAPL --epochs 100
    python -m model.train --ticker TSLA --window 60 --horizon 1
"""

import argparse
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── make repo root importable ─────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.fetch_data import fetch_stock_data, save_raw_data
from data.preprocess import prepare_dataset, inverse_transform_close
from model.cnn_model import build_cnn_model, get_callbacks

ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")


def train(
    ticker: str,
    window_size: int = 60,
    horizon: int = 1,
    epochs: int = 100,
    batch_size: int = 64,
    test_ratio: float = 0.2,
):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    ticker = ticker.upper()

    # 1. Data ──────────────────────────────────────────
    df = fetch_stock_data(ticker)
    save_raw_data(df, ticker)

    (X_train, X_test,
     yp_train, yp_test,
     yt_train, yt_test), scaler, n_features = prepare_dataset(
        df, ticker, window_size, horizon, test_ratio
    )

    # 2. Model ─────────────────────────────────────────
    model = build_cnn_model(window_size, n_features)
    model.summary()

    ckpt_path = os.path.join(ARTIFACTS_DIR, f"{ticker}_best.keras")
    callbacks = get_callbacks(ckpt_path)

    # 3. Train ─────────────────────────────────────────
    history = model.fit(
        X_train,
        {"price_output": yp_train, "trend_output": yt_train},
        validation_data=(
            X_test,
            {"price_output": yp_test, "trend_output": yt_test},
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # 4. Evaluate ──────────────────────────────────────
    results = model.evaluate(
        X_test,
        {"price_output": yp_test, "trend_output": yt_test},
        verbose=0,
    )
    metric_names = model.metrics_names
    metrics_dict = dict(zip(metric_names, results))
    trend_acc = metrics_dict.get("trend_output_accuracy", 0)
    print(f"\n{'='*50}")
    print(f"  Ticker          : {ticker}")
    print(f"  Trend Accuracy  : {trend_acc:.4f}  ({trend_acc*100:.1f}%)")
    print(f"{'='*50}\n")

    # 5. Save history & metrics ────────────────────────
    hist_path = os.path.join(ARTIFACTS_DIR, f"{ticker}_history.json")
    with open(hist_path, "w") as f:
        # convert numpy floats → Python floats for JSON serialisation
        serialisable = {k: [float(v) for v in vals]
                        for k, vals in history.history.items()}
        json.dump(serialisable, f, indent=2)

    metrics_path = os.path.join(ARTIFACTS_DIR, f"{ticker}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics_dict.items()}, f, indent=2)

    # 6. Plot training curves ──────────────────────────
    _plot_training(history, ticker)

    # 7. Plot predictions ─────────────────────────────
    pred_price, pred_trend = model.predict(X_test, verbose=0)
    pred_price_real = inverse_transform_close(pred_price.ravel(), scaler)
    true_price_real = inverse_transform_close(yp_test, scaler)
    _plot_predictions(true_price_real, pred_price_real, ticker)

    print(f"[train] Artifacts saved to {ARTIFACTS_DIR}/")
    return model, metrics_dict


def _plot_training(history, ticker: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_title(f"{ticker} – Total Loss")
    axes[0].legend()

    acc_key = "trend_output_accuracy"
    val_acc_key = "val_trend_output_accuracy"
    if acc_key in history.history:
        axes[1].plot(history.history[acc_key], label="train acc")
        axes[1].plot(history.history[val_acc_key], label="val acc")
        axes[1].set_title(f"{ticker} – Trend Accuracy")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, f"{ticker}_training.png"), dpi=120)
    plt.close()


def _plot_predictions(true_vals, pred_vals, ticker: str):
    plt.figure(figsize=(14, 5))
    plt.plot(true_vals, label="Actual", linewidth=1.5)
    plt.plot(pred_vals, label="Predicted", linewidth=1.5, alpha=0.8)
    plt.title(f"{ticker} – Price Predictions (Test Set)")
    plt.xlabel("Days")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, f"{ticker}_predictions.png"), dpi=120)
    plt.close()


# ── CLI ───────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Stock CNN model")
    parser.add_argument("--ticker",     type=str,   default="AAPL")
    parser.add_argument("--window",     type=int,   default=60,  help="Lookback window (days)")
    parser.add_argument("--horizon",    type=int,   default=1,   help="Forecast horizon (days)")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    train(
        ticker=args.ticker,
        window_size=args.window,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
    )
