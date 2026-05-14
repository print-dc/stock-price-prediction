"""
eda/analysis.py
Exploratory Data Analysis for stock time-series data.

Run: python -m eda.analysis --ticker AAPL
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.fetch_data import fetch_stock_data
from data.preprocess import add_technical_indicators, FEATURE_COLS

EDA_DIR = os.path.join(ROOT, "eda", "plots")
os.makedirs(EDA_DIR, exist_ok=True)


def run_eda(ticker: str = "AAPL"):
    ticker = ticker.upper()
    print(f"\n{'='*55}")
    print(f"  EDA  ──  {ticker}")
    print(f"{'='*55}\n")

    df_raw = fetch_stock_data(ticker, period="5y")
    df     = add_technical_indicators(df_raw)

    # ── 1. Summary stats ──────────────────────────────
    print("[ Summary Statistics – OHLCV ]\n")
    print(df_raw[["Open", "High", "Low", "Close", "Volume"]].describe().round(2))
    print()

    # ── 2. Stationarity (ADF test) ───────────────────
    from statsmodels.tsa.stattools import adfuller
    adf_stat, p_val, *_ = adfuller(df["Close"].dropna())
    print(f"ADF Statistic : {adf_stat:.4f}")
    print(f"p-value       : {p_val:.6f}")
    print(f"Stationarity  : {'NON-STATIONARY' if p_val > 0.05 else 'STATIONARY'}\n")

    # ── 3. Return distribution ────────────────────────
    returns = df["Return_1d"].dropna()
    kurt    = float(stats.kurtosis(returns))
    skew    = float(stats.skew(returns))
    print(f"Daily Returns – Skewness: {skew:.3f}  Kurtosis: {kurt:.3f}")
    print(f"                (Normal ~ 0, 3 respectively)\n")

    # ── 4. Correlation – feature importance proxy ─────
    cols_present = [c for c in FEATURE_COLS if c in df.columns]
    corr = df[cols_present].corr()["Close"].drop("Close").sort_values(ascending=False)
    print("[ Feature Correlation with Close ]\n")
    print(corr.round(3).to_string())
    print()

    # ── Plot 1: Price + Volume + Indicators ───────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.5)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(df.index, df["Close"],   color="#00d4ff", lw=1.2, label="Close")
    ax0.plot(df.index, df["SMA_20"],  color="#ff9500", lw=1,   label="SMA 20")
    ax0.plot(df.index, df["SMA_50"],  color="#ff3d5a", lw=1,   label="SMA 50")
    ax0.fill_between(df.index, df["BB_Lower"], df["BB_Upper"], alpha=0.1, color="#00d4ff")
    ax0.set_title(f"{ticker} – Price, SMAs & Bollinger Bands", fontsize=11, color="#cccccc")
    ax0.legend(fontsize=8); ax0.set_facecolor("#0d1117"); ax0.tick_params(colors="#888")

    ax1 = fig.add_subplot(gs[1])
    ax1.bar(df.index, df["Volume"], color="#444", width=1)
    ax1.set_title("Volume", fontsize=10, color="#cccccc")
    ax1.set_facecolor("#0d1117"); ax1.tick_params(colors="#888")

    ax2 = fig.add_subplot(gs[2])
    ax2.plot(df.index, df["RSI_14"], color="#a78bfa", lw=1.2)
    ax2.axhline(70, color="#ff3d5a", lw=0.8, linestyle="--", label="Overbought 70")
    ax2.axhline(30, color="#34d399", lw=0.8, linestyle="--", label="Oversold 30")
    ax2.set_title("RSI (14)", fontsize=10, color="#cccccc")
    ax2.legend(fontsize=8); ax2.set_facecolor("#0d1117"); ax2.tick_params(colors="#888")

    ax3 = fig.add_subplot(gs[3])
    ax3.plot(df.index, df["MACD"],        color="#00d4ff", lw=1.2, label="MACD")
    ax3.plot(df.index, df["MACD_Signal"], color="#ff9500", lw=1.2, label="Signal")
    ax3.bar(df.index, df["MACD"] - df["MACD_Signal"],
            color=np.where(df["MACD"] > df["MACD_Signal"], "#34d399", "#ff3d5a"),
            width=1, alpha=0.6)
    ax3.set_title("MACD", fontsize=10, color="#cccccc")
    ax3.legend(fontsize=8); ax3.set_facecolor("#0d1117"); ax3.tick_params(colors="#888")

    fig.patch.set_facecolor("#0d1117")
    plt.savefig(os.path.join(EDA_DIR, f"{ticker}_indicators.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[eda] Saved indicators chart")

    # ── Plot 2: Return distribution ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d1117")
    axes[0].hist(returns, bins=80, color="#00d4ff", edgecolor="none", alpha=0.8)
    axes[0].set_title(f"{ticker} – Daily Return Distribution", color="#cccccc")
    axes[0].set_facecolor("#0d1117"); axes[0].tick_params(colors="#888")

    stats.probplot(returns, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot (Normal)", color="#cccccc")
    axes[1].set_facecolor("#0d1117"); axes[1].tick_params(colors="#888")
    axes[1].get_lines()[0].set(color="#00d4ff", markersize=2)
    axes[1].get_lines()[1].set(color="#ff9500")

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, f"{ticker}_returns.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[eda] Saved return distribution chart")

    # ── Plot 3: Correlation heatmap ───────────────────
    numeric_cols = [c for c in FEATURE_COLS if c in df.columns]
    corr_matrix  = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 12), facecolor="#0d1117")
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=False, cmap="coolwarm",
        vmin=-1, vmax=1, linewidths=0.3, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"{ticker} – Feature Correlation Matrix", fontsize=12, color="#cccccc", pad=15)
    ax.set_facecolor("#0d1117"); ax.tick_params(colors="#999")
    fig.patch.set_facecolor("#0d1117")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, f"{ticker}_correlation.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[eda] Saved correlation heatmap")
    print(f"\n[eda] All plots saved to {EDA_DIR}/\n")

    return corr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    args = parser.parse_args()
    run_eda(args.ticker)
