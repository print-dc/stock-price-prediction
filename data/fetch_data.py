"""
fetch_data.py
Fetches historical stock data from Yahoo Finance via yfinance.
"""

import yfinance as yf
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")


def fetch_stock_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV data for a given ticker symbol.

    Args:
        ticker:   Stock symbol (e.g. 'AAPL', 'TSLA')
        period:   Data window  – '1y', '2y', '5y', 'max'
        interval: Candle size  – '1d', '1wk', '1mo'

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    print(f"[fetch] Downloading {ticker} | period={period} interval={interval}")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")

    # Flatten multi-level columns produced by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    print(f"[fetch] {len(df)} rows fetched  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def save_raw_data(df: pd.DataFrame, ticker: str) -> str:
    """Persist raw data to CSV and return the file path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(path)
    print(f"[fetch] Saved → {path}")
    return path


def load_raw_data(ticker: str) -> pd.DataFrame:
    """Load previously saved CSV, or fetch fresh data if not found."""
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(path):
        print(f"[fetch] Loading cached data from {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return fetch_stock_data(ticker)


if __name__ == "__main__":
    for sym in ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]:
        df = fetch_stock_data(sym)
        save_raw_data(df, sym)
        print(df.tail(3), "\n")
