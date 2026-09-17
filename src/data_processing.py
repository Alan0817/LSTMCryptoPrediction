"""Reusable market-data download, validation, and feature engineering functions."""

from pathlib import Path

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

from config import (
    DATA_PATH,
    END_DATE,
    INTERVAL,
    START_DATE,
    SYMBOL,
    TARGET_THRESHOLD,
)


PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def download_market_data(
    symbol=SYMBOL,
    start=START_DATE,
    end=END_DATE,
    interval=INTERVAL,
):
    """Download daily OHLCV data and normalize Yahoo's possible MultiIndex columns."""
    data = yf.download(symbol, start=start, end=end, interval=interval)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def validate_ohlcv_data(data):
    """Validate that a dataframe has the columns needed by the existing pipeline."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Market data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("Market data is empty.")
    missing = set(PRICE_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError("Market data is missing required columns: {}".format(sorted(missing)))


def add_target(data, threshold=TARGET_THRESHOLD):
    """Add the existing next-day-positive-return classification target unchanged."""
    result = data.copy()
    result["Return"] = result["Close"].pct_change()
    result["Future_Return"] = result["Close"].pct_change().shift(-1)
    result["Target"] = (result["Future_Return"] > threshold).astype(int)
    return result


def engineer_features(data, target_threshold=TARGET_THRESHOLD):
    """Apply the project's original cleaning, target, and indicator calculations."""
    validate_ohlcv_data(data)
    result = data.dropna().loc[:, list(PRICE_COLUMNS)].copy()
    result = add_target(result, threshold=target_threshold)

    result["RSI"] = RSIIndicator(close=result["Close"]).rsi()
    macd = MACD(close=result["Close"])
    result["MACD"] = macd.macd()
    result["MACD_Signal"] = macd.macd_signal()

    bands = BollingerBands(close=result["Close"])
    result["BB_High"] = bands.bollinger_hband()
    result["BB_Low"] = bands.bollinger_lband()
    result["MA_7"] = result["Close"].rolling(window=7).mean()
    result["MA_30"] = result["Close"].rolling(window=30).mean()
    result["EMA_7"] = result["Close"].ewm(span=7).mean()
    result["Volatility"] = result["Return"].rolling(window=7).std()
    return result.dropna()


def validate_processed_data(data):
    """Validate the processed data needed by model preparation and backtesting."""
    required = {"Future_Return", "Target"}
    from config import FEATURE_COLUMNS

    missing = required.union(FEATURE_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError("Processed data is missing required columns: {}".format(sorted(missing)))
    if data.empty:
        raise ValueError("Processed data is empty.")
    if data.loc[:, list(required.union(FEATURE_COLUMNS))].isnull().any().any():
        raise ValueError("Processed data contains missing required values.")


def save_processed_data(data, output_path=DATA_PATH):
    """Save processed data, creating only the configured output directory if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path)
    return output_path


def run_data_pipeline(output_path=DATA_PATH):
    """Download, engineer, and persist the project's default BTC dataset."""
    raw_data = download_market_data()
    processed_data = engineer_features(raw_data)
    save_processed_data(processed_data, output_path)
    return processed_data
