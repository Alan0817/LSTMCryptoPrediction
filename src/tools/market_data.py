"""Structured adapter for the existing Yahoo Finance market-data downloader."""

from datetime import date, datetime

import pandas as pd

from data_processing import PRICE_COLUMNS, download_market_data, validate_ohlcv_data


def _iso_timestamp(value):
    return pd.Timestamp(value).isoformat()


def validate_date_range(start_date, end_date):
    """Validate and normalize a requested date range for market-data adapters."""
    try:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
    except (TypeError, ValueError) as error:
        raise ValueError("start_date and end_date must be valid dates.") from error
    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError("start_date must be earlier than end_date.")
    return start, end


def get_market_data(
    symbol: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    downloader=download_market_data,
) -> dict:
    """Return a JSON-safe OHLCV summary using an injectable market-data source."""
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string.")
    start, end = validate_date_range(start_date, end_date)
    data = downloader(symbol=symbol, start=start.date().isoformat(), end=end.date().isoformat())
    validate_ohlcv_data(data)

    indexed_data = data.copy()
    indexed_data.index = pd.to_datetime(indexed_data.index)
    latest = indexed_data.iloc[-1]
    recent_returns = indexed_data["Close"].pct_change().dropna().tail(5)
    return {
        "symbol": symbol,
        "requested_date_range": {
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
        },
        "available_date_range": {
            "start": _iso_timestamp(indexed_data.index[0]),
            "end": _iso_timestamp(indexed_data.index[-1]),
        },
        "observations": int(len(indexed_data)),
        "latest_ohlcv": {
            column: float(latest[column])
            for column in PRICE_COLUMNS
        },
        "recent_returns": [
            {"timestamp": _iso_timestamp(timestamp), "return": float(value)}
            for timestamp, value in recent_returns.items()
        ],
    }
