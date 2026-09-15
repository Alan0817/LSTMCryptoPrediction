"""Structured adapter for the project's existing feature-engineering definitions."""

import pandas as pd

from data_processing import engineer_features


INDICATOR_COLUMNS = ("RSI", "MACD", "MACD_Signal", "MA_7", "MA_30", "Volatility")


def get_technical_indicators(market_data: pd.DataFrame) -> dict:
    """Return the latest project-defined technical indicators as plain Python values."""
    engineered = engineer_features(market_data)
    if engineered.empty:
        raise ValueError("Insufficient historical data to calculate technical indicators.")

    latest = engineered.iloc[-1]
    return {
        "timestamp": pd.Timestamp(engineered.index[-1]).isoformat(),
        "indicators": {
            column: float(latest[column])
            for column in INDICATOR_COLUMNS
        },
    }
