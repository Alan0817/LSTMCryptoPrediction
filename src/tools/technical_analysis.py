"""Structured adapter for the project's existing feature-engineering definitions."""

import pandas as pd

from data_processing import engineer_features


INDICATOR_COLUMNS = ("RSI", "MACD", "MACD_Signal", "MA_7", "MA_30", "Volatility")


def get_technical_indicators(market_data: pd.DataFrame) -> dict:
    """Return the latest project-defined technical indicators as plain Python values."""
    return get_technical_indicators_from_features(engineer_features(market_data))


def get_technical_indicators_from_features(feature_data: pd.DataFrame) -> dict:
    """Summarize already-engineered project features without recalculating them."""
    if not isinstance(feature_data, pd.DataFrame):
        raise TypeError("Feature data must be a pandas DataFrame.")
    missing = set(INDICATOR_COLUMNS).difference(feature_data.columns)
    if missing:
        raise ValueError("Feature data is missing technical indicator columns: {}".format(sorted(missing)))
    if feature_data.empty:
        raise ValueError("Insufficient historical data to calculate technical indicators.")

    latest = feature_data.iloc[-1]
    return {
        "timestamp": pd.Timestamp(feature_data.index[-1]).isoformat(),
        "indicators": {
            column: float(latest[column])
            for column in INDICATOR_COLUMNS
        },
    }
