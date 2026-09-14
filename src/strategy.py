"""Pure trading-signal, return, and performance-metric functions."""

import numpy as np

from config import LOWER_SIGNAL_THRESHOLD, UPPER_SIGNAL_THRESHOLD


def generate_signals(
    probabilities,
    upper_threshold=UPPER_SIGNAL_THRESHOLD,
    lower_threshold=LOWER_SIGNAL_THRESHOLD,
):
    """Map probabilities to the existing long/short/cash signal values."""
    probabilities = np.asarray(probabilities)
    return np.where(
        probabilities > upper_threshold,
        1,
        np.where(probabilities < lower_threshold, -1, 0),
    )


def calculate_strategy_returns(signals, market_returns):
    """Calculate returns using the project's current, intentionally preserved convention.

    TODO: Review the sign convention as a separate trading-logic change.  At
    present a +1 signal is multiplied by -1, matching the original backtest.
    """
    signals = np.asarray(signals)
    market_returns = np.asarray(market_returns)
    if len(signals) != len(market_returns):
        raise ValueError("Signals and market returns must have the same length.")
    return -signals * market_returns


def calculate_cumulative_returns(returns):
    return (1 + np.asarray(returns)).cumprod()


def sharpe_ratio(returns, risk_free_rate=0):
    excess_returns = np.asarray(returns) - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def max_drawdown(cumulative_returns):
    cumulative_returns = np.asarray(cumulative_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()


def trade_count(signals):
    return np.sum(np.abs(np.diff(np.asarray(signals))))
