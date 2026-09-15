"""Pure trading-signal, return, and performance-metric functions."""

import numpy as np

from config import LOWER_SIGNAL_THRESHOLD, UPPER_SIGNAL_THRESHOLD


def generate_signals(
    probabilities,
    upper_threshold=UPPER_SIGNAL_THRESHOLD,
    lower_threshold=LOWER_SIGNAL_THRESHOLD,
):
    """Map ``P(Target=1)`` to raw threshold labels, not trading exposure.

    A raw value of +1 means the probability exceeds the upper threshold, -1
    means it is below the lower threshold, and 0 is neutral. The current
    backtest converts these labels to contrarian exposure separately.
    """
    probabilities = np.asarray(probabilities)
    return np.where(
        probabilities > upper_threshold,
        1,
        np.where(probabilities < lower_threshold, -1, 0),
    )


def contrarian_exposure(raw_signals):
    """Convert raw threshold labels to the current backtest's exposure.

    The frozen convention maps raw +1 to -1 exposure and raw -1 to +1
    exposure. A zero raw signal remains flat.
    """
    return -np.asarray(raw_signals)


def calculate_strategy_returns(raw_signals, market_returns):
    """Calculate returns from raw labels using the frozen contrarian exposure.

    ``raw_signals`` are threshold labels derived from model probabilities;
    they are not long/short positions. The resulting trading exposure is
    ``contrarian_exposure(raw_signals)`` before multiplying market returns.
    """
    raw_signals = np.asarray(raw_signals)
    market_returns = np.asarray(market_returns)
    if len(raw_signals) != len(market_returns):
        raise ValueError("Signals and market returns must have the same length.")
    return contrarian_exposure(raw_signals) * market_returns


def calculate_documented_strategy_returns(signals, market_returns):
    """Candidate correction matching the documented +1 long / -1 short semantics.

    This is deliberately separate from ``calculate_strategy_returns`` so the
    existing reported backtest remains unchanged during the audit.
    """
    signals = np.asarray(signals)
    market_returns = np.asarray(market_returns)
    if len(signals) != len(market_returns):
        raise ValueError("Signals and market returns must have the same length.")
    return signals * market_returns


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
