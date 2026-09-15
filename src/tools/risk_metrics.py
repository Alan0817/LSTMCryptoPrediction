"""Structured adapter for the project's existing performance-metric functions."""

import numpy as np

from strategy import calculate_cumulative_returns, max_drawdown, sharpe_ratio


TRADING_DAYS_PER_YEAR = 252


def get_risk_metrics(returns) -> dict:
    """Return JSON-safe daily-return metrics using the project's current conventions."""
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1 or len(returns) == 0:
        raise ValueError("returns must be a non-empty one-dimensional sequence.")
    if not np.isfinite(returns).all():
        raise ValueError("returns must contain only finite values.")
    if np.std(returns) == 0:
        raise ValueError("returns must have non-zero variation to calculate Sharpe ratio.")

    cumulative = calculate_cumulative_returns(returns)
    return {
        "observations": int(len(returns)),
        "annualization_factor": TRADING_DAYS_PER_YEAR,
        "annualized_volatility": float(np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "sharpe_ratio": float(sharpe_ratio(returns)),
        "maximum_drawdown": float(max_drawdown(cumulative)),
        "cumulative_return": float(cumulative[-1] - 1),
        "assumptions": {
            "observation_frequency": "daily",
            "risk_free_rate": 0.0,
            "sharpe_annualization": "sqrt(252)",
        },
    }
