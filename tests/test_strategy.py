import numpy as np

from strategy import (
    calculate_strategy_returns,
    max_drawdown,
    sharpe_ratio,
    trade_count,
    generate_signals,
)


def test_signal_thresholds_are_strict_and_preserved():
    probabilities = np.array([0.52, 0.5201, 0.48, 0.4799, 0.5])
    assert generate_signals(probabilities).tolist() == [0, 1, 0, -1, 0]


def test_sharpe_ratio_matches_existing_formula():
    returns = np.array([0.01, 0.02, 0.03])
    expected = returns.mean() / returns.std() * np.sqrt(252)
    assert np.isclose(sharpe_ratio(returns), expected)


def test_max_drawdown_matches_running_peak_formula():
    assert np.isclose(max_drawdown(np.array([1.0, 1.2, 0.9, 1.1])), -0.25)


def test_trade_count_is_total_absolute_position_change():
    assert trade_count(np.array([0, 1, 1, -1])) == 3


def test_strategy_return_sign_convention_is_intentionally_preserved():
    """A +1 signal currently receives the negative market return; do not silently fix it."""
    signals = np.array([1, -1, 0])
    market_returns = np.array([0.10, 0.20, 0.30])
    assert np.allclose(
        calculate_strategy_returns(signals, market_returns),
        np.array([-0.10, 0.20, 0.0]),
    )
