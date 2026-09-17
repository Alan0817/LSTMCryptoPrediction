import numpy as np

from strategy import (
    calculate_documented_strategy_returns,
    calculate_strategy_returns,
    contrarian_exposure,
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
    """Raw labels map to contrarian exposure; do not silently change this."""
    raw_signals = np.array([1, -1, 0])
    market_returns = np.array([0.10, 0.20, 0.30])
    assert np.array_equal(contrarian_exposure(raw_signals), np.array([-1, 1, 0]))
    assert np.allclose(
        calculate_strategy_returns(raw_signals, market_returns),
        np.array([-0.10, 0.20, 0.0]),
    )


def test_documented_long_short_semantics_are_the_opposite_of_current_convention():
    signals = np.array([1, -1, 0])
    market_returns = np.array([0.10, 0.20, 0.30])
    assert np.allclose(
        calculate_documented_strategy_returns(signals, market_returns),
        np.array([0.10, -0.20, 0.0]),
    )
