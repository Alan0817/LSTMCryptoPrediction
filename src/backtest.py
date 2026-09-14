"""Backtest entry point built from reusable inference, strategy, metric, and plotting code."""

from dataclasses import dataclass

import numpy as np

from config import DATA_PATH, PLOT_PATH, SEQUENCE_LENGTH
from dataset import chronological_train_test_split, load_processed_data
from plotting import plot_cumulative_comparison
from prediction import predict_features
from strategy import (
    calculate_cumulative_returns,
    calculate_strategy_returns,
    max_drawdown,
    sharpe_ratio,
    trade_count,
)


@dataclass
class BacktestResult:
    probabilities: np.ndarray
    signals: np.ndarray
    strategy_returns: np.ndarray
    buy_hold_returns: np.ndarray
    strategy_cumulative: np.ndarray
    buy_hold_cumulative: np.ndarray
    strategy_sharpe: float
    buy_hold_sharpe: float
    strategy_mdd: float
    buy_hold_mdd: float
    number_of_trades: int


def run_backtest(data_path=DATA_PATH, plot_path=PLOT_PATH):
    """Run the original held-out-period backtest using the saved model artifact."""
    data = load_processed_data(data_path)
    _, test_df = chronological_train_test_split(data)
    predictions = predict_features(test_df)
    probabilities = predictions["probability_up"].to_numpy()
    signals = predictions["signal"].to_numpy()
    test_returns = test_df["Future_Return"].to_numpy()[SEQUENCE_LENGTH:]
    if len(signals) != len(test_returns):
        raise AssertionError("Signals and test returns must have equal length.")

    # Intentionally retains the original inverted signal-return convention.
    strategy_returns = calculate_strategy_returns(signals, test_returns)
    buy_hold_returns = test_returns
    strategy_cumulative = calculate_cumulative_returns(strategy_returns)
    buy_hold_cumulative = calculate_cumulative_returns(buy_hold_returns)
    result = BacktestResult(
        probabilities=probabilities,
        signals=signals,
        strategy_returns=strategy_returns,
        buy_hold_returns=buy_hold_returns,
        strategy_cumulative=strategy_cumulative,
        buy_hold_cumulative=buy_hold_cumulative,
        strategy_sharpe=sharpe_ratio(strategy_returns),
        buy_hold_sharpe=sharpe_ratio(buy_hold_returns),
        strategy_mdd=max_drawdown(strategy_cumulative),
        buy_hold_mdd=max_drawdown(buy_hold_cumulative),
        number_of_trades=int(trade_count(signals)),
    )
    plot_cumulative_comparison(strategy_cumulative, buy_hold_cumulative, plot_path)
    return result


def main():
    result = run_backtest()
    print(np.unique(result.signals, return_counts=True))
    print("Strategy Sharpe:", result.strategy_sharpe)
    print("Buy & Hold Sharpe:", result.buy_hold_sharpe)
    print("Strategy MDD:", result.strategy_mdd)
    print("Buy & Hold MDD:", result.buy_hold_mdd)
    print("Number of trades:", result.number_of_trades)


if __name__ == "__main__":
    main()
