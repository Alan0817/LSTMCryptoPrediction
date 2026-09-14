"""Plotting functions for deterministic backtest artifacts."""

from pathlib import Path

import matplotlib.pyplot as plt

from config import PLOT_PATH


def plot_cumulative_comparison(strategy_cumulative, buy_hold_cumulative, output_path=PLOT_PATH):
    """Save the original cumulative-return comparison chart."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(strategy_cumulative, label="LSTM Strategy")
    axis.plot(buy_hold_cumulative, label="Buy & Hold")
    axis.legend()
    axis.set_title("Strategy Backtest")
    axis.set_xlabel("Time")
    axis.set_ylabel("Portfolio Value")
    figure.savefig(output_path)
    plt.close(figure)
    return output_path
