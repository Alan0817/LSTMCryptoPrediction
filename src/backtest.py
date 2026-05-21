import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

from model import LSTMClassifier
from dataset import X_test, y_test, test_df, SEQUENCE_LENGTH


def sharpe_ratio(returns, risk_free_rate=0):

    excess_returns = returns - risk_free_rate

    return (
        np.mean(excess_returns)
        / np.std(excess_returns)
    ) * np.sqrt(252)

def max_drawdown(cumulative_returns):

    running_max = np.maximum.accumulate(
        cumulative_returns
    )

    drawdown = (
        cumulative_returns - running_max
    ) / running_max

    return drawdown.min()

# Load the trained model
script_path = Path(__file__).resolve().parent
model_path = os.path.join(script_path, "model_weight/lstm_model.pth")
model = torch.load(model_path)
model.eval()

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# DataLoader parameters
BATCH_SIZE = 64

test_dataset = TensorDataset(X_test, y_test)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

probs = []

with torch.no_grad():

    for X_batch, _ in test_loader:

        X_batch = X_batch.to(device)

        outputs = model(X_batch).squeeze()

        probs.extend(outputs.cpu().numpy())

# Convert probabilities to binary signals
# Buy / Cash signals: 1 = Buy, 0 = Cash
signals = (np.array(probs) > 0.5).astype(int)

# Long / Short signals: 1 = Long, -1 = Short
signals = np.where(
    np.array(probs) > 0.52,
    1,
    np.where(
        np.array(probs) < 0.48,
        -1,
        0
    )
)

# signals = pd.Series(signals).shift(1).fillna(0)
print(np.unique(signals, return_counts=True))

# Check the length of signals and test returns
test_returns = test_df['Future_Return'].values[
    SEQUENCE_LENGTH:
]
print("Signals length:", len(signals))
print(test_df[['Close', 'Future_Return']].head(35))
assert len(signals) == len(test_returns)


# Calculate strategy returns
# Buy / Cash signals: 1 = Buy, 0 = Cash
strategy_returns = -signals * test_returns

# Long / Short signals: 1 = Long, -1 = Short
# strategy_returns = np.where(
#     signals == 1,
#     test_returns,
#     -test_returns
# )

# Baseline buy-and-hold returns
buy_hold_returns = test_returns

# Cumulative returns
strategy_cumulative = (
    1 + strategy_returns
).cumprod()

buy_hold_cumulative = (
    1 + buy_hold_returns
).cumprod()

strategy_sharpe = sharpe_ratio(strategy_returns)
buy_hold_sharpe = sharpe_ratio(buy_hold_returns)

print("Strategy Sharpe:", strategy_sharpe)
print("Buy & Hold Sharpe:", buy_hold_sharpe)

# Calculate max drawdown
strategy_mdd = max_drawdown(
    strategy_cumulative
)

buy_hold_mdd = max_drawdown(
    buy_hold_cumulative
)

print("Strategy MDD:", strategy_mdd)
print("Buy & Hold MDD:", buy_hold_mdd)

# Visualize cumulative returns
plt.figure(figsize=(12, 6))

plt.plot(
    strategy_cumulative,
    label='LSTM Strategy'
)

plt.plot(
    buy_hold_cumulative,
    label='Buy & Hold'
)

plt.legend()

plt.title('Strategy Backtest')

plt.xlabel('Time')
plt.ylabel('Portfolio Value')

plt.savefig(os.path.join(script_path, "plots/cumulative_comparison.png"))

# Trading counts
num_trades = np.sum(np.abs(np.diff(signals)))

print("Number of trades:", num_trades)