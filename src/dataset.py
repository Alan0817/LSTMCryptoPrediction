import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch


def create_sequences(features, targets, sequence_length=30):
    X = []
    y = []

    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(targets[i])

    return np.array(X), np.array(y)


df = pd.read_csv("data/btc_processed.csv", index_col=0)

# Check the data
print(df.head())
print(df.isnull().sum())
print(df.dtypes)

# Define features and target
FEATURES = [
    'Close',
    'Volume',
    'RSI',
    'MACD',
    'MACD_Signal',
    'MA_7',
    'MA_30',
    'Volatility'
]

TARGET = 'Target'

# Split into train and test sets
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

scaler = StandardScaler()

train_features = scaler.fit_transform(train_df[FEATURES])
test_features = scaler.transform(test_df[FEATURES])

train_targets = train_df[TARGET].values
test_targets = test_df[TARGET].values

SEQUENCE_LENGTH = 30

X_train, y_train = create_sequences(
    train_features,
    train_targets,
    SEQUENCE_LENGTH
)

X_test, y_test = create_sequences(
    test_features,
    test_targets,
    SEQUENCE_LENGTH
)

# Check shapes
print(X_train.shape)
print(y_train.shape)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print(X_train.shape)