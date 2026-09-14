"""Dataset preparation functions with no import-time data loading or fitting."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from config import FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMN, TRAIN_SPLIT
from data_processing import validate_processed_data


@dataclass
class PreparedDatasets:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    scaler: StandardScaler
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor


def load_processed_data(data_path):
    data = pd.read_csv(data_path, index_col=0)
    validate_processed_data(data)
    return data


def chronological_train_test_split(data, train_split=TRAIN_SPLIT):
    """Split ordered observations without shuffling, preserving the original 80/20 split."""
    if not 0 < train_split < 1:
        raise ValueError("train_split must be between 0 and 1.")
    split_index = int(len(data) * train_split)
    if split_index == 0 or split_index == len(data):
        raise ValueError("train_split leaves an empty train or test dataset.")
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def scale_feature_splits(train_df, test_df, feature_columns=FEATURE_COLUMNS):
    """Fit a scaler only on the training period and transform both periods."""
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df.loc[:, list(feature_columns)])
    test_features = scaler.transform(test_df.loc[:, list(feature_columns)])
    return train_features, test_features, scaler


def create_sequences(features, targets, sequence_length=SEQUENCE_LENGTH):
    """Convert 2D features and 1D targets to the existing LSTM sequence alignment."""
    features = np.asarray(features)
    targets = np.asarray(targets)
    if features.ndim != 2:
        raise ValueError("features must be a 2D array.")
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same length.")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    X, y = [], []
    for index in range(sequence_length, len(features)):
        X.append(features[index - sequence_length:index])
        y.append(targets[index])
    return np.asarray(X), np.asarray(y)


def create_feature_sequences(features, sequence_length=SEQUENCE_LENGTH):
    """Create inference sequences using the same input alignment as training."""
    placeholder_targets = np.zeros(len(features), dtype=np.float32)
    sequences, _ = create_sequences(features, placeholder_targets, sequence_length)
    return sequences


def prepare_datasets(
    data,
    feature_columns=FEATURE_COLUMNS,
    target_column=TARGET_COLUMN,
    train_split=TRAIN_SPLIT,
    sequence_length=SEQUENCE_LENGTH,
):
    """Validate, split, scale, sequence, and convert data for model training."""
    validate_processed_data(data)
    if target_column not in data:
        raise ValueError("Target column is missing: {}".format(target_column))
    train_df, test_df = chronological_train_test_split(data, train_split)
    train_features, test_features, scaler = scale_feature_splits(train_df, test_df, feature_columns)
    X_train, y_train = create_sequences(train_features, train_df[target_column].to_numpy(), sequence_length)
    X_test, y_test = create_sequences(test_features, test_df[target_column].to_numpy(), sequence_length)
    return PreparedDatasets(
        train_df=train_df,
        test_df=test_df,
        scaler=scaler,
        X_train=torch.tensor(X_train, dtype=torch.float32),
        y_train=torch.tensor(y_train, dtype=torch.float32),
        X_test=torch.tensor(X_test, dtype=torch.float32),
        y_test=torch.tensor(y_test, dtype=torch.float32),
    )
