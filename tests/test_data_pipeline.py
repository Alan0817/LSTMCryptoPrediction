import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS
from data_processing import add_target
from dataset import (
    chronological_train_test_split,
    create_sequences,
    create_test_sequences_with_train_context,
)


def test_expected_model_feature_columns_are_stable():
    assert FEATURE_COLUMNS == (
        "Close", "Volume", "RSI", "MACD", "MACD_Signal", "MA_7", "MA_30", "Volatility"
    )


def test_target_uses_next_day_return_and_existing_threshold():
    data = pd.DataFrame({"Close": [100.0, 101.0, 100.0]})
    result = add_target(data)
    assert np.isclose(result.loc[0, "Future_Return"], 0.01)
    assert result["Target"].tolist() == [1, 0, 0]


def test_chronological_split_preserves_order():
    data = pd.DataFrame({"value": range(10)})
    train, test = chronological_train_test_split(data, train_split=0.8)
    assert train["value"].tolist() == list(range(8))
    assert test["value"].tolist() == [8, 9]


def test_sequence_shape_and_target_alignment():
    features = np.arange(20).reshape(10, 2)
    targets = np.arange(10)
    sequences, sequence_targets = create_sequences(features, targets, sequence_length=3)
    assert sequences.shape == (7, 3, 2)
    assert sequences[0].tolist() == features[:3].tolist()
    assert sequence_targets.tolist() == list(range(3, 10))


def test_sequence_target_uses_only_observations_before_its_target_date():
    features = np.arange(6).reshape(6, 1)
    targets = np.arange(100, 106)
    sequences, sequence_targets = create_sequences(features, targets, sequence_length=3)
    # The first target corresponds to index 3, while the final input is index 2.
    assert sequences[0, :, 0].tolist() == [0, 1, 2]
    assert sequence_targets[0] == 103


def test_train_context_makes_all_test_targets_evaluable():
    train_features = np.arange(6).reshape(6, 1)
    test_features = np.arange(6, 8).reshape(2, 1)
    test_targets = np.array([106, 107])
    sequences, targets = create_test_sequences_with_train_context(
        train_features, test_features, test_targets, sequence_length=3
    )
    assert sequences.shape == (2, 3, 1)
    assert sequences[0, :, 0].tolist() == [3, 4, 5]
    assert sequences[1, :, 0].tolist() == [4, 5, 6]
    assert targets.tolist() == [106, 107]
