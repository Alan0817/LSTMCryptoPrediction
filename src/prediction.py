"""Standalone inference over engineered feature data and a saved model artifact."""

import pandas as pd
import torch

from artifacts import load_model_artifact
from dataset import create_feature_sequences
from strategy import generate_signals


def predict_features(feature_data, artifact_path=None, device=None):
    """Return probability, 0.5 prediction, and current trading signal for each sequence.

    ``feature_data`` must already contain the artifact's engineered feature columns.
    Results begin after the artifact's lookback window, matching training alignment.
    """
    if not isinstance(feature_data, pd.DataFrame):
        raise TypeError("feature_data must be a pandas DataFrame.")
    if artifact_path is None:
        loaded = load_model_artifact(device=device)
    else:
        loaded = load_model_artifact(artifact_path, device)
    missing = set(loaded.feature_columns).difference(feature_data.columns)
    if missing:
        raise ValueError("Feature data is missing columns: {}".format(sorted(missing)))
    features = feature_data.loc[:, list(loaded.feature_columns)]
    if features.isnull().any().any():
        raise ValueError("Feature data contains missing feature values.")
    scaled_features = loaded.scaler.transform(features)
    sequences = create_feature_sequences(scaled_features, loaded.sequence_length)
    if len(sequences) == 0:
        return pd.DataFrame(
            columns=["probability_up", "prediction", "signal"],
            index=feature_data.index[loaded.sequence_length:],
        )
    device = next(loaded.model.parameters()).device
    with torch.no_grad():
        probabilities = loaded.model(
            torch.tensor(sequences, dtype=torch.float32, device=device)
        ).squeeze(-1).cpu().numpy()
    signals = generate_signals(
        probabilities,
        loaded.upper_signal_threshold,
        loaded.lower_signal_threshold,
    )
    return pd.DataFrame(
        {
            "probability_up": probabilities,
            "prediction": (probabilities > 0.5).astype(int),
            "signal": signals,
        },
        index=feature_data.index[loaded.sequence_length:],
    )
