"""Standalone inference over engineered feature data and a saved model artifact."""

import pandas as pd
import torch

from artifacts import load_model_artifact
from dataset import create_feature_sequences
from strategy import generate_signals


def predict_features(feature_data, artifact_path=None, device=None, historical_context=None):
    """Return ``P(Target=1)``, a 0.5 prediction, and a raw threshold label.

    ``feature_data`` must already contain the artifact's engineered feature columns.
    The returned ``signal`` column is a raw label, not a trading exposure; the
    frozen backtest converts it to contrarian exposure separately. Results begin
    after the artifact's lookback window unless historical context is supplied.
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
    if historical_context is not None:
        context_features = historical_context.loc[:, list(loaded.feature_columns)]
        if context_features.isnull().any().any():
            raise ValueError("Historical context contains missing feature values.")
        if len(context_features) < loaded.sequence_length:
            raise ValueError("Historical context is shorter than the sequence length.")
        features_for_sequences = pd.concat(
            [context_features.tail(loaded.sequence_length), features]
        )
    else:
        features_for_sequences = features
    scaled_features = loaded.scaler.transform(features_for_sequences)
    sequences = create_feature_sequences(scaled_features, loaded.sequence_length)
    if historical_context is not None:
        sequences = sequences[-len(features):]
        output_index = feature_data.index
    else:
        output_index = feature_data.index[loaded.sequence_length:]
    if len(sequences) == 0:
        return pd.DataFrame(
            columns=["probability_up", "prediction", "signal"],
            index=output_index,
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
        index=output_index,
    )
