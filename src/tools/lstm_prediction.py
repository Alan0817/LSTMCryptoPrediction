"""Structured adapter for existing LSTM inference."""

import pandas as pd

from prediction import predict_features


TARGET_DEFINITION = "Target = 1 when Future_Return > 0.005."
RAW_SIGNAL_DEFINITION = "Raw threshold label; it is not a long/short exposure."


def get_lstm_prediction(
    feature_data: pd.DataFrame,
    artifact_path=None,
    historical_context: pd.DataFrame | None = None,
    predictor=predict_features,
) -> dict:
    """Return the latest LSTM prediction without reinterpreting raw-signal semantics."""
    if not isinstance(feature_data, pd.DataFrame):
        raise TypeError("feature_data must be a pandas DataFrame.")
    predictions = predictor(
        feature_data,
        artifact_path=artifact_path,
        historical_context=historical_context,
    )
    if predictions.empty:
        raise ValueError("Insufficient feature history to produce an LSTM prediction.")

    latest = predictions.iloc[-1]
    return {
        "timestamp": pd.Timestamp(predictions.index[-1]).isoformat(),
        "probability_up": float(latest["probability_up"]),
        "prediction": int(latest["prediction"]),
        "raw_signal": int(latest["signal"]),
        "target_definition": TARGET_DEFINITION,
        "raw_signal_definition": RAW_SIGNAL_DEFINITION,
    }
