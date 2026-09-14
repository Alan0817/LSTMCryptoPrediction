import numpy as np
from sklearn.preprocessing import StandardScaler

from artifacts import load_model_artifact, save_model_artifact
from model import LSTMClassifier


def test_model_artifact_round_trip_preserves_scaler_and_metadata(tmp_path):
    scaler = StandardScaler().fit(np.array([[1.0] * 8, [3.0] * 8]))
    model = LSTMClassifier(input_size=8)
    artifact_path = tmp_path / "model.pth"
    save_model_artifact(model, scaler, {"epochs": 1}, output_path=artifact_path)

    loaded = load_model_artifact(artifact_path, device="cpu")
    assert loaded.feature_columns == (
        "Close", "Volume", "RSI", "MACD", "MACD_Signal", "MA_7", "MA_30", "Volatility"
    )
    assert loaded.sequence_length == 30
    assert loaded.training_metadata == {"epochs": 1}
    assert np.allclose(loaded.scaler.mean_, scaler.mean_)
    assert np.allclose(loaded.scaler.scale_, scaler.scale_)
