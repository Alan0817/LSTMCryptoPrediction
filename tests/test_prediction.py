from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import prediction


class CapturingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(0.0))
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs.detach().cpu().numpy()
        return torch.full((len(inputs), 1), 0.6, device=inputs.device)


def test_predict_features_with_context_returns_one_prediction_per_test_target(monkeypatch):
    model = CapturingModel()
    artifact = SimpleNamespace(
        model=model,
        scaler=SimpleNamespace(transform=lambda values: values.to_numpy()),
        feature_columns=("feature",),
        sequence_length=3,
        upper_signal_threshold=0.52,
        lower_signal_threshold=0.48,
    )
    monkeypatch.setattr(prediction, "load_model_artifact", lambda *args, **kwargs: artifact)
    historical_context = pd.DataFrame({"feature": range(6)}, index=range(6))
    test_data = pd.DataFrame({"feature": range(6, 9)}, index=range(6, 9))

    result = prediction.predict_features(test_data, historical_context=historical_context)

    assert result.index.tolist() == [6, 7, 8]
    assert len(result) == len(test_data)
    assert model.inputs[:, :, 0].tolist() == [[3, 4, 5], [4, 5, 6], [5, 6, 7]]
