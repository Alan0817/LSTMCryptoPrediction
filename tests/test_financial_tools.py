import json

import numpy as np
import pandas as pd
import pytest

from tools.lstm_prediction import get_lstm_prediction
from tools.market_data import get_market_data
from tools.risk_metrics import get_risk_metrics
from tools.technical_analysis import get_technical_indicators


def make_ohlcv(rows=60):
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = np.linspace(100.0, 160.0, rows) + np.sin(np.arange(rows))
    return pd.DataFrame(
        {
            "Open": close - 1.0,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Volume": np.linspace(1_000.0, 2_000.0, rows),
        },
        index=index,
    )


def test_market_data_tool_returns_json_safe_summary():
    calls = []

    def fake_downloader(**kwargs):
        calls.append(kwargs)
        return make_ohlcv(6)

    result = get_market_data("BTC-USD", "2024-01-01", "2024-02-01", downloader=fake_downloader)

    assert calls == [{"symbol": "BTC-USD", "start": "2024-01-01", "end": "2024-02-01"}]
    assert result["observations"] == 6
    assert result["latest_ohlcv"]["Close"] == pytest.approx(make_ohlcv(6).iloc[-1]["Close"])
    assert len(result["recent_returns"]) == 5
    json.dumps(result)


def test_market_data_tool_rejects_empty_data():
    def empty_downloader(**kwargs):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    with pytest.raises(ValueError, match="Market data is empty"):
        get_market_data("BTC-USD", "2024-01-01", "2024-02-01", downloader=empty_downloader)


def test_market_data_tool_rejects_invalid_date_range():
    with pytest.raises(ValueError, match="start_date must be earlier"):
        get_market_data("BTC-USD", "2024-02-01", "2024-01-01", downloader=lambda **kwargs: make_ohlcv())


def test_technical_indicators_reuse_feature_engineering_and_are_json_safe():
    result = get_technical_indicators(make_ohlcv())

    assert result["timestamp"].startswith("2024-02")
    assert set(result["indicators"]) == {"RSI", "MACD", "MACD_Signal", "MA_7", "MA_30", "Volatility"}
    assert all(isinstance(value, float) for value in result["indicators"].values())
    json.dumps(result)


def test_technical_indicators_require_sufficient_history():
    with pytest.raises(ValueError, match="Insufficient historical data"):
        get_technical_indicators(make_ohlcv(10))


def test_lstm_prediction_returns_raw_signal_without_exposure_reinterpretation():
    feature_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

    def fake_predictor(data, artifact_path=None, historical_context=None):
        assert data is feature_data
        return pd.DataFrame(
            {
                "probability_up": [0.6],
                "prediction": [1],
                "signal": [1],
            },
            index=pd.DatetimeIndex(["2024-03-01"]),
        )

    result = get_lstm_prediction(feature_data, predictor=fake_predictor)

    assert result["probability_up"] == 0.6
    assert result["prediction"] == 1
    assert result["raw_signal"] == 1
    assert "not a long/short exposure" in result["raw_signal_definition"]
    json.dumps(result)


def test_lstm_prediction_preserves_missing_artifact_error():
    def unavailable_predictor(*args, **kwargs):
        raise FileNotFoundError("Model artifact not found: missing.pth")

    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        get_lstm_prediction(pd.DataFrame({"feature": [1.0]}), predictor=unavailable_predictor)


def test_risk_metrics_reuse_existing_definitions_and_are_json_safe():
    returns = [0.01, -0.02, 0.03]

    result = get_risk_metrics(returns)

    expected_cumulative_return = (1.01 * 0.98 * 1.03) - 1
    assert result["observations"] == 3
    assert result["annualization_factor"] == 252
    assert result["cumulative_return"] == pytest.approx(expected_cumulative_return)
    assert result["assumptions"]["sharpe_annualization"] == "sqrt(252)"
    json.dumps(result)
