import json

import numpy as np
import pandas as pd
import pytest

from data_processing import engineer_features
from tools.market_analysis import analyze_market
from tools.registry import ToolRegistry


def make_ohlcv(rows=80):
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = np.linspace(100.0, 180.0, rows) + np.sin(np.arange(rows))
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


def fake_prediction(feature_data, artifact_path=None, historical_context=None):
    assert historical_context is None
    return pd.DataFrame(
        {"probability_up": [0.61], "prediction": [1], "signal": [1]},
        index=pd.DatetimeIndex([feature_data.index[-1]]),
    )


def test_btc_market_analysis_coordinates_tools_and_is_json_safe():
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        return make_ohlcv()

    result = analyze_market(
        "BTC-USD", "2024-01-01", "2024-04-01", downloader=downloader, predictor=fake_prediction
    )

    assert calls == [{"symbol": "BTC-USD", "start": "2024-01-01", "end": "2024-04-01"}]
    assert result["market"]["observations"] == 80
    assert set(result["technical_analysis"]["indicators"]) == {
        "RSI", "MACD", "MACD_Signal", "MA_7", "MA_30", "Volatility"
    }
    assert result["lstm_prediction"]["status"] == "available"
    assert result["lstm_prediction"]["probability_up"] == 0.61
    assert result["lstm_prediction"]["probability_up_definition"] == "P(Target = 1)"
    assert result["lstm_prediction"]["target_definition"] == "Target = 1 when Future_Return > 0.005."
    assert result["lstm_prediction"]["raw_signal_is_exposure"] is False
    assert result["risk"]["observations"] == 79
    assert result["risk"]["assumptions"]["sharpe_annualization"] == "sqrt(252)"
    assert any(item["code"] == "investment_recommendation" for item in result["limitations"])
    json.dumps(result)


def test_unsupported_symbol_preserves_market_technical_and_risk_without_lstm_call():
    def predictor(*args, **kwargs):
        raise AssertionError("unsupported symbols must not reach the LSTM")

    result = analyze_market(
        "ETH-USD", "2024-01-01", "2024-04-01", downloader=lambda **kwargs: make_ohlcv(), predictor=predictor
    )

    assert result["technical_analysis"]["timestamp"]
    assert result["risk"]["observations"] == 79
    assert result["lstm_prediction"] == {
        "status": "not_applicable",
        "reason": "The available LSTM artifact was trained for BTC-USD.",
    }
    applicability = next(item for item in result["limitations"] if item["code"] == "model_symbol_applicability")
    assert applicability["applicable"] is False


def test_insufficient_lstm_history_returns_partial_analysis():
    result = analyze_market(
        "BTC-USD", "2024-01-01", "2024-03-01", downloader=lambda **kwargs: make_ohlcv(50), predictor=fake_prediction
    )

    assert result["technical_analysis"]["timestamp"]
    assert result["risk"]["observations"] == 49
    assert result["lstm_prediction"]["status"] == "insufficient_history"


def test_missing_lstm_artifact_returns_partial_analysis():
    def unavailable_predictor(*args, **kwargs):
        raise FileNotFoundError("Model artifact not found: missing.pth")

    result = analyze_market(
        "BTC-USD",
        "2024-01-01",
        "2024-04-01",
        downloader=lambda **kwargs: make_ohlcv(),
        predictor=unavailable_predictor,
    )

    assert result["technical_analysis"]["timestamp"]
    assert result["risk"]["observations"] == 79
    assert result["lstm_prediction"] == {
        "status": "artifact_unavailable",
        "reason": "Model artifact not found: missing.pth",
    }


def test_invalid_market_data_fails_clearly():
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    with pytest.raises(ValueError, match="Market data is empty"):
        analyze_market("BTC-USD", "2024-01-01", "2024-04-01", downloader=lambda **kwargs: empty)


def test_registry_exposes_only_simple_market_analysis_arguments_and_uses_controlled_dependencies():
    registry = ToolRegistry(
        market_data_downloader=lambda **kwargs: make_ohlcv(),
        market_analysis_predictor=fake_prediction,
    )
    schema = registry.get("analyze_market").parameters_schema
    schema_text = json.dumps(schema).lower()

    assert set(schema["properties"]) == {"symbol", "start_date", "end_date"}
    assert "downloader" not in schema_text
    assert "predictor" not in schema_text
    assert "artifact" not in schema_text
    result = registry.execute(
        "analyze_market", {"symbol": "BTC-USD", "start_date": "2024-01-01", "end_date": "2024-04-01"}
    )
    assert result["lstm_prediction"]["status"] == "available"
    json.dumps(result)


def test_orchestrator_reuses_one_engineered_frame_for_technical_and_prediction():
    calls = []

    def feature_engineer(raw_data):
        result = engineer_features(raw_data)
        calls.append(result)
        return result

    def predictor(feature_data, **kwargs):
        assert feature_data is calls[0]
        return fake_prediction(feature_data, **kwargs)

    result = analyze_market(
        "BTC-USD",
        "2024-01-01",
        "2024-04-01",
        downloader=lambda **kwargs: make_ohlcv(),
        feature_engineer=feature_engineer,
        predictor=predictor,
    )

    assert len(calls) == 1
    assert result["technical_analysis"]["timestamp"] == calls[0].index[-1].isoformat()
